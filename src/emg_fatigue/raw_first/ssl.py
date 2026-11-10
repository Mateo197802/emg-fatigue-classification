"""Compact masked-reconstruction CNN for label-free sEMG representation learning."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .augmentation import ALL_FAMILIES, corrupt_signal


class MaskedConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        latent_map = self.encoder(values.unsqueeze(1))
        return torch.mean(latent_map, dim=-1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_map = self.encoder(values.unsqueeze(1))
        latent = torch.mean(latent_map, dim=-1)
        reconstruction = self.decoder(latent_map)
        reconstruction = F.interpolate(reconstruction, size=values.shape[-1], mode="linear", align_corners=False)
        return reconstruction.squeeze(1), latent


def fit_signal_scale(train_windows: np.ndarray) -> dict[str, float]:
    values = np.asarray(train_windows, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Signal standardization expects a finite 2D array of training windows.")
    mean = float(values.mean())
    scale = float(values.std())
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("Training signals have no usable global variance.")
    return {"mean": mean, "scale": scale, "method": "global training-group mean/std; no per-window amplitude normalization"}


def apply_signal_scale(values: np.ndarray, scale: dict[str, float]) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float32) - float(scale["mean"])) / float(scale["scale"])).astype(np.float32)


def _mask_batch(batch: torch.Tensor, rng: np.random.Generator, fraction: float, span_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    masked = batch.clone()
    mask = torch.zeros_like(batch, dtype=torch.bool)
    length = batch.shape[-1]
    for row in range(batch.shape[0]):
        target = max(1, int(round(length * fraction)))
        marked = 0
        while marked < target:
            start = int(rng.integers(0, max(1, length - span_samples + 1)))
            end = min(length, start + span_samples)
            mask[row, start:end] = True
            marked = int(mask[row].sum().item())
        masked[row][mask[row]] = 0.0
    return masked, mask


def _augment_batch(
    batch: np.ndarray,
    epoch: int,
    seed: int,
    sampling_rate_hz: float,
    levels: tuple[float, ...],
    line_hz: float | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    families = tuple(ALL_FAMILIES)
    output = np.empty_like(batch, dtype=np.float32)
    for index, row in enumerate(batch):
        family = families[int(rng.integers(0, len(families)))]
        severity = float(levels[int(rng.integers(0, len(levels)))])
        row_seed = int(seed + epoch * 2_000_003 + index * 101 + int(rng.integers(0, 1_000_000)))
        corrupted, _ = corrupt_signal(row, family, severity, row_seed, sampling_rate_hz, line_hz)
        output[index] = corrupted
    return output


def _reconstruction_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = (predicted - target) ** 2
    masked = (squared * mask).sum() / mask.sum().clamp_min(1)
    unmasked = (squared * (~mask)).sum() / (~mask).sum().clamp_min(1)
    return masked + 0.25 * unmasked


def _evaluate_loss(
    model: MaskedConvAutoencoder,
    values: np.ndarray,
    batch_size: int,
    seed: int,
    mask_fraction: float,
    span_samples: int,
) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            target = torch.from_numpy(values[start : start + batch_size])
            masked, mask = _mask_batch(target, rng, mask_fraction, span_samples)
            predicted, _ = model(masked)
            losses.append(float(_reconstruction_loss(predicted, target, mask).item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_masked_encoder(
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    sampling_rate_hz: float,
    output_dir: Path,
    condition: str,
    seed: int,
    line_hz: float | None,
    levels: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    mask_fraction: float = 0.15,
    mask_span_ms: float = 20.0,
) -> tuple[MaskedConvAutoencoder, list[dict[str, float]], dict[str, object]]:
    if condition not in {"clean", "augmented"}:
        raise ValueError("SSL condition must be 'clean' or 'augmented'.")
    train_windows = np.asarray(train_windows, dtype=np.float32)
    validation_windows = np.asarray(validation_windows, dtype=np.float32)
    if train_windows.ndim != 2 or train_windows.shape[1] < 64 or len(train_windows) == 0:
        raise ValueError("Training windows must be a nonempty matrix of at least 64 samples each.")
    if validation_windows.ndim != 2 or validation_windows.shape[1] != train_windows.shape[1] or len(validation_windows) == 0:
        raise ValueError("Validation windows must be nonempty and match training window length.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epoch count and batch size must be positive.")

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = MaskedConvAutoencoder().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    span_samples = max(1, int(round(mask_span_ms * sampling_rate_hz / 1000.0)))
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_windows))
        batch_losses: list[float] = []
        for batch_index, batch_start in enumerate(range(0, len(order), batch_size)):
            selected = train_windows[order[batch_start : batch_start + batch_size]]
            target = torch.from_numpy(selected)
            if condition == "augmented":
                augmented = _augment_batch(selected, epoch, seed + batch_index * 17, sampling_rate_hz, levels, line_hz)
                inputs = torch.from_numpy(augmented)
            else:
                inputs = target
            masked, mask = _mask_batch(inputs, rng, mask_fraction, span_samples)
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = model(masked)
            loss = _reconstruction_loss(predicted, target, mask)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))

        validation_loss = _evaluate_loss(model, validation_windows, batch_size, seed + 77, mask_fraction, span_samples)
        row = {
            "epoch": float(epoch + 1),
            "training_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            "validation_masked_reconstruction_loss": validation_loss,
        }
        history.append(row)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No checkpoint was selected from validation reconstruction.")
    model.load_state_dict(best_state)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"encoder_{condition}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "condition": condition,
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_validation_masked_reconstruction_loss": float(best_validation),
            "torch_version": torch.__version__,
            "architecture": "three-stage 1D convolutional encoder with transposed-convolution masked-reconstruction decoder",
        },
        checkpoint,
    )
    history_path = output_dir / f"history_{condition}.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    # A round-trip check protects the frozen checkpoint from serialization drift.
    loaded = MaskedConvAutoencoder().cpu().eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    loaded.load_state_dict(saved["state_dict"])
    probe = torch.from_numpy(validation_windows[: min(8, len(validation_windows))])
    with torch.no_grad():
        before, _ = model(probe)
        after, _ = loaded(probe)
    exact_roundtrip = bool(torch.equal(before, after))
    if not exact_roundtrip:
        raise RuntimeError("Serialized encoder did not reproduce identical reconstruction output.")

    metadata: dict[str, object] = {
        "condition": condition,
        "seed": int(seed),
        "training_window_count": int(len(train_windows)),
        "validation_window_count": int(len(validation_windows)),
        "epochs_requested": int(epochs),
        "best_epoch_by_validation_reconstruction": int(best_epoch),
        "best_validation_masked_reconstruction_loss": float(best_validation),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "mask_fraction": float(mask_fraction),
        "mask_span_ms": float(mask_span_ms),
        "training_augmentation_levels": list(levels) if condition == "augmented" else [],
        "training_augmentation_families": list(ALL_FAMILIES) if condition == "augmented" else [],
        "device": "cpu",
        "torch_version": torch.__version__,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_filename": checkpoint.name,
        "checkpoint_round_trip_exact": exact_roundtrip,
    }
    (output_dir / f"metadata_{condition}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model, history, metadata


def model_embedding(model: MaskedConvAutoencoder, values: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(np.asarray(values[start : start + batch_size], dtype=np.float32))
            embeddings.append(model.encode(batch).cpu().numpy())
    return np.vstack(embeddings) if embeddings else np.empty((0, 64), dtype=np.float32)


def model_reconstruct(model: MaskedConvAutoencoder, values: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    reconstructions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(np.asarray(values[start : start + batch_size], dtype=np.float32))
            reconstruction, _ = model(batch)
            reconstructions.append(reconstruction.cpu().numpy())
    return np.vstack(reconstructions) if reconstructions else np.empty_like(values)
