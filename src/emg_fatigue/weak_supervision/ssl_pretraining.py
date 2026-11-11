"""Fold-limited masked-span pretraining for a fresh sEMG encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .models_neural import MaskedSignalAutoencoder, SignalEncoder


@dataclass(frozen=True)
class EncoderFit:
    encoder: SignalEncoder
    reconstruction_losses: tuple[float, ...]
    seed: int
    optimizer: str
    train_window_count: int


def _as_signal_tensor(windows: np.ndarray | torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(windows, dtype=torch.float32, device="cpu")
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] != 1 or values.shape[2] < 64:
        raise ValueError("train_windows must have shape (n, samples) or (n, 1, samples), with n>0")
    if not torch.isfinite(values).all():
        raise ValueError("train_windows must contain only finite values")
    return values.contiguous()


def _masked_spans(batch: torch.Tensor, generator: torch.Generator, fraction: float) -> torch.Tensor:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("mask_fraction must be between zero and one")
    mask = torch.zeros_like(batch, dtype=torch.bool)
    sample_count = int(batch.shape[-1])
    target_count = max(1, int(round(sample_count * float(fraction))))
    minimum_span = max(2, int(round(sample_count * 0.02)))
    maximum_span = max(minimum_span, int(round(sample_count * 0.08)))
    for row in range(batch.shape[0]):
        while int(mask[row].sum().item()) < target_count:
            start = int(torch.randint(sample_count, (1,), generator=generator).item())
            width = int(
                torch.randint(minimum_span, maximum_span + 1, (1,), generator=generator).item()
            )
            end = min(sample_count, start + width)
            mask[row, :, start:end] = True
    return mask


def pretrain_encoder(
    train_windows: np.ndarray | torch.Tensor,
    *,
    epochs: int = 6,
    seed: int = 20261113,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    mask_fraction: float = 0.2,
) -> EncoderFit:
    """Fit a fresh encoder using only the supplied training-fold raw windows.

    The caller controls group membership and fold-specific normalization before
    passing windows here. This function has no validation-data input, so held-out
    groups cannot influence reconstruction fitting or checkpoint selection.
    """
    values = _as_signal_tensor(train_windows)
    epoch_count = int(epochs)
    batch_count = int(batch_size)
    if epoch_count < 1 or batch_count < 1 or float(learning_rate) <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    torch.manual_seed(int(seed))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1)
    autoencoder = MaskedSignalAutoencoder().cpu()
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    reconstruction_losses: list[float] = []

    autoencoder.train()
    for _ in range(epoch_count):
        permutation = torch.randperm(values.shape[0], generator=generator)
        weighted_loss = 0.0
        processed = 0
        for start in range(0, values.shape[0], batch_count):
            batch = values[permutation[start : start + batch_count]]
            mask = _masked_spans(batch, generator, float(mask_fraction))
            masked_input = batch.masked_fill(mask, 0.0)
            reconstruction, _ = autoencoder(masked_input)
            loss = torch.sum((reconstruction - batch).square() * mask) / mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Masked reconstruction loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_rows = int(batch.shape[0])
            weighted_loss += float(loss.detach()) * batch_rows
            processed += batch_rows
        reconstruction_losses.append(weighted_loss / max(processed, 1))

    autoencoder.eval()
    return EncoderFit(
        encoder=autoencoder.encoder,
        reconstruction_losses=tuple(reconstruction_losses),
        seed=int(seed),
        optimizer="AdamW",
        train_window_count=int(values.shape[0]),
    )
