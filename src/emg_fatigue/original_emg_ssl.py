"""Masked temporal reconstruction for unlabeled original sEMG recordings.

This module learns a signal representation without creating fatigue or movement
labels. Its outputs are reconstruction and embedding vectors only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def extract_fixed_windows(
    time_s: np.ndarray,
    signal: np.ndarray,
    *,
    window_samples: int,
    hop_samples: int,
    max_gap_s: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed sample-count windows and reject gaps or invalid edges."""
    time = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float32)
    if time.ndim != 1 or values.ndim != 1 or len(time) != len(values):
        raise ValueError("time_s and signal must be matching one-dimensional arrays")
    if len(time) < 2 or not np.isfinite(time).all() or not np.isfinite(values).all():
        raise ValueError("time_s and signal must contain finite samples")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if not isinstance(window_samples, int) or window_samples < 2:
        raise ValueError("window_samples must be an integer of at least two")
    if not isinstance(hop_samples, int) or hop_samples < 1:
        raise ValueError("hop_samples must be a positive integer")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    if valid_mask is None:
        valid = np.ones(len(time), dtype=bool)
    else:
        valid = np.asarray(valid_mask)
        if valid.dtype != np.bool_ or valid.shape != time.shape:
            raise ValueError("valid_mask must be a boolean vector matching time_s")

    windows: list[np.ndarray] = []
    starts: list[int] = []
    for start in range(0, len(time) - window_samples + 1, hop_samples):
        end = start + window_samples
        if not valid[start:end].all():
            continue
        if np.any(np.diff(time[start:end]) > max_gap_s):
            continue
        windows.append(values[start:end])
        starts.append(start)
    if not windows:
        return np.empty((0, window_samples), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(windows), np.asarray(starts, dtype=np.int64)


def split_groups(
    group_ids: Sequence[str], *, validation_fraction: float, seed: int
) -> dict[str, list[str]]:
    """Make a deterministic group-level split before any windowing."""
    groups = sorted(set(map(str, group_ids)))
    if len(groups) < 2:
        raise ValueError("at least two distinct groups are required for validation")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    validation_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * validation_fraction)))
    rng = np.random.default_rng(seed)
    validation = sorted(rng.choice(groups, size=validation_count, replace=False).tolist())
    validation_set = set(validation)
    train = [group for group in groups if group not in validation_set]
    return {"train": train, "validation": validation}


def random_block_mask(
    *,
    batch_size: int,
    n_samples: int,
    block_size: int,
    mask_fraction: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Mask reproducibly sampled contiguous blocks for each signal window."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least two")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not np.isfinite(mask_fraction) or not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between zero and one")
    device = torch.device(device)
    n_blocks = math.ceil(n_samples / block_size)
    n_masked_blocks = min(n_blocks, max(1, round(n_blocks * mask_fraction)))
    mask = torch.zeros((batch_size, 1, n_samples), dtype=torch.bool, device=device)
    for row in range(batch_size):
        selected = torch.randperm(n_blocks, generator=generator, device=device)[:n_masked_blocks]
        for block in selected.tolist():
            start = block * block_size
            mask[row, 0, start : min(start + block_size, n_samples)] = True
    return mask


def masked_reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error evaluated only at deliberately hidden samples."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have matching BxCxL shapes")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor")
    if mask.shape == (prediction.shape[0], prediction.shape[2]):
        mask = mask.unsqueeze(1)
    if mask.shape != prediction.shape:
        raise ValueError("mask must match prediction or have shape BxL")
    if not mask.any():
        raise ValueError("at least one sample must be masked")
    return torch.mean(torch.square(prediction[mask] - target[mask]))


class MaskedConvAutoencoder(nn.Module):
    """Compact 1-D CNN with a 64-channel temporal encoder and signal decoder."""

    embedding_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    @staticmethod
    def _validate_signal(signal: torch.Tensor) -> None:
        if signal.ndim != 3 or signal.shape[1] != 1:
            raise ValueError("signal must have shape Bx1xL")
        if not torch.is_floating_point(signal) or not torch.isfinite(signal).all():
            raise ValueError("signal must contain finite floating-point values")

    def _encode(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._validate_signal(signal)
        if mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor")
        if mask.shape == (signal.shape[0], signal.shape[2]):
            mask = mask.unsqueeze(1)
        if mask.shape != signal.shape:
            raise ValueError("mask must match the Bx1xL signal shape")
        model_input = torch.cat((signal.masked_fill(mask, 0.0), mask.to(signal.dtype)), dim=1)
        return self.encoder(model_input)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        latent = self._encode(signal, mask)
        expanded = F.interpolate(latent, size=signal.shape[-1], mode="linear", align_corners=False)
        return self.decoder(expanded)

    def embedding(self, signal: torch.Tensor) -> torch.Tensor:
        """Return the global-average-pooled 64-D embedding for an unmasked window."""
        self._validate_signal(signal)
        visible = torch.zeros_like(signal, dtype=torch.bool)
        latent = self._encode(signal, visible)
        return F.adaptive_avg_pool1d(latent, 1).squeeze(-1)
