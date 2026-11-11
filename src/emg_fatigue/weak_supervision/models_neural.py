"""Small raw-signal models for weakly supervised fatigue-proxy learning."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SignalEncoder(nn.Module):
    """Compact single-channel encoder for 500 ms sEMG windows."""

    def __init__(self, embedding_dim: int = 32) -> None:
        super().__init__()
        if int(embedding_dim) < 2:
            raise ValueError("embedding_dim must be at least 2")
        self.convolutions = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, int(embedding_dim)),
            nn.LayerNorm(int(embedding_dim)),
            nn.GELU(),
        )
        self.embedding_dim = int(embedding_dim)

    def forward_features(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 1:
            raise ValueError("SignalEncoder input must have shape (batch, 1, samples)")
        if signal.shape[-1] < 64:
            raise ValueError("SignalEncoder requires at least 64 samples per window")
        return self.convolutions(signal)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.projection(self.forward_features(signal))


class BinarySignalClassifier(nn.Module):
    """Single-logit classifier for the fixed binary proxy target."""

    def __init__(self, encoder: SignalEncoder | None = None, embedding_dim: int = 32) -> None:
        super().__init__()
        self.encoder = encoder or SignalEncoder(embedding_dim=embedding_dim)
        self.head = nn.Linear(self.encoder.embedding_dim, 1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(signal)).squeeze(-1)


class SoftGatedExperts(nn.Module):
    """Soft mixture of latent experts predicting one and the same proxy label."""

    def __init__(
        self,
        expert_count: int = 3,
        encoder: SignalEncoder | None = None,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        if int(expert_count) not in (2, 3):
            raise ValueError("Only two or three latent experts are supported")
        self.encoder = encoder or SignalEncoder(embedding_dim=embedding_dim)
        self.gate = nn.Linear(self.encoder.embedding_dim, int(expert_count))
        self.experts = nn.ModuleList(
            nn.Linear(self.encoder.embedding_dim, 1) for _ in range(int(expert_count))
        )
        self.expert_count = int(expert_count)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encoder(signal)
        gate_probabilities = torch.softmax(self.gate(embedding), dim=1)
        expert_logits = torch.cat([head(embedding) for head in self.experts], dim=1)
        mixture_logit = torch.sum(gate_probabilities * expert_logits, dim=1)
        return mixture_logit, gate_probabilities, expert_logits


class MaskedSignalAutoencoder(nn.Module):
    """Convolutional encoder-decoder trained to reconstruct masked sample spans."""

    def __init__(self, encoder: SignalEncoder | None = None, embedding_dim: int = 32) -> None:
        super().__init__()
        self.encoder = encoder or SignalEncoder(embedding_dim=embedding_dim)
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),
        )

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder.forward_features(signal)
        upsampled = F.interpolate(features, size=signal.shape[-1], mode="linear", align_corners=False)
        reconstruction = self.decoder(upsampled)
        embedding = self.encoder.projection(features)
        return reconstruction, embedding


def gate_load_balance_penalty(gates: torch.Tensor) -> torch.Tensor:
    """Return a differentiable penalty for nonuniform mean soft-gate load."""
    if gates.ndim != 2 or gates.shape[1] < 2:
        raise ValueError("gates must have shape (batch, experts>=2)")
    mean_load = gates.mean(dim=0)
    uniform = torch.full_like(mean_load, 1.0 / gates.shape[1])
    return torch.sum((mean_load - uniform) ** 2) * gates.shape[1]


def normalized_gate_entropy(gates: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Mean gate entropy normalized by log(expert_count), approximately in [0, 1]."""
    if gates.ndim != 2 or gates.shape[1] < 2:
        raise ValueError("gates must have shape (batch, experts>=2)")
    entropy = -torch.sum(gates * torch.log(gates.clamp_min(float(epsilon))), dim=1)
    return entropy.mean() / math.log(gates.shape[1])
