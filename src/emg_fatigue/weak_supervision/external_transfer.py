"""Weight-only transfer adapters for externally trained EMG encoders."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .models_neural import SignalEncoder


def initialize_signal_encoder_from_cerqueira(
    encoder: SignalEncoder,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Map the four-channel Cerqueira CNN trunk into the one-channel local encoder.

    The external first convolution is averaged over its four source channels and
    center-cropped from 17 to 9 taps. Subsequent convolution and projection
    weights map shape-for-shape. The external three-class head is never copied.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"external Cerqueira checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("external checkpoint does not contain a state_dict")
    source = payload["state_dict"]
    target = encoder.state_dict()
    mapping = {
        "convolutions.1.weight": "encoder.1.weight",
        "convolutions.1.bias": "encoder.1.bias",
        "convolutions.3.weight": "encoder.3.weight",
        "convolutions.4.weight": "encoder.4.weight",
        "convolutions.4.bias": "encoder.4.bias",
        "convolutions.6.weight": "encoder.6.weight",
        "convolutions.7.weight": "encoder.7.weight",
        "convolutions.7.bias": "encoder.7.bias",
        "projection.2.weight": "encoder.11.weight",
        "projection.2.bias": "encoder.11.bias",
        "projection.3.weight": "encoder.12.weight",
        "projection.3.bias": "encoder.12.bias",
    }
    first = source.get("encoder.0.weight")
    if first is None or first.ndim != 3 or first.shape[1] != 4:
        raise ValueError("external first convolution is not a four-channel kernel")
    desired_width = int(target["convolutions.0.weight"].shape[-1])
    if first.shape[-1] < desired_width:
        raise ValueError("external first kernel is narrower than the local kernel")
    crop_start = (int(first.shape[-1]) - desired_width) // 2
    adapted_first = first[..., crop_start : crop_start + desired_width].mean(dim=1, keepdim=True)
    if adapted_first.shape != target["convolutions.0.weight"].shape:
        raise ValueError("adapted external first convolution does not match local encoder shape")
    target["convolutions.0.weight"] = adapted_first.detach().clone()
    for destination, origin in mapping.items():
        value = source.get(origin)
        if value is None:
            raise ValueError(f"external encoder checkpoint is missing {origin}")
        if value.shape != target[destination].shape:
            raise ValueError(f"external encoder shape mismatch for {destination}: {tuple(value.shape)} vs {tuple(target[destination].shape)}")
        target[destination] = value.detach().clone()
    encoder.load_state_dict(target, strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "status": "initialized_from_external_cerqueira_supervision",
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "source_task": "Cerqueira source-reported self-perceived fatigue 0/1/2",
        "target_head_copied": False,
        "adapter": "mean over four external input-channel filters, center crop 17-to-9 taps, shape-matched deeper conv and projection layers",
        "target_encoder_input": "single-channel local 500 ms sEMG windows",
    }
