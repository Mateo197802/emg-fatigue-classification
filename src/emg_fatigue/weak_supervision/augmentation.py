"""Deterministic, RMS-audited perturbations for proxy-model development."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


CORRUPTION_FAMILIES = (
    "gaussian_white",
    "colored_ar1_rho_0_9",
    "line_interference_50_60_hz",
    "baseline_wander_0_5_2_hz",
    "motion_transients_0_5_2_hz",
    "sparse_impulses",
    "temporal_shift",
    "gain_perturbation",
    "combined",
)
TRAINING_LEVELS = (0.05, 0.10, 0.15, 0.20)
EVALUATION_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50)
MAX_SHIFT_FRACTION = 0.05


def stable_seed(*parts: object) -> int:
    """Return a process-independent 32-bit seed derived from stable identifiers."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def rms(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("corruption input must be a finite, non-empty one-dimensional signal")
    return float(np.sqrt(np.mean(data * data)))


def _scaled_additive(clean: np.ndarray, component: np.ndarray, level: float) -> tuple[np.ndarray, float]:
    clean_rms = rms(clean)
    component_rms = rms(component)
    if clean_rms <= np.finfo(float).eps:
        return np.zeros_like(clean, dtype=np.float64), 0.0
    if component_rms <= np.finfo(float).eps:
        return np.zeros_like(clean, dtype=np.float64), 0.0
    delta = np.asarray(component, dtype=np.float64) * (float(level) * clean_rms / component_rms)
    achieved = rms(delta) / clean_rms
    return delta, achieved


def _ar1_noise(generator: np.random.Generator, size: int, rho: float = 0.9) -> np.ndarray:
    innovations = generator.normal(0.0, 1.0, size=int(size))
    output = np.empty_like(innovations)
    output[0] = innovations[0]
    innovation_scale = np.sqrt(max(0.0, 1.0 - float(rho) ** 2))
    for index in range(1, len(output)):
        output[index] = float(rho) * output[index - 1] + innovation_scale * innovations[index]
    return output


def _shift_reflect(values: np.ndarray, samples: int) -> np.ndarray:
    shift = int(samples)
    if shift == 0:
        return np.asarray(values, dtype=np.float64).copy()
    pad = abs(shift)
    padded = np.pad(np.asarray(values, dtype=np.float64), (pad, pad), mode="reflect")
    start = pad - shift
    return padded[start : start + len(values)]


def corrupt_window(
    values: np.ndarray,
    family: str,
    level: float,
    seed: int,
    *,
    sampling_rate_hz: float = 1000.0,
    max_shift_fraction: float = MAX_SHIFT_FRACTION,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Corrupt one processed window and report requested and achieved RMS ratios.

    Additive families are scaled so ``RMS(delta) / RMS(clean)`` equals ``level``.
    Gain perturbations use that ratio exactly. For temporal shift, ``level`` sets
    the displacement as a fraction of the configured maximum shift; its achieved
    relative RMS difference is recorded because it depends on waveform shape.
    """
    clean = np.asarray(values, dtype=np.float64)
    clean_rms = rms(clean)
    family_name = str(family)
    requested = float(level)
    fs = float(sampling_rate_hz)
    if family_name not in CORRUPTION_FAMILIES:
        raise ValueError(f"unknown corruption family: {family_name}")
    if not 0.0 < requested <= 1.0:
        raise ValueError("corruption level must be in (0, 1]")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("sampling_rate_hz must be positive and finite")
    generator = np.random.default_rng(int(seed))
    parameters: dict[str, Any] = {}

    if family_name == "gain_perturbation":
        direction = int(generator.choice((-1, 1)))
        gain = 1.0 + direction * requested
        corrupted = clean * gain
        delta = corrupted - clean
        achieved = rms(delta) / clean_rms if clean_rms > 0.0 else 0.0
        parameters["gain"] = float(gain)
    elif family_name == "temporal_shift":
        maximum_samples = max(1, int(round(float(max_shift_fraction) * len(clean))))
        shift_fraction = min(1.0, requested / 0.50)
        shift_samples = max(1, int(round(maximum_samples * shift_fraction)))
        shift_samples = min(maximum_samples, shift_samples) * int(generator.choice((-1, 1)))
        corrupted = _shift_reflect(clean, shift_samples)
        delta = corrupted - clean
        achieved = rms(delta) / clean_rms if clean_rms > 0.0 else 0.0
        parameters.update(
            {
                "shift_samples": int(shift_samples),
                "shift_fraction_of_window": float(abs(shift_samples) / len(clean)),
                "maximum_shift_fraction": float(max_shift_fraction),
            }
        )
    else:
        time = np.arange(len(clean), dtype=np.float64) / fs
        if family_name == "gaussian_white":
            component = generator.normal(0.0, 1.0, len(clean))
        elif family_name == "colored_ar1_rho_0_9":
            component = _ar1_noise(generator, len(clean), rho=0.9)
            parameters["ar1_rho"] = 0.9
        elif family_name == "line_interference_50_60_hz":
            frequency = float(generator.choice((50.0, 60.0)))
            phase = float(generator.uniform(0.0, 2.0 * np.pi))
            component = np.sin(2.0 * np.pi * frequency * time + phase)
            parameters.update({"frequency_hz": frequency, "phase_rad": phase})
        elif family_name == "baseline_wander_0_5_2_hz":
            frequency = float(generator.uniform(0.5, 2.0))
            phase = float(generator.uniform(0.0, 2.0 * np.pi))
            component = np.sin(2.0 * np.pi * frequency * time + phase)
            parameters.update({"frequency_hz": frequency, "phase_rad": phase})
        elif family_name == "motion_transients_0_5_2_hz":
            frequency = float(generator.uniform(0.5, 2.0))
            center = float(generator.uniform(0.1, 0.4))
            width = float(generator.uniform(0.015, 0.06))
            phase = float(generator.uniform(0.0, 2.0 * np.pi))
            envelope = np.exp(-0.5 * ((time - center) / width) ** 2)
            component = envelope * np.sin(2.0 * np.pi * frequency * time + phase)
            parameters.update(
                {
                    "carrier_frequency_hz": frequency,
                    "burst_center_s": center,
                    "burst_width_s": width,
                    "phase_rad": phase,
                }
            )
        elif family_name == "sparse_impulses":
            component = np.zeros(len(clean), dtype=np.float64)
            count = max(1, int(round(0.01 * len(clean))))
            positions = generator.choice(len(clean), size=min(count, len(clean)), replace=False)
            component[positions] = generator.choice((-1.0, 1.0), size=len(positions))
            parameters["impulse_count"] = int(len(positions))
            parameters["impulse_probability"] = 0.01
        elif family_name == "combined":
            pieces = [
                generator.normal(0.0, 1.0, len(clean)),
                _ar1_noise(generator, len(clean), rho=0.9),
                np.sin(2.0 * np.pi * float(generator.choice((50.0, 60.0))) * time + generator.uniform(0.0, 2.0 * np.pi)),
                np.sin(2.0 * np.pi * float(generator.uniform(0.5, 2.0)) * time + generator.uniform(0.0, 2.0 * np.pi)),
            ]
            motion_frequency = float(generator.uniform(0.5, 2.0))
            motion_center = float(generator.uniform(0.1, 0.4))
            motion_width = float(generator.uniform(0.015, 0.06))
            motion_phase = float(generator.uniform(0.0, 2.0 * np.pi))
            pieces.append(
                np.exp(-0.5 * ((time - motion_center) / motion_width) ** 2)
                * np.sin(2.0 * np.pi * motion_frequency * time + motion_phase)
            )
            impulse = np.zeros(len(clean), dtype=np.float64)
            count = max(1, int(round(0.01 * len(clean))))
            positions = generator.choice(len(clean), size=min(count, len(clean)), replace=False)
            impulse[positions] = generator.choice((-1.0, 1.0), size=len(positions))
            pieces.append(impulse)
            component = np.sum(np.stack(pieces, axis=0), axis=0)
            parameters.update(
                {
                    "components": ["gaussian_white", "colored_ar1_rho_0_9", "line_interference_50_60_hz", "baseline_wander_0_5_2_hz", "motion_transients_0_5_2_hz", "sparse_impulses"],
                    "ar1_rho": 0.9,
                    "impulse_count": int(len(positions)),
                }
            )
        else:  # pragma: no cover - guarded by the family validation above
            raise RuntimeError("unreachable corruption family")
        delta, achieved = _scaled_additive(clean, component, requested)
        corrupted = clean + delta

    if corrupted.shape != clean.shape or not np.isfinite(corrupted).all():
        raise FloatingPointError("corruption produced a shape mismatch or non-finite values")
    final_delta = corrupted - clean
    final_achieved = rms(final_delta) / clean_rms if clean_rms > 0.0 else 0.0
    receipt: dict[str, Any] = {
        "family": family_name,
        "requested_level": requested,
        "achieved_rms_ratio": float(final_achieved),
        "clean_rms": clean_rms,
        "delta_rms": rms(final_delta),
        "seed": int(seed),
        "sampling_rate_hz": fs,
        "parameters": parameters,
        "level_semantics": (
            "requested relative RMS difference; achieved value recorded"
            if family_name == "temporal_shift"
            else "RMS(delta)/RMS(clean)"
        ),
    }
    return np.asarray(corrupted, dtype=np.float32), receipt
