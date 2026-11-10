"""Physically scaled corruption functions used for SSL and robustness analysis."""

from __future__ import annotations

import numpy as np
from scipy import signal


ADDITIVE_FAMILIES = (
    "gaussian_white",
    "colored_correlated",
    "powerline_50hz",
    "powerline_60hz",
    "baseline_shift_wander",
    "impulsive_spikes",
    "motion_transients",
    "combined_additive",
)
NONADDITIVE_FAMILIES = ("temporal_shift", "amplitude_perturbation")
ALL_FAMILIES = ADDITIVE_FAMILIES + NONADDITIVE_FAMILIES


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _fit_rms(noise: np.ndarray, target_rms: float, center: bool = True) -> np.ndarray:
    noise = np.asarray(noise, dtype=np.float64)
    if center:
        noise = noise - float(np.mean(noise))
    actual = _rms(noise)
    if actual <= np.finfo(float).eps:
        return np.zeros_like(noise)
    return noise * (target_rms / actual)


def _additive_noise(family: str, n: int, fs: float, rng: np.random.Generator, line_hz: float | None = None) -> np.ndarray:
    if family == "gaussian_white":
        return rng.normal(size=n)
    if family == "colored_correlated":
        white = rng.normal(size=n)
        cutoff_hz = min(200.0, 0.35 * fs)
        taps = signal.firwin(31, cutoff_hz / (fs / 2.0), window="hann")
        return signal.lfilter(taps, [1.0], white)
    if family in {"powerline_50hz", "powerline_60hz"}:
        frequency = 50.0 if family.endswith("50hz") else 60.0
        if frequency >= fs / 2.0:
            raise ValueError("Power-line corruption is unsupported above the Nyquist frequency.")
        phase = float(rng.uniform(0, 2 * np.pi))
        return np.sin(2 * np.pi * frequency * np.arange(n, dtype=np.float64) / fs + phase)
    if family == "baseline_shift_wander":
        time_s = np.arange(n, dtype=np.float64) / fs
        frequency = float(rng.uniform(0.2, min(5.0, fs * 0.1)))
        phase = float(rng.uniform(0, 2 * np.pi))
        dc = float(rng.choice((-1.0, 1.0))) * 0.5
        return dc + np.sin(2 * np.pi * frequency * time_s + phase)
    if family == "impulsive_spikes":
        noise = np.zeros(n, dtype=np.float64)
        count = max(1, int(round(0.002 * n)))
        indices = rng.choice(n, size=min(count, n), replace=False)
        noise[indices] = rng.choice((-1.0, 1.0), size=len(indices))
        return noise
    if family == "motion_transients":
        time_s = np.arange(n, dtype=np.float64) / fs
        noise = np.zeros(n, dtype=np.float64)
        pulse_count = max(1, int(round(n / max(1.0, fs * 0.25))))
        for _ in range(pulse_count):
            center = int(rng.integers(0, n))
            duration = float(rng.uniform(0.05, 0.2))
            width = max(1, int(round(duration * fs)))
            start = max(0, center - width // 2)
            end = min(n, start + width)
            local_t = time_s[start:end] - time_s[center]
            sigma = max(duration / 4.0, 1.0 / fs)
            noise[start:end] += float(rng.choice((-1.0, 1.0))) * np.exp(-0.5 * (local_t / sigma) ** 2)
        return noise
    if family == "combined_additive":
        families: tuple[str, ...] = (
            "gaussian_white",
            "colored_correlated",
            "baseline_shift_wander",
            "impulsive_spikes",
            "motion_transients",
            "powerline_60hz" if line_hz != 50.0 else "powerline_50hz",
        )
        components = [_fit_rms(_additive_noise(name, n, fs, rng, line_hz), 1.0) for name in families]
        return np.sum(components, axis=0)
    raise ValueError(f"Unknown additive corruption family: {family}")


def corrupt_signal(
    values: np.ndarray,
    family: str,
    severity: float,
    seed: int,
    sampling_rate_hz: float,
    line_hz: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply one corruption with additive RMS tied to the clean window RMS."""
    if not 0 <= severity <= 0.5:
        raise ValueError("Corruption severity must be between 0 and 0.5.")
    clean = np.asarray(values, dtype=np.float64)
    if clean.ndim != 1 or clean.size < 2 or not np.isfinite(clean).all():
        raise ValueError("Corruption input must be a finite one-dimensional signal window.")
    if severity == 0:
        return clean.astype(np.float32), {
            "family": "clean",
            "requested_severity": 0.0,
            "achieved_corruption_ratio": 0.0,
            "snr_db": float("inf"),
            "seed": int(seed),
            "sampling_rate_hz": float(sampling_rate_hz),
        }

    rng = np.random.default_rng(seed)
    clean_rms = max(_rms(clean), np.finfo(float).eps)
    target_noise_rms = severity * clean_rms
    result = clean.copy()
    details: dict[str, object] = {}
    if family in ADDITIVE_FAMILIES:
        base = _additive_noise(family, clean.size, sampling_rate_hz, rng, line_hz)
        noise = _fit_rms(base, target_noise_rms, center=family != "baseline_shift_wander")
        result = clean + noise
        details["achieved_noise_rms"] = _rms(noise)
        details["component_snr_db"] = 20.0 * np.log10(clean_rms / max(_rms(noise), np.finfo(float).eps))
        details["line_frequency_hz"] = (50.0 if family.endswith("50hz") else 60.0) if family.startswith("powerline_") else line_hz
        details["impulse_fraction"] = 0.002 if family == "impulsive_spikes" else None
        details["motion_duration_s"] = "random 50-200 ms Gaussian-shaped transients" if family == "motion_transients" else None
    elif family == "amplitude_perturbation":
        gain = float(1.0 + rng.choice((-1.0, 1.0)) * severity)
        result = clean * gain
        details["gain"] = gain
    elif family == "temporal_shift":
        maximum_shift_s = min(0.025, 0.05 * severity)
        max_samples = max(1, int(round(maximum_shift_s * sampling_rate_hz)))
        shift = int(rng.integers(-max_samples, max_samples + 1))
        positions = np.arange(clean.size, dtype=np.float64)
        result = np.interp(positions - shift, positions, clean, left=clean[0], right=clean[-1])
        details["shift_samples"] = shift
        details["shift_ms"] = 1000.0 * shift / sampling_rate_hz
        details["maximum_shift_ms"] = 1000.0 * maximum_shift_s
    else:
        raise ValueError(f"Unknown corruption family: {family}")

    delta_rms = _rms(result - clean)
    ratio = delta_rms / clean_rms
    return result.astype(np.float32), {
        "family": family,
        "requested_severity": float(severity),
        "achieved_corruption_ratio": float(ratio),
        "snr_db": float(20.0 * np.log10(clean_rms / max(delta_rms, np.finfo(float).eps))),
        "seed": int(seed),
        "sampling_rate_hz": float(sampling_rate_hz),
        **details,
    }


def augmentation_recipe(seed: int, levels: tuple[float, ...], line_hz: float | None) -> dict[str, object]:
    return {
        "training_severity_levels": list(levels),
        "families": list(ALL_FAMILIES),
        "additive_definition": "noise RMS / clean window RMS; achieved ratio and SNR are recorded per corruption",
        "colored_noise": "31-tap Hann-window low-pass FIR filtered white noise, RMS-normalized",
        "powerline": "separate 50 Hz and 60 Hz sinusoidal variants, each RMS-normalized; notch is independent and evidence-selected",
        "baseline": "random DC shift plus a 0.2-5 Hz sinusoidal wander, total RMS normalized",
        "impulsive": "random signed spikes at approximately 0.2% of samples, RMS-normalized",
        "motion": "random signed Gaussian-shaped 50-200 ms transients, RMS-normalized",
        "temporal_shift": "severity-scaled translation capped at 25 ms, edge-held interpolation",
        "amplitude_perturbation": "multiplicative gain 1 +/- severity",
        "combined_additive": "equal-RMS independent mixture of Gaussian, colored, baseline, impulse, motion, and one line-frequency component; total mixture RMS normalized",
        "selected_notch_frequency_hz": line_hz,
        "seed": int(seed),
    }
