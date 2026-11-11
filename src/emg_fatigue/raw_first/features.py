"""Temporal, spectral, and Haar wavelet features for processed windows."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .contracts import SignalWindow


BANDS_HZ = ((20.0, 60.0), (60.0, 120.0), (120.0, 250.0), (250.0, 400.0))


def feature_schema() -> list[dict[str, str]]:
    rows = [
        ("rms", "sqrt(mean(x^2))", "ADC-derived units", "Amplitude; not calibrated physical voltage."),
        ("mav", "mean(abs(x))", "ADC-derived units", "Mean absolute amplitude."),
        ("iemg", "sum(abs(x))/fs", "ADC-derived units·s", "Integrated absolute amplitude."),
        ("variance", "mean((x-mean(x))^2)", "ADC-derived units^2", "Window variance."),
        ("waveform_length", "sum(abs(diff(x)))", "ADC-derived units", "Cumulative waveform change."),
        ("zero_crossings", "count(x[i]*x[i+1] < 0)", "count", "Sign changes about zero."),
        ("slope_sign_changes", "count(diff[i]*diff[i+1] < 0)", "count", "Local slope reversals."),
        ("willison_amplitude", "count(abs(diff(x)) > train_threshold)", "count", "Threshold is 1% of train-window median RMS."),
        ("hjorth_activity", "var(x)", "ADC-derived units^2", "Signal activity."),
        ("hjorth_mobility", "sqrt(var(diff(x))/var(x))", "per sample", "Relative rate of change."),
        ("hjorth_complexity", "mobility(diff(x))/mobility(x)", "dimensionless", "Normalized change in mobility."),
        ("mean_frequency_hz", "sum(f*P(f))/sum(P(f))", "Hz", "Welch PSD weighted mean."),
        ("median_frequency_hz", "first f where cumulative P(f) >= 0.5 total", "Hz", "Welch PSD median frequency."),
        ("spectral_entropy", "-sum(p*log(p))/log(K)", "dimensionless", "Normalized PSD entropy in the supported passband."),
        ("high_low_power_ratio", "P(150-400 Hz)/P(20-150 Hz)", "dimensionless", "Band ratio, clipped to Nyquist-supported frequencies."),
    ]
    for lo, hi in BANDS_HZ:
        rows.append((f"band_power_{int(lo)}_{int(hi)}_hz", f"integral(P(f), {lo}-{hi} Hz)", "ADC-derived units^2", "Welch PSD integral, limited by Nyquist."))
        rows.append((f"relative_band_power_{int(lo)}_{int(hi)}_hz", f"band_power({lo}-{hi} Hz)/total_supported_power", "fraction", "Relative to the supported 20 Hz-to-high-cut band."))
    for level in range(1, 5):
        rows.append((f"haar_detail_energy_ratio_l{level}", f"sum(detail_l{level}^2)/sum(x^2)", "fraction", "Orthonormal Haar detail energy ratio."))
    return [{"name": n, "formula": f, "units": u, "interpretation": i} for n, f, u, i in rows]


def _haar_detail_ratios(x: np.ndarray, levels: int = 4) -> dict[str, float]:
    approximation = np.asarray(x, dtype=np.float64)
    total_energy = float(np.sum(approximation**2))
    result: dict[str, float] = {}
    if total_energy <= np.finfo(float).eps:
        return {f"haar_detail_energy_ratio_l{level}": float("nan") for level in range(1, levels + 1)}
    for level in range(1, levels + 1):
        if approximation.size < 2:
            result[f"haar_detail_energy_ratio_l{level}"] = float("nan")
            continue
        if approximation.size % 2:
            approximation = approximation[:-1]
        even = approximation[0::2]
        odd = approximation[1::2]
        detail = (even - odd) / np.sqrt(2.0)
        approximation = (even + odd) / np.sqrt(2.0)
        result[f"haar_detail_energy_ratio_l{level}"] = float(np.sum(detail**2) / total_energy)
    return result


def extract_features(window: SignalWindow, willison_threshold: float) -> dict[str, float]:
    x = np.asarray(window.signal, dtype=np.float64)
    fs = float(window.sampling_rate_hz)
    if x.size < 16 or not np.isfinite(x).all() or not np.isfinite(fs) or fs <= 0:
        return {}
    dx = np.diff(x)
    mean = float(np.mean(x))
    centered = x - mean
    variance = float(np.mean(centered**2))
    rms = float(np.sqrt(np.mean(x**2)))
    mav = float(np.mean(np.abs(x)))
    hjorth_mobility = float(np.sqrt(np.var(dx) / variance)) if variance > 0 and dx.size else float("nan")
    ddx = np.diff(dx)
    d_mobility = float(np.sqrt(np.var(ddx) / np.var(dx))) if dx.size > 1 and np.var(dx) > 0 else float("nan")
    hjorth_complexity = d_mobility / hjorth_mobility if hjorth_mobility > 0 else float("nan")

    freq, psd = signal.welch(x, fs=fs, nperseg=min(512, x.size), noverlap=None, scaling="density")
    upper_hz = min(400.0, fs * 0.45)
    passband = (freq >= min(20.0, upper_hz * 0.25)) & (freq <= upper_hz)
    f = freq[passband]
    p = np.maximum(psd[passband], 0.0)
    total_power = float(np.trapezoid(p, f)) if f.size > 1 else 0.0
    if total_power > 0 and f.size > 1:
        weights = p / p.sum()
        mean_frequency = float(np.sum(f * weights))
        cumulative = np.cumsum(weights)
        median_frequency = float(f[min(np.searchsorted(cumulative, 0.5, side="left"), f.size - 1)])
        nonzero = weights[weights > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(max(2, weights.size)))
    else:
        mean_frequency = median_frequency = entropy = float("nan")

    result = {
        "rms": rms,
        "mav": mav,
        "iemg": float(np.sum(np.abs(x)) / fs),
        "variance": variance,
        "waveform_length": float(np.sum(np.abs(dx))),
        "zero_crossings": float(np.sum(x[:-1] * x[1:] < 0)),
        "slope_sign_changes": float(np.sum(dx[:-1] * dx[1:] < 0)) if dx.size > 1 else 0.0,
        "willison_amplitude": float(np.sum(np.abs(dx) > willison_threshold)),
        "hjorth_activity": variance,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
        "mean_frequency_hz": mean_frequency,
        "median_frequency_hz": median_frequency,
        "spectral_entropy": entropy,
        "high_low_power_ratio": float("nan"),
    }
    low = (f >= 20.0) & (f < min(150.0, upper_hz))
    high = (f >= 150.0) & (f <= upper_hz)
    low_power = float(np.trapezoid(p[low], f[low])) if low.sum() > 1 else 0.0
    high_power = float(np.trapezoid(p[high], f[high])) if high.sum() > 1 else 0.0
    result["high_low_power_ratio"] = high_power / low_power if low_power > 0 else float("nan")

    for lo, hi in BANDS_HZ:
        supported = (f >= lo) & (f <= min(hi, upper_hz))
        band_power = float(np.trapezoid(p[supported], f[supported])) if supported.sum() > 1 else 0.0
        key = f"band_power_{int(lo)}_{int(hi)}_hz"
        result[key] = band_power
        result[f"relative_{key}"] = band_power / total_power if total_power > 0 else float("nan")
    result.update(_haar_detail_ratios(centered))
    return result
