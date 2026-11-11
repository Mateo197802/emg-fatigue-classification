"""Fresh temporal, spectral, and Haar-wavelet features for processed windows."""

from __future__ import annotations

import numpy as np
from scipy import signal, stats


TIME_FEATURES = (
    "rms",
    "mav",
    "iemg",
    "variance",
    "waveform_length",
    "waveform_length_per_second",
    "zero_crossings",
    "slope_sign_changes",
    "willison_amplitude",
    "median_absolute_deviation",
    "interquartile_range",
    "skewness",
    "kurtosis_excess",
    "hjorth_activity",
    "hjorth_mobility_hz",
    "hjorth_complexity",
)

SPECTRAL_FEATURES = (
    "median_frequency_hz",
    "mean_frequency_hz",
    "spectral_entropy",
    "high_low_power_ratio_150_400_over_20_150",
    "line_excluded_low_high_log_ratio",
    "line_excluded_low_frequency_fraction_20_80",
    "band_power_20_60",
    "band_power_60_120",
    "band_power_120_250",
    "band_power_250_400",
    "relative_band_power_20_60",
    "relative_band_power_60_120",
    "relative_band_power_120_250",
    "relative_band_power_250_400",
    "haar_detail_energy_ratio_d1",
    "haar_detail_energy_ratio_d2",
    "haar_detail_energy_ratio_d3",
    "haar_detail_energy_ratio_d4",
    "haar_detail_energy_ratio_d5",
    "haar_approx_energy_ratio_a5",
)

FEATURE_VIEWS: dict[str, tuple[str, ...]] = {
    "A1_morphology_only": TIME_FEATURES,
    "spectral_only": SPECTRAL_FEATURES,
    "combined_diagnostic": TIME_FEATURES + SPECTRAL_FEATURES,
}

LABEL_DEFINING_FEATURES = frozenset(
    {
        "median_frequency_hz",
        "mean_frequency_hz",
        "line_excluded_low_high_log_ratio",
        "line_excluded_low_frequency_fraction_20_80",
    }
)

BANDS_HZ = ((20.0, 60.0), (60.0, 120.0), (120.0, 250.0), (250.0, 400.0))


def window_key(window) -> str:
    """Stable key independent of window enumeration or overlap sensitivity."""
    duration_ms = int(round(float(window.duration_s) * 1000.0))
    return (
        f"{window.group_id}:{window.recording_id}:s{window.segment_index}:"
        f"d{duration_ms:04d}:p{int(window.start_sample):010d}"
    )


def feature_schema() -> list[dict[str, object]]:
    """Return formulas, units, interpretation, and feature-view boundaries."""
    descriptions = {
        "rms": ("sqrt(mean(x^2))", "ADC-derived units", "Window amplitude; sensor calibration is not assumed."),
        "mav": ("mean(abs(x))", "ADC-derived units", "Mean absolute window amplitude."),
        "iemg": ("sum(abs(x))/fs", "ADC-derived unit-seconds", "Integrated absolute amplitude."),
        "variance": ("mean((x-mean(x))^2)", "ADC-derived units^2", "Window amplitude dispersion."),
        "waveform_length": ("sum(abs(diff(x)))", "ADC-derived units", "Cumulative waveform change."),
        "waveform_length_per_second": ("waveform_length * fs / n", "ADC-derived units/s", "Duration-normalized waveform change."),
        "zero_crossings": ("count(sign changes where both samples exceed 0.01 * RMS)", "count", "Oscillation morphology; not movement phase."),
        "slope_sign_changes": ("count(local slope reversals with both slopes > 0.01 * RMS)", "count", "Local waveform-shape changes."),
        "willison_amplitude": ("count(abs(diff(x)) > 0.01 * RMS)", "count", "Large adjacent-sample changes under a fixed relative threshold."),
        "median_absolute_deviation": ("median(abs(x - median(x)))", "ADC-derived units", "Robust amplitude dispersion."),
        "interquartile_range": ("q75(x) - q25(x)", "ADC-derived units", "Central amplitude spread."),
        "skewness": ("mean(((x-mean(x))/sd(x))^3)", "dimensionless", "Amplitude-distribution asymmetry."),
        "kurtosis_excess": ("mean(((x-mean(x))/sd(x))^4) - 3", "dimensionless", "Amplitude-distribution tail weight."),
        "hjorth_activity": ("var(x)", "ADC-derived units^2", "Signal activity."),
        "hjorth_mobility_hz": ("fs * sqrt(var(diff(x)) / var(x))", "Hz", "Rate of signal change."),
        "hjorth_complexity": ("mobility(diff(x)) / mobility(x)", "dimensionless", "Normalized change in mobility."),
        "median_frequency_hz": ("first f where cumulative in-band PSD reaches 0.5 total", "Hz", "Label indicator used only in spectral/combined diagnostics."),
        "mean_frequency_hz": ("sum(f * PSD) / sum(PSD) over 20-400 Hz", "Hz", "Label indicator used only in spectral/combined diagnostics."),
        "spectral_entropy": ("-sum(p * log2(p)) / log2(K), p=PSD/sum(PSD)", "dimensionless", "Normalized in-band spectral spread."),
        "high_low_power_ratio_150_400_over_20_150": ("P(150-400 Hz) / P(20-150 Hz)", "dimensionless", "Spectral balance diagnostic."),
        "line_excluded_low_high_log_ratio": ("ln(P(20-80 Hz) / P(80-400 Hz)); detected line plus/minus 2 Hz excluded", "log ratio", "Label indicator; technical line bins are omitted."),
        "line_excluded_low_frequency_fraction_20_80": ("P(20-80 Hz) / P(20-400 Hz); detected line plus/minus 2 Hz excluded", "fraction", "Alternative low-frequency label indicator."),
    }
    rows: list[dict[str, object]] = []
    described_names = tuple(name for name in TIME_FEATURES + SPECTRAL_FEATURES if name in descriptions)
    for name in described_names:
        formula, units, interpretation = descriptions[name]
        rows.append(
            {
                "name": name,
                "formula": formula,
                "units": units,
                "interpretation": interpretation,
                "family": "temporal_morphology" if name in TIME_FEATURES else "spectral_or_time_frequency",
                "feature_views": [view for view, names in FEATURE_VIEWS.items() if name in names],
                "label_defining": name in LABEL_DEFINING_FEATURES,
            }
        )
    for low_hz, high_hz in BANDS_HZ:
        suffix = f"{int(low_hz)}_{int(high_hz)}"
        for prefix, formula, units, interpretation in (
            ("band_power", f"sum(PSD * df) in [{low_hz}, {high_hz}) Hz", "ADC-derived units^2", "Welch power band."),
            ("relative_band_power", f"band_power_{suffix} / total_power_20_400", "fraction", "Band share of supported power."),
        ):
            name = f"{prefix}_{suffix}"
            rows.append(
                {
                    "name": name,
                    "formula": formula,
                    "units": units,
                    "interpretation": interpretation,
                    "family": "spectral_or_time_frequency",
                    "feature_views": [view for view, names in FEATURE_VIEWS.items() if name in names],
                    "label_defining": False,
                }
            )
    for level in range(1, 6):
        name = f"haar_detail_energy_ratio_d{level}"
        rows.append(
            {
                "name": name,
                "formula": f"sum(Haar detail D{level}^2) / sum(all level-5 Haar coefficient energies)",
                "units": "fraction",
                "interpretation": "Decimated orthonormal Haar wavelet energy share; boundary-free for even-length windows.",
                "family": "spectral_or_time_frequency",
                "feature_views": ["spectral_only", "combined_diagnostic"],
                "label_defining": False,
            }
        )
    rows.append(
        {
            "name": "haar_approx_energy_ratio_a5",
            "formula": "sum(Haar approximation A5^2) / sum(all level-5 Haar coefficient energies)",
            "units": "fraction",
            "interpretation": "Coarse-scale decimated orthonormal Haar energy share.",
            "family": "spectral_or_time_frequency",
            "feature_views": ["spectral_only", "combined_diagnostic"],
            "label_defining": False,
        }
    )
    return rows


def _haar_energy_ratios(centered: np.ndarray, levels: int = 5) -> dict[str, float]:
    approximation = np.asarray(centered, dtype=np.float64)
    energies: dict[str, float] = {}
    for level in range(1, levels + 1):
        if approximation.size < 2:
            energies[f"haar_detail_energy_ratio_d{level}"] = float("nan")
            continue
        if approximation.size % 2:
            approximation = np.pad(approximation, (0, 1), mode="reflect")
        even = approximation[0::2]
        odd = approximation[1::2]
        detail = (even - odd) / np.sqrt(2.0)
        approximation = (even + odd) / np.sqrt(2.0)
        energies[f"haar_detail_energy_ratio_d{level}"] = float(np.sum(detail**2))
    energies["haar_approx_energy_ratio_a5"] = float(np.sum(approximation**2))
    total = sum(energies.values())
    if total <= np.finfo(float).eps:
        return {name: float("nan") for name in energies}
    return {name: value / total for name, value in energies.items()}


def _band_power(frequency: np.ndarray, psd: np.ndarray, low: float, high: float, line_frequency_hz: float | None, line_exclusion_hz: float) -> float:
    mask = (frequency >= low) & (frequency < high)
    if line_frequency_hz is not None:
        mask &= np.abs(frequency - line_frequency_hz) > line_exclusion_hz
    if mask.sum() < 1 or frequency.size < 2:
        return 0.0
    resolution = float(frequency[1] - frequency[0])
    return float(np.sum(np.maximum(psd[mask], 0.0)) * resolution)


def extract_window_features(
    values: np.ndarray,
    fs_hz: float,
    *,
    line_frequency_hz: float | None = None,
    line_exclusion_hz: float = 2.0,
) -> dict[str, float | int | str]:
    """Calculate deterministic features from one processed signal window."""
    x = np.asarray(values, dtype=np.float64)
    fs = float(fs_hz)
    if x.ndim != 1 or x.size < 32 or not np.isfinite(x).all() or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Feature window must be a finite 1D signal with at least 32 samples and positive fs.")
    if line_frequency_hz is not None and (not np.isfinite(line_frequency_hz) or line_frequency_hz <= 0):
        raise ValueError("Line frequency must be positive and finite when supplied.")
    mean = float(np.mean(x))
    centered = x - mean
    variance = float(np.mean(centered**2))
    rms = float(np.sqrt(np.mean(x**2)))
    mav = float(np.mean(np.abs(x)))
    dx = np.diff(x)
    ddx = np.diff(dx)
    threshold = 0.01 * rms
    crossing = ((x[:-1] > threshold) & (x[1:] < -threshold)) | ((x[:-1] < -threshold) & (x[1:] > threshold))
    slope_changes = (dx[:-1] * dx[1:] < 0) & (np.abs(dx[:-1]) > threshold) & (np.abs(dx[1:]) > threshold)
    diff_variance = float(np.var(dx)) if dx.size else 0.0
    hjorth_mobility = fs * np.sqrt(diff_variance / variance) if variance > 0 and diff_variance > 0 else float("nan")
    second_diff_variance = float(np.var(ddx)) if ddx.size else 0.0
    derivative_mobility = fs * np.sqrt(second_diff_variance / diff_variance) if diff_variance > 0 and second_diff_variance > 0 else float("nan")
    hjorth_complexity = derivative_mobility / hjorth_mobility if np.isfinite(hjorth_mobility) and hjorth_mobility > 0 and np.isfinite(derivative_mobility) else float("nan")
    standard_deviation = float(np.sqrt(variance))
    if standard_deviation > 0:
        standardized = centered / standard_deviation
        skewness = float(np.mean(standardized**3))
        kurtosis = float(np.mean(standardized**4) - 3.0)
    else:
        skewness = kurtosis = float("nan")

    result: dict[str, float | int | str] = {
        "feature_valid": int(variance > np.finfo(float).eps),
        "feature_status": "valid" if variance > np.finfo(float).eps else "zero_energy",
        "rms": rms,
        "mav": mav,
        "iemg": float(np.sum(np.abs(x)) / fs),
        "variance": variance,
        "waveform_length": float(np.sum(np.abs(dx))),
        "waveform_length_per_second": float(np.sum(np.abs(dx)) * fs / x.size),
        "zero_crossings": float(crossing.sum()),
        "slope_sign_changes": float(slope_changes.sum()),
        "willison_amplitude": float(np.sum(np.abs(dx) > threshold)),
        "median_absolute_deviation": float(np.median(np.abs(x - np.median(x)))),
        "interquartile_range": float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
        "skewness": skewness,
        "kurtosis_excess": kurtosis,
        "hjorth_activity": variance,
        "hjorth_mobility_hz": float(hjorth_mobility),
        "hjorth_complexity": float(hjorth_complexity),
    }

    frequency, psd = signal.welch(x, fs=fs, nperseg=min(512, x.size), scaling="density")
    high_frequency = min(400.0, fs / 2.0)
    supported = (frequency >= 20.0) & (frequency <= high_frequency)
    supported_frequency = frequency[supported]
    supported_psd = np.maximum(psd[supported], 0.0)
    total_supported = float(supported_psd.sum())
    if supported_frequency.size >= 2 and total_supported > np.finfo(float).eps:
        probabilities = supported_psd / total_supported
        positive_probabilities = probabilities[probabilities > 0]
        entropy = float(-np.sum(positive_probabilities * np.log2(positive_probabilities)) / np.log2(probabilities.size))
        mean_frequency = float(np.sum(supported_frequency * probabilities))
        median_frequency = float(supported_frequency[min(np.searchsorted(np.cumsum(probabilities), 0.5, side="left"), supported_frequency.size - 1)])
        total_power = total_supported * float(frequency[1] - frequency[0])
    else:
        entropy = mean_frequency = median_frequency = total_power = float("nan")
    result["mean_frequency_hz"] = mean_frequency
    result["median_frequency_hz"] = median_frequency
    result["spectral_entropy"] = entropy

    p20_60 = _band_power(frequency, psd, 20.0, 60.0, None, line_exclusion_hz)
    p60_120 = _band_power(frequency, psd, 60.0, 120.0, None, line_exclusion_hz)
    p120_250 = _band_power(frequency, psd, 120.0, 250.0, None, line_exclusion_hz)
    p250_400 = _band_power(frequency, psd, 250.0, high_frequency + np.finfo(float).eps, None, line_exclusion_hz)
    for suffix, value in (("20_60", p20_60), ("60_120", p60_120), ("120_250", p120_250), ("250_400", p250_400)):
        result[f"band_power_{suffix}"] = value
        result[f"relative_band_power_{suffix}"] = value / total_power if np.isfinite(total_power) and total_power > 0 else float("nan")
    low_power = _band_power(frequency, psd, 20.0, 80.0, line_frequency_hz, line_exclusion_hz)
    high_power = _band_power(frequency, psd, 80.0, high_frequency + np.finfo(float).eps, line_frequency_hz, line_exclusion_hz)
    p20_150 = _band_power(frequency, psd, 20.0, 150.0, None, line_exclusion_hz)
    p150_400 = _band_power(frequency, psd, 150.0, high_frequency + np.finfo(float).eps, None, line_exclusion_hz)
    result["high_low_power_ratio_150_400_over_20_150"] = p150_400 / p20_150 if p20_150 > 0 else float("nan")
    result["line_excluded_low_high_log_ratio"] = float(np.log(low_power / high_power)) if low_power > 0 and high_power > 0 else float("nan")
    result["line_excluded_low_frequency_fraction_20_80"] = low_power / (low_power + high_power) if low_power + high_power > 0 else float("nan")
    result.update(_haar_energy_ratios(centered))
    return result
