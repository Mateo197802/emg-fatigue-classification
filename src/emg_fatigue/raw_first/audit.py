"""Acquisition timing, signal-quality, ADC/voltage, and PSD audit."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import signal

from .contracts import RawRecording


def _longest_repeat(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.r_[0, changes, values.size]
    return int(np.diff(bounds).max(initial=0))


def _linear_relation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    predicted = slope * x[valid] + intercept
    residual = y[valid] - predicted
    ss_total = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_total if ss_total > 0 else float("nan")
    correlation = float(np.corrcoef(x[valid], y[valid])[0, 1])
    return float(slope), float(intercept), r2, correlation


def _valid_timing(time_s: np.ndarray) -> tuple[np.ndarray, float, dict[str, float | int]]:
    finite = np.isfinite(time_s)
    t = time_s[finite]
    if t.size < 2:
        return finite, float("nan"), {
            "timestamp_nonfinite_count": int((~finite).sum()),
            "timestamp_duplicate_count": 0,
            "clock_reset_count": 0,
            "gap_count": 0,
            "estimated_missing_samples": 0,
            "interval_median_s": float("nan"),
            "interval_p01_s": float("nan"),
            "interval_p99_s": float("nan"),
            "interval_mad_s": float("nan"),
            "jitter_mad_percent": float("nan"),
            "gap_threshold_s": float("nan"),
        }

    dt = np.diff(t)
    positive = dt[dt > 0]
    median_dt = float(np.median(positive)) if positive.size else float("nan")
    mad_dt = float(np.median(np.abs(positive - median_dt))) if positive.size else float("nan")
    gap_threshold = max(5.0 * median_dt, median_dt + 8.0 * 1.4826 * mad_dt) if np.isfinite(median_dt) else float("nan")
    gaps = dt > gap_threshold if np.isfinite(gap_threshold) else np.zeros(dt.shape, dtype=bool)
    valid_intervals = positive[positive <= gap_threshold] if np.isfinite(gap_threshold) else positive
    fs = 1.0 / float(np.median(valid_intervals)) if valid_intervals.size else float("nan")
    estimated_missing = int(np.maximum(np.rint(dt[gaps] / median_dt).astype(int) - 1, 0).sum()) if median_dt > 0 else 0
    stats: dict[str, float | int] = {
        "timestamp_nonfinite_count": int((~finite).sum()),
        "timestamp_duplicate_count": int((dt == 0).sum()),
        "clock_reset_count": int((dt < 0).sum()),
        "gap_count": int(gaps.sum()),
        "estimated_missing_samples": estimated_missing,
        "interval_median_s": median_dt,
        "interval_p01_s": float(np.quantile(valid_intervals, 0.01)) if valid_intervals.size else float("nan"),
        "interval_p99_s": float(np.quantile(valid_intervals, 0.99)) if valid_intervals.size else float("nan"),
        "interval_mad_s": mad_dt,
        "jitter_mad_percent": 100.0 * mad_dt / median_dt if median_dt > 0 else float("nan"),
        "gap_threshold_s": float(gap_threshold),
    }
    return finite, fs, stats


def _longest_contiguous_signal(recording: RawRecording, fs: float, gap_threshold_s: float) -> np.ndarray:
    t = recording.relative_time_s
    x = recording.adc_value
    best_start = 0
    best_end = 0
    run_start = 0
    for index in range(1, len(t)):
        dt = t[index] - t[index - 1]
        broken = (
            not np.isfinite(t[index - 1])
            or not np.isfinite(t[index])
            or not np.isfinite(x[index - 1])
            or not np.isfinite(x[index])
            or dt <= 0
            or (np.isfinite(gap_threshold_s) and dt > gap_threshold_s)
        )
        if broken:
            if index - run_start > best_end - best_start:
                best_start, best_end = run_start, index
            run_start = index
    if len(t) - run_start > best_end - best_start:
        best_start, best_end = run_start, len(t)
    segment = x[best_start:best_end]
    return segment[np.isfinite(segment)] if fs > 0 else np.asarray([], dtype=np.float64)


def _spectral_summary(x: np.ndarray, fs: float) -> dict[str, float | bool]:
    empty: dict[str, float | bool] = {
        "psd_peak_hz": float("nan"),
        "spectral_median_hz": float("nan"),
        "power_20_400_fraction": float("nan"),
        "line_50_excess_db": float("nan"),
        "line_60_excess_db": float("nan"),
        "line_50_detected": False,
        "line_60_detected": False,
    }
    if x.size < 256 or not np.isfinite(fs) or fs <= 0:
        return empty
    centered = signal.detrend(x, type="linear")
    freq, power = signal.welch(centered, fs=fs, nperseg=min(4096, x.size), noverlap=None, scaling="density")
    valid = (freq > 0) & np.isfinite(power)
    if valid.sum() < 3 or float(power[valid].sum()) <= 0:
        return empty
    pv = power[valid]
    fv = freq[valid]
    cdf = np.cumsum(pv) / pv.sum()
    median_frequency = float(fv[np.searchsorted(cdf, 0.5, side="left")])
    passband = (freq >= 20.0) & (freq <= min(400.0, fs * 0.45))
    denominator = float(power[valid].sum())
    passband_fraction = float(power[passband].sum() / denominator) if passband.any() else float("nan")
    result: dict[str, float | bool] = {
        "psd_peak_hz": float(freq[np.argmax(power[valid]) + np.flatnonzero(valid)[0]]),
        "spectral_median_hz": median_frequency,
        "power_20_400_fraction": passband_fraction,
    }
    for line_hz in (50.0, 60.0):
        line_band = (freq >= line_hz - 1.0) & (freq <= line_hz + 1.0)
        background = ((freq >= line_hz - 5.0) & (freq <= line_hz - 2.0)) | ((freq >= line_hz + 2.0) & (freq <= line_hz + 5.0))
        if line_band.any() and background.any() and fs / 2 > line_hz + 5:
            line_power = float(power[line_band].max())
            background_power = float(np.median(power[background]))
            excess = 10.0 * np.log10(max(line_power, np.finfo(float).tiny) / max(background_power, np.finfo(float).tiny))
        else:
            excess = float("nan")
        result[f"line_{int(line_hz)}_excess_db"] = float(excess)
        result[f"line_{int(line_hz)}_detected"] = bool(np.isfinite(excess) and excess >= 6.0)
    return result


def audit_recording(recording: RawRecording) -> dict[str, object]:
    t = recording.relative_time_s
    x = recording.adc_value
    v = recording.voltage_mv
    finite_t, fs, timing = _valid_timing(t)
    finite_x = np.isfinite(x)
    valid_x = x[finite_x]
    finite_pair = finite_x & np.isfinite(v)
    slope, intercept, r2, corr = _linear_relation(x[finite_pair], v[finite_pair])
    duration = float(np.nanmax(t) - np.nanmin(t)) if finite_t.any() else float("nan")
    adc_rms = float(np.sqrt(np.mean(valid_x**2))) if valid_x.size else float("nan")
    longest = _longest_contiguous_signal(recording, fs, float(timing["gap_threshold_s"]))
    spectrum = _spectral_summary(longest, fs)
    scale = float(np.max(np.abs(valid_x))) if valid_x.size else float("nan")
    int_like = float(np.mean(np.isclose(valid_x, np.rint(valid_x), atol=1e-6))) if valid_x.size else float("nan")
    row: dict[str, object] = {
        "group_id": recording.group_id,
        "recording_id": recording.recording_id,
        "sample_count": int(x.size),
        "finite_signal_sample_count": int(finite_x.sum()),
        "finite_voltage_sample_count": int(np.isfinite(v).sum()),
        "duration_s": duration,
        "effective_sampling_rate_hz": fs,
        "nyquist_hz": fs / 2.0 if np.isfinite(fs) else float("nan"),
        "adc_min": float(np.min(valid_x)) if valid_x.size else float("nan"),
        "adc_max": float(np.max(valid_x)) if valid_x.size else float("nan"),
        "adc_mean": float(np.mean(valid_x)) if valid_x.size else float("nan"),
        "adc_std": float(np.std(valid_x)) if valid_x.size else float("nan"),
        "adc_rms": adc_rms,
        "adc_zero_fraction": float(np.mean(valid_x == 0)) if valid_x.size else float("nan"),
        "adc_integer_like_fraction": int_like,
        "adc_unique_fraction": float(np.unique(valid_x).size / valid_x.size) if valid_x.size else float("nan"),
        "adc_max_exact_repeat_samples": _longest_repeat(valid_x),
        "voltage_min_mv": float(np.nanmin(v)) if np.isfinite(v).any() else float("nan"),
        "voltage_max_mv": float(np.nanmax(v)) if np.isfinite(v).any() else float("nan"),
        "adc_to_voltage_slope_mv_per_unit": slope,
        "adc_to_voltage_intercept_mv": intercept,
        "adc_to_voltage_r2": r2,
        "adc_to_voltage_correlation": corr,
        "adc_to_voltage_physical_calibration_verified": False,
        "source_sha256": recording.source_sha256,
        **timing,
        **spectrum,
    }
    return row


def audit_all(recordings: Iterable[RawRecording]) -> list[dict[str, object]]:
    return [audit_recording(recording) for recording in recordings]


def choose_sampling_plan(audits: list[dict[str, object]]) -> dict[str, object]:
    rates = np.asarray([row["effective_sampling_rate_hz"] for row in audits], dtype=np.float64)
    rates = rates[np.isfinite(rates) & (rates > 0)]
    if rates.size == 0:
        raise ValueError("No valid effective sampling rate could be estimated from timestamps.")
    median_rate = float(np.median(rates))
    within_1000 = float(np.mean(np.abs(rates - 1000.0) / 1000.0 <= 0.02))
    if abs(median_rate - 1000.0) / 1000.0 <= 0.01 and within_1000 >= 0.95:
        target_rate = 1000.0
        target_reason = "1000 Hz supported: median is within 1% and at least 95% of valid recordings are within 2%."
    else:
        target_rate = float(round(median_rate))
        target_reason = "1000 Hz support threshold not met; rounded robust median measured rate selected as common rate."
    min_rate = float(np.min(rates))
    high_cut = min(400.0, target_rate * 0.45, min_rate * 0.45)
    low_cut = 20.0 if high_cut > 20.0 else min(5.0, high_cut * 0.25)
    line50 = np.mean([bool(row.get("line_50_detected", False)) for row in audits])
    line60 = np.mean([bool(row.get("line_60_detected", False)) for row in audits])
    line_hz = 50.0 if line50 >= 0.20 and line50 >= line60 else (60.0 if line60 >= 0.20 else None)
    return {
        "target_sampling_rate_hz": target_rate,
        "target_rate_selection": target_reason,
        "valid_recording_rate_count": int(rates.size),
        "effective_rate_min_hz": min_rate,
        "effective_rate_median_hz": median_rate,
        "effective_rate_max_hz": float(np.max(rates)),
        "fraction_within_2_percent_of_1000_hz": within_1000,
        "bandpass_low_hz": low_cut,
        "bandpass_high_hz": high_cut,
        "notch_frequency_hz": line_hz,
        "notch_selection": "Detected PSD line peak in at least 20% of recordings at 50/60 Hz." if line_hz else "No line frequency crossed the prespecified prevalence threshold; no notch applied.",
    }
