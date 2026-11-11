"""Acquisition QC, gap-preserving preprocessing, and signal-quality receipts."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import ndimage, signal

from .contracts import AcquisitionAudit, ProcessedRecording, ProcessedSegment, RawRecording


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if not values.size:
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.r_[0, changes, values.size]
    return [
        (int(start), int(stop))
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
        if values[start]
    ]


def _continuous_ranges(times: np.ndarray, values: np.ndarray, gap_threshold_s: float) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for index, (time_s, sample) in enumerate(zip(times, values, strict=True)):
        if not np.isfinite(time_s) or not np.isfinite(sample):
            if start is not None and previous is not None and previous + 1 - start >= 2:
                ranges.append((start, previous + 1))
            start = None
            previous = None
            continue
        if previous is None:
            start = index
            previous = index
            continue
        delta = float(times[index] - times[previous])
        if delta <= 0 or delta > gap_threshold_s:
            if start is not None and previous + 1 - start >= 2:
                ranges.append((start, previous + 1))
            start = index
        previous = index
    if start is not None and previous is not None and previous + 1 - start >= 2:
        ranges.append((start, previous + 1))
    return tuple(ranges)


def _above_source_rate_minimum(sampling_rate_hz: float, minimum_rate_hz: float) -> bool:
    # Float timestamp quantization can make an exact boundary rate appear a few
    # trillionths above the configured minimum (for example, 800.000000000017).
    return bool(
        sampling_rate_hz > minimum_rate_hz
        and not np.isclose(sampling_rate_hz, minimum_rate_hz, rtol=1e-9, atol=1e-9)
    )


def _adc_voltage_relationship(
    adc_value: np.ndarray, voltage_mv: np.ndarray | None
) -> tuple[int, float, float, float, float]:
    adc = np.asarray(adc_value, dtype=np.float64)
    if voltage_mv is None:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    voltage = np.asarray(voltage_mv, dtype=np.float64)
    if voltage.shape != adc.shape:
        raise ValueError("ADC and voltage arrays must have matching shapes for the acquisition audit.")
    valid = np.isfinite(adc) & np.isfinite(voltage)
    x = adc[valid]
    y = voltage[valid]
    if not x.size:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    integer_like = float(np.mean(np.abs(x - np.rint(x)) <= 1e-6))
    if x.size < 2:
        return int(x.size), float("nan"), float("nan"), float("nan"), integer_like
    centered_x = x - float(np.mean(x))
    centered_y = y - float(np.mean(y))
    sum_xx = float(np.dot(centered_x, centered_x))
    sum_yy = float(np.dot(centered_y, centered_y))
    if sum_xx <= np.finfo(float).eps or sum_yy <= np.finfo(float).eps:
        return int(x.size), float("nan"), float("nan"), float("nan"), integer_like
    slope = float(np.dot(centered_x, centered_y) / sum_xx)
    intercept = float(np.mean(y) - slope * np.mean(x))
    residual = y - (slope * x + intercept)
    r_squared = float(1.0 - np.dot(residual, residual) / sum_yy)
    return int(x.size), slope, intercept, r_squared, integer_like


def _line_prominence(frequency: np.ndarray, psd: np.ndarray, center_hz: float) -> float:
    peak_mask = np.abs(frequency - center_hz) <= 1.0
    background_mask = ((frequency >= center_hz - 8.0) & (frequency <= center_hz - 3.0)) | (
        (frequency >= center_hz + 3.0) & (frequency <= center_hz + 8.0)
    )
    if not peak_mask.any() or not background_mask.any():
        return float("nan")
    peak = float(np.max(psd[peak_mask]))
    background = float(np.median(psd[background_mask]))
    floor = np.finfo(np.float64).tiny
    return float(10.0 * np.log10(max(peak, floor) / max(background, floor)))


def _spectral_audit(
    values: np.ndarray, ranges: tuple[tuple[int, int], ...], sampling_rate_hz: float
) -> tuple[float, float, float, dict[str, float]]:
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        return float("nan"), float("nan"), float("nan"), {"50": float("nan"), "60": float("nan")}
    total_weighted = 0.0
    inband_weighted = 0.0
    low_weighted = 0.0
    high_weighted = 0.0
    line_values: dict[str, list[float]] = {"50": [], "60": []}
    for start, stop in ranges:
        segment = np.asarray(values[start:stop], dtype=np.float64)
        if segment.size < max(32, int(round(sampling_rate_hz * 0.25))):
            continue
        segment = signal.detrend(segment, type="linear")
        frequency, psd = signal.welch(
            segment,
            fs=sampling_rate_hz,
            nperseg=min(segment.size, max(256, int(round(sampling_rate_hz)))),
            scaling="density",
        )
        power = float(np.trapezoid(psd, frequency)) if frequency.size > 1 else 0.0
        if power <= np.finfo(float).eps:
            continue
        weight = float(segment.size)
        total_weighted += power * weight
        nyquist = sampling_rate_hz / 2.0
        inband_mask = (frequency >= 20.0) & (frequency <= min(400.0, nyquist))
        low_mask = frequency < 20.0
        high_mask = frequency > 400.0
        inband_power = float(np.trapezoid(psd[inband_mask], frequency[inband_mask])) if inband_mask.sum() > 1 else 0.0
        low_power = float(np.trapezoid(psd[low_mask], frequency[low_mask])) if low_mask.sum() > 1 else 0.0
        high_power = float(np.trapezoid(psd[high_mask], frequency[high_mask])) if high_mask.sum() > 1 else 0.0
        inband_weighted += inband_power * weight
        low_weighted += low_power * weight
        high_weighted += high_power * weight
        for line in (50.0, 60.0):
            line_values[str(int(line))].append(_line_prominence(frequency, psd, line))
    if total_weighted <= np.finfo(float).eps:
        return 0.0, 0.0, 0.0, {"50": float("nan"), "60": float("nan")}
    fractions = (
        inband_weighted / total_weighted,
        low_weighted / total_weighted,
        high_weighted / total_weighted,
    )
    prominence = {
        key: float(np.nanmax(values)) if values and np.isfinite(values).any() else float("nan")
        for key, values in line_values.items()
    }
    return *fractions, prominence


def _quality_mask(
    values: np.ndarray, fs_hz: float, config: dict
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    x = np.asarray(values, dtype=np.float64)
    reason_masks: dict[str, np.ndarray] = {"nonfinite_sample": ~np.isfinite(x)}
    flags: list[str] = []
    flatline_samples = max(1, int(round(float(config["flatline_duration_s"]) * fs_hz)))
    equal_steps = np.r_[False, x[1:] == x[:-1]] if x.size else np.asarray([], dtype=bool)
    for start, stop in _runs(equal_steps):
        run_start = max(0, start - 1)
        if stop - run_start >= flatline_samples:
            reason_masks.setdefault("flatline_run_at_least_100ms", np.zeros(x.size, dtype=bool))[run_start:stop] = True
            flags.append("flatline_run_at_least_100ms")

    finite = x[np.isfinite(x)]
    if finite.size:
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        extremes = np.isfinite(x) & ((x == minimum) | (x == maximum))
        minimum_run = int(config["clipping_minimum_samples"])
        minimum_fraction = float(config["clipping_minimum_fraction"])
        clipped = False
        for start, stop in _runs(extremes):
            plateau = x[start:stop]
            for extreme in (minimum, maximum):
                local = plateau == extreme
                if local.size >= minimum_run and int(local.sum()) / max(1, x.size) >= minimum_fraction:
                    reason_masks.setdefault("clipping_suspect_plateau", np.zeros(x.size, dtype=bool))[start:stop] = True
                    clipped = True
        if clipped:
            flags.append("clipping_suspect_plateau")

        width = max(5, int(round(0.051 * fs_hz)))
        if width % 2 == 0:
            width += 1
        local_median = ndimage.median_filter(np.nan_to_num(x, nan=0.0), size=width, mode="nearest")
        residual = np.abs(np.nan_to_num(x, nan=0.0) - local_median)
        local_mad = ndimage.median_filter(residual, size=width, mode="nearest") * 1.4826
        global_scale = float(np.median(np.abs(finite - np.median(finite))) * 1.4826)
        floor = max(global_scale * 0.05, np.finfo(float).eps)
        threshold = float(config["impulse_local_robust_scale_multiple"]) * np.maximum(local_mad, floor)
        impulses = np.isfinite(x) & (residual > threshold)
        if impulses.any():
            dilation = max(1, int(round(0.005 * fs_hz)))
            expanded = ndimage.binary_dilation(impulses, structure=np.ones(2 * dilation + 1, dtype=bool))
            reason_masks["impulsive_artifact"] = expanded
            flags.append("impulsive_artifact")
    mask = np.zeros(x.size, dtype=bool)
    for reason_mask in reason_masks.values():
        mask |= reason_mask
    return mask, tuple(sorted(set(flags))), reason_masks


def audit_recording(recording: RawRecording, config: dict) -> AcquisitionAudit:
    acquisition = config["acquisition"]
    times = np.asarray(recording.relative_time_s, dtype=np.float64)
    values = np.asarray(recording.adc_value, dtype=np.float64)
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
        raise ValueError("Timestamp and signal arrays must be matching one-dimensional vectors.")
    dt = np.diff(times)
    finite_pairs = np.isfinite(times[:-1]) & np.isfinite(times[1:]) & np.isfinite(values[:-1]) & np.isfinite(values[1:])
    valid_dt = dt[finite_pairs]
    positive = valid_dt[valid_dt > 0]
    median_dt = float(np.median(positive)) if positive.size else float("nan")
    gap_threshold = (
        max(float(acquisition["gap_median_dt_multiplier"]) * median_dt, float(acquisition["minimum_gap_s"]))
        if np.isfinite(median_dt)
        else float("nan")
    )
    if not np.isfinite(gap_threshold):
        ranges: tuple[tuple[int, int], ...] = ()
    else:
        ranges = _continuous_ranges(times, values, gap_threshold)
    valid_intervals = positive[positive <= gap_threshold] if np.isfinite(gap_threshold) else positive
    if valid_intervals.size:
        median_dt = float(np.median(valid_intervals))
        dt_p05 = float(np.quantile(valid_intervals, 0.05))
        dt_p95 = float(np.quantile(valid_intervals, 0.95))
        jitter = float(100.0 * (dt_p95 - dt_p05) / median_dt) if median_dt > 0 else float("inf")
        sampling_rate = float(1.0 / median_dt)
        gap_threshold = max(float(acquisition["gap_median_dt_multiplier"]) * median_dt, float(acquisition["minimum_gap_s"]))
        ranges = _continuous_ranges(times, values, gap_threshold)
    else:
        dt_p05 = dt_p95 = jitter = sampling_rate = float("nan")

    duplicate_count = int(np.sum(valid_dt == 0))
    reset_count = int(np.sum(valid_dt < 0))
    gap_count = int(np.sum(valid_dt > gap_threshold)) if np.isfinite(gap_threshold) else 0
    nyquist = float(sampling_rate / 2.0) if np.isfinite(sampling_rate) else float("nan")
    adc_voltage_pair_count, adc_voltage_slope, adc_voltage_intercept, adc_voltage_r_squared, adc_integer_like_fraction = (
        _adc_voltage_relationship(values, recording.voltage_mv)
    )
    sampling_compatible = bool(
        np.isfinite(sampling_rate)
        and _above_source_rate_minimum(sampling_rate, float(acquisition["minimum_source_rate_hz"]))
        and nyquist > float(acquisition["bandpass_high_hz"])
        and jitter <= float(acquisition["maximum_jitter_percent"])
    )
    inband, low_fraction, high_fraction, line_prominence = _spectral_audit(values, ranges, sampling_rate)
    signal_compatible = bool(
        sampling_compatible
        and inband >= float(acquisition["minimum_in_band_power_fraction"])
        and low_fraction <= float(acquisition["maximum_below_20_hz_power_fraction"])
        and high_fraction <= float(acquisition["maximum_above_400_hz_power_fraction"])
    )
    flags: list[str] = []
    if not np.isfinite(sampling_rate) or not _above_source_rate_minimum(
        sampling_rate, float(acquisition["minimum_source_rate_hz"])
    ):
        flags.append("sampling_rate_below_supported_minimum")
    if np.isfinite(jitter) and jitter > float(acquisition["maximum_jitter_percent"]):
        flags.append("timestamp_jitter_exceeds_5_percent")
    if duplicate_count:
        flags.append("duplicate_timestamps_split_into_separate_runs")
    if reset_count:
        flags.append("clock_resets_split_into_separate_runs")
    if inband < float(acquisition["minimum_in_band_power_fraction"]):
        flags.append("in_band_power_below_80_percent")
    if low_fraction > float(acquisition["maximum_below_20_hz_power_fraction"]):
        flags.append("below_20_hz_power_above_20_percent")
    if high_fraction > float(acquisition["maximum_above_400_hz_power_fraction"]):
        flags.append("above_400_hz_power_above_5_percent")
    quality_mask, artifact_flags, artifact_masks = _quality_mask(
        values, sampling_rate if np.isfinite(sampling_rate) else 1.0, acquisition
    )
    flags.extend(artifact_flags)
    notch_limit = float(acquisition["line_prominence_notch_db"])
    eligible_lines = [(float(value), float(line_prominence[str(int(value))])) for value in (50, 60) if np.isfinite(line_prominence[str(int(value))]) and line_prominence[str(int(value))] > notch_limit]
    notch_frequency = max(eligible_lines, key=lambda pair: pair[1])[0] if eligible_lines else None
    duration = float(times[-1] - times[0]) if times.size > 1 and np.isfinite(times[[0, -1]]).all() else float("nan")
    return AcquisitionAudit(
        group_id=recording.group_id,
        recording_id=recording.recording_id,
        sample_count=int(values.size),
        duration_s=duration,
        sampling_rate_hz=sampling_rate,
        median_dt_s=median_dt,
        dt_p05_s=dt_p05,
        dt_p95_s=dt_p95,
        jitter_percent=jitter,
        gap_threshold_s=gap_threshold,
        continuous_ranges=ranges,
        duplicate_timestamp_count=duplicate_count,
        clock_reset_count=reset_count,
        gap_count=gap_count,
        nyquist_hz=nyquist,
        adc_voltage_pair_count=adc_voltage_pair_count,
        adc_voltage_slope_mv_per_unit=adc_voltage_slope,
        adc_voltage_intercept_mv=adc_voltage_intercept,
        adc_voltage_r_squared=adc_voltage_r_squared,
        adc_value_integer_like_fraction=adc_integer_like_fraction,
        sampling_compatible=sampling_compatible,
        in_band_power_fraction=float(inband),
        below_20_hz_power_fraction=float(low_fraction),
        above_400_hz_power_fraction=float(high_fraction),
        signal_compatible=signal_compatible,
        line_prominence_db=line_prominence,
        notch_frequency_hz=notch_frequency,
        quality_flags=tuple(sorted(set(flags))),
        quality_invalid_mask=quality_mask,
        artifact_masks=artifact_masks,
    )


def preprocess_recording(recording: RawRecording, config: dict) -> ProcessedRecording:
    audit = audit_recording(recording, config)
    acquisition = config["acquisition"]
    target_rate = float(acquisition["target_sampling_rate_hz"])
    source_rate = float(audit.sampling_rate_hz)
    if not audit.signal_compatible:
        return ProcessedRecording(recording.group_id, recording.recording_id, audit, ())
    low = float(acquisition["bandpass_low_hz"])
    high = float(acquisition["bandpass_high_hz"])
    if not (0 < low < high < target_rate / 2.0):
        raise ValueError("Configured passband is incompatible with the target Nyquist frequency.")
    ratio = Fraction(target_rate / source_rate).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    bandpass_sos = signal.butter(
        int(acquisition["filter_order"]), [low, high], btype="bandpass", fs=target_rate, output="sos"
    )
    notch_sos = None
    if audit.notch_frequency_hz is not None:
        b, a = signal.iirnotch(audit.notch_frequency_hz, float(acquisition["notch_q"]), fs=target_rate)
        notch_sos = signal.tf2sos(b, a)

    segments: list[ProcessedSegment] = []
    edge_samples = int(round(float(acquisition["filter_edge_mask_s"]) * target_rate))
    for segment_index, (start, stop) in enumerate(audit.continuous_ranges, start=1):
        source_values = np.asarray(recording.adc_value[start:stop], dtype=np.float64)
        if source_values.size < max(32, 3 * (2 * int(acquisition["filter_order"]) + 1)):
            continue
        filtered = signal.detrend(source_values, type="linear")
        if up != down:
            filtered = signal.resample_poly(filtered, up, down, window=("kaiser", 5.0))
            source_indices = np.rint(np.arange(filtered.size) * down / up).astype(int)
            source_indices = np.clip(source_indices, 0, source_values.size - 1)
            quality_invalid = np.asarray(audit.quality_invalid_mask[start:stop], dtype=bool)[source_indices]
            artifact_masks = {
                name: np.asarray(mask[start:stop], dtype=bool)[source_indices]
                for name, mask in audit.artifact_masks.items()
            }
        else:
            quality_invalid = np.asarray(audit.quality_invalid_mask[start:stop], dtype=bool).copy()
            artifact_masks = {
                name: np.asarray(mask[start:stop], dtype=bool).copy()
                for name, mask in audit.artifact_masks.items()
            }
        try:
            if notch_sos is not None:
                filtered = signal.sosfiltfilt(notch_sos, filtered)
            filtered = signal.sosfiltfilt(bandpass_sos, filtered)
        except ValueError:
            continue
        edge_mask = np.zeros(filtered.size, dtype=bool)
        edge_mask[: min(edge_samples, filtered.size)] = True
        edge_mask[max(0, filtered.size - edge_samples) :] = True
        segments.append(
            ProcessedSegment(
                group_id=recording.group_id,
                recording_id=recording.recording_id,
                segment_index=segment_index,
                signal=np.asarray(filtered, dtype=np.float32),
                sampling_rate_hz=target_rate,
                start_time_s=float(recording.relative_time_s[start]),
                original_sampling_rate_hz=source_rate,
                edge_mask=edge_mask,
                quality_invalid_mask=quality_invalid,
                artifact_masks=artifact_masks,
                quality_flags=audit.quality_flags,
                acquisition_compatible=audit.signal_compatible,
            )
        )
    return ProcessedRecording(recording.group_id, recording.recording_id, audit, tuple(segments))
