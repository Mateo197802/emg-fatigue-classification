"""Gap-preserving, evidence-configured signal preprocessing."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from .contracts import ProcessedSegment, RawRecording


def contiguous_raw_segments(
    recording: RawRecording, gap_threshold_s: float
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """Split on clock resets/gaps and discard nonfinite or duplicate-time samples."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    current_t: list[float] = []
    current_x: list[float] = []
    counts = {"nonfinite_samples": 0, "duplicate_timestamp_samples_dropped": 0, "segments_split": 0}

    def close_segment() -> None:
        if len(current_x) >= 2:
            segments.append((np.asarray(current_t, dtype=np.float64), np.asarray(current_x, dtype=np.float64)))

    last_time: float | None = None
    for time_s, sample in zip(recording.relative_time_s, recording.adc_value, strict=True):
        if not np.isfinite(time_s) or not np.isfinite(sample):
            counts["nonfinite_samples"] += 1
            if current_x:
                close_segment()
                current_t.clear()
                current_x.clear()
                last_time = None
            continue

        if last_time is not None:
            delta = float(time_s - last_time)
            if delta == 0:
                counts["duplicate_timestamp_samples_dropped"] += 1
                continue
            if delta < 0 or delta > gap_threshold_s:
                close_segment()
                current_t.clear()
                current_x.clear()
                last_time = None
                counts["segments_split"] += 1

        current_t.append(float(time_s))
        current_x.append(float(sample))
        last_time = float(time_s)

    if current_x:
        close_segment()
    return segments, counts


def _resample_ratio(source_rate_hz: float, target_rate_hz: float) -> tuple[int, int]:
    ratio = Fraction(target_rate_hz / source_rate_hz).limit_denominator(10000)
    return ratio.numerator, ratio.denominator


def preprocess_recording(
    recording: RawRecording,
    audit: dict[str, object],
    sampling_plan: dict[str, object],
    filter_order: int = 4,
    notch_q: float = 30.0,
) -> tuple[list[ProcessedSegment], dict[str, int]]:
    source_rate = float(audit["effective_sampling_rate_hz"])
    target_rate = float(sampling_plan["target_sampling_rate_hz"])
    low_hz = float(sampling_plan["bandpass_low_hz"])
    high_hz = float(sampling_plan["bandpass_high_hz"])
    if not np.isfinite(source_rate) or source_rate <= 0:
        return [], {"segments_split": 0, "short_segments_rejected": 0, "processed_segments": 0, "nonfinite_samples": 0, "duplicate_timestamp_samples_dropped": 0}
    if not (0 < low_hz < high_hz < target_rate / 2):
        raise ValueError("The selected EMG passband is incompatible with the common sampling rate.")

    raw_segments, counts = contiguous_raw_segments(recording, float(audit["gap_threshold_s"]))
    processed: list[ProcessedSegment] = []
    short_rejected = 0
    up, down = _resample_ratio(source_rate, target_rate)
    bandpass_sos = signal.butter(filter_order, [low_hz, high_hz], btype="bandpass", fs=target_rate, output="sos")
    notch_hz = sampling_plan.get("notch_frequency_hz")
    notch_sos = None
    if notch_hz is not None and float(notch_hz) < target_rate / 2:
        b, a = signal.iirnotch(float(notch_hz), notch_q, fs=target_rate)
        notch_sos = signal.tf2sos(b, a)

    for segment_index, (times, values) in enumerate(raw_segments, start=1):
        if values.size < max(32, 3 * filter_order * 2):
            short_rejected += 1
            continue
        signal_values = signal.detrend(values, type="linear")
        if up != down:
            signal_values = signal.resample_poly(signal_values, up, down, window=("kaiser", 5.0))
        try:
            if notch_sos is not None:
                signal_values = signal.sosfiltfilt(notch_sos, signal_values)
            signal_values = signal.sosfiltfilt(bandpass_sos, signal_values)
        except ValueError:
            short_rejected += 1
            continue
        if signal_values.size < 8:
            short_rejected += 1
            continue
        processed.append(
            ProcessedSegment(
                group_id=recording.group_id,
                recording_id=recording.recording_id,
                segment_index=segment_index,
                signal=np.asarray(signal_values, dtype=np.float32),
                sampling_rate_hz=target_rate,
                start_time_s=float(times[0]),
                original_sampling_rate_hz=source_rate,
                resample_up=up,
                resample_down=down,
            )
        )
    counts["short_segments_rejected"] = short_rejected
    counts["processed_segments"] = len(processed)
    return processed, counts
