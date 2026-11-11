"""Fixed-duration windows and prespecified signal-quality rejection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .contracts import ProcessedSegment, SignalWindow


def window_quality_reason(values: np.ndarray, sampling_rate_hz: float) -> str | None:
    if values.size == 0 or not np.isfinite(values).all():
        return "nonfinite_or_empty"
    centered = values - float(np.median(values))
    rms = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(rms) or rms <= np.finfo(np.float32).eps:
        return "zero_energy"
    max_repeat = 0
    if values.size > 1:
        changes = np.flatnonzero(values[1:] != values[:-1]) + 1
        bounds = np.r_[0, changes, values.size]
        max_repeat = int(np.diff(bounds).max(initial=0))
    if max_repeat >= max(4, int(round(0.05 * sampling_rate_hz))):
        return "flatline_run_at_least_50ms"

    peak = float(np.max(np.abs(centered)))
    crest_factor = peak / rms
    energy = centered.astype(np.float64) ** 2
    largest_sample_share = float(energy.max() / energy.sum()) if energy.sum() else 0.0
    if crest_factor > 40.0 and largest_sample_share > 0.25:
        return "single_sample_transient_candidate"
    return None


def segment_windows(
    segments: Iterable[ProcessedSegment], duration_s: float, overlap: float = 0.5
) -> tuple[list[SignalWindow], list[dict[str, object]]]:
    if duration_s <= 0 or not (0 <= overlap < 1):
        raise ValueError("Window duration must be positive and overlap must be in [0, 1).")
    windows: list[SignalWindow] = []
    rejected: list[dict[str, object]] = []
    for segment in segments:
        fs = segment.sampling_rate_hz
        width = int(round(duration_s * fs))
        hop = int(round(width * (1.0 - overlap)))
        if width < 2 or hop < 1:
            raise ValueError("Window dimensions are invalid at the selected sampling rate.")
        local_index = 0
        for start in range(0, max(0, len(segment.signal) - width + 1), hop):
            values = segment.signal[start : start + width]
            reason = window_quality_reason(values, fs)
            entry = {
                "group_id": segment.group_id,
                "recording_id": segment.recording_id,
                "segment_index": segment.segment_index,
                "window_index": local_index,
                "start_sample": start,
                "start_time_s": segment.start_time_s + start / fs,
                "duration_s": duration_s,
                "sampling_rate_hz": fs,
            }
            if reason:
                rejected.append({**entry, "rejection_reason": reason})
            else:
                windows.append(
                    SignalWindow(
                        group_id=segment.group_id,
                        recording_id=segment.recording_id,
                        segment_index=segment.segment_index,
                        window_index=local_index,
                        start_sample=start,
                        start_time_s=segment.start_time_s + start / fs,
                        duration_s=duration_s,
                        sampling_rate_hz=fs,
                        signal=np.asarray(values, dtype=np.float32),
                    )
                )
            local_index += 1
    return windows, rejected
