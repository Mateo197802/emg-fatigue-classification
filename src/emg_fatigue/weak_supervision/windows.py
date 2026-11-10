"""Overlap-aware signal windows with explicit activity and quality eligibility."""

from __future__ import annotations

import numpy as np

from .contracts import ActivitySegments, ProcessedRecording, SegmentActivity, SignalWindow


def _rejection_reason(
    segment,
    activity: SegmentActivity | None,
    start: int,
    stop: int,
    occupancy: float,
    minimum_occupancy: float,
) -> str | None:
    if not segment.acquisition_compatible:
        return "acquisition_or_spectral_gate_failed"
    if activity is None or activity.status != "determinate":
        return "activity_indeterminate"
    if segment.edge_mask[start:stop].any():
        return "filter_edge_mask"
    for reason_key, reason_text in (
        ("nonfinite_sample", "nonfinite_sample"),
        ("flatline_run_at_least_100ms", "flatline_run_at_least_100ms"),
        ("clipping_suspect_plateau", "clipping_suspect_plateau"),
        ("impulsive_artifact", "impulse_artifact"),
    ):
        reason_mask = segment.artifact_masks.get(reason_key)
        if reason_mask is not None and reason_mask[start:stop].any():
            return reason_text
    if segment.quality_invalid_mask[start:stop].any():
        return "signal_quality_artifact"
    if occupancy < minimum_occupancy:
        return "insufficient_activity_occupancy"
    return None


def make_windows(
    processed: ProcessedRecording,
    activity: ActivitySegments,
    duration_s: float,
    overlap: float,
) -> list[SignalWindow]:
    if duration_s <= 0 or not 0.0 <= overlap < 1.0:
        raise ValueError("Window duration must be positive and overlap must be in [0, 1).")
    activity_by_segment = {item.segment_index: item for item in activity.segments}
    windows: list[SignalWindow] = []
    minimum_occupancy = 0.5
    for segment in processed.segments:
        fs = float(segment.sampling_rate_hz)
        width = int(round(duration_s * fs))
        hop = int(round(width * (1.0 - overlap)))
        if width < 2 or hop < 1:
            raise ValueError("Window dimensions are invalid at the segment sampling rate.")
        segment_activity = activity_by_segment.get(segment.segment_index)
        sample_activity = (
            segment_activity.sample_active_mask
            if segment_activity is not None
            else np.zeros(segment.signal.size, dtype=bool)
        )
        for window_index, start in enumerate(range(0, max(0, segment.signal.size - width + 1), hop)):
            stop = start + width
            occupancy = float(np.mean(sample_activity[start:stop]))
            reason = _rejection_reason(
                segment, segment_activity, start, stop, occupancy, minimum_occupancy
            )
            windows.append(
                SignalWindow(
                    group_id=segment.group_id,
                    recording_id=segment.recording_id,
                    segment_index=segment.segment_index,
                    window_index=window_index,
                    start_sample=start,
                    start_time_s=segment.start_time_s + start / fs,
                    duration_s=float(duration_s),
                    sampling_rate_hz=fs,
                    signal=np.asarray(segment.signal[start:stop], dtype=np.float32).copy(),
                    activity_occupancy=occupancy,
                    eligible_for_label=reason is None,
                    rejection_reason=reason,
                )
            )
    return windows
