"""Candidate activity envelopes; these are not movement-phase or fatigue labels."""

from __future__ import annotations

import numpy as np

from .contracts import ActivitySegments, ProcessedRecording, SegmentActivity


def _otsu_threshold(values: np.ndarray, bins: int = 128) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2 or float(np.ptp(values)) <= 1e-10:
        return float("nan")
    histogram, edges = np.histogram(values, bins=bins)
    weights = histogram.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(weights.sum())
    cumulative_weight = np.cumsum(weights)
    cumulative_mean = np.cumsum(weights * centers)
    total_mean = cumulative_mean[-1]
    left_weight = cumulative_weight
    right_weight = total - left_weight
    valid = (left_weight > 0) & (right_weight > 0)
    between = np.zeros_like(centers)
    mean_left = np.divide(cumulative_mean, left_weight, out=np.zeros_like(centers), where=left_weight > 0)
    mean_right = np.divide(
        total_mean - cumulative_mean,
        right_weight,
        out=np.zeros_like(centers),
        where=right_weight > 0,
    )
    between[valid] = left_weight[valid] * right_weight[valid] * (mean_left[valid] - mean_right[valid]) ** 2
    return float(centers[int(np.argmax(between))])


def _fill_valid_short_gaps(active: np.ndarray, valid: np.ndarray, maximum_gap_frames: int) -> np.ndarray:
    result = np.asarray(active, dtype=bool).copy()
    flags = np.asarray(valid, dtype=bool)
    if result.size < 3 or maximum_gap_frames < 1:
        return result
    index = 0
    while index < result.size:
        if result[index]:
            index += 1
            continue
        start = index
        while index < result.size and not result[index]:
            index += 1
        stop = index
        gap_length = stop - start
        bounded = start > 0 and stop < result.size and result[start - 1] and result[stop]
        if bounded and gap_length <= maximum_gap_frames and flags[start:stop].all():
            result[start:stop] = True
    return result


def _true_ranges(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(mask, dtype=bool)
    if not values.size:
        return ()
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.r_[0, changes, values.size]
    return tuple((int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if values[a])


def detect_candidate_activity(processed: ProcessedRecording, config: dict) -> ActivitySegments:
    settings = config["activity"]
    results: list[SegmentActivity] = []
    frame_weights: list[int] = []
    active_weights: list[int] = []
    frame_samples_by_segment: dict[int, int] = {}
    for segment in processed.segments:
        fs = float(segment.sampling_rate_hz)
        frame_samples = max(2, int(round(float(settings["rms_frame_s"]) * fs)))
        hop_samples = max(1, int(round(float(settings["rms_hop_s"]) * fs)))
        starts = np.arange(0, max(0, segment.signal.size - frame_samples + 1), hop_samples, dtype=np.int64)
        rms_values = np.full(starts.size, np.nan, dtype=np.float64)
        frame_valid = np.zeros(starts.size, dtype=bool)
        for index, start in enumerate(starts):
            stop = int(start) + frame_samples
            invalid = segment.edge_mask[start:stop] | segment.quality_invalid_mask[start:stop]
            if invalid.any():
                continue
            block = np.asarray(segment.signal[start:stop], dtype=np.float64)
            if not np.isfinite(block).all():
                continue
            rms_values[index] = float(np.sqrt(np.mean(block * block)))
            frame_valid[index] = True
        valid_rms = rms_values[frame_valid]
        if valid_rms.size < 4 or np.ptp(np.log10(valid_rms + np.finfo(float).eps)) <= 1e-8:
            threshold_rms = float("nan")
            active = np.zeros(starts.size, dtype=bool)
            status = "indeterminate"
            active_fraction = float("nan")
        else:
            log_rms = np.log10(valid_rms + np.finfo(float).eps)
            threshold_log = _otsu_threshold(log_rms)
            threshold_rms = float(10.0**threshold_log) if np.isfinite(threshold_log) else float("nan")
            active = frame_valid & (rms_values > threshold_rms)
            max_gap = int(np.floor(float(settings["fill_inactive_gap_s"]) / float(settings["rms_hop_s"])))
            active = _fill_valid_short_gaps(active, frame_valid, max_gap)
            active_count = int(np.sum(active & frame_valid))
            active_fraction = active_count / int(np.sum(frame_valid)) if frame_valid.any() else float("nan")
            if (
                not np.isfinite(active_fraction)
                or active_fraction < float(settings["minimum_active_frame_fraction"])
                or active_fraction > float(settings["maximum_active_frame_fraction"])
            ):
                status = "indeterminate"
                active[:] = False
            else:
                status = "determinate"
        sample_mask = np.zeros(segment.signal.size, dtype=bool)
        for start, is_active, is_valid in zip(starts, active, frame_valid, strict=True):
            if is_active and is_valid:
                sample_mask[int(start) : int(start) + frame_samples] = True
        results.append(
            SegmentActivity(
                segment_index=segment.segment_index,
                status=status,
                threshold_rms=threshold_rms,
                active_fraction=float(active_fraction),
                valid_frame_count=int(frame_valid.sum()),
                frame_rms=rms_values,
                frame_start_samples=starts,
                frame_active=active,
                frame_valid=frame_valid,
                sample_active_mask=sample_mask,
                active_ranges=_true_ranges(sample_mask),
            )
        )
        frame_weights.append(int(frame_valid.sum()))
        active_weights.append(int(np.sum(active & frame_valid)))
        frame_samples_by_segment[segment.segment_index] = int(frame_valid.sum())

    total_frames = int(sum(frame_weights))
    total_active = int(sum(active_weights))
    overall_fraction = total_active / total_frames if total_frames else float("nan")
    overall_status = "determinate" if any(item.status == "determinate" for item in results) else "indeterminate"
    return ActivitySegments(overall_status, float(overall_fraction), tuple(results))
