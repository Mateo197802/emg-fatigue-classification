"""Signal QC, SOS filtering and reference-aligned windows for perceived fatigue."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .perceived_dataset import (
    CHANNEL_ORDER,
    TRIAL_PRIME_MOVER,
    extract_perceived_features,
    label_window,
)
from .preprocessing import line_prominence


@dataclass(frozen=True)
class PerceivedTrialProcessing:
    estimated_fs_hz: float
    median_interval_s: float
    jitter_cv: float
    gap_threshold_s: float
    gap_count: int
    window_samples: int
    hop_samples: int
    nominal_bin_spacing_hz: float
    feature_rows: tuple[dict, ...]
    window_audit_rows: tuple[dict, ...]
    sampling_rate_rows: tuple[dict, ...]
    qc_rows: tuple[dict, ...]
    filter_design_rows: tuple[dict, ...]
    exclusion_counts: dict[str, int]


def _longest_constant_run(values):
    if len(values) == 0:
        return 0
    same = np.diff(values) == 0
    best = current = 1
    for equal in same:
        current = current + 1 if equal else 1
        best = max(best, current)
    return int(best)


def _signal_segments(time_s, channel_values, gap_threshold_s):
    finite = np.isfinite(time_s)
    for values in channel_values.values():
        finite &= np.isfinite(values)
    segments = []
    start = None
    for index in range(len(time_s)):
        if not finite[index]:
            if start is not None:
                segments.append((start, index))
                start = None
            continue
        if index > 0:
            interval = time_s[index] - time_s[index - 1]
            if interval <= 0:
                raise ValueError("EMG time axis contains a clock reset or duplicate")
            if interval > gap_threshold_s and start is not None:
                segments.append((start, index))
                start = None
        if start is None:
            start = index
    if start is not None:
        segments.append((start, len(time_s)))
    return segments, finite


def _filter_segment(values, fs_hz, *, band_hz, order, edge_mask_s):
    low, high = map(float, band_hz)
    if not 0 < low < high or fs_hz <= 2 * high:
        raise ValueError("configured EMG band is not supported by the measured Nyquist frequency")
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs_hz, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    edge_samples = max(padlen, int(np.ceil(edge_mask_s * fs_hz)))
    if len(values) <= 2 * edge_samples:
        raise ValueError("continuous run too short after the filter edge mask")
    centered = values - float(np.mean(values))
    filtered = signal.sosfiltfilt(sos, centered, padlen=padlen)
    prominence = line_prominence(centered, fs_hz)
    notches = []
    for frequency_text, db in prominence.items():
        if db is None or db <= 10.0:
            continue
        frequency = float(frequency_text)
        if frequency >= fs_hz / 2:
            continue
        b, a = signal.iirnotch(frequency, 30.0, fs_hz)
        notch_sos = signal.tf2sos(b, a)
        filtered = signal.sosfiltfilt(notch_sos, filtered, padlen=padlen)
        notches.append({
            "frequency_hz": frequency,
            "quality_factor": 30.0,
            "prominence_db": float(db),
            "sos_coefficients": notch_sos.tolist(),
        })
    design = {
        "bandpass": {
            "family": "digital Butterworth band-pass",
            "prototype_order": int(order),
            "digital_order": int(2 * order),
            "cutoff_hz": [low, high],
            "sampling_rate_hz": float(fs_hz),
            "nyquist_hz": float(fs_hz / 2),
            "application": "scipy.signal.sosfiltfilt",
            "zero_phase_offline": True,
            "sos_pad_samples": int(padlen),
            "sos_coefficients": sos.tolist(),
            "edge_mask_samples_each_end": int(edge_samples),
            "edge_mask_seconds_each_end": float(edge_samples / fs_hz),
        },
        "line_prominence_db": prominence,
        "notch_trigger": "measured prominence above 10 dB",
        "notches_applied": notches,
    }
    return filtered, edge_samples, design


def process_perceived_trial(
    trial,
    *,
    window_s=0.5,
    overlap=0.5,
    band_hz=(20.0, 450.0),
    filter_order=4,
    edge_mask_s=0.10,
):
    """Build QC-backed EMG feature windows and align stable 0/2 labels.

    State 0 is no perceived fatigue, state 2 is perceived fatigue and state 1
    is excluded as transition. No time, participant, trial or window identifier
    is included in the returned feature dictionary.
    """
    time = np.asarray(trial.time_s, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("trial time axis must be finite and one-dimensional")
    intervals = np.diff(time)
    if np.any(intervals <= 0):
        raise ValueError("EMG time axis contains a clock reset or duplicate")
    median_dt = float(np.median(intervals))
    if median_dt <= 0 or not np.isfinite(median_dt):
        raise ValueError("could not estimate EMG sampling interval")
    fs_hz = 1.0 / median_dt
    gap_threshold_s = max(0.0015, 1.5 * median_dt)
    gap_mask = intervals > gap_threshold_s
    gap_count = int(np.count_nonzero(gap_mask))
    ordinary = intervals[~gap_mask]
    jitter_cv = float(np.std(ordinary, ddof=0) / np.mean(ordinary)) if len(ordinary) else float("nan")
    window_samples = int(round(float(window_s) * fs_hz))
    hop_samples = max(1, int(round(window_samples * (1.0 - float(overlap)))))
    if window_samples < 56 or not 0 <= overlap < 1 or window_s <= 0:
        raise ValueError("window configuration is invalid or too short")
    if not np.isfinite(edge_mask_s) or edge_mask_s <= 0:
        raise ValueError("edge mask must be finite and positive")
    channels = {name: np.asarray(trial.channels_v[name], dtype=float) for name in CHANNEL_ORDER}
    if any(values.shape != time.shape for values in channels.values()):
        raise ValueError("all active-muscle channels must match the EMG time axis")
    segments, finite_rows = _signal_segments(time, channels, gap_threshold_s)
    if not segments:
        raise ValueError("trial contains no finite continuous EMG samples")

    sampling_rows = []
    qc_rows = []
    for name in CHANNEL_ORDER:
        values = channels[name]
        finite_values = values[np.isfinite(values)]
        differences = np.diff(values[np.isfinite(values)])
        sampling_rows.append({
            "subject_id": trial.subject_id,
            "trial_id": trial.trial_id,
            "channel": name,
            "samples": int(len(time)),
            "median_interval_s": median_dt,
            "effective_fs_hz": fs_hz,
            "interval_jitter_cv": jitter_cv,
            "gap_threshold_s": gap_threshold_s,
            "gap_count": gap_count,
            "nonmonotonic_intervals": 0,
            "start_s": float(time[0]),
            "end_s": float(time[-1]),
        })
        qc_rows.append({
            "subject_id": trial.subject_id,
            "trial_id": trial.trial_id,
            "channel": name,
            "finite_samples": int(len(finite_values)),
            "nonfinite_samples": int(len(values) - len(finite_values)),
            "flat_sample_fraction": float(np.mean(differences == 0)) if len(differences) else 0.0,
            "longest_constant_run_samples": _longest_constant_run(finite_values),
            "minimum_v": float(np.min(finite_values)) if len(finite_values) else None,
            "maximum_v": float(np.max(finite_values)) if len(finite_values) else None,
            "standard_deviation_v": float(np.std(finite_values)) if len(finite_values) else None,
            "sensor_rail_clipping": "not_assessable_sensor_rail_limits_not_supplied",
        })

    feature_rows = []
    window_audit_rows = []
    exclusion_counts = Counter()
    filter_design_rows = []
    run_number = 0
    for segment_start, segment_end in segments:
        segment_time = time[segment_start:segment_end]
        filtered_by_channel = {}
        edge_samples = None
        segment_designs = {}
        try:
            for name in CHANNEL_ORDER:
                filtered, channel_edge, design = _filter_segment(
                    channels[name][segment_start:segment_end],
                    fs_hz,
                    band_hz=band_hz,
                    order=filter_order,
                    edge_mask_s=edge_mask_s,
                )
                filtered_by_channel[name] = filtered
                edge_samples = channel_edge if edge_samples is None else max(edge_samples, channel_edge)
                segment_designs[name] = design
        except ValueError as error:
            exclusion_counts[str(error)] += 1
            continue
        run_number += 1
        for name, design in segment_designs.items():
            filter_design_rows.append({
                "subject_id": trial.subject_id,
                "trial_id": trial.trial_id,
                "run_id": run_number,
                "channel": name,
                **design,
            })
        usable_start = int(edge_samples)
        usable_end = len(segment_time) - int(edge_samples)
        last_start = usable_end - window_samples
        if last_start < usable_start:
            exclusion_counts["no_full_window_after_filter_edge_mask"] += 1
            continue
        for local_start in range(usable_start, last_start + 1, hop_samples):
            local_end = local_start + window_samples
            global_start = segment_start + local_start
            global_end = segment_start + local_end
            start_s = float(time[global_start])
            end_exclusive_s = start_s + float(window_s)
            label, status = label_window(
                start_s,
                end_exclusive_s,
                trial.label_time_s,
                trial.perceived_level,
                initial_state=0,
            )
            audit_row = {
                "subject_id": trial.subject_id,
                "trial_id": trial.trial_id,
                "run_id": run_number,
                "window_start_s": start_s,
                "window_end_s": end_exclusive_s,
                "window_status": status,
                "binary_label": label,
            }
            window_audit_rows.append(audit_row)
            if label is None:
                exclusion_counts[status] += 1
                continue
            feature_values = {}
            try:
                for name in CHANNEL_ORDER:
                    values = filtered_by_channel[name][local_start:local_end]
                    for feature_name, feature_value in extract_perceived_features(
                        values, fs_hz, band_hz=band_hz
                    ).items():
                        feature_values[f"{name}__{feature_name}"] = feature_value
            except ValueError as error:
                exclusion_counts[f"feature:{error}"] += 1
                audit_row["window_status"] = f"feature:{error}"
                audit_row["binary_label"] = None
                continue
            feature_rows.append({
                "window_id": f"{trial.subject_id}_{trial.trial_id}_r{run_number:02d}_i{global_start:08d}",
                "subject_id": trial.subject_id,
                "trial_id": trial.trial_id,
                "prime_mover": TRIAL_PRIME_MOVER[int(trial.trial_id.split("_")[-1])],
                "window_start_s": start_s,
                "window_end_s": end_exclusive_s,
                "window_center_s": float((start_s + end_exclusive_s) / 2),
                "label": int(label),
                "label_status": status,
                "features": feature_values,
            })
            exclusion_counts["eligible_labeled_window"] += 1

    return PerceivedTrialProcessing(
        estimated_fs_hz=fs_hz,
        median_interval_s=median_dt,
        jitter_cv=jitter_cv,
        gap_threshold_s=gap_threshold_s,
        gap_count=gap_count,
        window_samples=window_samples,
        hop_samples=hop_samples,
        nominal_bin_spacing_hz=float(fs_hz / window_samples),
        feature_rows=tuple(feature_rows),
        window_audit_rows=tuple(window_audit_rows),
        sampling_rate_rows=tuple(sampling_rows),
        qc_rows=tuple(qc_rows),
        filter_design_rows=tuple(filter_design_rows),
        exclusion_counts=dict(sorted(exclusion_counts.items())),
    )
