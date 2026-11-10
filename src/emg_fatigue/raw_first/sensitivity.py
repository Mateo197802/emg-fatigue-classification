"""Unsupervised filter and window-size feature sensitivity summaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .contracts import RawRecording, SignalWindow
from .features import extract_features
from .preprocessing import preprocess_recording
from .windows import segment_windows


def filter_sensitivity(
    recordings: Sequence[RawRecording],
    audits: dict[str, dict[str, object]],
    primary_plan: dict[str, object],
    primary_windows: Sequence[SignalWindow],
    validation_groups: Sequence[str],
    willison_threshold: float,
    filter_order: int,
    overlap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = set(validation_groups)
    primary_map = {
        (window.recording_id, window.segment_index, window.start_sample): window
        for window in primary_windows
        if window.group_id in validation
    }
    alternative = dict(primary_plan)
    primary_high = float(primary_plan["bandpass_high_hz"])
    alternative["bandpass_high_hz"] = max(float(primary_plan["bandpass_low_hz"]) + 5.0, primary_high * 0.8)
    alternative["notch_frequency_hz"] = None
    alternative_name = f"lower_high_cut_{alternative['bandpass_high_hz']:.1f}hz_no_notch"
    alt_rows: list[dict[str, object]] = []
    matched = 0
    for recording in recordings:
        if recording.group_id not in validation:
            continue
        recording_audit = audits[recording.recording_id]
        processed, _ = preprocess_recording(recording, recording_audit, alternative, filter_order=filter_order)
        windows, _ = segment_windows(processed, 0.5, overlap)
        for window in windows:
            key = (window.recording_id, window.segment_index, window.start_sample)
            base = primary_map.get(key)
            if base is None:
                continue
            primary_features = extract_features(base, willison_threshold)
            alternative_features = extract_features(window, willison_threshold)
            for name in sorted(set(primary_features) & set(alternative_features)):
                first = float(primary_features[name])
                second = float(alternative_features[name])
                if np.isfinite(first) and np.isfinite(second):
                    alt_rows.append(
                        {
                            "feature": name,
                            "relative_absolute_change": abs(second - first) / max(abs(first), 1e-8),
                            "absolute_change": abs(second - first),
                        }
                    )
            matched += 1
    if matched == 0:
        return pd.DataFrame(), pd.DataFrame([{"alternative": alternative_name, "matched_windows": 0, "status": "no matched held-out windows"}])
    details = pd.DataFrame(alt_rows).groupby("feature", as_index=False).agg(
        median_relative_absolute_change=("relative_absolute_change", "median"),
        median_absolute_change=("absolute_change", "median"),
    )
    details.insert(0, "alternative", alternative_name)
    details["matched_validation_windows"] = matched
    summary = pd.DataFrame(
        [
            {
                "alternative": alternative_name,
                "matched_validation_windows": matched,
                "median_relative_absolute_feature_change": float(details["median_relative_absolute_change"].median()),
                "feature_count": int(details["feature"].nunique()),
                "status": "held-out source-group feature sensitivity only; not selected by predictive performance",
            }
        ]
    )
    return details, summary


def window_sensitivity(feature_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    primary = feature_frames.get("0.5")
    if primary is None:
        return pd.DataFrame()
    metadata = {"window_id", "group_id", "recording_id", "segment_index", "window_index", "start_sample", "start_time_s", "duration_s"}
    common = sorted(set(primary.columns) - metadata)
    rows: list[dict[str, object]] = []
    for duration, frame in sorted(feature_frames.items(), key=lambda item: float(item[0])):
        for feature in common:
            if feature not in frame.columns:
                continue
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "window_duration_s": float(duration),
                    "feature": feature,
                    "finite_window_count": int(finite.size),
                    "finite_fraction": float(finite.size / max(1, values.size)),
                    "mean": float(np.mean(finite)) if finite.size else float("nan"),
                    "median": float(np.median(finite)) if finite.size else float("nan"),
                    "std": float(np.std(finite)) if finite.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)
