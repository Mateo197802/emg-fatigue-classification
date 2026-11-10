"""Development-only feature extraction, proxy labels, and label sensitivities."""

from __future__ import annotations

import csv
import copy
import hashlib
import importlib.metadata
import json
import subprocess
import tomllib
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .activity import detect_candidate_activity
from .features import FEATURE_VIEWS, feature_schema, extract_window_features, window_key
from .io import load_training_zip
from .preprocess import preprocess_recording
from .pseudo_labels import (
    label_log_mdf_slope_rule,
    label_low_frequency_fraction_rule,
    label_recording,
    technical_reference,
)
from .windows import make_windows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _feature_base(window, *, status: str, feature_valid: int) -> dict[str, object]:
    return {
        "window_key": window_key(window),
        "group_id": window.group_id,
        "recording_id": window.recording_id,
        "segment_index": window.segment_index,
        "window_index": window.window_index,
        "start_sample": window.start_sample,
        "start_time_s": window.start_time_s,
        "duration_s": window.duration_s,
        "sampling_rate_hz": window.sampling_rate_hz,
        "activity_occupancy": window.activity_occupancy,
        "window_eligible_at_primary_occupancy": window.eligible_for_label,
        "window_rejection_reason": window.rejection_reason or "",
        "feature_valid": feature_valid,
        "feature_status": status,
    }


def _extract_rows_for_windows(windows, line_frequency_hz: float | None, line_exclusion_hz: float):
    feature_map: dict[str, dict[str, object]] = {}
    output_rows: list[dict[str, object]] = []
    feature_names = tuple(item["name"] for item in feature_schema())
    for window in windows:
        key = window_key(window)
        if window.eligible_for_label or window.rejection_reason == "insufficient_activity_occupancy":
            values = extract_window_features(
                window.signal,
                window.sampling_rate_hz,
                line_frequency_hz=line_frequency_hz,
                line_exclusion_hz=line_exclusion_hz,
            )
            feature_map[key] = values
            feature_row = _feature_base(
                window,
                status=str(values["feature_status"]),
                feature_valid=int(values["feature_valid"]),
            )
            feature_row.update({name: values.get(name, float("nan")) for name in feature_names})
        else:
            feature_row = _feature_base(
                window,
                status="window_ineligible_before_feature_extraction",
                feature_valid=0,
            )
            feature_row.update({name: float("nan") for name in feature_names})
        output_rows.append(feature_row)
    return output_rows, feature_map


def _label_row(label) -> dict[str, object]:
    row = asdict(label)
    for name in (
        "reference_window_keys",
        "reference_center",
        "reference_scaled_mad",
        "observed_indicators",
        "indicator_deltas",
        "indicator_thresholds",
    ):
        row[name] = json.dumps(row[name], sort_keys=True, separators=(",", ":"))
    return row


def _counts_by_class(labels) -> dict[str, int]:
    counts = Counter(str(label.class_id) for label in labels)
    return {name: counts.get(name, 0) for name in ("-1", "0", "1")}


def _occupancy_variant(windows, threshold: float):
    changed = []
    for window in windows:
        if window.rejection_reason == "insufficient_activity_occupancy":
            is_eligible = window.activity_occupancy >= threshold
            reason = None if is_eligible else "insufficient_activity_occupancy"
            changed.append(replace(window, eligible_for_label=is_eligible, rejection_reason=reason))
        elif window.eligible_for_label and window.activity_occupancy < threshold:
            changed.append(replace(window, eligible_for_label=False, rejection_reason="insufficient_activity_occupancy"))
        else:
            changed.append(window)
    return changed


def _sensitivity_row(kind: str, settings: dict[str, object], reference, labels, group_id: str, recording_id: str, segment_index: int) -> dict[str, object]:
    return {
        "sensitivity_type": kind,
        "settings_json": json.dumps(settings, sort_keys=True, separators=(",", ":")),
        "group_id": group_id,
        "recording_id": recording_id,
        "segment_index": segment_index,
        "reference_status": reference.status,
        "reference_window_count": reference.window_count,
        "reference_reasons": ";".join(reference.reasons),
        "ambiguous_count": sum(label.class_id == -1 for label in labels),
        "non_fatigue_like_count": sum(label.class_id == 0 for label in labels),
        "fatigue_like_count": sum(label.class_id == 1 for label in labels),
        "total_windows": len(labels),
        "label_coverage": (sum(label.class_id >= 0 for label in labels) / len(labels)) if labels else 0.0,
    }


def _correlation_frame(feature_rows: list[dict[str, object]], feature_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(feature_rows)
    eligible = frame[(frame["duration_s"] == 0.5) & (frame["window_eligible_at_primary_occupancy"] == True) & (frame["feature_valid"] == 1)]
    return eligible.loc[:, list(feature_columns)].corr(method="spearman", min_periods=3)


def run_label_audit(training_zip: Path, run_dir: Path, config_path: Path) -> dict[str, object]:
    """Extract fresh features and conservative weak labels from development groups only."""
    repo_root = Path(__file__).resolve().parents[3]
    training_zip = Path(training_zip).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    private_dir = run_dir / "private"
    run_manifest_path = private_dir / "run_manifest.json"
    split_manifest_path = private_dir / "split_manifest.json"
    if not run_manifest_path.is_file() or not split_manifest_path.is_file():
        raise FileNotFoundError("Prepare the training ZIP and split before feature and label auditing.")
    output_dir = private_dir / "features_labels"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("The private feature-label directory is not empty.")
    if output_dir.exists():
        output_dir.rmdir()
    output_dir.mkdir(parents=True)
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    training_sha = _sha256_file(training_zip)
    if training_sha != run_manifest.get("input_archive_sha256"):
        raise ValueError("Training ZIP differs from the archive prepared for this run.")
    recordings, inventory = load_training_zip(
        training_zip,
        seed=int(config["run"]["split_seed"]),
        train_fraction=float(config["run"]["train_fraction"]),
    )
    if list(inventory.split.development_groups) != split_manifest["development_group_ids"]:
        raise ValueError("Development split does not match the prepared private receipt.")
    if list(inventory.split.test_groups) != split_manifest["test_group_ids"]:
        raise ValueError("Locked internal-test split does not match the prepared private receipt.")
    if any(item.parsed_numeric_values for item in inventory.members if item.group_id in set(split_manifest["test_group_ids"])):
        raise RuntimeError("Internal-test numeric signals were parsed during feature extraction.")

    primary_occupancy = float(config["activity"]["primary_window_occupancy"])
    overlap = float(config["run"]["window_overlap"])
    durations = [float(value) for value in config["run"]["window_durations_s"]]
    primary_duration = float(config["run"]["primary_window_s"])
    feature_rows: list[dict[str, object]] = []
    primary_features: dict[str, dict[str, object]] = {}
    feature_maps_by_duration: dict[float, dict[str, dict[str, object]]] = {duration: {} for duration in durations}
    labels_rows: list[dict[str, object]] = []
    alternative_label_rows: list[dict[str, object]] = []
    window_size_label_rows: list[dict[str, object]] = []
    label_audit_rows: list[dict[str, object]] = []
    label_by_rule: dict[str, list] = {"primary_v1": [], "log_mdf_slope_low_fraction_v1": [], "low_frequency_fraction_persistence_v1": []}
    group_counts: Counter[tuple[str, int]] = Counter()
    recording_counts: Counter[tuple[str, str, int]] = Counter()
    metadata_counts: Counter[tuple[str, str, int]] = Counter()
    temporal_counts: Counter[tuple[str, int, int]] = Counter()
    recording_bounds: dict[str, tuple[float, float]] = {}
    recording_metadata: dict[str, tuple[str, str]] = {}
    feature_schema_rows = feature_schema()
    feature_names = tuple(item["name"] for item in feature_schema_rows)
    run_count = 0
    reference_unavailable_count = 0
    reference_unstable_count = 0
    sensitivity_ref_counts = sorted(set([3, 5]))

    for recording in recordings:
        finite_times = np.asarray(recording.relative_time_s, dtype=np.float64)
        recording_bounds[recording.recording_id] = (float(np.min(finite_times)), float(np.max(finite_times)))
        recording_metadata[recording.recording_id] = (
            str(recording.filename_metadata.get("series_token") or "unknown"),
            str(recording.filename_metadata.get("side_token") or "unknown"),
        )
        processed = preprocess_recording(recording, config)
        activity = detect_candidate_activity(processed, config)
        windows_by_duration = {}
        for duration in durations:
            windows = make_windows(
                processed,
                activity,
                duration,
                overlap,
                minimum_occupancy=primary_occupancy,
            )
            windows_by_duration[duration] = windows
            rows, features_for_duration = _extract_rows_for_windows(
                windows,
                processed.audit.notch_frequency_hz,
                float(config["labels"]["line_exclusion_hz"]),
            )
            feature_rows.extend(rows)
            feature_maps_by_duration[duration].update(features_for_duration)
            if duration == primary_duration:
                primary_features.update(features_for_duration)
        primary_windows = windows_by_duration[primary_duration]
        reference_windows = make_windows(
            processed,
            activity,
            primary_duration,
            0.0,
            minimum_occupancy=primary_occupancy,
        )
        notch_sensitivity_by_segment: dict[int, list[dict[str, object]]] = {}
        finite_prominences = [
            float(value)
            for value in processed.audit.line_prominence_db.values()
            if np.isfinite(value)
        ]
        if finite_prominences and max(finite_prominences) > 7.5:
            for threshold_db, variant_name in (
                (float("inf"), "notch_off"),
                (7.5, "notch_prominence_7_5db"),
                (12.5, "notch_prominence_12_5db"),
            ):
                variant_config = copy.deepcopy(config)
                variant_config["acquisition"]["line_prominence_notch_db"] = threshold_db
                variant_processed = preprocess_recording(recording, variant_config)
                variant_activity = detect_candidate_activity(variant_processed, variant_config)
                variant_windows = make_windows(
                    variant_processed,
                    variant_activity,
                    primary_duration,
                    overlap,
                    minimum_occupancy=primary_occupancy,
                )
                variant_references = make_windows(
                    variant_processed,
                    variant_activity,
                    primary_duration,
                    0.0,
                    minimum_occupancy=primary_occupancy,
                )
                _, variant_features = _extract_rows_for_windows(
                    variant_windows,
                    variant_processed.audit.notch_frequency_hz,
                    float(config["labels"]["line_exclusion_hz"]),
                )
                for variant_segment in variant_processed.segments:
                    variant_segment_windows = [item for item in variant_windows if item.segment_index == variant_segment.segment_index]
                    variant_segment_references = [item for item in variant_references if item.segment_index == variant_segment.segment_index]
                    variant_reference = technical_reference(variant_segment_references, variant_features, config)
                    variant_labels = label_recording(variant_segment_windows, variant_features, variant_reference, config)
                    notch_sensitivity_by_segment.setdefault(variant_segment.segment_index, []).append(
                        _sensitivity_row(
                            "notch_threshold",
                            {
                                "variant": variant_name,
                                "line_prominence_threshold_db": threshold_db if np.isfinite(threshold_db) else None,
                                "selected_notch_hz": variant_processed.audit.notch_frequency_hz,
                            },
                            variant_reference,
                            variant_labels,
                            recording.group_id,
                            recording.recording_id,
                            variant_segment.segment_index,
                        )
                    )

        for segment in processed.segments:
            segment_windows = [window for window in primary_windows if window.segment_index == segment.segment_index]
            segment_reference_windows = [window for window in reference_windows if window.segment_index == segment.segment_index]
            if not segment_windows:
                continue
            reference = technical_reference(segment_reference_windows, primary_features, config)
            segment_labels = label_recording(segment_windows, primary_features, reference, config)
            labels_rows.extend(_label_row(label) for label in segment_labels)
            label_by_rule["primary_v1"].extend(segment_labels)
            run_count += 1
            reference_unavailable_count += int(not reference.available)
            reference_unstable_count += int(reference.available and not reference.stable)
            for label in segment_labels:
                group_counts[(recording.group_id, label.class_id)] += 1
                recording_counts[(recording.group_id, recording.recording_id, label.class_id)] += 1
                series_token, side_token = recording_metadata[recording.recording_id]
                metadata_counts[(series_token, side_token, label.class_id)] += 1
                lower_time, upper_time = recording_bounds[recording.recording_id]
                fraction = (
                    (label.start_time_s - lower_time) / (upper_time - lower_time)
                    if upper_time > lower_time
                    else 0.0
                )
                quartile = min(3, max(0, int(np.floor(np.clip(fraction, 0.0, 0.999999) * 4.0))))
                temporal_counts[(recording.group_id, quartile, label.class_id)] += 1
            label_audit_rows.append(
                _sensitivity_row(
                    "primary_v1",
                    {
                        "rule_version": config["labels"]["version"],
                        "reference_window_count": int(config["labels"]["reference_window_count"]),
                        "reference_stability_percent": float(config["labels"]["reference_instability_percent"]),
                        "occupancy_threshold": primary_occupancy,
                        "relative_change": 0.0,
                        "scaled_mad_multiple": float(config["labels"]["minimum_scaled_mad_multiple"]),
                        "window_duration_s": primary_duration,
                        "window_overlap": overlap,
                    },
                    reference,
                    segment_labels,
                    recording.group_id,
                    recording.recording_id,
                    segment.segment_index,
                )
            )
            label_audit_rows.extend(notch_sensitivity_by_segment.get(segment.segment_index, []))

            for sensitivity_duration in durations:
                if sensitivity_duration == primary_duration:
                    continue
                sensitivity_windows = windows_by_duration[sensitivity_duration]
                sensitivity_run_windows = [
                    window for window in sensitivity_windows if window.segment_index == segment.segment_index
                ]
                sensitivity_reference_windows = make_windows(
                    processed,
                    activity,
                    sensitivity_duration,
                    0.0,
                    minimum_occupancy=primary_occupancy,
                )
                sensitivity_reference_windows = [
                    window for window in sensitivity_reference_windows if window.segment_index == segment.segment_index
                ]
                size_reference = technical_reference(
                    sensitivity_reference_windows,
                    feature_maps_by_duration[sensitivity_duration],
                    config,
                )
                size_labels = label_recording(
                    sensitivity_run_windows,
                    feature_maps_by_duration[sensitivity_duration],
                    size_reference,
                    config,
                    rule_version=f"spectral_persistence_v1_{int(round(sensitivity_duration * 1000))}ms_sensitivity",
                )
                window_size_label_rows.extend(_label_row(label) for label in size_labels)
                label_audit_rows.append(
                    _sensitivity_row(
                        "window_size",
                        {"window_duration_s": sensitivity_duration, "window_overlap": overlap},
                        size_reference,
                        size_labels,
                        recording.group_id,
                        recording.recording_id,
                        segment.segment_index,
                    )
                )

            low_fraction_reference = technical_reference(
                segment_reference_windows,
                primary_features,
                config,
                stability_components=("low_fraction",),
            )
            low_fraction_labels = label_low_frequency_fraction_rule(
                segment_windows, primary_features, low_fraction_reference, config
            )
            alternative_label_rows.extend(
                {**_label_row(label), "comparison_rule": "low_frequency_fraction_persistence_v1"}
                for label in low_fraction_labels
            )
            label_by_rule["low_frequency_fraction_persistence_v1"].extend(low_fraction_labels)
            label_audit_rows.append(
                _sensitivity_row(
                    "alternative_low_frequency_fraction",
                    {"reference_window_count": int(config["labels"]["reference_window_count"]), "threshold": "3*scaled_MAD_or_5_percent_fraction_floor"},
                    low_fraction_reference,
                    low_fraction_labels,
                    recording.group_id,
                    recording.recording_id,
                    segment.segment_index,
                )
            )

            slope_reference = technical_reference(
                segment_reference_windows,
                primary_features,
                config,
                stability_components=("mdf", "low_fraction"),
            )
            slope_labels = label_log_mdf_slope_rule(segment_windows, primary_features, slope_reference, config)
            alternative_label_rows.extend(
                {**_label_row(label), "comparison_rule": "log_mdf_slope_low_fraction_v1"}
                for label in slope_labels
            )
            label_by_rule["log_mdf_slope_low_fraction_v1"].extend(slope_labels)
            label_audit_rows.append(
                _sensitivity_row(
                    "alternative_log_mdf_slope_low_fraction",
                    {"reference_window_count": int(config["labels"]["reference_window_count"]), "trend": "negative log-MDF slope", "low_fraction_threshold": "3*scaled_MAD_or_5_percent_fraction_floor"},
                    slope_reference,
                    slope_labels,
                    recording.group_id,
                    recording.recording_id,
                    segment.segment_index,
                )
            )

            # Reference-length and reference-stability grid.
            for reference_count in sensitivity_ref_counts:
                for stability_percent in config["labels"]["reference_instability_sensitivity_percent"]:
                    reference_variant = technical_reference(
                        segment_reference_windows,
                        primary_features,
                        config,
                        count=int(reference_count),
                        stability_percent=float(stability_percent),
                    )
                    variant_labels = label_recording(segment_windows, primary_features, reference_variant, config)
                    label_audit_rows.append(
                        _sensitivity_row(
                            "reference_length_stability",
                            {"reference_window_count": int(reference_count), "reference_stability_percent": float(stability_percent)},
                            reference_variant,
                            variant_labels,
                            recording.group_id,
                            recording.recording_id,
                            segment.segment_index,
                        )
                    )

            # Occupancy sensitivity changes eligibility only; it never alters signal features.
            for occupancy in config["activity"]["window_occupancy_sensitivity"]:
                occupancy = float(occupancy)
                variant_primary = _occupancy_variant(segment_windows, occupancy)
                variant_reference_windows = _occupancy_variant(segment_reference_windows, occupancy)
                occupancy_reference = technical_reference(variant_reference_windows, primary_features, config)
                occupancy_labels = label_recording(variant_primary, primary_features, occupancy_reference, config)
                label_audit_rows.append(
                    _sensitivity_row(
                        "activity_occupancy",
                        {"occupancy_threshold": occupancy},
                        occupancy_reference,
                        occupancy_labels,
                        recording.group_id,
                        recording.recording_id,
                        segment.segment_index,
                    )
                )

            # Relative-change, scaled-MAD, and persistence sensitivity grid.
            for relative_percent in config["labels"]["sensitivity_relative_change_percent"]:
                for mad_multiple in config["labels"]["sensitivity_scaled_mad_multiples"]:
                    for persistence in config["labels"]["sensitivity_persistence_windows"]:
                        for reference_count in config["labels"]["sensitivity_reference_window_counts"]:
                            settings_variant = copy.deepcopy(config)
                            settings_variant["labels"]["entry_window_count"] = int(persistence)
                            settings_variant["labels"]["entry_rolling_window_count"] = int(persistence)
                            settings_variant["labels"]["minimum_entry_span_s"] = 0.0
                            settings_variant["labels"]["reference_window_count"] = int(reference_count)
                            reference_variant = technical_reference(
                                segment_reference_windows,
                                primary_features,
                                settings_variant,
                                count=int(reference_count),
                            )
                            variant_labels = label_recording(
                                segment_windows,
                                primary_features,
                                reference_variant,
                                settings_variant,
                                relative_change=float(relative_percent) / 100.0,
                                scaled_mad_multiple=float(mad_multiple),
                            )
                            label_audit_rows.append(
                                _sensitivity_row(
                                    "threshold_persistence",
                                    {
                                        "relative_change_percent": float(relative_percent),
                                        "scaled_mad_multiple": float(mad_multiple),
                                        "persistence_windows": int(persistence),
                                        "reference_window_count": int(reference_count),
                                    },
                                    reference_variant,
                                    variant_labels,
                                    recording.group_id,
                                    recording.recording_id,
                                    segment.segment_index,
                                )
                            )

    feature_fields = (
        "window_key", "group_id", "recording_id", "segment_index", "window_index", "start_sample",
        "start_time_s", "duration_s", "sampling_rate_hz", "activity_occupancy",
        "window_eligible_at_primary_occupancy", "window_rejection_reason", "feature_valid", "feature_status",
    ) + feature_names
    label_fields = (
        "window_key", "group_id", "recording_id", "segment_index", "start_time_s", "duration_s",
        "class_id", "class_name", "rule_version", "label_confidence", "label_reason", "state_transition",
        "reference_window_keys", "reference_center", "reference_scaled_mad", "observed_indicators",
        "indicator_deltas", "indicator_thresholds", "persistence_candidates", "persistence_span_s", "state_latched",
    )
    audit_fields = (
        "sensitivity_type", "settings_json", "group_id", "recording_id", "segment_index", "reference_status",
        "reference_window_count", "reference_reasons", "ambiguous_count", "non_fatigue_like_count",
        "fatigue_like_count", "total_windows", "label_coverage",
    )
    feature_path = output_dir / "features.csv"
    label_path = output_dir / "pseudo_labels.csv"
    alternative_label_path = output_dir / "pseudo_labels_alternatives.csv"
    window_size_label_path = output_dir / "window_size_sensitivity_labels.csv"
    audit_path = output_dir / "label_sensitivity_by_run.csv"
    feature_row_count = _write_csv(feature_path, feature_fields, feature_rows)
    label_row_count = _write_csv(label_path, label_fields, labels_rows)
    alternative_row_count = _write_csv(alternative_label_path, label_fields + ("comparison_rule",), alternative_label_rows)
    window_size_label_count = _write_csv(window_size_label_path, label_fields, window_size_label_rows)
    audit_row_count = _write_csv(audit_path, audit_fields, label_audit_rows)
    pd.DataFrame(feature_rows).to_parquet(output_dir / "features.parquet", index=False, engine="pyarrow")
    pd.DataFrame(labels_rows).to_parquet(output_dir / "pseudo_labels.parquet", index=False, engine="pyarrow")
    pd.DataFrame(alternative_label_rows).to_parquet(
        output_dir / "pseudo_labels_alternatives.parquet", index=False, engine="pyarrow"
    )
    _write_json(output_dir / "feature_schema.json", {"features": feature_schema_rows, "views": {name: list(names) for name, names in FEATURE_VIEWS.items()}})
    _write_json(
        output_dir / "label_rule_config.json",
        {
            "label_configuration": config["labels"],
            "primary_occupancy": primary_occupancy,
            "window_overlap": overlap,
            "primary_duration_s": primary_duration,
            "label_scope": "EMG-derived operational fatigue-like proxy; not clinical, perceived, or mechanical ground truth",
            "feature_independence": "A1 excludes MDF, MNF, spectral ratios, spectral entropy, bands, and wavelet energies used as label indicators",
        },
    )

    primary_feature_rows = [row for row in feature_rows if float(row["duration_s"]) == primary_duration and row["window_eligible_at_primary_occupancy"] and int(row["feature_valid"]) == 1]
    correlation = _correlation_frame(primary_feature_rows, feature_names)
    correlation.to_csv(output_dir / "feature_correlation_spearman.csv", index=True)

    group_summary_rows = []
    for group_id in sorted(inventory.split.development_groups):
        per_class = {str(label_id): group_counts[(group_id, label_id)] for label_id in (-1, 0, 1)}
        total = sum(per_class.values())
        group_summary_rows.append({
            "group_id": group_id,
            "ambiguous_count": per_class["-1"],
            "non_fatigue_like_count": per_class["0"],
            "fatigue_like_count": per_class["1"],
            "total_windows": total,
            "label_coverage": (per_class["0"] + per_class["1"]) / total if total else 0.0,
        })
    _write_csv(
        output_dir / "label_counts_by_group.csv",
        ("group_id", "ambiguous_count", "non_fatigue_like_count", "fatigue_like_count", "total_windows", "label_coverage"),
        group_summary_rows,
    )
    recording_summary_rows = []
    for group_id, recording_id in sorted({(key[0], key[1]) for key in recording_counts}):
        counts = {str(label_id): recording_counts[(group_id, recording_id, label_id)] for label_id in (-1, 0, 1)}
        total = sum(counts.values())
        recording_summary_rows.append({
            "group_id": group_id,
            "recording_id": recording_id,
            "ambiguous_count": counts["-1"],
            "non_fatigue_like_count": counts["0"],
            "fatigue_like_count": counts["1"],
            "total_windows": total,
            "label_coverage": (counts["0"] + counts["1"]) / total if total else 0.0,
        })
    _write_csv(
        output_dir / "label_counts_by_recording.csv",
        ("group_id", "recording_id", "ambiguous_count", "non_fatigue_like_count", "fatigue_like_count", "total_windows", "label_coverage"),
        recording_summary_rows,
    )
    metadata_summary_rows = []
    for series_token, side_token in sorted({(key[0], key[1]) for key in metadata_counts}):
        counts = {str(label_id): metadata_counts[(series_token, side_token, label_id)] for label_id in (-1, 0, 1)}
        total = sum(counts.values())
        metadata_summary_rows.append({
            "series_token": series_token,
            "side_token": side_token,
            "ambiguous_count": counts["-1"],
            "non_fatigue_like_count": counts["0"],
            "fatigue_like_count": counts["1"],
            "total_windows": total,
            "label_coverage": (counts["0"] + counts["1"]) / total if total else 0.0,
        })
    _write_csv(
        output_dir / "label_counts_by_metadata.csv",
        ("series_token", "side_token", "ambiguous_count", "non_fatigue_like_count", "fatigue_like_count", "total_windows", "label_coverage"),
        metadata_summary_rows,
    )
    temporal_summary_rows = []
    for group_id in inventory.split.development_groups:
        for quartile in range(4):
            counts = {str(label_id): temporal_counts[(group_id, quartile, label_id)] for label_id in (-1, 0, 1)}
            total = sum(counts.values())
            temporal_summary_rows.append({
                "group_id": group_id,
                "relative_recording_time_quartile": quartile + 1,
                "ambiguous_count": counts["-1"],
                "non_fatigue_like_count": counts["0"],
                "fatigue_like_count": counts["1"],
                "total_windows": total,
                "label_coverage": (counts["0"] + counts["1"]) / total if total else 0.0,
            })
    _write_csv(
        output_dir / "label_counts_by_temporal_quartile.csv",
        ("group_id", "relative_recording_time_quartile", "ambiguous_count", "non_fatigue_like_count", "fatigue_like_count", "total_windows", "label_coverage"),
        temporal_summary_rows,
    )

    agreement_rows = []
    rule_names = tuple(label_by_rule)
    by_key = {
        rule: {label.window_key: label for label in values}
        for rule, values in label_by_rule.items()
    }
    for left_index, left_rule in enumerate(rule_names):
        for right_rule in rule_names[left_index + 1 :]:
            common_keys = sorted(set(by_key[left_rule]) & set(by_key[right_rule]))
            comparable = [
                (by_key[left_rule][key], by_key[right_rule][key])
                for key in common_keys
                if by_key[left_rule][key].class_id >= 0 and by_key[right_rule][key].class_id >= 0
            ]
            exact = sum(left.class_id == right.class_id for left, right in comparable)
            left_coverage = sum(item.class_id >= 0 for item in by_key[left_rule].values())
            right_coverage = sum(item.class_id >= 0 for item in by_key[right_rule].values())
            agreement_rows.append(
                {
                    "rule_a": left_rule,
                    "rule_b": right_rule,
                    "shared_windows": len(common_keys),
                    "both_rules_assignable": len(comparable),
                    "exact_class_agreement_count": exact,
                    "agreement_when_both_assign": exact / len(comparable) if comparable else None,
                    "rule_a_assigned_count": left_coverage,
                    "rule_b_assigned_count": right_coverage,
                    "rule_a_coverage": left_coverage / len(by_key[left_rule]) if by_key[left_rule] else 0.0,
                    "rule_b_coverage": right_coverage / len(by_key[right_rule]) if by_key[right_rule] else 0.0,
                }
            )
    agreement_path = output_dir / "label_rule_agreement.csv"
    _write_csv(
        agreement_path,
        ("rule_a", "rule_b", "shared_windows", "both_rules_assignable", "exact_class_agreement_count", "agreement_when_both_assign", "rule_a_assigned_count", "rule_b_assigned_count", "rule_a_coverage", "rule_b_coverage"),
        agreement_rows,
    )

    class_counts = Counter(str(row["class_id"]) for row in labels_rows)
    alternative_counts = {
        rule: {label_id: sum(label.class_id == int(label_id) for label in values) for label_id in ("-1", "0", "1")}
        for rule, values in label_by_rule.items()
    }
    recipe = {
        "stage": "development_feature_extraction_and_weak_label_audit",
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_archive_sha256": training_sha,
        "config_sha256": _sha256_file(config_path),
        "git_branch": _git(repo_root, "branch", "--show-current"),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "development_group_ids": list(inventory.split.development_groups),
        "internal_test_group_ids": list(inventory.split.test_groups),
        "development_recording_count": len(recordings),
        "development_continuous_run_count": run_count,
        "internal_test_numeric_values_parsed": 0,
        "external_archive_opened": False,
        "label_scope": "EMG-derived operational proxy only; not clinical, perceived, or mechanically verified fatigue",
        "primary_label_rule": config["labels"]["version"],
        "primary_label_counts": {label_id: class_counts.get(label_id, 0) for label_id in ("-1", "0", "1")},
        "label_counts_by_rule": alternative_counts,
        "reference_run_counts": {
            "unavailable_fewer_than_five": reference_unavailable_count,
            "available_but_unstable": reference_unstable_count,
        },
        "feature_row_count": feature_row_count,
        "primary_label_row_count": label_row_count,
        "alternative_label_row_count": alternative_row_count,
        "window_size_sensitivity_label_row_count": window_size_label_count,
        "sensitivity_run_row_count": audit_row_count,
        "feature_valid_primary_window_count": len(primary_feature_rows),
        "sensitivity_grid": {
            "reference_window_counts": [3, 5],
            "reference_stability_percent": list(config["labels"]["reference_instability_sensitivity_percent"]),
            "activity_occupancy": list(config["activity"]["window_occupancy_sensitivity"]),
            "relative_change_percent": list(config["labels"]["sensitivity_relative_change_percent"]),
            "scaled_mad_multiples": list(config["labels"]["sensitivity_scaled_mad_multiples"]),
            "persistence_windows": list(config["labels"]["sensitivity_persistence_windows"]),
        },
        "artifacts": [
            "features.parquet",
            "features.csv",
            "pseudo_labels.parquet",
            "pseudo_labels.csv",
            "pseudo_labels_alternatives.parquet",
            "pseudo_labels_alternatives.csv",
            "window_size_sensitivity_labels.csv",
            "label_sensitivity_by_run.csv",
            "label_rule_agreement.csv",
            "label_counts_by_group.csv",
            "label_counts_by_recording.csv",
            "label_counts_by_metadata.csv",
            "label_counts_by_temporal_quartile.csv",
            "feature_schema.json",
            "feature_correlation_spearman.csv",
            "label_rule_config.json",
        ],
    }
    source_paths = (
        "src/emg_fatigue/weak_supervision/features.py",
        "src/emg_fatigue/weak_supervision/pseudo_labels.py",
        "src/emg_fatigue/weak_supervision/label_audit.py",
        "src/emg_fatigue/weak_supervision/preprocess.py",
        "src/emg_fatigue/weak_supervision/activity.py",
        "src/emg_fatigue/weak_supervision/windows.py",
        "src/emg_fatigue/weak_supervision/io.py",
        "src/emg_fatigue/weak_supervision/contracts.py",
        "src/emg_fatigue/weak_supervision/runner.py",
        "tools/run_fatigue_proxy.py",
    )
    recipe["source_files_sha256"] = {path: _sha256_file(repo_root / path) for path in source_paths}
    recipe["software_versions"] = {
        "python": __import__("sys").version.split()[0],
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "pandas": importlib.metadata.version("pandas"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }
    recipe_path = output_dir / "feature_label_stage_manifest.json"
    _write_json(recipe_path, recipe)

    run_manifest["stage"] = "development_feature_extraction_and_weak_label_audit"
    run_manifest["feature_label_stage_manifest_sha256"] = _sha256_file(recipe_path)
    run_manifest["git_branch"] = _git(repo_root, "branch", "--show-current")
    run_manifest["git_commit"] = _git(repo_root, "rev-parse", "HEAD")
    run_manifest["git_working_tree_status"] = _git(repo_root, "status", "--short")
    source_files = sorted((repo_root / "src" / "emg_fatigue" / "weak_supervision").glob("*.py"))
    source_files.append(repo_root / "tools" / "run_fatigue_proxy.py")
    source_hashes = {
        path.relative_to(repo_root).as_posix(): _sha256_file(path)
        for path in source_files
        if path.is_file()
    }
    run_manifest["source_fingerprint_files"] = source_hashes
    run_manifest["source_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_manifest["output_sha256"] = {
        path.relative_to(private_dir).as_posix(): _sha256_file(path)
        for path in sorted(private_dir.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    _write_json(run_manifest_path, run_manifest)
    return {
        "development_recording_count": len(recordings),
        "feature_row_count": feature_row_count,
        "primary_label_row_count": label_row_count,
        "alternative_label_row_count": alternative_row_count,
        "window_size_sensitivity_label_row_count": window_size_label_count,
        "sensitivity_run_row_count": audit_row_count,
        "primary_label_counts": recipe["primary_label_counts"],
        "label_counts_by_rule": recipe["label_counts_by_rule"],
        "reference_run_counts": recipe["reference_run_counts"],
        "feature_label_manifest_path": str(recipe_path),
    }
