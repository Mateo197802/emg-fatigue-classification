"""Private, target-free descriptive audit for exploratory sEMG features."""

import argparse
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .activity import detect_activity
from .candidate_features import CANDIDATE_FEATURE_COLUMNS, CANDIDATE_FEATURE_SCHEMA, candidate_window_features
from .contracts import output_root, read_config, sha256_file, write_json, implementation_provenance
from .features import FEATURE_COLUMNS, window_features
from .io import load_recordings
from .preprocessing import preprocess


def _pair_correlations(frame, columns, *, analysis_scope):
    rows = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            a = pd.to_numeric(frame[left], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(frame[right], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            a = a[valid]
            b = b[valid]
            status = "defined"
            rho = None
            if len(a) < 3:
                status = "insufficient_observations"
            elif np.ptp(a) == 0 or np.ptp(b) == 0:
                status = "undefined_constant_feature"
            else:
                result = spearmanr(a, b)
                if np.isfinite(result.statistic):
                    rho = float(result.statistic)
                else:
                    status = "undefined_nonfinite_result"
            rows.append({
                "feature_a": left,
                "feature_b": right,
                "spearman_rho": rho,
                "n_observations": int(len(a)),
                "analysis_scope": analysis_scope,
                "status": status,
            })
    return pd.DataFrame(rows)


def _extract_target_free_windows(recordings, config):
    rows = []
    run_audit = []
    for recording in recordings:
        if recording.series == "R0":
            continue
        processed = detect_activity(preprocess(recording, config), config)
        for run in processed["runs"]:
            fs = float(run["fs_hz"])
            window_samples = int(round(config.window_s * fs))
            hop_samples = int(round(window_samples * (1.0 - config.overlap)))
            if window_samples < 56 or hop_samples < 1:
                raise ValueError("configured window produces invalid feature or hop length")
            valid = np.asarray(run["valid"], dtype=bool)
            active = np.asarray(run["active"], dtype=bool)
            x = np.asarray(run["x"], dtype=float)
            count_before = len(rows)
            rejected_invalid = 0
            rejected_inactive = 0
            rejected_feature = 0
            for start in range(0, len(x) - window_samples + 1, hop_samples):
                end = start + window_samples
                if not valid[start:end].all():
                    rejected_invalid += 1
                    continue
                occupancy = float(active[start:end].mean())
                if occupancy < config.activity_occupancy:
                    rejected_inactive += 1
                    continue
                try:
                    primary = window_features(x[start:end], fs)
                    candidate = candidate_window_features(x[start:end], fs)
                except ValueError:
                    rejected_feature += 1
                    continue
                row = {
                    "recording_id": recording.recording_id,
                    "group_id": recording.subject_id,
                    "side": recording.side,
                    "series": recording.series,
                    "run_id": int(run["run_id"]),
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "start_time_s": float(run["t"][start]),
                    "end_time_s": float(run["t"][end - 1]),
                    "sampling_rate_hz": fs,
                    "activity_occupancy": occupancy,
                    **primary,
                    **candidate,
                }
                rows.append(row)
            run_audit.append({
                "recording_id": recording.recording_id,
                "group_id": recording.subject_id,
                "run_id": int(run["run_id"]),
                "sampling_rate_hz": fs,
                "window_samples": window_samples,
                "hop_samples": hop_samples,
                "valid_candidate_windows": len(rows) - count_before,
                "rejected_invalid_window": rejected_invalid,
                "rejected_activity_occupancy": rejected_inactive,
                "rejected_feature_calculation": rejected_feature,
            })
    if not rows:
        raise ValueError("no target-free QC-valid candidate windows were available")
    return pd.DataFrame(rows), pd.DataFrame(run_audit)


def build_candidate_feature_audit(recordings, config, *, output_dir, source_sha256=None):
    """Write a private feature/redundancy audit; never reads or writes labels."""
    root = output_root(output_dir)
    frame, run_audit = _extract_target_free_windows(recordings, config)
    candidate_columns = list(CANDIDATE_FEATURE_COLUMNS)
    full_columns = list(FEATURE_COLUMNS) + candidate_columns
    group_summary = (
        frame.groupby("group_id", sort=True)[full_columns]
        .median()
        .reset_index()
    )
    group_summary.insert(1, "recording_count", frame.groupby("group_id")["recording_id"].nunique().reindex(group_summary.group_id).to_numpy())
    group_summary.insert(2, "window_count", frame.groupby("group_id").size().reindex(group_summary.group_id).to_numpy())

    window_correlations = _pair_correlations(
        frame,
        full_columns,
        analysis_scope="descriptive_overlapping_windows",
    )
    group_correlations = _pair_correlations(
        group_summary,
        full_columns,
        analysis_scope="descriptive_group_medians_identity_unverified",
    )
    correlations = pd.concat([window_correlations, group_correlations], ignore_index=True)

    feature_path = root / "candidate_features_private.parquet"
    run_path = root / "candidate_feature_run_audit_private.csv"
    group_path = root / "candidate_feature_group_summary_private.csv"
    correlation_path = root / "candidate_feature_correlations_private.csv"
    frame.to_parquet(feature_path, index=False)
    run_audit.to_csv(run_path, index=False)
    group_summary.to_csv(group_path, index=False)
    correlations.to_csv(correlation_path, index=False)

    synthetic = bool(recordings) and all(r.source == "synthetic" for r in recordings)
    receipt = {
        "schema": CANDIDATE_FEATURE_SCHEMA,
        "evidence_status": "synthetic_only" if synthetic else "descriptive_candidate_audit_only",
        "target_used": False,
        "label_columns_written": [],
        "source_sha256": source_sha256,
        "recording_count": int(frame["recording_id"].nunique()),
        "group_count": int(frame["group_id"].nunique()),
        "group_identity_verified": False,
        "candidate_window_count": int(len(frame)),
        "window_s": float(config.window_s),
        "overlap": float(config.overlap),
        "primary_feature_columns": list(FEATURE_COLUMNS),
        "candidate_feature_columns": candidate_columns,
        "descriptive_only_limitations": [
            "No fatigue or movement-phase target was read or fitted.",
            "Overlapping windows are correlated and are not independent observations.",
            "Group identifiers are folder-derived and participant identity is unverified.",
            "Feature correlations do not establish predictive value or improve Accuracy/F1.",
        ],
        "configuration": {
            "window_s": float(config.window_s),
            "overlap": float(config.overlap),
            "band_hz": list(config.band_hz),
            "filter_order": int(config.filter_order),
            "edge_mask_s": float(config.edge_s),
            "activity_occupancy": float(config.activity_occupancy),
            "notch": bool(config.notch),
            "config_sha256": config.digest(),
        },
        "implementation": implementation_provenance(),
        "outputs_sha256": {
            feature_path.name: sha256_file(feature_path),
            run_path.name: sha256_file(run_path),
            group_path.name: sha256_file(group_path),
            correlation_path.name: sha256_file(correlation_path),
        },
    }
    write_json(root / "candidate_feature_audit.receipt.json", receipt)
    return {
        "windows": frame,
        "run_audit": run_audit,
        "group_summary": group_summary,
        "window_correlations": window_correlations,
        "group_correlations": group_correlations,
        "receipt": receipt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    recordings = load_recordings(archive, read_config(args.config))
    result = build_candidate_feature_audit(
        recordings,
        read_config(args.config),
        output_dir=args.output_dir,
        source_sha256=sha256_file(archive),
    )
    print(json.dumps(result["receipt"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
