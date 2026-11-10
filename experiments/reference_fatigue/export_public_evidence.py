"""Export aggregate benchmark evidence without publishing row-level data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PUBLIC_METRIC_COLUMNS = [
    "architecture",
    "n_windows",
    "participants",
    "accuracy",
    "f1_macro",
    "balanced_accuracy",
    "precision_fatigue",
    "recall_fatigue",
    "specificity_no_fatigue",
    "roc_auc",
    "pr_auc",
    "accuracy_participant_macro",
    "accuracy_ci95_low",
    "accuracy_ci95_high",
    "f1_macro_participant_macro",
    "f1_macro_ci95_low",
    "f1_macro_ci95_high",
    "balanced_accuracy_participant_macro",
    "balanced_accuracy_ci95_low",
    "balanced_accuracy_ci95_high",
    "sensitivity_fatigue_participant_macro",
    "sensitivity_fatigue_ci95_low",
    "sensitivity_fatigue_ci95_high",
    "specificity_no_fatigue_participant_macro",
    "specificity_no_fatigue_ci95_low",
    "specificity_no_fatigue_ci95_high",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_public_evidence(run_dir: Path, output_dir: Path, *, paired_replicates=50_000):
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "run_manifest.json",
        "dataset_audit.json",
        "sampling_rate_audit.csv",
        "qc_summary.csv",
        "benchmark_results.csv",
        "benchmark_by_participant.csv",
        "confusion_matrices.csv",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"run directory lacks evidence: {missing}")

    private_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    dataset_audit = json.loads((run_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "benchmark_results.csv")
    by_participant = pd.read_csv(run_dir / "benchmark_by_participant.csv")
    confusion = pd.read_csv(run_dir / "confusion_matrices.csv")
    sampling = pd.read_csv(run_dir / "sampling_rate_audit.csv")
    qc = pd.read_csv(run_dir / "qc_summary.csv")

    missing_metrics = sorted(set(PUBLIC_METRIC_COLUMNS) - set(metrics.columns))
    if missing_metrics:
        raise ValueError(f"benchmark metrics are missing columns: {missing_metrics}")
    if by_participant["subject_id"].nunique() != 13:
        raise ValueError("expected 13 participants for paired comparisons")
    if by_participant.duplicated(["architecture", "subject_id"]).any():
        raise ValueError("participant metrics contain duplicate model/participant rows")
    if not {"time_only_control", "majority"}.issubset(set(by_participant["architecture"])):
        raise ValueError("required reference controls are missing")

    metrics[PUBLIC_METRIC_COLUMNS].to_csv(
        output_dir / "benchmark_metrics.csv",
        index=False,
        float_format="%.10g",
        lineterminator="\n",
    )
    pooled = confusion.loc[confusion["scope"].eq("pooled_oof_windows")].copy()
    confusion_support = pooled.groupby("architecture")["count"].sum().sort_index()
    metric_support = metrics.set_index("architecture")["n_windows"].sort_index()
    if not confusion_support.equals(metric_support):
        raise ValueError("pooled confusion counts do not match benchmark OOF support")
    if pooled.groupby("architecture").size().ne(4).any():
        raise ValueError("each pooled binary confusion matrix must contain four cells")
    confusion_columns = [
        "architecture",
        "actual_label",
        "predicted_label",
        "count",
        "row_normalized",
        "n_windows",
    ]
    pooled[confusion_columns].to_csv(
        output_dir / "confusion_matrices.csv",
        index=False,
        float_format="%.10g",
        lineterminator="\n",
    )

    paired = by_participant.pivot(
        index="subject_id", columns="architecture", values=["accuracy", "f1_macro"]
    )
    if paired.isna().any().any():
        raise ValueError("paired participant metrics contain missing values")
    rng = np.random.default_rng(20261109)
    draws_idx = rng.integers(0, len(paired), size=(paired_replicates, len(paired)))
    delta_rows = []
    for metric in ("accuracy", "f1_macro"):
        control = paired[(metric, "time_only_control")].to_numpy(dtype=float)
        for model in sorted(set(by_participant["architecture"]) - {"time_only_control", "majority"}):
            delta = paired[(metric, model)].to_numpy(dtype=float) - control
            draws = delta[draws_idx].mean(axis=1)
            delta_rows.append(
                {
                    "emg_model": model,
                    "metric": metric,
                    "comparison": "model_minus_time_only_control",
                    "participants": int(len(delta)),
                    "mean_paired_difference": float(delta.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "participants_model_higher": int((delta > 0).sum()),
                    "bootstrap_replicates": int(paired_replicates),
                    "bootstrap_seed": 20261109,
                }
            )
    pd.DataFrame(delta_rows).to_csv(
        output_dir / "paired_delta_vs_time_only.csv",
        index=False,
        float_format="%.10g",
        lineterminator="\n",
    )

    artifact_hashes = {
        name: _sha256(output_dir / name)
        for name in (
            "benchmark_metrics.csv",
            "confusion_matrices.csv",
            "paired_delta_vs_time_only.csv",
        )
    }
    receipt = {
        "run_id": private_manifest["run_id"],
        "dataset_id": private_manifest["dataset_id"],
        "dataset_version": private_manifest["dataset_version"],
        "source_record": dataset_audit["source_url"],
        "dataset_license": dataset_audit["license"],
        "task_id": private_manifest["task_id"],
        "label_definition_id": private_manifest["label_definition_id"],
        "prediction_unit": private_manifest["prediction_unit"],
        "cohort": {
            "participants": int(dataset_audit["participants"]),
            "trials": int(dataset_audit["trials"]),
            "channels_per_trial": int(dataset_audit["channels_per_trial"]),
            "channel_names": dataset_audit["verified_active_muscles"],
            "signal_unit": dataset_audit["signal_unit"],
            "measured_fs_hz_min": float(sampling["effective_fs_hz"].min()),
            "measured_fs_hz_max": float(sampling["effective_fs_hz"].max()),
        },
        "label_counts": dataset_audit["eligible_windows_by_class"],
        "eligible_windows": int(private_manifest["feature_rows"]),
        "preprocessing_qc": {
            "channel_trial_records": int(len(sampling)),
            "gaps": int(sampling["gap_count"].sum()),
            "nonmonotonic_intervals": int(sampling["nonmonotonic_intervals"].sum()),
            "nonfinite_samples": int(qc["nonfinite_samples"].sum()),
            "terminal_clock_reset_rows_dropped": int(dataset_audit["raw_terminal_reset_rows_dropped"]),
            "trials_with_terminal_clock_reset": int(dataset_audit["raw_terminal_reset_trials"]),
            "sensor_clipping": "not assessable; sensor rail limits absent",
            "filter": "Butterworth bandpass prototype order 4, 20-450 Hz, SOS, zero-phase",
            "notch": "50/60 Hz conditional on prominence; none triggered",
        },
        "window": {
            "length_ms": 500,
            "overlap_percent": 50,
            "rule": "stable reference label over the entire window; transition label 1 excluded",
        },
        "outer_validation": "13-fold leave-one-participant-out; all records of a held-out participant excluded from training",
        "training_sampling": "Even temporal-bin sampling; maximum 50 windows per training participant x trial x class; all test windows retained",
        "model_selection": "Fixed model settings; no hyperparameter search or selection on outer test folds",
        "seed": int(private_manifest["seed"]),
        "participant_bootstrap_replicates": 5000,
        "paired_difference_bootstrap": {
            "replicates": int(paired_replicates),
            "seed": 20261109,
            "resampling_unit": "participant",
        },
        "code_revision": private_manifest["code_revision"],
        "source_archive_hashes": private_manifest["dataset_source_hashes"],
        "private_feature_table_sha256": private_manifest["feature_table_sha256"],
        "private_oof_predictions_sha256": private_manifest["predictions_sha256"],
        "models_completed": private_manifest["models_completed"],
        "aggregate_artifact_sha256": artifact_hashes,
        "phase_reference_available": False,
        "mechanical_force_reference_available": False,
        "scope_note": "Self-reported fatigue only. Aggregate evidence is public; per-window OOF predictions, per-participant metrics, raw signals and subject-level audit tables remain local and are not tracked.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json"}:
            content = path.read_text(encoding="utf-8")
            if "subject_id" in content or "subject_" in content:
                raise ValueError(f"public artifact contains participant-level identifiers: {path.name}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paired-replicates", type=int, default=50_000)
    args = parser.parse_args()
    receipt = export_public_evidence(
        args.run_dir, args.output_dir, paired_replicates=args.paired_replicates
    )
    print(f"Exported aggregate evidence for {receipt['run_id']}")
    print(f"Code revision: {receipt['code_revision']}")
    print(f"Models: {', '.join(receipt['models_completed'])}")


if __name__ == "__main__":
    main()
