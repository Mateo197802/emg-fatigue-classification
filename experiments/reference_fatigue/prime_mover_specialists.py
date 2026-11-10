"""Compare global and prime-mover specialist EMG fatigue models.

This exploratory architecture uses the dataset's known prime-mover task as a
router. It does not predict knee extension/flexion phase. All labels remain
the time-aligned self-reported fatigue states from the Zenodo source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from emg_fatigue.perceived_benchmark import (
    _make_estimator,
    _participant_bootstrap,
    _predict,
    binary_metric_summary,
    equalized_training_sample,
    leave_one_participant_out,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO_ROOT / "experiments" / "reference_fatigue" / "run_001" / "features.parquet"
PRIVATE_OUTPUT = REPO_ROOT / "private" / "zenodo_14182446" / "prime_mover_specialists_run_001"
PUBLIC_OUTPUT = REPO_ROOT / "results" / "reference_fatigue"
SEED = 20261109
SAMPLE_CAP = 50
MODEL_NAME = "hist_gradient_boosting"

METADATA_COLUMNS = {
    "window_id", "subject_id", "trial_id", "prime_mover", "window_start_s",
    "window_end_s", "window_center_s", "label", "label_status",
}


def _default_expert_factory(seed: int):
    return _make_estimator(MODEL_NAME, seed)


def fit_prime_mover_experts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int = SEED,
    estimator_factory=None,
) -> pd.DataFrame:
    """Fit one EMG fatigue expert per known prime mover and route test rows.

    ``prime_mover`` is used only as an externally supplied task identifier for
    routing. Each expert receives only sEMG features. Participant overlap is
    rejected so the helper cannot accidentally evaluate a seen participant.
    """
    required = {"window_id", "subject_id", "trial_id", "prime_mover", "label"}
    for name, frame in (("train", train), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} rows lack required columns: {sorted(missing)}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} rows contain missing participant, task, or label values")
        if not set(frame["label"].astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"{name} labels must be binary 0/1")
        absent_features = set(feature_columns) - set(frame.columns)
        if absent_features:
            raise ValueError(f"{name} rows lack features: {sorted(absent_features)}")
        values = frame[feature_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} features contain nonfinite values")
    overlap = set(train["subject_id"].astype(str)) & set(test["subject_id"].astype(str))
    if overlap:
        raise ValueError(f"participant overlap between train and test: {sorted(overlap)}")
    if train["window_id"].duplicated().any() or test["window_id"].duplicated().any():
        raise ValueError("window_id must be unique within each split")

    factory = estimator_factory or _default_expert_factory
    outputs = []
    test_movers = sorted(test["prime_mover"].astype(str).unique())
    train_movers = set(train["prime_mover"].astype(str).unique())
    for index, mover in enumerate(test_movers):
        if mover not in train_movers:
            raise ValueError(f"no training specialist exists for test task {mover!r}")
        expert_train = train[train["prime_mover"].astype(str) == mover]
        expert_test = test[test["prime_mover"].astype(str) == mover]
        if set(expert_train["label"].astype(int).unique()) != {0, 1}:
            raise ValueError(f"specialist {mover!r} requires both fatigue labels in training")
        estimator = factory(seed + index)
        estimator.fit(
            expert_train[feature_columns].to_numpy(dtype=float),
            expert_train["label"].to_numpy(dtype=int),
        )
        prediction, score, score_type = _predict(
            estimator, expert_test[feature_columns].to_numpy(dtype=float)
        )
        outputs.append(pd.DataFrame({
            "model": "specialist_by_prime_mover",
            "expert_id": mover,
            "window_id": expert_test["window_id"].astype(str).to_numpy(),
            "subject_id": expert_test["subject_id"].astype(str).to_numpy(),
            "trial_id": expert_test["trial_id"].astype(str).to_numpy(),
            "y_true": expert_test["label"].astype(int).to_numpy(),
            "y_pred": prediction,
            "score": score,
            "score_type": score_type,
        }))
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    if len(result) != len(test) or result["window_id"].duplicated().any():
        raise RuntimeError("specialist routing did not produce exactly one prediction per test window")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summarize_model(rows: pd.DataFrame, *, replicates: int, seed: int) -> tuple[dict, pd.DataFrame]:
    metrics = binary_metric_summary(rows["y_true"], rows["y_pred"], rows["score"])
    bootstrap, by_participant = _participant_bootstrap(
        rows, replicates=replicates, seed=seed
    )
    return {
        "model": str(rows["model"].iloc[0]),
        "n_windows": int(len(rows)),
        "participants": int(rows["subject_id"].nunique()),
        **{key: value for key, value in metrics.items() if key != "n_windows"},
        **bootstrap,
    }, pd.DataFrame.from_dict(by_participant, orient="index").rename_axis("subject_id").reset_index()


def _paired_bootstrap_delta(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    metric: str,
    replicates: int = 50_000,
    seed: int = SEED,
) -> dict:
    grouped = {}
    for name, frame in (("candidate", candidate), ("reference", reference)):
        grouped[name] = {
            str(subject): binary_metric_summary(group["y_true"], group["y_pred"], group["score"])[metric]
            for subject, group in frame.groupby("subject_id", sort=True)
        }
    subjects = sorted(set(grouped["candidate"]) & set(grouped["reference"]))
    differences = np.asarray([
        grouped["candidate"][subject] - grouped["reference"][subject]
        for subject in subjects
    ], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
    draws = differences[indices].mean(axis=1)
    return {
        "metric": metric,
        "comparison": "specialist_by_prime_mover_minus_global_hist_gradient_boosting",
        "participants": len(subjects),
        "mean_paired_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "participants_candidate_higher": int(np.sum(differences > 0)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def run_comparison(
    features_path: Path = FEATURES_PATH,
    private_output: Path = PRIVATE_OUTPUT,
    public_output: Path = PUBLIC_OUTPUT,
    *,
    sample_cap: int = SAMPLE_CAP,
    seed: int = SEED,
    bootstrap_replicates: int = 5000,
) -> tuple[pd.DataFrame, dict]:
    """Run a leave-one-participant-out global vs task-specialist comparison."""
    features_path = Path(features_path).resolve()
    private_output = Path(private_output).resolve()
    public_output = Path(public_output).resolve()
    private_output.mkdir(parents=True, exist_ok=True)
    public_output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(features_path)
    feature_columns = [column for column in frame.columns if column not in METADATA_COLUMNS]
    if frame["subject_id"].nunique() != 13:
        raise ValueError("expected 13 verified participants in the source benchmark")

    outputs = {"global_hist_gradient_boosting": [], "specialist_by_prime_mover": []}
    for fold_number, split in enumerate(leave_one_participant_out(frame), start=1):
        train_full = frame.iloc[split["train_indices"]].copy()
        test = frame.iloc[split["test_indices"]].copy()
        train = equalized_training_sample(
            train_full, max_per_trial_class=sample_cap, seed=seed + fold_number
        )
        if set(train["label"].astype(int).unique()) != {0, 1}:
            raise ValueError(f"outer fold {fold_number} training sample lacks both fatigue labels")

        model_seed = seed + fold_number * 100 + 4
        global_model = _make_estimator(MODEL_NAME, model_seed)
        global_model.fit(
            train[feature_columns].to_numpy(dtype=float), train["label"].to_numpy(dtype=int)
        )
        pred, score, score_type = _predict(
            global_model, test[feature_columns].to_numpy(dtype=float)
        )
        outputs["global_hist_gradient_boosting"].append(pd.DataFrame({
            "model": "global_hist_gradient_boosting",
            "expert_id": "all_prime_movers",
            "window_id": test["window_id"].astype(str).to_numpy(),
            "subject_id": test["subject_id"].astype(str).to_numpy(),
            "trial_id": test["trial_id"].astype(str).to_numpy(),
            "y_true": test["label"].astype(int).to_numpy(),
            "y_pred": pred,
            "score": score,
            "score_type": score_type,
        }))
        specialist_rows = fit_prime_mover_experts(
            train,
            test,
            feature_columns,
            seed=model_seed,
        )
        outputs["specialist_by_prime_mover"].append(specialist_rows)
        print(
            f"LOPO {fold_number}/13 held out one participant; "
            f"global={len(test)} windows; specialists={len(specialist_rows)} windows",
            flush=True,
        )

    oof = {name: pd.concat(parts, ignore_index=True) for name, parts in outputs.items()}
    for name, rows in oof.items():
        if len(rows) != len(frame) or rows["window_id"].duplicated().any():
            raise RuntimeError(f"{name}: OOF output must contain one prediction per eligible window")
    oof_path = private_output / "oof_predictions.parquet"
    pd.concat(oof.values(), ignore_index=True).to_parquet(oof_path, index=False)

    metric_rows = []
    participant_rows = []
    confusion_rows = []
    for index, (name, rows) in enumerate(oof.items()):
        summary, participant = _summarize_model(
            rows, replicates=bootstrap_replicates, seed=seed + index
        )
        metric_rows.append(summary)
        participant["model"] = name
        participant_rows.append(participant)
        matrix = confusion_matrix(rows["y_true"], rows["y_pred"], labels=[0, 1])
        totals = matrix.sum(axis=1)
        for actual in (0, 1):
            for predicted in (0, 1):
                confusion_rows.append({
                    "model": name,
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "count": int(matrix[actual, predicted]),
                    "row_normalized": float(matrix[actual, predicted] / totals[actual]) if totals[actual] else float("nan"),
                    "n_windows": int(len(rows)),
                })

    global_rows = oof["global_hist_gradient_boosting"]
    expert_rows = oof["specialist_by_prime_mover"]
    paired_rows = [
        _paired_bootstrap_delta(
            expert_rows,
            global_rows,
            metric=metric,
            replicates=50_000,
            seed=seed + index,
        )
        for index, metric in enumerate(("accuracy", "f1_macro"))
    ]
    aggregate_names = {
        "prime_mover_specialist_metrics.csv": pd.DataFrame(metric_rows),
        "prime_mover_specialist_confusions.csv": pd.DataFrame(confusion_rows),
        "prime_mover_specialist_paired_differences.csv": pd.DataFrame(paired_rows),
    }
    hashes = {}
    for name, table in aggregate_names.items():
        path = public_output / name
        table.to_csv(path, index=False, float_format="%.10g")
        hashes[name] = _sha256(path)
    private_participant_path = private_output / "participant_metrics.csv"
    pd.concat(participant_rows, ignore_index=True).to_csv(private_participant_path, index=False)

    manifest = {
        "status": "completed",
        "dataset_id": "10.5281/zenodo.14182446",
        "task_id": "PERCEIVED_FATIGUE_CLASSIFICATION",
        "label_definition": "stable self-report 0=no perceived fatigue; stable self-report 2=perceived fatigue; transition state excluded",
        "prediction_unit": "reference-aligned 500 ms windows; one OOF prediction per eligible window",
        "models": ["global_hist_gradient_boosting", "specialist_by_prime_mover"],
        "specialist_router": "recorded prime_mover task metadata; no phase inference",
        "outer_validation": "13-fold leave-one-participant-out; all participant windows held out together",
        "training_sample": f"even temporal sampling, maximum {sample_cap} windows per training participant x trial x class; test windows retained in full",
        "feature_columns": feature_columns,
        "hyperparameters": "same fixed HistGradientBoostingClassifier settings as run_001; no tuning on outer participants",
        "seed": seed,
        "participant_bootstrap_replicates": bootstrap_replicates,
        "paired_bootstrap_replicates": 50_000,
        "paired_bootstrap_unit": "participant",
        "features_sha256": _sha256(features_path),
        "baseline_run_manifest_sha256": _sha256(REPO_ROOT / "results" / "reference_fatigue" / "run_manifest.json"),
        "private_oof_sha256": _sha256(oof_path),
        "private_participant_metrics_sha256": _sha256(private_participant_path),
        "public_aggregate_sha256": hashes,
        "scope_limit": "self-reported fatigue only; specialists are routed by task/prime mover, not by extension/flexion phase; objective muscle-fatigue labels are unavailable",
    }
    manifest_path = public_output / "prime_mover_specialist_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = _sha256(manifest_path)
    private_manifest_path = private_output / "run_manifest.json"
    private_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metric_frame = pd.DataFrame(metric_rows)
    return metric_frame, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--private-output", type=Path, default=PRIVATE_OUTPUT)
    parser.add_argument("--public-output", type=Path, default=PUBLIC_OUTPUT)
    parser.add_argument("--sample-cap", type=int, default=SAMPLE_CAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    metrics, manifest = run_comparison(
        args.features,
        args.private_output,
        args.public_output,
        sample_cap=args.sample_cap,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(metrics.to_string(index=False))
    print(f"status={manifest['status']}; public_aggregates={manifest['public_aggregate_sha256']}")


if __name__ == "__main__":
    main()
