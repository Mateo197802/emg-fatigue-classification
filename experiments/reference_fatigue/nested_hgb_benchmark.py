"""Nested participant-grouped tuning for the direct perceived-fatigue model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

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
BASELINE_OOF_PATH = REPO_ROOT / "experiments" / "reference_fatigue" / "run_001" / "oof_predictions.parquet"
PRIVATE_OUTPUT = REPO_ROOT / "private" / "zenodo_14182446" / "nested_hgb_run_001"
PUBLIC_OUTPUT = REPO_ROOT / "results" / "reference_fatigue"
SEED = 20261109
SAMPLE_CAP = 50
MODEL_NAME = "hist_gradient_boosting"
METADATA_COLUMNS = {
    "window_id", "subject_id", "trial_id", "prime_mover", "window_start_s",
    "window_end_s", "window_center_s", "label", "label_status",
}


def _estimator(params: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=int(params.get("max_iter", 150)),
        learning_rate=float(params.get("learning_rate", 0.08)),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
        l2_regularization=float(params.get("l2_regularization", 1.0)),
        early_stopping=False,
        random_state=seed,
    )


def default_candidate_grid() -> list[dict]:
    """A compact, fixed grid containing the original HGB configuration."""
    return [
        {
            "max_iter": int(iterations),
            "learning_rate": float(rate),
            "max_leaf_nodes": int(leaves),
            "l2_regularization": float(l2),
        }
        for iterations, rate, leaves, l2 in itertools.product(
            (75, 150), (0.03, 0.08), (7, 15), (1.0, 10.0)
        )
    ]


def _participant_macro_f1(valid: pd.DataFrame, predicted: np.ndarray) -> float:
    rows = valid[["subject_id", "label"]].copy()
    rows["prediction"] = np.asarray(predicted, dtype=int)
    values = [
        float(f1_score(group["label"], group["prediction"], labels=[0, 1], average="macro", zero_division=0))
        for _, group in rows.groupby("subject_id", sort=True)
    ]
    return float(np.mean(values)) if values else float("nan")


def select_hgb_params(
    train: pd.DataFrame,
    feature_columns: list[str],
    *,
    candidates: list[dict] | None = None,
    inner_splits: int = 3,
    sample_cap: int = SAMPLE_CAP,
    seed: int = SEED,
    outer_fold: int = 0,
) -> tuple[dict, list[dict]]:
    """Select parameters using only participant-disjoint inner validation.

    Candidate score is mean macro-F1 across participants in the inner
    validation folds. Every fitted scaler/model sees only its inner training
    groups; the outer participant is absent from this function entirely.
    """
    required = {"subject_id", "trial_id", "window_id", "window_start_s", "label"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"training frame lacks columns: {sorted(missing)}")
    if train[list(required)].isna().any().any():
        raise ValueError("inner training data contain missing grouping or label values")
    if set(train["label"].astype(int).unique()) != {0, 1}:
        raise ValueError("inner training data must include both perceived-fatigue labels")
    groups = train["subject_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    split_count = min(int(inner_splits), len(unique_groups))
    if split_count < 2:
        raise ValueError("inner tuning requires at least two training participants")
    splitter = GroupKFold(n_splits=split_count)
    splits = list(splitter.split(train, train["label"], groups))
    grid = candidates if candidates is not None else default_candidate_grid()
    if not grid:
        raise ValueError("candidate grid must not be empty")

    history = []
    scores_by_candidate = []
    for candidate_index, params in enumerate(grid):
        scores = []
        for inner_index, (train_local, valid_local) in enumerate(splits, start=1):
            inner_train_full = train.iloc[train_local].copy()
            inner_valid = train.iloc[valid_local].copy()
            train_subjects = sorted(inner_train_full["subject_id"].astype(str).unique())
            valid_subjects = sorted(inner_valid["subject_id"].astype(str).unique())
            if set(train_subjects) & set(valid_subjects):
                raise RuntimeError("participant leakage in inner GroupKFold split")
            inner_train = equalized_training_sample(
                inner_train_full,
                max_per_trial_class=sample_cap,
                seed=seed + outer_fold * 100_000 + inner_index,
            )
            if set(inner_train["label"].astype(int).unique()) != {0, 1}:
                raise ValueError("sampled inner training rows lack one perceived-fatigue class")
            estimator = _estimator(params, seed + outer_fold * 10_000 + inner_index)
            estimator.fit(
                inner_train[feature_columns].to_numpy(dtype=float),
                inner_train["label"].to_numpy(dtype=int),
            )
            prediction = np.asarray(
                estimator.predict(inner_valid[feature_columns].to_numpy(dtype=float)), dtype=int
            )
            score = _participant_macro_f1(inner_valid, prediction)
            scores.append(score)
            history.append({
                "outer_fold": int(outer_fold),
                "candidate_index": candidate_index,
                "params": dict(params),
                "inner_fold": inner_index,
                "train_subjects": train_subjects,
                "valid_subjects": valid_subjects,
                "participant_macro_f1": score,
                "training_windows_used": int(len(inner_train)),
                "validation_windows": int(len(inner_valid)),
            })
        scores_by_candidate.append(float(np.mean(scores)))
    best_index = max(range(len(grid)), key=lambda index: (scores_by_candidate[index], -index))
    return dict(grid[best_index]), history


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_delta(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str, *, replicates=50_000, seed=SEED):
    candidate_subject = {
        str(subject): binary_metric_summary(group.y_true, group.y_pred, group.score)[metric]
        for subject, group in candidate.groupby("subject_id", sort=True)
    }
    reference_subject = {
        str(subject): binary_metric_summary(group.y_true, group.y_pred, group.score)[metric]
        for subject, group in reference.groupby("subject_id", sort=True)
    }
    subjects = sorted(set(candidate_subject) & set(reference_subject))
    differences = np.asarray([candidate_subject[s] - reference_subject[s] for s in subjects], dtype=float)
    rng = np.random.default_rng(seed)
    sample = differences[rng.integers(0, len(subjects), size=(replicates, len(subjects)))].mean(axis=1)
    return {
        "metric": metric,
        "comparison": "nested_hgb_minus_fixed_hgb_run_001",
        "participants": len(subjects),
        "mean_paired_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "participants_nested_higher": int(np.sum(differences > 0)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def _load_fixed_hgb_predictions(path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path)
    required = {"architecture", "window_id", "subject_id", "y_true", "y_pred", "score"}
    if not required.issubset(rows.columns):
        raise ValueError("fixed benchmark OOF prediction schema is incomplete")
    fixed = rows[rows["architecture"] == MODEL_NAME].copy()
    if fixed.empty or fixed["window_id"].duplicated().any():
        raise ValueError("expected exactly one fixed HGB OOF prediction per eligible window")
    return fixed.rename(columns={"architecture": "model"})[[
        "model", "window_id", "subject_id", "y_true", "y_pred", "score"
    ]]


def reproduce_fixed_hgb_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    *,
    sample_cap: int = SAMPLE_CAP,
    seed: int = SEED,
) -> pd.DataFrame:
    """Recreate the fixed HGB OOF rows from the recorded original protocol."""
    rows = []
    for fold_number, split in enumerate(leave_one_participant_out(frame), start=1):
        train_full = frame.iloc[split["train_indices"]].copy()
        test = frame.iloc[split["test_indices"]].copy()
        train = equalized_training_sample(
            train_full, max_per_trial_class=sample_cap, seed=seed + fold_number
        )
        estimator = _make_estimator(MODEL_NAME, seed + fold_number * 100 + 4)
        estimator.fit(train[feature_columns].to_numpy(dtype=float), train["label"].to_numpy(dtype=int))
        prediction, score, score_type = _predict(
            estimator, test[feature_columns].to_numpy(dtype=float)
        )
        rows.append(pd.DataFrame({
            "model": MODEL_NAME,
            "window_id": test["window_id"].astype(str).to_numpy(),
            "subject_id": test["subject_id"].astype(str).to_numpy(),
            "y_true": test["label"].astype(int).to_numpy(),
            "y_pred": prediction,
            "score": score,
            "score_type": score_type,
        }))
    result = pd.concat(rows, ignore_index=True)
    if len(result) != len(frame) or result["window_id"].duplicated().any():
        raise RuntimeError("recreated fixed HGB OOF must cover each window exactly once")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def run_nested_benchmark(
    features_path: Path = FEATURES_PATH,
    baseline_oof_path: Path = BASELINE_OOF_PATH,
    private_output: Path = PRIVATE_OUTPUT,
    public_output: Path = PUBLIC_OUTPUT,
    *,
    sample_cap: int = SAMPLE_CAP,
    seed: int = SEED,
    bootstrap_replicates: int = 5000,
    candidates: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    features_path = Path(features_path).resolve()
    baseline_oof_path = Path(baseline_oof_path).resolve()
    private_output = Path(private_output).resolve()
    public_output = Path(public_output).resolve()
    private_output.mkdir(parents=True, exist_ok=True)
    public_output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(features_path)
    if frame["subject_id"].nunique() != 13:
        raise ValueError("expected 13 verified participants in the perceived-fatigue corpus")
    feature_columns = [column for column in frame.columns if column not in METADATA_COLUMNS]
    grid = candidates if candidates is not None else default_candidate_grid()
    outputs = []
    selected_rows = []
    full_search_history = []
    for outer_fold, split in enumerate(leave_one_participant_out(frame), start=1):
        train = frame.iloc[split["train_indices"]].copy()
        test = frame.iloc[split["test_indices"]].copy()
        if set(train["subject_id"].astype(str)) & set(test["subject_id"].astype(str)):
            raise RuntimeError("participant leakage in outer leave-one-participant-out split")
        params, history = select_hgb_params(
            train,
            feature_columns,
            candidates=grid,
            inner_splits=3,
            sample_cap=sample_cap,
            seed=seed,
            outer_fold=outer_fold,
        )
        full_search_history.extend(history)
        train_sample = equalized_training_sample(
            train, max_per_trial_class=sample_cap, seed=seed + outer_fold
        )
        estimator = _estimator(params, seed + outer_fold * 100 + 4)
        estimator.fit(
            train_sample[feature_columns].to_numpy(dtype=float),
            train_sample["label"].to_numpy(dtype=int),
        )
        prediction, score, score_type = _predict(estimator, test[feature_columns].to_numpy(dtype=float))
        fold_rows = pd.DataFrame({
            "model": "nested_hist_gradient_boosting",
            "outer_fold": outer_fold,
            "window_id": test["window_id"].astype(str).to_numpy(),
            "subject_id": test["subject_id"].astype(str).to_numpy(),
            "trial_id": test["trial_id"].astype(str).to_numpy(),
            "y_true": test["label"].astype(int).to_numpy(),
            "y_pred": prediction,
            "score": score,
            "score_type": score_type,
        })
        outputs.append(fold_rows)
        selected_rows.append({
            "outer_fold": outer_fold,
            "test_window_count": int(len(test)),
            "train_window_count_full": int(len(train)),
            "train_window_count_sampled": int(len(train_sample)),
            **{f"param_{key}": value for key, value in params.items()},
        })
        print(
            f"Nested LOPO {outer_fold}/13 held out one participant; "
            f"selected={json.dumps(params, sort_keys=True)}; test_windows={len(test)}",
            flush=True,
        )

    private_selection_path = private_output / "selected_params.csv"
    pd.DataFrame(selected_rows).to_csv(private_selection_path, index=False)
    private_history_path = private_output / "inner_search_history.json"
    private_history_path.write_text(json.dumps(full_search_history, indent=2) + "\n", encoding="utf-8")

    oof = pd.concat(outputs, ignore_index=True)
    if len(oof) != len(frame) or oof["window_id"].duplicated().any():
        raise RuntimeError("nested OOF must contain exactly one prediction per eligible window")
    private_oof_path = private_output / "oof_predictions.parquet"
    oof.to_parquet(private_oof_path, index=False)
    summary = binary_metric_summary(oof["y_true"], oof["y_pred"], oof["score"])
    bootstrap, participant_metrics = _participant_bootstrap(
        oof, replicates=bootstrap_replicates, seed=seed
    )
    metrics = {
        "model": "nested_hist_gradient_boosting",
        "n_windows": int(len(oof)),
        "participants": int(oof["subject_id"].nunique()),
        **{key: value for key, value in summary.items() if key != "n_windows"},
        **bootstrap,
    }
    matrix = confusion_matrix(oof["y_true"], oof["y_pred"], labels=[0, 1])
    totals = matrix.sum(axis=1)
    confusion_rows = [
        {
            "model": "nested_hist_gradient_boosting",
            "actual_label": actual,
            "predicted_label": predicted,
            "count": int(matrix[actual, predicted]),
            "row_normalized": float(matrix[actual, predicted] / totals[actual]) if totals[actual] else float("nan"),
            "n_windows": int(len(oof)),
        }
        for actual in (0, 1)
        for predicted in (0, 1)
    ]
    if baseline_oof_path.is_file():
        resolved_fixed_oof_path = baseline_oof_path
        fixed = _load_fixed_hgb_predictions(baseline_oof_path)
    else:
        resolved_fixed_oof_path = private_output / "fixed_hgb_reference_oof.parquet"
        fixed = reproduce_fixed_hgb_predictions(
            frame,
            feature_columns,
            resolved_fixed_oof_path,
            sample_cap=sample_cap,
            seed=seed,
        )
    merged = oof.merge(fixed, on=["window_id", "subject_id", "y_true"], suffixes=("_nested", "_fixed"), validate="one_to_one")
    if len(merged) != len(oof):
        raise RuntimeError("nested and fixed HGB models must be compared on identical windows")
    fixed_comparison = fixed.rename(columns={
        "model": "fixed_model", "y_true": "y_true_fixed", "y_pred": "y_pred_fixed", "score": "score_fixed"
    })
    paired_rows = [
        _paired_delta(
            oof[["subject_id", "y_true", "y_pred", "score"]],
            fixed_comparison.rename(columns={"y_true_fixed": "y_true", "y_pred_fixed": "y_pred", "score_fixed": "score"}),
            metric,
            replicates=50_000,
            seed=seed + index,
        )
        for index, metric in enumerate(("accuracy", "f1_macro"))
    ]

    public_metrics_path = public_output / "nested_hgb_metrics.csv"
    pd.DataFrame([metrics]).to_csv(public_metrics_path, index=False, float_format="%.10g")
    confusion_path = public_output / "nested_hgb_confusion.csv"
    pd.DataFrame(confusion_rows).to_csv(confusion_path, index=False, float_format="%.10g")
    paired_path = public_output / "nested_hgb_paired_differences.csv"
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False, float_format="%.10g")
    selected_path = public_output / "nested_hgb_selected_params.csv"
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False, float_format="%.10g")
    private_participant_path = private_output / "participant_metrics.csv"
    pd.DataFrame.from_dict(participant_metrics, orient="index").rename_axis("subject_id").reset_index().to_csv(private_participant_path, index=False)

    public_hashes = {
        path.name: _sha256(path)
        for path in (public_metrics_path, confusion_path, paired_path, selected_path)
    }
    manifest = {
        "status": "completed",
        "dataset_id": "10.5281/zenodo.14182446",
        "task_id": "PERCEIVED_FATIGUE_CLASSIFICATION",
        "target": "stable 0=no self-reported fatigue, stable 2=self-reported fatigue; 1 transition excluded",
        "prediction_unit": "reference-aligned 500 ms sEMG windows; one OOF prediction per eligible window",
        "participants": 13,
        "outer_validation": "13-fold leave-one-participant-out",
        "inner_validation": "3-fold GroupKFold by participant; maximize mean participant-macro F1",
        "candidate_count_per_outer_fold": len(grid),
        "candidate_grid": grid,
        "train_sampling": f"even temporal bins; max {sample_cap} windows per training participant x trial x class; test windows kept in full",
        "features_sha256": _sha256(features_path),
        "fixed_oof_sha256": _sha256(resolved_fixed_oof_path),
        "private_oof_sha256": _sha256(private_oof_path),
        "private_search_history_sha256": _sha256(private_history_path),
        "private_participant_metrics_sha256": _sha256(private_participant_path),
        "participant_bootstrap_replicates": bootstrap_replicates,
        "paired_bootstrap_replicates": 50_000,
        "public_aggregate_sha256": public_hashes,
        "scope_limit": "self-reported fatigue only; no synchronized objective force/torque labels or extension/flexion phase labels",
    }
    manifest_path = public_output / "nested_hgb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = _sha256(manifest_path)
    (private_output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return pd.DataFrame([metrics]), manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--baseline-oof", type=Path, default=BASELINE_OOF_PATH)
    parser.add_argument("--private-output", type=Path, default=PRIVATE_OUTPUT)
    parser.add_argument("--public-output", type=Path, default=PUBLIC_OUTPUT)
    parser.add_argument("--sample-cap", type=int, default=SAMPLE_CAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    metrics, manifest = run_nested_benchmark(
        args.features,
        args.baseline_oof,
        args.private_output,
        args.public_output,
        sample_cap=args.sample_cap,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(metrics.to_string(index=False))
    print(f"status={manifest['status']}; public_hashes={manifest['public_aggregate_sha256']}")


if __name__ == "__main__":
    main()
