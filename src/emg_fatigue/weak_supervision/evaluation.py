"""Nested, source-grouped evaluation for weak fatigue-proxy labels.

This module reads only development feature and label tables. Ambiguous labels
remain in the prediction receipt as unscored rows and never enter model fitting.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .contracts import GroupSplit
from .features import FEATURE_VIEWS, LABEL_DEFINING_FEATURES
from .group_cv import FoldPlan, group_folds
from .metrics import compute_metrics
from .models_tabular import (
    MODEL_NAMES,
    build_tabular_model,
    candidate_complexity_key,
    candidate_grid,
)


@dataclass(frozen=True)
class OOFResults:
    """Development-only nested grouped out-of-fold results."""

    predictions: pd.DataFrame
    pooled_metrics: pd.DataFrame
    group_metrics: pd.DataFrame
    bootstrap_intervals: pd.DataFrame
    model_receipts: pd.DataFrame
    fold_manifest: dict[str, object]
    baseline_comparisons: pd.DataFrame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not safe:
        raise ValueError("Artifact name cannot be empty")
    return safe


def _relative_time_fraction(frame: pd.DataFrame) -> pd.Series:
    """Normalize elapsed time within a recording for the shortcut control only."""
    if "start_time_s" not in frame or "recording_id" not in frame:
        raise ValueError("relative_time_only requires start_time_s and recording_id")
    times = pd.to_numeric(frame["start_time_s"], errors="coerce")
    bounds = frame.assign(_time=times).groupby("recording_id", sort=False)["_time"].agg(["min", "max"])
    span = frame["recording_id"].map(bounds["max"] - bounds["min"]).astype(float)
    origin = frame["recording_id"].map(bounds["min"]).astype(float)
    values = (times - origin) / span.replace(0.0, np.nan)
    return values.fillna(0.0).clip(0.0, 1.0)


def _prepare_windows_and_labels(
    windows: pd.DataFrame,
    labels: pd.DataFrame,
    split: GroupSplit,
    feature_views: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    required_window = {"window_key", "group_id", "recording_id", "start_time_s"}
    required_label = {"window_key", "group_id", "class_id"}
    if missing := required_window - set(windows.columns):
        raise ValueError(f"Window table is missing required columns: {sorted(missing)}")
    if missing := required_label - set(labels.columns):
        raise ValueError(f"Label table is missing required columns: {sorted(missing)}")
    if windows["window_key"].duplicated().any() or labels["window_key"].duplicated().any():
        raise ValueError("window_key must be unique in both feature and label tables")
    development_groups = set(str(group) for group in split.development_groups)
    window_groups = set(windows["group_id"].astype(str))
    label_groups = set(labels["group_id"].astype(str))
    if window_groups - development_groups or label_groups - development_groups:
        raise ValueError("Feature or label table contains a non-development source group")
    if window_groups != label_groups:
        raise ValueError("Feature and label tables do not have the same development groups")

    needed_features = {name for names in feature_views.values() for name in names}
    frame = windows.copy()
    if "relative_time_fraction" in needed_features and "relative_time_fraction" not in frame:
        frame["relative_time_fraction"] = _relative_time_fraction(frame)
    if missing := needed_features - set(frame.columns):
        raise ValueError(f"Feature view requests unavailable columns: {sorted(missing)}")
    if "A1_morphology_only" in feature_views:
        leaked = set(feature_views["A1_morphology_only"]) & LABEL_DEFINING_FEATURES
        if leaked:
            raise ValueError(f"A1 contains label-defining spectral features: {sorted(leaked)}")

    kept_labels = [name for name in ("window_key", "group_id", "class_id", "label_confidence", "rule_version", "label_reason") if name in labels]
    merged = frame.merge(
        labels[kept_labels],
        on="window_key",
        how="inner",
        suffixes=("", "_label"),
        validate="one_to_one",
    )
    if len(merged) != len(labels):
        raise ValueError("Some pseudo-label rows have no matching feature window")
    if "group_id_label" in merged and not merged["group_id"].astype(str).eq(merged["group_id_label"].astype(str)).all():
        raise ValueError("Feature and label source-group IDs disagree")
    if not merged["class_id"].isin((-1, 0, 1)).all():
        raise ValueError("Pseudo-label class vocabulary must be {-1, 0, 1}")
    for name, default in (("label_confidence", 0.0), ("rule_version", "unspecified"), ("label_reason", "")):
        if name not in merged:
            merged[name] = default
    merged["class_id"] = merged["class_id"].astype(int)
    merged["group_id"] = merged["group_id"].astype(str)
    merged["recording_id"] = merged["recording_id"].astype(str)
    return merged


def _group_calibration_splits(groups: Sequence[str], y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    group_array = np.asarray(groups, dtype=object)
    unique_groups = sorted(set(group_array.tolist()))
    if len(unique_groups) < 3:
        raise ValueError("Group-calibrated SVM requires at least three training groups")
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=int(seed))
    splits = list(splitter.split(np.zeros((len(y), 1)), y, groups=group_array))
    for train_index, calibration_index in splits:
        if set(np.asarray(y)[train_index].tolist()) != {0, 1}:
            raise ValueError("SVM calibration training split lacks a proxy class")
        if set(np.asarray(y)[calibration_index].tolist()) != {0, 1}:
            raise ValueError("SVM calibration group fold lacks a proxy class")
    return splits


def _fit_estimator(
    model_name: str,
    params: Mapping[str, object],
    x: pd.DataFrame,
    y: np.ndarray,
    groups: Sequence[str],
    seed: int,
) -> tuple[object, list[str], float]:
    estimator = build_tabular_model(model_name, params, seed)
    if model_name == "svm_rbf":
        splits = _group_calibration_splits(groups, y, seed)
        estimator.set_params(model__cv=splits)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(x, y)
    elapsed = time.perf_counter() - start
    warning_messages = [f"{item.category.__name__}: {item.message}" for item in caught]
    return estimator, warning_messages, elapsed


def _positive_probability(estimator: object, x: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(x), dtype=np.float64)
    classes = np.asarray(estimator.classes_)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(x):
        raise ValueError("Estimator returned an invalid probability matrix")
    aligned = np.zeros((len(x), 2), dtype=np.float64)
    for column, class_id in enumerate(classes):
        if int(class_id) in (0, 1):
            aligned[:, int(class_id)] = probabilities[:, column]
    if not np.isfinite(aligned).all() or not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Estimator probabilities are non-finite or not normalized")
    return aligned


def _group_macro_f1(y: np.ndarray, pred: np.ndarray, group_ids: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    groups = np.asarray(group_ids, dtype=object)
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        record = compute_metrics(y[selected], None, pred[selected], np.zeros(int(selected.sum()), dtype=bool))
        if record.macro_f1 is None:
            raise ValueError(f"Validation group {group} does not contain both classes")
        result[str(group)] = float(record.macro_f1)
    return result


def _select_candidate(
    model_name: str,
    candidate_params: Sequence[Mapping[str, object]],
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    inner_folds: Sequence[object],
    *,
    seed: int,
) -> tuple[dict[str, object], float, list[dict[str, object]]]:
    if not candidate_params:
        raise ValueError(f"No candidate hyperparameters for {model_name}")
    candidate_receipts: list[dict[str, object]] = []
    for candidate_index, params in enumerate(candidate_params):
        scores: list[float] = []
        group_scores: list[dict[str, float]] = []
        warnings_seen: list[str] = []
        errors: list[str] = []
        for fold in inner_folds:
            train = frame[frame.group_id.isin(fold.train_group_ids) & frame.class_id.isin((0, 1))]
            valid = frame[frame.group_id.isin(fold.validation_group_ids) & frame.class_id.isin((0, 1))]
            if set(train.class_id.unique()) != {0, 1} or set(valid.class_id.unique()) != {0, 1}:
                errors.append(f"inner fold {fold.fold_id}: missing class in train or validation")
                continue
            fold_seed = int(seed) + int(fold.fold_id) * 101 + candidate_index
            try:
                estimator, found_warnings, _ = _fit_estimator(
                    model_name,
                    params,
                    train[list(feature_names)],
                    train.class_id.to_numpy(dtype=np.int64),
                    train.group_id.tolist(),
                    fold_seed,
                )
                warnings_seen.extend(found_warnings)
                pred = np.asarray(estimator.predict(valid[list(feature_names)]), dtype=np.int64)
                by_group = _group_macro_f1(valid.class_id.to_numpy(dtype=np.int64), pred, valid.group_id.tolist())
                group_scores.append(by_group)
                scores.extend(by_group.values())
            except Exception as exc:  # candidate failures remain visible in the receipt
                errors.append(f"inner fold {fold.fold_id}: {type(exc).__name__}: {str(exc)[:300]}")
        mean_score = float(np.mean(scores)) if scores else float("nan")
        candidate_receipts.append(
            {
                "params": dict(params),
                "mean_validation_group_macro_f1": mean_score if np.isfinite(mean_score) else None,
                "validation_group_count": len(scores),
                "per_inner_fold_group_macro_f1": group_scores,
                "warnings": sorted(set(warnings_seen)),
                "errors": errors,
            }
        )
    valid_candidates = [item for item in candidate_receipts if item["mean_validation_group_macro_f1"] is not None]
    if not valid_candidates:
        raise RuntimeError(f"All candidates failed for {model_name}: {_json(candidate_receipts)}")
    winner = min(
        valid_candidates,
        key=lambda item: (
            -float(item["mean_validation_group_macro_f1"]),
            candidate_complexity_key(model_name, item["params"]),
            _json(item["params"]),
        ),
    )
    return (
        dict(winner["params"]),
        float(winner["mean_validation_group_macro_f1"]),
        candidate_receipts,
    )


def _outer_assignments(plan: FoldPlan) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for fold in plan.outer_folds:
        for group_id in fold.validation_group_ids:
            if group_id in assignments:
                raise ValueError(f"Development group {group_id} appears in multiple outer validation folds")
            assignments[group_id] = fold.fold_id
    if set(assignments) != set(plan.development_group_ids):
        raise ValueError("Outer folds do not assign every development group exactly once")
    return assignments


def _prediction_rows(
    frame: pd.DataFrame,
    *,
    feature_view: str,
    model_name: str,
    fold_by_group: Mapping[str, int],
    probabilities: Mapping[str, tuple[float, float]],
    direct_rule: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        class_id = int(row.class_id)
        if class_id == -1:
            p0 = p1 = confidence = float("nan")
            prediction = -1
            abstain = True
        elif direct_rule:
            p0, p1 = ((1.0, 0.0) if class_id == 0 else (0.0, 1.0))
            confidence = 1.0
            prediction = class_id
            abstain = False
        else:
            p0, p1 = probabilities[str(row.window_key)]
            confidence = max(p0, p1)
            prediction = 0 if p0 >= p1 else 1
            abstain = False
        rows.append(
            {
                "window_key": str(row.window_key),
                "group_id": str(row.group_id),
                "recording_id": str(row.recording_id),
                "feature_view": feature_view,
                "model_name": model_name,
                "fold_id": int(fold_by_group[str(row.group_id)]),
                "true_class_id": class_id,
                "predicted_class_id": int(prediction),
                "probability_class_0": p0,
                "probability_class_1": p1,
                "confidence": confidence,
                "abstain": bool(abstain),
                "label_confidence": float(row.label_confidence),
                "label_reason": str(row.label_reason),
            }
        )
    return rows


def _confusion_from_predictions(
    truth: np.ndarray,
    predictions: np.ndarray,
    group_ids: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    groups = np.asarray(group_ids, dtype=object)
    unique = sorted(set(groups.tolist()))
    matrices: list[np.ndarray] = []
    pred_column = {0: 0, 1: 1, -1: 2}
    for group in unique:
        selected = groups == group
        matrix = np.zeros((2, 3), dtype=np.int64)
        for actual, predicted in zip(truth[selected], predictions[selected], strict=True):
            matrix[int(actual), pred_column[int(predicted)]] += 1
        matrices.append(matrix)
    return np.stack(matrices, axis=0), unique


def _metrics_from_confusions(matrices: np.ndarray) -> dict[str, np.ndarray]:
    total = matrices.sum(axis=(1, 2)).astype(float)
    true0 = matrices[:, 0, :].sum(axis=1).astype(float)
    true1 = matrices[:, 1, :].sum(axis=1).astype(float)
    tn, fp, abstain0 = matrices[:, 0, 0].astype(float), matrices[:, 0, 1].astype(float), matrices[:, 0, 2].astype(float)
    fn, tp, abstain1 = matrices[:, 1, 0].astype(float), matrices[:, 1, 1].astype(float), matrices[:, 1, 2].astype(float)
    accuracy = np.divide(tn + tp, total, out=np.full_like(total, np.nan), where=total > 0)
    recall0 = np.divide(tn, true0, out=np.full_like(true0, np.nan), where=true0 > 0)
    recall1 = np.divide(tp, true1, out=np.full_like(true1, np.nan), where=true1 > 0)
    f1_0_den = 2 * tn + fp + abstain0 + fn
    f1_1_den = 2 * tp + fn + abstain1 + fp
    f1_0 = np.divide(2 * tn, f1_0_den, out=np.full_like(f1_0_den, np.nan), where=f1_0_den > 0)
    f1_1 = np.divide(2 * tp, f1_1_den, out=np.full_like(f1_1_den, np.nan), where=f1_1_den > 0)
    macro = np.nanmean(np.column_stack((f1_0, f1_1)), axis=1)
    balanced = np.nanmean(np.column_stack((recall0, recall1)), axis=1)
    coverage = np.divide(total - abstain0 - abstain1, total, out=np.full_like(total, np.nan), where=total > 0)
    return {"accuracy": accuracy, "macro_f1": macro, "balanced_accuracy": balanced, "coverage": coverage}


def _group_bootstrap(
    truth: np.ndarray,
    predictions: np.ndarray,
    group_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | int | str]]:
    matrices, groups = _confusion_from_predictions(truth, predictions, group_ids)
    if len(groups) < 2:
        return {name: {"lower": float("nan"), "upper": float("nan"), "valid_replicates": 0, "requested_replicates": int(replicates), "method": "source-group percentile bootstrap"} for name in ("accuracy", "macro_f1", "balanced_accuracy", "coverage")}
    rng = np.random.default_rng(int(seed))
    choices = rng.integers(0, len(groups), size=(int(replicates), len(groups)))
    samples = matrices[choices].sum(axis=1)
    values = _metrics_from_confusions(samples)
    result: dict[str, dict[str, float | int | str]] = {}
    for name, metric_values in values.items():
        finite = metric_values[np.isfinite(metric_values)]
        if finite.size:
            lo, hi = np.quantile(finite, [0.025, 0.975])
            result[name] = {
                "lower": float(lo),
                "upper": float(hi),
                "valid_replicates": int(finite.size),
                "requested_replicates": int(replicates),
                "method": "pooled source-group percentile bootstrap; source groups sampled with replacement",
            }
        else:
            result[name] = {
                "lower": float("nan"),
                "upper": float("nan"),
                "valid_replicates": 0,
                "requested_replicates": int(replicates),
                "method": "pooled source-group percentile bootstrap; all replicates undefined",
            }
    return result


def _summarize_predictions(
    predictions: pd.DataFrame,
    *,
    pseudo_label_coverage: float,
    thresholds: Sequence[float],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for (view, model), rows in predictions.groupby(["feature_view", "model_name"], sort=True):
        scored = rows[rows.true_class_id.isin((0, 1))].copy()
        truth = scored.true_class_id.to_numpy(dtype=np.int64)
        raw_prediction = scored.predicted_class_id.to_numpy(dtype=np.int64)
        confidence = scored.confidence.to_numpy(dtype=np.float64)
        groups = scored.group_id.to_numpy(dtype=object)
        if not len(scored):
            continue
        for threshold in thresholds:
            abstain = (~np.isfinite(confidence)) | (confidence < float(threshold))
            predicted = np.where(abstain, -1, raw_prediction).astype(np.int64)
            metric = compute_metrics(truth, np.column_stack((scored.probability_class_0, scored.probability_class_1)), predicted, abstain)
            metric_dict = metric.to_dict()
            pooled_rows.append(
                {
                    "feature_view": view,
                    "model_name": model,
                    "confidence_threshold": float(threshold),
                    "n_evaluated_windows": len(scored),
                    "pseudo_label_coverage": float(pseudo_label_coverage),
                    "evaluation_group_count": int(scored.group_id.nunique()),
                    "groups_with_class_0": int(scored.loc[scored.true_class_id.eq(0), "group_id"].nunique()),
                    "groups_with_class_1": int(scored.loc[scored.true_class_id.eq(1), "group_id"].nunique()),
                    "accuracy": metric.accuracy,
                    "macro_f1": metric.macro_f1,
                    "balanced_accuracy": metric.balanced_accuracy,
                    "coverage": metric.coverage,
                    "selective_accuracy": metric.selective_accuracy,
                    "selective_error": metric.selective_error,
                    "confusion_matrix_json": _json(metric.confusion_matrix),
                    "confusion_true_labels_json": _json(metric.confusion_true_labels),
                    "confusion_predicted_labels_json": _json(metric.confusion_predicted_labels),
                    "per_class_json": _json(metric.per_class),
                    "calibration_json": _json(metric.calibration),
                    "undefined_reasons_json": _json(metric.undefined_reasons),
                }
            )
            for group in sorted(set(groups.tolist())):
                selected = groups == group
                group_metric = compute_metrics(
                    truth[selected],
                    np.column_stack((scored.probability_class_0.to_numpy()[selected], scored.probability_class_1.to_numpy()[selected])),
                    predicted[selected],
                    abstain[selected],
                )
                group_rows.append(
                    {
                        "feature_view": view,
                        "model_name": model,
                        "confidence_threshold": float(threshold),
                        "group_id": group,
                        "n_evaluated_windows": int(selected.sum()),
                        "class_0_support": int(np.sum(truth[selected] == 0)),
                        "class_1_support": int(np.sum(truth[selected] == 1)),
                        "accuracy": group_metric.accuracy,
                        "macro_f1": group_metric.macro_f1,
                        "balanced_accuracy": group_metric.balanced_accuracy,
                        "coverage": group_metric.coverage,
                        "confusion_matrix_json": _json(group_metric.confusion_matrix),
                    }
                )
            intervals = _group_bootstrap(
                truth,
                predicted,
                groups,
                replicates=bootstrap_replicates,
                seed=int(seed) + int(round(float(threshold) * 1000)),
            )
            for metric_name, interval in intervals.items():
                bootstrap_rows.append(
                    {
                        "feature_view": view,
                        "model_name": model,
                        "confidence_threshold": float(threshold),
                        "metric": metric_name,
                        **interval,
                    }
                )
    return pd.DataFrame(pooled_rows), pd.DataFrame(group_rows), pd.DataFrame(bootstrap_rows)


def _baseline_comparisons(group_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = group_metrics[
        group_metrics.feature_view.eq("relative_time_only")
        & group_metrics.model_name.eq("logistic_regression")
        & group_metrics.confidence_threshold.eq(0.0)
    ][["group_id", "accuracy", "macro_f1"]].rename(
        columns={"accuracy": "temporal_baseline_group_accuracy", "macro_f1": "temporal_baseline_group_macro_f1"}
    )
    if baseline.empty:
        return pd.DataFrame()
    comparisons = group_metrics[group_metrics.confidence_threshold.eq(0.0)].merge(baseline, on="group_id", how="inner")
    comparisons["accuracy_delta_vs_temporal_baseline"] = comparisons.accuracy - comparisons.temporal_baseline_group_accuracy
    comparisons["macro_f1_delta_vs_temporal_baseline"] = comparisons.macro_f1 - comparisons.temporal_baseline_group_macro_f1
    result = (
        comparisons.groupby(["feature_view", "model_name"], as_index=False)
        .agg(
            groups_compared=("group_id", "nunique"),
            mean_group_accuracy_delta=("accuracy_delta_vs_temporal_baseline", "mean"),
            median_group_accuracy_delta=("accuracy_delta_vs_temporal_baseline", "median"),
            groups_higher_accuracy=("accuracy_delta_vs_temporal_baseline", lambda values: int((values > 0).sum())),
            mean_group_macro_f1_delta=("macro_f1_delta_vs_temporal_baseline", "mean"),
            median_group_macro_f1_delta=("macro_f1_delta_vs_temporal_baseline", "median"),
            groups_higher_macro_f1=("macro_f1_delta_vs_temporal_baseline", lambda values: int((values > 0).sum())),
        )
    )
    return result


def nested_group_predictions(
    windows: pd.DataFrame,
    labels: pd.DataFrame,
    feature_views: Mapping[str, Sequence[str]],
    models: Mapping[str, Sequence[Mapping[str, object]]],
    split: GroupSplit,
    config: Mapping[str, object],
    *,
    artifact_dir: Path | None = None,
) -> OOFResults:
    """Tune and evaluate model candidates using development-only source groups.

    Fold membership is generated from one row per source group, so window count
    and class count cannot change which source groups enter a fold.
    """
    if not feature_views:
        raise ValueError("At least one feature view is required")
    frame = _prepare_windows_and_labels(windows, labels, split, feature_views)
    dev_groups = tuple(sorted(set(frame.group_id.tolist())))
    if dev_groups != tuple(sorted(set(str(group) for group in split.development_groups))):
        raise ValueError("Feature/label rows must include all and only development groups")
    cv = config.get("cv", {})
    if not isinstance(cv, Mapping):
        raise ValueError("config.cv must be a mapping")
    seed = int(cv.get("seed", 20261113))
    outer_count = int(cv.get("outer_group_folds", 4))
    inner_count = int(cv.get("inner_group_folds", 3))
    plan = group_folds(dev_groups, outer_folds=outer_count, inner_folds=inner_count, seed=seed)
    fold_by_group = _outer_assignments(plan)
    pseudo_label_coverage = float(frame.class_id.isin((0, 1)).mean())
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=False)

    prediction_records: list[dict[str, object]] = []
    receipt_rows: list[dict[str, object]] = []
    requested_models = tuple(models.keys())
    unknown_models = set(requested_models) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unsupported model names: {sorted(unknown_models)}")
    model_settings = config.get("models", {})
    base_model_seed = int(model_settings.get("random_state", seed)) if isinstance(model_settings, Mapping) else seed

    for view_name, feature_names_raw in feature_views.items():
        feature_names = tuple(str(name) for name in feature_names_raw)
        view_frame = frame.copy()
        # Baselines are deterministic and are included in each view for direct comparison.
        prediction_records.extend(
            _prediction_rows(
                view_frame,
                feature_view=view_name,
                model_name="direct_rule_reference",
                fold_by_group=fold_by_group,
                probabilities={},
                direct_rule=True,
            )
        )
        for model_name in requested_models:
            probabilities_by_key: dict[str, tuple[float, float]] = {}
            for outer_fold in plan.outer_folds:
                outer_train = view_frame[
                    view_frame.group_id.isin(outer_fold.train_group_ids)
                    & view_frame.class_id.isin((0, 1))
                ]
                outer_valid = view_frame[view_frame.group_id.isin(outer_fold.validation_group_ids)]
                valid_scored = outer_valid[outer_valid.class_id.isin((0, 1))]
                if set(outer_train.class_id.unique()) != {0, 1} or set(valid_scored.class_id.unique()) != {0, 1}:
                    raise ValueError(f"Outer fold {outer_fold.fold_id} lacks a proxy class for {view_name}/{model_name}")
                selected_params, selection_score, candidate_receipts = _select_candidate(
                    model_name,
                    models[model_name],
                    view_frame,
                    feature_names,
                    plan.inner_folds[outer_fold.fold_id],
                    seed=base_model_seed + outer_fold.fold_id * 1000,
                )
                fold_seed = base_model_seed + outer_fold.fold_id * 1000 + 999
                estimator, fit_warnings, fit_seconds = _fit_estimator(
                    model_name,
                    selected_params,
                    outer_train[list(feature_names)],
                    outer_train.class_id.to_numpy(dtype=np.int64),
                    outer_train.group_id.tolist(),
                    fold_seed,
                )
                outer_probabilities = _positive_probability(estimator, valid_scored[list(feature_names)])
                for key, probability in zip(valid_scored.window_key.astype(str), outer_probabilities, strict=True):
                    probabilities_by_key[key] = (float(probability[0]), float(probability[1]))

                artifact_path_str: str | None = None
                artifact_hash: str | None = None
                if artifact_dir is not None:
                    target = (
                        artifact_dir
                        / _safe_name(view_name)
                        / _safe_name(model_name)
                        / f"outer_{outer_fold.fold_id:02d}.joblib"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    import joblib

                    joblib.dump(estimator, target, compress=3)
                    artifact_path_str = str(target.resolve())
                    artifact_hash = _sha256_file(target)
                receipt_rows.append(
                    {
                        "feature_view": view_name,
                        "model_name": model_name,
                        "outer_fold_id": outer_fold.fold_id,
                        "fit_status": "completed",
                        "selected_params_json": _json(selected_params),
                        "selection_metric": "mean_validation_group_macro_f1",
                        "selection_score": selection_score,
                        "inner_candidate_scores_json": _json(candidate_receipts),
                        "feature_names_json": _json(feature_names),
                        "train_group_ids_json": _json(outer_fold.train_group_ids),
                        "validation_group_ids_json": _json(outer_fold.validation_group_ids),
                        "training_assigned_window_count": len(outer_train),
                        "training_class_0_windows": int(outer_train.class_id.eq(0).sum()),
                        "training_class_1_windows": int(outer_train.class_id.eq(1).sum()),
                        "seed": fold_seed,
                        "fit_seconds": fit_seconds,
                        "fit_warnings_json": _json(fit_warnings),
                        "model_artifact_path": artifact_path_str,
                        "model_artifact_sha256": artifact_hash,
                    }
                )
            missing_keys = set(view_frame.loc[view_frame.class_id.isin((0, 1)), "window_key"].astype(str)) - set(probabilities_by_key)
            if missing_keys:
                raise RuntimeError(f"OOF predictions missing {len(missing_keys)} assigned windows for {view_name}/{model_name}")
            combo_rows = _prediction_rows(
                view_frame,
                feature_view=view_name,
                model_name=model_name,
                fold_by_group=fold_by_group,
                probabilities=probabilities_by_key,
            )
            prediction_records.extend(combo_rows)

    predictions = pd.DataFrame(prediction_records)
    if predictions.duplicated(["feature_view", "model_name", "window_key"]).any():
        raise RuntimeError("OOF result contains duplicate window predictions")
    expected_rows = len(frame)
    sizes = predictions.groupby(["feature_view", "model_name"], sort=False).size()
    if not sizes.eq(expected_rows).all():
        raise RuntimeError("Each model/view must emit one prediction record per development window")
    prediction_groups = predictions[predictions.model_name.ne("direct_rule_reference")]
    if prediction_groups.groupby(["feature_view", "model_name"]).window_key.nunique().ne(expected_rows).any():
        raise RuntimeError("A model/view did not produce exactly-once OOF rows")

    guardrails = config.get("guardrails", {})
    thresholds = sorted(set([0.0] + [float(value) for value in guardrails.get("confidence_thresholds", [])]))
    replicates = int(cv.get("group_bootstrap_replicates", 2000))
    pooled, per_group, bootstrap = _summarize_predictions(
        predictions,
        pseudo_label_coverage=pseudo_label_coverage,
        thresholds=thresholds,
        bootstrap_replicates=replicates,
        seed=seed,
    )
    comparisons = _baseline_comparisons(per_group)
    fold_manifest: dict[str, object] = {
        "seed": plan.seed,
        "development_groups": list(plan.development_group_ids),
        "outer_folds": [
            {
                "fold_id": fold.fold_id,
                "train_group_ids": list(fold.train_group_ids),
                "validation_group_ids": list(fold.validation_group_ids),
                "inner_folds": [
                    {
                        "fold_id": inner.fold_id,
                        "train_group_ids": list(inner.train_group_ids),
                        "validation_group_ids": list(inner.validation_group_ids),
                    }
                    for inner in plan.inner_folds[fold.fold_id]
                ],
            }
            for fold in plan.outer_folds
        ],
    }
    return OOFResults(
        predictions=predictions,
        pooled_metrics=pooled,
        group_metrics=per_group,
        bootstrap_intervals=bootstrap,
        model_receipts=pd.DataFrame(receipt_rows),
        fold_manifest=fold_manifest,
        baseline_comparisons=comparisons,
    )


def _label_rule_tables(private_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    feature_dir = private_dir / "features_labels"
    primary = pd.read_parquet(feature_dir / "pseudo_labels.parquet")
    result = [("spectral_persistence_v1", primary.copy())]
    alternatives = pd.read_parquet(feature_dir / "pseudo_labels_alternatives.parquet")
    for rule_name, table in alternatives.groupby("comparison_rule", sort=True):
        result.append((str(rule_name), table.drop(columns=["comparison_rule"]).copy()))
    return result


def _fold_support(frame: pd.DataFrame, group_ids: Sequence[str]) -> dict[int, tuple[int, int, int]]:
    subset = frame[frame.group_id.isin(group_ids) & frame.class_id.isin((0, 1))]
    support: dict[int, tuple[int, int, int]] = {}
    for class_id in (0, 1):
        selected = subset[subset.class_id.eq(class_id)]
        support[class_id] = (
            int(len(selected)),
            int(selected.group_id.nunique()),
            int(selected.recording_id.nunique()),
        )
    return support


def _audit_label_support(
    label_tables: Sequence[tuple[str, pd.DataFrame]],
    split: GroupSplit,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]], FoldPlan]:
    cv = config.get("cv", {})
    seed = int(cv.get("seed", 20261113))
    plan = group_folds(
        tuple(sorted(str(value) for value in split.development_groups)),
        outer_folds=int(cv.get("outer_group_folds", 4)),
        inner_folds=int(cv.get("inner_group_folds", 3)),
        seed=seed,
    )
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    by_rule: dict[str, dict[str, object]] = {}
    for rule, raw in label_tables:
        frame = raw[raw.group_id.astype(str).isin(split.development_groups)].copy()
        if frame.window_key.duplicated().any():
            raise ValueError(f"Label rule {rule} has duplicate window keys")
        counts = frame.class_id.value_counts().to_dict()
        assigned = frame[frame.class_id.isin((0, 1))]
        trials = assigned.groupby(["group_id", "recording_id"]).class_id.agg(
            has_0=lambda values: bool((values == 0).any()),
            has_1=lambda values: bool((values == 1).any()),
        )
        pass_gate = True
        issues: list[str] = []
        role_frames: list[tuple[str, int, Sequence[str], Sequence[str]]] = []
        for outer in plan.outer_folds:
            role_frames.append(("outer_validation", outer.fold_id, outer.train_group_ids, outer.validation_group_ids))
            role_frames.append(("outer_training", outer.fold_id, outer.train_group_ids, outer.train_group_ids))
            for inner in plan.inner_folds[outer.fold_id]:
                role_frames.append(("inner_validation", (outer.fold_id - 1) * len(plan.inner_folds[outer.fold_id]) + inner.fold_id, inner.train_group_ids, inner.validation_group_ids))
                role_frames.append(("inner_training", (outer.fold_id - 1) * len(plan.inner_folds[outer.fold_id]) + inner.fold_id, inner.train_group_ids, inner.train_group_ids))
        for role, fold_id, train_groups, evaluated_groups in role_frames:
            support = _fold_support(frame, evaluated_groups)
            for class_id in (0, 1):
                windows_n, groups_n, trials_n = support[class_id]
                fold_rows.append(
                    {
                        "rule_version": rule,
                        "partition": role,
                        "fold_id": fold_id,
                        "class_id": class_id,
                        "assigned_windows_descriptive_only": windows_n,
                        "distinct_source_groups": groups_n,
                        "distinct_recordings_trials": trials_n,
                    }
                )
                if windows_n == 0 or groups_n < 2 or trials_n < 2:
                    pass_gate = False
                    issues.append(f"{role} {fold_id} class {class_id}: windows={windows_n}, groups={groups_n}, trials={trials_n}")
        summary = {
            "rule_version": rule,
            "total_windows": int(len(frame)),
            "ambiguous_windows": int(counts.get(-1, 0)),
            "class_0_windows": int(counts.get(0, 0)),
            "class_1_windows": int(counts.get(1, 0)),
            "assigned_label_coverage": float(len(assigned) / len(frame)) if len(frame) else 0.0,
            "groups_with_class_0": int(assigned.loc[assigned.class_id.eq(0), "group_id"].nunique()),
            "groups_with_class_1": int(assigned.loc[assigned.class_id.eq(1), "group_id"].nunique()),
            "trials_with_class_0": int(trials.has_0.sum()),
            "trials_with_class_1": int(trials.has_1.sum()),
            "trials_with_both_classes": int((trials.has_0 & trials.has_1).sum()),
            "fold_coverage_gate_pass": bool(pass_gate),
            "gate_issues_json": _json(issues),
        }
        summary_rows.append(summary)
        by_rule[rule] = summary
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), by_rule, plan


def run_supervised_stage(
    run_dir: Path,
    config_path: Path,
    *,
    label_rule: str,
) -> dict[str, object]:
    """Run model selection/OOF only from the existing private development tables."""
    import tomllib

    run_dir = Path(run_dir)
    private_dir = run_dir / "private"
    feature_dir = private_dir / "features_labels"
    split_doc = json.loads((private_dir / "split_manifest.json").read_text(encoding="utf-8"))
    split = GroupSplit(
        seed=int(split_doc["seed"]),
        train_fraction=float(split_doc["train_fraction"]),
        development_groups=tuple(str(value) for value in split_doc["development_group_ids"]),
        test_groups=tuple(str(value) for value in split_doc["test_group_ids"]),
    )
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    label_tables = _label_rule_tables(private_dir)
    summaries, fold_support, by_rule, _plan = _audit_label_support(label_tables, split, config)
    stage_dir = private_dir / "supervised_models"
    if stage_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing supervised model stage: {stage_dir}")
    stage_dir.mkdir(parents=True)
    summaries.to_csv(stage_dir / "label_support_summary.csv", index=False)
    fold_support.to_csv(stage_dir / "label_support_by_fold.csv", index=False)
    if label_rule not in by_rule:
        raise ValueError(f"Unknown label rule {label_rule}; available: {sorted(by_rule)}")
    chosen_support = by_rule[label_rule]
    if not chosen_support["fold_coverage_gate_pass"]:
        receipt = {
            "stage": "development_nested_grouped_supervised_models",
            "status": "stopped_before_training_insufficient_grouped_class_support",
            "label_rule": label_rule,
            "coverage_gate": chosen_support,
            "internal_test_signal_values_parsed": 0,
            "external_rar_opened": False,
        }
        (stage_dir / "supervised_model_stage_manifest.json").write_text(_json(receipt), encoding="utf-8")
        return receipt

    labels = next(table for rule, table in label_tables if rule == label_rule)
    features = pd.read_parquet(feature_dir / "features.parquet")
    primary_window_s = float(config.get("run", {}).get("primary_window_s", 0.5))
    windows = features.loc[np.isclose(features.duration_s.to_numpy(dtype=float), primary_window_s)].copy()
    if windows.empty:
        raise ValueError("No primary-window features were found")
    if "feature_valid" in windows:
        invalid_assigned = labels.loc[labels.class_id.isin((0, 1)), "window_key"].isin(
            windows.loc[windows.feature_valid.ne(1), "window_key"]
        )
        if invalid_assigned.any():
            raise ValueError("Assigned pseudo-labels include feature-invalid windows")

    views: dict[str, Sequence[str]] = {name: tuple(names) for name, names in FEATURE_VIEWS.items()}
    views["relative_time_only"] = ("relative_time_fraction",)
    model_candidates = candidate_grid(config)
    result = nested_group_predictions(
        windows,
        labels,
        views,
        model_candidates,
        split,
        config,
        artifact_dir=stage_dir / "checkpoints",
    )
    result.predictions.to_parquet(stage_dir / "oof_predictions.parquet", index=False)
    result.pooled_metrics.to_csv(stage_dir / "pooled_metrics.csv", index=False)
    result.group_metrics.to_csv(stage_dir / "group_metrics.csv", index=False)
    result.bootstrap_intervals.to_csv(stage_dir / "group_bootstrap_intervals.csv", index=False)
    result.model_receipts.to_csv(stage_dir / "model_receipts.csv", index=False)
    result.baseline_comparisons.to_csv(stage_dir / "temporal_baseline_comparisons.csv", index=False)
    (stage_dir / "fold_manifest.json").write_text(_json(result.fold_manifest), encoding="utf-8")

    input_paths = [
        feature_dir / "features.parquet",
        feature_dir / "pseudo_labels.parquet",
        feature_dir / "pseudo_labels_alternatives.parquet",
        private_dir / "split_manifest.json",
        Path(config_path),
    ]
    output_paths = [
        stage_dir / "label_support_summary.csv",
        stage_dir / "label_support_by_fold.csv",
        stage_dir / "oof_predictions.parquet",
        stage_dir / "pooled_metrics.csv",
        stage_dir / "group_metrics.csv",
        stage_dir / "group_bootstrap_intervals.csv",
        stage_dir / "model_receipts.csv",
        stage_dir / "temporal_baseline_comparisons.csv",
        stage_dir / "fold_manifest.json",
    ]
    checkpoint_paths = sorted((stage_dir / "checkpoints").rglob("*.joblib"))
    output_paths.extend(checkpoint_paths)
    receipt = {
        "stage": "development_nested_grouped_supervised_models",
        "status": "completed",
        "label_rule": label_rule,
        "label_rule_type": "weak_supervision_proxy_not_independent_fatigue_ground_truth",
        "coverage_gate": chosen_support,
        "cv_seed": int(config["cv"]["seed"]),
        "outer_group_folds": int(config["cv"]["outer_group_folds"]),
        "inner_group_folds": int(config["cv"]["inner_group_folds"]),
        "feature_views": {name: list(names) for name, names in views.items()},
        "model_families": list(model_candidates),
        "oof_predictions_exactly_once_per_development_window": True,
        "ambiguous_labels_used_for_fit": False,
        "overlapping_windows_treated_as_independent_groups": False,
        "independent_unit": "source folder group; participant identity is unverified",
        "internal_test_signal_values_parsed": 0,
        "external_rar_opened": False,
        "inputs_sha256": {str(path): _sha256_file(path) for path in input_paths},
        "outputs_sha256": {str(path): _sha256_file(path) for path in output_paths},
        "checkpoint_count": len(checkpoint_paths),
        "assigned_development_windows": int(labels.class_id.isin((0, 1)).sum()),
        "development_windows_total": int(len(labels)),
        "pseudo_label_coverage": float(labels.class_id.isin((0, 1)).mean()),
        "pooled_metric_confidence_thresholds": sorted(set([0.0] + [float(v) for v in config.get("guardrails", {}).get("confidence_thresholds", [])])),
    }
    (stage_dir / "supervised_model_stage_manifest.json").write_text(_json(receipt), encoding="utf-8")
    return receipt
