"""Nested, source-grouped evaluation for weak fatigue-proxy labels.

This module reads only development feature and label tables. Ambiguous labels
remain in the prediction receipt as unscored rows and never enter model fitting.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold

from .contracts import CandidateThresholdSelection, GroupSplit
from .features import FEATURE_VIEWS, LABEL_DEFINING_FEATURES
from .group_cv import FoldPlan, GroupFold, group_folds, group_folds_with_class_coverage
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


@dataclass(frozen=True)
class NestedThresholdResults:
    """Nested OOF outputs with matched fixed and inner-selected decisions."""

    predictions: pd.DataFrame
    pooled_metrics: pd.DataFrame
    group_metrics: pd.DataFrame
    paired_comparisons: pd.DataFrame
    model_receipts: pd.DataFrame
    fold_manifest: dict[str, object]


@dataclass(frozen=True)
class ThresholdSelection:
    """Threshold selected from inner OOF scores with equal source-group weight."""

    threshold: float
    group_balanced_macro_f1: float
    group_balanced_precision: float
    group_balanced_recall: float
    candidate_count: int
    tie_resolution: str


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


def group_calibration_splits(
    group_ids: Sequence[str],
    y: Sequence[int],
    seed: int,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build group-disjoint calibration folds and reject unsupported splits."""
    group_array = np.asarray(group_ids, dtype=object)
    labels = np.asarray(y, dtype=np.int64)
    if group_array.ndim != 1 or labels.ndim != 1 or labels.shape != group_array.shape or labels.size == 0:
        raise ValueError("group_ids and y must be non-empty, one-dimensional arrays of equal length")
    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least two")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Calibration data must contain both proxy classes")
    if any(group is None or str(group) == "" for group in group_array.tolist()):
        raise ValueError("group_ids must be non-empty")
    unique_groups = sorted(set(group_array.tolist()), key=str)
    if len(unique_groups) < int(n_splits):
        raise ValueError(f"Group calibration requires at least {int(n_splits)} training groups")
    splitter = GroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    splits = list(splitter.split(np.zeros((len(labels), 1)), labels, groups=group_array))
    for train_index, calibration_index in splits:
        if set(labels[train_index].tolist()) != {0, 1}:
            raise ValueError("Calibration training split lacks a proxy class")
        if set(labels[calibration_index].tolist()) != {0, 1}:
            raise ValueError("Calibration group fold lacks a proxy class")
        if set(group_array[train_index].tolist()) & set(group_array[calibration_index].tolist()):
            raise ValueError("Calibration split has source-group overlap")
    return splits


def _group_calibration_splits(groups: Sequence[str], y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Backwards-compatible private wrapper for the original SVM helper."""
    return group_calibration_splits(groups, y, seed, n_splits=3)


def _fit_estimator(
    model_name: str,
    params: Mapping[str, object],
    x: pd.DataFrame,
    y: np.ndarray,
    groups: Sequence[str],
    seed: int,
    calibration_method: str | None = None,
) -> tuple[object, list[str], float]:
    estimator = build_tabular_model(model_name, params, seed)
    method = calibration_method
    if method is None:
        method = "sigmoid" if model_name == "svm_rbf" else "none"
    if method not in {"none", "sigmoid"}:
        raise ValueError(f"Unsupported calibration method: {method}")
    if model_name == "svm_rbf" and method == "none":
        raise ValueError("svm_rbf requires sigmoid calibration to produce probabilities")
    if method == "sigmoid":
        splits = group_calibration_splits(groups, y, seed, n_splits=3)
        estimator = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=splits,
            ensemble=False,
        )
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


def _group_balanced_macro_f1(y: np.ndarray, pred: np.ndarray, group_ids: Sequence[str]) -> float:
    """Compute macro-F1 after giving each source group equal total weight."""
    truth = np.asarray(y, dtype=np.int64)
    predicted = np.asarray(pred, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=object)
    if truth.ndim != 1 or predicted.shape != truth.shape or groups.shape != truth.shape:
        raise ValueError("y, pred, and group_ids must be one-dimensional arrays of equal length")
    if set(np.unique(truth).tolist()) != {0, 1}:
        raise ValueError("Group-balanced macro-F1 requires both classes in the validation partition")
    weights = np.zeros(truth.size, dtype=np.float64)
    for group_id in sorted(set(groups.tolist())):
        selected = groups == group_id
        weights[selected] = 1.0 / int(selected.sum())
    f1_values: list[float] = []
    for class_id in (0, 1):
        true_class = truth == class_id
        predicted_class = predicted == class_id
        tp = float(weights[true_class & predicted_class].sum())
        fp = float(weights[~true_class & predicted_class].sum())
        fn = float(weights[true_class & ~predicted_class].sum())
        denominator = 2.0 * tp + fp + fn
        f1_values.append((2.0 * tp / denominator) if denominator > 0.0 else 0.0)
    return float(np.mean(f1_values))


def select_group_balanced_threshold(
    y_true: Sequence[int],
    probability_class_1: Sequence[float],
    group_ids: Sequence[str],
) -> ThresholdSelection:
    """Select the best threshold without allowing large groups to dominate.

    Candidate values are the observed probabilities plus the fixed 0, 0.5,
    and 1 boundaries. Ties first prefer the value closest to 0.5, then the
    higher threshold. Callers must supply inner out-of-fold probabilities;
    this function cannot distinguish or inspect an outer evaluation fold.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probability_class_1, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=object)
    if truth.ndim != 1 or scores.ndim != 1 or groups.ndim != 1:
        raise ValueError("y_true, probability_class_1, and group_ids must be one-dimensional")
    if truth.size == 0 or scores.shape != truth.shape or groups.shape != truth.shape:
        raise ValueError("y_true, probabilities, and group_ids must be non-empty and have equal length")
    if set(np.unique(truth).tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both classes")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Threshold selection requires finite probabilities in [0, 1]")
    if any(group is None or str(group) == "" for group in groups.tolist()):
        raise ValueError("group_ids must be non-empty")

    candidates = np.unique(np.concatenate((np.asarray([0.0, 0.5, 1.0]), scores)))
    weights = np.zeros(truth.size, dtype=np.float64)
    for group_id in sorted(set(groups.tolist()), key=str):
        selected = groups == group_id
        weights[selected] = 1.0 / int(selected.sum())

    scored: list[tuple[float, float]] = []
    for threshold in candidates:
        predicted = (scores >= threshold).astype(np.int64)
        scored.append((float(threshold), _group_balanced_macro_f1(truth, predicted, groups)))
    best_score = max(score for _, score in scored)
    tied = [threshold for threshold, score in scored if np.isclose(score, best_score, rtol=1e-12, atol=1e-12)]
    if len(tied) == 1:
        selected_threshold = tied[0]
        tie_resolution = "unique_optimum"
    else:
        distances = np.asarray([abs(threshold - 0.5) for threshold in tied], dtype=np.float64)
        closest = [threshold for threshold, distance in zip(tied, distances, strict=True) if np.isclose(distance, distances.min(), rtol=0.0, atol=1e-12)]
        if len(closest) == 1:
            selected_threshold = closest[0]
            tie_resolution = "closest_to_0.5"
        else:
            selected_threshold = max(closest)
            tie_resolution = "higher_threshold"

    predicted = (scores >= selected_threshold).astype(np.int64)
    tp = float(weights[(truth == 1) & (predicted == 1)].sum())
    fp = float(weights[(truth == 0) & (predicted == 1)].sum())
    fn = float(weights[(truth == 1) & (predicted == 0)].sum())
    precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    return ThresholdSelection(
        threshold=float(selected_threshold),
        group_balanced_macro_f1=float(best_score),
        group_balanced_precision=float(precision),
        group_balanced_recall=float(recall),
        candidate_count=int(candidates.size),
        tie_resolution=tie_resolution,
    )


def _group_equal_sample_weights(group_ids: Sequence[str]) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("group_ids must be a non-empty one-dimensional sequence")
    weights = np.zeros(groups.size, dtype=np.float64)
    for group_id in sorted(set(groups.tolist()), key=str):
        selected = groups == group_id
        weights[selected] = 1.0 / int(selected.sum())
    return weights


def _candidate_calibration_methods(model_name: str) -> tuple[str, ...]:
    if model_name == "svm_rbf":
        return ("sigmoid",)
    if model_name == "dummy_majority":
        return ("none",)
    return ("none", "sigmoid")


def _candidate_support(frame: pd.DataFrame) -> dict[str, object]:
    scored = frame[frame.class_id.isin((0, 1))]
    by_class: dict[str, dict[str, int]] = {}
    for class_id in (0, 1):
        selected = scored[scored.class_id.eq(class_id)]
        by_class[str(class_id)] = {
            "assigned_windows": int(len(selected)),
            "source_groups": int(selected.group_id.nunique()),
            "recordings_trials": int(selected.recording_id.nunique()),
        }
    return {
        "assigned_windows": int(len(scored)),
        "source_groups": int(scored.group_id.nunique()),
        "recordings_trials": int(scored.recording_id.nunique()),
        "by_class": by_class,
    }


def _validate_candidate_partition(frame: pd.DataFrame, role: str, fold_id: int) -> None:
    scored = frame[frame.class_id.isin((0, 1))]
    if set(scored.class_id.unique()) != {0, 1}:
        raise ValueError(f"inner fold {fold_id} {role} lacks a proxy class")
    support = _candidate_support(scored)
    for class_id in (0, 1):
        class_support = support["by_class"][str(class_id)]
        if int(class_support["source_groups"]) < 1 or int(class_support["recordings_trials"]) < 1:
            raise ValueError(f"inner fold {fold_id} {role} lacks independent support for class {class_id}")


def _inner_oof_candidate_predictions(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    inner_folds: Sequence[object],
    model_name: str,
    params: Mapping[str, object],
    calibration_method: str,
    seed: int,
) -> pd.DataFrame:
    """Fit one candidate on each inner-training side and score held-out groups once."""
    required = {"window_key", "group_id", "recording_id", "class_id", *feature_names}
    if missing := required - set(frame.columns):
        raise ValueError(f"Candidate frame is missing columns: {sorted(missing)}")
    if frame["window_key"].duplicated().any():
        raise ValueError("Candidate frame window_key values must be unique")
    if frame["group_id"].isna().any() or frame["recording_id"].isna().any():
        raise ValueError("Candidate frame group and recording IDs must be non-missing")
    if not frame.class_id.isin((-1, 0, 1)).all():
        raise ValueError("Candidate frame class_id values must use {-1, 0, 1}")
    if not inner_folds:
        raise ValueError("At least one inner fold is required")

    working = frame.copy()
    working["group_id"] = working.group_id.astype(str)
    working["recording_id"] = working.recording_id.astype(str)
    all_groups = set(working.group_id.tolist())
    validation_counts: dict[str, int] = {}
    for fold in inner_folds:
        train_groups = set(str(value) for value in fold.train_group_ids)
        validation_groups = set(str(value) for value in fold.validation_group_ids)
        if train_groups & validation_groups:
            raise ValueError(f"inner fold {fold.fold_id} has group overlap")
        if not validation_groups or train_groups != all_groups - validation_groups:
            raise ValueError(f"inner fold {fold.fold_id} groups must exactly partition the candidate frame")
        for group_id in validation_groups:
            validation_counts[group_id] = validation_counts.get(group_id, 0) + 1
    if set(validation_counts) != all_groups or any(count != 1 for count in validation_counts.values()):
        raise ValueError("inner validation groups must exactly cover each candidate group once")

    output: list[pd.DataFrame] = []
    for fold in inner_folds:
        train = working[
            working.group_id.isin(fold.train_group_ids) & working.class_id.isin((0, 1))
        ].copy()
        valid = working[
            working.group_id.isin(fold.validation_group_ids) & working.class_id.isin((0, 1))
        ].copy()
        _validate_candidate_partition(train, "training", int(fold.fold_id))
        _validate_candidate_partition(valid, "validation", int(fold.fold_id))
        if set(train.group_id.tolist()) & set(valid.group_id.tolist()):
            raise ValueError(f"inner fold {fold.fold_id} fit and validation source groups overlap")
        fold_seed = int(seed) + int(fold.fold_id) * 101
        estimator, found_warnings, fit_seconds = _fit_estimator(
            model_name,
            params,
            train[list(feature_names)],
            train.class_id.to_numpy(dtype=np.int64),
            train.group_id.tolist(),
            fold_seed,
            calibration_method=calibration_method,
        )
        probabilities = _positive_probability(estimator, valid[list(feature_names)])
        output.append(
            pd.DataFrame(
                {
                    "window_key": valid.window_key.astype(str).to_numpy(),
                    "group_id": valid.group_id.astype(str).to_numpy(),
                    "recording_id": valid.recording_id.astype(str).to_numpy(),
                    "inner_fold_id": int(fold.fold_id),
                    "true_class_id": valid.class_id.to_numpy(dtype=np.int64),
                    "probability_class_0": probabilities[:, 0],
                    "probability_class_1": probabilities[:, 1],
                    "fit_warnings_json": _json(found_warnings),
                    "fit_seconds": float(fit_seconds),
                }
            )
        )

    predictions = pd.concat(output, ignore_index=True)
    if predictions.window_key.duplicated().any():
        raise RuntimeError("An inner candidate emitted more than one OOF score for a window")
    if set(predictions.group_id) != all_groups:
        raise RuntimeError("An inner candidate did not score every assigned source group")
    support = _candidate_support(predictions.rename(columns={"true_class_id": "class_id"}))
    if any(
        int(support["by_class"][str(class_id)][field]) < 2
        for class_id in (0, 1)
        for field in ("source_groups", "recordings_trials")
    ):
        raise ValueError("inner OOF support is insufficient across independent groups or trials")
    return predictions.sort_values(["inner_fold_id", "group_id", "window_key"]).reset_index(drop=True)


def _group_weighted_threshold_stats(
    truth: np.ndarray,
    scores: np.ndarray,
    group_ids: Sequence[str],
    threshold: float,
) -> tuple[float, float]:
    predicted = scores >= float(threshold)
    weights = _group_equal_sample_weights(group_ids)
    tp = float(weights[(truth == 1) & predicted].sum())
    fp = float(weights[(truth == 0) & predicted].sum())
    fn = float(weights[(truth == 1) & ~predicted].sum())
    precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    return float(precision), float(recall)


def _score_inner_candidate(predictions: pd.DataFrame) -> dict[str, object]:
    truth = predictions.true_class_id.to_numpy(dtype=np.int64)
    scores = predictions.probability_class_1.to_numpy(dtype=np.float64)
    groups = predictions.group_id.astype(str).tolist()
    selection = select_group_balanced_threshold(truth, scores, groups)
    fixed_macro = _group_balanced_macro_f1(truth, (scores >= 0.5).astype(np.int64), groups)
    fixed_precision, fixed_recall = _group_weighted_threshold_stats(truth, scores, groups, 0.5)
    optimized_precision, optimized_recall = _group_weighted_threshold_stats(truth, scores, groups, selection.threshold)
    weights = _group_equal_sample_weights(groups)
    precision_curve, recall_curve, _ = precision_recall_curve(truth, scores, sample_weight=weights)
    support = _candidate_support(predictions.rename(columns={"true_class_id": "class_id"}))
    return {
        "optimized_threshold": float(selection.threshold),
        "optimized_threshold_tie_resolution": selection.tie_resolution,
        "threshold_candidate_count": int(selection.candidate_count),
        "group_balanced_macro_f1": float(selection.group_balanced_macro_f1),
        "group_balanced_precision": optimized_precision,
        "group_balanced_recall": optimized_recall,
        "fixed_0_5_group_balanced_macro_f1": float(fixed_macro),
        "fixed_0_5_group_balanced_precision": fixed_precision,
        "fixed_0_5_group_balanced_recall": fixed_recall,
        "group_weighted_pr_auc": float(auc(recall_curve, precision_curve)),
        "group_weighted_average_precision": float(average_precision_score(truth, scores, sample_weight=weights)),
        "group_weighted_roc_auc": float(roc_auc_score(truth, scores, sample_weight=weights)),
        "support": support,
    }


def select_candidate_and_threshold(
    model_name: str,
    candidate_params: Sequence[Mapping[str, object]],
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    inner_folds: Sequence[object],
    seed: int,
) -> CandidateThresholdSelection:
    """Jointly choose model settings and threshold from inner-group OOF only."""
    if not candidate_params:
        raise ValueError(f"No candidate hyperparameters for {model_name}")
    candidate_receipts: list[dict[str, object]] = []
    candidate_predictions: dict[int, pd.DataFrame] = {}
    calibration_methods = _candidate_calibration_methods(model_name)
    candidate_index = 0
    for params in candidate_params:
        for calibration_method in calibration_methods:
            receipt: dict[str, object] = {
                "candidate_index": candidate_index,
                "params": dict(params),
                "calibration_method": calibration_method,
                "fit_status": "unsupported",
                "selection_metric": "group_balanced_macro_f1_after_inner_oof_threshold_selection",
                "errors": [],
            }
            try:
                predictions = _inner_oof_candidate_predictions(
                    frame,
                    feature_names,
                    inner_folds,
                    model_name,
                    params,
                    calibration_method,
                    int(seed) + candidate_index * 10007,
                )
                metrics = _score_inner_candidate(predictions)
                receipt.update(metrics)
                receipt["fit_status"] = "completed"
                receipt["warnings"] = sorted(
                    {
                        warning
                        for value in predictions.fit_warnings_json
                        for warning in json.loads(str(value))
                    }
                )
                receipt["inner_fit_seconds"] = float(predictions.fit_seconds.sum())
                candidate_predictions[candidate_index] = predictions
            except Exception as exc:  # unsupported candidate settings remain in the receipt
                receipt["errors"] = [f"{type(exc).__name__}: {str(exc)[:400]}"]
            candidate_receipts.append(receipt)
            candidate_index += 1

    viable = [item for item in candidate_receipts if item["fit_status"] == "completed"]
    if not viable:
        raise RuntimeError(f"All model/calibration candidates failed for {model_name}: {_json(candidate_receipts)}")
    calibrated_params = {
        _json(item["params"])
        for item in viable
        if item["calibration_method"] == "sigmoid"
    }
    eligible: list[dict[str, object]] = []
    for item in viable:
        has_supported_calibrator = _json(item["params"]) in calibrated_params
        is_eligible = (
            item["calibration_method"] == "sigmoid"
            if has_supported_calibrator
            else item["calibration_method"] == "none"
        )
        item["eligible_for_selection"] = bool(is_eligible)
        if not is_eligible and item["calibration_method"] == "none" and has_supported_calibrator:
            item["selection_exclusion_reason"] = "group-aware sigmoid calibration is supported for these hyperparameters"
        elif not is_eligible:
            item["selection_exclusion_reason"] = "calibration candidate failed support or fit checks"
        if is_eligible:
            eligible.append(item)
    selected = min(
        eligible,
        key=lambda item: (
            -float(item["group_balanced_macro_f1"]),
            candidate_complexity_key(model_name, item["params"]),
            0 if item["calibration_method"] == "sigmoid" else 1,
            _json(item["params"]),
            int(item["candidate_index"]),
        ),
    )
    selected_index = int(selected["candidate_index"])
    for item in candidate_receipts:
        item["selected"] = int(item["candidate_index"]) == selected_index
    selected_predictions = candidate_predictions[selected_index]
    return CandidateThresholdSelection(
        model_name=model_name,
        params=dict(selected["params"]),
        calibration_method=str(selected["calibration_method"]),
        threshold=float(selected["optimized_threshold"]),
        group_balanced_macro_f1=float(selected["group_balanced_macro_f1"]),
        group_balanced_precision=float(selected["group_balanced_precision"]),
        group_balanced_recall=float(selected["group_balanced_recall"]),
        selected_candidate_index=selected_index,
        candidate_receipts=tuple(candidate_receipts),
        inner_oof_predictions=tuple(selected_predictions.to_dict(orient="records")),
    )


def _select_candidate(
    model_name: str,
    candidate_params: Sequence[Mapping[str, object]],
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    inner_folds: Sequence[object],
    *,
    seed: int,
    group_balanced_selection: bool = False,
) -> tuple[dict[str, object], float, list[dict[str, object]]]:
    if not candidate_params:
        raise ValueError(f"No candidate hyperparameters for {model_name}")
    candidate_receipts: list[dict[str, object]] = []
    metric_key = (
        "mean_inner_fold_group_balanced_macro_f1"
        if group_balanced_selection
        else "mean_validation_group_macro_f1"
    )
    for candidate_index, params in enumerate(candidate_params):
        scores: list[float] = []
        group_scores: list[dict[str, object]] = []
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
                if group_balanced_selection:
                    score = _group_balanced_macro_f1(
                        valid.class_id.to_numpy(dtype=np.int64),
                        pred,
                        valid.group_id.tolist(),
                    )
                    group_scores.append(
                        {
                            "fold_id": int(fold.fold_id),
                            "group_balanced_macro_f1": score,
                            "validation_group_count": int(valid.group_id.nunique()),
                        }
                    )
                    scores.append(score)
                else:
                    by_group = _group_macro_f1(valid.class_id.to_numpy(dtype=np.int64), pred, valid.group_id.tolist())
                    group_scores.append(by_group)
                    scores.extend(by_group.values())
            except Exception as exc:  # candidate failures remain visible in the receipt
                errors.append(f"inner fold {fold.fold_id}: {type(exc).__name__}: {str(exc)[:300]}")
        mean_score = float(np.mean(scores)) if scores else float("nan")
        candidate_receipts.append(
            {
                "params": dict(params),
                "mean_validation_group_macro_f1": (
                    mean_score if np.isfinite(mean_score) and not group_balanced_selection else None
                ),
                "mean_inner_fold_group_balanced_macro_f1": (
                    mean_score if np.isfinite(mean_score) and group_balanced_selection else None
                ),
                "selection_metric": metric_key,
                "validation_group_count": (
                    sum(int(item.get("validation_group_count", 0)) for item in group_scores)
                    if group_balanced_selection
                    else len(scores)
                ),
                "per_inner_fold_group_macro_f1": group_scores if not group_balanced_selection else [],
                "per_inner_fold_group_balanced_scores": group_scores if group_balanced_selection else [],
                "warnings": sorted(set(warnings_seen)),
                "errors": errors,
            }
        )
    valid_candidates = [item for item in candidate_receipts if item[metric_key] is not None]
    if not valid_candidates:
        raise RuntimeError(f"All candidates failed for {model_name}: {_json(candidate_receipts)}")
    winner = min(
        valid_candidates,
        key=lambda item: (
            -float(item[metric_key]),
            candidate_complexity_key(model_name, item["params"]),
            _json(item["params"]),
        ),
    )
    return (
        dict(winner["params"]),
        float(winner[metric_key]),
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


def _paired_group_bootstrap_vs_temporal(
    predictions: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Compare morphology models with the time-only logistic baseline by group.

    Each source group's confusion matrix is normalized to unit mass before the
    paired bootstrap, so replicating overlapping windows inside one group does
    not give that group more inferential weight.
    """
    scored = predictions[predictions.true_class_id.isin((0, 1))].copy()
    baseline = scored[
        scored.feature_view.eq("relative_time_only")
        & scored.model_name.eq("logistic_regression")
    ][["window_key", "group_id", "true_class_id", "predicted_class_id"]].rename(
        columns={
            "group_id": "baseline_group_id",
            "true_class_id": "baseline_true_class_id",
            "predicted_class_id": "baseline_predicted_class_id",
        }
    )
    if baseline.empty:
        return pd.DataFrame()
    morphology = scored[scored.feature_view.eq("A1_morphology_only")]
    models = sorted(set(morphology.model_name.astype(str)) - {"direct_rule_reference"})
    group_ids = tuple(sorted(baseline.baseline_group_id.astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    requested = int(replicates)
    if requested < 1:
        raise ValueError("paired group bootstrap replicates must be positive")

    def metrics(confusion: np.ndarray) -> tuple[float, float]:
        tn, fp, fn, tp = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
        support0 = tn + fp
        support1 = fn + tp
        total = float(confusion.sum())
        if total <= 0.0:
            return float("nan"), float("nan")
        accuracy = (tn + tp) / total
        if support0 <= 0.0 or support1 <= 0.0:
            return float(accuracy), float("nan")
        f1_0_denominator = 2.0 * tn + fp + fn
        f1_1_denominator = 2.0 * tp + fp + fn
        f1_0 = 2.0 * tn / f1_0_denominator if f1_0_denominator > 0.0 else 0.0
        f1_1 = 2.0 * tp / f1_1_denominator if f1_1_denominator > 0.0 else 0.0
        return float(accuracy), float((f1_0 + f1_1) / 2.0)

    output_rows: list[dict[str, object]] = []
    bootstrap_groups = rng.integers(0, len(group_ids), size=(requested, len(group_ids)))
    for model_name in models:
        model_rows = morphology[morphology.model_name.eq(model_name)][
            ["window_key", "group_id", "true_class_id", "predicted_class_id"]
        ]
        paired = model_rows.merge(
            baseline,
            on="window_key",
            how="inner",
            validate="one_to_one",
        )
        if (
            not paired.group_id.astype(str).eq(paired.baseline_group_id.astype(str)).all()
            or not paired.true_class_id.eq(paired.baseline_true_class_id).all()
        ):
            raise ValueError("Morphology and temporal baseline predictions are not paired on identical groups/labels")
        paired_confusions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        group_accuracy_deltas: list[float] = []
        mixed_group_macro_deltas: list[float] = []
        mixed_group_count = 0
        for group_id in group_ids:
            rows = paired[paired.group_id.astype(str).eq(group_id)]
            matrix = np.zeros((2, 2), dtype=np.float64)
            baseline_matrix = np.zeros((2, 2), dtype=np.float64)
            np.add.at(
                matrix,
                (rows.true_class_id.to_numpy(dtype=int), rows.predicted_class_id.to_numpy(dtype=int)),
                1.0,
            )
            np.add.at(
                baseline_matrix,
                (rows.true_class_id.to_numpy(dtype=int), rows.baseline_predicted_class_id.to_numpy(dtype=int)),
                1.0,
            )
            paired_confusions[group_id] = (matrix / matrix.sum(), baseline_matrix / baseline_matrix.sum())
            model_accuracy, model_macro = metrics(matrix)
            baseline_accuracy, baseline_macro = metrics(baseline_matrix)
            group_accuracy_deltas.append(model_accuracy - baseline_accuracy)
            if np.unique(rows.true_class_id.to_numpy(dtype=int)).size == 2:
                mixed_group_count += 1
                mixed_group_macro_deltas.append(model_macro - baseline_macro)

        model_point_confusion = np.mean([paired_confusions[group][0] for group in group_ids], axis=0)
        baseline_point_confusion = np.mean([paired_confusions[group][1] for group in group_ids], axis=0)
        model_accuracy, model_macro = metrics(model_point_confusion)
        baseline_accuracy, baseline_macro = metrics(baseline_point_confusion)

        model_bootstrap = np.mean(
            np.asarray([[paired_confusions[group_ids[index]][0] for index in draw] for draw in bootstrap_groups]),
            axis=1,
        )
        baseline_bootstrap = np.mean(
            np.asarray([[paired_confusions[group_ids[index]][1] for index in draw] for draw in bootstrap_groups]),
            axis=1,
        )
        model_values = [metrics(matrix) for matrix in model_bootstrap]
        baseline_values = [metrics(matrix) for matrix in baseline_bootstrap]
        delta_accuracy = np.asarray([a[0] - b[0] for a, b in zip(model_values, baseline_values, strict=True)])
        delta_macro = np.asarray([a[1] - b[1] for a, b in zip(model_values, baseline_values, strict=True)])
        valid_accuracy = np.isfinite(delta_accuracy)
        valid_macro = np.isfinite(delta_macro)
        output_rows.append(
            {
                "feature_view": "A1_morphology_only",
                "model_name": model_name,
                "comparison_model": "relative_time_only/logistic_regression",
                "source_group_count": len(group_ids),
                "groups_with_both_classes": mixed_group_count,
                "mean_group_accuracy_delta": float(np.mean(group_accuracy_deltas)),
                "mean_group_macro_f1_delta_on_both_class_groups": float(np.mean(mixed_group_macro_deltas)) if mixed_group_macro_deltas else None,
                "group_balanced_accuracy": model_accuracy,
                "time_baseline_group_balanced_accuracy": baseline_accuracy,
                "group_balanced_accuracy_delta": model_accuracy - baseline_accuracy,
                "group_balanced_macro_f1": model_macro,
                "time_baseline_group_balanced_macro_f1": baseline_macro,
                "group_balanced_macro_f1_delta": model_macro - baseline_macro,
                "bootstrap_method": "paired source-group bootstrap; each group confusion matrix normalized to unit mass",
                "bootstrap_seed": int(seed),
                "bootstrap_requested_replicates": requested,
                "accuracy_delta_ci_low": float(np.quantile(delta_accuracy[valid_accuracy], 0.025)) if valid_accuracy.any() else None,
                "accuracy_delta_ci_high": float(np.quantile(delta_accuracy[valid_accuracy], 0.975)) if valid_accuracy.any() else None,
                "accuracy_delta_valid_replicates": int(valid_accuracy.sum()),
                "macro_f1_delta_ci_low": float(np.quantile(delta_macro[valid_macro], 0.025)) if valid_macro.any() else None,
                "macro_f1_delta_ci_high": float(np.quantile(delta_macro[valid_macro], 0.975)) if valid_macro.any() else None,
                "macro_f1_delta_valid_replicates": int(valid_macro.sum()),
            }
        )
    return pd.DataFrame(output_rows)


def nested_group_predictions(
    windows: pd.DataFrame,
    labels: pd.DataFrame,
    feature_views: Mapping[str, Sequence[str]],
    models: Mapping[str, Sequence[Mapping[str, object]]],
    split: GroupSplit,
    config: Mapping[str, object],
    *,
    artifact_dir: Path | None = None,
    fold_plan: FoldPlan | None = None,
    group_balanced_selection: bool = False,
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
    plan = fold_plan or group_folds(dev_groups, outer_folds=outer_count, inner_folds=inner_count, seed=seed)
    if tuple(sorted(plan.development_group_ids)) != dev_groups:
        raise ValueError("fold_plan must contain all and only development source groups")
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
                    group_balanced_selection=group_balanced_selection,
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
                        "selection_metric": (
                            "mean_inner_fold_group_balanced_macro_f1"
                            if group_balanced_selection
                            else "mean_validation_group_macro_f1"
                        ),
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


def _threshold_prediction_rows(
    frame: pd.DataFrame,
    *,
    feature_view: str,
    model_name: str,
    fold_by_group: Mapping[str, int],
    probabilities: Mapping[str, tuple[float, float]],
    threshold_by_fold: Mapping[int, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        class_id = int(row.class_id)
        fold_id = int(fold_by_group[str(row.group_id)])
        if class_id == -1:
            p0 = p1 = confidence = float("nan")
            decisions = (("fixed_0_5", 0.5, -1), ("inner_oof_selected", float(threshold_by_fold[fold_id]), -1))
        else:
            p0, p1 = probabilities[str(row.window_key)]
            confidence = max(p0, p1)
            selected_threshold = float(threshold_by_fold[fold_id])
            decisions = (
                ("fixed_0_5", 0.5, int(p1 >= 0.5)),
                ("inner_oof_selected", selected_threshold, int(p1 >= selected_threshold)),
            )
        for strategy, threshold, prediction in decisions:
            rows.append(
                {
                    "window_key": str(row.window_key),
                    "group_id": str(row.group_id),
                    "recording_id": str(row.recording_id),
                    "feature_view": feature_view,
                    "model_name": model_name,
                    "threshold_strategy": strategy,
                    "decision_threshold": float(threshold),
                    "fold_id": fold_id,
                    "true_class_id": class_id,
                    "predicted_class_id": prediction,
                    "probability_class_0": p0,
                    "probability_class_1": p1,
                    "confidence": confidence,
                    "abstain": bool(class_id == -1),
                    "label_confidence": float(row.label_confidence),
                    "label_reason": str(row.label_reason),
                }
            )
    return rows


def _weighted_confusion_metrics(confusion: np.ndarray) -> dict[str, float | None]:
    tn, fp = float(confusion[0, 0]), float(confusion[0, 1])
    fn, tp = float(confusion[1, 0]), float(confusion[1, 1])
    support0, support1 = tn + fp, fn + tp
    total = support0 + support1
    if total <= 0.0 or support0 <= 0.0 or support1 <= 0.0:
        return {
            "accuracy": None,
            "sensitivity": None,
            "specificity": None,
            "macro_f1": None,
            "balanced_accuracy": None,
            "mcc": None,
            "cohen_kappa": None,
        }
    recall0 = tn / support0
    recall1 = tp / support1
    precision0 = tn / (tn + fn) if tn + fn else 0.0
    precision1 = tp / (tp + fp) if tp + fp else 0.0
    f1_0_denominator = 2.0 * tn + fp + fn
    f1_1_denominator = 2.0 * tp + fp + fn
    f1_0 = 2.0 * tn / f1_0_denominator if f1_0_denominator else 0.0
    f1_1 = 2.0 * tp / f1_1_denominator if f1_1_denominator else 0.0
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / np.sqrt(denominator)) if denominator > 0.0 else 0.0
    observed = (tn + tp) / total
    expected = (support0 * (tn + fn) + support1 * (fp + tp)) / (total * total)
    kappa = (observed - expected) / (1.0 - expected) if not np.isclose(expected, 1.0) else 0.0
    return {
        "accuracy": float(observed),
        "sensitivity": float(recall1),
        "specificity": float(recall0),
        "macro_f1": float((f1_0 + f1_1) / 2.0),
        "balanced_accuracy": float((recall0 + recall1) / 2.0),
        "mcc": float(mcc),
        "cohen_kappa": float(kappa),
    }


def _group_confusion_matrices(
    frame: pd.DataFrame,
    *,
    truth_column: str = "true_class_id",
    prediction_column: str = "predicted_class_id",
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for group_id, rows in frame.groupby("group_id", sort=True):
        matrix = np.zeros((2, 2), dtype=np.float64)
        np.add.at(
            matrix,
            (
                rows[truth_column].to_numpy(dtype=np.int64),
                rows[prediction_column].to_numpy(dtype=np.int64),
            ),
            1.0,
        )
        matrices[str(group_id)] = matrix / float(matrix.sum())
    return matrices


def _group_balanced_probability_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    predicted: np.ndarray,
    group_ids: Sequence[str],
) -> dict[str, object]:
    weights = _group_equal_sample_weights(group_ids)
    confusion = np.zeros((2, 2), dtype=np.float64)
    np.add.at(confusion, (truth, predicted), weights)
    result: dict[str, object] = _weighted_confusion_metrics(confusion)
    result["confusion_matrix"] = confusion.tolist()
    if all(np.any(truth == class_id) for class_id in (0, 1)):
        result["pr_auc_average_precision"] = float(average_precision_score(truth, scores, sample_weight=weights))
        result["roc_auc"] = float(roc_auc_score(truth, scores, sample_weight=weights))
    else:
        result["pr_auc_average_precision"] = None
        result["roc_auc"] = None
    result["brier_score"] = float(np.average((scores - truth) ** 2, weights=weights))
    return result


def _nested_threshold_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for (view, model, strategy), rows in predictions.groupby(
        ["feature_view", "model_name", "threshold_strategy"], sort=True
    ):
        scored = rows[rows.true_class_id.isin((0, 1))].copy()
        truth = scored.true_class_id.to_numpy(dtype=np.int64)
        predicted = scored.predicted_class_id.to_numpy(dtype=np.int64)
        probabilities = np.column_stack(
            (scored.probability_class_0.to_numpy(dtype=np.float64), scored.probability_class_1.to_numpy(dtype=np.float64))
        )
        groups = scored.group_id.astype(str).to_numpy(dtype=object)
        metric = compute_metrics(truth, probabilities, predicted, np.zeros(len(scored), dtype=bool))
        group_balanced = _group_balanced_probability_metrics(
            truth,
            probabilities[:, 1],
            predicted,
            groups,
        )
        pooled_rows.append(
            {
                "feature_view": view,
                "model_name": model,
                "threshold_strategy": strategy,
                "decision_thresholds_json": _json(
                    sorted({float(value) for value in scored.decision_threshold.to_numpy(dtype=float)})
                ),
                "n_evaluated_windows": len(scored),
                "evaluation_group_count": int(scored.group_id.nunique()),
                "groups_with_class_0": int(scored.loc[scored.true_class_id.eq(0), "group_id"].nunique()),
                "groups_with_class_1": int(scored.loc[scored.true_class_id.eq(1), "group_id"].nunique()),
                "accuracy": metric.accuracy,
                "precision_class_1": metric.per_class["1"]["precision"],
                "sensitivity": metric.sensitivity,
                "specificity": metric.specificity,
                "macro_f1": metric.macro_f1,
                "balanced_accuracy": metric.balanced_accuracy,
                "mcc": metric.matthews_correlation_coefficient,
                "cohen_kappa": metric.cohen_kappa,
                "pr_auc_average_precision": metric.pr_auc_average_precision,
                "roc_auc": metric.roc_auc,
                "brier_score": metric.calibration["brier_score"],
                "log_loss": metric.log_loss,
                "group_balanced_accuracy": group_balanced["accuracy"],
                "group_balanced_sensitivity": group_balanced["sensitivity"],
                "group_balanced_specificity": group_balanced["specificity"],
                "group_balanced_macro_f1": group_balanced["macro_f1"],
                "group_balanced_balanced_accuracy": group_balanced["balanced_accuracy"],
                "group_balanced_mcc": group_balanced["mcc"],
                "group_balanced_cohen_kappa": group_balanced["cohen_kappa"],
                "group_weighted_pr_auc_average_precision": group_balanced["pr_auc_average_precision"],
                "group_weighted_roc_auc": group_balanced["roc_auc"],
                "group_weighted_brier_score": group_balanced["brier_score"],
                "expected_calibration_error_10_bins": metric.calibration["expected_calibration_error_10_bins"],
                "confusion_matrix_json": _json(metric.confusion_matrix),
                "normalized_confusion_matrix_json": _json(metric.confusion_matrix_normalized_by_true),
                "group_balanced_confusion_matrix_json": _json(group_balanced["confusion_matrix"]),
                "per_class_json": _json(metric.per_class),
                "undefined_reasons_json": _json(metric.undefined_reasons),
            }
        )
        for group_id, group in scored.groupby("group_id", sort=True):
            group_metric = compute_metrics(
                group.true_class_id.to_numpy(dtype=np.int64),
                np.column_stack(
                    (group.probability_class_0.to_numpy(dtype=float), group.probability_class_1.to_numpy(dtype=float))
                ),
                group.predicted_class_id.to_numpy(dtype=np.int64),
                np.zeros(len(group), dtype=bool),
            )
            group_rows.append(
                {
                    "feature_view": view,
                    "model_name": model,
                    "threshold_strategy": strategy,
                    "group_id": str(group_id),
                    "n_evaluated_windows": int(len(group)),
                    "class_0_support": int(group.true_class_id.eq(0).sum()),
                    "class_1_support": int(group.true_class_id.eq(1).sum()),
                    "accuracy": group_metric.accuracy,
                    "precision_class_1": group_metric.per_class["1"]["precision"],
                    "sensitivity": group_metric.sensitivity,
                    "specificity": group_metric.specificity,
                    "macro_f1": group_metric.macro_f1,
                    "balanced_accuracy": group_metric.balanced_accuracy,
                    "mcc": group_metric.matthews_correlation_coefficient,
                    "cohen_kappa": group_metric.cohen_kappa,
                    "pr_auc_average_precision": group_metric.pr_auc_average_precision,
                    "roc_auc": group_metric.roc_auc,
                    "brier_score": group_metric.calibration["brier_score"],
                    "expected_calibration_error_10_bins": group_metric.calibration["expected_calibration_error_10_bins"],
                    "log_loss": group_metric.log_loss,
                    "confusion_matrix_json": _json(group_metric.confusion_matrix),
                    "normalized_confusion_matrix_json": _json(group_metric.confusion_matrix_normalized_by_true),
                    "undefined_reasons_json": _json(group_metric.undefined_reasons),
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(group_rows)


def _paired_group_metric_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    comparison: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    key_columns = ["window_key"]
    fields = ["group_id", "true_class_id", "predicted_class_id"]
    paired = left[key_columns + fields].merge(
        right[key_columns + fields],
        on="window_key",
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(f"{comparison} did not use the same scored windows")
    for field in ("group_id", "true_class_id"):
        if not paired[f"{field}_left"].astype(str).eq(paired[f"{field}_right"].astype(str)).all():
            raise ValueError(f"{comparison} is not paired on identical groups and labels")
    paired_rows_left = paired[["window_key", "group_id_left", "true_class_id_left", "predicted_class_id_left"]].rename(
        columns={"group_id_left": "group_id", "true_class_id_left": "true_class_id", "predicted_class_id_left": "predicted_class_id"}
    )
    paired_rows_right = paired[["window_key", "group_id_right", "true_class_id_right", "predicted_class_id_right"]].rename(
        columns={"group_id_right": "group_id", "true_class_id_right": "true_class_id", "predicted_class_id_right": "predicted_class_id"}
    )
    left_matrices = _group_confusion_matrices(paired_rows_left)
    right_matrices = _group_confusion_matrices(paired_rows_right)
    groups = tuple(sorted(set(left_matrices) & set(right_matrices)))
    if len(groups) < 2:
        raise ValueError(f"{comparison} requires at least two paired source groups")
    if set(left_matrices) != set(right_matrices):
        raise ValueError(f"{comparison} source-group support differs between paired results")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(groups), size=(int(replicates), len(groups)))
    metric_names = ("accuracy", "macro_f1", "balanced_accuracy", "sensitivity", "specificity", "mcc", "cohen_kappa")

    def aggregate(matrices: Mapping[str, np.ndarray], selected: Sequence[int]) -> dict[str, float | None]:
        mean_matrix = np.mean([matrices[groups[index]] for index in selected], axis=0)
        return _weighted_confusion_metrics(mean_matrix)

    left_point = aggregate(left_matrices, tuple(range(len(groups))))
    right_point = aggregate(right_matrices, tuple(range(len(groups))))
    output: dict[str, object] = {
        "comparison": comparison,
        "source_group_count": len(groups),
        "paired_window_count": len(paired),
        "bootstrap_method": "paired source-group percentile bootstrap with each group confusion normalized to unit mass",
        "bootstrap_seed": int(seed),
        "bootstrap_requested_replicates": int(replicates),
    }
    for metric_name in metric_names:
        delta = None
        if left_point[metric_name] is not None and right_point[metric_name] is not None:
            delta = float(left_point[metric_name] - right_point[metric_name])
        bootstrap_deltas: list[float] = []
        for draw in draws:
            left_metrics = aggregate(left_matrices, draw)
            right_metrics = aggregate(right_matrices, draw)
            lvalue, rvalue = left_metrics[metric_name], right_metrics[metric_name]
            if lvalue is not None and rvalue is not None:
                bootstrap_deltas.append(float(lvalue - rvalue))
        output[f"{metric_name}_left"] = left_point[metric_name]
        output[f"{metric_name}_right"] = right_point[metric_name]
        output[f"{metric_name}_delta"] = delta
        output[f"{metric_name}_delta_ci_low"] = float(np.quantile(bootstrap_deltas, 0.025)) if bootstrap_deltas else None
        output[f"{metric_name}_delta_ci_high"] = float(np.quantile(bootstrap_deltas, 0.975)) if bootstrap_deltas else None
        output[f"{metric_name}_delta_valid_replicates"] = len(bootstrap_deltas)
    return output


def _paired_nested_threshold_comparisons(
    predictions: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    scored = predictions[predictions.true_class_id.isin((0, 1))].copy()
    rows: list[dict[str, object]] = []
    for (view, model), model_rows in scored.groupby(["feature_view", "model_name"], sort=True):
        fixed = model_rows[model_rows.threshold_strategy.eq("fixed_0_5")]
        tuned = model_rows[model_rows.threshold_strategy.eq("inner_oof_selected")]
        comparison = _paired_group_metric_delta(
            tuned,
            fixed,
            comparison="inner_oof_selected_minus_fixed_0_5",
            replicates=replicates,
            seed=seed,
        )
        rows.append({"feature_view": view, "model_name": model, "threshold_strategy": "inner_oof_selected", **comparison})

    for strategy in ("fixed_0_5", "inner_oof_selected"):
        temporal = scored[
            scored.feature_view.eq("relative_time_only")
            & scored.model_name.eq("logistic_regression")
            & scored.threshold_strategy.eq(strategy)
        ]
        if temporal.empty:
            continue
        for model in sorted(set(scored.loc[scored.feature_view.eq("A1_morphology_only"), "model_name"])):
            morphology = scored[
                scored.feature_view.eq("A1_morphology_only")
                & scored.model_name.eq(model)
                & scored.threshold_strategy.eq(strategy)
            ]
            comparison = _paired_group_metric_delta(
                morphology,
                temporal,
                comparison="A1_morphology_minus_relative_time_logistic",
                replicates=replicates,
                seed=seed,
            )
            rows.append(
                {
                    "feature_view": "A1_morphology_only",
                    "model_name": model,
                    "threshold_strategy": strategy,
                    **comparison,
                }
            )
    return pd.DataFrame(rows)


def _serialized_fold_plan(plan: FoldPlan) -> dict[str, object]:
    return {
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


def _fold_plan_from_manifest(path: Path) -> FoldPlan:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    development_groups = tuple(str(value) for value in document["development_groups"])
    outer_folds: list[GroupFold] = []
    inner_folds: dict[int, tuple[GroupFold, ...]] = {}
    for outer in document["outer_folds"]:
        fold_id = int(outer["fold_id"])
        outer_folds.append(
            GroupFold(
                fold_id=fold_id,
                train_group_ids=tuple(str(value) for value in outer["train_group_ids"]),
                validation_group_ids=tuple(str(value) for value in outer["validation_group_ids"]),
            )
        )
        inner_folds[fold_id] = tuple(
            GroupFold(
                fold_id=int(inner["fold_id"]),
                train_group_ids=tuple(str(value) for value in inner["train_group_ids"]),
                validation_group_ids=tuple(str(value) for value in inner["validation_group_ids"]),
            )
            for inner in outer["inner_folds"]
        )
    plan = FoldPlan(
        seed=int(document["seed"]),
        development_group_ids=development_groups,
        outer_folds=tuple(outer_folds),
        inner_folds=inner_folds,
    )
    _outer_assignments(plan)
    if len(plan.outer_folds) != 4 or any(len(plan.inner_folds[fold.fold_id]) != 3 for fold in plan.outer_folds):
        raise ValueError("Historical fold manifest must contain exactly 4 outer x 3 inner folds")
    for fold in plan.outer_folds:
        for inner in plan.inner_folds[fold.fold_id]:
            if set(inner.train_group_ids) & set(inner.validation_group_ids):
                raise ValueError(f"Historical inner fold {fold.fold_id}/{inner.fold_id} has group overlap")
            if set(inner.train_group_ids) | set(inner.validation_group_ids) != set(fold.train_group_ids):
                raise ValueError(f"Historical inner fold {fold.fold_id}/{inner.fold_id} does not partition outer training groups")
    return plan


def nested_group_threshold_predictions(
    windows: pd.DataFrame,
    labels: pd.DataFrame,
    feature_views: Mapping[str, Sequence[str]],
    models_by_view: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    split: GroupSplit,
    config: Mapping[str, object],
    *,
    fold_plan: FoldPlan,
    artifact_dir: Path,
) -> NestedThresholdResults:
    """Fit the revised nested selector and evaluate both thresholds on identical OOF rows."""
    if not feature_views or set(feature_views) != set(models_by_view):
        raise ValueError("Each feature view must have an explicit candidate model mapping")
    frame = _prepare_windows_and_labels(windows, labels, split, feature_views)
    dev_groups = tuple(sorted(set(frame.group_id.astype(str))))
    if dev_groups != tuple(sorted(str(value) for value in split.development_groups)):
        raise ValueError("Feature/label rows must include all and only development source groups")
    if dev_groups != tuple(sorted(fold_plan.development_group_ids)):
        raise ValueError("Historical fold plan must contain all and only development source groups")
    if len(fold_plan.outer_folds) != 4 or any(len(fold_plan.inner_folds[fold.fold_id]) != 3 for fold in fold_plan.outer_folds):
        raise ValueError("Revised nested selection requires the supported historical 4x3 fold plan")
    if Path(artifact_dir).exists():
        raise FileExistsError(f"Refusing to overwrite existing model artifacts: {artifact_dir}")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    fold_by_group = _outer_assignments(fold_plan)
    settings = config.get("models", {})
    cv = config.get("cv", {})
    if not isinstance(settings, Mapping) or not isinstance(cv, Mapping):
        raise ValueError("config.models and config.cv must be mappings")
    base_seed = int(settings.get("random_state", cv.get("seed", fold_plan.seed)))
    prediction_records: list[dict[str, object]] = []
    receipt_rows: list[dict[str, object]] = []

    for view_name, feature_names_raw in feature_views.items():
        feature_names = tuple(str(value) for value in feature_names_raw)
        candidates_for_view = models_by_view[view_name]
        unknown_models = set(candidates_for_view) - set(MODEL_NAMES)
        if unknown_models:
            raise ValueError(f"Unsupported model names in {view_name}: {sorted(unknown_models)}")
        view_frame = frame.copy()
        for model_name, candidate_params in candidates_for_view.items():
            probabilities_by_key: dict[str, tuple[float, float]] = {}
            threshold_by_fold: dict[int, float] = {}
            for outer_fold in fold_plan.outer_folds:
                outer_train = view_frame[view_frame.group_id.isin(outer_fold.train_group_ids)]
                outer_train_scored = outer_train[outer_train.class_id.isin((0, 1))]
                outer_valid = view_frame[view_frame.group_id.isin(outer_fold.validation_group_ids)]
                valid_scored = outer_valid[outer_valid.class_id.isin((0, 1))]
                _validate_candidate_partition(outer_train_scored, "outer training", outer_fold.fold_id)
                _validate_candidate_partition(valid_scored, "outer validation", outer_fold.fold_id)
                selection_seed = base_seed + outer_fold.fold_id * 1000
                selection = select_candidate_and_threshold(
                    model_name,
                    candidate_params,
                    outer_train,
                    feature_names,
                    fold_plan.inner_folds[outer_fold.fold_id],
                    selection_seed,
                )
                fit_seed = selection_seed + 999
                estimator, fit_warnings, fit_seconds = _fit_estimator(
                    model_name,
                    selection.params,
                    outer_train_scored[list(feature_names)],
                    outer_train_scored.class_id.to_numpy(dtype=np.int64),
                    outer_train_scored.group_id.tolist(),
                    fit_seed,
                    calibration_method=selection.calibration_method,
                )
                probabilities = _positive_probability(estimator, valid_scored[list(feature_names)])
                for key, probability in zip(valid_scored.window_key.astype(str), probabilities, strict=True):
                    probabilities_by_key[key] = (float(probability[0]), float(probability[1]))
                threshold_by_fold[outer_fold.fold_id] = float(selection.threshold)

                target = (
                    artifact_dir
                    / _safe_name(view_name)
                    / _safe_name(model_name)
                    / f"outer_{outer_fold.fold_id:02d}.joblib"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                import joblib

                joblib.dump(estimator, target, compress=3)
                view_token = "m" if view_name == "A1_morphology_only" else "t"
                model_token = {
                    "dummy_majority": "dum",
                    "logistic_regression": "log",
                    "svm_rbf": "svm",
                    "random_forest": "rf",
                    "hist_gradient_boosting": "hist",
                    "mlp": "mlp",
                }[model_name]
                # Keep this evidence path short: the private run root is already nested deeply on Windows.
                inner_oof_path = artifact_dir / "s" / f"{view_token}_{model_token}_o{outer_fold.fold_id:02d}.parquet"
                inner_oof_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(selection.inner_oof_predictions).to_parquet(inner_oof_path, index=False)
                receipt_rows.append(
                    {
                        "feature_view": view_name,
                        "model_name": model_name,
                        "outer_fold_id": outer_fold.fold_id,
                        "fit_status": "completed",
                        "selected_params_json": _json(selection.params),
                        "selected_calibration_method": selection.calibration_method,
                        "selected_threshold": selection.threshold,
                        "selection_group_balanced_macro_f1": selection.group_balanced_macro_f1,
                        "selection_group_balanced_precision": selection.group_balanced_precision,
                        "selection_group_balanced_recall": selection.group_balanced_recall,
                        "selection_candidate_index": selection.selected_candidate_index,
                        "inner_candidate_receipts_json": _json(selection.candidate_receipts),
                        "feature_names_json": _json(feature_names),
                        "train_group_ids_json": _json(outer_fold.train_group_ids),
                        "validation_group_ids_json": _json(outer_fold.validation_group_ids),
                        "training_assigned_window_count": int(len(outer_train_scored)),
                        "training_class_0_windows": int(outer_train_scored.class_id.eq(0).sum()),
                        "training_class_1_windows": int(outer_train_scored.class_id.eq(1).sum()),
                        "seed": fit_seed,
                        "fit_seconds": float(fit_seconds),
                        "inner_selection_fit_warnings_json": _json(
                            sorted(
                                {
                                    warning
                                    for candidate in selection.candidate_receipts
                                    for warning in candidate.get("warnings", [])
                                }
                            )
                        ),
                        "outer_fit_warnings_json": _json(fit_warnings),
                        "model_artifact_path": str(target.resolve()),
                        "model_artifact_sha256": _sha256_file(target),
                        "inner_oof_artifact_path": str(inner_oof_path.resolve()),
                        "inner_oof_artifact_sha256": _sha256_file(inner_oof_path),
                    }
                )
            expected = set(view_frame.loc[view_frame.class_id.isin((0, 1)), "window_key"].astype(str))
            if expected != set(probabilities_by_key):
                raise RuntimeError(
                    f"Nested threshold OOF coverage mismatch for {view_name}/{model_name}: "
                    f"missing={len(expected - set(probabilities_by_key))}, extra={len(set(probabilities_by_key) - expected)}"
                )
            prediction_records.extend(
                _threshold_prediction_rows(
                    view_frame,
                    feature_view=view_name,
                    model_name=model_name,
                    fold_by_group=fold_by_group,
                    probabilities=probabilities_by_key,
                    threshold_by_fold=threshold_by_fold,
                )
            )

    predictions = pd.DataFrame(prediction_records)
    key = ["feature_view", "model_name", "threshold_strategy", "window_key"]
    if predictions.duplicated(key).any():
        raise RuntimeError("Nested threshold results contain duplicate prediction rows")
    expected_rows = len(frame)
    row_counts = predictions.groupby(["feature_view", "model_name", "threshold_strategy"], sort=False).size()
    if not row_counts.eq(expected_rows).all():
        raise RuntimeError("Each model/view/threshold strategy must emit one row per development window")
    pooled, per_group = _nested_threshold_metrics(predictions)
    paired = _paired_nested_threshold_comparisons(
        predictions,
        replicates=int(cv.get("group_bootstrap_replicates", 2000)),
        seed=int(cv.get("seed", fold_plan.seed)),
    )
    return NestedThresholdResults(
        predictions=predictions,
        pooled_metrics=pooled,
        group_metrics=per_group,
        paired_comparisons=paired,
        model_receipts=pd.DataFrame(receipt_rows),
        fold_manifest=_serialized_fold_plan(fold_plan),
    )


def run_nested_threshold_stage(
    run_dir: Path,
    config_path: Path,
    *,
    output_subdir: str = "lowfreq_threshold_v2",
) -> dict[str, object]:
    """Run development-only low-frequency-proxy nested threshold selection."""
    import tomllib

    run_dir = Path(run_dir)
    private_dir = run_dir / "private"
    feature_dir = private_dir / "features_labels"
    if Path(output_subdir).name != output_subdir or output_subdir in {"", ".", ".."}:
        raise ValueError("output_subdir must be a single directory name")
    split_doc = json.loads((private_dir / "split_manifest.json").read_text(encoding="utf-8"))
    split = GroupSplit(
        seed=int(split_doc["seed"]),
        train_fraction=float(split_doc["train_fraction"]),
        development_groups=tuple(str(value) for value in split_doc["development_group_ids"]),
        test_groups=tuple(str(value) for value in split_doc["test_group_ids"]),
    )
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    active_rule = "low_frequency_fraction_persistence_v1"
    label_tables_all = _label_rule_tables(private_dir)
    label_rules = {rule for rule, _ in label_tables_all}
    if active_rule not in label_rules:
        raise ValueError(f"Active low-frequency proxy label rule is unavailable; found {sorted(label_rules)}")
    features = pd.read_parquet(feature_dir / "features.parquet")
    primary_window_s = float(config.get("run", {}).get("primary_window_s", 0.5))
    windows = features.loc[np.isclose(features.duration_s.to_numpy(dtype=float), primary_window_s)].copy()
    if windows.empty:
        raise ValueError("No primary-window features were found")
    labels = next(table for rule, table in label_tables_all if rule == active_rule).copy()
    expected_keys = set(windows.window_key.astype(str))
    if set(labels.window_key.astype(str)) - expected_keys:
        raise ValueError("Primary pseudo-label table contains keys outside the primary-window feature table")
    labels = labels[labels.group_id.astype(str).isin(split.development_groups)].copy()
    windows = windows[windows.group_id.astype(str).isin(split.development_groups)].copy()
    if set(windows.group_id.astype(str)) != set(split.development_groups):
        raise ValueError("Development feature table does not contain all split development groups")

    historical_fold_path = private_dir / "supervised_models" / "fold_manifest.json"
    if not historical_fold_path.is_file():
        raise FileNotFoundError(f"Historical 4x3 fold manifest is required: {historical_fold_path}")
    fold_plan = _fold_plan_from_manifest(historical_fold_path)
    label_tables = [
        (rule, table[table.group_id.astype(str).isin(split.development_groups)].copy())
        for rule, table in label_tables_all
    ]
    support_summary, support_by_fold, support_by_rule, checked_plan = _audit_label_support(
        label_tables,
        split,
        config,
        plan=fold_plan,
    )
    if _serialized_fold_plan(checked_plan) != json.loads(historical_fold_path.read_text(encoding="utf-8")):
        # Legacy manifests may differ only in JSON object key ordering; normalize through serialization.
        historical = json.loads(historical_fold_path.read_text(encoding="utf-8"))
        if _json(_serialized_fold_plan(checked_plan)) != _json(historical):
            raise ValueError("Loaded fold plan does not exactly match the historical fold assignment")

    stage_dir = private_dir / output_subdir
    if stage_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing threshold-selection stage: {stage_dir}")
    stage_dir.mkdir(parents=True)
    support_summary.to_csv(stage_dir / "label_support_summary.csv", index=False)
    support_by_fold.to_csv(stage_dir / "label_support_by_fold.csv", index=False)
    chosen_support = support_by_rule[active_rule]
    if not chosen_support["fold_coverage_gate_pass"]:
        receipt = {
            "stage": "development_nested_grouped_inner_oof_threshold_selection",
            "status": "stopped_before_training_insufficient_grouped_class_support",
            "label_rule": active_rule,
            "label_rule_type": "weak_supervision_proxy_not_independent_fatigue_ground_truth",
            "coverage_gate": chosen_support,
            "fold_manifest_sha256": _sha256_file(historical_fold_path),
            "outer_group_folds": 4,
            "inner_group_folds": 3,
            "internal_test_signal_values_parsed": 0,
            "external_rar_opened": False,
        }
        (stage_dir / "supervised_threshold_stage_manifest.json").write_text(_json(receipt), encoding="utf-8")
        return receipt
    if "feature_valid" in windows:
        invalid_keys = set(windows.loc[windows.feature_valid.ne(1), "window_key"].astype(str))
        invalid_assigned = set(labels.loc[labels.class_id.isin((0, 1)), "window_key"].astype(str)) & invalid_keys
        if invalid_assigned:
            raise ValueError(f"Assigned pseudo-labels include {len(invalid_assigned)} feature-invalid windows")

    candidate_models = candidate_grid(config)
    views: dict[str, Sequence[str]] = {
        "A1_morphology_only": tuple(FEATURE_VIEWS["A1_morphology_only"]),
        "relative_time_only": ("relative_time_fraction",),
    }
    models_by_view: dict[str, Mapping[str, Sequence[Mapping[str, object]]]] = {
        "A1_morphology_only": candidate_models,
        "relative_time_only": {"logistic_regression": candidate_models["logistic_regression"]},
    }
    result = nested_group_threshold_predictions(
        windows,
        labels,
        views,
        models_by_view,
        split,
        config,
        fold_plan=fold_plan,
        artifact_dir=stage_dir / "artifacts",
    )
    result.predictions.to_parquet(stage_dir / "nested_oof_predictions.parquet", index=False)
    result.pooled_metrics.to_csv(stage_dir / "pooled_metrics_by_threshold.csv", index=False)
    result.group_metrics.to_csv(stage_dir / "metrics_by_group.csv", index=False)
    result.paired_comparisons.to_csv(stage_dir / "paired_threshold_and_temporal_comparisons.csv", index=False)
    result.model_receipts.to_csv(stage_dir / "outer_model_receipts.csv", index=False)
    (stage_dir / "fold_manifest.json").write_text(_json(result.fold_manifest), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
    code_paths = {
        "evaluation.py": Path(__file__).resolve(),
        "contracts.py": Path(__file__).with_name("contracts.py").resolve(),
        "metrics.py": Path(__file__).with_name("metrics.py").resolve(),
        "models_tabular.py": Path(__file__).with_name("models_tabular.py").resolve(),
        "run_fatigue_proxy.py": repo_root / "tools" / "run_fatigue_proxy.py",
    }
    branch_result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    input_paths = [
        feature_dir / "features.parquet",
        feature_dir / "pseudo_labels.parquet",
        feature_dir / "pseudo_labels_alternatives.parquet",
        private_dir / "split_manifest.json",
        historical_fold_path,
        Path(config_path),
        *code_paths.values(),
    ]
    output_paths = [
        stage_dir / "label_support_summary.csv",
        stage_dir / "label_support_by_fold.csv",
        stage_dir / "nested_oof_predictions.parquet",
        stage_dir / "pooled_metrics_by_threshold.csv",
        stage_dir / "metrics_by_group.csv",
        stage_dir / "paired_threshold_and_temporal_comparisons.csv",
        stage_dir / "outer_model_receipts.csv",
        stage_dir / "fold_manifest.json",
    ]
    output_paths.extend(sorted((stage_dir / "artifacts").rglob("*.joblib")))
    output_paths.extend(sorted((stage_dir / "artifacts").rglob("*.parquet")))
    receipt = {
        "stage": "development_nested_grouped_inner_oof_threshold_selection",
        "status": "completed",
        "label_rule": active_rule,
        "label_rule_type": "weak_supervision_proxy_not_independent_fatigue_ground_truth",
        "coverage_gate": chosen_support,
        "sensitivity_label_support": {
            rule: support_by_rule[rule]
            for rule in sorted(support_by_rule)
            if rule != active_rule
        },
        "fold_source": str(historical_fold_path),
        "fold_manifest_sha256": _sha256_file(historical_fold_path),
        "git_branch": branch_result.stdout.strip() if branch_result.returncode == 0 else None,
        "git_commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "working_tree_dirty_at_run": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
        "source_files_sha256": {name: _sha256_file(path) for name, path in code_paths.items()},
        "cv_seed": fold_plan.seed,
        "outer_group_folds": 4,
        "inner_group_folds": 3,
        "selection_metric": "group-balanced Macro-F1 from concatenated inner out-of-fold probabilities",
        "threshold_comparison": [0.5, "inner_oof_selected"],
        "calibration": "group-disjoint sigmoid when support permits; uncalibrated candidate retained where viable",
        "primary_feature_view": "A1_morphology_only",
        "temporal_baseline": "relative_time_only/logistic_regression",
        "paired_comparison_unit": "same outer fold, group, window_key, class_id, and proxy-label rule",
        "training_evaluation_window_subset": "all primary 500 ms windows; 50 percent overlap",
        "ambiguous_labels_used_for_fit": False,
        "overlapping_windows_treated_as_independent_groups": False,
        "independent_unit": "source folder group; participant identity is unverified",
        "internal_test_signal_values_parsed": 0,
        "external_rar_opened": False,
        "inputs_sha256": {str(path): _sha256_file(path) for path in input_paths},
        "outputs_sha256": {str(path): _sha256_file(path) for path in output_paths},
        "checkpoint_count": len(list((stage_dir / "artifacts").rglob("*.joblib"))),
        "inner_oof_selection_receipt_count": len(list((stage_dir / "artifacts").rglob("*.parquet"))),
        "assigned_development_windows": int(labels.class_id.isin((0, 1)).sum()),
        "development_windows_total": int(len(labels)),
        "pseudo_label_coverage": float(labels.class_id.isin((0, 1)).mean()),
    }
    manifest_path = stage_dir / "supervised_threshold_stage_manifest.json"
    manifest_path.write_text(_json(receipt), encoding="utf-8")
    return {**receipt, "manifest_path": str(manifest_path.resolve())}


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
    *,
    plan: FoldPlan | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]], FoldPlan]:
    cv = config.get("cv", {})
    seed = int(cv.get("seed", 20261113))
    if plan is None:
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


def run_a1_optimization_stage(
    run_dir: Path,
    config_path: Path,
    *,
    parent_subdir: str = "lowfreq_threshold_v2",
    output_subdir: str = "a1_lowfreq_optimization_v1",
) -> dict[str, object]:
    """Nested 4x3 A1 search for morphology regularization and feature subsets.

    All candidate hyperparameters, calibration choices, and decision thresholds
    are selected from inner group-held-out predictions. Outer groups are used
    once for paired OOF performance estimation only.
    """
    import itertools
    import tomllib

    run_dir = Path(run_dir).expanduser().resolve()
    private_dir = run_dir / "private"
    feature_dir = private_dir / "features_labels"
    for name in (parent_subdir, output_subdir):
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"{name!r} must be a single directory name")
    parent_dir = private_dir / parent_subdir
    parent_receipt_path = parent_dir / "supervised_threshold_stage_manifest.json"
    historical_fold_path = parent_dir / "fold_manifest.json"
    if not parent_receipt_path.is_file() or not historical_fold_path.is_file():
        raise FileNotFoundError("Completed low-frequency 4x3 threshold parent is required")
    parent_receipt = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
    if (
        parent_receipt.get("status") != "completed"
        or parent_receipt.get("label_rule") != "low_frequency_fraction_persistence_v1"
        or int(parent_receipt.get("outer_group_folds", -1)) != 4
        or int(parent_receipt.get("inner_group_folds", -1)) != 3
    ):
        raise ValueError("Parent stage does not match the active low-frequency 4x3 contract")
    expected_fold_hash = parent_receipt.get("outputs_sha256", {}).get(str(historical_fold_path))
    if not expected_fold_hash or _sha256_file(historical_fold_path) != expected_fold_hash:
        raise ValueError("Parent 4x3 fold manifest checksum mismatch")

    split_doc = json.loads((private_dir / "split_manifest.json").read_text(encoding="utf-8"))
    split = GroupSplit(
        seed=int(split_doc["seed"]),
        train_fraction=float(split_doc["train_fraction"]),
        development_groups=tuple(str(value) for value in split_doc["development_group_ids"]),
        test_groups=tuple(str(value) for value in split_doc["test_group_ids"]),
    )
    fold_plan = _fold_plan_from_manifest(historical_fold_path)
    if tuple(sorted(fold_plan.development_group_ids)) != tuple(sorted(split.development_groups)):
        raise ValueError("Parent fold groups differ from the frozen development groups")
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))

    # Use Parquet row-group filtering so reserved groups are not materialized.
    development_groups = list(split.development_groups)
    features = pd.read_parquet(
        feature_dir / "features.parquet", filters=[("group_id", "in", development_groups)]
    )
    labels = pd.read_parquet(
        feature_dir / "pseudo_labels_alternatives.parquet",
        filters=[("group_id", "in", development_groups)],
    )
    labels = labels.loc[
        labels.comparison_rule.eq("low_frequency_fraction_persistence_v1")
    ].drop(columns=["comparison_rule"]).copy()
    primary_window_s = float(config.get("run", {}).get("primary_window_s", 0.5))
    windows = features.loc[
        np.isclose(features.duration_s.to_numpy(dtype=float), primary_window_s)
    ].copy()
    if windows.empty or set(windows.group_id.astype(str)) != set(split.development_groups):
        raise ValueError("Filtered development morphology features do not cover the frozen groups")
    if set(labels.group_id.astype(str)) != set(split.development_groups):
        raise ValueError("Filtered primary labels do not cover the frozen development groups")
    support_summary, support_by_fold, support_by_rule, checked_plan = _audit_label_support(
        [("low_frequency_fraction_persistence_v1", labels)], split, config, plan=fold_plan
    )
    if not support_by_rule["low_frequency_fraction_persistence_v1"]["fold_coverage_gate_pass"]:
        raise ValueError("Primary low-frequency label support failed a nested 4x3 class/group/trial gate")
    if _json(_serialized_fold_plan(checked_plan)) != _json(json.loads(historical_fold_path.read_text(encoding="utf-8"))):
        raise ValueError("A1 optimization fold plan differs from the frozen historical 4x3 plan")

    output_dir = private_dir / output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite A1 optimization stage: {output_dir}")
    output_dir.mkdir(parents=True)
    support_summary.to_csv(output_dir / "label_support_summary.csv", index=False)
    support_by_fold.to_csv(output_dir / "label_support_by_fold.csv", index=False)

    # C, class weighting, robust/standard scaling, and univariate feature counts
    # are all fit within each inner training partition by the sklearn Pipeline.
    candidate_params: list[dict[str, object]] = []
    for c_value, class_weight, scaler, feature_k in itertools.product(
        (0.01, 0.1, 1.0, 10.0, 100.0),
        ("balanced", None),
        ("standard", "robust"),
        (None, 8, 12),
    ):
        row: dict[str, object] = {
            "C": c_value,
            "class_weight": class_weight,
            "scaler": scaler,
        }
        if feature_k is not None:
            row["feature_k"] = feature_k
        candidate_params.append(row)

    from .features import FEATURE_VIEWS

    views: dict[str, Sequence[str]] = {
        "A1_morphology_only": tuple(FEATURE_VIEWS["A1_morphology_only"]),
        "relative_time_only": ("relative_time_fraction",),
    }
    models_by_view: dict[str, Mapping[str, Sequence[Mapping[str, object]]]] = {
        "A1_morphology_only": {"logistic_regression": candidate_params},
        # Keep the historical temporal-only candidate family for a direct paired control.
        "relative_time_only": {"logistic_regression": candidate_grid(config)["logistic_regression"]},
    }
    result = nested_group_threshold_predictions(
        windows,
        labels,
        views,
        models_by_view,
        split,
        config,
        fold_plan=fold_plan,
        artifact_dir=output_dir / "checkpoints",
    )
    result.predictions.to_parquet(output_dir / "nested_oof_predictions.parquet", index=False)
    result.pooled_metrics.to_csv(output_dir / "pooled_metrics_by_threshold.csv", index=False)
    result.group_metrics.to_csv(output_dir / "metrics_by_group.csv", index=False)
    result.paired_comparisons.to_csv(output_dir / "paired_group_comparisons.csv", index=False)
    result.model_receipts.to_csv(output_dir / "outer_model_receipts.csv", index=False)
    fold_path = output_dir / "fold_manifest.json"
    fold_path.write_text(_json(result.fold_manifest), encoding="utf-8")

    inputs = [
        feature_dir / "features.parquet",
        feature_dir / "pseudo_labels_alternatives.parquet",
        private_dir / "split_manifest.json",
        historical_fold_path,
        parent_receipt_path,
        Path(config_path),
        Path(__file__),
        Path(__file__).with_name("models_tabular.py"),
        Path(__file__).with_name("metrics.py"),
    ]
    outputs = [
        output_dir / "label_support_summary.csv",
        output_dir / "label_support_by_fold.csv",
        output_dir / "nested_oof_predictions.parquet",
        output_dir / "pooled_metrics_by_threshold.csv",
        output_dir / "metrics_by_group.csv",
        output_dir / "paired_group_comparisons.csv",
        output_dir / "outer_model_receipts.csv",
        fold_path,
        *sorted((output_dir / "checkpoints").rglob("*.joblib")),
        *sorted((output_dir / "checkpoints").rglob("*.parquet")),
    ]
    receipt = {
        "stage": "development_nested_grouped_a1_morphology_optimization",
        "status": "completed",
        "label_rule": "low_frequency_fraction_persistence_v1",
        "label_rule_type": "weak_supervision_proxy_not_independent_fatigue_ground_truth",
        "outer_group_folds": 4,
        "inner_group_folds": 3,
        "fold_source_sha256": _sha256_file(historical_fold_path),
        "selection": "candidate hyperparameters and calibration selected on inner group OOF probabilities; threshold selected by group-balanced Macro-F1 on selected-candidate inner OOF probabilities",
        "candidate_count_per_outer_fold": len(candidate_params),
        "candidate_families": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "class_weight": ["balanced", None],
            "scaler": ["standard", "robust"],
            "feature_k": ["all", 8, 12],
            "feature_selection_fit_scope": "each inner training split only",
        },
        "calibration": "group-disjoint sigmoid calibration selected when support permits; uncalibrated variant retained only when calibrated fit is unsupported",
        "threshold_comparison": [0.5, "inner_oof_selected"],
        "primary_selection_metric": "group-balanced Macro-F1",
        "paired_control": "temporal-only logistic baseline from the same development groups/windows/labels and historical 4x3 folds",
        "independent_unit": "source folder group; participant identity is unverified",
        "overlapping_windows_treated_as_independent_groups": False,
        "reserved_test_member_payloads_read": 0,
        "internal_test_signal_values_parsed": 0,
        "external_rar_opened": False,
        "checkpoint_count": len(list((output_dir / "checkpoints").rglob("*.joblib"))),
        "inputs_sha256": {str(path.resolve()): _sha256_file(path) for path in inputs},
        "outputs_sha256": {str(path.resolve()): _sha256_file(path) for path in outputs},
        "interpretation": "agreement with a weak EMG proxy; no independent physiological fatigue validation",
    }
    manifest_path = output_dir / "a1_optimization_stage_manifest.json"
    manifest_path.write_text(_json(receipt), encoding="utf-8")
    return {**receipt, "manifest_path": str(manifest_path.resolve())}


def run_supervised_stage(
    run_dir: Path,
    config_path: Path,
    *,
    label_rule: str,
    output_subdir: str = "supervised_models",
    coverage_aware_folds: bool = False,
    nonoverlap_only: bool = False,
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
    if Path(output_subdir).name != output_subdir or output_subdir in {"", ".", ".."}:
        raise ValueError("output_subdir must be a single directory name")
    label_tables_all = _label_rule_tables(private_dir)
    if label_rule not in {rule for rule, _ in label_tables_all}:
        raise ValueError(f"Unknown label rule {label_rule}; available: {sorted(rule for rule, _ in label_tables_all)}")
    features = pd.read_parquet(feature_dir / "features.parquet")
    primary_window_s = float(config.get("run", {}).get("primary_window_s", 0.5))
    all_primary_windows = features.loc[
        np.isclose(features.duration_s.to_numpy(dtype=float), primary_window_s)
    ].copy()
    if all_primary_windows.empty:
        raise ValueError("No primary-window features were found")

    active_labels_all = next(table for rule, table in label_tables_all if rule == label_rule).copy()
    expected_keys = set(all_primary_windows.window_key.astype(str))
    if set(active_labels_all.window_key.astype(str)) - expected_keys:
        raise ValueError("Pseudo-label table contains keys outside the primary-window feature table")

    # Keep the full overlapping-window distribution for audit, then optionally
    # restrict fitting and OOF scoring to one non-overlapping parity.
    active_assigned_all = active_labels_all[active_labels_all.class_id.isin((0, 1))]
    all_trial_support = active_assigned_all.groupby(["group_id", "recording_id", "class_id"]).size()
    class_summary_rows: list[dict[str, object]] = []
    for class_id, class_name in ((0, "non_fatigue_like"), (1, "fatigue_like"), (-1, "ambiguous")):
        selected = active_labels_all[active_labels_all.class_id.eq(class_id)]
        assigned_selected = selected[selected.class_id.isin((0, 1))]
        class_summary_rows.append(
            {
                "label_rule": label_rule,
                "class_id": class_id,
                "class_name": class_name,
                "window_count_descriptive_only": int(len(selected)),
                "distinct_source_groups": int(selected.group_id.nunique()),
                "distinct_recordings_trials": int(selected[["group_id", "recording_id"]].drop_duplicates().shape[0]),
                "assigned_window_coverage": float(len(active_assigned_all) / len(active_labels_all)),
            }
        )
    active_group_counts = (
        active_labels_all.groupby(["group_id", "class_id"]).size().unstack(fill_value=0)
        .reindex(columns=[-1, 0, 1], fill_value=0)
        .rename(columns={-1: "ambiguous_windows", 0: "class_0_windows", 1: "class_1_windows"})
    )
    active_group_trial_counts = all_trial_support.rename("assigned_windows_descriptive_only").reset_index()
    active_trial_counts = (
        active_labels_all.groupby(["group_id", "recording_id", "class_id"]).size().unstack(fill_value=0)
        .reindex(columns=[-1, 0, 1], fill_value=0)
        .rename(columns={-1: "ambiguous_windows", 0: "class_0_windows", 1: "class_1_windows"})
        .reset_index()
    )

    rule_distribution_rows: list[dict[str, object]] = []
    for rule, table in label_tables_all:
        scored = table[table.class_id.isin((0, 1))]
        trial_flags = scored.groupby(["group_id", "recording_id"]).class_id.agg(
            has_class_0=lambda values: bool((values == 0).any()),
            has_class_1=lambda values: bool((values == 1).any()),
        )
        rule_distribution_rows.append(
            {
                "rule_version": rule,
                "total_primary_windows": int(len(table)),
                "ambiguous_windows": int(table.class_id.eq(-1).sum()),
                "class_0_windows_descriptive_only": int(table.class_id.eq(0).sum()),
                "class_1_windows_descriptive_only": int(table.class_id.eq(1).sum()),
                "assigned_label_coverage": float(len(scored) / len(table)) if len(table) else 0.0,
                "groups_with_class_0": int(scored.loc[scored.class_id.eq(0), "group_id"].nunique()),
                "groups_with_class_1": int(scored.loc[scored.class_id.eq(1), "group_id"].nunique()),
                "trials_with_class_0": int(trial_flags.has_class_0.sum()),
                "trials_with_class_1": int(trial_flags.has_class_1.sum()),
                "trials_with_both_classes": int((trial_flags.has_class_0 & trial_flags.has_class_1).sum()),
            }
        )

    labels = active_labels_all
    windows = all_primary_windows
    label_tables = label_tables_all
    if nonoverlap_only:
        if "window_index" not in windows:
            raise ValueError("Non-overlap filtering requires window_index in the feature table")
        windows = windows.loc[windows.window_index.astype(int).mod(2).eq(0)].copy()
        retained_keys = set(windows.window_key.astype(str))
        labels = labels[labels.window_key.astype(str).isin(retained_keys)].copy()
        label_tables = [
            (rule, table[table.window_key.astype(str).isin(retained_keys)].copy())
            for rule, table in label_tables_all
        ]

    if coverage_aware_folds:
        fold_plan = group_folds_with_class_coverage(
            labels.group_id.astype(str).tolist(),
            labels.recording_id.astype(str).tolist(),
            labels.class_id.astype(int).tolist(),
            outer_folds=3,
            inner_folds=2,
            seed=int(config.get("cv", {}).get("seed", 20261113)),
            min_groups_per_class=2,
            min_trials_per_class=2,
        )
        effective_config = dict(config)
        effective_cv = dict(config.get("cv", {}))
        effective_cv.update({"outer_group_folds": 3, "inner_group_folds": 2})
        effective_config["cv"] = effective_cv
    else:
        fold_plan = None
        effective_config = config

    summaries, fold_support, by_rule, fold_plan = _audit_label_support(
        label_tables,
        split,
        effective_config,
        plan=fold_plan,
    )
    stage_dir = private_dir / output_subdir
    if stage_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing supervised model stage: {stage_dir}")
    stage_dir.mkdir(parents=True)
    summaries.to_csv(stage_dir / "label_support_summary.csv", index=False)
    fold_support.to_csv(stage_dir / "label_support_by_fold.csv", index=False)
    pd.DataFrame(rule_distribution_rows).to_csv(
        stage_dir / "label_support_all_overlapping_windows.csv", index=False
    )
    pd.DataFrame(class_summary_rows).to_csv(stage_dir / "active_label_distribution_by_class.csv", index=False)
    active_group_counts.to_csv(stage_dir / "active_label_distribution_by_group.csv")
    active_trial_counts.to_csv(stage_dir / "active_label_distribution_by_trial.csv", index=False)
    active_group_trial_counts.to_csv(stage_dir / "active_label_trial_support_by_group_class.csv", index=False)
    chosen_support = by_rule[label_rule]
    if not chosen_support["fold_coverage_gate_pass"]:
        receipt = {
            "stage": "development_nested_grouped_supervised_models",
            "status": "stopped_before_training_insufficient_grouped_class_support",
            "label_rule": label_rule,
            "coverage_gate": chosen_support,
            "fold_strategy": "coverage_aware_group_trial_presence" if coverage_aware_folds else "group_kfold",
            "outer_group_folds": len(fold_plan.outer_folds),
            "inner_group_folds": len(fold_plan.inner_folds[fold_plan.outer_folds[0].fold_id]),
            "training_evaluation_window_subset": "even-index non-overlapping 500 ms windows" if nonoverlap_only else "all 500 ms windows",
            "active_label_distribution_all_overlapping_windows": rule_distribution_rows[
                next(index for index, item in enumerate(rule_distribution_rows) if item["rule_version"] == label_rule)
            ],
            "internal_test_signal_values_parsed": 0,
            "external_rar_opened": False,
        }
        (stage_dir / "supervised_model_stage_manifest.json").write_text(_json(receipt), encoding="utf-8")
        return receipt

    if "feature_valid" in windows:
        invalid_assigned = labels.loc[labels.class_id.isin((0, 1)), "window_key"].isin(
            windows.loc[windows.feature_valid.ne(1), "window_key"]
        )
        if invalid_assigned.any():
            raise ValueError("Assigned pseudo-labels include feature-invalid windows")

    if coverage_aware_folds:
        views: dict[str, Sequence[str]] = {
            "A1_morphology_only": tuple(FEATURE_VIEWS["A1_morphology_only"]),
            "relative_time_only": ("relative_time_fraction",),
        }
        all_model_candidates = candidate_grid(effective_config)
        model_candidates = {
            name: candidates
            for name, candidates in all_model_candidates.items()
            if name != "svm_rbf"
        }
        excluded_models = {
            "svm_rbf": "Skipped: its three-way probability-calibration split cannot reliably retain both classes in each inner training partition at this support level."
        }
    else:
        views = {name: tuple(names) for name, names in FEATURE_VIEWS.items()}
        views["relative_time_only"] = ("relative_time_fraction",)
        model_candidates = candidate_grid(effective_config)
        excluded_models = {}
    result = nested_group_predictions(
        windows,
        labels,
        views,
        model_candidates,
        split,
        effective_config,
        artifact_dir=stage_dir / "checkpoints",
        fold_plan=fold_plan,
        group_balanced_selection=coverage_aware_folds,
    )
    result.predictions.to_parquet(stage_dir / "oof_predictions.parquet", index=False)
    result.pooled_metrics.to_csv(stage_dir / "pooled_metrics.csv", index=False)
    result.group_metrics.to_csv(stage_dir / "group_metrics.csv", index=False)
    result.bootstrap_intervals.to_csv(stage_dir / "group_bootstrap_intervals.csv", index=False)
    result.model_receipts.to_csv(stage_dir / "model_receipts.csv", index=False)
    result.baseline_comparisons.to_csv(stage_dir / "temporal_baseline_comparisons.csv", index=False)
    paired_group_comparisons = _paired_group_bootstrap_vs_temporal(
        result.predictions,
        replicates=int(effective_config.get("cv", {}).get("group_bootstrap_replicates", 2000)),
        seed=int(effective_config.get("cv", {}).get("seed", 20261113)),
    )
    paired_group_comparisons.to_csv(stage_dir / "paired_group_bootstrap_vs_time.csv", index=False)
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
        stage_dir / "label_support_all_overlapping_windows.csv",
        stage_dir / "active_label_distribution_by_class.csv",
        stage_dir / "active_label_distribution_by_group.csv",
        stage_dir / "active_label_distribution_by_trial.csv",
        stage_dir / "active_label_trial_support_by_group_class.csv",
        stage_dir / "oof_predictions.parquet",
        stage_dir / "pooled_metrics.csv",
        stage_dir / "group_metrics.csv",
        stage_dir / "group_bootstrap_intervals.csv",
        stage_dir / "model_receipts.csv",
        stage_dir / "temporal_baseline_comparisons.csv",
        stage_dir / "paired_group_bootstrap_vs_time.csv",
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
        "cv_seed": int(effective_config["cv"]["seed"]),
        "outer_group_folds": len(fold_plan.outer_folds),
        "inner_group_folds": len(fold_plan.inner_folds[fold_plan.outer_folds[0].fold_id]),
        "fold_strategy": "coverage_aware_group_trial_presence" if coverage_aware_folds else "group_kfold",
        "coverage_minima": {"source_groups_per_class": 2, "recording_trials_per_class": 2} if coverage_aware_folds else None,
        "training_evaluation_window_subset": "even-index non-overlapping 500 ms windows" if nonoverlap_only else "all 500 ms windows",
        "overlap_between_selected_windows": "none; 500 ms start-to-start hop" if nonoverlap_only else "50 percent",
        "excluded_model_families": excluded_models,
        "selection_metric": "mean inner-fold macro-F1 with equal total weight per source group" if coverage_aware_folds else "mean validation-group macro-F1",
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
        "assigned_development_windows_all_overlapping": int(active_assigned_all.shape[0]),
        "development_windows_total": int(len(labels)),
        "pseudo_label_coverage": float(labels.class_id.isin((0, 1)).mean()),
        "pooled_metric_confidence_thresholds": sorted(set([0.0] + [float(v) for v in config.get("guardrails", {}).get("confidence_thresholds", [])])),
    }
    (stage_dir / "supervised_model_stage_manifest.json").write_text(_json(receipt), encoding="utf-8")
    return receipt
