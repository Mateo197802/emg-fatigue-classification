"""Binary proxy classification metrics and source-group bootstrap intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MetricRecord:
    n_samples: int
    n_predicted: int
    n_abstained: int
    coverage: float | None
    abstention_rate: float | None
    accuracy: float | None
    selective_accuracy: float | None
    selective_error: float | None
    macro_f1: float | None
    balanced_accuracy: float | None
    per_class: dict[str, dict[str, float | int | None]]
    confusion_matrix: list[list[int]]
    confusion_true_labels: tuple[int, int]
    confusion_predicted_labels: tuple[int, int, str]
    calibration: dict[str, float | int | None]
    undefined_reasons: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _probability_matrix(y_score: Sequence[float] | np.ndarray | None, n: int) -> np.ndarray | None:
    if y_score is None:
        return None
    score = np.asarray(y_score, dtype=np.float64)
    if score.ndim == 1 and score.size == n:
        positive = score
        score = np.column_stack((1.0 - positive, positive))
    if score.shape != (n, 2):
        raise ValueError("y_score must be None, one positive-class probability per row, or an n-by-2 matrix")
    return score


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray | None,
    y_pred: Sequence[int] | np.ndarray,
    abstain: Sequence[bool] | np.ndarray,
) -> MetricRecord:
    """Compute binary metrics with abstentions explicit in the confusion table."""
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    abstained = np.asarray(abstain, dtype=bool)
    if truth.ndim != 1 or pred.shape != truth.shape or abstained.shape != truth.shape:
        raise ValueError("y_true, y_pred, and abstain must be one-dimensional arrays with equal lengths")
    if np.any(~np.isin(truth, (0, 1))):
        raise ValueError("y_true must contain only assigned proxy classes 0 and 1")
    if np.any(~np.isin(pred, (-1, 0, 1))):
        raise ValueError("y_pred must contain only 0, 1, or -1 for abstention")
    n = int(truth.size)
    scores = _probability_matrix(y_score, n)
    abstained = abstained | (pred == -1)
    effective_pred = np.where(abstained, -1, pred)
    predicted_mask = ~abstained
    n_predicted = int(predicted_mask.sum())
    n_abstained = n - n_predicted
    undefined: dict[str, str] = {}

    if n:
        coverage: float | None = n_predicted / n
        abstention_rate: float | None = n_abstained / n
        accuracy: float | None = float(np.mean((effective_pred == truth) & predicted_mask))
    else:
        coverage = abstention_rate = accuracy = None
        undefined.update(
            {
                "coverage": "no evaluation samples",
                "abstention_rate": "no evaluation samples",
                "accuracy": "no evaluation samples",
            }
        )

    if n_predicted:
        selective_accuracy: float | None = float(np.mean(effective_pred[predicted_mask] == truth[predicted_mask]))
        selective_error: float | None = 1.0 - selective_accuracy
    else:
        selective_accuracy = selective_error = None
        undefined["selective_error"] = "all evaluation samples were abstained"

    confusion = np.zeros((2, 3), dtype=np.int64)
    predicted_column = {0: 0, 1: 1, -1: 2}
    for actual, predicted in zip(truth, effective_pred, strict=True):
        confusion[int(actual), predicted_column[int(predicted)]] += 1

    per_class: dict[str, dict[str, float | int | None]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for class_id in (0, 1):
        true_class = truth == class_id
        pred_class = (effective_pred == class_id) & predicted_mask
        tp = int(np.sum(true_class & pred_class))
        fp = int(np.sum(~true_class & pred_class))
        support = int(true_class.sum())
        fn = int(np.sum(true_class & ~pred_class))
        predicted_count = int(pred_class.sum())
        precision = tp / predicted_count if predicted_count else None
        recall = tp / support if support else None
        f1_denominator = 2 * tp + fp + fn
        f1 = 2 * tp / f1_denominator if f1_denominator else None
        per_class[str(class_id)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted_count,
        }
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1_values.append(f1)
        if precision is None:
            undefined[f"precision_class_{class_id}"] = "no predictions for class"
        if recall is None:
            undefined[f"recall_class_{class_id}"] = f"class {class_id} absent from truth"

    if all(int(np.sum(truth == class_id)) > 0 for class_id in (0, 1)):
        macro_f1 = float(np.mean([float(per_class[str(class_id)]["f1"]) for class_id in (0, 1)]))
        balanced_accuracy: float | None = float(np.mean(recalls)) if len(recalls) == 2 else None
        if balanced_accuracy is None:
            undefined["balanced_accuracy"] = "both classes must have defined recall"
    else:
        macro_f1 = None
        balanced_accuracy = None
        missing = [str(class_id) for class_id in (0, 1) if not np.any(truth == class_id)]
        reason = "class " + ", ".join(missing) + " absent from truth"
        undefined["macro_f1"] = reason
        undefined["balanced_accuracy"] = reason

    calibration: dict[str, float | int | None] = {
        "brier_score": None,
        "expected_calibration_error_10_bins": None,
        "calibration_bin_count": None,
    }
    if scores is None:
        undefined["calibration"] = "probability scores unavailable"
    elif n == 0:
        undefined["calibration"] = "no evaluation samples"
    elif (
        not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
        or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5)
    ):
        undefined["calibration"] = "probability scores are non-finite or not normalized"
    else:
        probability = scores[:, 1]
        calibration["brier_score"] = float(np.mean((probability - truth) ** 2))
        bin_index = np.minimum((probability * 10).astype(int), 9)
        ece = 0.0
        nonempty_bins = 0
        for index in range(10):
            selected = bin_index == index
            if np.any(selected):
                nonempty_bins += 1
                ece += float(selected.sum() / n) * abs(float(np.mean(probability[selected])) - float(np.mean(truth[selected])))
        calibration["expected_calibration_error_10_bins"] = float(ece)
        calibration["calibration_bin_count"] = nonempty_bins

    return MetricRecord(
        n_samples=n,
        n_predicted=n_predicted,
        n_abstained=n_abstained,
        coverage=coverage,
        abstention_rate=abstention_rate,
        accuracy=accuracy,
        selective_accuracy=selective_accuracy,
        selective_error=selective_error,
        macro_f1=macro_f1,
        balanced_accuracy=balanced_accuracy,
        per_class=per_class,
        confusion_matrix=confusion.tolist(),
        confusion_true_labels=(0, 1),
        confusion_predicted_labels=(0, 1, "abstain"),
        calibration=calibration,
        undefined_reasons=undefined,
    )


def grouped_bootstrap_intervals(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray | None,
    y_pred: Sequence[int] | np.ndarray,
    abstain: Sequence[bool] | np.ndarray,
    group_ids: Sequence[str] | np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 20261113,
) -> dict[str, dict[str, float | int | str | None]]:
    """Percentile intervals from resampling source groups with replacement."""
    truth = np.asarray(y_true, dtype=np.int64)
    scores = _probability_matrix(y_score, truth.size)
    pred = np.asarray(y_pred, dtype=np.int64)
    abstained = np.asarray(abstain, dtype=bool)
    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or groups.size != truth.size:
        raise ValueError("group_ids must have one value for each evaluation sample")
    if truth.size == 0:
        raise ValueError("grouped bootstrap requires at least one evaluation sample")
    if any(value is None or (isinstance(value, float) and np.isnan(value)) for value in groups):
        raise ValueError("group_ids cannot contain missing values")
    replicate_count = int(replicates)
    if replicate_count < 1:
        raise ValueError("replicates must be positive")

    normalized_groups = np.asarray([str(value) for value in groups], dtype=object)
    unique_groups = np.asarray(sorted(set(normalized_groups.tolist())), dtype=object)
    indices_by_group = {
        group_id: np.flatnonzero(normalized_groups == group_id)
        for group_id in unique_groups
    }
    metric_names = (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "coverage",
        "selective_error",
        "brier_score",
        "expected_calibration_error_10_bins",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(int(seed))
    for _ in range(replicate_count):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group_id] for group_id in sampled])
        record = compute_metrics(
            truth[indices],
            scores[indices] if scores is not None else None,
            pred[indices],
            abstained[indices],
        )
        values = record.to_dict()
        for name in metric_names:
            if name == "brier_score" or name == "expected_calibration_error_10_bins":
                value = record.calibration[name]
            else:
                value = values[name]
            if value is not None and np.isfinite(float(value)):
                samples[name].append(float(value))

    intervals: dict[str, dict[str, float | int | str | None]] = {}
    for name, values in samples.items():
        if values:
            lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
            intervals[name] = {
                "lower": float(lower),
                "upper": float(upper),
                "valid_replicates": len(values),
                "requested_replicates": replicate_count,
                "method": "source-group percentile bootstrap; groups sampled with replacement",
            }
        else:
            intervals[name] = {
                "lower": None,
                "upper": None,
                "valid_replicates": 0,
                "requested_replicates": replicate_count,
                "method": "source-group percentile bootstrap; groups sampled with replacement",
                "undefined_reason": "no replicate had a defined metric",
            }
    return intervals
