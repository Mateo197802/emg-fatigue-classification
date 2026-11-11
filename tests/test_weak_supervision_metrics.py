from __future__ import annotations

import numpy as np

from emg_fatigue.weak_supervision.metrics import (
    compute_metrics,
    grouped_bootstrap_intervals,
)


def test_metrics_count_abstentions_as_uncovered_and_report_selective_performance():
    y_true = np.asarray([0, 1, 0, 1])
    y_score = np.asarray([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5], [0.4, 0.6]])
    y_pred = np.asarray([0, 1, 1, 0])
    abstain = np.asarray([False, False, True, False])

    result = compute_metrics(y_true, y_score, y_pred, abstain)

    assert result.accuracy == 0.5
    assert result.coverage == 0.75
    assert result.abstention_rate == 0.25
    assert np.isclose(result.selective_error, 1 / 3)
    assert result.confusion_matrix == [[1, 0, 1], [1, 1, 0]]
    assert result.per_class["1"]["recall"] == 0.5
    assert np.isclose(result.calibration["brier_score"], 0.115)


def test_metrics_mark_class_absence_as_undefined_instead_of_inventing_recall():
    result = compute_metrics(
        np.asarray([0, 0]),
        np.asarray([[0.9, 0.1], [0.8, 0.2]]),
        np.asarray([0, 0]),
        np.asarray([False, False]),
    )

    assert result.macro_f1 is None
    assert result.balanced_accuracy is None
    assert result.per_class["1"]["recall"] is None
    assert "class 1 absent from truth" in result.undefined_reasons["balanced_accuracy"]
    assert result.matthews_correlation_coefficient is None
    assert result.cohen_kappa is None
    assert result.pr_auc_average_precision is None
    assert result.roc_auc is None


def test_metrics_report_full_binary_metrics_and_normalized_confusion():
    result = compute_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]]),
        np.asarray([0, 1, 1, 0]),
        np.zeros(4, dtype=bool),
    )

    assert result.accuracy == 0.5
    assert result.sensitivity == 0.5
    assert result.specificity == 0.5
    assert result.macro_f1 == 0.5
    assert result.balanced_accuracy == 0.5
    assert result.matthews_correlation_coefficient == 0.0
    assert result.cohen_kappa == 0.0
    assert np.isclose(result.pr_auc_average_precision, 7 / 12)
    assert np.isclose(result.roc_auc, 0.5)
    assert np.isclose(result.calibration["brier_score"], 0.375)
    assert np.isfinite(result.log_loss)
    assert result.confusion_matrix == [[1, 1, 0], [1, 1, 0]]
    assert result.confusion_matrix_normalized_by_true == [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]


def test_grouped_bootstrap_is_reproducible_and_resamples_whole_groups():
    y_true = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    y_score = np.asarray([[0.8, 0.2], [0.2, 0.8]] * 4)
    y_pred = np.asarray([0, 1] * 4)
    abstain = np.zeros(8, dtype=bool)
    groups = np.asarray(["A", "A", "B", "B", "C", "C", "D", "D"])

    first = grouped_bootstrap_intervals(
        y_true, y_score, y_pred, abstain, groups, replicates=40, seed=91
    )
    second = grouped_bootstrap_intervals(
        y_true, y_score, y_pred, abstain, groups, replicates=40, seed=91
    )

    assert first == second
    assert first["macro_f1"]["requested_replicates"] == 40
    assert first["macro_f1"]["valid_replicates"] == 40
    assert first["macro_f1"]["lower"] == 1.0
    assert first["macro_f1"]["upper"] == 1.0
