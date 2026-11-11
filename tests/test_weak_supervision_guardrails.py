from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from emg_fatigue.weak_supervision.evaluation_guardrails import (
    OOD_FEATURES,
    OOD_METHODS,
    _fit_fold_ood_detectors,
    _fit_ood_scorer,
    _risk_coverage_points,
)


def test_all_ood_scorers_fit_reference_only_and_return_finite_scores():
    rng = np.random.default_rng(41)
    fit = rng.normal(size=(80, 15))
    query = rng.normal(loc=0.4, size=(9, 15))

    for method in OOD_METHODS:
        scores, _, receipt = _fit_ood_scorer(fit, query, method)
        assert scores.shape == (9,)
        assert np.isfinite(scores).all()
        assert receipt["method"]


def test_fold_ood_calibration_uses_source_groups_and_keeps_signal_reasons_distinct():
    rng = np.random.default_rng(47)
    rows = []
    for group_index in range(5):
        for window_index in range(20):
            row = {
                "window_key": f"G{group_index}:{window_index}",
                "group_id": f"G{group_index}",
                "recording_id": f"G{group_index}_F1",
                "class_id": int(window_index % 2),
                "eligible_for_label": True,
                "feature_valid": True,
                "raw_signal_compatible": True,
            }
            row.update({feature: float(rng.normal(loc=group_index * 0.3)) for feature in OOD_FEATURES})
            rows.append(row)
    frame = pd.DataFrame(rows)

    scores, receipts = _fit_fold_ood_detectors(frame.iloc[:80], frame.iloc[80:100], quantile=0.95)

    assert set(scores.ood_detector) == set(OOD_METHODS)
    assert len(scores) == 20 * len(OOD_METHODS)
    assert scores.groupby("ood_detector").window_key.nunique().eq(20).all()
    assert len(receipts) == len(OOD_METHODS)
    assert all(receipt["outer_validation_used_for_fit_or_threshold"] is False for receipt in receipts)
    assert scores.quality_pass.all()
    assert scores.non_classifier_abstention_reasons.str.contains("classifier_uncertainty").sum() == 0
    assert scores.unsupported_signal.any()


def test_risk_coverage_combines_fixed_gates_with_confidence_and_keeps_class_coverage():
    rows = pd.DataFrame(
        {
            "group_id": ["G1", "G1", "G2", "G2"],
            "true_class_id": [0, 1, 0, 1],
            "predicted_class_id": [0, 1, 1, 0],
            "confidence": [0.9, 0.8, 0.7, 0.6],
        }
    )
    points = _risk_coverage_points(rows, [0.5, 0.75], fixed_abstain=[False, False, True, False])

    assert points.loc[points.confidence_threshold.eq(0.5), "coverage"].item() == 0.75
    assert points.loc[points.confidence_threshold.eq(0.5), "class_0_coverage"].item() == 0.5
    assert points.loc[points.confidence_threshold.eq(0.5), "class_1_coverage"].item() == 1.0
    assert points.loc[points.confidence_threshold.eq(0.5), "fixed_quality_ood_or_unsupported_abstentions"].item() == 1
