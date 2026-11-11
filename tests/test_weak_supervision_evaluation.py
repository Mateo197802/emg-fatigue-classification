from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from emg_fatigue.weak_supervision.contracts import GroupSplit
from emg_fatigue.weak_supervision.evaluation import (
    _fit_estimator,
    _inner_oof_candidate_predictions,
    _group_balanced_macro_f1,
    _paired_group_bootstrap_vs_temporal,
    group_calibration_splits,
    nested_group_predictions,
    select_candidate_and_threshold,
    select_group_balanced_threshold,
)
from emg_fatigue.weak_supervision.group_cv import GroupFold


def _synthetic_oof_data() -> tuple[pd.DataFrame, pd.DataFrame, GroupSplit]:
    groups = tuple(f"G{index:02d}" for index in range(8))
    windows = []
    labels = []
    for group_index, group_id in enumerate(groups):
        recording_id = f"{group_id}_R1"
        for window_index, class_id in enumerate((0, 1, 0, -1)):
            key = f"{group_id}:{recording_id}:s1:d0500:p{window_index * 250:010d}"
            amplitude = (-1.0 if class_id == 0 else 1.0) + group_index * 0.05
            windows.append(
                {
                    "window_key": key,
                    "group_id": group_id,
                    "recording_id": recording_id,
                    "segment_index": 1,
                    "window_index": window_index,
                    "start_sample": window_index * 250,
                    "start_time_s": window_index * 0.25,
                    "duration_s": 0.5,
                    "sampling_rate_hz": 1000.0,
                    "activity_occupancy": 1.0,
                    "window_eligible_at_primary_occupancy": True,
                    "feature_valid": 1,
                    "rms": amplitude,
                }
            )
            labels.append(
                {
                    "window_key": key,
                    "group_id": group_id,
                    "class_id": class_id,
                    "label_confidence": 0.8 if class_id >= 0 else 0.0,
                    "rule_version": "spectral_persistence_v1",
                    "label_reason": "synthetic fixture",
                }
            )
    split = GroupSplit(
        seed=101,
        train_fraction=0.8,
        development_groups=groups,
        test_groups=("LOCKED",),
    )
    return pd.DataFrame(windows), pd.DataFrame(labels), split


def test_nested_group_evaluation_emits_exactly_once_oof_and_private_fold_models(tmp_path: Path):
    windows, labels, split = _synthetic_oof_data()
    config = {
        "run": {"primary_window_s": 0.5},
        "cv": {
            "outer_group_folds": 2,
            "inner_group_folds": 2,
            "seed": 101,
            "group_bootstrap_replicates": 20,
        },
        "models": {"random_state": 17},
        "guardrails": {"confidence_thresholds": [0.55, 0.75]},
    }
    views = {
        "A1_morphology_only": ("rms",),
        "relative_time_only": ("relative_time_fraction",),
    }
    candidates = {
        "dummy_majority": [{"strategy": "prior"}],
        "logistic_regression": [{"C": 1.0}],
    }

    result = nested_group_predictions(
        windows,
        labels,
        views,
        candidates,
        split,
        config,
        artifact_dir=tmp_path / "private_models",
    )

    assert result.predictions.groupby(["feature_view", "model_name"]).size().eq(len(windows)).all()
    assert not result.predictions.duplicated(["feature_view", "model_name", "window_key"]).any()
    assert set(result.predictions.group_id) == set(split.development_groups)
    assert "LOCKED" not in set(result.predictions.group_id)
    assert {"dummy_majority", "logistic_regression", "direct_rule_reference"} <= set(result.predictions.model_name)
    assert "relative_time_only" in set(result.predictions.feature_view)
    assert not result.fold_manifest["outer_folds"][0]["train_group_ids"] == result.fold_manifest["outer_folds"][0]["validation_group_ids"]

    logistic_receipt = result.model_receipts.query(
        "feature_view == 'A1_morphology_only' and model_name == 'logistic_regression'"
    ).iloc[0]
    artifact_path = Path(logistic_receipt.model_artifact_path)
    assert artifact_path.is_file()
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == logistic_receipt.model_artifact_sha256
    estimator = joblib.load(artifact_path)
    training_groups = json.loads(logistic_receipt.train_group_ids_json)
    train_mask = windows.group_id.isin(training_groups) & labels.class_id.isin((0, 1))
    expected_mean = float(windows.loc[train_mask, "rms"].mean())
    assert np.isclose(estimator.named_steps["scaler"].mean_[0], expected_mean)

    direct = result.pooled_metrics.query(
        "model_name == 'direct_rule_reference' and confidence_threshold == 0.0"
    ).iloc[0]
    assert direct.accuracy == 1.0
    assert direct.pseudo_label_coverage == 0.75
    assert result.group_metrics.group_id.nunique() == len(split.development_groups)


def test_group_balanced_selection_score_is_invariant_to_overlapping_window_replication():
    score = _group_balanced_macro_f1(
        np.asarray([0, 1, 0, 1]),
        np.asarray([0, 0, 0, 1]),
        ["G01", "G01", "G02", "G02"],
    )
    replicated_score = _group_balanced_macro_f1(
        np.asarray([0, 1] + [0, 1] * 100),
        np.asarray([0, 0] + [0, 1] * 100),
        ["G01", "G01"] + ["G02", "G02"] * 100,
    )

    assert np.isclose(score, replicated_score)


def test_paired_temporal_comparison_bootstraps_groups_not_window_count():
    rows = []
    truth_by_group = {"G01": (0, 1), "G02": (0, 1), "G03": (0, 1)}
    morphology_by_group = {"G01": (0, 0), "G02": (0, 1), "G03": (1, 1)}
    temporal_by_group = {"G01": (0, 1), "G02": (0, 1), "G03": (0, 1)}
    for group_id, truth_values in truth_by_group.items():
        for window_index, (truth, morphology, temporal) in enumerate(
            zip(truth_values, morphology_by_group[group_id], temporal_by_group[group_id], strict=True)
        ):
            key = f"{group_id}:{window_index}"
            for feature_view, model_name, predicted in (
                ("A1_morphology_only", "logistic_regression", morphology),
                ("relative_time_only", "logistic_regression", temporal),
            ):
                rows.append(
                    {
                        "window_key": key,
                        "group_id": group_id,
                        "true_class_id": truth,
                        "predicted_class_id": predicted,
                        "confidence": 1.0,
                        "feature_view": feature_view,
                        "model_name": model_name,
                    }
                )
    original = _paired_group_bootstrap_vs_temporal(pd.DataFrame(rows), replicates=100, seed=71).iloc[0]

    replicated = list(rows)
    for duplicate_index in range(50):
        for window_index, (truth, morphology, temporal) in enumerate(
            zip(truth_by_group["G01"], morphology_by_group["G01"], temporal_by_group["G01"], strict=True)
        ):
            key = f"G01:duplicate:{duplicate_index}:{window_index}"
            for feature_view, model_name, predicted in (
                ("A1_morphology_only", "logistic_regression", morphology),
                ("relative_time_only", "logistic_regression", temporal),
            ):
                replicated.append(
                    {
                        "window_key": key,
                        "group_id": "G01",
                        "true_class_id": truth,
                        "predicted_class_id": predicted,
                        "confidence": 1.0,
                        "feature_view": feature_view,
                        "model_name": model_name,
                    }
                )
    repeated = _paired_group_bootstrap_vs_temporal(pd.DataFrame(replicated), replicates=100, seed=71).iloc[0]

    assert original.source_group_count == repeated.source_group_count == 3
    assert np.isfinite(original.mean_group_accuracy_delta)
    assert np.isclose(original.group_balanced_accuracy_delta, repeated.group_balanced_accuracy_delta)
    assert np.isclose(original.group_balanced_macro_f1_delta, repeated.group_balanced_macro_f1_delta)
    assert np.isclose(original.macro_f1_delta_ci_low, repeated.macro_f1_delta_ci_low)
    assert np.isclose(original.macro_f1_delta_ci_high, repeated.macro_f1_delta_ci_high)


def test_group_balanced_threshold_recovers_positive_predictions_below_half():
    y_true = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    probability_class_1 = np.asarray([0.02, 0.08, 0.21, 0.28, 0.04, 0.11, 0.24, 0.31])
    groups = ["G01"] * 4 + ["G02"] * 4

    result = select_group_balanced_threshold(y_true, probability_class_1, groups)

    assert result.threshold < 0.5
    assert result.group_balanced_macro_f1 == 1.0
    assert result.group_balanced_precision == 1.0
    assert result.group_balanced_recall == 1.0
    assert result.candidate_count >= len(np.unique(probability_class_1))


def test_group_balanced_threshold_ties_prefer_half_then_higher_threshold():
    closest = select_group_balanced_threshold(
        np.asarray([0, 1, 0, 1]),
        np.asarray([0.1, 0.9, 0.1, 0.9]),
        ["G01", "G01", "G02", "G02"],
    )
    higher = select_group_balanced_threshold(
        np.asarray([1, 0, 1, 0]),
        np.asarray([0.0, 0.999, 0.0, 0.999]),
        ["G01", "G01", "G02", "G02"],
    )

    assert closest.threshold == 0.5
    assert closest.tie_resolution == "closest_to_0.5"
    assert higher.threshold == 1.0
    assert higher.tie_resolution == "higher_threshold"


def test_group_balanced_threshold_rejects_missing_class_and_invalid_scores():
    with pytest.raises(ValueError, match="both classes"):
        select_group_balanced_threshold(np.asarray([0, 0]), np.asarray([0.2, 0.4]), ["G01", "G02"])

    with pytest.raises(ValueError, match="finite probabilities"):
        select_group_balanced_threshold(np.asarray([0, 1]), np.asarray([0.2, np.nan]), ["G01", "G02"])


def test_group_balanced_threshold_is_invariant_to_replication_within_a_group():
    y_true = np.asarray([0, 1, 0, 1, 0, 1])
    probability_class_1 = np.asarray([0.2, 0.7, 0.6, 0.3, 0.1, 0.9])
    groups = ["G01", "G01", "G02", "G02", "G03", "G03"]
    original = select_group_balanced_threshold(y_true, probability_class_1, groups)

    repeated = select_group_balanced_threshold(
        np.concatenate((y_true, y_true[:2], y_true[:2])),
        np.concatenate((probability_class_1, probability_class_1[:2], probability_class_1[:2])),
        groups + ["G01"] * 4,
    )

    assert original.threshold == repeated.threshold
    assert np.isclose(original.group_balanced_macro_f1, repeated.group_balanced_macro_f1)
    assert np.isclose(original.group_balanced_precision, repeated.group_balanced_precision)
    assert np.isclose(original.group_balanced_recall, repeated.group_balanced_recall)


def test_group_calibration_splits_are_disjoint_and_retain_both_classes():
    groups = np.repeat(np.asarray([f"G{index:02d}" for index in range(6)]), 2)
    y = np.tile(np.asarray([0, 1]), 6)

    splits = group_calibration_splits(groups, y, seed=19, n_splits=3)

    assert len(splits) == 3
    for train_index, calibration_index in splits:
        assert set(groups[train_index]).isdisjoint(set(groups[calibration_index]))
        assert set(y[train_index]) == {0, 1}
        assert set(y[calibration_index]) == {0, 1}


def test_group_calibration_splits_reject_a_fold_without_positive_group_support():
    groups = np.repeat(np.asarray(["G01", "G02", "G03"]), 2)
    y = np.asarray([0, 0, 0, 0, 1, 1])

    with pytest.raises(ValueError, match="lacks a proxy class"):
        group_calibration_splits(groups, y, seed=23, n_splits=3)


def test_sigmoid_calibration_wraps_full_training_pipeline_and_returns_probabilities():
    rows = []
    labels = []
    groups = []
    for index in range(6):
        group_id = f"G{index:02d}"
        rows.extend(([-1.0 - index * 0.01, 0.0], [1.0 + index * 0.01, 1.0]))
        labels.extend((0, 1))
        groups.extend((group_id, group_id))
    x = pd.DataFrame(rows, columns=["feature_a", "feature_b"])

    estimator, warnings_seen, _ = _fit_estimator(
        "logistic_regression",
        {"C": 1.0},
        x,
        np.asarray(labels),
        groups,
        seed=31,
        calibration_method="sigmoid",
    )
    probabilities = estimator.predict_proba(x.iloc[:4])

    assert estimator.__class__.__name__ == "CalibratedClassifierCV"
    assert estimator.ensemble is False
    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert all("CalibratedClassifierCV" not in item for item in warnings_seen)


def _inner_oof_selection_fixture(*, sparse_positive_groups: bool = False):
    groups = tuple(f"G{index:02d}" for index in range(12))
    train_groups = groups[:9]
    inner_folds = tuple(
        GroupFold(
            fold_id=index + 1,
            train_group_ids=tuple(group for group in train_groups if group not in train_groups[index * 3 : (index + 1) * 3]),
            validation_group_ids=train_groups[index * 3 : (index + 1) * 3],
        )
        for index in range(3)
    )
    positive_groups = {"G00", "G03", "G06"} if sparse_positive_groups else set(train_groups)
    rows = []
    for group_index, group_id in enumerate(groups):
        recording_id = f"{group_id}_R1"
        for class_id in (0, 1):
            value = (-1.0 if class_id == 0 else 1.0) + group_index * 0.02
            # Keep the split support intentionally sparse for the calibration failure fixture.
            if sparse_positive_groups and group_id not in positive_groups and class_id == 1:
                continue
            rows.append(
                {
                    "window_key": f"{group_id}:{class_id}",
                    "group_id": group_id,
                    "recording_id": recording_id,
                    "class_id": class_id,
                    "feature": value,
                }
            )
    return pd.DataFrame(rows), inner_folds, groups


def test_inner_oof_predictions_are_exactly_once_and_group_disjoint(monkeypatch):
    import emg_fatigue.weak_supervision.evaluation as evaluation

    frame, inner_folds, groups = _inner_oof_selection_fixture()
    fitting_groups: list[set[str]] = []
    calibration_receipts: list[tuple[set[str], set[str]]] = []
    original_fit = evaluation._fit_estimator
    original_splits = evaluation.group_calibration_splits

    def recording_fit(model_name, params, x, y, groups, seed, calibration_method=None):
        fitting_groups.append(set(str(group) for group in groups))
        return original_fit(model_name, params, x, y, groups, seed, calibration_method)

    def recording_splits(group_ids, y, seed, n_splits=3):
        splits = original_splits(group_ids, y, seed, n_splits=n_splits)
        group_array = np.asarray(group_ids, dtype=object)
        fit_groups = set(str(group) for group in group_array)
        calibration_receipts.extend(
            (fit_groups, set(str(group) for group in group_array[calibration]))
            for _, calibration in splits
        )
        return splits

    monkeypatch.setattr(evaluation, "_fit_estimator", recording_fit)
    monkeypatch.setattr(evaluation, "group_calibration_splits", recording_splits)
    result = _inner_oof_candidate_predictions(
        frame[frame.group_id.isin(groups[:9])],
        ("feature",),
        inner_folds,
        "logistic_regression",
        {"C": 1.0},
        "sigmoid",
        901,
    )

    assert result.window_key.is_unique
    assert set(result.group_id) == set(groups[:9])
    assert len(result) == 18
    assert result.groupby("group_id").size().eq(2).all()
    assert len(fitting_groups) == len(inner_folds)
    for fold, observed_train in zip(inner_folds, fitting_groups, strict=True):
        assert observed_train == set(fold.train_group_ids)
        assert observed_train.isdisjoint(fold.validation_group_ids)
    assert calibration_receipts
    assigned_groups = set(groups[:9])
    for fit_group_set, calibration_group_set in calibration_receipts:
        held_out_groups = assigned_groups - fit_group_set
        assert fit_group_set.isdisjoint(held_out_groups)
        assert calibration_group_set.issubset(fit_group_set)
        assert calibration_group_set.isdisjoint(held_out_groups)


def test_inner_oof_rejects_outer_validation_groups_in_candidate_frame():
    frame, inner_folds, _ = _inner_oof_selection_fixture()

    with pytest.raises(ValueError, match="exactly partition"):
        _inner_oof_candidate_predictions(
            frame,
            ("feature",),
            inner_folds,
            "logistic_regression",
            {"C": 1.0},
            "none",
            902,
        )


def test_candidate_selection_records_unsupported_group_calibration_and_keeps_uncalibrated_candidate():
    frame, inner_folds, groups = _inner_oof_selection_fixture(sparse_positive_groups=True)
    frame = frame[frame.group_id.isin(groups[:9])]

    result = select_candidate_and_threshold(
        "logistic_regression",
        [{"C": 1.0}],
        frame,
        ("feature",),
        inner_folds,
        903,
    )

    failed = [item for item in result.candidate_receipts if item["calibration_method"] == "sigmoid"]
    viable = [item for item in result.candidate_receipts if item["calibration_method"] == "none"]
    assert failed and all(item["fit_status"] == "unsupported" for item in failed)
    assert all("lacks a proxy class" in " ".join(item["errors"]) for item in failed)
    assert viable and viable[0]["fit_status"] == "completed"
    assert result.calibration_method == "none"
    assert pd.DataFrame(result.inner_oof_predictions).group_id.nunique() == 9


def test_candidate_selection_uses_sigmoid_probabilities_when_group_support_is_sufficient():
    frame, inner_folds, groups = _inner_oof_selection_fixture()
    frame = frame[frame.group_id.isin(groups[:9])]

    result = select_candidate_and_threshold(
        "logistic_regression",
        [{"C": 1.0}],
        frame,
        ("feature",),
        inner_folds,
        904,
    )

    assert result.calibration_method == "sigmoid"
    uncalibrated = [item for item in result.candidate_receipts if item["calibration_method"] == "none"]
    calibrated = [item for item in result.candidate_receipts if item["calibration_method"] == "sigmoid"]
    assert uncalibrated and all(not item["eligible_for_selection"] for item in uncalibrated)
    assert calibrated and all(item["eligible_for_selection"] for item in calibrated)
