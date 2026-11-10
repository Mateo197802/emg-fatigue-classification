from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from emg_fatigue.weak_supervision.contracts import GroupSplit
from emg_fatigue.weak_supervision.evaluation import nested_group_predictions


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
