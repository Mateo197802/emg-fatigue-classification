import pandas as pd
import pytest


def test_paired_metric_contrast_bootstraps_matched_groups_deterministically():
    from emg_fatigue.evaluation import paired_metric_contrast

    results = pd.DataFrame(
        [
            {"model": "base", "fold_id": "g1", "accuracy": 0.50},
            {"model": "base", "fold_id": "g2", "accuracy": 0.50},
            {"model": "base", "fold_id": "g3", "accuracy": 0.50},
            {"model": "candidate", "fold_id": "g1", "accuracy": 0.60},
            {"model": "candidate", "fold_id": "g2", "accuracy": 0.40},
            {"model": "candidate", "fold_id": "g3", "accuracy": 0.50},
        ]
    )

    first = paired_metric_contrast(
        results,
        candidate="candidate",
        baseline="base",
        metric="accuracy",
        bootstrap_replicates=1000,
        seed=17,
    )
    second = paired_metric_contrast(
        results,
        candidate="candidate",
        baseline="base",
        metric="accuracy",
        bootstrap_replicates=1000,
        seed=17,
    )

    assert first == second
    assert first["mean_difference"] == pytest.approx(0.0)
    assert first["n_paired_groups"] == 3
    assert first["groups_improved"] == 1
    assert first["groups_worse"] == 1
    assert first["groups_equal"] == 1
    assert first["ci95_lower"] < 0 < first["ci95_upper"]


def test_paired_metric_contrast_rejects_unmatched_or_duplicate_groups():
    from emg_fatigue.evaluation import paired_metric_contrast

    mismatched = pd.DataFrame(
        [
            {"model": "base", "fold_id": "g1", "accuracy": 0.5},
            {"model": "base", "fold_id": "g2", "accuracy": 0.5},
            {"model": "candidate", "fold_id": "g1", "accuracy": 0.6},
            {"model": "candidate", "fold_id": "g3", "accuracy": 0.6},
        ]
    )
    with pytest.raises(ValueError, match="identical group support"):
        paired_metric_contrast(mismatched, candidate="candidate", baseline="base")

    duplicated = pd.concat([mismatched, mismatched.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one metric row per model and group"):
        paired_metric_contrast(duplicated, candidate="candidate", baseline="base")
