import numpy as np
import pandas as pd
import pytest


def test_equalized_training_sample_is_capped_per_participant_trial_and_class():
    from emg_fatigue.perceived_benchmark import equalized_training_sample

    frame = pd.DataFrame(
        [
            {
                "subject_id": subject,
                "trial_id": trial,
                "label": label,
                    "window_id": f"{subject}_{trial}_{label}_{index}",
                "window_start_s": float(index),
                "x": float(index),
            }
            for subject in ("s1", "s2")
            for trial in ("t1", "t2")
            for label in (0, 1)
            for index in range(10)
        ]
    )

    result = equalized_training_sample(frame, max_per_trial_class=3, seed=11)

    counts = result.groupby(["subject_id", "trial_id", "label"]).size()
    assert set(counts) == {3}
    assert set(result["label"]) == {0, 1}
    assert result.equals(equalized_training_sample(frame, max_per_trial_class=3, seed=11))


def test_binary_metrics_report_fatigue_sensitivity_and_specificity():
    from emg_fatigue.perceived_benchmark import binary_metric_summary

    result = binary_metric_summary(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.8, 0.9, 0.2]),
    )

    assert result["accuracy"] == pytest.approx(0.5)
    assert result["f1_macro"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["sensitivity_fatigue"] == pytest.approx(0.5)
    assert result["specificity_no_fatigue"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(0.75)


def test_outer_splits_leave_each_participant_out_once():
    from emg_fatigue.perceived_benchmark import leave_one_participant_out

    frame = pd.DataFrame(
        {"subject_id": np.repeat(["s1", "s2", "s3", "s4"], 4), "label": [0, 0, 1, 1] * 4}
    )
    splits = list(leave_one_participant_out(frame))

    assert len(splits) == 4
    assert {split["test_subject"] for split in splits} == {"s1", "s2", "s3", "s4"}
    for split in splits:
        assert split["test_subject"] not in split["train_subjects"]
