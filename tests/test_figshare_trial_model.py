import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat


def test_condition_label_is_trial_outcome_not_per_window_fatigue():
    from experiments.reference_fatigue.figshare_trial_model import condition_task_label

    assert condition_task_label("30") == 0
    assert condition_task_label("TLim") == 1
    with pytest.raises(ValueError):
        condition_task_label("MVC")


def test_window_metrics_are_pooled_to_one_trial_feature_row():
    from experiments.reference_fatigue.figshare_trial_model import summarize_window_metrics

    windows = {
        "rms_log10": np.array([1.0, 2.0, 3.0]),
        "mdf_hz": np.array([100.0, 90.0, 80.0]),
    }
    result = summarize_window_metrics(windows, np.array([0.0, 0.5, 1.0]))

    assert len(result) == 6
    assert result["rms_log10__median"] == 2.0
    assert result["rms_log10__iqr"] == 1.0
    assert result["rms_log10__slope"] == pytest.approx(2.0)
    assert result["mdf_hz__slope"] == pytest.approx(-20.0)


def test_outer_splits_hold_out_complete_participant_pairs():
    from experiments.reference_fatigue.figshare_trial_model import outer_leave_one_subject_out

    frame = pd.DataFrame(
        [
            {"subject_id": f"S{i}", "condition": condition, "label": label}
            for i in range(1, 5)
            for condition, label in (("30", 0), ("TLim", 1))
        ]
    )
    splits = list(outer_leave_one_subject_out(frame))

    assert len(splits) == 4
    for split in splits:
        test = frame.iloc[split["test_indices"]]
        train = frame.iloc[split["train_indices"]]
        assert test["subject_id"].nunique() == 1
        assert test["subject_id"].iloc[0] not in set(train["subject_id"])
        assert sorted(test["label"].tolist()) == [0, 1]


def test_multiple_window_logits_pool_to_single_trial_score():
    from experiments.reference_fatigue.figshare_trial_model import mil_mean_logit

    assert mil_mean_logit(np.array([-2.0, 0.0, 2.0])) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        mil_mean_logit(np.array([]))


def test_trial_level_metrics_confusion_counts_are_not_window_counts():
    from experiments.reference_fatigue.figshare_trial_benchmark import metric_summary

    result = metric_summary(
        np.array([0, 1, 0, 1]),
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
    )

    assert result["n_trials"] == 4
    assert result["tn"] == 1 and result["fp"] == 1
    assert result["fn"] == 1 and result["tp"] == 1
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["f1_macro"] == pytest.approx(0.5)


def test_participant_bootstrap_resamples_complete_two_trial_groups():
    from experiments.reference_fatigue.figshare_trial_benchmark import _bootstrap_intervals

    rows = []
    for subject, predictions in (("S1", (0, 1)), ("S2", (0, 1)), ("S3", (1, 1)), ("S4", (0, 0))):
        for label, prediction in enumerate(predictions):
            rows.append({
                "model": "toy",
                "subject_id": subject,
                "y_true": label,
                "y_pred": prediction,
                "score": float(prediction),
            })
    result = _bootstrap_intervals(pd.DataFrame(rows), replicates=100, seed=3)

    assert result.loc[0, "participant_count"] == 4
    assert 0.0 <= result.loc[0, "accuracy_ci95_low"] <= result.loc[0, "accuracy_ci95_high"] <= 1.0
    assert 0.0 <= result.loc[0, "f1_macro_ci95_low"] <= result.loc[0, "f1_macro_ci95_high"] <= 1.0


def test_process_trial_file_keeps_one_trial_target_and_excludes_short_record_parts(tmp_path):
    from experiments.reference_fatigue.figshare_trial_model import process_trial_file

    rng = np.random.default_rng(17)
    fs = 1000.0
    n_samples = 25_000
    time = np.arange(n_samples) / fs
    force = np.full(n_samples, -1.0)
    force[2_000:13_000] = 10.0
    sig = np.empty((13, 5), dtype=object)
    for index in np.ndindex(sig.shape):
        sig[index] = np.empty(0)
    for channel in range(32):
        sig.flat[channel] = (
            0.2 * np.sin(2 * np.pi * (70 + channel) * time)
            + 0.1 * np.sin(2 * np.pi * 130 * time)
            + 0.02 * rng.standard_normal(n_samples)
        )[None, :]
    discard = np.zeros((13, 5), dtype=np.uint8)
    mat_path = tmp_path / "Sx_30_RF_edited_final.mat"
    savemat(
        mat_path,
        {
            "SIG": sig,
            "discardChannelsVec": discard,
            "ref_signal": force[None, :],
            "fsamp": np.array([[fs]]),
        },
    )

    result = process_trial_file(mat_path, "Sx", "30")

    assert result["trial_row"]["label"] == 0
    assert result["trial_row"]["eligible_window_count"] > 2
    assert result["trial_row"]["valid_channel_count"] == 32
    assert result["raw_bag"].shape[1:] == (65, 500)
    assert "window_label" not in result["trial_row"]
