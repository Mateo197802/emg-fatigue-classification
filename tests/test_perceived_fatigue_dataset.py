import numpy as np
import pandas as pd
import pytest
import io
import zipfile


def test_parse_measurement_frame_uses_raw_voltage_and_canonical_muscles():
    from emg_fatigue.perceived_dataset import parse_measurement_frame

    frame = pd.DataFrame(
        {
            "X [s]": [0.0, 0.001, 0.002],
            "R BICEPS BRACHII: EMG 1 [V]": [0.1, 0.2, 0.3],
            "X [s].1": [0.0, 0.001, 0.002],
            "R DELTOID ANTERIOR: EMG 2 [V]": [0.4, 0.5, 0.6],
            "X [s].2": [0.0, 0.001, 0.002],
            "R DELTOID MEDIUS: EMG 6 [V]": [0.7, 0.8, 0.9],
            "X [s].3": [0.0, 0.001, 0.002],
            "R DELTOID POSTERIOR: EMG 7 [V]": [1.0, 1.1, 1.2],
        }
    )

    time, channels, metadata = parse_measurement_frame(frame)

    assert time.tolist() == [0.0, 0.001, 0.002]
    assert list(channels) == ["biceps", "deltoid_anterior", "deltoid_medius", "deltoid_posterior"]
    assert channels["biceps"].tolist() == [0.1, 0.2, 0.3]
    assert metadata["units"] == "V"
    assert metadata["channel_time_axes_aligned"] is True


def test_parse_measurement_frame_drops_only_a_zeroed_terminal_clock_reset():
    from emg_fatigue.perceived_dataset import parse_measurement_frame

    frame = pd.DataFrame(
        {
            "X [s]": [0.0, 0.001, 0.002, 0.0],
            "R BICEPS BRACHII: EMG 1 [V]": [0.1, 0.2, 0.3, 0.0],
            "X [s].1": [0.0, 0.001, 0.002, 0.003],
            "R DELTOID ANTERIOR: EMG 2 [V]": [0.4, 0.5, 0.6, 0.0],
            "X [s].2": [0.0, 0.001, 0.002, 0.003],
            "R DELTOID MEDIUS: EMG 6 [V]": [0.7, 0.8, 0.9, 1.0],
            "X [s].3": [0.0, 0.001, 0.002, 0.003],
            "R DELTOID POSTERIOR: EMG 7 [V]": [1.0, 1.1, 1.2, 1.3],
        }
    )

    time, channels, metadata = parse_measurement_frame(frame)

    assert time.tolist() == [0.0, 0.001, 0.002]
    assert all(len(values) == 3 for values in channels.values())
    assert metadata["terminal_reset_rows_dropped"] == 1

    frame.iloc[-1, 1] = 0.25
    with pytest.raises(ValueError, match="ambiguous terminal clock reset"):
        parse_measurement_frame(frame)


def test_terminal_backward_label_row_is_dropped_only_when_duplicate_state_is_clear():
    from emg_fatigue.perceived_dataset import clean_label_timeline

    time, level, receipt = clean_label_timeline(
        [0.02, 0.04, 0.06, 0.08, 0.06], [0, 0, 1, 2, 2]
    )

    assert time.tolist() == [0.02, 0.04, 0.06, 0.08]
    assert level.tolist() == [0, 0, 1, 2]
    assert receipt["terminal_backward_duplicate_dropped"] is True
    assert receipt["rows_dropped"] == 1

    with pytest.raises(ValueError, match="ambiguous nonmonotonic"):
        clean_label_timeline([0.02, 0.04, 0.08, 0.06, 0.10], [0, 0, 1, 2, 2])


def test_window_label_requires_coverage_and_one_stable_reference_state():
    from emg_fatigue.perceived_dataset import label_window

    times = np.arange(0.02, 1.001, 0.02)
    labels = np.zeros(len(times), dtype=int)
    labels[times >= 0.50] = 1
    labels[times >= 0.70] = 2

    assert label_window(0.0, 0.48, times, labels, initial_state=0)[0] == 0
    assert label_window(0.30, 0.80, times, labels, initial_state=0)[1] == "mixed_reference_state"
    assert label_window(0.50, 0.68, times, labels, initial_state=0)[1] == "transition_state_excluded"
    assert label_window(0.70, 0.90, times, labels, initial_state=0)[0] == 1
    assert label_window(1.10, 1.30, times, labels, initial_state=0)[1] == "outside_reference_support"


def test_perceived_feature_formulas_on_known_sinusoid():
    from emg_fatigue.perceived_dataset import extract_perceived_features

    fs = 1000.0
    time = np.arange(500) / fs
    signal = 2.0 * np.sin(2 * np.pi * 100 * time)

    features = extract_perceived_features(signal, fs)

    assert features["rms"] == pytest.approx(np.sqrt(2.0), rel=0.02)
    centered = signal - signal.mean()
    expected_mav = np.mean(np.abs(centered))
    assert features["mav"] == pytest.approx(expected_mav, rel=1e-12)
    assert features["iemg_v_s"] == pytest.approx(expected_mav * len(signal) / fs, rel=1e-12)
    assert features["mdf_hz"] == pytest.approx(100.0, abs=2.1)
    assert features["mnf_hz"] == pytest.approx(100.0, abs=2.1)
    assert 0.0 <= features["spectral_entropy_norm"] <= 1.0
    assert all(np.isfinite(list(features.values())))


def test_nested_archive_reader_matches_subject_and_trial_not_zip_order(tmp_path):
    from emg_fatigue.perceived_dataset import iter_perceived_trials

    def nested_zip(files):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, contents in files.items():
                archive.writestr(name, contents)
        return buffer.getvalue()

    raw_csv = (
        "X [s],R BICEPS BRACHII: EMG 1 [V],X [s].1,R DELTOID ANTERIOR: EMG 2 [V],"
        "X [s].2,R DELTOID MEDIUS: EMG 6 [V],X [s].3,R DELTOID POSTERIOR: EMG 7 [V]\n"
        "0,0.1,0,0.2,0,0.3,0,0.4\n"
        "0.001,0.2,0.001,0.3,0.001,0.4,0.001,0.5\n"
    )
    label_csv = "time,label\n0.02,0\n0.04,0\n"
    raw_subject = nested_zip({"subject_1/trial_1.csv": raw_csv})
    label_subject = nested_zip({"subject_1/Trial_1.csv": label_csv})
    emg_archive = tmp_path / "raw.zip"
    label_archive = tmp_path / "labels.zip"
    with zipfile.ZipFile(emg_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sEMG_data/subject_1.zip", raw_subject)
    with zipfile.ZipFile(label_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("self_perceived_fatigue_index/subject_1.zip", label_subject)

    trial = next(iter_perceived_trials(emg_archive, label_archive, subject_ids=(1,), trial_ids=(1,)))

    assert (trial.subject_id, trial.trial_id) == ("subject_1", "trial_1")
    assert trial.raw_member == "subject_1/trial_1.csv"
    assert trial.label_member == "subject_1/Trial_1.csv"
    assert trial.signal_metadata["units"] == "V"
    assert trial.label_cleanup["estimated_rate_hz"] == pytest.approx(50.0)


def _synthetic_perceived_trial(*, include_gap=False):
    from emg_fatigue.perceived_dataset import PerceivedFatigueTrial

    fs = 1000.0
    time = np.arange(0, 6, 1 / fs)
    if include_gap:
        time[3000:] += 1.0
    channels = {
        name: 1e-3 * np.sin(2 * np.pi * (80 + index * 20) * time)
        for index, name in enumerate(
            ("biceps", "deltoid_anterior", "deltoid_medius", "deltoid_posterior")
        )
    }
    label_time = np.arange(0.02, 6.001, 0.02)
    level = np.zeros(len(label_time), dtype=int)
    level[label_time >= 2.5] = 1
    level[label_time >= 3.5] = 2
    return PerceivedFatigueTrial(
        subject_id="subject_1",
        trial_id="trial_1",
        time_s=time,
        channels_v=channels,
        label_time_s=label_time,
        perceived_level=level,
        raw_member="subject_1/trial_1.csv",
        label_member="subject_1/Trial_1.csv",
        raw_member_crc32="00000000",
        raw_member_bytes=1,
        label_member_crc32="00000000",
        label_member_bytes=1,
        signal_metadata={"units": "V", "channel_time_axes_aligned": True},
        label_cleanup={"rows_dropped": 0},
    )


def test_real_trial_preprocessing_emits_stable_window_labels_and_four_channel_features():
    from emg_fatigue.perceived_preprocessing import process_perceived_trial

    result = process_perceived_trial(_synthetic_perceived_trial())

    assert result.window_samples == 500
    assert result.hop_samples == 250
    assert result.estimated_fs_hz == pytest.approx(1000.0, rel=1e-3)
    assert len(result.feature_rows) > 0
    assert set(row["label"] for row in result.feature_rows) == {0, 1}
    assert all("time_fraction" not in row for row in result.feature_rows)
    assert all("subject_id" not in row for row in result.feature_rows[0]["features"])
    assert len(result.feature_rows[0]["features"]) == 4 * 15
    assert result.exclusion_counts["transition_state_excluded"] > 0
    assert result.exclusion_counts["mixed_reference_state"] > 0


def test_real_trial_preprocessing_splits_at_acquisition_gap():
    from emg_fatigue.perceived_preprocessing import process_perceived_trial

    result = process_perceived_trial(_synthetic_perceived_trial(include_gap=True))

    assert result.gap_count == 1
    assert all(
        not (row["window_start_s"] < 4.0 and row["window_end_s"] > 3.0)
        for row in result.feature_rows
    )
