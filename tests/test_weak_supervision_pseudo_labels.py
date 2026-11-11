from __future__ import annotations

import importlib

import numpy as np
import pytest

from emg_fatigue.weak_supervision.contracts import SignalWindow


def _api():
    try:
        features = importlib.import_module("emg_fatigue.weak_supervision.features")
        labels = importlib.import_module("emg_fatigue.weak_supervision.pseudo_labels")
    except ImportError:
        pytest.fail("technical-reference pseudo-label engine is not implemented", pytrace=False)
    return features, labels


def _window(start_s: float, index: int, *, duration_s: float = 0.5, eligible: bool = True) -> SignalWindow:
    fs_hz = 1000.0
    width = int(duration_s * fs_hz)
    return SignalWindow(
        group_id="G01",
        recording_id="G01_F001",
        segment_index=1,
        window_index=index,
        start_sample=int(round(start_s * fs_hz)),
        start_time_s=start_s,
        duration_s=duration_s,
        sampling_rate_hz=fs_hz,
        signal=np.zeros(width, dtype=np.float32),
        activity_occupancy=1.0 if eligible else 0.0,
        eligible_for_label=eligible,
        rejection_reason=None if eligible else "insufficient_activity_occupancy",
    )


def _indicator_row(*, fatigue_like: bool = False) -> dict[str, float | int | str]:
    if fatigue_like:
        return {
            "feature_valid": 1,
            "feature_status": "valid",
            "median_frequency_hz": 95.0,
            "mean_frequency_hz": 115.0,
            "rms": 1.1,
            "line_excluded_low_high_log_ratio": 0.1,
            "line_excluded_low_frequency_fraction_20_80": 0.55,
        }
    return {
        "feature_valid": 1,
        "feature_status": "valid",
        "median_frequency_hz": 100.0,
        "mean_frequency_hz": 120.0,
        "rms": 1.0,
        "line_excluded_low_high_log_ratio": 0.0,
        "line_excluded_low_frequency_fraction_20_80": 0.5,
    }


def _label_config() -> dict:
    return {
        "run": {"primary_window_s": 0.5},
        "labels": {
            "version": "spectral_persistence_v1",
            "reference_window_count": 5,
            "reference_instability_percent": 20.0,
            "minimum_mdf_mnf_drop_hz": 4.0,
            "minimum_scaled_mad_multiple": 3.0,
            "minimum_log_ratio_increase_when_mad_zero": 0.04879016416943205,
            "minimum_low_frequency_fraction_increase_when_mad_zero": 0.05,
            "entry_window_count": 3,
            "entry_rolling_window_count": 4,
            "recovery_consecutive_window_count": 3,
            "minimum_entry_span_s": 1.25,
        }
    }


def test_technical_reference_uses_first_five_valid_nonoverlapping_windows():
    features, labels = _api()
    windows = [_window(i * 0.5, i) for i in range(6)]
    feature_map = {features.window_key(window): _indicator_row() for window in windows}
    windows[1] = _window(0.5, 1, eligible=False)

    reference = labels.technical_reference(windows, feature_map, _label_config())

    assert reference.available is True
    assert reference.stable is True
    assert reference.window_count == 5
    assert reference.reference_window_keys[0] == features.window_key(windows[0])
    assert features.window_key(windows[1]) not in reference.reference_window_keys


def test_insufficient_or_unstable_reference_blocks_all_supervised_labels():
    features, labels = _api()
    windows = [_window(i * 0.5, i) for i in range(4)]
    feature_map = {features.window_key(window): _indicator_row() for window in windows}
    insufficient = labels.technical_reference(windows, feature_map, _label_config())
    assert insufficient.available is False
    assert insufficient.status == "insufficient_reference_windows"

    windows = [_window(i * 0.5, i) for i in range(5)]
    unstable_values = [70.0, 85.0, 100.0, 115.0, 130.0]
    feature_map = {}
    for window, mdf in zip(windows, unstable_values, strict=True):
        row = _indicator_row()
        row["median_frequency_hz"] = mdf
        feature_map[features.window_key(window)] = row
    unstable = labels.technical_reference(windows, feature_map, _label_config())
    assert unstable.available is True
    assert unstable.stable is False
    assert "unstable_mdf_reference" in unstable.reasons


def test_persistent_entry_and_recovery_follow_exact_nonretroactive_state_rules():
    features, labels = _api()
    reference_windows = [_window(i * 0.5, i) for i in range(5)]
    reference_map = {features.window_key(window): _indicator_row() for window in reference_windows}
    reference = labels.technical_reference(reference_windows, reference_map, _label_config())
    windows = reference_windows.copy()
    windows.append(_window(2.25, 5))
    observations = {}
    # Three positive indicators within the last four eligible windows, spanning 1.25 s.
    states = [True, True, False, True, False, False, False]
    first_start = 2.5
    for index, fatigue in enumerate(states):
        window = _window(first_start + 0.25 * index, index + 6)
        windows.append(window)
        observations[features.window_key(window)] = _indicator_row(fatigue_like=fatigue)
    feature_map = {**reference_map, **observations}

    assigned = labels.label_recording(windows, feature_map, reference, _label_config())
    by_key = {item.window_key: item for item in assigned}
    post_reference = windows[6:]

    assert all(by_key[features.window_key(window)].class_id == -1 for window in reference_windows)
    assert by_key[features.window_key(windows[5])].label_reason == "technical_reference_period"
    assert by_key[features.window_key(post_reference[0])].class_id == -1
    assert by_key[features.window_key(post_reference[1])].class_id == -1
    assert by_key[features.window_key(post_reference[2])].class_id == 0
    assert by_key[features.window_key(post_reference[3])].class_id == 1
    assert by_key[features.window_key(post_reference[3])].state_latched is True
    assert by_key[features.window_key(post_reference[4])].class_id == -1
    assert by_key[features.window_key(post_reference[5])].class_id == -1
    assert by_key[features.window_key(post_reference[6])].class_id == 0
    assert by_key[features.window_key(post_reference[6])].state_latched is False


def test_quality_invalid_windows_abstain_and_reset_persistence_counter():
    features, labels = _api()
    reference_windows = [_window(i * 0.5, i) for i in range(5)]
    reference_map = {features.window_key(window): _indicator_row() for window in reference_windows}
    reference = labels.technical_reference(reference_windows, reference_map, _label_config())
    windows = reference_windows.copy()
    feature_map = dict(reference_map)
    for index in range(4):
        window = _window(2.25 + 0.25 * index, index + 5, eligible=index != 2)
        windows.append(window)
        if window.eligible_for_label:
            feature_map[features.window_key(window)] = _indicator_row(fatigue_like=True)

    assigned = labels.label_recording(windows, feature_map, reference, _label_config())

    assert all(item.class_id == -1 for item in assigned[-4:])
    assert assigned[-2].label_reason == "insufficient_activity_occupancy"


def test_indicator_values_exactly_on_primary_threshold_do_not_trigger_entry():
    features, labels = _api()
    reference_windows = [_window(i * 0.5, i) for i in range(5)]
    feature_map = {features.window_key(window): _indicator_row() for window in reference_windows}
    reference = labels.technical_reference(reference_windows, feature_map, _label_config())
    window = _window(2.5, 5)
    threshold_row = _indicator_row()
    threshold_row["median_frequency_hz"] = 96.0
    threshold_row["mean_frequency_hz"] = 116.0
    threshold_row["line_excluded_low_high_log_ratio"] = 0.04879016416943205
    feature_map[features.window_key(window)] = threshold_row

    assigned = labels.label_recording(reference_windows + [window], feature_map, reference, _label_config())

    assert assigned[-1].class_id == 0
    assert assigned[-1].label_reason == "within_reference_envelope"


def test_low_frequency_alternative_is_independent_of_mdf_mnf_values():
    features, labels = _api()
    reference_windows = [_window(i * 0.5, i) for i in range(5)]
    feature_map = {}
    for index, window in enumerate(reference_windows):
        row = _indicator_row()
        row["median_frequency_hz"] = 50.0 + 30.0 * index
        feature_map[features.window_key(window)] = row
    reference = labels.technical_reference(
        reference_windows,
        feature_map,
        _label_config(),
        stability_components=("low_fraction",),
    )
    assert reference.stable is True
    windows = reference_windows + [_window(2.25, 5)]
    for index in range(7):
        window = _window(2.5 + 0.25 * index, index + 6)
        windows.append(window)
        row = _indicator_row()
        row["median_frequency_hz"] = 200.0 - 20.0 * index
        row["mean_frequency_hz"] = 350.0 - 10.0 * index
        if index in (0, 1, 3):
            row["line_excluded_low_frequency_fraction_20_80"] = 0.60
        feature_map[features.window_key(window)] = row

    assigned = labels.label_low_frequency_fraction_rule(windows, feature_map, reference, _label_config())

    assert assigned[-4].class_id == 1
    assert assigned[-4].rule_version == "low_frequency_fraction_persistence_v1"


def test_log_mdf_slope_alternative_requires_both_frequency_trend_and_low_band_shift():
    features, labels = _api()
    reference_windows = [_window(i * 0.5, i) for i in range(5)]
    reference_map = {features.window_key(window): _indicator_row() for window in reference_windows}
    reference = labels.technical_reference(
        reference_windows,
        reference_map,
        _label_config(),
        stability_components=("mdf", "low_fraction"),
    )
    windows = reference_windows + [_window(2.25, 5)]
    feature_map = dict(reference_map)
    observations = []
    for index, (mdf, low_fraction) in enumerate(
        ((100.0, 0.5), (95.0, 0.6), (95.0, 0.6), (95.0, 0.6), (100.0, 0.5), (100.0, 0.5), (100.0, 0.5))
    ):
        window = _window(2.5 + 0.25 * index, index + 6)
        windows.append(window)
        row = _indicator_row()
        row["median_frequency_hz"] = mdf
        row["line_excluded_low_frequency_fraction_20_80"] = low_fraction
        feature_map[features.window_key(window)] = row
        observations.append(window)

    assigned = labels.label_log_mdf_slope_rule(windows, feature_map, reference, _label_config())
    by_key = {item.window_key: item for item in assigned}

    assert by_key[features.window_key(observations[3])].class_id == 1
    assert by_key[features.window_key(observations[3])].rule_version == "log_mdf_slope_low_fraction_v1"
