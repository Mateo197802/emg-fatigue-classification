from __future__ import annotations

import importlib

import numpy as np
import pytest


def _feature_api():
    try:
        return importlib.import_module("emg_fatigue.weak_supervision.features")
    except ImportError:
        pytest.fail("fresh weak-supervision feature extraction is not implemented", pytrace=False)


def test_temporal_and_spectral_features_recover_known_sine_properties():
    features = _feature_api()
    fs_hz = 1000.0
    amplitude = 2.0
    time_s = np.arange(500, dtype=np.float64) / fs_hz
    x = amplitude * np.sin(2 * np.pi * 50 * time_s)

    row = features.extract_window_features(x, fs_hz)

    assert row["rms"] == pytest.approx(amplitude / np.sqrt(2), abs=0.01)
    assert row["mav"] == pytest.approx(2 * amplitude / np.pi, abs=0.02)
    assert row["iemg"] == pytest.approx((2 * amplitude / np.pi) * 0.5, abs=0.01)
    assert row["mean_frequency_hz"] == pytest.approx(50.0, abs=2.0)
    assert row["median_frequency_hz"] == pytest.approx(50.0, abs=2.0)
    assert row["feature_valid"] == 1
    assert sum(row[f"haar_detail_energy_ratio_d{level}"] for level in range(1, 6)) + row[
        "haar_approx_energy_ratio_a5"
    ] == pytest.approx(1.0, abs=1e-10)


def test_line_exclusion_changes_only_requested_low_high_ratio():
    features = _feature_api()
    fs_hz = 1000.0
    time_s = np.arange(1000, dtype=np.float64) / fs_hz
    x = np.sin(2 * np.pi * 60 * time_s) + 0.5 * np.sin(2 * np.pi * 100 * time_s)

    with_line = features.extract_window_features(x, fs_hz, line_frequency_hz=None)
    line_excluded = features.extract_window_features(x, fs_hz, line_frequency_hz=60.0)

    assert with_line["line_excluded_low_high_log_ratio"] != pytest.approx(
        line_excluded["line_excluded_low_high_log_ratio"], abs=1e-3
    )
    assert with_line["median_frequency_hz"] == line_excluded["median_frequency_hz"]


def test_zero_energy_window_is_explicit_and_never_emits_fake_spectral_values():
    features = _feature_api()

    row = features.extract_window_features(np.zeros(500, dtype=np.float64), 1000.0)

    assert row["feature_valid"] == 0
    assert row["feature_status"] == "zero_energy"
    assert np.isnan(row["median_frequency_hz"])
    assert np.isnan(row["mean_frequency_hz"])
    assert np.isnan(row["line_excluded_low_high_log_ratio"])
    assert row["rms"] == 0.0


def test_primary_feature_view_excludes_every_label_defining_spectral_indicator():
    features = _feature_api()
    schema = features.feature_schema()
    primary = set(features.FEATURE_VIEWS["A1_morphology_only"])
    label_indicators = {row["name"] for row in schema if row["label_defining"]}

    assert label_indicators.isdisjoint(primary)
    assert {"median_frequency_hz", "mean_frequency_hz", "line_excluded_low_high_log_ratio"}.issubset(
        label_indicators
    )
