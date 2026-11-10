import numpy as np
import pytest


def test_candidate_feature_formulas_have_declared_units_and_are_not_primary_schema():
    from emg_fatigue.candidate_features import (
        CANDIDATE_FEATURE_SCHEMA,
        candidate_window_features,
    )
    from emg_fatigue.features import FEATURE_COLUMNS, FEATURE_COLUMNS_V2

    fs = 1000.0
    t = np.arange(1000) / fs
    x = 2 * np.sin(2 * np.pi * 125 * t)
    result = candidate_window_features(x, fs)

    assert CANDIDATE_FEATURE_SCHEMA == "emg_candidate_features_v1"
    assert result["zero_crossings_per_s"] == pytest.approx(250.0, abs=0.1)
    assert result["slope_sign_changes_per_s"] == pytest.approx(250.0, abs=0.1)
    assert result["integrated_emg_counts_s"] == pytest.approx((4 + 4 * np.sqrt(2)) / 8, rel=0.01)
    assert result["variance_counts2"] == pytest.approx(2.0, rel=0.01)
    assert 0 <= result["spectral_entropy_norm"] <= 1
    assert set(result).isdisjoint(FEATURE_COLUMNS)
    assert set(result).isdisjoint(FEATURE_COLUMNS_V2)


def test_candidate_features_reject_zero_energy_and_bad_thresholds():
    from emg_fatigue.candidate_features import candidate_window_features

    with pytest.raises(ValueError, match="energy"):
        candidate_window_features(np.zeros(1000), 1000.0)
    with pytest.raises(ValueError, match="threshold"):
        candidate_window_features(
            np.sin(2 * np.pi * 100 * np.arange(1000) / 1000),
            1000.0,
            amplitude_threshold=-1.0,
        )
