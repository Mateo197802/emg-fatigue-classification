import numpy as np
import pytest

import emg_fatigue.features as feature_module


def test_corrected_waveform_length_schema_uses_sample_span_and_preserves_legacy():
    extractor = getattr(feature_module, "window_features_v2", None)
    assert callable(extractor), "versioned feature extractor is missing"

    rng = np.random.default_rng(481)
    x = rng.normal(size=256)
    fs = 1000.0
    legacy = feature_module.window_features(x, fs)
    corrected = extractor(x, fs)
    centered = x - x.mean()
    expected = np.abs(np.diff(centered)).sum() / ((len(x) - 1) / fs)

    assert "waveform_length_per_s" in feature_module.FEATURE_COLUMNS
    assert "waveform_length_per_s_sample_span" in feature_module.FEATURE_COLUMNS_V2
    assert "waveform_length_per_s" not in feature_module.FEATURE_COLUMNS_V2
    assert feature_module.FEATURE_SCHEMA_V2 != "emg_window_features_v1"
    assert corrected["waveform_length_per_s_sample_span"] == pytest.approx(expected)
    assert corrected["waveform_length_per_s_sample_span"] != pytest.approx(
        legacy["waveform_length_per_s"]
    )
