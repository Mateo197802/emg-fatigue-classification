import numpy as np
import pytest


@pytest.mark.parametrize("sample_rate", [0.0, -1.0, float("nan"), float("inf"), 800.0])
def test_both_feature_schemas_reject_invalid_or_sub_nyquist_sample_rates(sample_rate):
    from emg_fatigue.features import window_features, window_features_v2

    signal = np.sin(2 * np.pi * 100 * np.arange(250) / 1000)
    for extractor in (window_features, window_features_v2):
        with pytest.raises(ValueError, match="sampling rate"):
            extractor(signal, sample_rate)


@pytest.mark.parametrize(
    "signal, sample_rate, message",
    [
        (np.ones((1000, 1)), 1000.0, "one-dimensional"),
        (np.r_[np.ones(999), np.nan], 1000.0, "finite"),
        (np.ones(1000), float("nan"), "sampling rate"),
        (np.ones(1000), 800.0, "Nyquist"),
    ],
)
def test_filter_run_rejects_malformed_signal_or_unusable_sample_rate(signal, sample_rate, message):
    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.preprocessing import filter_run

    with pytest.raises(ValueError, match=message):
        filter_run(signal, sample_rate, StudyConfig())


def test_notch_is_applied_only_when_measured_mains_prominence_exceeds_threshold():
    from dataclasses import replace

    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.preprocessing import filter_run, line_prominence

    fs = 1000.0
    t = np.arange(8000) / fs
    clean = 2.0 * np.sin(2 * np.pi * 100 * t)
    contaminated = clean + 5.0 * np.sin(2 * np.pi * 50 * t)
    assert line_prominence(contaminated, fs)["50"] > 10.0

    base = StudyConfig(notch=False)
    notch = replace(base, notch=True)
    clean_base, mask = filter_run(clean, fs, base)
    clean_conditional, _ = filter_run(clean, fs, notch)
    contaminated_base, _ = filter_run(contaminated, fs, base)
    contaminated_notch, _ = filter_run(contaminated, fs, notch)

    np.testing.assert_allclose(clean_conditional, clean_base, rtol=0, atol=1e-12)
    valid_t = t[mask]

    def amplitude(values, frequency):
        design = np.column_stack((np.sin(2 * np.pi * frequency * valid_t),
                                  np.cos(2 * np.pi * frequency * valid_t)))
        coefficients, *_ = np.linalg.lstsq(design, values[mask], rcond=None)
        return float(np.hypot(*coefficients))

    assert amplitude(contaminated_notch, 50) < 0.1 * amplitude(contaminated_base, 50)
    assert amplitude(contaminated_notch, 100) == pytest.approx(
        amplitude(contaminated_base, 100), rel=0.03
    )
