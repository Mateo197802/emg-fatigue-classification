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


@pytest.mark.parametrize("mains_hz", [50.0, 60.0])
def test_notch_is_applied_only_when_measured_mains_prominence_exceeds_threshold(mains_hz):
    from dataclasses import replace

    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.preprocessing import filter_run, line_prominence

    fs = 1000.0
    t = np.arange(8000) / fs
    clean = 2.0 * np.sin(2 * np.pi * 100 * t)
    contaminated = clean + 5.0 * np.sin(2 * np.pi * mains_hz * t)
    assert line_prominence(contaminated, fs)[str(int(mains_hz))] > 10.0

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

    assert amplitude(contaminated_notch, mains_hz) < 0.1 * amplitude(contaminated_base, mains_hz)
    assert amplitude(contaminated_notch, 100) == pytest.approx(
        amplitude(contaminated_base, 100), rel=0.03
    )


def test_bandpass_removes_baseline_drift_while_preserving_emg_band_tone():
    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.preprocessing import filter_run

    fs = 1000.0
    t = np.arange(8000) / fs
    signal = 60.0 + 20.0 * np.sin(2 * np.pi * 0.5 * t) + 2.0 * np.sin(2 * np.pi * 100 * t)
    filtered, mask = filter_run(signal, fs, StudyConfig())

    def amplitude(values, frequency):
        valid_t = t[mask]
        design = np.column_stack((np.sin(2 * np.pi * frequency * valid_t),
                                  np.cos(2 * np.pi * frequency * valid_t)))
        coefficients, *_ = np.linalg.lstsq(design, values[mask], rcond=None)
        return float(np.hypot(*coefficients))

    assert amplitude(filtered, 0.5) < 0.02 * amplitude(signal, 0.5)
    assert amplitude(filtered, 100.0) == pytest.approx(amplitude(signal, 100.0), rel=0.03)


def test_qc_reports_uncalibrated_amplitude_and_linear_trend_as_descriptive_counts():
    from emg_fatigue.contracts import Recording, StudyConfig
    from emg_fatigue.qc import qc_recording

    fs = 1000.0
    t = np.arange(4000) / fs
    x = 150.0 + 3.0 * t + 2.0 * np.sin(2 * np.pi * 100.0 * t)
    recording = Recording("R", "G", "test", "left", "R1", t, x, "hash")

    result = qc_recording(recording, StudyConfig())

    assert len(result) == 1
    row = result[0]
    assert row["mean_count"] == pytest.approx(float(x.mean()), abs=1e-10)
    assert row["rms_count"] == pytest.approx(float(np.sqrt(np.mean(x * x))), abs=1e-10)
    assert row["peak_to_peak_count"] == pytest.approx(float(np.ptp(x)), abs=1e-10)
    assert row["p01_count"] == pytest.approx(float(np.quantile(x, 0.01)), abs=1e-10)
    assert row["p99_count"] == pytest.approx(float(np.quantile(x, 0.99)), abs=1e-10)
    assert row["linear_trend_count_per_s"] == pytest.approx(3.0, abs=0.01)
    assert row["clipping_status"] == "no_extreme_repeat_flag"
