from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import numpy as np
import pytest

from emg_fatigue.weak_supervision.contracts import RawRecording


def _preprocess_api():
    try:
        preprocess = importlib.import_module("emg_fatigue.weak_supervision.preprocess")
        activity = importlib.import_module("emg_fatigue.weak_supervision.activity")
        windows = importlib.import_module("emg_fatigue.weak_supervision.windows")
    except ImportError:
        pytest.fail("weak-supervision acquisition and segmentation components are not implemented", pytrace=False)
    return preprocess, activity, windows


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "fatigue_proxy.toml"
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _recording(duration_s: float = 6.0, fs_hz: float = 1000.0, *, gap_index: int | None = None, spike: bool = False):
    count = int(duration_s * fs_hz)
    time_s = np.arange(count, dtype=np.float64) / fs_hz
    if gap_index is not None:
        time_s[gap_index:] += 0.25
    rng = np.random.default_rng(912)
    values = rng.normal(0.0, 0.002, count)
    active = (time_s >= 1.5) & (time_s < 4.5)
    values[active] += 0.25 * np.sin(2 * np.pi * 90 * time_s[active])
    values[active] += 0.12 * np.sin(2 * np.pi * 170 * time_s[active])
    if spike:
        values[int(2.0 * fs_hz)] += 4.0
    return RawRecording(
        group_id="G01",
        recording_id="G01_F001",
        relative_time_s=time_s,
        adc_value=values,
        voltage_mv=values * 0.1,
        source_sha256="a" * 64,
        filename_metadata={"series_token": None, "side_token": None, "use": "descriptive_only"},
    )


def test_acquisition_audit_detects_clock_gap_without_inflating_sampling_rate():
    preprocess, _, _ = _preprocess_api()

    audit = preprocess.audit_recording(_recording(gap_index=3000), _config())

    assert audit.sampling_rate_hz == pytest.approx(1000.0, abs=0.1)
    assert audit.gap_count == 1
    assert len(audit.continuous_ranges) == 2
    assert audit.sampling_compatible is True


def test_acquisition_audit_quantifies_adc_to_voltage_relationship():
    preprocess, _, _ = _preprocess_api()

    audit = preprocess.audit_recording(_recording(), _config())

    assert audit.adc_voltage_pair_count == 6000
    assert audit.adc_voltage_slope_mv_per_unit == pytest.approx(0.1, abs=1e-8)
    assert audit.adc_voltage_intercept_mv == pytest.approx(0.0, abs=1e-8)
    assert audit.adc_voltage_r_squared == pytest.approx(1.0, abs=1e-10)


def test_acquisition_audit_rejects_rate_at_or_below_800_hz():
    preprocess, _, _ = _preprocess_api()

    audit = preprocess.audit_recording(_recording(duration_s=6.0, fs_hz=800.0), _config())

    assert audit.sampling_rate_hz == pytest.approx(800.0, abs=0.1)
    assert audit.sampling_compatible is False
    assert "sampling_rate_below_supported_minimum" in audit.quality_flags


def test_preprocessing_filters_each_run_and_marks_both_filter_edges():
    preprocess, _, _ = _preprocess_api()

    processed = preprocess.preprocess_recording(_recording(gap_index=3000), _config())

    assert len(processed.segments) == 2
    for segment in processed.segments:
        assert segment.sampling_rate_hz == 1000.0
        assert segment.edge_mask[:100].all()
        assert segment.edge_mask[-100:].all()
        assert not segment.edge_mask[100:-100].any()


def test_preprocessing_does_not_filter_when_raw_spectrum_fails_band_gate():
    preprocess, _, _ = _preprocess_api()
    recording = _recording()
    low_frequency_only = 0.2 * np.sin(2 * np.pi * 10 * recording.relative_time_s)
    incompatible = RawRecording(
        group_id=recording.group_id,
        recording_id=recording.recording_id,
        relative_time_s=recording.relative_time_s,
        adc_value=low_frequency_only,
        voltage_mv=low_frequency_only * 0.1,
        source_sha256=recording.source_sha256,
        filename_metadata=recording.filename_metadata,
    )

    processed = preprocess.preprocess_recording(incompatible, _config())

    assert processed.audit.signal_compatible is False
    assert processed.segments == ()


def test_activity_and_windows_require_candidate_occupancy_and_qc():
    preprocess, activity, windows = _preprocess_api()
    processed = preprocess.preprocess_recording(_recording(), _config())
    activity_segments = activity.detect_candidate_activity(processed, _config())

    assert activity_segments.status == "determinate"
    assert 0.05 <= activity_segments.active_fraction <= 0.95
    primary = windows.make_windows(processed, activity_segments, 0.5, 0.5)
    eligible = [window for window in primary if window.eligible_for_label]

    assert eligible
    assert all(window.activity_occupancy >= 0.5 for window in eligible)
    assert all(window.rejection_reason is None for window in eligible)
    stricter = windows.make_windows(processed, activity_segments, 0.5, 0.5, minimum_occupancy=0.75)
    stricter_eligible = [window for window in stricter if window.eligible_for_label]
    assert len(stricter_eligible) <= len(eligible)
    assert all(window.activity_occupancy >= 0.75 for window in stricter_eligible)


def test_impulsive_artifact_makes_overlapping_windows_ineligible():
    preprocess, activity, windows = _preprocess_api()
    processed = preprocess.preprocess_recording(_recording(spike=True), _config())
    activity_segments = activity.detect_candidate_activity(processed, _config())
    primary = windows.make_windows(processed, activity_segments, 0.5, 0.5)

    around_spike = [window for window in primary if window.start_time_s <= 2.0 < window.start_time_s + window.duration_s]

    assert around_spike
    assert all(not window.eligible_for_label for window in around_spike)
    assert all("impulse" in (window.rejection_reason or "") for window in around_spike)
