import importlib
import importlib.util
from dataclasses import replace

import numpy as np

from emg_fatigue.reference_labels import (
    ReferenceContract,
    fatigue_labels_from_torque,
    phase_from_flexion_angle,
)


def _builder():
    spec = importlib.util.find_spec("emg_fatigue.reference_windows")
    assert spec is not None, "reference-backed fixed-window builder is missing"
    return importlib.import_module("emg_fatigue.reference_windows").make_reference_windows


def _contract(kind, level, *, angle_convention=None, repeatability_id=None):
    return ReferenceContract(
        reference_kind=kind,
        reference_level=level,
        source_id="synthetic-source",
        unit="N*m" if kind == "torque" else "deg",
        time_axis_id="synthetic-reference-clock",
        emg_time_axis_id="synthetic-emg-clock",
        participant_id="synthetic-person",
        session_id="synthetic-session",
        calibration_verified=True,
        calibration_id="synthetic-calibration",
        synchronized=True,
        synchronization_id="synthetic-sync",
        synchronization_residual_s=0.001,
        synchronization_tolerance_s=0.01,
        label_definition_id="synthetic-label-v1",
        label_definition_approved=False,
        synthetic_fixture=True,
        repeatability_id=repeatability_id,
        angle_convention=angle_convention,
    )


def test_fixed_500_ms_windows_use_verified_rate_and_half_window_hop():
    make_windows = _builder()
    fs = 1000.0
    time = np.arange(0, 1.501, 1 / fs)
    result = make_windows(time_s=time, fs_hz=fs, window_s=0.5, overlap=0.5)

    assert result.window_samples == 500
    assert result.hop_samples == 250
    assert result.nominal_bin_spacing_hz == 2.0
    assert len(result.windows) == 5
    assert all(window.fatigue_state is None for window in result.windows)
    assert all(window.fatigue_status == "reference_missing" for window in result.windows)


def test_window_straddling_phase_transition_is_unlabeled():
    make_windows = _builder()
    fs = 1000.0
    time = np.arange(0, 1.001, 1 / fs)
    angle = np.r_[np.linspace(0, 30, 501), np.linspace(30, 0, 500)]
    phase = phase_from_flexion_angle(
        time,
        angle,
        velocity_deadband_deg_s=1.0,
        reference_contract=_contract("knee_angle", "time_series", angle_convention="flexion_positive"),
        max_gap_s=0.002,
    )
    result = make_windows(
        time_s=time, fs_hz=fs, window_s=0.5, overlap=0.5, phase_result=phase
    )

    assert any(window.phase_status == "phase_transition_or_mixed" for window in result.windows)
    assert all(
        window.phase is None
        for window in result.windows
        if window.phase_status == "phase_transition_or_mixed"
    )
    assert all(window.phase_evidence_status == "synthetic_only" for window in result.windows)


def test_fatigue_onset_boundary_window_is_unlabeled_and_marked_synthetic():
    make_windows = _builder()
    fs = 1000.0
    time = np.arange(0, 2.001, 1 / fs)
    reps = np.arange(8, dtype=float)
    fatigue = fatigue_labels_from_torque(
        [100, 101, 99, 70, 70, 70, 70, 70],
        reference_contract=_contract("torque", "repetition", repeatability_id="synthetic-sem"),
        repetition_time_s=reps,
        sem_nm=1.0,
        valid_reps=np.ones(8, dtype=bool),
        angle_matched=np.ones(8, dtype=bool),
    )
    fatigue = replace(fatigue, onset_time_s=1.0)
    result = make_windows(
        time_s=time,
        fs_hz=fs,
        window_s=0.5,
        overlap=0.5,
        fatigue_result=fatigue,
    )

    crossing = [window for window in result.windows if window.fatigue_status == "crosses_fatigue_onset"]
    assert crossing
    assert all(window.fatigue_state is None for window in crossing)
    assert all(window.fatigue_evidence_status == "synthetic_only" for window in result.windows)
