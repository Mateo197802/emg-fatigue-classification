from dataclasses import replace

import numpy as np
import pytest

from emg_fatigue.reference_labels import (
    ReferenceContract,
    assert_reference_labels_exportable,
    fatigue_labels_from_torque,
    phase_from_flexion_angle,
)


def measured_reference(n, **overrides):
    valid_reps = overrides.pop("valid_reps", np.ones(n, dtype=bool))
    angle_matched = overrides.pop("angle_matched", np.ones(n, dtype=bool))
    contract = ReferenceContract(
        reference_kind="torque",
        reference_level="repetition",
        source_id="synthetic-torque-series",
        unit="N*m",
        time_axis_id="synthetic-torque-clock",
        emg_time_axis_id="synthetic-emg-clock",
        participant_id="synthetic-participant",
        session_id="synthetic-session",
        calibration_verified=True,
        calibration_id="synthetic-calibration",
        synchronized=True,
        synchronization_id="synthetic-sync",
        synchronization_residual_s=0.001,
        synchronization_tolerance_s=0.01,
        label_definition_id="candidate-torque-rule-v1",
        label_definition_approved=False,
        synthetic_fixture=True,
        repeatability_id="synthetic-repeatability",
    )
    contract = replace(contract, **overrides)
    return {
        "reference_contract": contract,
        "repetition_time_s": np.arange(n, dtype=float),
        "valid_reps": valid_reps,
        "angle_matched": angle_matched,
    }


def phase_reference(**overrides):
    contract = ReferenceContract(
        reference_kind="knee_angle",
        reference_level="time_series",
        source_id="synthetic-angle-series",
        unit="deg",
        time_axis_id="synthetic-angle-clock",
        emg_time_axis_id="synthetic-emg-clock",
        participant_id="synthetic-participant",
        session_id="synthetic-session",
        calibration_verified=True,
        calibration_id="synthetic-angle-calibration",
        synchronized=True,
        synchronization_id="synthetic-sync",
        synchronization_residual_s=0.001,
        synchronization_tolerance_s=0.01,
        label_definition_id="synthetic-knee-phase-v1",
        label_definition_approved=False,
        synthetic_fixture=True,
        angle_convention="flexion_positive",
    )
    return replace(contract, **overrides)


def test_force_decline_label_requires_two_confirmed_windows():
    values = [100, 101, 99, 95, 90, 88, 80, 78]
    result = fatigue_labels_from_torque(
        values,
        sem_nm=2.0,
        **measured_reference(len(values)),
    )

    assert result.mdc95_nm == pytest.approx(1.96 * np.sqrt(2) * 2.0)
    assert result.threshold_nm == pytest.approx(100.0 - result.mdc95_nm)
    assert result.onset_index == 5
    assert result.labels.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert result.baseline_indices == (0, 1, 2)
    assert result.source_id == "synthetic-torque-series"
    assert result.calibration_id == "synthetic-calibration"
    assert result.synchronization_id == "synthetic-sync"
    assert result.repeatability_id == "synthetic-repeatability"
    assert result.evidence_status == "synthetic_only"
    with pytest.raises(ValueError, match="synthetic"):
        assert_reference_labels_exportable(result)


def test_isolated_torque_drop_does_not_create_fatigue_label():
    values = [100, 101, 99, 90, 99, 88, 99]
    result = fatigue_labels_from_torque(
        values,
        sem_nm=2.0,
        **measured_reference(len(values)),
    )

    assert result.onset_index is None
    assert result.labels.tolist() == [0] * 7


@pytest.mark.parametrize("sem_nm", [0.0, -1.0, float("nan")])
def test_force_label_requires_valid_repeatability_error(sem_nm):
    values = [100, 101, 99, 95, 90, 88]
    with pytest.raises(ValueError, match="SEM"):
        fatigue_labels_from_torque(values, sem_nm=sem_nm, **measured_reference(len(values)))


def test_force_labels_require_verified_calibration_synchronization_and_masks():
    values = [100, 101, 99, 95, 90, 88]
    with pytest.raises(ValueError, match="valid_reps"):
        fatigue_labels_from_torque(
            values,
            sem_nm=2.0,
            reference_contract=measured_reference(len(values))["reference_contract"],
            repetition_time_s=np.arange(len(values)),
        )
    with pytest.raises(ValueError, match="calibration"):
        fatigue_labels_from_torque(
            values, sem_nm=2.0,
            **measured_reference(len(values), calibration_verified=False),
        )
    with pytest.raises(ValueError, match="synchron"):
        fatigue_labels_from_torque(
            values, sem_nm=2.0,
            **measured_reference(len(values), synchronized=False),
        )
    with pytest.raises(ValueError, match="source_id"):
        fatigue_labels_from_torque(
            values, sem_nm=2.0,
            **measured_reference(len(values), source_id=""),
        )


def test_invalid_repetitions_are_unlabeled_and_break_the_rolling_confirmation():
    values = [100, 101, 99, 70, 70, 70]
    valid = [True, True, True, False, True, True]
    result = fatigue_labels_from_torque(
        values, sem_nm=1.0, **measured_reference(len(values), valid_reps=valid)
    )

    assert result.onset_index is None
    assert result.labels.tolist() == [0, 0, 0, -1, 0, 0]


def test_angle_mismatched_repetitions_are_unlabeled_and_not_used_as_baseline():
    values = [70, 100, 101, 99, 90, 88, 80, 78]
    valid = [True] * len(values)
    matched = [False, True, True, True, True, True, True, True]
    result = fatigue_labels_from_torque(
        values,
        sem_nm=2.0,
        **measured_reference(len(values), valid_reps=valid, angle_matched=matched),
    )

    assert result.baseline_indices == (1, 2, 3)
    assert result.baseline_torque_nm == pytest.approx(100.0)
    assert result.labels[0] == -1


def test_invalid_gap_does_not_allow_fatigue_onset_to_be_confirmed_across_it():
    values = [100, 101, 99, 80, 79, 78, 77, 76]
    valid = [True, True, True, True, False, True, True, True]
    result = fatigue_labels_from_torque(
        values, sem_nm=1.0, **measured_reference(len(values), valid_reps=valid)
    )

    assert result.onset_index is None
    assert result.labels[4] == -1


def test_phase_labels_come_from_flexion_angle_direction_and_deadband():
    labels = phase_from_flexion_angle(
        time_s=[0, 1, 2, 3, 4, 5],
        flexion_angle_deg=[10, 15, 20, 20, 15, 10],
        velocity_deadband_deg_s=3.0,
        reference_contract=phase_reference(),
        max_gap_s=1.1,
    )

    assert labels.labels.tolist() == ["flexion", "flexion", "transition", "transition", "extension", "extension"]
    assert labels.angle_calibration_id == "synthetic-angle-calibration"
    assert labels.synchronization_id == "synthetic-sync"


def test_phase_labels_reject_unsynchronized_or_invalid_time_axis():
    with pytest.raises(ValueError, match="strictly increasing"):
        phase_from_flexion_angle(
            time_s=[0, 1, 1, 3],
            flexion_angle_deg=[10, 15, 14, 10],
            velocity_deadband_deg_s=1.0,
            reference_contract=phase_reference(),
            max_gap_s=1.1,
        )


def test_phase_labels_require_calibrated_synchronized_angle():
    values = dict(
        time_s=[0, 1, 2], flexion_angle_deg=[10, 15, 10], velocity_deadband_deg_s=1.0,
        reference_contract=phase_reference(), max_gap_s=1.1,
    )
    with pytest.raises(ValueError, match="calibration"):
        phase_from_flexion_angle(
            **{**values, "reference_contract": phase_reference(calibration_verified=False)}
        )
    with pytest.raises(ValueError, match="synchron"):
        phase_from_flexion_angle(
            **{**values, "reference_contract": phase_reference(synchronized=False)}
        )
    with pytest.raises(ValueError, match="calibration_id"):
        phase_from_flexion_angle(
            **{**values, "reference_contract": phase_reference(calibration_id="")}
        )


def test_phase_labels_reject_gaps_that_would_corrupt_angular_velocity():
    with pytest.raises(ValueError, match="gap"):
        phase_from_flexion_angle(
            time_s=[0.0, 0.01, 0.02, 0.20, 0.21],
            flexion_angle_deg=[10, 11, 12, 11, 10],
            velocity_deadband_deg_s=1.0,
            reference_contract=phase_reference(),
            max_gap_s=0.03,
        )


def test_real_fatigue_labels_require_approved_definition_and_approval_receipt():
    values = [100, 101, 99, 95, 90, 88]
    contract = replace(
        measured_reference(len(values))["reference_contract"],
        synthetic_fixture=False,
        label_definition_approved=False,
    )
    args = measured_reference(len(values))
    args["reference_contract"] = contract
    with pytest.raises(ValueError, match="approved"):
        fatigue_labels_from_torque(values, sem_nm=2.0, **args)

    contract = replace(contract, label_definition_approved=True, approval_record_id="review-001")
    args["reference_contract"] = contract
    result = fatigue_labels_from_torque(values, sem_nm=2.0, **args)
    assert result.evidence_status == "reference_approved"
    assert_reference_labels_exportable(result)


def test_fatigue_labeler_rejects_session_level_labels_for_repetition_task():
    values = [100, 101, 99, 95, 90, 88]
    args = measured_reference(len(values))
    args["reference_contract"] = replace(args["reference_contract"], reference_level="session")
    with pytest.raises(ValueError, match="reference_level"):
        fatigue_labels_from_torque(values, sem_nm=2.0, **args)


def test_reference_contract_rejects_synchronization_residual_above_tolerance():
    values = [100, 101, 99, 95, 90, 88]
    args = measured_reference(len(values))
    args["reference_contract"] = replace(
        args["reference_contract"], synchronization_residual_s=0.02
    )
    with pytest.raises(ValueError, match="drift"):
        fatigue_labels_from_torque(values, sem_nm=2.0, **args)


def test_phase_result_retains_reference_and_emg_time_axis_ids():
    result = phase_from_flexion_angle(
        time_s=[0, 1, 2],
        flexion_angle_deg=[0, 5, 10],
        velocity_deadband_deg_s=1.0,
        reference_contract=phase_reference(),
        max_gap_s=1.1,
    )

    assert result.time_axis_id == "synthetic-angle-clock"
    assert result.emg_time_axis_id == "synthetic-emg-clock"


def test_extension_positive_angle_convention_reverses_the_velocity_sign():
    result = phase_from_flexion_angle(
        time_s=[0, 1, 2, 3, 4, 5],
        flexion_angle_deg=[10, 15, 20, 20, 15, 10],
        velocity_deadband_deg_s=3.0,
        angle_convention="extension_positive",
        reference_contract=phase_reference(angle_convention="extension_positive"),
        max_gap_s=1.1,
    )

    assert result.labels.tolist() == [
        "extension", "extension", "transition", "transition", "flexion", "flexion"
    ]
    assert result.angle_convention == "extension_positive"


def test_phase_labeler_rejects_synchronization_drift_above_tolerance():
    contract = phase_reference(synchronization_residual_s=0.02)
    with pytest.raises(ValueError, match="drift"):
        phase_from_flexion_angle(
            time_s=[0, 1, 2],
            flexion_angle_deg=[10, 15, 20],
            velocity_deadband_deg_s=1.0,
            reference_contract=contract,
            max_gap_s=1.1,
        )
