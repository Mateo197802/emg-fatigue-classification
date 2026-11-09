import inspect

import numpy as np
import pytest

import emg_fatigue.reference_labels as reference_labels
from emg_fatigue.reference_labels import (
    ReferenceContract,
    fatigue_labels_from_torque,
    phase_from_flexion_angle,
)


def _torque_arguments(n):
    return {
        "reference_contract": ReferenceContract(
            reference_kind="torque",
            reference_level="repetition",
            source_id="synthetic-unapproved-torque",
            unit="N*m",
            time_axis_id="torque-clock",
            emg_time_axis_id="emg-clock",
            participant_id="participant-001",
            session_id="session-001",
            calibration_verified=True,
            calibration_id="calibration-001",
            synchronized=True,
            synchronization_id="sync-001",
            synchronization_residual_s=0.001,
            synchronization_tolerance_s=0.01,
            label_definition_id="candidate-rule-v1",
            label_definition_approved=False,
            approval_record_id=None,
            synthetic_fixture=False,
            repeatability_id="repeatability-001",
        ),
        "repetition_time_s": np.arange(n, dtype=float),
        "sem_nm": 2.0,
        "valid_reps": np.ones(n, dtype=bool),
        "angle_matched": np.ones(n, dtype=bool),
    }


def test_candidate_fatigue_rule_cannot_emit_labels_without_approval():
    torque = [100.0, 101.0, 99.0, 95.0, 90.0, 88.0]

    with pytest.raises(ValueError, match="approved"):
        fatigue_labels_from_torque(torque, **_torque_arguments(len(torque)))


def test_phase_interface_requires_explicit_angle_convention():
    parameters = inspect.signature(phase_from_flexion_angle).parameters

    assert "angle_convention" in parameters


def test_reference_contract_exists_and_records_label_approval_and_grouping():
    contract = getattr(reference_labels, "ReferenceContract", None)

    assert contract is not None
    assert {
        "reference_kind",
        "reference_level",
        "source_id",
        "unit",
        "time_axis_id",
        "participant_id",
        "session_id",
        "calibration_verified",
        "synchronized",
        "synchronization_residual_s",
        "synchronization_tolerance_s",
        "label_definition_id",
        "label_definition_approved",
        "approval_record_id",
        "synthetic_fixture",
    }.issubset(contract.__dataclass_fields__)


def test_labelers_consume_the_reference_contract():
    assert "reference_contract" in inspect.signature(fatigue_labels_from_torque).parameters
    assert "reference_contract" in inspect.signature(phase_from_flexion_angle).parameters
