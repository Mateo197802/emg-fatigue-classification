import importlib
from dataclasses import replace

import pytest


def test_reference_config_is_separate_and_blocks_training_until_g0_passes():
    module = importlib.import_module("emg_fatigue.contracts")
    reader = getattr(module, "read_reference_config", None)
    assert callable(reader), "separate reference-backed config reader is missing"

    config = reader("configs/reference_study.yaml")
    assert config.primary_window_s == 0.5
    assert config.sensitivity_window_s == 0.25
    assert config.overlap == 0.5
    assert config.psd_method == "periodogram"
    assert config.data_gate == "failed"
    with pytest.raises(ValueError, match="fatigue reference"):
        config.require_training_ready(
            fatigue_reference_approved=False,
            phase_reference_approved=False,
            participant_ids_verified=False,
        )


def test_reference_config_requires_phase_reference_for_auxiliary_architectures():
    module = importlib.import_module("emg_fatigue.contracts")
    reader = getattr(module, "read_reference_config", None)
    assert callable(reader), "separate reference-backed config reader is missing"
    config = reader("configs/reference_study.yaml")
    config = replace(
        config,
        fatigue_data_gate="passed",
        fatigue_definition_approved=True,
        participant_ids_verified=True,
    )

    with pytest.raises(ValueError, match="phase reference"):
        config.require_training_ready(
            fatigue_reference_approved=True,
            phase_reference_approved=False,
            participant_ids_verified=True,
        )


def test_direct_fatigue_training_gate_does_not_require_kinematics():
    module = importlib.import_module("emg_fatigue.contracts")
    config = module.read_reference_config("configs/reference_study.yaml")
    direct_config = replace(
        config,
        fatigue_data_gate="passed",
        fatigue_definition_approved=True,
        participant_ids_verified=True,
    )

    direct_config.require_fatigue_training_ready(
        fatigue_reference_approved=True,
        participant_ids_verified=True,
    )
    with pytest.raises(ValueError, match="phase reference"):
        direct_config.require_phase_conditioned_training_ready(
            fatigue_reference_approved=True,
            phase_reference_approved=False,
            participant_ids_verified=True,
        )


def test_phase_training_gate_does_not_require_fatigue_reference():
    module = importlib.import_module("emg_fatigue.contracts")
    config = module.read_reference_config("configs/reference_study.yaml")
    phase_config = replace(
        config,
        phase_data_gate="passed",
        phase_definition_approved=True,
        participant_ids_verified=True,
    )

    phase_config.require_phase_training_ready(
        phase_reference_approved=True,
        participant_ids_verified=True,
    )
    with pytest.raises(ValueError, match="fatigue reference"):
        phase_config.require_fatigue_training_ready(
            fatigue_reference_approved=False,
            participant_ids_verified=True,
        )


def test_phase_conditioned_training_requires_both_independent_data_gates():
    module = importlib.import_module("emg_fatigue.contracts")
    config = module.read_reference_config("configs/reference_study.yaml")
    ready_config = replace(
        config,
        fatigue_data_gate="passed",
        phase_data_gate="passed",
        fatigue_definition_approved=True,
        phase_definition_approved=True,
        participant_ids_verified=True,
    )

    ready_config.require_phase_conditioned_training_ready(
        fatigue_reference_approved=True,
        phase_reference_approved=True,
        participant_ids_verified=True,
    )
