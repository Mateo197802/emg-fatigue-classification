import importlib

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

    with pytest.raises(ValueError, match="phase reference"):
        config.require_training_ready(
            fatigue_reference_approved=True,
            phase_reference_approved=False,
            participant_ids_verified=True,
        )
