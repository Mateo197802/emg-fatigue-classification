import numpy as np
import pandas as pd
import pytest


def _synthetic_phase_fatigue_fixture():
    rng = np.random.default_rng(20261109)
    rows = []
    for group_index in range(6):
        for row_index in range(24):
            phase = row_index % 2
            fatigue = (row_index // 2) % 2
            noise = rng.normal(0.0, 0.15, size=4)
            rows.append(
                {
                    "participant": f"synthetic-{group_index}",
                    "phase": "extension" if phase == 0 else "flexion",
                    "fatigue": fatigue,
                    "rms": float(fatigue + 0.2 * phase + noise[0]),
                    "mdf_hz": float(90 - 12 * fatigue + 8 * phase + noise[1]),
                    "mav": float(0.5 * fatigue + noise[2]),
                    "waveform_length_per_s": float(2 * fatigue + phase + noise[3]),
                }
            )
    frame = pd.DataFrame(rows)
    return (
        frame[["rms", "mdf_hz", "mav", "waveform_length_per_s"]],
        frame["fatigue"].to_numpy(dtype=int),
        frame["phase"].to_numpy(dtype=object),
        frame["participant"].to_numpy(dtype=object),
    )


def test_grouped_oof_phase_probabilities_are_complete_and_group_disjoint():
    from emg_fatigue.reference_models import grouped_oof_phase_probabilities

    X, _, phase, groups = _synthetic_phase_fatigue_fixture()
    encoded = (phase == "flexion").astype(int)
    result = grouped_oof_phase_probabilities(X, encoded, groups, n_splits=3, seed=4)

    assert result.probabilities.shape == (len(X), 2)
    assert np.isfinite(result.probabilities).all()
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.oof_count == len(X)
    assert all(not (set(fold["train_groups"]) & set(fold["validation_groups"])) for fold in result.folds)


def test_reference_architectures_use_matched_grouped_fatigue_support_in_synthetic_mode():
    from emg_fatigue.contracts import read_reference_config
    from emg_fatigue.reference_models import evaluate_reference_architectures

    X, fatigue, phase, groups = _synthetic_phase_fatigue_fixture()
    config = read_reference_config("configs/reference_study.yaml")
    result = evaluate_reference_architectures(
        X,
        fatigue,
        phase,
        groups,
        config=config,
        synthetic_fixture=True,
        inner_splits=3,
        seed=7,
    )

    expected = {"majority", "logistic_rms_mdf", "logistic_all", "svm_rbf", "random_forest",
                "phase_then_fatigue", "phase_experts"}
    assert set(result.fatigue_predictions["architecture"]) == expected
    support = result.fatigue_predictions.groupby("architecture")["row_index"].nunique()
    assert set(support) == {len(X)}
    assert not result.fatigue_predictions.duplicated(["architecture", "row_index"]).any()
    assert set(result.fatigue_predictions["evidence_status"]) == {"synthetic_only"}
    assert set(result.phase_predictions["evidence_status"]) == {"synthetic_only"}
    for fold in result.fold_audit.to_dict("records"):
        assert not (set(fold["train_groups"]) & set(fold["test_groups"]))

    from emg_fatigue.reference_models import build_reference_evidence_tables

    tables = build_reference_evidence_tables(result)
    assert set(tables["fatigue_group_summary"]["metric"]) >= {"accuracy", "f1_macro"}
    assert set(tables["fatigue_paired_differences"]["candidate"]) == {
        "phase_then_fatigue", "phase_experts"
    }
    assert set(tables["fatigue_confusion_matrix"]["aggregation"]) == {"pooled_oof_windows"}
    assert set(tables["phase_group_summary"]["uncertainty_unit"]) == {"group_id"}
    assert all(set(frame["evidence_status"]) == {"synthetic_only"} for frame in tables.values())


def test_reference_architectures_refuse_empirical_training_while_gates_fail():
    from emg_fatigue.contracts import read_reference_config
    from emg_fatigue.reference_models import evaluate_reference_architectures

    X, fatigue, phase, groups = _synthetic_phase_fatigue_fixture()
    config = read_reference_config("configs/reference_study.yaml")
    with pytest.raises(ValueError, match="fatigue reference"):
        evaluate_reference_architectures(X, fatigue, phase, groups, config=config)


def test_phase_conditioned_comparison_uses_only_matched_fatigue_and_phase_rows():
    from emg_fatigue.contracts import read_reference_config
    from emg_fatigue.reference_models import evaluate_reference_architectures

    X, fatigue, phase, groups = _synthetic_phase_fatigue_fixture()
    phase = phase.copy()
    phase[::6] = "transition"
    expected_support = int(np.count_nonzero(np.isin(phase, ["extension", "flexion"])))

    result = evaluate_reference_architectures(
        X,
        fatigue,
        phase,
        groups,
        config=read_reference_config("configs/reference_study.yaml"),
        synthetic_fixture=True,
        inner_splits=3,
        seed=17,
    )

    support = result.fatigue_predictions.groupby("architecture")["row_index"].nunique()
    assert set(support) == {expected_support}
    assert result.phase_predictions["row_index"].nunique() == expected_support
    assert result.fold_audit["excluded_no_valid_phase_rows"].sum() == len(X) - expected_support


def test_phase_oof_rejects_single_class_labels():
    from emg_fatigue.reference_models import grouped_oof_phase_probabilities

    X, _, _, groups = _synthetic_phase_fatigue_fixture()
    with pytest.raises(ValueError, match="both classes"):
        grouped_oof_phase_probabilities(X, np.zeros(len(X), dtype=int), groups, n_splits=3)


def test_direct_fatigue_evaluator_needs_fatigue_gate_but_not_phase_gate():
    from dataclasses import replace

    from emg_fatigue.contracts import read_reference_config
    from emg_fatigue.reference_models import evaluate_direct_fatigue

    X, fatigue, _, groups = _synthetic_phase_fatigue_fixture()
    config = replace(
        read_reference_config("configs/reference_study.yaml"),
        fatigue_data_gate="passed",
        fatigue_definition_approved=True,
        participant_ids_verified=True,
    )
    result = evaluate_direct_fatigue(
        X,
        fatigue,
        groups,
        config=config,
        fatigue_reference_approved=True,
        participant_ids_verified=True,
    )

    assert result.evidence_status == "reference_gated_training"
    assert result.phase_predictions.empty
    assert set(result.fatigue_predictions["architecture"]) == {
        "majority", "logistic_rms_mdf", "logistic_all", "svm_rbf", "random_forest"
    }
    assert set(result.fatigue_predictions["evidence_status"]) == {"reference_gated_training"}


def test_phase_evaluator_needs_phase_gate_but_not_fatigue_gate():
    from dataclasses import replace

    from emg_fatigue.contracts import read_reference_config
    from emg_fatigue.reference_models import evaluate_movement_phase

    X, _, phase, groups = _synthetic_phase_fatigue_fixture()
    config = replace(
        read_reference_config("configs/reference_study.yaml"),
        phase_data_gate="passed",
        phase_definition_approved=True,
        participant_ids_verified=True,
    )
    result = evaluate_movement_phase(
        X,
        phase,
        groups,
        config=config,
        phase_reference_approved=True,
        participant_ids_verified=True,
    )

    assert result.evidence_status == "reference_gated_training"
    assert result.fatigue_predictions.empty
    assert set(result.phase_predictions["architecture"]) == {
        "majority", "logistic_all", "svm_rbf", "random_forest"
    }


def test_empirical_direct_evaluator_refuses_missing_fatigue_gate():
    from dataclasses import replace

    from emg_fatigue.contracts import read_reference_config
    from emg_fatigue.reference_models import evaluate_direct_fatigue

    X, fatigue, _, groups = _synthetic_phase_fatigue_fixture()
    config = replace(
        read_reference_config("configs/reference_study.yaml"),
        fatigue_definition_approved=True,
        participant_ids_verified=True,
    )
    with pytest.raises(ValueError, match="fatigue data gate"):
        evaluate_direct_fatigue(
            X,
            fatigue,
            groups,
            config=config,
            fatigue_reference_approved=True,
            participant_ids_verified=True,
        )


def test_empirical_phase_evaluator_refuses_missing_phase_gate():
    from dataclasses import replace

    from emg_fatigue.contracts import read_reference_config
    from emg_fatigue.reference_models import evaluate_movement_phase

    X, _, phase, groups = _synthetic_phase_fatigue_fixture()
    config = replace(
        read_reference_config("configs/reference_study.yaml"),
        phase_definition_approved=True,
        participant_ids_verified=True,
    )
    with pytest.raises(ValueError, match="phase data gate"):
        evaluate_movement_phase(
            X,
            phase,
            groups,
            config=config,
            phase_reference_approved=True,
            participant_ids_verified=True,
        )


def test_mixed_type_group_identifiers_are_rejected_explicitly():
    from emg_fatigue.reference_models import grouped_oof_phase_probabilities

    X, _, phase, groups = _synthetic_phase_fatigue_fixture()
    mixed = groups.copy()
    mixed[0] = 0
    with pytest.raises(ValueError, match="one identifier type"):
        grouped_oof_phase_probabilities(X, (phase == "flexion").astype(int), mixed, n_splits=3)


def test_rms_mdf_baseline_rejects_unidentified_first_two_columns():
    from emg_fatigue.reference_models import _family_features

    with pytest.raises(ValueError, match="named rms and mdf_hz"):
        _family_features(pd.DataFrame({"activity": [1.0], "time": [2.0]}), "logistic_rms_mdf")
