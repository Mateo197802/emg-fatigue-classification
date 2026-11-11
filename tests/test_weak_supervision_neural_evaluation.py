import pandas as pd
import pytest

pytest.importorskip("torch")

from emg_fatigue.weak_supervision.evaluation_neural import (
    _confusion_metrics,
    _group_balanced_macro_f1,
    audit_group_fold_support,
    select_fold_pretraining_indices,
    verify_tabular_parent_stage,
)
from emg_fatigue.weak_supervision.group_cv import FoldPlan, GroupFold


def test_pretraining_selector_uses_only_training_group_eligible_nonoverlap_rows():
    metadata = pd.DataFrame(
        {
            "group_id": ["G1", "G1", "G1", "G1", "G1", "G1", "G2"],
            "recording_id": ["T1", "T1", "T1", "T1", "T1", "T1", "T2"],
            "window_index": [0, 1, 2, 3, 4, 6, 0],
            "start_time_s": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 0.0],
            "eligible_for_label": [True, True, True, True, True, False, True],
        }
    )

    selected = select_fold_pretraining_indices(metadata, ("G1",))
    assert selected.tolist() == [0, 2, 4]
    assert metadata.iloc[selected].start_time_s.diff().dropna().tolist() == [0.5, 0.5]


def test_fold_support_audit_rejects_a_role_with_too_few_independent_groups():
    groups = tuple(f"G{index}" for index in range(8))
    rows = []
    for group in groups:
        for class_id in (0, 1):
            rows.append(
                {
                    "group_id": group,
                    "recording_id": f"{group}_C{class_id}",
                    "class_id": class_id,
                }
            )
    metadata = pd.DataFrame(rows)
    outer = GroupFold(1, groups[:4], groups[4:])
    inner = GroupFold(1, groups[:2], groups[2:4])
    plan = FoldPlan(7, groups, (outer,), {1: (inner,)})

    support = audit_group_fold_support(metadata, plan)
    assert set(support.distinct_source_groups) == {2, 4}
    assert set(support.distinct_recording_trials) == {2, 4}

    unsupported = metadata.loc[~(metadata.group_id.isin(("G5", "G6", "G7")) & metadata.class_id.eq(1))]
    with pytest.raises(ValueError, match="group/trial support gate failed"):
        audit_group_fold_support(unsupported, plan)


def test_group_balanced_macro_f1_does_not_penalize_perfect_one_class_groups():
    truth = [0, 0, 1, 1]
    prediction = [0, 0, 1, 1]
    groups = ["G0", "G1", "G2", "G3"]

    assert _group_balanced_macro_f1(truth, prediction, groups) == pytest.approx(1.0)


def test_one_class_group_metrics_are_undefined_with_a_reason():
    metric = _confusion_metrics([[2, 0], [0, 0]])

    assert metric["accuracy"] == pytest.approx(1.0)
    assert metric["macro_f1"] is None
    assert metric["balanced_accuracy"] is None
    assert metric["undefined_reasons"]["macro_f1"] == "class 1 absent from truth"


def test_neural_stage_rejects_a_temporal_baseline_changed_after_parent_receipt(tmp_path):
    stage = tmp_path / "tabular"
    stage.mkdir()
    fold_path = stage / "fold_manifest.json"
    oof_path = stage / "oof_predictions.parquet"
    fold_path.write_bytes(b'{"development_groups":[],"outer_folds":[]}')
    oof_path.write_bytes(b"frozen OOF predictions")
    import hashlib
    import json

    parent = {
        "status": "completed",
        "label_rule": "spectral_persistence_v1",
        "label_rule_type": "weak_supervision_proxy_not_independent_fatigue_ground_truth",
        "training_evaluation_window_subset": "even-index non-overlapping 500 ms windows",
        "overlap_between_selected_windows": "none; 500 ms start-to-start hop",
        "coverage_gate": {"fold_coverage_gate_pass": True},
        "internal_test_signal_values_parsed": 0,
        "external_rar_opened": False,
        "outputs_sha256": {
            str(fold_path.resolve()): hashlib.sha256(fold_path.read_bytes()).hexdigest(),
            str(oof_path.resolve()): hashlib.sha256(oof_path.read_bytes()).hexdigest(),
        },
    }
    (stage / "supervised_model_stage_manifest.json").write_text(json.dumps(parent), encoding="utf-8")
    oof_path.write_bytes(b"changed probabilities")

    with pytest.raises(ValueError, match="checksum"):
        verify_tabular_parent_stage(stage)
