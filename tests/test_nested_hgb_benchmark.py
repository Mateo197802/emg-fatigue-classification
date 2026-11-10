import pandas as pd


def test_nested_hgb_selection_uses_participant_disjoint_inner_folds():
    from experiments.reference_fatigue.nested_hgb_benchmark import select_hgb_params

    rows = []
    for participant in range(6):
        for trial in range(2):
            for label in (0, 1):
                rows.append({
                    "window_id": f"p{participant}-t{trial}-y{label}",
                    "subject_id": f"p{participant}",
                    "trial_id": f"t{trial}",
                    "prime_mover": "biceps",
                    "window_start_s": float(label),
                    "window_center_s": float(label) + 0.25,
                    "label": label,
                    "signal_feature": float(label) + participant * 0.001,
                })
    frame = pd.DataFrame(rows)

    selected, history = select_hgb_params(
        frame,
        ["signal_feature"],
        candidates=[{"max_iter": 10, "max_leaf_nodes": 3}],
        inner_splits=3,
        sample_cap=1,
        seed=9,
    )

    assert selected == {"max_iter": 10, "max_leaf_nodes": 3}
    assert len(history) == 3
    assert all(not set(row["train_subjects"]) & set(row["valid_subjects"]) for row in history)
    assert all(row["participant_macro_f1"] >= 0 for row in history)


def test_fixed_hgb_reference_oof_can_be_recreated_from_seeded_protocol(tmp_path):
    from experiments.reference_fatigue.nested_hgb_benchmark import reproduce_fixed_hgb_predictions

    rows = []
    for participant in range(4):
        for trial in range(2):
            for label in (0, 1):
                rows.append({
                    "window_id": f"p{participant}-t{trial}-y{label}",
                    "subject_id": f"p{participant}",
                    "trial_id": f"t{trial}",
                    "prime_mover": "biceps",
                    "window_start_s": float(label),
                    "label": label,
                    "signal_feature": float(label) + participant * 0.001,
                })
    frame = pd.DataFrame(rows)
    path = tmp_path / "fixed_oof.parquet"

    oof = reproduce_fixed_hgb_predictions(
        frame, ["signal_feature"], path, sample_cap=1, seed=7
    )

    assert path.is_file()
    assert len(oof) == len(frame)
    assert oof.window_id.is_unique
    assert set(oof.subject_id) == set(frame.subject_id)


def test_nested_manifest_hashes_recreated_fixed_oof_when_no_baseline_is_supplied(tmp_path):
    import json

    from experiments.reference_fatigue.nested_hgb_benchmark import (
        _sha256,
        run_nested_benchmark,
    )

    rows = []
    for participant in range(13):
        for trial in range(2):
            for label in (0, 1):
                rows.append({
                    "window_id": f"p{participant}-t{trial}-y{label}",
                    "subject_id": f"p{participant}",
                    "trial_id": f"t{trial}",
                    "prime_mover": "biceps",
                    "window_start_s": float(label),
                    "window_center_s": float(label) + 0.25,
                    "label": label,
                    "signal_feature": float(label) + participant * 0.001,
                })
    frame = pd.DataFrame(rows)
    features_path = tmp_path / "features.parquet"
    frame.to_parquet(features_path, index=False)
    private_output = tmp_path / "private"
    public_output = tmp_path / "public"
    missing_baseline = tmp_path / "not-present.parquet"

    _, manifest = run_nested_benchmark(
        features_path,
        missing_baseline,
        private_output,
        public_output,
        sample_cap=1,
        seed=11,
        bootstrap_replicates=20,
        candidates=[{"max_iter": 10, "learning_rate": 0.08, "max_leaf_nodes": 3, "l2_regularization": 1.0}],
    )

    generated_fixed_oof = private_output / "fixed_hgb_reference_oof.parquet"
    written_manifest = json.loads((public_output / "nested_hgb_manifest.json").read_text())
    assert manifest["fixed_oof_sha256"] == _sha256(generated_fixed_oof)
    assert written_manifest["fixed_oof_sha256"] == _sha256(generated_fixed_oof)
