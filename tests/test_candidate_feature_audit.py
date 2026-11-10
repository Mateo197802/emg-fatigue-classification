import pandas as pd


def test_candidate_feature_audit_is_target_free_and_marks_synthetic_data(tmp_path):
    from emg_fatigue.candidate_feature_audit import build_candidate_feature_audit
    from emg_fatigue.contracts import StudyConfig
    from emg_fatigue.workflow import synthetic_recordings

    out = tmp_path / "private-audit"
    result = build_candidate_feature_audit(
        synthetic_recordings(groups=4, seconds=4, seed=73),
        StudyConfig(window_s=0.25),
        output_dir=out,
    )

    assert result["receipt"]["evidence_status"] == "synthetic_only"
    assert result["receipt"]["target_used"] is False
    assert result["receipt"]["label_columns_written"] == []
    features = pd.read_parquet(out / "candidate_features_private.parquet")
    assert "label" not in features.columns
    assert "early_proxy_nonfatigue" not in set(features.columns)
    assert set(result["window_correlations"]["analysis_scope"]) == {"descriptive_overlapping_windows"}
    assert (out / "candidate_feature_audit.receipt.json").is_file()
