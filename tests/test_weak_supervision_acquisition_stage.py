from __future__ import annotations

import csv
import importlib
import json
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pytest


def _signal_csv() -> bytes:
    fs_hz = 1000
    time_s = np.arange(6000, dtype=np.float64) / fs_hz
    rng = np.random.default_rng(442)
    values = rng.normal(0.0, 0.002, time_s.size)
    active = (time_s >= 1.5) & (time_s < 4.5)
    values[active] += 0.25 * np.sin(2 * np.pi * 90 * time_s[active])
    values[active] += 0.12 * np.sin(2 * np.pi * 170 * time_s[active])
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("Relative_Time_s", "ADC_Value", "Voltage_mV"))
    writer.writerows(zip(time_s, values, values * 0.1, strict=True))
    return output.getvalue().encode("utf-8")


def test_acquisition_stage_writes_private_dev_only_receipts_and_plots(tmp_path: Path):
    io_module = importlib.import_module("emg_fatigue.weak_supervision.io")
    stage_module = importlib.import_module("emg_fatigue.weak_supervision.acquisition_stage")
    label_stage_module = importlib.import_module("emg_fatigue.weak_supervision.label_audit")
    repo_root = Path(__file__).resolve().parents[1]
    config_source = repo_root / "configs" / "fatigue_proxy.toml"
    config = tomllib.loads(config_source.read_text(encoding="utf-8"))
    config["run"]["expected_training_csv_count"] = 2
    config["run"]["expected_source_group_count"] = 2
    config_copy = tmp_path / "test_config.toml"
    config_copy.write_text(
        config_source.read_text(encoding="utf-8")
        .replace("expected_training_csv_count = 140", "expected_training_csv_count = 2")
        .replace("expected_source_group_count = 15", "expected_source_group_count = 2"),
        encoding="utf-8",
    )
    archive = tmp_path / "training.zip"
    payload = _signal_csv()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("source_a/recording.csv", payload)
        output.writestr("source_b/recording.csv", payload + b"\n")

    run_dir = tmp_path / "private_run"
    prepared = io_module.load_training_zip(archive, seed=20261112, train_fraction=0.8)
    from emg_fatigue.weak_supervision.runner import prepare_training_run

    preparation = prepare_training_run(archive, run_dir, config_copy)
    result = stage_module.run_acquisition_stage(archive, run_dir, config_copy)
    label_result = label_stage_module.run_label_audit(archive, run_dir, config_copy)

    assert preparation["development_record_count"] == 1
    assert result["development_record_count"] == 1
    assert result["unparsed_internal_test_member_count"] == 1
    acquisition_dir = run_dir / "private" / "acquisition"
    for name in (
        "acquisition_audit.csv",
        "activity_audit.csv",
        "window_manifest.csv",
        "preprocessing_coefficients.json",
        "acquisition_stage_manifest.json",
    ):
        assert (acquisition_dir / name).is_file()
    assert list((acquisition_dir / "plots").glob("*.png"))
    receipt = json.loads((acquisition_dir / "acquisition_stage_manifest.json").read_text())
    assert receipt["internal_test_numeric_values_parsed"] == 0
    assert receipt["external_archive_opened"] is False
    assert prepared[1].unparsed_internal_test_member_count == 1
    assert label_result["feature_row_count"] > 0
    assert label_result["primary_label_row_count"] > 0
    feature_dir = run_dir / "private" / "features_labels"
    for name in (
        "features.parquet",
        "features.csv",
        "pseudo_labels.parquet",
        "pseudo_labels.csv",
        "pseudo_labels_alternatives.parquet",
        "pseudo_labels_alternatives.csv",
        "window_size_sensitivity_labels.csv",
        "label_sensitivity_by_run.csv",
        "label_rule_agreement.csv",
        "label_counts_by_group.csv",
        "label_counts_by_recording.csv",
        "label_counts_by_metadata.csv",
        "label_counts_by_temporal_quartile.csv",
        "feature_schema.json",
        "feature_correlation_spearman.csv",
        "label_rule_config.json",
        "feature_label_stage_manifest.json",
    ):
        assert (feature_dir / name).is_file()
    label_receipt = json.loads((feature_dir / "feature_label_stage_manifest.json").read_text())
    assert label_receipt["internal_test_numeric_values_parsed"] == 0
    assert label_receipt["primary_label_counts"].keys() == {"-1", "0", "1"}
