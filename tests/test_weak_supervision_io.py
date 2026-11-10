from __future__ import annotations

import importlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


TRAINING_GROUPS = [f"group_{index:02d}" for index in range(1, 16)]
EXPECTED_TEST_ALIASES = {"G04", "G10", "G13"}


def _inventory_api():
    try:
        module = importlib.import_module("emg_fatigue.weak_supervision.io")
    except ImportError:
        pytest.fail("weak-supervision raw inventory API is not implemented", pytrace=False)
    splitter = getattr(module, "create_group_split", None)
    loader = getattr(module, "load_training_zip", None)
    assert callable(splitter), "group splitter must be available for the raw inventory workflow"
    assert callable(loader), "training ZIP loader must be available for the raw inventory workflow"
    return splitter, loader


def _csv_payload(*, malformed: bool = False, offset: int = 0) -> bytes:
    sample = "not-a-number" if malformed else str(1.25 + offset)
    return (
        "Relative_Time_s,ADC_Value,Voltage_mV\n"
        f"0.000,{offset},0.0\n0.001,{sample},0.1\n0.002,{2.5 + offset},0.2\n"
    ).encode("utf-8")


def _make_zip(path: Path, *, malformed_groups: set[str] | None = None) -> Path:
    malformed_groups = malformed_groups or set()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, group in enumerate(TRAINING_GROUPS):
            archive.writestr(
                f"{group}/recording.csv",
                _csv_payload(malformed=group in malformed_groups, offset=index),
            )
    return path


def test_group_split_is_repeatable_and_reserves_three_of_fifteen_groups():
    splitter, _ = _inventory_api()
    groups = [f"G{index:02d}" for index in range(1, 16)]

    first = splitter(groups, seed=20261112)
    second = splitter(groups, seed=20261112)

    assert first == second
    assert set(first.test_groups) == EXPECTED_TEST_ALIASES
    assert len(first.development_groups) == 12
    assert len(first.test_groups) == 3
    assert set(first.development_groups).isdisjoint(first.test_groups)
    assert set(first.development_groups) | set(first.test_groups) == set(groups)


def test_training_zip_loader_does_not_parse_locked_test_signal_values(tmp_path: Path):
    _, loader = _inventory_api()
    archive_path = _make_zip(
        tmp_path / "training.zip",
        malformed_groups={f"group_{int(alias[1:]):02d}" for alias in EXPECTED_TEST_ALIASES},
    )

    recordings, inventory = loader(archive_path)

    assert len(recordings) == 12
    assert {record.group_id for record in recordings} == set(inventory.split.development_groups)
    assert set(inventory.split.test_groups) == EXPECTED_TEST_ALIASES
    assert inventory.unparsed_internal_test_member_count == 3
    assert inventory.csv_count == 15
    assert inventory.non_csv_member_count == 0


def test_training_zip_loader_rejects_identical_signal_content_across_source_groups(tmp_path: Path):
    _, loader = _inventory_api()
    archive_path = tmp_path / "duplicates.zip"
    payload = _csv_payload()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("group_01/a.csv", payload)
        archive.writestr("group_02/b.csv", payload)

    with pytest.raises(ValueError, match="(?i)duplicate"):
        loader(archive_path)


def test_training_zip_loader_rejects_unsafe_member_paths(tmp_path: Path):
    _, loader = _inventory_api()
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../group_01/recording.csv", _csv_payload())

    with pytest.raises(ValueError, match="path"):
        loader(archive_path)


def test_prepare_stage_writes_private_split_and_provenance_receipts(tmp_path: Path):
    try:
        runner = importlib.import_module("emg_fatigue.weak_supervision.runner")
    except ImportError:
        pytest.fail("private prepare-stage runner is not implemented", pytrace=False)
    prepare = getattr(runner, "prepare_training_run", None)
    assert callable(prepare), "prepare stage must create its private reproducibility receipts"

    archive_path = _make_zip(tmp_path / "training.zip")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[run]\n"
        "seed = 20261110\n"
        "split_seed = 20261112\n"
        "train_fraction = 0.8\n"
        "expected_training_csv_count = 15\n"
        "expected_source_group_count = 15\n"
        'expected_training_archive_name = "expected-training.zip"\n',
        encoding="utf-8",
    )
    run_dir = tmp_path / "private-run"

    result = prepare(archive_path, run_dir, config_path)

    assert result["development_record_count"] == 12
    assert result["internal_test_group_count"] == 3
    split_manifest = __import__("json").loads((run_dir / "private" / "split_manifest.json").read_text())
    assert set(split_manifest["test_group_ids"]) == EXPECTED_TEST_ALIASES
    assert split_manifest["unparsed_test_member_count"] == 3
    run_manifest = __import__("json").loads((run_dir / "private" / "run_manifest.json").read_text())
    assert run_manifest["external_archive_opened_this_run"] is False
    assert run_manifest["prior_project_external_exposure"]
    assert run_manifest["input_archive_name_matches_expected"] is False
    assert run_manifest["git_commit"]
    assert not (Path(__file__).resolve().parents[1] / "private-run").exists()


def test_cli_help_runs_from_a_source_checkout():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_fatigue_proxy.py"

    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "--prepare-training-zip" in result.stdout
