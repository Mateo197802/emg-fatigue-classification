"""Private raw inventory and development/test split preparation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import load_training_zip

PRIOR_EXTERNAL_EXPOSURE = (
    "The same supplied RAR was inspected during a prior project experiment after that experiment's freeze; "
    "it is not pristine at project level and participant independence remains unverified."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": os.sys.version.split()[0]}
    for import_name, distribution_name in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit_learn", "scikit-learn"),
        ("torch", "torch"),
        ("pyarrow", "pyarrow"),
        ("PyWavelets", "PyWavelets"),
        ("joblib", "joblib"),
    ):
        try:
            versions[import_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[import_name] = None
    return versions


def _source_fingerprint(repo_root: Path) -> dict[str, object]:
    package_dir = repo_root / "src" / "emg_fatigue" / "weak_supervision"
    source_paths = sorted(package_dir.glob("*.py")) + [repo_root / "tools" / "run_fatigue_proxy.py"]
    hashes: dict[str, str] = {}
    for source_path in source_paths:
        if source_path.is_file():
            hashes[source_path.relative_to(repo_root).as_posix()] = _sha256_file(source_path)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    return {"sha256": digest, "files": hashes}


def _output_hashes(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            result[path.relative_to(run_dir).as_posix()] = _sha256_file(path)
    return result


def prepare_training_run(
    training_zip: Path,
    run_dir: Path,
    config_path: Path,
) -> dict[str, object]:
    """Inventory all archive bytes, parse development signals only, and persist private receipts."""
    repo_root = _repo_root().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    training_zip = Path(training_zip).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    if run_dir == repo_root or repo_root in run_dir.parents:
        raise ValueError("Restricted run outputs must be stored outside the Git checkout.")
    if not config_path.is_file():
        raise FileNotFoundError("Fatigue proxy configuration was not found.")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    run_config = config["run"]

    recordings, inventory = load_training_zip(
        training_zip,
        seed=int(run_config["split_seed"]),
        train_fraction=float(run_config["train_fraction"]),
    )
    expected_csv_count = int(run_config.get("expected_training_csv_count", inventory.csv_count))
    expected_group_count = int(run_config.get("expected_source_group_count", inventory.source_group_count))
    if inventory.csv_count != expected_csv_count or inventory.source_group_count != expected_group_count:
        raise ValueError("Training archive counts do not match the frozen run configuration.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("The private run directory already contains files.")
    run_dir.mkdir(parents=True, exist_ok=True)
    private = run_dir / "private"
    private.mkdir(parents=True, exist_ok=True)

    config_copy = private / "fatigue_proxy_config.toml"
    config_copy.write_bytes(config_path.read_bytes())
    inventory_dict = inventory.to_private_dict()
    _write_json(private / "input_manifest.json", inventory_dict)
    _write_json(private / "source_group_map.json", inventory.source_group_map)
    test_groups = set(inventory.split.test_groups)
    split_manifest = {
        "seed": inventory.split.seed,
        "train_fraction": inventory.split.train_fraction,
        "development_group_ids": list(inventory.split.development_groups),
        "test_group_ids": list(inventory.split.test_groups),
        "method": inventory.split.method,
        "identity_verified": False,
        "development_members": [
            {"group_id": row.group_id, "recording_id": row.recording_id, "member_name": row.member_name}
            for row in inventory.members
            if row.group_id in set(inventory.split.development_groups)
        ],
        "test_members": [
            {
                "group_id": row.group_id,
                "recording_id": row.recording_id,
                "member_name": row.member_name,
                "sha256": row.sha256,
                "numeric_values_parsed": row.parsed_numeric_values,
            }
            for row in inventory.members
            if row.group_id in test_groups
        ],
        "unparsed_test_member_count": inventory.unparsed_internal_test_member_count,
    }
    _write_json(private / "split_manifest.json", split_manifest)
    _write_json(private / "runtime_versions.json", _runtime_versions())

    source = _source_fingerprint(repo_root)
    run_manifest: dict[str, Any] = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "fresh_raw_inventory_development_values_only",
        "input_archive_sha256": inventory.input_archive_sha256,
        "config_sha256": _sha256_file(config_path),
        "input_archive_name_matches_expected": inventory.input_archive_name
        == run_config.get("expected_training_archive_name"),
        "csv_count": inventory.csv_count,
        "source_group_count": inventory.source_group_count,
        "development_record_count": len(recordings),
        "development_group_count": len(inventory.split.development_groups),
        "internal_test_group_count": len(inventory.split.test_groups),
        "unparsed_internal_test_member_count": inventory.unparsed_internal_test_member_count,
        "external_archive_opened_this_run": False,
        "prior_project_external_exposure": PRIOR_EXTERNAL_EXPOSURE,
        "prior_features_labels_or_checkpoints_reused": False,
        "git_branch": _git_value(repo_root, "branch", "--show-current"),
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_working_tree_status": _git_value(repo_root, "status", "--short"),
        "source_fingerprint_sha256": source["sha256"],
        "source_fingerprint_files": source["files"],
        "software_versions": _runtime_versions(),
    }
    run_manifest["output_sha256"] = _output_hashes(run_dir)
    _write_json(private / "run_manifest.json", run_manifest)
    return {
        "run_id": run_dir.name,
        "csv_count": inventory.csv_count,
        "source_group_count": inventory.source_group_count,
        "development_record_count": len(recordings),
        "development_group_count": len(inventory.split.development_groups),
        "internal_test_group_count": len(inventory.split.test_groups),
        "unparsed_internal_test_member_count": inventory.unparsed_internal_test_member_count,
        "run_manifest_path": str(private / "run_manifest.json"),
    }
