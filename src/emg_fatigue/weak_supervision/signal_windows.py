"""Development-only loading and materialization of processed sEMG windows."""

from __future__ import annotations

import hashlib
import json
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from .activity import detect_candidate_activity
from .contracts import RawRecording
from .features import window_key
from .io import REQUIRED_COLUMNS, _parse_development_record
from .preprocess import preprocess_recording
from .windows import make_windows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(private_dir: Path, name: str) -> Path:
    path = Path(private_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"Required private development manifest is missing: {path.name}")
    return path


def load_development_raw_recordings(
    archive_path: Path,
    private_dir: Path,
) -> tuple[list[RawRecording], dict[str, Any]]:
    """Read only members explicitly listed as development in the split manifest.

    This function deliberately does not call ``ZipFile.testzip`` or iterate over
    member contents. It reads the archive central directory, verifies the whole
    archive hash, and opens only the allow-listed development members.
    """
    archive_path = Path(archive_path).expanduser().resolve()
    private_dir = Path(private_dir).expanduser().resolve()
    if archive_path.suffix.casefold() != ".zip":
        raise ValueError("Development signal input must be the training ZIP archive")
    if not archive_path.is_file():
        raise FileNotFoundError("Development training ZIP was not found")

    input_manifest_path = _manifest_path(private_dir, "input_manifest.json")
    split_manifest_path = _manifest_path(private_dir, "split_manifest.json")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    expected_archive_hash = str(input_manifest.get("input_archive_sha256", ""))
    observed_archive_hash = _sha256_file(archive_path)
    if not expected_archive_hash or observed_archive_hash != expected_archive_hash:
        raise ValueError("Training ZIP SHA-256 does not match the frozen private input manifest")

    development_group_ids = tuple(sorted(str(value) for value in split_manifest.get("development_group_ids", ())))
    if not development_group_ids:
        raise ValueError("Split manifest contains no development source groups")
    development_members = split_manifest.get("development_members")
    if not isinstance(development_members, list) or not development_members:
        raise ValueError("Split manifest contains no allow-listed development archive members")
    test_members = split_manifest.get("test_members", ())
    test_names = {str(item["member_name"]) for item in test_members if isinstance(item, dict) and "member_name" in item}

    records_by_name: dict[str, dict[str, Any]] = {}
    for record in input_manifest.get("members", ()):
        name = str(record.get("member_name", ""))
        if not name or name in records_by_name:
            raise ValueError("Input manifest contains an empty or duplicated archive member name")
        records_by_name[name] = record

    allow_list: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_recording_ids: set[str] = set()
    for item in development_members:
        if not isinstance(item, dict):
            raise ValueError("Development member manifest entry is malformed")
        group_id = str(item.get("group_id", ""))
        recording_id = str(item.get("recording_id", ""))
        member_name = str(item.get("member_name", ""))
        path_parts = PurePosixPath(member_name)
        if path_parts.is_absolute() or ".." in path_parts.parts or len(path_parts.parts) < 2:
            raise ValueError("Split manifest contains an unsafe development member path")
        if group_id not in development_group_ids or member_name in test_names:
            raise ValueError("Development allow-list includes a reserved test member or group")
        if member_name in seen_names or recording_id in seen_recording_ids:
            raise ValueError("Split manifest contains duplicate development member identities")
        source = records_by_name.get(member_name)
        if source is None:
            raise ValueError("Development member is absent from the frozen input manifest")
        if str(source.get("group_id")) != group_id or str(source.get("recording_id")) != recording_id:
            raise ValueError("Development member identity disagrees with the frozen input manifest")
        parsed = bool(source.get("parsed_numeric_values", source.get("numeric_values_parsed", False)))
        if not parsed:
            raise ValueError("Development member was not marked as numerically parsed in the input manifest")
        if tuple(source.get("header", ())) != REQUIRED_COLUMNS:
            raise ValueError("Development member header disagrees with the required raw CSV schema")
        seen_names.add(member_name)
        seen_recording_ids.add(recording_id)
        allow_list.append({**item, "expected_sha256": str(source.get("sha256", "")), "expected_byte_size": int(source.get("byte_size", -1))})

    input_development_names = {
        str(record["member_name"])
        for record in input_manifest.get("members", ())
        if str(record.get("group_id")) in development_group_ids
        and bool(record.get("parsed_numeric_values", record.get("numeric_values_parsed", False)))
    }
    if seen_names != input_development_names:
        raise ValueError("Split allow-list does not exactly cover parsed development members")

    recordings: list[RawRecording] = []
    read_receipts: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for item in sorted(allow_list, key=lambda row: (str(row["group_id"]), str(row["recording_id"]))):
            member_name = str(item["member_name"])
            try:
                info = archive.getinfo(member_name)
            except KeyError as error:
                raise ValueError("Allow-listed development member is missing from the ZIP central directory") from error
            if info.is_dir() or info.file_size != int(item["expected_byte_size"]):
                raise ValueError("Development ZIP member type or size disagrees with the frozen input manifest")
            # The only archive payload read in this loop is a member on the development allow-list.
            with archive.open(info, "r") as stream:
                payload = stream.read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != item["expected_sha256"]:
                raise ValueError("Development ZIP member SHA-256 does not match the frozen input manifest")
            recording, row_count = _parse_development_record(
                payload,
                str(item["group_id"]),
                str(item["recording_id"]),
                digest,
                member_name,
            )
            recordings.append(recording)
            read_receipts.append(
                {
                    "group_id": recording.group_id,
                    "recording_id": recording.recording_id,
                    "member_name": member_name,
                    "sha256": digest,
                    "byte_size": len(payload),
                    "numeric_row_count": int(row_count),
                }
            )

    receipt: dict[str, Any] = {
        "input_archive_sha256": observed_archive_hash,
        "development_group_ids": list(development_group_ids),
        "development_member_count": len(recordings),
        "development_member_receipts": read_receipts,
        "reserved_test_member_count": len(test_names),
        "reserved_test_member_payloads_read": 0,
        "internal_test_signal_values_parsed": 0,
        "external_rar_opened": False,
    }
    return recordings, receipt


def build_development_signal_window_bank(
    archive_path: Path,
    private_dir: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Rebuild primary 500 ms development signals and join frozen weak labels."""
    archive_path = Path(archive_path).expanduser().resolve()
    private_dir = Path(private_dir).expanduser().resolve()
    output_dir = Path(output_dir).resolve() if output_dir else private_dir / "neural_data"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing private neural data: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    config_path = _manifest_path(private_dir, "fatigue_proxy_config.toml")
    feature_dir = private_dir / "features_labels"
    feature_path = feature_dir / "features.parquet"
    label_path = feature_dir / "pseudo_labels.parquet"
    for required in (feature_path, label_path):
        if not required.is_file():
            raise FileNotFoundError(f"Frozen development table is missing: {required.name}")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    primary_window_s = float(config["run"]["primary_window_s"])
    overlap = float(config["run"]["window_overlap"])
    minimum_occupancy = float(config["activity"]["primary_window_occupancy"])
    recordings, read_receipt = load_development_raw_recordings(archive_path, private_dir)

    metadata_rows: list[dict[str, object]] = []
    waveform_rows: list[np.ndarray] = []
    for recording in recordings:
        processed = preprocess_recording(recording, config)
        activity = detect_candidate_activity(processed, config)
        for window in make_windows(
            processed,
            activity,
            primary_window_s,
            overlap,
            minimum_occupancy=minimum_occupancy,
        ):
            key = window_key(window)
            metadata_rows.append(
                {
                    "window_key": key,
                    "group_id": window.group_id,
                    "recording_id": window.recording_id,
                    "segment_index": int(window.segment_index),
                    "window_index": int(window.window_index),
                    "start_sample": int(window.start_sample),
                    "start_time_s": float(window.start_time_s),
                    "duration_s": float(window.duration_s),
                    "sampling_rate_hz": float(window.sampling_rate_hz),
                    "activity_occupancy": float(window.activity_occupancy),
                    "eligible_for_label": bool(window.eligible_for_label),
                    "rejection_reason": window.rejection_reason or "",
                    "source_sha256": recording.source_sha256,
                }
            )
            waveform_rows.append(np.asarray(window.signal, dtype=np.float32))

    if not metadata_rows:
        raise ValueError("No development signal windows were reconstructed")
    metadata = pd.DataFrame(metadata_rows)
    if metadata.window_key.duplicated().any():
        raise ValueError("Reconstructed signal bank contains duplicate window keys")
    signal_array = np.stack(waveform_rows).astype(np.float32, copy=False)
    expected_samples = int(round(primary_window_s * float(config["acquisition"]["target_sampling_rate_hz"])))
    if signal_array.ndim != 2 or signal_array.shape[1] != expected_samples:
        raise ValueError("Reconstructed signal windows do not match the configured duration and sample rate")

    features = pd.read_parquet(feature_path)
    expected_windows = features.loc[
        np.isclose(features.duration_s.to_numpy(dtype=float), primary_window_s),
        ["window_key", "group_id", "recording_id", "window_index", "feature_valid"],
    ].copy()
    if set(metadata.window_key.astype(str)) != set(expected_windows.window_key.astype(str)):
        missing = set(expected_windows.window_key.astype(str)) - set(metadata.window_key.astype(str))
        extra = set(metadata.window_key.astype(str)) - set(expected_windows.window_key.astype(str))
        raise ValueError(f"Reconstructed development window keys differ from frozen features (missing={len(missing)}, extra={len(extra)})")
    aligned = metadata.merge(
        expected_windows,
        on="window_key",
        how="inner",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )
    if not aligned.group_id.astype(str).eq(aligned.group_id_feature.astype(str)).all():
        raise ValueError("Reconstructed group IDs differ from the frozen feature table")
    if not aligned.recording_id.astype(str).eq(aligned.recording_id_feature.astype(str)).all():
        raise ValueError("Reconstructed recording IDs differ from the frozen feature table")
    if not aligned.window_index.astype(int).eq(aligned.window_index_feature.astype(int)).all():
        raise ValueError("Reconstructed window indices differ from the frozen feature table")
    labels = pd.read_parquet(label_path)
    label_columns = [name for name in ("window_key", "class_id", "label_confidence", "rule_version", "label_reason") if name in labels]
    metadata = metadata.merge(labels[label_columns], on="window_key", how="left", validate="one_to_one")
    if metadata.class_id.isna().any():
        raise ValueError("Some reconstructed development signals have no frozen pseudo-label row")

    bank_path = output_dir / "signals_500ms_development.npz"
    metadata_path = output_dir / "signal_window_manifest.parquet"
    np.savez_compressed(
        bank_path,
        signals=signal_array,
        window_keys=metadata.window_key.astype(str).to_numpy(dtype=np.str_),
    )
    metadata.to_parquet(metadata_path, index=False)

    input_paths = [
        archive_path,
        private_dir / "input_manifest.json",
        private_dir / "split_manifest.json",
        config_path,
        feature_path,
        label_path,
    ]
    output_paths = [bank_path, metadata_path]
    receipt = {
        "stage": "development_signal_window_bank",
        "status": "completed",
        "input_archive_sha256": read_receipt["input_archive_sha256"],
        "development_group_ids": read_receipt["development_group_ids"],
        "development_record_count": read_receipt["development_member_count"],
        "development_window_count": int(len(metadata)),
        "signal_shape": list(signal_array.shape),
        "window_duration_s": primary_window_s,
        "window_overlap": overlap,
        "sampling_rate_hz": float(config["acquisition"]["target_sampling_rate_hz"]),
        "pseudo_label_rule": str(metadata.rule_version.dropna().iloc[0]) if "rule_version" in metadata and metadata.rule_version.notna().any() else "unspecified",
        "assigned_proxy_windows_descriptive_only": int(metadata.class_id.isin((0, 1)).sum()),
        "ambiguous_proxy_windows": int(metadata.class_id.eq(-1).sum()),
        "reserved_test_member_count": read_receipt["reserved_test_member_count"],
        "reserved_test_member_payloads_read": 0,
        "internal_test_signal_values_parsed": 0,
        "external_rar_opened": False,
        "overlapping_windows_treated_as_independent_groups": False,
        "input_sha256": {str(path): _sha256_file(path) for path in input_paths},
        "output_sha256": {str(path): _sha256_file(path) for path in output_paths},
    }
    receipt_path = output_dir / "signal_window_bank_manifest.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return {**receipt, "manifest_path": str(receipt_path.resolve())}
