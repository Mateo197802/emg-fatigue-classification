"""Verify that private defense figures use the frozen run's signal sources."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_archive_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive_matches_frozen_sources(archive_path: Path, frozen_root: Path) -> dict[str, object]:
    """Bind a supplied archive to the per-recording hashes in a frozen run.

    Container hashes are recorded, while every mapped recording is also
    checked against the immutable ``input_hashes.json`` used by the frozen
    analysis. This rejects same-shape archives with different signal bytes.
    """
    archive_path = Path(archive_path)
    frozen_root = Path(frozen_root)
    input_hashes_path = frozen_root / "input_hashes.json"
    mapping_path = frozen_root / "source_mapping.csv"
    processing_receipt_path = frozen_root / "processing_receipt.json"
    for path in (archive_path, input_hashes_path, mapping_path, processing_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    input_hash_bytes = input_hashes_path.read_bytes()
    processing_receipt = json.loads(processing_receipt_path.read_text(encoding="utf-8"))
    if processing_receipt.get("input_hashes_sha256") != hashlib.sha256(input_hash_bytes).hexdigest():
        raise ValueError("input_hashes receipt does not match the frozen input hash table")
    frozen_hashes = json.loads(input_hash_bytes.decode("utf-8"))
    if not isinstance(frozen_hashes, list) or not frozen_hashes:
        raise ValueError("frozen input hash table must be a nonempty list")
    expected = {}
    for row in frozen_hashes:
        if not isinstance(row, dict) or not row.get("recording_id") or not row.get("sha256"):
            raise ValueError("frozen input hash rows require recording_id and sha256")
        recording_id = str(row["recording_id"])
        digest = str(row["sha256"]).lower()
        if recording_id in expected:
            raise ValueError(f"duplicate recording_id in frozen input hashes: {recording_id}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid frozen SHA-256 for {recording_id}")
        expected[recording_id] = digest

    with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
        mapping_rows = list(csv.DictReader(stream))
    if not mapping_rows or not {"recording_id", "source_member"}.issubset(mapping_rows[0]):
        raise ValueError("source mapping must include recording_id and source_member")
    member_by_id = {}
    for row in mapping_rows:
        recording_id = str(row["recording_id"])
        member = str(row["source_member"])
        if not recording_id or not member or recording_id in member_by_id:
            raise ValueError("source mapping contains an empty or duplicate recording/member mapping")
        member_by_id[recording_id] = member
    if set(member_by_id) != set(expected):
        raise ValueError("source mapping IDs do not match the frozen input hash table")
    if len(set(member_by_id.values())) != len(member_by_id):
        raise ValueError("source mapping contains duplicate archive member paths")

    verified_rows = []
    with zipfile.ZipFile(archive_path, "r") as source:
        infos = [info for info in source.infolist() if not info.is_dir()]
        counts = Counter(info.filename for info in infos)
        duplicated = sorted(name for name, count in counts.items() if count > 1)
        if duplicated:
            raise ValueError(f"source archive contains duplicate member paths: {duplicated[:3]}")
        info_by_name = {info.filename: info for info in infos}
        for recording_id, member in sorted(member_by_id.items()):
            info = info_by_name.get(member)
            if info is None:
                raise ValueError(f"source archive is missing frozen recording member: {member}")
            actual = _sha256_archive_member(source, info)
            if actual != expected[recording_id]:
                raise ValueError(
                    f"source archive does not match frozen source recording {recording_id} ({member})"
                )
            verified_rows.append({
                "recording_id": recording_id,
                "source_member": member,
                "sha256": actual,
            })

    manifest = json.dumps(
        verified_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "archive_sha256": _sha256_file(archive_path),
        "source_member_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "recordings_verified": len(verified_rows),
    }
