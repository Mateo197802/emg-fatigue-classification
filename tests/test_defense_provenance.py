import hashlib
import json
import zipfile

import pandas as pd
import pytest

from emg_fatigue.defense_provenance import verify_archive_matches_frozen_sources


def _write_frozen_source(root, member, payload):
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    input_hashes = [{"recording_id": "REC001", "sha256": digest}]
    input_hashes_path = root / "input_hashes.json"
    input_hashes_path.write_text(json.dumps(input_hashes), encoding="utf-8")
    pd.DataFrame([{"recording_id": "REC001", "source_member": member}]).to_csv(
        root / "source_mapping.csv", index=False
    )
    (root / "processing_receipt.json").write_text(
        json.dumps({"input_hashes_sha256": hashlib.sha256(input_hashes_path.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )


def test_archive_provenance_matches_each_frozen_recording_hash(tmp_path):
    member = "participant/session.csv"
    payload = b"time,emg\n0,1\n0.001,2\n"
    frozen = tmp_path / "frozen"
    _write_frozen_source(frozen, member, payload)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(member, payload)

    result = verify_archive_matches_frozen_sources(archive, frozen)

    assert result["recordings_verified"] == 1
    assert result["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert len(result["source_member_manifest_sha256"]) == 64


def test_archive_provenance_rejects_same_shape_different_source(tmp_path):
    member = "participant/session.csv"
    payload = b"time,emg\n0,1\n0.001,2\n"
    frozen = tmp_path / "frozen"
    _write_frozen_source(frozen, member, payload)
    archive = tmp_path / "different.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(member, b"time,emg\n0,9\n0.001,8\n")

    with pytest.raises(ValueError, match="does not match frozen source recording"):
        verify_archive_matches_frozen_sources(archive, frozen)


def test_archive_provenance_rejects_tampered_frozen_hash_receipt(tmp_path):
    member = "participant/session.csv"
    payload = b"time,emg\n0,1\n0.001,2\n"
    frozen = tmp_path / "frozen"
    _write_frozen_source(frozen, member, payload)
    (frozen / "input_hashes.json").write_text("[]", encoding="utf-8")
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(member, payload)

    with pytest.raises(ValueError, match="input_hashes receipt does not match"):
        verify_archive_matches_frozen_sources(archive, frozen)
