from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from emg_fatigue.weak_supervision.signal_windows import load_development_raw_recordings


def _csv_payload(value: int) -> bytes:
    rows = ["Relative_Time_s,ADC_Value,Voltage_mV"]
    rows.extend(f"{index / 1000:.3f},{value + index},{(value + index) * 0.8:.3f}" for index in range(8))
    return ("\n".join(rows) + "\n").encode("utf-8")


def test_development_loader_does_not_parse_reserved_test_payload(tmp_path: Path):
    dev_name = "development/record.csv"
    test_name = "test/record.csv"
    dev_payload = _csv_payload(1)
    # This payload would fail the numeric CSV parser if the reserved member were touched.
    test_payload = b"reserved test values are not readable as a training CSV\n"
    archive_path = tmp_path / "training.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(dev_name, dev_payload)
        archive.writestr(test_name, test_payload)

    private_dir = tmp_path / "private"
    private_dir.mkdir()
    members = []
    for group_id, recording_id, member_name, payload, parsed in (
        ("G01", "G01_F001", dev_name, dev_payload, True),
        ("G02", "G02_F001", test_name, test_payload, False),
    ):
        members.append(
            {
                "group_id": group_id,
                "recording_id": recording_id,
                "member_name": member_name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
                "header": ["Relative_Time_s", "ADC_Value", "Voltage_mV"],
                "parsed_numeric_values": parsed,
                "numeric_row_count": 8 if parsed else None,
            }
        )
    (private_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "input_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "members": members,
            }
        ),
        encoding="utf-8",
    )
    (private_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": 7,
                "development_group_ids": ["G01"],
                "test_group_ids": ["G02"],
                "development_members": [
                    {"group_id": "G01", "recording_id": "G01_F001", "member_name": dev_name}
                ],
                "test_members": [
                    {"group_id": "G02", "recording_id": "G02_F001", "member_name": test_name}
                ],
            }
        ),
        encoding="utf-8",
    )

    recordings, receipt = load_development_raw_recordings(archive_path, private_dir)

    assert [recording.recording_id for recording in recordings] == ["G01_F001"]
    assert recordings[0].adc_value.tolist() == list(range(1, 9))
    assert receipt["development_member_count"] == 1
    assert receipt["reserved_test_member_payloads_read"] == 0
