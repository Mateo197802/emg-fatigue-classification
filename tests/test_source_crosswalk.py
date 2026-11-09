import importlib
import importlib.util


def _builder():
    spec = importlib.util.find_spec("emg_fatigue.source_crosswalk")
    assert spec is not None, "auditable source crosswalk builder is missing"
    module = importlib.import_module("emg_fatigue.source_crosswalk")
    return module.build_source_crosswalk


def test_crosswalk_keeps_archive_records_separate_without_identity_keys():
    build = _builder()
    digest_a = "a" * 64
    digest_b = "b" * 64
    records = build(
        inventory_rows=[
            {
                "recording_id": "ZIP_REC0001",
                "subject_id": "folder-derived-01",
                "source_member": "group_a/recording.csv",
                "sha256": digest_a,
                "n_samples": "5000",
                "start_s": "0.001",
                "end_s": "5.0",
                "median_fs_hz": "1000.0",
            }
        ],
        rar_audit_rows=[
            {
                "archive_member": "P11_set01_dr.csv",
                "sha256": digest_b,
                "rows": 4000,
                "duration_s": 4.0,
                "fs_median_hz": 1000.0,
            }
        ],
        archive_hashes={"project_zip": "c" * 64, "raw_rar": "d" * 64},
    )

    assert len(records) == 2
    assert len({row["recording_key"] for row in records}) == 2
    assert {row["source_archive"] for row in records} == {"project_zip", "raw_rar"}
    assert all(row["participant_id_verified"] is False for row in records)
    assert all(row["cross_archive_identity_link"] == "unresolved" for row in records)
    assert "ZIP_REC0001" in {row["recording_key"] for row in records}
    assert any(row["recording_key"].startswith("RAW_RAR_") for row in records)


def test_exact_duplicate_bytes_are_flagged_without_claiming_same_person():
    build = _builder()
    digest = "e" * 64
    records = build(
        inventory_rows=[{"recording_id": "ZIP_REC0001", "subject_id": "folder-01",
                         "source_member": "group/one.csv", "sha256": digest}],
        rar_audit_rows=[{"archive_member": "P11_set01_dr.csv", "sha256": digest}],
        archive_hashes={"project_zip": "f" * 64, "raw_rar": "0" * 64},
    )

    assert all(row["same_sha256_in_other_archive"] is True for row in records)
    assert all(row["cross_archive_identity_link"] == "unresolved" for row in records)
