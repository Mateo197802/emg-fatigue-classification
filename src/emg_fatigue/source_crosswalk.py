"""Build a private source-to-recording crosswalk without inferring identity."""

import hashlib


def _valid_sha256(value, field):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a hexadecimal SHA-256 digest") from exc
    return value.lower()


def build_source_crosswalk(*, inventory_rows, rar_audit_rows, archive_hashes):
    """Return one row per observed source recording, keeping archives distinct.

    ``source_group_key`` preserves the existing folder-derived grouping key
    when available, but ``participant_id_verified`` is always false until an
    independent participant/session crosswalk is supplied.
    """
    expected_archives = {"project_zip", "raw_rar"}
    if set(archive_hashes) != expected_archives:
        raise ValueError("archive_hashes must identify project_zip and raw_rar")
    archive_hashes = {
        key: _valid_sha256(value, f"{key}_sha256")
        for key, value in archive_hashes.items()
    }
    records = []
    seen_recording_keys = set()

    for row in inventory_rows:
        recording_key = str(row.get("recording_id", "")).strip()
        member = str(row.get("source_member", "")).strip()
        if not recording_key or not member:
            raise ValueError("project ZIP inventory requires recording_id and source_member")
        content_hash = _valid_sha256(row.get("sha256"), "member_sha256")
        if recording_key in seen_recording_keys:
            raise ValueError(f"duplicate recording key: {recording_key}")
        seen_recording_keys.add(recording_key)
        records.append({
            "source_archive": "project_zip",
            "archive_sha256": archive_hashes["project_zip"],
            "source_member": member,
            "member_sha256": content_hash,
            "recording_key": recording_key,
            "source_group_key": str(row.get("subject_id", "")).strip() or "unavailable",
            "participant_id_verified": False,
            "cross_archive_identity_link": "unresolved",
            "modality": "sEMG",
            "unit": "ADC_count",
            "time_axis": "relative_time_s",
            "sample_count": row.get("n_samples", ""),
            "duration_s": row.get("end_s", ""),
            "sampling_rate_hz": row.get("median_fs_hz", ""),
            "calibration_state": "unverified",
            "synchronization_state": "independent_reference_not_found",
            "privacy": "private",
        })

    for row in rar_audit_rows:
        member = str(row.get("archive_member", "")).strip()
        if not member:
            raise ValueError("RAR audit requires archive_member")
        content_hash = _valid_sha256(row.get("sha256"), "member_sha256")
        recording_key = "RAW_RAR_" + hashlib.sha256(
            f"{member}\0{content_hash}".encode("utf-8")
        ).hexdigest()[:12]
        if recording_key in seen_recording_keys:
            raise ValueError(f"duplicate recording key: {recording_key}")
        seen_recording_keys.add(recording_key)
        records.append({
            "source_archive": "raw_rar",
            "archive_sha256": archive_hashes["raw_rar"],
            "source_member": member,
            "member_sha256": content_hash,
            "recording_key": recording_key,
            "source_group_key": "unverified_filename_code",
            "participant_id_verified": False,
            "cross_archive_identity_link": "unresolved",
            "modality": "sEMG",
            "unit": "ADC_count",
            "time_axis": "relative_time_s",
            "sample_count": row.get("rows", ""),
            "duration_s": row.get("duration_s", ""),
            "sampling_rate_hz": row.get("fs_median_hz", ""),
            "calibration_state": "unverified",
            "synchronization_state": "independent_reference_not_found",
            "privacy": "private",
        })

    hashes_by_archive = {
        archive: {
            row["member_sha256"]
            for row in records
            if row["source_archive"] == archive
        }
        for archive in expected_archives
    }
    for row in records:
        other_archive = "raw_rar" if row["source_archive"] == "project_zip" else "project_zip"
        row["same_sha256_in_other_archive"] = row["member_sha256"] in hashes_by_archive[other_archive]
    return sorted(records, key=lambda row: (row["source_archive"], row["source_member"]))
