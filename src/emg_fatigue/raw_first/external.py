"""Final-only inspection of the locked external RAR after training freeze."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import locale
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch

from .audit import audit_recording
from .contracts import RawRecording
from .io import raw_first_source_fingerprint, sha256_file, write_json
from .preprocessing import preprocess_recording
from .runtime import software_versions
from .ssl import MaskedConvAutoencoder, apply_signal_scale, model_embedding, model_reconstruct
from .windows import segment_windows

MAX_EXTERNAL_MEMBERS = 1000
MAX_EXTERNAL_MEMBER_BYTES = 64 * 1024 * 1024
MAX_EXTERNAL_TOTAL_BYTES = 256 * 1024 * 1024
MAX_EXTERNAL_LISTING_BYTES = 8 * 1024 * 1024


def _find_7z() -> str | None:
    for command in ("7z", "7za", "7zz"):
        path = shutil.which(command)
        if path:
            return path
    candidates = (
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _find_unrar() -> str | None:
    path = shutil.which("unrar")
    if path:
        return path
    candidates = (
        Path(r"C:\Program Files\WinRAR\UnRAR.exe"),
        Path(r"C:\Program Files (x86)\WinRAR\UnRAR.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _list_rar_members(executable: str, archive: Path, tool: str) -> list[str]:
    if tool == "unrar":
        # UnRAR's -inul switch suppresses the member listing itself.
        args = [executable, "lb", "-p-", str(archive)]
    else:
        args = [executable, "l", "-slt", str(archive)]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    listing = bytearray()
    try:
        if process.stdout is None:
            raise RuntimeError("RAR member listing pipe could not be opened.")
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            if len(listing) + len(chunk) > MAX_EXTERNAL_LISTING_BYTES:
                process.kill()
                process.wait()
                raise ValueError("The external RAR member listing exceeded the configured size limit.")
            listing.extend(chunk)
        return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0:
        message = "UnRAR could not list the external archive." if tool == "unrar" else "The available RAR utility could not list the external archive."
        raise RuntimeError(message)
    text = bytes(listing).decode(locale.getpreferredencoding(False), errors="replace")
    if tool == "unrar":
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().endswith(("/", "\\"))
        ]
    members: list[str] = []
    for section in re.split(r"\r?\n\s*\r?\n", text):
        fields = {}
        for line in section.splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                fields[key.strip()] = value.strip()
        value = fields.get("Path")
        if value and value != str(archive) and "Size" in fields and fields.get("Folder") != "+":
            members.append(value)
    return members


def _read_rar_member(executable: str, archive: Path, member: str, tool: str, max_bytes: int) -> bytes:
    if tool == "unrar":
        # -inul also suppresses streamed member bytes with this UnRAR build.
        args = [executable, "p", "-p-", str(archive), member]
    else:
        args = [executable, "x", "-so", "-bso0", "-bsp0", str(archive), member]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    payload = bytearray()
    try:
        if process.stdout is None:
            raise RuntimeError("RAR member output pipe could not be opened.")
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            if len(payload) + len(chunk) > max_bytes:
                process.kill()
                process.wait()
                raise ValueError("A RAR CSV member exceeded the configured private inspection size limit.")
            payload.extend(chunk)
        return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0:
        message = "UnRAR could not stream a CSV member from the locked external RAR." if tool == "unrar" else "A CSV member could not be streamed from the locked external RAR."
        raise RuntimeError(message)
    return bytes(payload)


def _verify_frozen_run(private: Path, freeze: dict[str, object]) -> None:
    """Recompute frozen configuration and encoder hashes before touching the RAR."""
    manifest_path = private / "run_manifest.json"
    configuration_path = private / "frozen_configuration.json"
    if not manifest_path.is_file() or not configuration_path.is_file():
        raise ValueError("The frozen run is missing its manifest or configuration receipt.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    if configuration_hash != freeze.get("configuration_sha256"):
        raise ValueError("Frozen configuration hash does not match its freeze receipt.")
    if sha256_file(configuration_path) != manifest.get("output_sha256", {}).get("private/frozen_configuration.json"):
        raise ValueError("Frozen configuration does not match the private run manifest.")
    if manifest.get("input_archive_sha256") != freeze.get("input_archive_sha256"):
        raise ValueError("Training input hash does not match the private run manifest and freeze receipt.")
    if manifest.get("external_rar_opened") is not False:
        raise ValueError("The private run manifest records that the external archive was already opened.")

    frozen_versions = freeze.get("software_versions")
    recorded_versions = manifest.get("software_versions")
    current_versions = software_versions()
    if not isinstance(frozen_versions, dict) or not isinstance(recorded_versions, dict):
        raise ValueError("Runtime version provenance is missing from the freeze receipt or run manifest.")
    if frozen_versions != recorded_versions or current_versions != frozen_versions:
        expected = frozen_versions
        mismatches = sorted(
            key
            for key in set(expected) | set(recorded_versions) | set(current_versions)
            if expected.get(key) != recorded_versions.get(key) or expected.get(key) != current_versions.get(key)
        )
        raise ValueError(
            "Runtime versions changed since the frozen training run: " + ", ".join(mismatches)
        )

    scaler_path = private / "signal_standardization.json"
    scaler_hash = sha256_file(scaler_path) if scaler_path.is_file() else None
    if (
        scaler_hash is None
        or scaler_hash != freeze.get("signal_standardization_sha256")
        or scaler_hash != manifest.get("signal_standardization_sha256")
        or scaler_hash != manifest.get("output_sha256", {}).get("private/signal_standardization.json")
    ):
        raise ValueError("The learned signal scaler does not match the freeze receipt and private run manifest.")

    source_fingerprint = raw_first_source_fingerprint()
    if (
        source_fingerprint["sha256"] != freeze.get("external_stage_code_sha256")
        or source_fingerprint["sha256"] != manifest.get("external_stage_code_sha256")
        or source_fingerprint["files"] != freeze.get("external_stage_code_files")
        or source_fingerprint["files"] != manifest.get("external_stage_code_files")
    ):
        raise ValueError("Raw-first source fingerprint does not match the freeze receipt and run manifest.")
    recorded_commit = freeze.get("external_stage_code_commit")
    if not isinstance(recorded_commit, str) or manifest.get("external_stage_code_commit") != recorded_commit:
        raise ValueError("External-stage source commit does not match the freeze receipt and run manifest.")
    try:
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("The current source-control revision could not be verified before external inspection.") from exc
    if current_commit != recorded_commit:
        raise ValueError("The current source-control revision differs from the frozen external-stage revision.")

    recorded = freeze.get("encoder_checkpoint_sha256")
    if not isinstance(recorded, dict) or set(recorded) != {"clean", "augmented"}:
        raise ValueError("Freeze receipt does not contain both encoder checkpoint hashes.")
    output_hashes = manifest.get("output_sha256", {})
    for condition in ("clean", "augmented"):
        checkpoint = private / "models" / f"encoder_{condition}.pt"
        relative = f"private/models/encoder_{condition}.pt"
        expected = recorded.get(condition)
        if not checkpoint.is_file() or not isinstance(expected, str):
            raise ValueError("A frozen encoder checkpoint is missing from the run.")
        actual = sha256_file(checkpoint)
        if actual != expected or output_hashes.get(relative) != expected:
            raise ValueError("An encoder checkpoint hash does not match the freeze receipt and run manifest.")


def _parse_csv(payload: bytes, group_id: str, recording_id: str, digest: str) -> RawRecording:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("An external CSV is not valid UTF-8 text.") from exc
    reader = csv.DictReader(io.StringIO(text))
    expected = ("Relative_Time_s", "ADC_Value", "Voltage_mV")
    if tuple(reader.fieldnames or ()) != expected:
        raise ValueError("An external CSV does not match the frozen three-column input schema.")
    times: list[float] = []
    adc: list[float] = []
    voltage: list[float] = []
    for row in reader:
        try:
            times.append(float(row["Relative_Time_s"]))
            adc.append(float(row["ADC_Value"]))
            voltage.append(float(row["Voltage_mV"]))
        except (TypeError, ValueError):
            times.append(float("nan"))
            adc.append(float("nan"))
            voltage.append(float("nan"))
    return RawRecording(
        group_id=group_id,
        recording_id=recording_id,
        relative_time_s=np.asarray(times, dtype=np.float64),
        adc_value=np.asarray(adc, dtype=np.float64),
        voltage_mv=np.asarray(voltage, dtype=np.float64),
        source_sha256=digest,
    )


def _assign_external_groups(member_names: list[str]) -> tuple[dict[str, str], list[str]]:
    raw_groups: dict[str, list[str]] = {}
    for name in member_names:
        parts = PurePosixPath(name.replace("\\", "/")).parts
        match = next((part.upper() for part in parts if re.fullmatch(r"P\d{2}", part, flags=re.IGNORECASE)), None)
        key = match or (parts[0] if len(parts) > 1 else "external_root")
        raw_groups.setdefault(key, []).append(name)
    mapping: dict[str, str] = {}
    for index, raw_group in enumerate(sorted(raw_groups, key=lambda item: (item.casefold(), item)), start=1):
        if raw_group.upper() in {"P11", "P12"}:
            group_id = raw_group.upper()
        else:
            group_id = f"EX{index:02d}"
        for member in raw_groups[raw_group]:
            mapping[member] = group_id
    return mapping, sorted(set(mapping.values()))


def _load_model(path: Path) -> MaskedConvAutoencoder:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = MaskedConvAutoencoder().cpu().eval()
    model.load_state_dict(bundle["state_dict"])
    return model


def _record_blocked_external(run_root: Path, output_dir: Path, status: str, archive_hash: str, contents_opened: bool) -> None:
    manifest_path = run_root / "private" / "run_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_stage_status"] = status
    manifest["external_archive_sha256"] = archive_hash
    manifest["external_rar_opened"] = bool(contents_opened)
    manifest["external_output_sha256"] = {
        str(path.relative_to(run_root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    manifest.setdefault("output_sha256", {}).update(manifest["external_output_sha256"])
    write_json(manifest_path, manifest)


def finalize_external(rar_path: Path, run_root: Path) -> dict[str, object]:
    """Inspect the RAR only when a completed frozen training run is present."""
    rar_path = rar_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    private = run_root / "private"
    freeze_path = private / "freeze.json"
    if not freeze_path.is_file():
        raise ValueError("External input is locked until the private training freeze receipt exists.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_training_configuration_and_self_supervised_encoders" or freeze.get("external_rar_opened") is not False:
        raise ValueError("External input freeze receipt is invalid or has already been used.")
    _verify_frozen_run(private, freeze)
    if rar_path.suffix.casefold() != ".rar" or not rar_path.is_file():
        raise ValueError("Final external input must be the supplied RAR archive.")

    output_dir = private / "external"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_hash = sha256_file(rar_path)
    executable = _find_7z()
    tool = "7z"
    if executable is None:
        executable = _find_unrar()
        tool = "unrar"
    if executable is None:
        result: dict[str, object] = {
            "status": "blocked_rar_extractor_unavailable",
            "actual_archive_name": rar_path.name,
            "expected_archive_name": "raw(2).rar",
            "archive_name_matches_objective": rar_path.name.casefold() == "raw(2).rar",
            "archive_sha256": archive_hash,
            "rar_test_passed": False,
            "csv_count": None,
            "independent_labels_found": None,
            "reference_search_complete": False,
            "fatigue_metrics_produced": False,
            "identity_independence_verified": False,
            "limitations": ["No 7-Zip/UnRAR extractor is available; the RAR contents were not inspected."],
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output_dir / "external_audit.json", result)
        _record_blocked_external(run_root, output_dir, str(result["status"]), archive_hash, False)
        return result

    test_args = [executable, "t", "-inul", "-p-", str(rar_path)] if tool == "unrar" else [executable, "t", str(rar_path)]
    test = subprocess.run(test_args, capture_output=True, text=True, errors="replace", check=False)
    if test.returncode != 0:
        result = {
            "status": "blocked_rar_integrity_check_failed",
            "actual_archive_name": rar_path.name,
            "expected_archive_name": "raw(2).rar",
            "archive_name_matches_objective": rar_path.name.casefold() == "raw(2).rar",
            "archive_sha256": archive_hash,
            "rar_test_passed": False,
            "csv_count": None,
            "independent_labels_found": None,
            "reference_search_complete": False,
            "fatigue_metrics_produced": False,
            "identity_independence_verified": False,
            "limitations": ["The RAR integrity test failed; no member data were used."],
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output_dir / "external_audit.json", result)
        _record_blocked_external(run_root, output_dir, str(result["status"]), archive_hash, False)
        return result

    member_names = _list_rar_members(executable, rar_path, tool)
    if len(member_names) > MAX_EXTERNAL_MEMBERS:
        raise ValueError("The external RAR exceeds the configured member-count limit.")
    csv_names = [name for name in member_names if PurePosixPath(name.replace("\\", "/")).suffix.casefold() == ".csv"]
    non_csv_names = [name for name in member_names if name not in csv_names]
    group_map, group_ids = _assign_external_groups(csv_names)
    records: list[RawRecording] = []
    private_hashes: list[dict[str, str]] = []
    grouped_indices: dict[str, int] = {}
    candidate_label_files: list[dict[str, object]] = []
    total_csv_bytes = 0
    for name in sorted(csv_names):
        group_id = group_map[name]
        grouped_indices[group_id] = grouped_indices.get(group_id, 0) + 1
        recording_id = f"{group_id}_F{grouped_indices[group_id]:03d}"
        remaining_bytes = MAX_EXTERNAL_TOTAL_BYTES - total_csv_bytes
        if remaining_bytes <= 0:
            raise ValueError("The external RAR exceeded the configured total CSV size limit.")
        content = _read_rar_member(
            executable,
            rar_path,
            name,
            tool,
            max_bytes=min(MAX_EXTERNAL_MEMBER_BYTES, remaining_bytes),
        )
        total_csv_bytes += len(content)
        digest = hashlib.sha256(content).hexdigest()
        private_hashes.append({"group_id": group_id, "recording_id": recording_id, "sha256": digest})
        try:
            records.append(_parse_csv(content, group_id, recording_id, digest))
        except ValueError:
            try:
                header = next(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
            except (UnicodeDecodeError, StopIteration):
                header = []
            header_lower = {value.strip().casefold() for value in header}
            label_like = bool(header_lower & {"fatigue", "borg", "borg_cr10", "mvc", "force", "torque", "phase", "extension", "flexion", "label"})
            candidate_label_files.append({"opaque_member_id": recording_id, "header": header, "reference_column_candidate": label_like})

    audits = [audit_recording(recording) for recording in records]
    training_config = json.loads((private / "frozen_configuration.json").read_text(encoding="utf-8"))
    plan = training_config["sampling_plan"]
    run_config = training_config["config"]["run"]
    by_recording = {str(row["recording_id"]): row for row in audits}
    all_windows = []
    incompatible = []
    for recording in records:
        audit = by_recording[recording.recording_id]
        if float(audit["nyquist_hz"]) < float(plan["bandpass_high_hz"]):
            incompatible.append(recording.recording_id)
            continue
        segments, _ = preprocess_recording(recording, audit, plan, filter_order=int(run_config["filter_order"]))
        windows, _ = segment_windows(segments, float(run_config["primary_window_s"]), float(run_config["window_overlap"]))
        all_windows.extend(windows)

    output_embeddings: dict[str, np.ndarray] = {
        "group_id": np.asarray([window.group_id for window in all_windows]),
        "recording_id": np.asarray([window.recording_id for window in all_windows]),
        "start_time_s": np.asarray([window.start_time_s for window in all_windows], dtype=np.float64),
    }
    if all_windows:
        signal_scale = json.loads((private / "signal_standardization.json").read_text(encoding="utf-8"))
        signals = np.stack([window.signal for window in all_windows]).astype(np.float32)
        normalized = apply_signal_scale(signals, signal_scale)
        for condition in ("clean", "augmented"):
            checkpoint = private / "models" / f"encoder_{condition}.pt"
            model = _load_model(checkpoint)
            output_embeddings[f"embedding_{condition}"] = model_embedding(model, normalized)
            reconstructions = model_reconstruct(model, normalized)
            output_embeddings[f"reconstruction_mse_{condition}"] = np.mean((reconstructions - normalized) ** 2, axis=1)
        np.savez_compressed(output_dir / "exploratory_embeddings.npz", **output_embeddings)
    else:
        np.savez_compressed(output_dir / "exploratory_embeddings.npz", **output_embeddings)

    csv_count = len(csv_names)
    schema_compatible = csv_count > 0 and len(records) == csv_count
    reference_search_complete = csv_count > 0 and len(non_csv_names) == 0 and len(records) == csv_count and not candidate_label_files
    independent_labels_found: bool | None = False if reference_search_complete else None
    group_counts = {group: sum(record.group_id == group for record in records) for group in group_ids}
    result = {
        "status": "held_out_file_analysis_no_frozen_supervised_model",
        "actual_archive_name": rar_path.name,
        "expected_archive_name": "raw(2).rar",
        "archive_name_matches_objective": rar_path.name.casefold() == "raw(2).rar",
        "archive_sha256": archive_hash,
        "rar_test_passed": True,
        "external_extractor": tool,
        "csv_count": csv_count,
        "signal_csv_count": len(records),
        "non_csv_member_count": len(non_csv_names),
        "reference_like_non_csv_member_count": sum(
            bool(re.search(r"borg|mvc|force|torque|fatigue|phase|label|annotation", name, flags=re.IGNORECASE))
            for name in non_csv_names
        ),
        "group_count": len(group_ids),
        "group_file_counts": group_counts,
        "private_csv_hashes": private_hashes,
        "schema_compatible": schema_compatible,
        "incompatible_recording_count_due_to_nyquist": len(incompatible),
        "analyzed_window_count": len(all_windows),
        "reference_search_complete": reference_search_complete,
        "reference_search_status": "complete_no_reference_columns_found" if reference_search_complete else "incomplete_or_ambiguous",
        "independent_labels_found": independent_labels_found,
        "reference_candidate_count": sum(bool(item["reference_column_candidate"]) for item in candidate_label_files),
        "reference_candidates_validated": False,
        "reference_like_csv_header_count": sum(bool(item["reference_column_candidate"]) for item in candidate_label_files),
        "private_reference_csv_headers": candidate_label_files,
        "fatigue_predictions_generated": False,
        "fatigue_metrics_produced": False,
        "phase_metrics_produced": False,
        "identity_independence_verified": False,
        "previous_exposure_ruled_out": False,
        "limitations": [
            (
                "A reference-like CSV header was present, but synchronization, modality, granularity, and independence were not validated; no supervised model was frozen."
                if any(bool(item["reference_column_candidate"]) for item in candidate_label_files)
                else (
                    "No independent synchronized fatigue or movement-phase reference was present in the complete observed three-column signal schema."
                    if reference_search_complete
                    else f"Reference search is incomplete: {len(non_csv_names)} non-CSV member(s) and {csv_count - len(records)} non-signal CSV member(s) were not validated; no label absence is claimed."
                )
            ),
            "The observed external source grouping cannot be verified as independent of training identities or prior project exposure.",
            "Encoder outputs are exploratory representations and reconstruction errors, not fatigue predictions or external validation metrics.",
        ],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "external_audit.json", result)
    public_result = {
        "status": result["status"],
        "csv_count": csv_count,
        "group_count": len(group_ids),
        "analyzed_window_count": len(all_windows),
        "reference_search_complete": reference_search_complete,
        "independent_labels_found": independent_labels_found,
        "fatigue_metrics_produced": False,
        "identity_independence_verified": False,
        "limitations": result["limitations"],
    }
    write_json(_repo_results_dir() / "external_summary.json", public_result)
    _refresh_execution_summary(result)

    manifest_path = private / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_archive_sha256"] = archive_hash
    manifest["external_output_sha256"] = {
        str(path.relative_to(run_root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest.setdefault("output_sha256", {}).update(manifest["external_output_sha256"])
    public_summary = _repo_results_dir() / "external_summary.json"
    manifest["output_sha256"]["public/external_summary.json"] = sha256_file(public_summary)
    manifest["output_sha256"]["public/EXECUTION_SUMMARY.md"] = sha256_file(_repo_results_dir() / "EXECUTION_SUMMARY.md")
    manifest["external_rar_opened"] = True
    manifest["external_stage_status"] = str(result["status"])
    manifest["external_checked_at_utc"] = result["checked_at_utc"]
    freeze["external_rar_opened"] = True
    freeze["external_archive_sha256"] = archive_hash
    freeze["external_check_status"] = result["status"]
    write_json(freeze_path, freeze)
    manifest.setdefault("output_sha256", {})["private/freeze.json"] = sha256_file(freeze_path)
    write_json(manifest_path, manifest)
    return result


def _repo_results_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "results" / "raw_first"


def _refresh_execution_summary(result: dict[str, object]) -> None:
    """Replace the pre-external placeholder with the post-freeze file audit."""
    summary_path = _repo_results_dir() / "EXECUTION_SUMMARY.md"
    if not summary_path.is_file():
        return
    expected_name = str(result["expected_archive_name"])
    filename_note = (
        f"The supplied archive filename matches expected `{expected_name}`."
        if bool(result.get("archive_name_matches_objective"))
        else f"The supplied archive filename differs from expected `{expected_name}`; the actual name is kept in the restricted audit."
    )
    csv_count = int(result.get("csv_count") or 0)
    group_count = int(result.get("group_count") or 0)
    group_word = "group" if group_count == 1 else "groups"
    signal_count = int(result.get("signal_csv_count") or 0)
    window_count = int(result.get("analyzed_window_count") or 0)
    reference_status = (
        "A complete reference search found no independent fatigue or movement-phase labels."
        if bool(result.get("reference_search_complete")) and result.get("independent_labels_found") is False
        else "Reference search was incomplete or ambiguous; no label absence is claimed and supervised outputs remain blocked."
    )
    schema_text = (
        f"{signal_count} of {csv_count} parsed CSVs matched"
        if bool(result.get("schema_compatible"))
        else f"not all parsed CSVs matched ({signal_count} of {csv_count} parsed)"
    )
    section = "\n".join(
        [
            "## External RAR",
            "",
            f"- The archive integrity check passed. {filename_note}",
            f"- Inspected {csv_count} CSV members across {group_count} opaque source {group_word}; {signal_count} matched the frozen signal schema, and {int(result.get('incompatible_recording_count_due_to_nyquist') or 0)} were excluded for sampling incompatibility.",
            f"- {schema_text} the frozen three-column schema. Frozen preprocessing and both encoders produced {window_count:,} exploratory windows after training freeze.",
            f"- {reference_status} No predictions or supervised metrics were produced.",
            "- Source identity independence and prior exposure remain unverified. Treat this as held-out file analysis, not independent external validation.",
            "",
        ]
    )
    text = summary_path.read_text(encoding="utf-8")
    prefix = text.split("## External RAR", 1)[0].rstrip()
    summary_path.write_text(prefix + "\n\n" + section, encoding="utf-8")
