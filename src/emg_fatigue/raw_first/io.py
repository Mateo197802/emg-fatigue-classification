"""Read and validate the training ZIP without extracting identifiable names."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from .contracts import RawRecording

REQUIRED_COLUMNS = ("Relative_Time_s", "ADC_Value", "Voltage_mV")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_training_zip(path: Path) -> tuple[list[RawRecording], dict[str, object]]:
    """Load CSV arrays and return a private-safe inventory without source names."""
    if path.suffix.casefold() != ".zip":
        raise ValueError("Training input must be a ZIP archive; RAR input is reserved for the final external stage.")
    if not path.is_file():
        raise FileNotFoundError("Training archive was not found.")

    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError("Training ZIP failed its CRC check.")

        csv_infos = [item for item in archive.infolist() if not item.is_dir() and item.filename.casefold().endswith(".csv")]
        if not csv_infos:
            raise ValueError("Training ZIP contains no CSV files.")

        members_by_group: dict[str, list[zipfile.ZipInfo]] = {}
        for item in csv_infos:
            member = PurePosixPath(item.filename)
            if member.is_absolute() or ".." in member.parts or len(member.parts) < 2:
                raise ValueError("Training ZIP contains a CSV path outside a source folder.")
            members_by_group.setdefault(member.parts[0], []).append(item)

        source_groups = sorted(members_by_group, key=lambda value: value.casefold())
        if len(source_groups) > 99:
            raise ValueError("Training ZIP has too many source groups for opaque aliasing.")
        aliases = {name: f"G{index:02d}" for index, name in enumerate(source_groups, start=1)}

        records: list[RawRecording] = []
        file_hashes: list[dict[str, str]] = []
        header_counts: dict[str, int] = {}
        malformed_rows = 0
        for source_group in source_groups:
            group_infos = sorted(members_by_group[source_group], key=lambda item: PurePosixPath(item.filename).name.casefold())
            group_id = aliases[source_group]
            for file_index, info in enumerate(group_infos, start=1):
                recording_id = f"{group_id}_F{file_index:03d}"
                payload = archive.read(info)
                digest = hashlib.sha256(payload).hexdigest()
                text = payload.decode("utf-8-sig")
                reader = csv.DictReader(io.StringIO(text))
                headers = tuple(reader.fieldnames or ())
                header_counts[",".join(headers)] = header_counts.get(",".join(headers), 0) + 1
                if headers != REQUIRED_COLUMNS:
                    raise ValueError(f"Recording {recording_id} does not match the required three-column schema.")

                time_values: list[float] = []
                adc_values: list[float] = []
                voltage_values: list[float] = []
                for row in reader:
                    try:
                        time_values.append(float(row["Relative_Time_s"]))
                        adc_values.append(float(row["ADC_Value"]))
                        voltage_values.append(float(row["Voltage_mV"]))
                    except (TypeError, ValueError):
                        malformed_rows += 1
                        time_values.append(float("nan"))
                        adc_values.append(float("nan"))
                        voltage_values.append(float("nan"))

                records.append(
                    RawRecording(
                        group_id=group_id,
                        recording_id=recording_id,
                        relative_time_s=np.asarray(time_values, dtype=np.float64),
                        adc_value=np.asarray(adc_values, dtype=np.float64),
                        voltage_mv=np.asarray(voltage_values, dtype=np.float64),
                        source_sha256=digest,
                    )
                )
                file_hashes.append({"group_id": group_id, "recording_id": recording_id, "sha256": digest})

    inventory: dict[str, object] = {
        "input_archive_name": path.name,
        "input_archive_sha256": sha256_file(path),
        "csv_count": len(records),
        "source_group_count": len(source_groups),
        "source_groups": sorted(aliases.values()),
        "files_by_group": {aliases[name]: len(members_by_group[name]) for name in source_groups},
        "non_csv_member_count": len([item for item in archive.infolist() if not item.is_dir()]) - len(csv_infos),
        "header_counts": header_counts,
        "malformed_row_count": malformed_rows,
        "file_hashes": file_hashes,
        "group_alias_policy": "case-insensitive sort of source folder names mapped to G01..Gnn; original names are not written",
    }
    return records, inventory
