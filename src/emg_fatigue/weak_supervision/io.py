"""Fresh, group-aware inventory and development-only signal parsing."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np

from .contracts import ArchiveInventory, ArchiveMember, GroupSplit, RawRecording

REQUIRED_COLUMNS = ("Relative_Time_s", "ADC_Value", "Voltage_mV")
SPLIT_SEED = 20261112


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def create_group_split(
    group_ids: Sequence[str], seed: int = SPLIT_SEED, train_fraction: float = 0.8
) -> GroupSplit:
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) != len(group_ids):
        raise ValueError("Group split input must contain one occurrence of each source group.")
    if len(unique_groups) < 2:
        raise ValueError("At least two source groups are required for a held-out group split.")
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("Training fraction must be in [0.5, 1).")
    order = np.random.default_rng(seed).permutation(unique_groups).tolist()
    test_count = max(1, int(round((1.0 - train_fraction) * len(unique_groups))))
    test_count = min(test_count, len(unique_groups) - 1)
    return GroupSplit(
        seed=int(seed),
        train_fraction=float(train_fraction),
        development_groups=tuple(sorted(order[test_count:])),
        test_groups=tuple(sorted(order[:test_count])),
    )


def _member_header(payload: bytes, member_name: str) -> tuple[str, ...]:
    first_line = payload.splitlines()[0] if payload.splitlines() else b""
    try:
        text = first_line.decode("utf-8-sig")
        return tuple(next(csv.reader([text])))
    except (UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise ValueError(f"CSV header is invalid for an archive member: {member_name}") from error


def _filename_metadata(member_name: str) -> dict[str, str | None]:
    name = PurePosixPath(member_name).name
    series = re.search(r"(?<![A-Za-z0-9])R[0-5](?![A-Za-z0-9])", name, flags=re.IGNORECASE)
    side = re.search(r"(?<![A-Za-z0-9])(?:dr|iz)(?![A-Za-z0-9])", name, flags=re.IGNORECASE)
    return {
        "series_token": series.group(0).upper() if series else None,
        "side_token": side.group(0).lower() if side else None,
        "use": "descriptive_only_never_a_target_or_primary_predictor",
    }


def _parse_development_record(
    payload: bytes,
    group_id: str,
    recording_id: str,
    source_sha256: str,
    member_name: str,
) -> tuple[RawRecording, int]:
    try:
        decoded = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded))
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(f"CSV columns do not match the required schema: {member_name}")
        timestamps: list[float] = []
        samples: list[float] = []
        voltage: list[float] = []
        rows = 0
        for row in reader:
            rows += 1
            try:
                timestamps.append(float(row["Relative_Time_s"]))
                samples.append(float(row["ADC_Value"]))
                voltage.append(float(row["Voltage_mV"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV contains a malformed numeric row: {member_name}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"CSV is not valid UTF-8: {member_name}") from error
    t = np.asarray(timestamps, dtype=np.float64)
    x = np.asarray(samples, dtype=np.float64)
    v = np.asarray(voltage, dtype=np.float64)
    if rows < 2 or not np.isfinite(t).all() or not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError(f"CSV has too few rows or non-finite numeric values: {member_name}")
    return (
        RawRecording(
            group_id=group_id,
            recording_id=recording_id,
            relative_time_s=t,
            adc_value=x,
            voltage_mv=v,
            source_sha256=source_sha256,
            filename_metadata=_filename_metadata(member_name),
        ),
        rows,
    )


def load_training_zip(
    path: Path,
    seed: int = SPLIT_SEED,
    train_fraction: float = 0.8,
) -> tuple[list[RawRecording], ArchiveInventory]:
    """Hash and inventory the full ZIP; decode signal values for development groups only."""
    path = Path(path).expanduser().resolve()
    if path.suffix.casefold() != ".zip":
        raise ValueError("Training input must be a ZIP archive; RAR is reserved for the final external stage.")
    if not path.is_file():
        raise FileNotFoundError("Training ZIP was not found.")

    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError("Training ZIP failed its CRC check.")
        all_files = [item for item in archive.infolist() if not item.is_dir()]
        csv_infos = [item for item in all_files if item.filename.casefold().endswith(".csv")]
        if not csv_infos:
            raise ValueError("Training ZIP contains no CSV signals.")

        names_by_group: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for info in csv_infos:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or len(member.parts) < 2:
                raise ValueError("Training ZIP contains an unsafe CSV member path.")
            names_by_group[member.parts[0]].append(info)

        raw_groups = sorted(names_by_group, key=str.casefold)
        if len(raw_groups) < 2 or len(raw_groups) > 99:
            raise ValueError("Training ZIP has an unsupported number of source groups.")
        alias_by_source = {name: f"G{index:02d}" for index, name in enumerate(raw_groups, start=1)}
        split = create_group_split(tuple(alias_by_source.values()), seed=seed, train_fraction=train_fraction)
        development = set(split.development_groups)
        test = set(split.test_groups)

        group_counts: dict[str, int] = {}
        recording_names: list[tuple[str, str, zipfile.ZipInfo]] = []
        for source_name in raw_groups:
            infos = sorted(names_by_group[source_name], key=lambda item: PurePosixPath(item.filename).name.casefold())
            group_id = alias_by_source[source_name]
            group_counts[group_id] = len(infos)
            for file_index, info in enumerate(infos, start=1):
                recording_names.append((group_id, f"{group_id}_F{file_index:03d}", info))

        member_records: list[ArchiveMember] = []
        recordings: list[RawRecording] = []
        hash_to_groups: dict[str, set[str]] = defaultdict(set)
        hash_to_group_counts: Counter[str] = Counter()
        header_counts: Counter[str] = Counter()
        malformed_rows = 0
        for group_id, recording_id, info in recording_names:
            payload = archive.read(info)
            digest = _sha256(payload)
            hash_to_groups[digest].add(group_id)
            hash_to_group_counts[digest] += 1
            header = _member_header(payload, info.filename)
            if header != REQUIRED_COLUMNS:
                raise ValueError(f"CSV columns do not match the required schema: {info.filename}")
            header_counts[",".join(header)] += 1
            if group_id in development:
                record, row_count = _parse_development_record(
                    payload, group_id, recording_id, digest, info.filename
                )
                recordings.append(record)
                parsed = True
            else:
                row_count = None
                parsed = False
            member_records.append(
                ArchiveMember(
                    group_id=group_id,
                    recording_id=recording_id,
                    member_name=info.filename,
                    sha256=digest,
                    byte_size=int(info.file_size),
                    header=header,
                    parsed_numeric_values=parsed,
                    numeric_row_count=row_count,
                )
            )

        cross_group_duplicates = [digest for digest, groups in hash_to_groups.items() if len(groups) > 1]
        if cross_group_duplicates:
            raise ValueError("Duplicate signal content appears across source groups; group isolation is unsafe.")
        within_group_duplicates = tuple(sorted(digest for digest, count in hash_to_group_counts.items() if count > 1))
        inventory = ArchiveInventory(
            input_archive_name=path.name,
            input_archive_sha256=_sha256_file(path),
            csv_count=len(csv_infos),
            source_group_count=len(raw_groups),
            source_groups=tuple(sorted(alias_by_source.values())),
            source_group_map=alias_by_source,
            split=split,
            members=tuple(member_records),
            header_counts=dict(header_counts),
            non_csv_member_count=len(all_files) - len(csv_infos),
            development_record_count=len(recordings),
            unparsed_internal_test_member_count=sum(item.group_id in test and not item.parsed_numeric_values for item in member_records),
            duplicate_within_group_hashes=within_group_duplicates,
            malformed_row_count=malformed_rows,
        )
    return recordings, inventory
