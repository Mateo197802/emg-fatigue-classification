"""Typed contracts for the fresh weak-supervision experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupSplit:
    seed: int
    train_fraction: float
    development_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    method: str = "deterministic source-group split; source groups are not verified participants"


@dataclass(frozen=True)
class RawRecording:
    group_id: str
    recording_id: str
    relative_time_s: np.ndarray
    adc_value: np.ndarray
    voltage_mv: np.ndarray | None
    source_sha256: str
    filename_metadata: dict[str, str | None]


@dataclass(frozen=True)
class ArchiveMember:
    group_id: str
    recording_id: str
    member_name: str
    sha256: str
    byte_size: int
    header: tuple[str, ...]
    parsed_numeric_values: bool
    numeric_row_count: int | None


@dataclass(frozen=True)
class ArchiveInventory:
    input_archive_name: str
    input_archive_sha256: str
    csv_count: int
    source_group_count: int
    source_groups: tuple[str, ...]
    source_group_map: dict[str, str]
    split: GroupSplit
    members: tuple[ArchiveMember, ...]
    header_counts: dict[str, int]
    non_csv_member_count: int
    development_record_count: int
    unparsed_internal_test_member_count: int
    duplicate_within_group_hashes: tuple[str, ...]
    malformed_row_count: int

    def to_private_dict(self) -> dict[str, object]:
        return {
            "input_archive_name": self.input_archive_name,
            "input_archive_sha256": self.input_archive_sha256,
            "csv_count": self.csv_count,
            "source_group_count": self.source_group_count,
            "source_groups": list(self.source_groups),
            "source_group_map": self.source_group_map,
            "split": {
                "seed": self.split.seed,
                "train_fraction": self.split.train_fraction,
                "development_groups": list(self.split.development_groups),
                "test_groups": list(self.split.test_groups),
                "method": self.split.method,
            },
            "members": [
                {
                    "group_id": item.group_id,
                    "recording_id": item.recording_id,
                    "member_name": item.member_name,
                    "sha256": item.sha256,
                    "byte_size": item.byte_size,
                    "header": list(item.header),
                    "parsed_numeric_values": item.parsed_numeric_values,
                    "numeric_row_count": item.numeric_row_count,
                }
                for item in self.members
            ],
            "header_counts": self.header_counts,
            "non_csv_member_count": self.non_csv_member_count,
            "development_record_count": self.development_record_count,
            "unparsed_internal_test_member_count": self.unparsed_internal_test_member_count,
            "duplicate_within_group_hashes": list(self.duplicate_within_group_hashes),
            "malformed_row_count": self.malformed_row_count,
        }
