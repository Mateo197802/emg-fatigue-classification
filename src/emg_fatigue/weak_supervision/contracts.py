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


@dataclass(frozen=True)
class AcquisitionAudit:
    group_id: str
    recording_id: str
    sample_count: int
    duration_s: float
    sampling_rate_hz: float
    median_dt_s: float
    dt_p05_s: float
    dt_p95_s: float
    jitter_percent: float
    gap_threshold_s: float
    continuous_ranges: tuple[tuple[int, int], ...]
    duplicate_timestamp_count: int
    clock_reset_count: int
    gap_count: int
    nyquist_hz: float
    adc_voltage_pair_count: int
    adc_voltage_slope_mv_per_unit: float
    adc_voltage_intercept_mv: float
    adc_voltage_r_squared: float
    adc_value_integer_like_fraction: float
    sampling_compatible: bool
    in_band_power_fraction: float
    below_20_hz_power_fraction: float
    above_400_hz_power_fraction: float
    signal_compatible: bool
    line_prominence_db: dict[str, float]
    notch_frequency_hz: float | None
    quality_flags: tuple[str, ...]
    quality_invalid_mask: np.ndarray
    artifact_masks: dict[str, np.ndarray]


@dataclass(frozen=True)
class ProcessedSegment:
    group_id: str
    recording_id: str
    segment_index: int
    signal: np.ndarray
    sampling_rate_hz: float
    start_time_s: float
    original_sampling_rate_hz: float
    edge_mask: np.ndarray
    quality_invalid_mask: np.ndarray
    artifact_masks: dict[str, np.ndarray]
    quality_flags: tuple[str, ...]
    acquisition_compatible: bool


@dataclass(frozen=True)
class ProcessedRecording:
    group_id: str
    recording_id: str
    audit: AcquisitionAudit
    segments: tuple[ProcessedSegment, ...]


@dataclass(frozen=True)
class SegmentActivity:
    segment_index: int
    status: str
    threshold_rms: float
    active_fraction: float
    valid_frame_count: int
    frame_rms: np.ndarray
    frame_start_samples: np.ndarray
    frame_active: np.ndarray
    frame_valid: np.ndarray
    sample_active_mask: np.ndarray
    active_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ActivitySegments:
    status: str
    active_fraction: float
    segments: tuple[SegmentActivity, ...]


@dataclass(frozen=True)
class SignalWindow:
    group_id: str
    recording_id: str
    segment_index: int
    window_index: int
    start_sample: int
    start_time_s: float
    duration_s: float
    sampling_rate_hz: float
    signal: np.ndarray
    activity_occupancy: float
    eligible_for_label: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class ReferenceEstimate:
    group_id: str
    recording_id: str
    segment_index: int
    available: bool
    stable: bool
    window_count: int
    reference_window_keys: tuple[str, ...]
    center: dict[str, float]
    scaled_mad: dict[str, float]
    stability_ratios: dict[str, float]
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class WeakLabel:
    window_key: str
    group_id: str
    recording_id: str
    segment_index: int
    start_time_s: float
    duration_s: float
    class_id: int
    class_name: str
    rule_version: str
    label_confidence: float
    label_reason: str
    state_transition: str
    reference_window_keys: tuple[str, ...]
    reference_center: dict[str, float]
    reference_scaled_mad: dict[str, float]
    observed_indicators: dict[str, float]
    indicator_deltas: dict[str, float]
    indicator_thresholds: dict[str, float]
    persistence_candidates: int
    persistence_span_s: float
    state_latched: bool
