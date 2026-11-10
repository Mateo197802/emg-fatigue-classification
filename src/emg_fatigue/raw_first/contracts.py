"""Small data contracts for the raw-first workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawRecording:
    group_id: str
    recording_id: str
    relative_time_s: np.ndarray
    adc_value: np.ndarray
    voltage_mv: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class ProcessedSegment:
    group_id: str
    recording_id: str
    segment_index: int
    signal: np.ndarray
    sampling_rate_hz: float
    start_time_s: float
    original_sampling_rate_hz: float
    resample_up: int
    resample_down: int


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


@dataclass(frozen=True)
class PipelinePaths:
    input_archive: Path
    output_root: Path
    run_root: Path
