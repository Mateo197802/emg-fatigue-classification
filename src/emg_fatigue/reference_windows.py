"""Window continuous sEMG runs without crossing signal or reference boundaries."""

from dataclasses import dataclass

import numpy as np

from .reference_labels import FatigueLabelResult, PhaseLabelResult


@dataclass(frozen=True)
class ReferenceWindow:
    window_id: str
    start_index: int
    end_index: int
    start_s: float
    end_s: float
    n_samples: int
    sample_span_s: float
    phase: str | None
    phase_status: str
    phase_evidence_status: str
    fatigue_state: int | None
    fatigue_status: str
    fatigue_evidence_status: str
    signal_status: str


@dataclass(frozen=True)
class ReferenceWindowSet:
    window_type: str
    window_s: float
    overlap: float
    window_samples: int
    hop_samples: int
    nominal_bin_spacing_hz: float
    verified_fs_hz: float
    windows: tuple[ReferenceWindow, ...]


def make_reference_windows(
    time_s,
    *,
    fs_hz,
    window_s=0.5,
    overlap=0.5,
    valid_mask=None,
    phase_result=None,
    fatigue_result=None,
    max_gap_s=None,
):
    """Create fixed sliding windows and abstain on mixed/invalid windows.

    A missing fatigue reference leaves every fatigue label absent. Phase and
    fatigue labels are copied only from their reference-result objects; there
    is no fallback to elapsed-time thirds or EMG burst halves.
    """
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("time_s must be a finite one-dimensional time axis")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and positive")
    if not np.isfinite(window_s) or window_s <= 0 or not 0 <= overlap < 1:
        raise ValueError("window_s and overlap are outside valid ranges")

    median_dt = float(np.median(np.diff(time)))
    observed_fs = 1.0 / median_dt
    if abs(observed_fs - fs_hz) / fs_hz > 0.05:
        raise ValueError("verified fs_hz disagrees with the supplied time axis")
    n = int(round(window_s * fs_hz))
    hop = max(1, int(round(n * (1.0 - overlap))))
    if n < 2:
        raise ValueError("window must contain at least two samples")

    if valid_mask is None:
        valid = np.ones(len(time), dtype=bool)
    else:
        valid = np.asarray(valid_mask)
        if valid.dtype != np.bool_ or valid.shape != time.shape:
            raise ValueError("valid_mask must be a boolean vector matching time_s")
    if phase_result is not None:
        if not isinstance(phase_result, PhaseLabelResult):
            raise ValueError("phase_result must be a PhaseLabelResult")
        if len(phase_result.labels) != len(time):
            raise ValueError("phase labels must match time_s")
        if phase_result.time_axis_id == phase_result.emg_time_axis_id:
            pass
        elif not phase_result.synchronization_id:
            raise ValueError("phase labels lack a synchronization receipt")
    if fatigue_result is not None and not isinstance(fatigue_result, FatigueLabelResult):
        raise ValueError("fatigue_result must be a FatigueLabelResult")

    if max_gap_s is None:
        max_gap_s = max(0.0015, 1.5 / fs_hz)
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")

    windows = []
    for start in range(0, len(time) - n + 1, hop):
        end = start + n
        start_s = float(time[start])
        end_s = float(time[end - 1])
        signal_status = "valid"
        if not valid[start:end].all():
            signal_status = "invalid_sample_mask"
        elif np.any(np.diff(time[start:end]) > max_gap_s):
            signal_status = "crosses_acquisition_gap"

        phase = None
        phase_status = "reference_missing"
        phase_evidence_status = "reference_missing"
        if phase_result is not None:
            phase_evidence_status = phase_result.evidence_status
            phase_values = np.asarray(phase_result.labels[start:end], dtype=object)
            unique = set(phase_values.tolist())
            if len(unique) == 1 and unique.issubset({"extension", "flexion"}):
                phase = str(phase_values[0])
                phase_status = "phase_reference_valid"
            else:
                phase_status = "phase_transition_or_mixed"

        fatigue_state = None
        fatigue_status = "reference_missing"
        fatigue_evidence_status = "reference_missing"
        if fatigue_result is not None:
            fatigue_evidence_status = fatigue_result.evidence_status
            onset = fatigue_result.onset_time_s
            if onset is None:
                fatigue_state = 0
                fatigue_status = "no_onset_observed"
            elif end_s < onset:
                fatigue_state = 0
                fatigue_status = "before_reference_onset"
            elif start_s >= onset:
                fatigue_state = 1
                fatigue_status = "after_reference_onset"
            else:
                fatigue_status = "crosses_fatigue_onset"

        if signal_status != "valid":
            phase = None
            phase_status = signal_status
            fatigue_state = None
            fatigue_status = signal_status

        windows.append(ReferenceWindow(
            window_id=f"W{start:08d}_{n}",
            start_index=start,
            end_index=end,
            start_s=start_s,
            end_s=end_s,
            n_samples=n,
            sample_span_s=float((n - 1) / fs_hz),
            phase=phase,
            phase_status=phase_status,
            phase_evidence_status=phase_evidence_status,
            fatigue_state=fatigue_state,
            fatigue_status=fatigue_status,
            fatigue_evidence_status=fatigue_evidence_status,
            signal_status=signal_status,
        ))
    return ReferenceWindowSet(
        window_type="fixed_length_sliding",
        window_s=float(window_s),
        overlap=float(overlap),
        window_samples=n,
        hop_samples=hop,
        nominal_bin_spacing_hz=float(fs_hz / n),
        verified_fs_hz=float(fs_hz),
        windows=tuple(windows),
    )
