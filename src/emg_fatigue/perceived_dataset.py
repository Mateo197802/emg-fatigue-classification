"""Auditable reader and signal features for the Zenodo perceived-fatigue set.

The target is the participant's reported fatigue state. It is not a mechanical
loss-of-force label. Raw trial files are read directly from the nested archives;
they are never copied into the repository.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import pywt
from scipy import signal


CHANNEL_ORDER = (
    "biceps",
    "deltoid_anterior",
    "deltoid_medius",
    "deltoid_posterior",
)

TRIAL_PRIME_MOVER = {
    1: "deltoid_anterior",
    2: "deltoid_anterior",
    3: "deltoid_posterior",
    4: "deltoid_posterior",
    5: "biceps",
    6: "biceps",
    7: "deltoid_medius",
    8: "deltoid_medius",
    9: "deltoid_anterior",
    10: "deltoid_anterior",
    11: "deltoid_posterior",
    12: "deltoid_posterior",
}


@dataclass(frozen=True)
class PerceivedFatigueTrial:
    """One movement trial with four simultaneous active-muscle channels."""

    subject_id: str
    trial_id: str
    time_s: np.ndarray
    channels_v: dict[str, np.ndarray]
    label_time_s: np.ndarray
    perceived_level: np.ndarray
    raw_member: str
    label_member: str
    raw_member_crc32: str
    raw_member_bytes: int
    label_member_crc32: str
    label_member_bytes: int
    signal_metadata: dict
    label_cleanup: dict


def _canonical_muscle(value):
    text = str(value).upper()
    if "BICEPS" in text:
        return "biceps"
    if "DELTOID" not in text:
        return None
    if "ANTERIOR" in text:
        return "deltoid_anterior"
    if "MEDIUS" in text:
        return "deltoid_medius"
    if "POSTERIOR" in text:
        return "deltoid_posterior"
    return None


def parse_measurement_frame(frame):
    """Parse a trial CSV as paired elapsed-time and raw-voltage columns."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[1] != 8:
        raise ValueError("trial CSV must contain four time/EMG column pairs")

    names = list(map(str, frame.columns))
    channels = {}
    time_axes = {}
    source_columns = {}
    for index in range(0, len(names), 2):
        time_name, signal_name = names[index : index + 2]
        if not time_name.lower().startswith("x [s]"):
            raise ValueError("each signal column must be preceded by its elapsed-time column")
        if "[v]" not in signal_name.lower():
            raise ValueError("raw EMG units are not identified as volts")
        muscle = _canonical_muscle(signal_name)
        if muscle is None:
            raise ValueError(f"unrecognized active-muscle channel: {signal_name}")
        if muscle in channels:
            raise ValueError(f"duplicate active-muscle channel: {muscle}")
        time_values = pd.to_numeric(frame[time_name], errors="coerce").to_numpy(dtype=float)
        signal_values = pd.to_numeric(frame[signal_name], errors="coerce").to_numpy(dtype=float)
        if time_values.ndim != 1 or signal_values.shape != time_values.shape:
            raise ValueError("time and EMG columns must be matching one-dimensional arrays")
        channels[muscle] = signal_values
        time_axes[muscle] = time_values
        source_columns[muscle] = {"time": time_name, "signal": signal_name}

    if set(channels) != set(CHANNEL_ORDER):
        raise ValueError("trial must contain one channel for each of the four active muscles")
    time = time_axes[CHANNEL_ORDER[0]]
    if len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("elapsed-time axis must contain at least two finite samples")
    terminal_rows_dropped = 0
    while len(time) > 2:
        reset_channels = []
        for muscle in CHANNEL_ORDER:
            axis = time_axes[muscle]
            if not np.isfinite(axis[-1]) or axis[-1] <= axis[-2]:
                if not np.isfinite(axis[-1]) or abs(float(axis[-1])) <= 1e-12:
                    reset_channels.append(muscle)
                else:
                    raise ValueError("EMG time axis contains a nonterminal clock reset")
        if not reset_channels:
            break
        for muscle in reset_channels:
            last_signal = channels[muscle][-1]
            if np.isfinite(last_signal) and last_signal != 0.0:
                raise ValueError("ambiguous terminal clock reset with nonzero EMG")
        # The terminal row is not synchronized across all four active channels.
        # Drop it as one acquisition row, never sort the time axis or interpolate.
        for muscle in CHANNEL_ORDER:
            time_axes[muscle] = time_axes[muscle][:-1]
            channels[muscle] = channels[muscle][:-1]
        terminal_rows_dropped += 1
        time = time_axes[CHANNEL_ORDER[0]]

    if len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("elapsed-time axis must contain at least two finite samples")
    dt = np.diff(time)
    positive = dt[dt > 0]
    if not len(positive):
        raise ValueError("elapsed-time axis has no positive sampling intervals")
    if np.any(dt <= 0):
        raise ValueError("EMG time axis contains an internal clock reset or duplicate")
    alignment_tolerance = max(1e-8, 0.05 * float(np.median(positive)))
    for muscle in CHANNEL_ORDER[1:]:
        axis = time_axes[muscle]
        if axis.shape != time.shape or not np.isfinite(axis).all():
            raise ValueError("four active channels do not share a complete time axis")
        if not np.allclose(axis, time, rtol=0.0, atol=alignment_tolerance):
            raise ValueError("four active channels are not aligned to a common time axis")
    ordered = {name: channels[name] for name in CHANNEL_ORDER}
    return time, ordered, {
        "units": "V",
        "channel_time_axes_aligned": True,
        "channel_alignment_tolerance_s": alignment_tolerance,
        "terminal_reset_rows_dropped": terminal_rows_dropped,
        "input_rows": int(len(time) + terminal_rows_dropped),
        "rows_dropped": terminal_rows_dropped,
        "source_columns": source_columns,
    }


def clean_label_timeline(time_s, levels):
    """Validate the 50 Hz step labels and drop one unambiguous bad terminal row."""
    time = np.asarray(time_s, dtype=float)
    label = np.asarray(levels)
    if time.ndim != 1 or label.ndim != 1 or len(time) != len(label) or len(time) < 2:
        raise ValueError("label time and state must be matching one-dimensional arrays")
    if not np.isfinite(time).all() or not np.isin(label, [0, 1, 2]).all():
        raise ValueError("label timeline contains nonfinite timestamps or unknown states")
    label = label.astype(int)
    dt = np.diff(time)
    nonpositive = np.flatnonzero(dt <= 0)
    dropped = False
    if len(nonpositive):
        terminal_duplicate = (
            len(nonpositive) == 1
            and int(nonpositive[0]) == len(time) - 2
            and label[-1] == label[-2]
        )
        if not terminal_duplicate:
            raise ValueError("ambiguous nonmonotonic label timeline")
        time = time[:-1]
        label = label[:-1]
        dt = np.diff(time)
        dropped = True
    if np.any(dt <= 0):
        raise ValueError("label timestamps must be strictly increasing after cleanup")
    median_dt = float(np.median(dt))
    if not np.isfinite(median_dt) or median_dt <= 0:
        raise ValueError("could not estimate label sampling interval")
    return time, label, {
        "input_rows": int(len(time) + int(dropped)),
        "rows_dropped": int(dropped),
        "terminal_backward_duplicate_dropped": bool(dropped),
        "median_interval_s": median_dt,
        "estimated_rate_hz": float(1.0 / median_dt),
        "nonmonotonic_intervals_remaining": int(np.count_nonzero(dt <= 0)),
        "large_gaps_remaining": int(np.count_nonzero(dt > 1.5 * median_dt)),
        "initial_state_from_protocol": 0,
    }


def label_window(
    start_s,
    end_s,
    label_time_s,
    perceived_level,
    *,
    initial_state=0,
):
    """Return binary no/yes only when a full window has one stable state.

    Level 0 maps to 0 (no perceived fatigue); level 2 maps to 1 (perceived
    fatigue). Level 1 is an explicit transition and is excluded from the
    binary target. The end time is exclusive.
    """
    times = np.asarray(label_time_s, dtype=float)
    states = np.asarray(perceived_level, dtype=int)
    if (
        times.ndim != 1
        or states.ndim != 1
        or len(times) != len(states)
        or len(times) < 2
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("label timeline must be aligned and strictly increasing")
    start = float(start_s)
    end = float(end_s)
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValueError("window bounds must be finite, nonnegative and increasing")
    median_dt = float(np.median(np.diff(times)))
    coverage_tolerance = max(1e-6, 1.1 * median_dt)
    if end > times[-1] + coverage_tolerance:
        return None, "outside_reference_support"

    # The acquisition protocol initializes the button state to 0. The first
    # recorded sample is normally at 20 ms rather than exactly at time zero.
    # Source timestamps are decimal exports of a 50 Hz clock; the tolerance
    # prevents a representational value such as 0.7000000000000001 from moving
    # a window boundary to the preceding state.
    first_index = int(np.searchsorted(times, start + 1e-9, side="right") - 1)
    if first_index < 0:
        if initial_state != 0 or times[0] > coverage_tolerance:
            return None, "outside_reference_support"
        state_at_start = int(initial_state)
    else:
        state_at_start = int(states[first_index])

    intersected = (times > start + 1e-9) & (times < end - 1e-9)
    event_states = set(int(value) for value in states[intersected])
    event_states.add(state_at_start)
    gaps = np.diff(times)
    crossed_gap = np.any(
        (times[:-1] < end)
        & (times[1:] > start)
        & (gaps > 1.5 * median_dt)
    )
    if crossed_gap:
        return None, "reference_gap"
    if len(event_states) != 1:
        return None, "mixed_reference_state"
    if state_at_start == 1:
        return None, "transition_state_excluded"
    if state_at_start == 0:
        return 0, "stable_no_perceived_fatigue"
    if state_at_start == 2:
        return 1, "stable_perceived_fatigue"
    return None, "unknown_reference_state"


def extract_perceived_features(x, fs_hz, *, band_hz=(20.0, 450.0)):
    """Compute interpretable time, spectral and db4 features for one window."""
    values = np.asarray(x, dtype=float)
    fs = float(fs_hz)
    low, high = map(float, band_hz)
    if values.ndim != 1 or len(values) < 56 or not np.isfinite(values).all():
        raise ValueError("feature window must contain at least 56 finite samples")
    if not np.isfinite(fs) or not 0 < low < high or fs <= 2 * high:
        raise ValueError("feature band must be below Nyquist for the measured rate")
    centered = values - float(values.mean())
    energy = float(np.mean(centered * centered))
    if energy <= np.finfo(float).tiny:
        raise ValueError("zero-energy feature window")
    frequency, density = signal.periodogram(
        centered,
        fs=fs,
        window="hann",
        detrend=False,
        scaling="density",
    )
    in_band = (frequency >= low) & (frequency <= high)
    f = frequency[in_band]
    p = density[in_band]
    if len(f) < 2 or not np.isfinite(p).all() or float(p.sum()) <= np.finfo(float).tiny:
        raise ValueError("insufficient in-band power")
    df = float(np.median(np.diff(frequency)))
    total_band_power = float(p.sum() * df)
    if total_band_power <= np.finfo(float).tiny:
        raise ValueError("zero in-band power")

    def band_power(lower, upper, *, include_upper=False):
        mask = (frequency >= lower) & (
            (frequency <= upper) if include_upper else (frequency < upper)
        )
        return float(density[mask].sum() * df)

    total_spectral_sum = float(p.sum())
    cumulative = np.cumsum(p)
    median_index = min(int(np.searchsorted(cumulative, 0.5 * total_spectral_sum)), len(f) - 1)
    probability = p / total_spectral_sum
    nonzero = probability > 0
    entropy = float(-np.sum(probability[nonzero] * np.log(probability[nonzero])))
    entropy_norm = entropy / float(np.log(len(probability))) if len(probability) > 1 else 0.0

    coefficients = pywt.wavedec(centered, "db4", mode="symmetric", level=3)
    wavelet_energy = np.asarray([np.sum(c * c) for c in coefficients], dtype=float)
    wavelet_total = float(wavelet_energy.sum())
    if wavelet_total <= np.finfo(float).tiny:
        raise ValueError("zero wavelet energy")
    relative_wavelet = wavelet_energy / wavelet_total
    sample_span_s = (len(centered) - 1) / fs
    result = {
        "rms": float(np.sqrt(energy)),
        "mav": float(np.mean(np.abs(centered))),
        "waveform_length_v": float(np.sum(np.abs(np.diff(centered)))),
        "waveform_length_v_per_s": float(np.sum(np.abs(np.diff(centered))) / sample_span_s),
        "iemg_v_s": float(np.sum(np.abs(centered)) / fs),
        "mdf_hz": float(f[median_index]),
        "mnf_hz": float(np.sum(f * p) / total_spectral_sum),
        "bandpower_20_60_v2": band_power(20.0, 60.0),
        "bandpower_60_150_v2": band_power(60.0, 150.0),
        "bandpower_150_450_v2": band_power(150.0, 450.0, include_upper=True),
        "spectral_entropy_norm": float(np.clip(entropy_norm, 0.0, 1.0)),
        "wavelet_a3_energy_rel": float(relative_wavelet[0]),
        "wavelet_d3_energy_rel": float(relative_wavelet[1]),
        "wavelet_d2_energy_rel": float(relative_wavelet[2]),
        "wavelet_d1_energy_rel": float(relative_wavelet[3]),
    }
    if not np.isfinite(list(result.values())).all():
        raise ValueError("nonfinite features")
    return result


def _member_index(archive):
    return {name.replace("\\", "/").lower(): name for name in archive.namelist()}


def _find_member(index, expected_suffix):
    suffix = expected_suffix.replace("\\", "/").lower()
    matches = [actual for normalized, actual in index.items() if normalized.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one archive member ending in {expected_suffix!r}; found {len(matches)}")
    return matches[0]


def _label_frame(frame):
    columns = {str(name).strip().lower(): name for name in frame.columns}
    if not {"time", "label"}.issubset(columns):
        raise ValueError("perceived-fatigue CSV must contain time and label columns")
    time, levels, receipt = clean_label_timeline(
        pd.to_numeric(frame[columns["time"]], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(frame[columns["label"]], errors="coerce").to_numpy(dtype=float),
    )
    return time, levels, receipt


def iter_perceived_trials(
    emg_archive_path,
    label_archive_path,
    *,
    subject_ids=range(1, 14),
    trial_ids=range(1, 13),
):
    """Yield matched subject/trial EMG and self-report records from nested ZIPs."""
    with zipfile.ZipFile(Path(emg_archive_path)) as emg_outer, zipfile.ZipFile(
        Path(label_archive_path)
    ) as label_outer:
        emg_index = _member_index(emg_outer)
        label_index = _member_index(label_outer)
        for subject_number in subject_ids:
            subject = int(subject_number)
            raw_subject_member = _find_member(emg_index, f"sEMG_data/subject_{subject}.zip")
            label_subject_member = _find_member(
                label_index, f"self_perceived_fatigue_index/subject_{subject}.zip"
            )
            raw_subject_bytes = emg_outer.read(raw_subject_member)
            label_subject_bytes = label_outer.read(label_subject_member)
            with zipfile.ZipFile(BytesIO(raw_subject_bytes)) as raw_subject, zipfile.ZipFile(
                BytesIO(label_subject_bytes)
            ) as label_subject:
                raw_index = _member_index(raw_subject)
                trial_label_index = _member_index(label_subject)
                for trial_number in trial_ids:
                    trial = int(trial_number)
                    raw_member = _find_member(
                        raw_index, f"subject_{subject}/trial_{trial}.csv"
                    )
                    label_member = _find_member(
                        trial_label_index,
                        f"subject_{subject}/trial_{trial}.csv",
                    )
                    raw_info = raw_subject.getinfo(raw_member)
                    label_info = label_subject.getinfo(label_member)
                    with raw_subject.open(raw_member) as stream:
                        raw_frame = pd.read_csv(stream)
                    with label_subject.open(label_member) as stream:
                        label_frame = pd.read_csv(stream)
                    time, channels, signal_metadata = parse_measurement_frame(raw_frame)
                    label_time, levels, label_cleanup = _label_frame(label_frame)
                    yield PerceivedFatigueTrial(
                        subject_id=f"subject_{subject}",
                        trial_id=f"trial_{trial}",
                        time_s=time,
                        channels_v=channels,
                        label_time_s=label_time,
                        perceived_level=levels,
                        raw_member=raw_member,
                        label_member=label_member,
                        raw_member_crc32=f"{raw_info.CRC:08x}",
                        raw_member_bytes=int(raw_info.file_size),
                        label_member_crc32=f"{label_info.CRC:08x}",
                        label_member_bytes=int(label_info.file_size),
                        signal_metadata=signal_metadata,
                        label_cleanup=label_cleanup,
                    )
