"""Trial-level reference-task benchmark for paired RF sEMG recordings.

Labels belong to complete trials: reference contractions versus a sustained
task performed until task failure. They are not instantaneous fatigue states.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.io import loadmat
from sklearn.model_selection import GroupKFold

from experiments.reference_fatigue.figshare_task_level import (
    DATASET_DOI,
    detect_force_bouts,
)


SEED = 20261109
WINDOW_SECONDS = 0.5
OVERLAP = 0.5
BAND_HZ = (20.0, 450.0)
FILTER_ORDER = 4
EDGE_MASK_SECONDS = 0.10
MINIMUM_FORCE_BOUT_SECONDS = 10.0
MAX_CNN_WINDOWS_PER_TRIAL = 32
METRIC_NAMES = (
    "rms_log10",
    "mav_log10",
    "waveform_length_log10",
    "mdf_hz",
    "mnf_hz",
    "bandpower_20_60_ratio",
    "bandpower_60_150_ratio",
    "spectral_entropy_norm",
)


def condition_task_label(condition: str) -> int:
    """Map protocol condition to a trial-outcome label, never to a window label."""
    normalized = str(condition).strip().lower()
    if normalized == "30":
        return 0
    if normalized == "tlim":
        return 1
    raise ValueError(f"unsupported task-outcome condition: {condition!r}")


def summarize_window_metrics(
    window_metrics: dict[str, np.ndarray], time_fraction: np.ndarray
) -> dict[str, float]:
    """Pool signal-window measurements into one feature vector for a trial."""
    time = np.asarray(time_fraction, dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all():
        raise ValueError("time_fraction must contain at least two finite window positions")
    if np.any(np.diff(time) < 0):
        raise ValueError("time_fraction must be nondecreasing")
    pooled: dict[str, float] = {}
    for name, values in window_metrics.items():
        vector = np.asarray(values, dtype=float)
        if vector.shape != time.shape or not np.isfinite(vector).all():
            raise ValueError(f"window metric {name!r} must match finite time_fraction")
        q25, median, q75 = np.percentile(vector, [25.0, 50.0, 75.0])
        centered_time = time - time.mean()
        denominator = float(np.dot(centered_time, centered_time))
        slope = float(np.dot(centered_time, vector - vector.mean()) / denominator) if denominator else 0.0
        pooled[f"{name}__median"] = float(median)
        pooled[f"{name}__iqr"] = float(q75 - q25)
        pooled[f"{name}__slope"] = slope
    return pooled


def mil_mean_logit(window_logits: np.ndarray) -> float:
    """Aggregate instance logits to one multiple-instance trial score."""
    values = np.asarray(window_logits, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("window_logits must be a nonempty finite one-dimensional array")
    return float(values.mean())


def outer_leave_one_subject_out(frame: pd.DataFrame):
    """Yield participant-disjoint outer folds for paired trial rows."""
    required = {"subject_id", "condition", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"trial table lacks columns: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("trial table has missing subject, condition, or label values")
    if frame.duplicated(["subject_id", "condition"]).any():
        raise ValueError("trial table must contain at most one row per subject and condition")
    subjects = sorted(frame["subject_id"].astype(str).unique())
    if len(subjects) < 4:
        raise ValueError("at least four independent participants are required")
    for subject in subjects:
        test_indices = np.flatnonzero(frame["subject_id"].astype(str).to_numpy() == subject)
        train_indices = np.flatnonzero(frame["subject_id"].astype(str).to_numpy() != subject)
        test = frame.iloc[test_indices]
        if set(test["label"].astype(int)) != {0, 1}:
            raise ValueError(f"held-out subject {subject!r} does not have both trial conditions")
        yield {
            "test_subject": subject,
            "train_subjects": sorted(set(frame.iloc[train_indices]["subject_id"].astype(str))),
            "train_indices": train_indices,
            "test_indices": test_indices,
        }


def inner_group_splits(frame: pd.DataFrame, indices: np.ndarray, n_splits: int = 3):
    """Return grouped training/validation indices for nested model selection."""
    subset = frame.iloc[np.asarray(indices, dtype=int)]
    groups = subset["subject_id"].astype(str).to_numpy()
    split_count = min(int(n_splits), len(np.unique(groups)))
    if split_count < 2:
        raise ValueError("nested grouped tuning requires at least two training participants")
    splitter = GroupKFold(n_splits=split_count)
    for local_train, local_valid in splitter.split(subset, subset["label"], groups):
        yield np.asarray(indices, dtype=int)[local_train], np.asarray(indices, dtype=int)[local_valid]


def _valid_mat_channels(sig, discard, expected_samples: int):
    if sig.shape != (13, 5) or discard.shape != (13, 5):
        raise ValueError("SIG and discardChannelsVec must both use the 13x5 electrode grid")
    grid = np.zeros((65, expected_samples), dtype=np.float64)
    valid = np.zeros(65, dtype=bool)
    for flat_index, (row, col) in enumerate(np.ndindex(13, 5)):
        values = np.asarray(sig[row, col]).squeeze()
        if bool(discard[row, col]) or values.ndim != 1 or values.size != expected_samples:
            continue
        if not np.isfinite(values).all():
            continue
        grid[flat_index] = values.astype(np.float64, copy=False)
        valid[flat_index] = True
    if int(valid.sum()) < 32:
        raise ValueError(f"only {int(valid.sum())} usable electrode channels remain")
    return grid, valid


def _line_prominence_db(values: np.ndarray, fs: float) -> dict[int, float]:
    """Estimate 50/60 Hz prominence from a pooled Welch PSD sample."""
    channel_scores: dict[int, list[float]] = {50: [], 60: []}
    segment = values[:, : min(values.shape[1], int(round(2.0 * fs)))].copy()
    segment -= segment.mean(axis=1, keepdims=True)
    f, psd = signal.welch(segment, fs=fs, nperseg=min(4096, segment.shape[1]), axis=-1)
    for target in (50, 60):
        if target >= fs / 2:
            continue
        center = int(np.argmin(np.abs(f - target)))
        local = (np.abs(f - target) <= 2.0) & (np.abs(f - target) >= 0.75)
        baseline = np.median(psd[:, local], axis=1)
        scores = 10.0 * np.log10(np.maximum(psd[:, center], 1e-30) / np.maximum(baseline, 1e-30))
        channel_scores[target] = scores[np.isfinite(scores)].tolist()
    return {
        frequency: float(np.median(scores)) if scores else float("nan")
        for frequency, scores in channel_scores.items()
    }


def _window_features(channel_windows: np.ndarray, fs: float) -> dict[str, float]:
    """Compute robust channel-median features for one 0.5 s multi-channel window."""
    values = np.asarray(channel_windows, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    mav = np.mean(np.abs(centered), axis=1)
    wave = np.sum(np.abs(np.diff(centered, axis=1)), axis=1) / ((centered.shape[1] - 1) / fs)
    f, psd = signal.periodogram(
        centered, fs=fs, window="hann", detrend=False, scaling="density", axis=-1
    )
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    low = (f >= 20.0) & (f < 60.0)
    middle = (f >= 60.0) & (f < 150.0)
    total = np.sum(psd[:, band], axis=1)
    if np.any(total <= 1e-20):
        raise ValueError("an active EMG window has insufficient in-band power")
    band_psd = psd[:, band]
    band_freq = f[band]
    cumulative = np.cumsum(band_psd, axis=1)
    md_index = np.argmax(cumulative >= (0.5 * total[:, None]), axis=1)
    mdf = band_freq[md_index]
    mnf = np.sum(band_psd * band_freq[None, :], axis=1) / total
    p = band_psd / total[:, None]
    entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-30)), 0.0), axis=1) / np.log(p.shape[1])
    low_ratio = np.sum(psd[:, low], axis=1) / total
    middle_ratio = np.sum(psd[:, middle], axis=1) / total
    by_channel = {
        "rms_log10": np.log10(np.maximum(rms, 1e-20)),
        "mav_log10": np.log10(np.maximum(mav, 1e-20)),
        "waveform_length_log10": np.log10(np.maximum(wave, 1e-20)),
        "mdf_hz": mdf,
        "mnf_hz": mnf,
        "bandpower_20_60_ratio": low_ratio,
        "bandpower_60_150_ratio": middle_ratio,
        "spectral_entropy_norm": entropy,
    }
    result = {name: float(np.median(vector)) for name, vector in by_channel.items()}
    if not np.isfinite(list(result.values())).all():
        raise ValueError("nonfinite window features")
    result["_channel_rms"] = rms
    return result


def process_trial_file(path: Path, subject_id: str, condition: str) -> dict:
    """Preprocess one MAT recording and retain only trial-level labels."""
    path = Path(path)
    label = condition_task_label(condition)
    data = loadmat(path, variable_names=["SIG", "discardChannelsVec", "ref_signal", "fsamp"])
    reference = np.asarray(data["ref_signal"], dtype=np.float64).squeeze()
    fs = float(np.asarray(data["fsamp"], dtype=float).squeeze())
    if reference.ndim != 1 or reference.size < 8 or not np.isfinite(reference).all():
        raise ValueError(f"{path.name}: reference force trace is invalid")
    if not np.isfinite(fs) or fs <= 2 * BAND_HZ[1]:
        raise ValueError(f"{path.name}: sampling rate is not above twice the high cutoff")
    grid, channel_valid = _valid_mat_channels(
        data["SIG"], np.asarray(data["discardChannelsVec"]).astype(bool), reference.size
    )
    bouts, threshold = detect_force_bouts(
        reference,
        fs,
        minimum_bout_s=MINIMUM_FORCE_BOUT_SECONDS,
        merge_gap_s=0.25,
        smooth_s=0.10,
        threshold_fraction=0.20,
    )
    if not bouts:
        raise ValueError(f"{path.name}: no active force bouts meet the minimum duration")

    first_start, first_end = bouts[0]
    diagnostic_samples = min(first_end - first_start, int(round(2.0 * fs)))
    line_prominence = _line_prominence_db(
        grid[channel_valid, first_start : first_start + diagnostic_samples], fs
    )
    candidates = [
        (frequency, prominence)
        for frequency, prominence in line_prominence.items()
        if np.isfinite(prominence) and prominence > 10.0
    ]
    notch_frequency = max(candidates, key=lambda row: row[1])[0] if candidates else None
    bandpass_sos = signal.butter(
        FILTER_ORDER, BAND_HZ, btype="bandpass", fs=fs, output="sos"
    )
    sos = bandpass_sos
    notch_coefficients = None
    if notch_frequency is not None:
        b, a = signal.iirnotch(notch_frequency, Q=30.0, fs=fs)
        notch_sos = signal.tf2sos(b, a)
        sos = np.vstack((notch_sos, bandpass_sos))
        notch_coefficients = {"b": b.tolist(), "a": a.tolist(), "q": 30.0}

    window_samples = int(round(WINDOW_SECONDS * fs))
    hop_samples = int(round(window_samples * (1.0 - OVERLAP)))
    edge_samples = int(round(EDGE_MASK_SECONDS * fs))
    feature_rows: list[dict[str, float]] = []
    candidate_slices: list[tuple[np.ndarray, int]] = []
    window_active_centers_s: list[float] = []
    cumulative_active = 0.0
    bout_durations = []
    sos_records = []
    for start, end in bouts:
        segment = grid[channel_valid, start:end]
        filtered = signal.sosfiltfilt(sos, segment, axis=-1)
        bout_seconds = (end - start) / fs
        bout_durations.append(bout_seconds)
        valid_start = edge_samples
        valid_stop = filtered.shape[1] - edge_samples
        local_positions = range(valid_start, valid_stop - window_samples + 1, hop_samples)
        for local_start in local_positions:
            local_end = local_start + window_samples
            result = _window_features(filtered[:, local_start:local_end], fs)
            feature_rows.append({name: result[name] for name in METRIC_NAMES})
            candidate_slices.append((filtered, local_start))
            window_active_centers_s.append(cumulative_active + (local_start + window_samples / 2) / fs)
        sos_records.append({"start_sample": int(start), "end_sample": int(end)})
        cumulative_active += bout_seconds
    if len(feature_rows) < 2:
        raise ValueError(f"{path.name}: fewer than two valid 500 ms windows")

    metrics = {
        name: np.asarray([row[name] for row in feature_rows], dtype=float)
        for name in METRIC_NAMES
    }
    time_fraction = np.asarray(window_active_centers_s, dtype=float) / cumulative_active
    trial_features = summarize_window_metrics(metrics, time_fraction)
    selected = np.linspace(
        0, len(candidate_slices) - 1,
        min(MAX_CNN_WINDOWS_PER_TRIAL, len(candidate_slices)),
        dtype=int,
    )
    bag = np.zeros((len(selected), 65, window_samples), dtype=np.float32)
    for out_index, candidate_index in enumerate(selected):
        filtered, local_start = candidate_slices[int(candidate_index)]
        window = filtered[:, local_start : local_start + window_samples]
        bag[out_index, channel_valid] = window.astype(np.float32, copy=False)
    channel_window_rms = np.stack(
        [
            np.sqrt(np.mean(candidate_slices[i][0][:, candidate_slices[i][1] : candidate_slices[i][1] + window_samples] ** 2, axis=1))
            for i in range(len(candidate_slices))
        ]
    )
    channel_rms = np.zeros(65, dtype=np.float32)
    channel_rms[channel_valid] = np.median(channel_window_rms, axis=0).astype(np.float32)
    durations = np.asarray(bout_durations, dtype=float)
    total_seconds = reference.size / fs
    active_seconds = float(durations.sum())
    row = {
        "subject_id": str(subject_id),
        "condition": "30" if label == 0 else "TLim",
        "label": int(label),
        **trial_features,
        "total_duration_s": float(total_seconds),
        "active_duration_s": active_seconds,
        "active_to_total_ratio": float(active_seconds / total_seconds),
        "active_bout_count": int(len(bouts)),
        "bout_duration_median_s": float(np.median(durations)),
        "bout_duration_max_s": float(np.max(durations)),
        "force_threshold_uncalibrated_units": float(threshold),
        "eligible_window_count": int(len(feature_rows)),
        "valid_channel_count": int(channel_valid.sum()),
        "sampling_rate_hz": float(fs),
        "notch_frequency_hz": notch_frequency,
        "notch_prominence_50_db": line_prominence[50],
        "notch_prominence_60_db": line_prominence[60],
        "raw_member": path.name,
        "raw_member_sha256": _sha256(path),
    }
    return {
        "trial_row": row,
        "raw_bag": bag,
        "channel_valid": channel_valid,
        "channel_rms": channel_rms,
        "window_samples": window_samples,
        "fs_hz": fs,
        "active_bouts": bouts,
        "sos_coefficients": sos.tolist(),
        "notch_coefficients": notch_coefficients,
        "notch_prominence_db": line_prominence,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_trial_dataset(source_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Create trial-level feature rows and private raw-window bags."""
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    receipt_path = source_dir / "source_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing source receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("dataset_doi") != DATASET_DOI:
        raise ValueError("input source receipt has an unexpected dataset DOI")
    if not receipt.get("member_crc32_verified"):
        raise ValueError("all source members must pass their ZIP CRC checks")
    output_dir.mkdir(parents=True, exist_ok=True)
    bags_dir = output_dir / "trial_bags"
    bags_dir.mkdir(parents=True, exist_ok=True)

    trial_rows = []
    processing_rows = []
    for member in receipt["members"]:
        path = source_dir / member["subject_id"] / Path(member["archive_member"]).name
        if not path.is_file() or _sha256(path) != member["sha256"]:
            raise ValueError(f"source member missing or SHA-256 mismatch: {path.name}")
        processed = process_trial_file(path, member["subject_id"], member["condition"])
        trial_rows.append(processed["trial_row"])
        normalized_condition = "TLim" if str(member["condition"]).lower() == "tlim" else "30"
        bag_path = bags_dir / f"{member['subject_id']}_{normalized_condition}.npz"
        np.savez_compressed(
            bag_path,
            windows=processed["raw_bag"],
            channel_valid=processed["channel_valid"],
            channel_rms=processed["channel_rms"],
        )
        processing_rows.append({
            "subject_id": member["subject_id"],
            "condition": member["condition"],
            "active_bouts_sample_bounds": json.dumps(processed["active_bouts"]),
            "window_samples": processed["window_samples"],
            "hop_samples": int(round(processed["window_samples"] * (1 - OVERLAP))),
            "edge_mask_samples_each_end": int(round(EDGE_MASK_SECONDS * processed["fs_hz"])),
            "notch_frequency_hz": processed["trial_row"]["notch_frequency_hz"],
            "notch_prominence_db": json.dumps(processed["notch_prominence_db"], sort_keys=True),
            "notch_coefficients": json.dumps(processed["notch_coefficients"], sort_keys=True),
            "bandpass_sos_coefficients": json.dumps(processed["sos_coefficients"][-len(signal.butter(FILTER_ORDER, BAND_HZ, btype="bandpass", fs=processed["fs_hz"], output="sos")):]),
            "trial_bag_path": str(bag_path),
            "trial_bag_sha256": _sha256(bag_path),
        })
        print(f"processed {member['subject_id']} {member['condition']}: {processed['trial_row']['eligible_window_count']} windows, {processed['trial_row']['valid_channel_count']} channels", flush=True)

    frame = pd.DataFrame(trial_rows).sort_values(["subject_id", "label"], kind="stable").reset_index(drop=True)
    if frame.shape[0] != 28 or frame["subject_id"].nunique() != 14:
        raise ValueError(f"expected 28 trials from 14 paired subjects, found {frame.shape[0]} trials/{frame['subject_id'].nunique()} subjects")
    pair_labels = frame.groupby("subject_id")["label"].apply(lambda values: sorted(values.tolist()))
    if any(labels != [0, 1] for labels in pair_labels):
        raise ValueError("each included participant must contribute one reference and one task-to-failure trial")
    feature_columns = [
        f"{metric}__{stat}"
        for metric in METRIC_NAMES
        for stat in ("median", "iqr", "slope")
    ]
    frame.to_csv(output_dir / "trial_features.csv", index=False)
    pd.DataFrame(processing_rows).sort_values(["subject_id", "condition"]).to_csv(
        output_dir / "processing_audit.csv", index=False
    )
    schema = {
        "schema_id": "figshare_rf_task_outcome_trial_features_v1",
        "dataset_doi": DATASET_DOI,
        "prediction_unit": "complete recording trial; one feature row and one prediction per trial",
        "labels": {"30": 0, "TLim": 1},
        "label_semantics": "reference contraction condition versus sustained 30% peak-torque knee-extension task carried to task failure; not an instantaneous or per-window physiological fatigue state",
        "force_trace_use": "uncalibrated reference force is used only to find active contraction bouts and exclude short post-task MVC; never enters any model feature or target",
        "sampling_rate_hz": "read from each MAT file; expected to be 2048 Hz",
        "windows": {
            "type": "fixed-length sliding windows inside detected active bouts",
            "duration_s": WINDOW_SECONDS,
            "overlap_fraction": OVERLAP,
            "sample_count": "round(duration_s * measured_fs_hz)",
            "hop_samples": "round(window_samples * (1 - overlap_fraction))",
            "edge_mask_s_each_end": EDGE_MASK_SECONDS,
            "force_bout_minimum_duration_s": MINIMUM_FORCE_BOUT_SECONDS,
            "window_target": "none; windows are pooled to a complete-trial feature vector, and the CNN pools window logits before computing one trial-level loss",
        },
        "filter": {
            "bandpass": f"scipy.signal.butter(N={FILTER_ORDER}, Wn=[20,450] Hz, btype='bandpass', fs=measured_fs, output='sos')",
            "bandpass_prototype_order": FILTER_ORDER,
            "digital_bandpass_order": 2 * FILTER_ORDER,
            "application": "zero-phase scipy.signal.sosfiltfilt independently within detected active bouts",
            "conditional_notch": "Q=30 at 50 or 60 Hz only if pooled Welch PSD prominence exceeds 10 dB; coefficients and decisions saved in processing_audit.csv",
        },
        "feature_formulas": {
            "rms_log10": "log10(sqrt(sum((x[n]-mean(x))^2)/N))",
            "mav_log10": "log10(sum(abs(x[n]-mean(x)))/N)",
            "waveform_length_log10": "log10(sum(abs(x[n]-x[n-1])) / ((N-1)/fs))",
            "psd": "one-sided periodogram using Hann window and density scaling",
            "mdf_hz": "first f_k where cumulative Pxx reaches 0.5 * sum(Pxx), for 20<=f<=450 Hz",
            "mnf_hz": "sum(f_k * Pxx[k]) / sum(Pxx[k]), for 20<=f<=450 Hz",
            "bandpower_20_60_ratio": "sum(Pxx, 20<=f<60) / sum(Pxx, 20<=f<=450)",
            "bandpower_60_150_ratio": "sum(Pxx, 60<=f<150) / sum(Pxx, 20<=f<=450)",
            "spectral_entropy_norm": "-sum(p_k * log(p_k))/log(K), p_k=Pxx[k]/sum(Pxx), for 20<=f<=450 Hz",
            "channel_pooling": "median over usable RF electrode channels per window",
            "trial_pooling": "median, interquartile range, and least-squares slope across active-time-normalized windows",
        },
        "feature_columns": feature_columns,
        "time_only_control_features": [
            "total_duration_s", "active_duration_s", "active_to_total_ratio",
            "active_bout_count", "bout_duration_median_s", "bout_duration_max_s",
        ],
        "feature_exclusions": ["subject_id", "condition", "label", "force_threshold_uncalibrated_units", "raw_member", "raw_member_sha256", "notch_frequency_hz", "notch_prominence_50_db", "notch_prominence_60_db", "eligible_window_count", "valid_channel_count", "sampling_rate_hz"],
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    provenance = {
        "dataset_doi": DATASET_DOI,
        "source_receipt_sha256": _sha256(receipt_path),
        "member_count": len(receipt["members"]),
        "trial_count": int(len(frame)),
        "participant_count": int(frame["subject_id"].nunique()),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
        "source_members": [
            {key: member[key] for key in ("subject_id", "condition", "archive_member", "uncompressed_bytes", "zip_crc32", "sha256")}
            for member in receipt["members"]
        ],
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return frame
