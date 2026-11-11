"""Participant-grouped training on independently recorded external EMG labels.

The Cerqueira target is the source's three-level, researcher-operated
self-perceived fatigue-state label (0/1/2). All outputs are private development
artifacts; window counts are descriptive and inference is clustered by subject.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import optimize, signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .external_provenance import code_provenance


CLASS_IDS = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = {0: "NO_FATIGUE", 1: "TRANSITION", 2: "FATIGUE"}
WINDOW_SECONDS = 4.0
WINDOW_HOP_SECONDS = 4.0  # non-overlapping; subject remains the statistical unit
LABEL_PURITY_MINIMUM = 0.80
LABEL_RATE_MINIMUM_HZ = 45.0
LABEL_RATE_MAXIMUM_HZ = 55.0
MAX_START_OFFSET_SECONDS = 0.05
MIN_GROUPS_PER_CLASS = 2
MIN_TRIALS_PER_CLASS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _subject_trial_from_path(path: Path) -> tuple[int, int] | None:
    subject_match = re.search(r"subject[_ -]?(\d+)", str(path), flags=re.IGNORECASE)
    trial_match = re.search(r"trial[_ -]?(\d+)", path.name, flags=re.IGNORECASE)
    if not subject_match or not trial_match:
        return None
    return int(subject_match.group(1)), int(trial_match.group(1))


def _label_map(label_root: Path) -> dict[tuple[int, int], pd.DataFrame]:
    result: dict[tuple[int, int], pd.DataFrame] = {}
    for path in label_root.rglob("*.csv"):
        identity = _subject_trial_from_path(path)
        if identity is None:
            continue
        raw = pd.read_csv(path)
        columns = {str(column).strip().casefold(): column for column in raw.columns}
        if "time" not in columns or "label" not in columns:
            continue
        table = raw[[columns["time"], columns["label"]]].copy()
        table.columns = ["time_s", "label"]
        table["time_s"] = pd.to_numeric(table["time_s"], errors="coerce")
        table["label"] = pd.to_numeric(table["label"], errors="coerce")
        table = table.dropna().sort_values("time_s", kind="stable")
        table = table[table["label"].isin(CLASS_IDS)]
        if table.empty:
            continue
        key = (identity[0], identity[1])
        if key in result:
            raise ValueError(f"duplicate Cerqueira label file for subject/trial {key}")
        result[key] = table.reset_index(drop=True)
    return result


def audit_cerqueira_labels(*, label_root: Path, output_dir: Path) -> dict[str, Any]:
    """Summarize source label coverage by participant and trial before EMG parsing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _label_map(label_root)
    trial_rows: list[dict[str, Any]] = []
    transition_counts: Counter[tuple[int, int]] = Counter()
    for (subject, trial), frame in sorted(labels.items()):
        y = frame["label"].to_numpy(dtype=np.int64)
        counts = Counter(int(value) for value in y)
        delta = np.diff(frame["time_s"].to_numpy(dtype=np.float64))
        transitions = Counter((int(left), int(right)) for left, right in zip(y[:-1], y[1:], strict=True) if left != right)
        transition_counts.update(transitions)
        row: dict[str, Any] = {
            "subject_id": f"S{subject:02d}",
            "subject_number": subject,
            "trial_id": f"T{trial:02d}",
            "trial_number": trial,
            "label_samples_descriptive_only": len(y),
            "start_time_s": float(frame["time_s"].min()),
            "end_time_s": float(frame["time_s"].max()),
            "duration_s": float(frame["time_s"].max() - frame["time_s"].min()),
            "median_sampling_rate_hz": float(1.0 / np.median(delta)) if len(delta) else None,
            "label_transition_count": int(sum(transitions.values())),
            "has_all_three_levels": len(counts) == len(CLASS_IDS),
        }
        for class_id in CLASS_IDS:
            row[f"class_{int(class_id)}_samples"] = int(counts.get(int(class_id), 0))
            row[f"class_{int(class_id)}_fraction"] = float(counts.get(int(class_id), 0) / len(y)) if len(y) else None
        trial_rows.append(row)
    trial_table = pd.DataFrame(trial_rows)
    subject_table = trial_table.groupby(["subject_id", "subject_number"], as_index=False).agg(
        trial_count=("trial_id", "nunique"),
        label_samples_descriptive_only=("label_samples_descriptive_only", "sum"),
        trials_with_all_three_levels=("has_all_three_levels", "sum"),
        class_0_samples=("class_0_samples", "sum"),
        class_1_samples=("class_1_samples", "sum"),
        class_2_samples=("class_2_samples", "sum"),
    )
    class_rows = []
    for class_id in CLASS_IDS:
        active = trial_table[trial_table[f"class_{int(class_id)}_samples"] > 0]
        class_rows.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "timestamp_count_descriptive_only": int(trial_table[f"class_{int(class_id)}_samples"].sum()),
            "independent_participant_count": int(active["subject_id"].nunique()),
            "trial_count": int(active["trial_id"].count()),
        })
    trial_table.to_csv(output_dir / "cerqueira_label_support_by_trial.csv", index=False)
    subject_table.to_csv(output_dir / "cerqueira_label_support_by_subject.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "cerqueira_label_distribution_by_class.csv", index=False)
    expected_keys = {(subject, trial) for subject in range(1, 14) for trial in range(1, 13)}
    summary = {
        "status": "completed",
        "dataset_key": "cerqueira_v2",
        "target_levels": CLASS_NAMES,
        "independent_unit": "participant",
        "participant_count": int(trial_table["subject_id"].nunique()),
        "trial_count": int(len(trial_table)),
        "expected_13_by_12_trials_present": set(labels) == expected_keys,
        "missing_participant_trials": sorted([list(key) for key in expected_keys - set(labels)]),
        "unexpected_participant_trials": sorted([list(key) for key in set(labels) - expected_keys]),
        "trials_with_all_three_levels": int(trial_table["has_all_three_levels"].sum()),
        "label_samples_descriptive_only": int(trial_table["label_samples_descriptive_only"].sum()),
        "median_sampling_rate_hz": float(trial_table["median_sampling_rate_hz"].median()),
        "class_distribution": class_rows,
        "state_transitions": {f"{left}->{right}": int(count) for (left, right), count in sorted(transition_counts.items())},
        "window_count_is_independent_n": False,
        "note": "50 Hz label timestamps are repeated measures; only 13 participants are independent units.",
        "support_by_trial_file": "cerqueira_label_support_by_trial.csv",
        "support_by_subject_file": "cerqueira_label_support_by_subject.csv",
        "code_provenance": code_provenance(__file__),
    }
    _write_json(output_dir / "cerqueira_label_audit.json", summary)
    return summary


def _read_emg_trial(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Read the four synchronized time/value pairs documented by the source."""
    if raw.shape[1] < 8:
        raise ValueError(f"expected at least four time/value pairs, got {raw.shape[1]} columns")
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for channel in range(4):
        time = pd.to_numeric(raw.iloc[:, channel * 2], errors="coerce").to_numpy(dtype=np.float64)
        values = pd.to_numeric(raw.iloc[:, channel * 2 + 1], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(time) & np.isfinite(values)
        time, values = time[valid], values[valid]
        if len(time) < 1000:
            raise ValueError(f"EMG channel {channel} has too few valid samples: {len(time)}")
        order = np.argsort(time, kind="stable")
        time, values = time[order], values[order]
        unique = np.concatenate(([True], np.diff(time) > 0))
        time, values = time[unique], values[unique]
        pairs.append((time, values))

    fs_values = [1.0 / np.median(np.diff(time)) for time, _ in pairs]
    fs = float(np.median(fs_values))
    if not 800.0 <= fs <= 2000.0:
        raise ValueError(f"unexpected EMG sample rate inferred from timestamps: {fs:.2f} Hz")
    start = max(time[0] for time, _ in pairs)
    end = min(time[-1] for time, _ in pairs)
    dt = 1.0 / fs
    common_time = np.arange(start, end, dt, dtype=np.float64)
    if len(common_time) < int(fs * WINDOW_SECONDS):
        raise ValueError("synchronized four-channel EMG overlap is shorter than one analysis window")
    channels = np.vstack([np.interp(common_time, time, values) for time, values in pairs])
    sos = signal.butter(4, (20.0, 450.0), btype="bandpass", fs=fs, output="sos")
    filtered = np.vstack([signal.sosfiltfilt(sos, values) for values in channels]).astype(np.float32)
    return common_time, filtered, fs


def _window_features(window: np.ndarray, fs: float) -> dict[str, float]:
    output: dict[str, float] = {}
    for channel, values in enumerate(window):
        prefix = f"ch{channel + 1}"
        values = np.asarray(values, dtype=np.float64)
        centered = values - np.mean(values)
        rms = float(np.sqrt(np.mean(centered * centered)))
        diff = np.diff(centered)
        scale = max(float(np.std(centered)), 1e-12)
        output[f"{prefix}_rms"] = rms
        output[f"{prefix}_mav"] = float(np.mean(np.abs(centered)))
        output[f"{prefix}_waveform_length"] = float(np.mean(np.abs(diff)))
        output[f"{prefix}_zero_crossing_rate"] = float(np.mean((centered[:-1] * centered[1:]) < -(0.01 * scale) ** 2))
        output[f"{prefix}_slope_sign_change_rate"] = float(np.mean((diff[:-1] * diff[1:]) < -(0.01 * scale) ** 2))
        frequencies, power = signal.welch(centered, fs=fs, nperseg=min(2048, len(centered)))
        total = float(np.sum(power))
        if total <= 0 or not np.isfinite(total):
            output[f"{prefix}_mean_frequency"] = 0.0
            output[f"{prefix}_median_frequency"] = 0.0
            for band in ("20_60", "60_150", "150_450"):
                output[f"{prefix}_band_{band}"] = 0.0
            output[f"{prefix}_spectral_entropy"] = 0.0
            continue
        output[f"{prefix}_mean_frequency"] = float(np.sum(frequencies * power) / total)
        cumulative = np.cumsum(power)
        output[f"{prefix}_median_frequency"] = float(frequencies[min(int(np.searchsorted(cumulative, total / 2)), len(frequencies) - 1)])
        for low, high, name in ((20, 60, "20_60"), (60, 150, "60_150"), (150, 450, "150_450")):
            selected = (frequencies >= low) & (frequencies < high)
            output[f"{prefix}_band_{name}"] = float(np.sum(power[selected]) / total)
        probability = power / total
        output[f"{prefix}_spectral_entropy"] = float(-np.sum(probability[probability > 0] * np.log(probability[probability > 0])))
    return output


def _build_one_trial(
    subject: int,
    trial: int,
    raw: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, Any]]:
    time, emg, fs = _read_emg_trial(raw)
    label_time = labels["time_s"].to_numpy(dtype=np.float64)
    label_value = labels["label"].to_numpy(dtype=np.int64)
    label_rate_hz = float(1.0 / np.median(np.diff(label_time))) if len(label_time) > 1 else float("nan")
    start_offset_s = float(label_time[0] - time[0]) if len(label_time) else float("nan")
    if not LABEL_RATE_MINIMUM_HZ <= label_rate_hz <= LABEL_RATE_MAXIMUM_HZ:
        raise ValueError(f"subject {subject} trial {trial} label rate is outside the audited 50 Hz range: {label_rate_hz:.3f}")
    if not np.isfinite(start_offset_s) or abs(start_offset_s) > MAX_START_OFFSET_SECONDS:
        raise ValueError(f"subject {subject} trial {trial} label and EMG clocks do not share a verified start: {start_offset_s:.6f} s")
    rows: list[dict[str, Any]] = []
    windows: list[np.ndarray] = []
    duration = min(float(time[-1] - time[0]), float(label_time[-1] - time[0]))
    window_samples = int(round(WINDOW_SECONDS * fs))
    hop_samples = int(round(WINDOW_HOP_SECONDS * fs))
    window_index = 0
    for left in np.arange(float(time[0]), float(time[0]) + duration - WINDOW_SECONDS + 1e-9, WINDOW_HOP_SECONDS):
        right = left + WINDOW_SECONDS
        start_index = int(np.searchsorted(time, left, side="left"))
        stop_index = int(np.searchsorted(time, right, side="left"))
        if stop_index - start_index < int(window_samples * 0.98):
            continue
        signal_window = emg[:, start_index:stop_index]
        if signal_window.shape[1] < window_samples:
            continue
        signal_window = signal_window[:, :window_samples]
        within = (label_time >= left) & (label_time < right)
        window_labels = label_value[within]
        if len(window_labels) < int(WINDOW_SECONDS * 40):
            continue
        counts = np.bincount(window_labels, minlength=len(CLASS_IDS))
        majority = int(np.argmax(counts))
        purity = float(counts[majority] / len(window_labels))
        if purity < LABEL_PURITY_MINIMUM:
            continue
        relative_time = float((left + WINDOW_SECONDS / 2 - time[0]) / max(duration, WINDOW_SECONDS))
        row: dict[str, Any] = {
            "subject_id": f"S{subject:02d}",
            "subject_number": subject,
            "trial_id": f"T{trial:02d}",
            "trial_number": trial,
            "recording_id": f"S{subject:02d}_T{trial:02d}",
            "window_index": window_index,
            "window_start_s": left,
            "window_center_s": left + WINDOW_SECONDS / 2,
            "trial_duration_s": duration,
            "relative_time": relative_time,
            "label": majority,
            "label_name": CLASS_NAMES[majority],
            "label_purity": purity,
            "label_sample_count": int(len(window_labels)),
            "sampling_rate_hz": fs,
            "window_overlap": False,
        }
        row.update(_window_features(signal_window, fs))
        rows.append(row)
        windows.append(signal_window)
        window_index += 1
    return rows, windows, {
        "subject_id": f"S{subject:02d}",
        "trial_id": f"T{trial:02d}",
        "emg_start_s": float(time[0]),
        "label_start_s": float(label_time[0]),
        "start_offset_s": start_offset_s,
        "emg_end_s": float(time[-1]),
        "label_end_s": float(label_time[-1]),
        "end_offset_s": float(label_time[-1] - time[-1]),
        "emg_rate_hz": float(fs),
        "label_rate_hz": label_rate_hz,
    }


def build_cerqueira_windows(
    *,
    signal_archive: Path,
    label_root: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Extract aligned, non-overlapping 4 s windows and source-auditable labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "cerqueira_v2_window_features.parquet"
    signal_path = output_dir / "cerqueira_v2_signal_windows.npy"
    manifest_path = output_dir / "window_bank_manifest.json"
    if not force and feature_path.exists() and signal_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signal_archive_sha256") == _sha256(signal_archive):
            return manifest

    if not signal_archive.is_file():
        raise FileNotFoundError(f"verified Cerqueira signal archive not found: {signal_archive}")
    labels_by_trial = _label_map(label_root)
    rows: list[dict[str, Any]] = []
    signal_windows: list[np.ndarray] = []
    synchronization_rows: list[dict[str, Any]] = []
    matched_trials: set[tuple[int, int]] = set()
    # The publisher checksum was verified during download. Reading each nested
    # archive/member below also exercises its ZIP CRC without a second full pass.
    raw_archive = zipfile.ZipFile(signal_archive)
    for member in raw_archive.infolist():
        if member.is_dir():
            continue
        member_path = Path(member.filename)
        subject_match = re.search(r"subject[_ -]?(\d+)\.zip$", member_path.name, flags=re.IGNORECASE)
        if subject_match:
            subject = int(subject_match.group(1))
            inner_bytes = raw_archive.read(member)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as subject_zip:
                for trial_member in subject_zip.infolist():
                    if trial_member.is_dir() or not trial_member.filename.casefold().endswith(".csv"):
                        continue
                    trial_match = re.search(r"trial[_ -]?(\d+)\.csv$", trial_member.filename, flags=re.IGNORECASE)
                    if trial_match is None:
                        continue
                    trial = int(trial_match.group(1))
                    if trial < 1 or (subject, trial) not in labels_by_trial:
                        continue
                    key = (subject, trial)
                    if key in matched_trials:
                        raise ValueError(f"duplicate raw EMG trial in publisher archive: {key}")
                    raw = pd.read_csv(subject_zip.open(trial_member))
                    new_rows, new_windows, synchronization = _build_one_trial(subject, trial, raw, labels_by_trial[key])
                    rows.extend(new_rows)
                    signal_windows.extend(new_windows)
                    synchronization_rows.append(synchronization)
                    matched_trials.add(key)
                    print(f"Parsed Cerqueira subject {subject:02d} trial {trial:02d}: {len(new_rows)} windows", flush=True)
        elif member_path.suffix.casefold() == ".csv":
            identity = _subject_trial_from_path(member_path)
            if identity is None or identity not in labels_by_trial:
                continue
            subject, trial = identity
            if identity in matched_trials:
                raise ValueError(f"duplicate raw EMG trial in publisher archive: {identity}")
            raw = pd.read_csv(raw_archive.open(member))
            new_rows, new_windows, synchronization = _build_one_trial(subject, trial, raw, labels_by_trial[identity])
            rows.extend(new_rows)
            signal_windows.extend(new_windows)
            synchronization_rows.append(synchronization)
            matched_trials.add(identity)
            print(f"Parsed Cerqueira subject {subject:02d} trial {trial:02d}: {len(new_rows)} windows", flush=True)
    raw_archive.close()

    expected_trials = set(labels_by_trial)
    if matched_trials != expected_trials:
        missing = sorted(expected_trials - matched_trials)
        extra = sorted(matched_trials - expected_trials)
        raise RuntimeError(f"raw/label trial pairing incomplete; missing={missing[:20]}, extra={extra[:20]}")
    if not rows:
        raise RuntimeError("no aligned Cerqueira windows passed signal and label quality gates")
    table = pd.DataFrame(rows)
    bank = np.stack(signal_windows).astype(np.float32, copy=False)
    if len(table) != len(bank):
        raise RuntimeError("feature rows and signal windows are not one-to-one")
    table.insert(0, "sample_id", [f"CERQ_{index:06d}" for index in range(len(table))])
    table.to_parquet(feature_path, index=False)
    np.save(signal_path, bank, allow_pickle=False)
    class_counts = Counter(int(x) for x in table["label"])
    group_support = table.groupby("subject_id")["label"].nunique().to_dict()
    trial_support = table.groupby("recording_id")["label"].nunique().to_dict()
    manifest = {
        "status": "completed",
        "dataset_key": "cerqueira_v2",
        "target": "source-reported self-perceived fatigue level: 0=no fatigue, 1=transition to fatigue, 2=fatigue",
        "target_provenance": "state selected by participants after researcher prompts; not a physiological fatigue ground truth",
        "signal_archive_sha256": _sha256(signal_archive),
        "source_archive_sha256": {
            str(path.relative_to(label_root.parent)).replace("\\", "/"): _sha256(path)
            for path in (
                signal_archive,
                label_root.parent / "originals" / "self_perceived_fatigue_index.zip",
                label_root.parent / "originals" / "Metadata.xlsx",
                label_root.parent / "originals" / "protocol.xlsx",
                label_root.parent / "official_metadata.json",
            )
            if path.is_file() and path.is_relative_to(label_root.parent)
        },
        "label_csv_sha256": {
            str(path.relative_to(label_root)).replace("\\", "/"): _sha256(path)
            for path in sorted(label_root.rglob("*.csv"), key=lambda value: str(value).casefold())
        },
        "label_root": str(label_root.resolve()),
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": WINDOW_HOP_SECONDS,
        "label_purity_minimum": LABEL_PURITY_MINIMUM,
        "class_counts_descriptive_windows": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "participant_count_independent_units": int(table["subject_id"].nunique()),
        "trial_count": int(table["recording_id"].nunique()),
        "groups_with_all_three_labels": int(sum(value == len(CLASS_IDS) for value in group_support.values())),
        "trials_with_all_three_labels": int(sum(value == len(CLASS_IDS) for value in trial_support.values())),
        "window_count_is_descriptive_only": True,
        "overlapping_windows_treated_as_independent": False,
        "preprocessing": "4th-order zero-phase Butterworth 20-450 Hz; timestamps synchronized on four active time/value channel pairs; channel 1 timebase with linear interpolation",
        "time_synchronization": {
            "method": "match elapsed-time windows from the shared EMG and source rating timestamps; all matched trials must start within 50 ms and ratings must sample at 45-55 Hz",
            "matched_trials": len(synchronization_rows),
            "max_abs_start_offset_s": float(max(abs(row["start_offset_s"]) for row in synchronization_rows)),
            "min_emg_rate_hz": float(min(row["emg_rate_hz"] for row in synchronization_rows)),
            "max_emg_rate_hz": float(max(row["emg_rate_hz"] for row in synchronization_rows)),
            "min_label_rate_hz": float(min(row["label_rate_hz"] for row in synchronization_rows)),
            "max_label_rate_hz": float(max(row["label_rate_hz"] for row in synchronization_rows)),
            "per_trial_offsets": synchronization_rows,
        },
        "feature_file": str(feature_path.resolve()),
        "feature_file_sha256": _sha256(feature_path),
        "signal_window_file": str(signal_path.resolve()),
        "signal_window_file_sha256": _sha256(signal_path),
        "signal_window_shape": list(bank.shape),
        "feature_columns": [column for column in table.columns if column.startswith("ch")],
        "paired_raw_label_trials": len(matched_trials),
        "code_provenance": code_provenance(__file__),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _group_class_weights(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    keys = pd.DataFrame({"group": frame["subject_id"].astype(str), "trial": frame["recording_id"].astype(str), "label": y})
    group_count = int(keys["group"].nunique())
    windows_per_group_class_trial = keys.groupby(["group", "label", "trial"], observed=True)["label"].transform("size").to_numpy(dtype=np.float64)
    classes_per_group = keys.groupby("group", observed=True)["label"].transform("nunique").to_numpy(dtype=np.float64)
    trials_per_group_class = keys.groupby(["group", "label"], observed=True)["trial"].transform("nunique").to_numpy(dtype=np.float64)
    weights = 1.0 / (group_count * classes_per_group * trials_per_group_class * windows_per_group_class_trial)
    return weights * (len(weights) / weights.sum())


def _weighted_standardize_fit(x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = weight.astype(np.float64)
    weight = weight / weight.sum()
    mean = np.sum(x.astype(np.float64) * weight[:, None], axis=0)
    variance = np.sum(np.square(x.astype(np.float64) - mean) * weight[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float64), scale.astype(np.float64)


def _fit_logistic(
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    c_value: float,
) -> tuple[dict[str, Any], np.ndarray]:
    weight = _group_class_weights(frame, y)
    mean, scale = _weighted_standardize_fit(x, weight)
    standardized = (x - mean) / scale
    estimator = LogisticRegression(C=float(c_value), max_iter=2000, solver="lbfgs", random_state=73)
    estimator.fit(standardized, y, sample_weight=weight)
    return {"estimator": estimator, "mean": mean, "scale": scale}, estimator.predict_proba(standardized)


def _predict_logistic(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return model["estimator"].predict_proba((x - model["mean"]) / model["scale"])


def _fit_temperature(probabilities: np.ndarray, y: np.ndarray, groups: Sequence[str]) -> float:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    group_values = np.asarray([str(value) for value in groups])
    sample_weights = np.zeros(len(y), dtype=np.float64)
    for group in sorted(set(group_values)):
        selected_group = group_values == group
        supported_classes = [int(class_id) for class_id in CLASS_IDS if np.any(selected_group & (y == class_id))]
        for class_id in supported_classes:
            selected = selected_group & (y == class_id)
            sample_weights[selected] = 1.0 / (len(set(group_values)) * len(supported_classes) * int(selected.sum()))
    sample_weights /= sample_weights.sum()
    def loss(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        adjusted = logits / temperature
        adjusted -= adjusted.max(axis=1, keepdims=True)
        normalized = np.exp(adjusted)
        normalized /= normalized.sum(axis=1, keepdims=True)
        row_loss = -np.log(np.clip(normalized[np.arange(len(y)), y], 1e-12, 1.0))
        return float(np.sum(sample_weights * row_loss))
    result = optimize.minimize_scalar(loss, bounds=(np.log(0.1), np.log(10.0)), method="bounded")
    return float(np.exp(result.x))


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _require_partition_support(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for class_id in CLASS_IDS:
        active = frame[frame["label"].astype(int) == int(class_id)]
        groups = int(active["subject_id"].nunique())
        trials = int(active["recording_id"].nunique())
        windows = int(len(active))
        details[str(int(class_id))] = {"windows_descriptive_only": windows, "participants": groups, "trials": trials}
        if groups < MIN_GROUPS_PER_CLASS or trials < MIN_TRIALS_PER_CLASS or windows == 0:
            raise ValueError(f"{name} lacks independent support for class {class_id}: {details[str(int(class_id))]}")
    return details


def _group_balanced_macro_f1(y: np.ndarray, prediction: np.ndarray, groups: Sequence[str]) -> float:
    scores = []
    for group in sorted(set(str(value) for value in groups)):
        selected = np.asarray([str(value) == group for value in groups], dtype=bool)
        scores.append(f1_score(y[selected], prediction[selected], labels=CLASS_IDS, average="macro", zero_division=0))
    return float(np.mean(scores))


def _metrics(y: np.ndarray, probabilities: np.ndarray, groups: Sequence[str]) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    group_metrics: list[dict[str, Any]] = []
    for group in sorted(set(str(value) for value in groups)):
        selected = np.asarray([str(value) == group for value in groups], dtype=bool)
        group_y = y[selected]
        group_pred = prediction[selected]
        group_prob = probabilities[selected]
        support = {str(int(c)): int(np.sum(group_y == c)) for c in CLASS_IDS}
        if any(value == 0 for value in support.values()):
            raise ValueError(f"held-out participant {group} lacks one or more class support: {support}")
        precision, recall, f1, _ = precision_recall_fscore_support(group_y, group_pred, labels=CLASS_IDS, zero_division=0)
        one_hot = np.eye(len(CLASS_IDS))[group_y]
        group_metrics.append({
            "subject_id": group,
            "windows_descriptive_only": int(np.sum(selected)),
            "accuracy": float(accuracy_score(group_y, group_pred)),
            "macro_f1": float(np.mean(f1)),
            "balanced_accuracy": float(balanced_accuracy_score(group_y, group_pred)),
            "mcc": float(matthews_corrcoef(group_y, group_pred)),
            "cohen_kappa": float(cohen_kappa_score(group_y, group_pred)),
            "brier_score": float(np.mean(np.sum((group_prob - one_hot) ** 2, axis=1))),
            "confusion_matrix": confusion_matrix(group_y, group_pred, labels=CLASS_IDS).tolist(),
            **{f"precision_class_{c}": float(precision[i]) for i, c in enumerate(CLASS_IDS)},
            **{f"recall_class_{c}": float(recall[i]) for i, c in enumerate(CLASS_IDS)},
            **{f"f1_class_{c}": float(f1[i]) for i, c in enumerate(CLASS_IDS)},
        })
    group_frame = pd.DataFrame(group_metrics)
    group_normalized = []
    for row in group_metrics:
        cm = np.asarray(row["confusion_matrix"], dtype=np.float64)
        group_normalized.append(cm / cm.sum())
    normalized_confusion = np.mean(group_normalized, axis=0)
    true_one_hot = np.eye(len(CLASS_IDS))[y]
    try:
        group_pr = [average_precision_score((y[np.asarray([str(v) == g for v in groups])] == c).astype(int), probabilities[np.asarray([str(v) == g for v in groups])][:, int(c)]) for g in sorted(set(str(v) for v in groups)) for c in CLASS_IDS]
        group_roc = [roc_auc_score((y[np.asarray([str(v) == g for v in groups])] == c).astype(int), probabilities[np.asarray([str(v) == g for v in groups])][:, int(c)]) for g in sorted(set(str(v) for v in groups)) for c in CLASS_IDS]
    except ValueError:
        group_pr, group_roc = [], []
    return {
        "participant_count": int(group_frame.shape[0]),
        "window_count_descriptive_only": int(len(y)),
        "group_macro_f1": float(group_frame["macro_f1"].mean()),
        "group_accuracy": float(group_frame["accuracy"].mean()),
        "group_balanced_accuracy": float(group_frame["balanced_accuracy"].mean()),
        "group_mcc": float(group_frame["mcc"].mean()),
        "group_cohen_kappa": float(group_frame["cohen_kappa"].mean()),
        "group_brier_score": float(group_frame["brier_score"].mean()),
        "group_macro_average_precision_ovr": float(np.mean(group_pr)) if group_pr else None,
        "group_macro_roc_auc_ovr": float(np.mean(group_roc)) if group_roc else None,
        "macro_f1_pooled_descriptive": float(f1_score(y, prediction, labels=CLASS_IDS, average="macro", zero_division=0)),
        "pooled_accuracy_descriptive": float(accuracy_score(y, prediction)),
        "group_balanced_confusion_matrix": normalized_confusion.tolist(),
        "pooled_confusion_matrix_descriptive": confusion_matrix(y, prediction, labels=CLASS_IDS).tolist(),
        "per_group": group_metrics,
        "window_count_is_independent_n": False,
    }


def _paired_group_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    metric: str,
    replicates: int = 2000,
    seed: int = 1811,
) -> dict[str, Any]:
    groups = sorted(set(left["subject_id"].astype(str)) & set(right["subject_id"].astype(str)))
    if len(groups) < 4:
        return {"metric": metric, "participant_count": len(groups), "status": "insufficient_independent_groups"}
    left_values = left.set_index("subject_id")[metric].to_dict()
    right_values = right.set_index("subject_id")[metric].to_dict()
    differences = np.asarray([float(left_values[g]) - float(right_values[g]) for g in groups])
    rng = np.random.default_rng(seed)
    samples = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        selected = rng.integers(0, len(groups), size=len(groups))
        samples[index] = float(np.mean(differences[selected]))
    return {
        "metric": metric,
        "bootstrap_unit": "participant; paired left-minus-right",
        "participant_count": len(groups),
        "delta": float(np.mean(differences)),
        "ci_95_low": float(np.quantile(samples, 0.025)),
        "ci_95_high": float(np.quantile(samples, 0.975)),
        "requested_replicates": int(replicates),
        "valid_replicates": int(replicates),
    }


def _temporal_features(frame: pd.DataFrame) -> np.ndarray:
    """Use only within-trial relative time for the predefined temporal control."""
    return frame["relative_time"].to_numpy(dtype=np.float64).reshape(-1, 1)


def _trial_condition_features(frame: pd.DataFrame) -> np.ndarray:
    """Encode relative time and the source's 12 repeated movement conditions."""
    trial_number = frame["trial_number"].to_numpy(dtype=np.int64)
    if np.any((trial_number < 1) | (trial_number > 12)):
        raise ValueError("Cerqueira trial-condition control expects trial numbers in 1..12")
    condition = np.eye(12, dtype=np.float64)[trial_number - 1]
    return np.column_stack((_temporal_features(frame), condition))


def run_cerqueira_grouped_stage(
    *,
    feature_path: Path,
    output_dir: Path,
    seed: int = 731,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Fit regularized multiclass and time-only models with nested subject CV."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite external development stage: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(feature_path).reset_index(drop=True)
    if frame["subject_id"].nunique() < 12:
        raise ValueError("Cerqueira outer 4-fold participant-grouped evaluation requires at least 12 participants")
    feature_columns = [column for column in frame.columns if re.match(r"ch\d+_", str(column))]
    if not feature_columns:
        raise ValueError("no signal-derived feature columns found")
    if not np.isfinite(frame[feature_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("non-finite EMG features must be audited before fitting")
    overall_support = _require_partition_support(frame, "Cerqueira full development set")
    labels = frame["label"].to_numpy(dtype=np.int64)
    groups = frame["subject_id"].astype(str).to_numpy()
    x = frame[feature_columns].to_numpy(dtype=np.float64)
    outer_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=int(seed))
    oof_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    fold_receipts: list[dict[str, Any]] = []
    for outer_fold, (train_index, validation_index) in enumerate(outer_cv.split(x, labels, groups), start=1):
        train_frame = frame.iloc[train_index].reset_index(drop=True)
        validation_frame = frame.iloc[validation_index].reset_index(drop=True)
        train_support = _require_partition_support(train_frame, f"outer {outer_fold} training")
        validation_support = _require_partition_support(validation_frame, f"outer {outer_fold} validation")
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=int(seed + outer_fold))
        inner_splits = list(inner_cv.split(train_frame[feature_columns], train_frame["label"], train_frame["subject_id"]))
        inner_oof_by_c = {float(c): np.full((len(train_frame), len(CLASS_IDS)), np.nan, dtype=np.float64) for c in (0.1, 1.0, 10.0)}
        inner_oof_time = np.full((len(train_frame), len(CLASS_IDS)), np.nan, dtype=np.float64)
        inner_oof_condition = np.full((len(train_frame), len(CLASS_IDS)), np.nan, dtype=np.float64)
        for inner_fold, (inner_train_index, inner_validation_index) in enumerate(inner_splits, start=1):
            inner_train = train_frame.iloc[inner_train_index].reset_index(drop=True)
            inner_validation = train_frame.iloc[inner_validation_index].reset_index(drop=True)
            inner_train_support = _require_partition_support(inner_train, f"outer {outer_fold} inner {inner_fold} training")
            inner_validation_support = _require_partition_support(inner_validation, f"outer {outer_fold} inner {inner_fold} validation")
            y_train = inner_train["label"].to_numpy(dtype=np.int64)
            y_validation = inner_validation["label"].to_numpy(dtype=np.int64)
            for c_value in inner_oof_by_c:
                model, _ = _fit_logistic(inner_train, inner_train[feature_columns].to_numpy(dtype=np.float64), y_train, c_value)
                probabilities = _predict_logistic(model, inner_validation[feature_columns].to_numpy(dtype=np.float64))
                inner_oof_by_c[c_value][inner_validation_index] = probabilities
                predictions = np.argmax(probabilities, axis=1)
                score = _group_balanced_macro_f1(y_validation, predictions, inner_validation["subject_id"].astype(str).tolist())
                candidate_rows.append({
                    "outer_fold_id": outer_fold,
                    "inner_fold_id": inner_fold,
                    "model_name": "signal_logistic",
                    "C": c_value,
                    "group_macro_f1": score,
                    "inner_train_participants": json.dumps(sorted(inner_train["subject_id"].astype(str).unique())),
                    "inner_validation_participants": json.dumps(sorted(inner_validation["subject_id"].astype(str).unique())),
                    "train_support": json.dumps(inner_train_support),
                    "validation_support": json.dumps(inner_validation_support),
                })

            time_x_train = _temporal_features(inner_train)
            time_x_validation = _temporal_features(inner_validation)
            time_model, _ = _fit_logistic(inner_train, time_x_train, y_train, 1.0)
            inner_oof_time[inner_validation_index] = _predict_logistic(time_model, time_x_validation)
            condition_x_train = _trial_condition_features(inner_train)
            condition_x_validation = _trial_condition_features(inner_validation)
            condition_model, _ = _fit_logistic(inner_train, condition_x_train, y_train, 1.0)
            inner_oof_condition[inner_validation_index] = _predict_logistic(condition_model, condition_x_validation)
        selected_c = max(inner_oof_by_c, key=lambda c: _group_balanced_macro_f1(train_frame["label"].to_numpy(dtype=np.int64), np.argmax(inner_oof_by_c[c], axis=1), train_frame["subject_id"].astype(str).tolist()))
        inner_probabilities = inner_oof_by_c[selected_c]
        if not np.isfinite(inner_probabilities).all() or not np.isfinite(inner_oof_time).all() or not np.isfinite(inner_oof_condition).all():
            raise RuntimeError(f"outer fold {outer_fold}: incomplete inner out-of-fold probability matrix")
        inner_temperature = _fit_temperature(inner_probabilities, train_frame["label"].to_numpy(dtype=np.int64), train_frame["subject_id"].astype(str).tolist())
        inner_time_temperature = _fit_temperature(inner_oof_time, train_frame["label"].to_numpy(dtype=np.int64), train_frame["subject_id"].astype(str).tolist())
        inner_condition_temperature = _fit_temperature(inner_oof_condition, train_frame["label"].to_numpy(dtype=np.int64), train_frame["subject_id"].astype(str).tolist())
        final_model, _ = _fit_logistic(train_frame, train_frame[feature_columns].to_numpy(dtype=np.float64), train_frame["label"].to_numpy(dtype=np.int64), selected_c)
        validation_probability = _apply_temperature(_predict_logistic(final_model, validation_frame[feature_columns].to_numpy(dtype=np.float64)), inner_temperature)
        time_x_train = _temporal_features(train_frame)
        time_x_validation = _temporal_features(validation_frame)
        final_time_model, _ = _fit_logistic(train_frame, time_x_train, train_frame["label"].to_numpy(dtype=np.int64), 1.0)
        time_probability = _apply_temperature(_predict_logistic(final_time_model, time_x_validation), inner_time_temperature)
        condition_x_train = _trial_condition_features(train_frame)
        condition_x_validation = _trial_condition_features(validation_frame)
        final_condition_model, _ = _fit_logistic(train_frame, condition_x_train, train_frame["label"].to_numpy(dtype=np.int64), 1.0)
        condition_probability = _apply_temperature(_predict_logistic(final_condition_model, condition_x_validation), inner_condition_temperature)
        validation_y = validation_frame["label"].to_numpy(dtype=np.int64)
        signal_metrics = _metrics(validation_y, validation_probability, validation_frame["subject_id"].astype(str).tolist())
        time_metrics = _metrics(validation_y, time_probability, validation_frame["subject_id"].astype(str).tolist())
        condition_metrics = _metrics(validation_y, condition_probability, validation_frame["subject_id"].astype(str).tolist())
        for model_name, probabilities, metrics in (
            ("signal_logistic", validation_probability, signal_metrics),
            ("time_only_logistic", time_probability, time_metrics),
            ("trial_condition_time_logistic", condition_probability, condition_metrics),
        ):
            group_rows.extend({"outer_fold_id": outer_fold, "model_name": model_name, **row} for row in metrics.pop("per_group"))
            for row_index, (_, record) in enumerate(validation_frame.iterrows()):
                oof_rows.append({
                    "sample_id": record["sample_id"],
                    "subject_id": record["subject_id"],
                    "recording_id": record["recording_id"],
                    "trial_number": int(record["trial_number"]),
                    "relative_time": float(record["relative_time"]),
                    "outer_fold_id": outer_fold,
                    "model_name": model_name,
                    "true_label": int(validation_y[row_index]),
                    "prediction": int(np.argmax(probabilities[row_index])),
                    "probability_class_0": float(probabilities[row_index, 0]),
                    "probability_class_1": float(probabilities[row_index, 1]),
                    "probability_class_2": float(probabilities[row_index, 2]),
                    "inner_oof_selected_C": selected_c if model_name == "signal_logistic" else 1.0,
                    "inner_oof_calibration_temperature": (
                        inner_temperature if model_name == "signal_logistic"
                        else inner_time_temperature if model_name == "time_only_logistic"
                        else inner_condition_temperature
                    ),
                })
            metrics["model_name"] = model_name
            metrics["outer_fold_id"] = outer_fold
            metrics["selected_C"] = selected_c if model_name == "signal_logistic" else 1.0
            metrics["inner_oof_temperature"] = (
                inner_temperature if model_name == "signal_logistic"
                else inner_time_temperature if model_name == "time_only_logistic"
                else inner_condition_temperature
            )
            metrics.pop("per_group", None)
            fold_receipts.append(metrics)
        checkpoint = {
            "model": final_model,
            "feature_columns": feature_columns,
            "class_ids": CLASS_IDS,
            "class_names": CLASS_NAMES,
            "temperature": inner_temperature,
            "selected_C": selected_c,
            "outer_fold_id": outer_fold,
            "training_participants": sorted(train_frame["subject_id"].astype(str).unique()),
            "validation_participants": sorted(validation_frame["subject_id"].astype(str).unique()),
            "label_rule": "cerqueira_v2_self_perceived_fatigue_state_0_1_2",
        }
        joblib.dump(checkpoint, output_dir / f"cerqueira_signal_logistic_outer_fold_{outer_fold}.joblib")
        _write_json(output_dir / f"outer_fold_{outer_fold}_support.json", {
            "training_participants": sorted(train_frame["subject_id"].astype(str).unique()),
            "validation_participants": sorted(validation_frame["subject_id"].astype(str).unique()),
            "training": train_support,
            "validation": validation_support,
        })
        print(f"Completed Cerqueira outer fold {outer_fold}/4; C={selected_c}; T={inner_temperature:.3f}; signal group Macro-F1={signal_metrics['group_macro_f1']:.3f}; time={time_metrics['group_macro_f1']:.3f}", flush=True)

    predictions = pd.DataFrame(oof_rows)
    groups_table = pd.DataFrame(group_rows)
    predictions.to_parquet(output_dir / "cerqueira_nested_oof_predictions.parquet", index=False)
    groups_table.to_csv(output_dir / "cerqueira_group_metrics.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output_dir / "cerqueira_inner_candidate_scores.csv", index=False)
    pd.DataFrame(fold_receipts).to_csv(output_dir / "cerqueira_outer_fold_metrics.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for model_name in ("signal_logistic", "time_only_logistic", "trial_condition_time_logistic"):
        subset = predictions[predictions["model_name"] == model_name]
        all_metrics = _metrics(subset["true_label"].to_numpy(dtype=np.int64), subset[["probability_class_0", "probability_class_1", "probability_class_2"]].to_numpy(dtype=np.float64), subset["subject_id"].astype(str).tolist())
        all_metrics.pop("per_group", None)
        all_metrics["model_name"] = model_name
        summary_rows.append(all_metrics)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "cerqueira_metrics_summary.csv", index=False)
    paired_rows = []
    signal_groups = groups_table[groups_table.model_name == "signal_logistic"]
    control_groups = {
        name: groups_table[groups_table.model_name == name]
        for name in ("time_only_logistic", "trial_condition_time_logistic")
    }
    for baseline_name, baseline_groups in control_groups.items():
        for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
            paired_rows.append({
                "comparison": f"signal_logistic_minus_{baseline_name}",
                **_paired_group_bootstrap(signal_groups, baseline_groups, metric=metric, replicates=bootstrap_replicates, seed=seed + len(paired_rows)),
            })
    pd.DataFrame(paired_rows).to_csv(output_dir / "cerqueira_paired_group_bootstrap.csv", index=False)

    final_c_values = [item["selected_C"] for item in fold_receipts if item["model_name"] == "signal_logistic"]
    final_c = float(pd.Series(final_c_values).median())
    final_model, _ = _fit_logistic(frame, x, labels, final_c)
    outer_oof = predictions[predictions.model_name == "signal_logistic"]
    final_temperature = _fit_temperature(
        outer_oof[["probability_class_0", "probability_class_1", "probability_class_2"]].to_numpy(dtype=np.float64),
        outer_oof["true_label"].to_numpy(dtype=np.int64),
        outer_oof["subject_id"].astype(str).tolist(),
    )
    joblib.dump({
        "model": final_model,
        "feature_columns": feature_columns,
        "class_ids": CLASS_IDS,
        "class_names": CLASS_NAMES,
        "temperature": final_temperature,
        "selected_C_median_outer": final_c,
        "training_participants": sorted(frame["subject_id"].astype(str).unique()),
        "label_rule": "cerqueira_v2_self_perceived_fatigue_state_0_1_2",
        "calibration_source": "outer participant OOF predictions, each participant held out once",
    }, output_dir / "cerqueira_signal_logistic_final_development.joblib")
    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir(), key=lambda value: value.name.casefold())
        if path.is_file()
    }
    manifest = {
        "status": "completed",
        "dataset_key": "cerqueira_v2",
        "target": "researcher-operated self-perceived fatigue state 0/1/2; not physiological ground truth",
        "outer_folds": 4,
        "inner_folds": 3,
        "independent_unit": "participant",
        "development_participants": int(frame["subject_id"].nunique()),
        "training_trials": int(frame["recording_id"].nunique()),
        "sample_count_descriptive_only": int(len(frame)),
        "class_support_full_data_descriptive_only": overall_support,
        "minimum_groups_per_class_per_partition": MIN_GROUPS_PER_CLASS,
        "minimum_trials_per_class_per_partition": MIN_TRIALS_PER_CLASS,
        "calibration": "multiclass temperature scaling with participant and class balanced weights; fit only to inner OOF probabilities per outer fold; final checkpoint calibrated from outer OOF",
        "decision_rule": "multiclass argmax after temperature scaling; no class threshold tuned on held-out groups",
        "models": ["signal_logistic", "time_only_logistic", "trial_condition_time_logistic"],
        "controls": {
            "time_only_logistic": "relative within-trial time only",
            "trial_condition_time_logistic": "relative within-trial time plus one-hot source trial/movement condition",
        },
        "checkpoint_files": [f"cerqueira_signal_logistic_outer_fold_{i}.joblib" for i in range(1, 5)] + ["cerqueira_signal_logistic_final_development.joblib"],
        "artifact_sha256": artifact_hashes,
        "code_provenance": code_provenance(__file__),
        "predictions_file": "cerqueira_nested_oof_predictions.parquet",
        "window_count_is_independent_n": False,
        "locked_course_internal_test_values_parsed": 0,
        "course_raw2_rar_opened": False,
    }
    _write_json(output_dir / "cerqueira_nested_stage_manifest.json", manifest)
    return {"status": "completed", "manifest_path": str((output_dir / "cerqueira_nested_stage_manifest.json").resolve()), "metrics": summary.to_dict(orient="records")}
