"""Run the complete raw-first, label-free sEMG experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy import signal

from .audit import audit_all, choose_sampling_plan
from .augmentation import ALL_FAMILIES, augmentation_recipe
from .contracts import ProcessedSegment, SignalWindow
from .features import extract_features, feature_schema
from .io import raw_first_source_fingerprint, read_training_zip, sha256_file, write_json
from .labels import audit_references
from .plots import make_before_after_plots, make_global_signal_summary, make_sampling_histogram
from .preprocessing import preprocess_recording
from .runtime import software_versions
from .robustness import evaluate_robustness
from .sensitivity import filter_sensitivity, window_sensitivity
from .splits import source_group_split
from .ssl import apply_signal_scale, fit_signal_scale, model_embedding, train_masked_encoder
from .windows import segment_windows


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "EMG_Biomecanica" / "raw-first-fatigue-v2"
    return Path.home() / "AppData" / "Local" / "EMG_Biomecanica" / "raw-first-fatigue-v2"


def _ensure_private_output(output_root: Path) -> None:
    output = output_root.expanduser().resolve()
    repo = _repo_root().resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Restricted run outputs must be stored outside the Git checkout.")
    output.mkdir(parents=True, exist_ok=True)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _git_metadata() -> dict[str, object]:
    repo = _repo_root()
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "branch": run_git(["branch", "--show-current"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "working_tree_status": run_git(["status", "--short"]),
    }


def _write_csv(path: Path, rows: list[dict[str, object]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "min": float("nan"), "median": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _aggregate_audit(
    audits: list[dict[str, object]],
    sampling_plan: dict[str, object],
    inventory: dict[str, object],
    label_audit: dict[str, object],
    window_counts: dict[str, int],
    rejection_counts: dict[str, int],
) -> pd.DataFrame:
    rates = np.asarray([row["effective_sampling_rate_hz"] for row in audits], dtype=np.float64)
    durations = np.asarray([row["duration_s"] for row in audits], dtype=np.float64)
    r2s = np.asarray([row["adc_to_voltage_r2"] for row in audits], dtype=np.float64)
    jitter = np.asarray([row["jitter_mad_percent"] for row in audits], dtype=np.float64)
    rms = np.asarray([row["adc_rms"] for row in audits], dtype=np.float64)
    rows: list[dict[str, object]] = []

    def add(metric: str, value: object, unit: str = "count", detail: str = "") -> None:
        rows.append({"metric": metric, "value": value, "unit": unit, "detail": detail})

    add("training_csv_count", int(inventory["csv_count"]))
    add("source_folder_count", int(inventory["source_group_count"]))
    add("expected_training_archive_name", inventory.get("expected_training_archive_name", "EMG_TRAIN_SOLO_CSV_ANONIMO.zip"), "filename")
    add("supplied_archive_name_matches_objective", bool(inventory.get("supplied_archive_name_matches_objective", False)), "boolean")
    add("source_group_names_anonymized_in_processing", True, "boolean", "Opaque aliases are generated in memory; the source-name map remains private and is not written.")
    add("source_folder_identity_verified", False, "boolean", "Opaque source-group folders do not establish participant identity.")
    add("timestamp_rate_valid_file_count", _finite_stats(rates)["count"])
    add("effective_sampling_rate_hz", json.dumps(_finite_stats(rates)), "Hz", "Distribution across source files; no file-level rows published.")
    add("duration_s", json.dumps(_finite_stats(durations)), "s", "Distribution across source files.")
    add("sampling_interval_jitter_mad_percent", json.dumps(_finite_stats(jitter)), "%", "Robust interval dispersion per source file, aggregated.")
    add("adc_rms", json.dumps(_finite_stats(rms)), "ADC-derived units", "Exported signal scale; no verified physical calibration.")
    add("adc_to_voltage_r2", json.dumps(_finite_stats(r2s)), "R^2", "Linear relationship only; does not verify sensor calibration or ADC representation.")
    add("effective_target_sampling_rate_hz", sampling_plan["target_sampling_rate_hz"], "Hz", str(sampling_plan["target_rate_selection"]))
    add("preprocessing_selection_scope", sampling_plan["selection_scope"], "scope")
    add("preprocessing_selection_group_count", sampling_plan["selection_group_count"])
    add("preprocessing_selection_recording_count", sampling_plan["selection_recording_count"])
    add("selected_bandpass_hz", f"{sampling_plan['bandpass_low_hz']}-{sampling_plan['bandpass_high_hz']}", "Hz")
    add("selected_notch_hz", sampling_plan["notch_frequency_hz"] if sampling_plan["notch_frequency_hz"] is not None else "none", "Hz", str(sampling_plan["notch_selection"]))
    add("reference_search_complete", label_audit["reference_search_complete"], "boolean")
    add("reference_audit_status", label_audit["status"], "status", str(label_audit["rationale"]))
    add("unreviewed_non_signal_csv_count", label_audit["unreviewed_non_signal_csv_count"])
    add("unreviewed_sidecar_count", label_audit["sidecar_count"])
    add("primary_500ms_valid_window_count", window_counts.get("0.5", 0))
    add("primary_500ms_rejected_window_count", rejection_counts.get("0.5", 0))
    for key, count in sorted(window_counts.items()):
        if key == "0.5":
            continue
        add(f"valid_{int(float(key) * 1000)}ms_window_count", count)
    for key, count in sorted(rejection_counts.items()):
        if key == "0.5":
            continue
        add(f"rejected_{int(float(key) * 1000)}ms_window_count", count)
    return pd.DataFrame(rows)


def _feature_frame(windows: list[SignalWindow], willison_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in windows:
        identifier = f"{window.recording_id}_S{window.segment_index:02d}_W{window.window_index:05d}"
        features = extract_features(window, willison_threshold)
        rows.append(
            {
                "window_id": identifier,
                "group_id": window.group_id,
                "recording_id": window.recording_id,
                "segment_index": window.segment_index,
                "window_index": window.window_index,
                "start_sample": window.start_sample,
                "start_time_s": window.start_time_s,
                "duration_s": window.duration_s,
                **features,
            }
        )
    return pd.DataFrame(rows)


def _fit_training_feature_transform(frame: pd.DataFrame, train_groups: list[str], output_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata_cols = {"window_id", "group_id", "recording_id", "segment_index", "window_index", "start_sample", "start_time_s", "duration_s"}
    feature_cols = [column for column in frame.columns if column not in metadata_cols]
    train_mask = frame["group_id"].isin(train_groups).to_numpy()
    if not train_mask.any():
        raise ValueError("No primary windows belong to the development source groups.")
    included: list[str] = []
    details: dict[str, dict[str, object]] = {}
    train_values = frame.loc[train_mask, feature_cols]
    for column in feature_cols:
        values = pd.to_numeric(train_values[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        coverage = float(finite.size / max(1, values.size))
        if coverage < 0.95 or finite.size < 2:
            details[column] = {"included": False, "reason": "less than 95% finite in training groups", "finite_fraction": coverage}
            continue
        median = float(np.median(finite))
        scale = float(np.std(finite))
        if not np.isfinite(scale) or scale <= 1e-12:
            details[column] = {"included": False, "reason": "zero or invalid training-group variance", "finite_fraction": coverage, "median": median}
            continue
        details[column] = {"included": True, "finite_fraction": coverage, "median": median, "training_mean": float(np.mean(finite)), "training_scale": scale}
        included.append(column)
    if not included:
        raise ValueError("No physiological feature has sufficient finite training-group variability.")

    filled = frame[included].copy()
    for column in included:
        filled[column] = pd.to_numeric(filled[column], errors="coerce").fillna(float(details[column]["median"]))
    scaler = StandardScaler().fit(filled.loc[train_mask, included].to_numpy(dtype=np.float64))
    transformed = scaler.transform(filled.to_numpy(dtype=np.float64))
    standardized = frame[["window_id", "group_id", "recording_id", "duration_s"]].copy()
    for index, column in enumerate(included):
        standardized[column] = transformed[:, index]
        details[column]["standardizer_mean"] = float(scaler.mean_[index])
        details[column]["standardizer_scale"] = float(scaler.scale_[index])
    metadata = {
        "fit_groups": sorted(train_groups),
        "fit_scope": "development source groups only; validation source groups transformed with fixed medians and StandardScaler",
        "included_feature_names": included,
        "feature_details": details,
        "excluded_feature_count": len(feature_cols) - len(included),
    }
    write_json(output_dir / "feature_standardization.json", metadata)
    return standardized, metadata


def _safe_aggregate_feature_summary(frame: pd.DataFrame, schema: list[dict[str, str]], output_path: Path) -> None:
    names = [item["name"] for item in schema if item["name"] in frame.columns]
    rows: list[dict[str, object]] = []
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "feature": name,
                "finite_window_count": int(finite.size),
                "finite_fraction": float(finite.size / max(1, values.size)),
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite)) if finite.size else float("nan"),
            }
        )
    _write_csv(output_path, rows)


def _save_private_embeddings(models: dict[str, Any], windows: list[SignalWindow], signal_scale: dict[str, float], output_path: Path) -> None:
    signals = np.stack([window.signal for window in windows]).astype(np.float32)
    normalized = apply_signal_scale(signals, signal_scale)
    data: dict[str, np.ndarray] = {
        "group_id": np.asarray([window.group_id for window in windows]),
        "recording_id": np.asarray([window.recording_id for window in windows]),
        "start_time_s": np.asarray([window.start_time_s for window in windows], dtype=np.float64),
    }
    for name, model in models.items():
        data[f"embedding_{name}"] = model_embedding(model, normalized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)


def _hash_outputs(run_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            hashes[str(path.relative_to(run_root)).replace("\\", "/")] = sha256_file(path)
    return hashes


def _filter_coefficients(sampling_plan: dict[str, object], order: int) -> dict[str, object]:
    fs = float(sampling_plan["target_sampling_rate_hz"])
    low = float(sampling_plan["bandpass_low_hz"])
    high = float(sampling_plan["bandpass_high_hz"])
    bandpass = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    notch = None
    notch_hz = sampling_plan.get("notch_frequency_hz")
    if notch_hz is not None and float(notch_hz) < fs / 2:
        b, a = signal.iirnotch(float(notch_hz), 30.0, fs=fs)
        notch = signal.tf2sos(b, a)
    return {
        "target_sampling_rate_hz": fs,
        "butterworth_bandpass_sos": bandpass.tolist(),
        "notch_sos": notch.tolist() if notch is not None else None,
    }


def _markdown_summary(
    inventory: dict[str, object],
    audits: list[dict[str, object]],
    sampling_plan: dict[str, object],
    label_audit: dict[str, object],
    split: dict[str, object],
    ssl_metadata: dict[str, dict[str, object]],
    robust_frame: pd.DataFrame,
    window_counts: dict[str, int],
    rejection_counts: dict[str, int],
) -> str:
    rates = _finite_stats(np.asarray([row["effective_sampling_rate_hz"] for row in audits], dtype=float))
    corruption_rows = robust_frame[robust_frame["corruption_family"] != "clean"]
    if len(corruption_rows):
        by_model = {
            condition: corruption_rows[corruption_rows["model_condition"] == condition]
            for condition in ("clean", "augmented")
        }
        cosine_at_severity = {
            (condition, severity): float(
                frame.loc[frame["requested_severity"] == severity, "median_embedding_cosine_similarity"].median()
            )
            for condition, frame in by_model.items()
            for severity in (0.20, 0.50)
        }
        worst_rows = {
            condition: frame[frame["requested_severity"] == 0.50]
            .sort_values("median_embedding_cosine_similarity")
            .iloc[0]
            for condition, frame in by_model.items()
        }
        worst_text = (
            f"Median across family-level embedding cosine summaries at 20% severity: clean {cosine_at_severity[('clean', 0.20)]:.4f}, "
            f"augmentation-trained {cosine_at_severity[('augmented', 0.20)]:.4f}; at 50% stress: "
            f"{cosine_at_severity[('clean', 0.50)]:.4f} and {cosine_at_severity[('augmented', 0.50)]:.4f}. "
            f"Worst 50% family: clean {worst_rows['clean']['corruption_family']} at "
            f"{worst_rows['clean']['median_embedding_cosine_similarity']:.3f}; augmentation-trained "
            f"{worst_rows['augmented']['corruption_family']} at {worst_rows['augmented']['median_embedding_cosine_similarity']:.3f}. "
            "These measure representation stability, not classification accuracy."
        )
    else:
        worst_text = "No robustness summary available."
    expected_archive_name = str(inventory.get("expected_training_archive_name", "EMG_TRAIN_SOLO_CSV_ANONIMO.zip"))
    if bool(inventory.get("supplied_archive_name_matches_objective", False)):
        archive_name_text = f"The supplied training archive filename matches expected `{expected_archive_name}`."
    else:
        archive_name_text = f"The supplied training archive filename differs from expected `{expected_archive_name}`; the actual name is retained only in the restricted manifest."
    notch_hz = sampling_plan["notch_frequency_hz"]
    if notch_hz is not None:
        notch_text = f"notch at {float(notch_hz):.1f} Hz; {sampling_plan['notch_selection']}"
    else:
        notch_text = "no notch applied because no line frequency met the prespecified prevalence threshold"
    return "\n".join(
        [
            "# Raw First sEMG Experiment Summary",
            "",
            "## Input audit",
            "",
            "- Training input: the supplied ZIP archive; its exact filename, SHA-256, and original-file hashes are recorded in the restricted local run manifest.",
            f"- Parsed {inventory['csv_count']} CSV files in {inventory['source_group_count']} opaque source groups. Source-group identity is not verified participant identity.",
            f"- {archive_name_text} Source folders were assigned opaque aliases in memory.",
            f"- Effective sampling rate across valid files: median {rates['median']:.3f} Hz, range {rates['min']:.3f}–{rates['max']:.3f} Hz.",
            f"- Sampling-rate, passband, and notch choices were derived from {sampling_plan['selection_group_count']} development source groups only ({sampling_plan['selection_recording_count']} recordings); held-out groups were excluded from selection.",
            f"- Chosen common rate: {sampling_plan['target_sampling_rate_hz']:.3f} Hz. {sampling_plan['target_rate_selection']}",
            f"- Filter: {sampling_plan['bandpass_low_hz']:.1f}–{sampling_plan['bandpass_high_hz']:.1f} Hz; {notch_text}.",
            f"- Valid windows: 500 ms {window_counts.get('0.5', 0):,}, 250 ms {window_counts.get('0.25', 0):,}, 1000 ms {window_counts.get('1.0', 0):,}. Primary 500 ms rejected: {rejection_counts.get('0.5', 0):,}.",
            "- `ADC_Value` is treated as an exported signed signal scale. The paired voltage column is analyzed for numerical relationship only; sensor calibration and raw ADC hardware representation are not verified.",
            "",
            "## Labels and supervised scope",
            "",
            f"- Status: **{label_audit['status']}**. {label_audit['rationale']}",
            "- Missing independent fatigue references: Borg CR-10, pre/post MVC, force/torque, repetition-level performance, or a mechanical failure criterion.",
            "- Missing independent phase references: synchronized extension/flexion or knee-angle/video annotations.",
            "- No supervised fatigue/phase model or classification metric (accuracy, macro-F1, sensitivity, or confusion matrix) was produced.",
            "",
            "## Self-supervised training",
            "",
            f"- Source-group split: {split['train_group_count']} development groups and {split['validation_group_count']} held-out groups. These are not verified participants.",
            "- Model: compact 1D CNN trained from scratch with masked reconstruction; clean and augmentation-trained versions use the same training/validation source groups.",
            *[
                f"- {name}: best epoch {meta['best_epoch_by_validation_reconstruction']}; validation masked reconstruction loss {meta['best_validation_masked_reconstruction_loss']:.6g}; checkpoint SHA-256 `{meta['checkpoint_sha256']}`."
                for name, meta in ssl_metadata.items()
            ],
            "",
            "## Noise robustness",
            "",
            f"- {worst_text}",
            "- All robustness numbers are feature, embedding, and reconstruction stability summaries under fixed matched corruptions; they do not establish a fatigue classifier's robustness.",
            "- Filtering sensitivity is summarized on held-out source groups using a lower high-cut/no-notch alternative; window-size sensitivity is summarized at 250, 500, and 1000 ms.",
            "- Detailed feature-family × severity × encoder tables, training histories, audit tables, waveforms, spectra, per-file hashes, and embeddings are stored in the restricted local run directory.",
            "",
            "## External RAR",
            "",
            "- Reserved for the final post-freeze check. Do not treat an unlabeled or identity-unverified RAR as independent external fatigue validation.",
        ]
    ) + "\n"


def run_training(training_zip: Path, output_root: Path | None = None, config_path: Path | None = None) -> Path:
    training_zip = training_zip.expanduser().resolve()
    output_root = (output_root or _default_output_root()).expanduser().resolve()
    config_path = config_path or (_repo_root() / "configs" / "raw_first.toml")
    _ensure_private_output(output_root)
    config = _load_config(config_path)
    run_config = config["run"]
    ssl_config = config["ssl"]
    robustness_config = config["robustness"]
    run_seed = int(run_config["seed"])
    input_digest = sha256_file(training_zip)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{input_digest[:10]}"
    run_root = output_root / run_id
    private = run_root / "private"
    private.mkdir(parents=True, exist_ok=False)

    recordings, inventory = read_training_zip(training_zip)
    expected_archive_name = str(run_config.get("expected_training_archive_name", "EMG_TRAIN_SOLO_CSV_ANONIMO.zip"))
    inventory["expected_training_archive_name"] = expected_archive_name
    inventory["supplied_archive_name_matches_objective"] = training_zip.name.casefold() == expected_archive_name.casefold()
    if int(inventory["csv_count"]) != int(run_config["expected_training_csv_count"]):
        raise ValueError("Training CSV count differs from the configured input contract.")
    if int(inventory["source_group_count"]) != int(run_config["expected_source_group_count"]):
        raise ValueError("Source-group count differs from the configured input contract.")
    write_json(private / "input_manifest.json", inventory)

    # Partition first so data-dependent sampling and filter choices see only
    # development-group acquisition audits. Validation audits stay descriptive.
    group_ids = sorted({recording.group_id for recording in recordings})
    split = source_group_split(group_ids, run_seed, float(run_config["train_fraction"]))
    write_json(private / "group_partition.json", split)
    development_groups = set(split["train_group_ids"])

    audits = audit_all(recordings)
    audit_frame = pd.DataFrame(audits)
    _write_csv(private / "acquisition_audit.csv", audit_frame)
    development_audits = [row for row in audits if row["group_id"] in development_groups]
    if not development_audits:
        raise ValueError("No development-group acquisition audits are available for preprocessing selection.")
    sampling_plan = choose_sampling_plan(development_audits)
    sampling_plan["selection_scope"] = "development source groups only; held-out groups excluded"
    sampling_plan["selection_group_count"] = len(development_groups)
    sampling_plan["selection_recording_count"] = len(development_audits)
    write_json(private / "sampling_plan.json", sampling_plan)
    make_sampling_histogram(audits, private / "plots" / "sampling_rate_distribution.png")
    make_global_signal_summary(audits, private / "plots" / "global_signal_spectral_summary.png")

    audit_by_recording = {str(row["recording_id"]): row for row in audits}
    segments: list[ProcessedSegment] = []
    segment_receipts: list[dict[str, object]] = []
    for recording in recordings:
        processed, receipt = preprocess_recording(
            recording,
            audit_by_recording[recording.recording_id],
            sampling_plan,
            filter_order=int(run_config["filter_order"]),
        )
        segments.extend(processed)
        segment_receipts.append({"group_id": recording.group_id, "recording_id": recording.recording_id, **receipt})
    write_json(
        private / "preprocessing_config.json",
        {
            **sampling_plan,
            "filter_order": int(run_config["filter_order"]),
            "filter_type": "Butterworth bandpass in SOS form, zero-phase SOS filtering by continuous segment",
            "detrending": "linear detrend per continuous segment before filtering",
            "resampling": "scipy.signal.resample_poly with Kaiser 5.0 antialiasing window; only when the measured rate differs from target",
            "amplitude_normalization": "none per recording; SSL global mean/std fitted on development-group windows only",
            "notch_q": 30.0,
            "window_qc": {
                "reject_nonfinite_or_empty": True,
                "reject_zero_energy": True,
                "reject_constant_run_ms": 50,
                "reject_single_sample_transient_when_crest_factor_exceeds": 40.0,
                "transient_single_sample_energy_share_exceeds": 0.25,
            },
        },
    )
    write_json(private / "filter_coefficients.json", _filter_coefficients(sampling_plan, int(run_config["filter_order"])))
    write_json(private / "segment_processing_receipts.json", segment_receipts)
    make_before_after_plots(recordings, segments, private / "plots" / "before_after", audit_by_recording, sampling_plan)

    label_audit = audit_references(inventory).to_dict()
    write_json(private / "label_audit.json", label_audit)

    durations = tuple(float(value) for value in run_config["window_durations_s"])
    overlap = float(run_config["window_overlap"])
    windows_by_duration: dict[str, list[SignalWindow]] = {}
    rejection_by_duration: dict[str, list[dict[str, object]]] = {}
    for duration in durations:
        key = str(duration)
        windows, rejected = segment_windows(segments, duration, overlap)
        windows_by_duration[key] = windows
        rejection_by_duration[key] = rejected
    window_counts = {key: len(value) for key, value in windows_by_duration.items()}
    rejection_counts = {key: len(value) for key, value in rejection_by_duration.items()}
    for key, windows in windows_by_duration.items():
        duration_ms = int(round(float(key) * 1000))
        _write_csv(
            private / f"window_manifest_{duration_ms}ms.csv",
            [
                {
                    "group_id": window.group_id,
                    "recording_id": window.recording_id,
                    "segment_index": window.segment_index,
                    "window_index": window.window_index,
                    "start_sample": window.start_sample,
                    "start_time_s": window.start_time_s,
                    "duration_s": window.duration_s,
                    "sampling_rate_hz": window.sampling_rate_hz,
                }
                for window in windows
            ],
        )
        _write_csv(private / f"rejected_windows_{duration_ms}ms.csv", rejection_by_duration[key])

    primary_key = str(float(run_config["primary_window_s"]))
    primary_windows = windows_by_duration[primary_key]
    train_groups = list(split["train_group_ids"])
    validation_groups = list(split["validation_group_ids"])
    train_windows = [window for window in primary_windows if window.group_id in train_groups]
    validation_windows = [window for window in primary_windows if window.group_id in validation_groups]
    if not train_windows or not validation_windows:
        raise ValueError("The source-group split did not yield both development and held-out windows.")

    train_rms = np.asarray([np.sqrt(np.mean(window.signal.astype(np.float64) ** 2)) for window in train_windows])
    willison_threshold = 0.01 * float(np.median(train_rms))
    schema = feature_schema()
    write_json(
        private / "feature_schema.json",
        {
            "features": schema,
            "willison_threshold_adc_derived_units": willison_threshold,
            "willison_threshold_source": "1% of median RMS across development-group 500 ms windows only",
            "sampling_rate_hz": sampling_plan["target_sampling_rate_hz"],
            "primary_window_s": float(run_config["primary_window_s"]),
            "overlap": overlap,
            "feature_importance": "not calculated; there are no independent fatigue labels",
        },
    )

    feature_frames: dict[str, pd.DataFrame] = {}
    for key, windows in windows_by_duration.items():
        frame = _feature_frame(windows, willison_threshold)
        feature_frames[key] = frame
        duration_ms = int(round(float(key) * 1000))
        _write_csv(private / f"features_{duration_ms}ms.csv", frame)

    primary_features = feature_frames[primary_key]
    standardized_features, feature_transform = _fit_training_feature_transform(primary_features, train_groups, private)
    _write_csv(private / "features_standardized_500ms.csv", standardized_features)
    feature_columns = feature_transform["included_feature_names"]
    train_feature_frame = standardized_features[standardized_features["group_id"].isin(train_groups)]
    correlation = train_feature_frame[feature_columns].corr(method="spearman", numeric_only=True)
    correlation.to_csv(private / "feature_correlations_train_only.csv")
    window_sensitivity_frame = window_sensitivity(feature_frames)
    _write_csv(private / "feature_window_sensitivity.csv", window_sensitivity_frame)
    filter_sensitivity_details, filter_sensitivity_summary = filter_sensitivity(
        recordings,
        audit_by_recording,
        sampling_plan,
        primary_windows,
        validation_groups,
        willison_threshold,
        int(run_config["filter_order"]),
        overlap,
    )
    _write_csv(private / "filter_sensitivity_by_feature.csv", filter_sensitivity_details)
    _write_csv(private / "filter_sensitivity_summary.csv", filter_sensitivity_summary)

    signal_scale = fit_signal_scale(np.stack([window.signal for window in train_windows]))
    signal_scale["fit_group_count"] = len(train_groups)
    signal_scale["fit_window_count"] = len(train_windows)
    write_json(private / "signal_standardization.json", signal_scale)
    train_signals = apply_signal_scale(np.stack([window.signal for window in train_windows]), signal_scale)
    validation_signals = apply_signal_scale(np.stack([window.signal for window in validation_windows]), signal_scale)

    checkpoint_dir = private / "models"
    ssl_models: dict[str, Any] = {}
    ssl_histories: dict[str, list[dict[str, float]]] = {}
    ssl_metadata: dict[str, dict[str, object]] = {}
    levels = tuple(float(value) for value in ssl_config["augmentation_levels"])
    for condition, seed_offset in (("clean", 0), ("augmented", 1)):
        model, history, metadata = train_masked_encoder(
            train_signals,
            validation_signals,
            float(sampling_plan["target_sampling_rate_hz"]),
            checkpoint_dir,
            condition,
            run_seed + seed_offset,
            float(sampling_plan["notch_frequency_hz"]) if sampling_plan["notch_frequency_hz"] is not None else None,
            levels=levels,
            epochs=int(ssl_config["epochs"]),
            batch_size=int(ssl_config["batch_size"]),
            learning_rate=float(ssl_config["learning_rate"]),
            mask_fraction=float(ssl_config["mask_fraction"]),
            mask_span_ms=float(ssl_config["mask_span_ms"]),
        )
        ssl_models[condition] = model
        ssl_histories[condition] = history
        ssl_metadata[condition] = metadata

    private_summary = _aggregate_audit(audits, sampling_plan, inventory, label_audit, window_counts, rejection_counts)
    _write_csv(private / "aggregate_acquisition_summary.csv", private_summary)
    _safe_aggregate_feature_summary(primary_features, schema, private / "aggregate_feature_summary.csv")
    _save_private_embeddings(ssl_models, validation_windows, signal_scale, private / "validation_embeddings.npz")

    robustness_recipe = augmentation_recipe(
        int(robustness_config["seed"]), levels, float(sampling_plan["notch_frequency_hz"]) if sampling_plan["notch_frequency_hz"] is not None else None
    )
    write_json(private / "augmentation_config.json", {**robustness_recipe, "robustness_levels": list(robustness_config["levels"])})
    robust_frame, reproducibility = evaluate_robustness(
        validation_windows,
        ssl_models,
        signal_scale,
        willison_threshold,
        private / "robustness",
        int(robustness_config["seed"]),
        float(sampling_plan["notch_frequency_hz"]) if sampling_plan["notch_frequency_hz"] is not None else None,
        levels=tuple(float(value) for value in robustness_config["levels"]),
        maximum_windows=int(robustness_config["maximum_windows"]),
    )
    write_json(private / "inference_reproducibility.json", reproducibility)

    public_dir = _repo_root() / "results" / "raw_first"
    public_dir.mkdir(parents=True, exist_ok=True)
    aggregate_frame = _aggregate_audit(audits, sampling_plan, inventory, label_audit, window_counts, rejection_counts)
    _write_csv(public_dir / "aggregate_audit.csv", aggregate_frame)
    _write_csv(public_dir / "feature_robustness.csv", robust_frame)
    _write_csv(public_dir / "feature_window_sensitivity.csv", window_sensitivity_frame)
    _write_csv(public_dir / "filter_sensitivity_summary.csv", filter_sensitivity_summary)
    _write_csv(
        public_dir / "ssl_training_summary.csv",
        [
            {
                "condition": name,
                "best_epoch_by_validation_reconstruction": metadata["best_epoch_by_validation_reconstruction"],
                "best_validation_masked_reconstruction_loss": metadata["best_validation_masked_reconstruction_loss"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "training_window_count": metadata["training_window_count"],
                "validation_window_count": metadata["validation_window_count"],
            }
            for name, metadata in ssl_metadata.items()
        ],
    )
    _safe_aggregate_feature_summary(primary_features, schema, public_dir / "feature_summary.csv")
    (public_dir / "EXECUTION_SUMMARY.md").write_text(
        _markdown_summary(inventory, audits, sampling_plan, label_audit, split, ssl_metadata, robust_frame, window_counts, rejection_counts),
        encoding="utf-8",
    )
    make_sampling_histogram(audits, public_dir / "sampling_rate_distribution.png")
    make_global_signal_summary(audits, public_dir / "global_signal_spectral_summary.png")

    config_snapshot = {
        "input_archive_sha256": input_digest,
        "config": config,
        "sampling_plan": sampling_plan,
        "preprocessing": {
            "filter_order": int(run_config["filter_order"]),
            "filter_type": "Butterworth bandpass SOS; zero-phase per continuous segment",
            "detrend": "linear per segment",
            "resampling": "polyphase Kaiser 5.0 only when needed",
        },
        "features": {"schema_count": len(schema), "willison_threshold_adc_derived_units": willison_threshold},
        "split": split,
        "ssl_models": ssl_metadata,
        "augmentation": robustness_recipe,
        "label_audit": label_audit,
    }
    write_json(private / "frozen_configuration.json", config_snapshot)
    runtime_versions = software_versions()
    source_fingerprint = raw_first_source_fingerprint()
    signal_standardization_hash = sha256_file(private / "signal_standardization.json")
    source_commit = _git_metadata().get("commit")
    write_json(
        private / "freeze.json",
        {
            "status": "frozen_training_configuration_and_self_supervised_encoders",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_archive_sha256": input_digest,
            "configuration_sha256": hashlib.sha256(json.dumps(config_snapshot, sort_keys=True).encode()).hexdigest(),
            "encoder_checkpoint_sha256": {name: metadata["checkpoint_sha256"] for name, metadata in ssl_metadata.items()},
            "signal_standardization_sha256": signal_standardization_hash,
            "external_stage_code_sha256": source_fingerprint["sha256"],
            "external_stage_code_files": source_fingerprint["files"],
            "external_stage_code_commit": source_commit,
            "software_versions": runtime_versions,
            "external_rar_opened": False,
            "supervised_fatigue_model": False,
        },
    )

    # Record checksums only after all restricted and aggregate outputs have been written.
    output_hashes = _hash_outputs(run_root)
    for path in sorted(public_dir.rglob("*")):
        if path.is_file():
            output_hashes[f"public/{path.relative_to(public_dir).as_posix()}"] = sha256_file(path)
    write_json(
        private / "run_manifest.json",
        {
            "run_id": run_id,
            "input_archive_sha256": input_digest,
            "input_original_file_hashes": inventory["file_hashes"],
            "output_sha256": output_hashes,
            "software_versions": runtime_versions,
            "git": _git_metadata(),
            "signal_standardization_sha256": signal_standardization_hash,
            "external_stage_code_sha256": source_fingerprint["sha256"],
            "external_stage_code_files": source_fingerprint["files"],
            "external_stage_code_commit": source_commit,
            "seed": run_seed,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "external_rar_opened": False,
            "supervised_metrics": "not produced; no independent synchronized labels",
            "public_aggregate_directory": str(public_dir),
        },
    )
    return run_root


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Execute the raw-first sEMG audit and label-free SSL pipeline.")
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--training-zip", type=Path, help="Authorized training ZIP; must not be the reserved RAR.")
    stages.add_argument("--finalize-external-archive", type=Path, help="Reserved RAR; accepted only after the selected run has a freeze receipt.")
    parser.add_argument("--run-dir", type=Path, help="Completed private training run directory required for the final external stage.")
    parser.add_argument("--output-root", type=Path, default=_default_output_root(), help="Private output directory outside the repository.")
    parser.add_argument("--config", type=Path, default=_repo_root() / "configs" / "raw_first.toml")
    args = parser.parse_args(argv)
    if args.finalize_external_archive:
        if args.run_dir is None:
            parser.error("--run-dir is required with --finalize-external-archive.")
        from .external import finalize_external

        result = finalize_external(args.finalize_external_archive, args.run_dir)
        print(f"External stage status: {result['status']}")
        return 0
    run_root = run_training(args.training_zip, args.output_root, args.config)
    print(f"Private run completed: {run_root}")
    print(f"Aggregate evidence: {_repo_root() / 'results' / 'raw_first'}")
    return 0
