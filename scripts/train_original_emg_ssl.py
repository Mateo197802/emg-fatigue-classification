"""Train a private, auditable masked-reconstruction encoder on original EMG.

The script intentionally does not train a fatigue classifier: the source
archives contain no independent fatigue labels or synchronized references.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import signal
from scipy.integrate import trapezoid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from emg_fatigue.contracts import StudyConfig, implementation_provenance, sha256_file, write_json
from emg_fatigue.features import FEATURE_COLUMNS_V2, window_features_v2
from emg_fatigue.io import load_recordings
from emg_fatigue.original_emg_ssl import (
    MaskedConvAutoencoder,
    extract_fixed_windows,
    masked_reconstruction_loss,
    random_block_mask,
    split_groups,
)
from emg_fatigue.preprocessing import preprocess


WINDOW_SAMPLES = 500
HOP_SAMPLES = 250
MASK_BLOCK_SAMPLES = 25
MASK_FRACTION = 0.25
PREANALYSIS_FEATURE_COLUMNS = (*FEATURE_COLUMNS_V2, "spectral_power_20_400")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy_verified(source: Path, destination: Path) -> dict:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_file(source)
    if destination.exists():
        observed = sha256_file(destination)
        if observed != expected:
            raise ValueError(f"existing source copy has a different SHA-256: {destination.name}")
    else:
        shutil.copy2(source, destination)
        observed = sha256_file(destination)
    if observed != expected:
        raise ValueError(f"copied source SHA-256 mismatch: {destination.name}")
    return {"filename": destination.name, "sha256": observed, "bytes": destination.stat().st_size}


def _source_recordings(path: Path, source_label: str, cfg: StudyConfig):
    recordings = load_recordings(path, cfg)
    if not recordings:
        raise ValueError(f"no recordings loaded from {source_label}")
    rows = []
    qc_rows = []
    line_rows = []
    for recording in recordings:
        processed = preprocess(recording, cfg)
        n_recording_windows = 0
        valid_runs = 0
        invalid_runs = 0
        for run_number, qc in enumerate(processed["qc"]):
            if qc["status"] != "valid":
                invalid_runs += 1
            else:
                valid_runs += 1
            qc_rows.append({
                "source": source_label,
                "recording_id": recording.recording_id,
                "group_id": recording.subject_id,
                "side": recording.side,
                "series": recording.series,
                "run_index": run_number,
                "start_sample": qc["start"],
                "end_sample_exclusive": qc["end"],
                "samples": qc["end"] - qc["start"],
                "sampling_rate_hz": qc["fs_hz"],
                "jitter_relative": qc["jitter_relative"],
                "qc_status": qc["status"],
                "mean_count": qc["mean_count"],
                "std_count": qc["std_count"],
                "rms_count": qc["rms_count"],
                "peak_to_peak_count": qc["peak_to_peak_count"],
                "extreme_repeat_fraction": qc["extreme_repeat_fraction"],
                "clipping_status": qc["clipping_status"],
            })
        for run in processed["runs"]:
            n = len(run["x"])
            starts = np.arange(0, max(0, n - WINDOW_SAMPLES + 1), HOP_SAMPLES, dtype=np.int64)
            # Keep only windows fully within the edge-valid mask and continuous time.
            candidate_windows, local_starts = extract_fixed_windows(
                run["t"],
                run["x"],
                window_samples=WINDOW_SAMPLES,
                hop_samples=HOP_SAMPLES,
                max_gap_s=max(cfg.gap_min_s, cfg.gap_multiplier / run["fs_hz"]),
                valid_mask=run["valid"],
            )
            design = run["filter_design"]
            notch_scores = design["notch"]["prominence_db_by_candidate_hz"]
            line_rows.append({
                "source": source_label,
                "recording_id": recording.recording_id,
                "group_id": recording.subject_id,
                "run_index": run["run_id"],
                "sampling_rate_hz": run["fs_hz"],
                "prominence_50_hz_db": notch_scores.get("50"),
                "prominence_60_hz_db": notch_scores.get("60"),
                "notch_applied": bool(design["notch"]["applied"]),
            })
            for window_index, (window, start) in enumerate(zip(candidate_windows, local_starts)):
                start_time = float(run["t"][start])
                end_time = float(run["t"][start + WINDOW_SAMPLES - 1])
                rows.append({
                    "signal": window,
                    "source": source_label,
                    "recording_id": recording.recording_id,
                    "group_id": recording.subject_id,
                    "side": recording.side,
                    "series": recording.series,
                    "run_index": run["run_id"],
                    "window_index_in_run": window_index,
                    "start_sample_in_run": int(start),
                    "start_time_s": start_time,
                    "end_time_s": end_time,
                    "duration_s": end_time - start_time,
                    "sampling_rate_hz": run["fs_hz"],
                    "window_id": f"{source_label}_{recording.recording_id}_R{run['run_id']:02d}_W{int(start):07d}",
                })
            n_recording_windows += len(candidate_windows)
        for row in qc_rows:
            if row["recording_id"] == recording.recording_id and row["source"] == source_label:
                row["valid_qc_runs"] = valid_runs
                row["nonvalid_qc_runs"] = invalid_runs
                row["recording_windows"] = n_recording_windows
    return recordings, rows, qc_rows, line_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_training_curve(history: pd.DataFrame, path: Path) -> None:
    """Render train/held-out masked MSE and its zero-prediction baseline."""
    required = {"epoch", "train_masked_mse", "validation_masked_mse", "validation_zero_baseline_mse"}
    if not required.issubset(history.columns) or history.empty:
        raise ValueError("training history is missing the fields required for the reconstruction curve")
    values = history.sort_values("epoch")
    best = values.loc[values["validation_masked_mse"].idxmin()]
    baseline = float(values["validation_zero_baseline_mse"].iloc[-1])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    axis.plot(values["epoch"], values["train_masked_mse"], color="#2563A6", linewidth=2.0, marker="o", markersize=3.5, label="Training")
    axis.plot(values["epoch"], values["validation_masked_mse"], color="#D18A00", linewidth=2.0, marker="o", markersize=3.5, label="Held-out validation")
    axis.axhline(baseline, color="#5B6573", linewidth=1.5, linestyle="--", label="Zero-prediction baseline")
    axis.scatter([float(best["epoch"])], [float(best["validation_masked_mse"])], color="#D18A00", edgecolor="white", linewidth=0.8, s=58, zorder=4)
    axis.annotate(
        f"Best validation: {float(best['validation_masked_mse']):.4f} (epoch {int(best['epoch'])})",
        xy=(float(best["epoch"]), float(best["validation_masked_mse"])),
        xytext=(8, 10), textcoords="offset points", fontsize=8.5, color="#303846",
    )
    axis.set_title("Original sEMG masked reconstruction", loc="left", fontsize=12, weight="bold", color="#233042", pad=12)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Masked-sample MSE (standardized signal²)")
    axis.set_xlim(float(values["epoch"].min()), float(values["epoch"].max()))
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#DCE2E8", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8792A2")
    axis.legend(frameon=False, ncol=1, loc="upper right", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(destination, format="png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _unique_group_count(recordings) -> int:
    """Count participant-like filename groups from loader Recording objects."""
    return len({recording.subject_id for recording in recordings})


def fit_train_only_feature_standardizer(frame: pd.DataFrame, feature_columns) -> tuple[pd.DataFrame, dict]:
    """Fit feature means/scales on course ZIP training windows only."""
    fit_mask = (
        frame["source"].eq("course_zip")
        & frame["split"].eq("train")
        & frame["feature_status"].eq("valid")
    )
    training = frame.loc[fit_mask, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    if training.empty or not np.isfinite(training.to_numpy(dtype=float)).all():
        raise ValueError("training rows must contain finite values for every preanalysis feature")
    means = training.mean(axis=0)
    scales = training.std(axis=0, ddof=0)
    scales = scales.where(scales > 1e-12, 1.0)
    standardized = frame.copy()
    for column in feature_columns:
        values = pd.to_numeric(standardized[column], errors="coerce")
        standardized[f"{column}_z"] = (values - means[column]) / scales[column]
    return standardized, {
        "fit_source": "course_zip train rows only",
        "fit_window_count": int(fit_mask.sum()),
        "fit_group_count": int(frame.loc[fit_mask, "group_id"].nunique()),
        "method": "population z-score: z=(feature - training_mean)/training_population_std; constant training features use scale 1",
        "mean": {column: float(means[column]) for column in feature_columns},
        "scale": {column: float(scales[column]) for column in feature_columns},
    }


def spectral_power_20_400(values, fs_hz: float) -> float:
    """Integrate one-sided Hann-periodogram density within 20-400 Hz."""
    values = np.asarray(values, dtype=float)
    fs_hz = float(fs_hz)
    if values.ndim != 1 or values.size < 8 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite one-dimensional vector with at least eight samples")
    if not np.isfinite(fs_hz) or fs_hz <= 800.0:
        raise ValueError("sampling rate must exceed 800 Hz for the 20-400 Hz analysis band")
    frequencies, density = signal.periodogram(
        values - values.mean(),
        fs=fs_hz,
        window="hann",
        detrend=False,
        scaling="density",
    )
    in_band = (frequencies >= 20.0) & (frequencies <= 400.0)
    if np.count_nonzero(in_band) < 2:
        raise ValueError("periodogram resolution is insufficient to integrate 20-400 Hz power")
    return float(trapezoid(density[in_band], frequencies[in_band]))


def build_preanalysis_feature_frames(course_rows, raw_rows, train_groups):
    """Extract descriptive features without assigning fatigue or phase labels."""
    train_groups = set(train_groups)
    feature_rows = []
    for source, rows in (("course_zip", course_rows), ("raw_rar", raw_rows)):
        for row in rows:
            split = (
                "train" if source == "course_zip" and row["group_id"] in train_groups
                else "validation" if source == "course_zip"
                else "source_holdout_descriptive"
            )
            output_row = {key: value for key, value in row.items() if key != "signal"}
            output_row["split"] = split
            try:
                output_row.update(window_features_v2(row["signal"], row["sampling_rate_hz"]))
                output_row["spectral_power_20_400"] = spectral_power_20_400(
                    row["signal"], row["sampling_rate_hz"]
                )
                output_row["feature_status"] = "valid"
            except ValueError as error:
                output_row.update({column: np.nan for column in PREANALYSIS_FEATURE_COLUMNS})
                output_row["feature_status"] = "invalid_feature_window"
                output_row["feature_error"] = str(error)
            feature_rows.append(output_row)
    raw_frame = pd.DataFrame(feature_rows)
    standardized, scaler = fit_train_only_feature_standardizer(raw_frame, PREANALYSIS_FEATURE_COLUMNS)
    scaler.update({
        "feature_schema": "emg_window_features_v2_sample_span_plus_band_power_v1",
        "feature_columns": list(PREANALYSIS_FEATURE_COLUMNS),
        "validation_and_raw_rar_rows_used_to_fit_scaler": False,
    })
    schema = {
        "schema": "course_emg_preanalysis_features_v2",
        "feature_schema": "emg_window_features_v2_sample_span_plus_band_power_v1",
        "feature_columns": list(PREANALYSIS_FEATURE_COLUMNS),
        "definitions": {
            "rms": "sqrt((1/N) * sum(x[n]^2))",
            "mav": "(1/N) * sum(abs(x[n]))",
            "waveform_length_per_s_sample_span": "sum(abs(x[n]-x[n-1])) / ((N-1)/fs)",
            "mdf_hz": "lowest f_m with cumulative Hann-periodogram power in 20-400 Hz reaching 50 percent of total in-band power",
            "mnf_hz": "sum(f_k * Pxx[k]) / sum(Pxx[k]) over 20-400 Hz",
            "power_ratio_low_high": "sum(Pxx, 20<=f<60) / sum(Pxx, 60<=f<=400)",
            "spectral_power_20_400": "integral of one-sided Hann periodogram density over 20-400 Hz; units ADC-counts^2",
            "wavelet_A3_D3_D2_D1": "relative squared-coefficient energy from db4 discrete wavelet transform, level 3, symmetric extension",
        },
        "feature_signal": "filtered centered ADC counts; feature formulas center each window",
        "periodogram": "one-sided Hann periodogram; detrend=False; density scaling",
        "analysis_band_hz": [20.0, 400.0],
        "standardization": scaler,
        "usage": "descriptive preanalysis only; not input to masked-reconstruction encoder and no fatigue labels assigned",
        "status_counts": {str(key): int(value) for key, value in raw_frame["feature_status"].value_counts().items()},
    }
    return raw_frame, standardized, schema


def write_preanalysis_feature_outputs(output, course_rows, raw_rows, train_groups) -> dict:
    raw, standardized, schema = build_preanalysis_feature_frames(course_rows, raw_rows, train_groups)
    raw.to_csv(Path(output) / "preanalysis_features_raw.csv", index=False, encoding="utf-8")
    standardized.to_csv(Path(output) / "preanalysis_features_zscore.csv", index=False, encoding="utf-8")
    write_json(Path(output) / "preanalysis_feature_schema.json", schema)
    return schema


def _write_training_config(output, args, source_receipts, outer_receipt) -> None:
    config = {
        "run_id": Path(output).name,
        "objective": "self-supervised masked temporal reconstruction; no fatigue or phase labels",
        "seed": args.seed,
        "input_sources": {"copied_inputs": source_receipts, "outer_course_archive_receipt": outer_receipt},
        "preprocessing": {
            "input_column": "ADC_Value",
            "input_units": "uncalibrated ADC counts",
            "sampling_rate_hz": 1000.0,
            "gap_rule": "split when dt > max(0.0015 s, 1.5 * median_dt)",
            "jitter_max": 0.05,
            "bandpass_hz": [20.0, 400.0],
            "butterworth_prototype_order": 4,
            "application": "SOS zero-phase forward-backward filter",
            "conditional_line_notch": {"frequencies_hz": [50.0, 60.0], "trigger_db": 10.0, "q": 30.0},
            "edge_mask_s_each_end": 0.1,
        },
        "windows": {"type": "fixed_length_sliding", "samples": WINDOW_SAMPLES, "seconds_at_1000_hz": 0.5, "hop_samples": HOP_SAMPLES, "overlap": 0.5},
        "normalization": "population z-score fitted on training windows only",
        "preanalysis_features": {"schema": "emg_window_features_v2_sample_span_plus_band_power_v1", "columns": list(PREANALYSIS_FEATURE_COLUMNS), "scaler_fit": "course_zip train windows only", "used_as_ssl_encoder_inputs": False},
        "self_supervised_task": {"name": "masked_temporal_reconstruction", "mask_fraction": MASK_FRACTION, "contiguous_block_samples": MASK_BLOCK_SAMPLES, "loss": "masked-sample mean squared error"},
        "architecture": {"name": "MaskedConvAutoencoder", "encoder_channels": [16, 32, 64, 64], "embedding_dimension": 64},
        "optimization": {"optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "max_epochs": args.epochs, "early_stopping_patience": args.patience, "minimum_validation_improvement": args.min_delta},
        "validation": {"method": "deterministic 20 percent group holdout from course ZIP, split before windows", "validation_fraction": args.validation_fraction},
        "raw_rar": "held out from fitting and checkpoint selection; descriptive reconstruction only; cross-archive identity unverified",
        "fatigue_classification": "not run because no independent fatigue labels are supplied",
    }
    path = Path(output) / "training_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def finalize_existing_run(args) -> Path:
    """Complete audit manifests from a finished checkpoint without retraining."""
    course_path = Path(args.course_archive).expanduser().resolve()
    raw_path = Path(args.raw_archive).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("private training outputs must be stored outside the Git repository")
    if (output / "run_manifest.json").exists() and not getattr(args, "rebuild_preanalysis_features", False):
        raise FileExistsError("run_manifest.json already exists; refusing to replace a finalized run")
    required = [
        "checkpoint.pt", "training_history.csv", "window_manifest.csv", "preprocessing_qc.csv",
        "split_manifest.json", "preprocessing_manifest.json", "feature_schema.json",
        "group_reconstruction_metrics.csv", "line_interference_qc.csv",
    ]
    if not getattr(args, "rebuild_preanalysis_features", False):
        required.extend(["preanalysis_features_raw.csv", "preanalysis_features_zscore.csv", "preanalysis_feature_schema.json"])
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot finalize incomplete run; missing {missing}")
    source_receipts = [
        _copy_verified(course_path, output / "source" / "course_source.zip"),
        _copy_verified(raw_path, output / "source" / "raw.rar"),
    ]
    outer_receipt = None
    if args.outer_archive:
        outer_path = Path(args.outer_archive).expanduser().resolve()
        outer_receipt = {"filename": outer_path.name, "sha256": sha256_file(outer_path), "bytes": outer_path.stat().st_size}
    _write_training_config(output, args, source_receipts, outer_receipt)
    _write_training_config(output, args, source_receipts, outer_receipt)

    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    history = pd.read_csv(output / "training_history.csv")
    windows = pd.read_csv(output / "window_manifest.csv")
    qc = pd.read_csv(output / "preprocessing_qc.csv")
    split = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    if getattr(args, "rebuild_preanalysis_features", False):
        cfg = StudyConfig(notch=True)
        _, course_rows, _, _ = _source_recordings(course_path, "course_zip", cfg)
        _, raw_rows, _, _ = _source_recordings(raw_path, "raw_rar", cfg)
        write_preanalysis_feature_outputs(output, course_rows, raw_rows, split["train_groups"])
    best_epoch = int(checkpoint["best_epoch"])
    write_training_curve(history, output / "training_curve.png")
    best_history = history.loc[history["epoch"] == best_epoch]
    if best_history.empty:
        raise ValueError("checkpoint best epoch is absent from training_history.csv")
    best_row = best_history.iloc[0]
    baseline = float(best_row["validation_zero_baseline_mse"])
    best_loss = float(checkpoint["best_validation_masked_mse"])
    improvement = 100 * (baseline - best_loss) / baseline if baseline else None
    course_qc = qc.loc[qc["source"] == "course_zip"]
    raw_qc = qc.loc[qc["source"] == "raw_rar"]
    course_windows = windows.loc[windows["source"] == "course_zip"]
    raw_windows = windows.loc[windows["source"] == "raw_rar"]
    rates = pd.to_numeric(windows["sampling_rate_hz"], errors="coerce").dropna()
    code = implementation_provenance()
    runner_path = Path(__file__).resolve()
    run_manifest = {
        "run_id": output.name,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "self-supervised masked temporal reconstruction on original unlabeled EMG",
        "fatigue_detector_trained": False,
        "movement_classifier_trained": False,
        "supervised_fatigue_gate": {
            "status": "not_trainable_from_attached_course_data",
            "reason": "no independent Borg/MVC/fatigue reference or synchronized event labels are included with these signals",
            "temporal_position_proxy_used": False,
            "synthetic_or_inferred_labels_used": False,
            "accuracy_f1_confusion_matrix": "not applicable; no fatigue classifier was trained",
            "public_candidates_screened": [
                {
                    "dataset": "MUniverse Avrillon et al. 2024 (NEMAR nm000159)",
                    "doi": "10.82901/nemar.nm000159",
                    "decision": "not compatible for supervised fatigue classification",
                    "reason": "records vastus lateralis isometric knee extensions at 10-80% MVC with force signals; it does not provide the course's rectus-femoris 70%-1RM repetition-level fatigue labels",
                },
                {
                    "dataset": "A versatile knee exoskeleton mitigates quadriceps fatigue in lifting, lowering, and carrying tasks (Dryad)",
                    "doi": "10.5061/dryad.z34tmpgks",
                    "decision": "not compatible for supervised EMG fatigue classification",
                    "reason": "the released EMG/kinematics/torque data are from non-fatiguing tasks; the fatigued squat portion provides performance and perceptual outcomes, not paired fatigue-task EMG",
                },
                {
                    "dataset": "sEMG and self-perceived fatigue dataset (Zenodo 14182446)",
                    "doi": "10.5281/zenodo.14182446",
                    "decision": "not compatible for the course target",
                    "reason": "upper-limb muscles and upper-body tasks; perceived fatigue cannot be relabeled as rectus-femoris knee-extension fatigue",
                },
                {
                    "dataset": "Use of Surface Electromyography to Measure Muscle Fatigue in Patients in an Acute Care Hospital",
                    "doi": "10.1093/ptj/pzaa035",
                    "decision": "scientifically related study; raw dataset not located as a directly downloadable public artifact",
                    "reason": "the paper describes synchronized rectus-femoris/vastus-lateralis EMG and IMU in isometric/dynamic time-to-task-failure tests, but this does not supply a verified downloadable, participant-level training dataset for the course's 70%-1RM protocol",
                },
            ],
            "missing_for_training": [
                "fatigue state tied to each recording or repetition by an approved operational definition",
                "verified participant/session crosswalk and synchronized reference clock",
                "for the intended mechanical criterion, compatible force/MVC or validated performance-decrement measurements",
            ],
        },
        "protocol_scope": {
            "course_guideline_describes": "rectus femoris, unilateral knee extension at 70% 1RM, Borg CR-10 and MVC/video references",
            "signal_level_mapping": "CSV files expose side and series but do not independently document muscle identity, exercise, load, or fatigue state per window",
        },
        "data": {
            "course_zip_recordings": int(course_qc["recording_id"].nunique()),
            "course_zip_filename_groups": _unique_group_count_from_ids(course_qc["group_id"]),
            "course_zip_windows": int(len(course_windows)),
            "raw_rar_recordings": int(raw_qc["recording_id"].nunique()),
            "raw_rar_filename_groups": _unique_group_count_from_ids(raw_qc["group_id"]),
            "raw_rar_windows": int(len(raw_windows)),
            "raw_rar_role": "source holdout for descriptive reconstruction; excluded from fitting and checkpoint selection",
            "raw_rar_cross_archive_identity": "unknown; no independent participant claim",
            "sampling_rate_range_hz": [float(rates.min()), float(rates.max())] if len(rates) else None,
        },
        "sources": {"copied_inputs": source_receipts, "outer_course_archive_receipt": outer_receipt},
        "training": {
            "framework": "PyTorch",
            "torch_version": torch.__version__,
            "device": "cpu",
            "seed": int(checkpoint["seed"]),
            "requested_epochs": int(args.epochs),
            "completed_epochs": int(len(history)),
            "best_epoch": best_epoch,
            "best_validation_masked_mse": best_loss,
            "validation_zero_baseline_mse": baseline,
            "best_validation_improvement_over_zero_baseline_pct": improvement,
            "parameter_count": int(checkpoint["architecture"]["parameter_count"]),
            "elapsed_training_seconds": float(pd.to_numeric(history["epoch_seconds"], errors="coerce").sum()),
            "optimizer": {"name": "AdamW", "learning_rate": float(args.learning_rate), "weight_decay": float(args.weight_decay)},
            "batch_size": int(args.batch_size),
            "early_stopping_patience": int(args.patience),
        },
        "split": {
            "method": "group-level filename-derived holdout before windowing",
            "train_group_count": len(split["train_groups"]),
            "validation_group_count": len(split["validation_groups"]),
            "groups_disjoint": set(split["train_groups"]).isdisjoint(split["validation_groups"]),
            "identity_status": "filename-derived grouping; participant identity crosswalk unverified",
        },
        "metrics_scope": "masked signal reconstruction only; no classification accuracy, F1, or confusion matrix",
        "supervised_classifier_artifacts": {
            "status": "not_applicable_no_supervised_fatigue_labels",
            "files_not_created": ["oof_predictions.csv", "benchmark_results.csv", "confusion_matrix.csv", "per_subject_metrics.csv"],
            "reason": "empty classification artifacts would imply a classifier evaluation that was not performed",
        },
        "preanalysis_features": None,
        "code_provenance": {
            "git_revision": code["git_revision"],
            "package_source_sha256": code["package_source_sha256"],
            "module_hashes": code["module_hashes"],
            "training_script_sha256": sha256_file(runner_path),
            "working_tree_changes_present_at_run": True,
        },
    }
    write_json(output / "supervised_gate.json", run_manifest["supervised_fatigue_gate"])
    write_json(output / "classification_artifacts_status.json", run_manifest["supervised_classifier_artifacts"])
    preanalysis_schema_path = output / "preanalysis_feature_schema.json"
    if preanalysis_schema_path.is_file():
        feature_schema = json.loads(preanalysis_schema_path.read_text(encoding="utf-8"))
        run_manifest["preanalysis_features"] = {
            "raw_rows": int(len(pd.read_csv(output / "preanalysis_features_raw.csv"))),
            "standardized_rows": int(len(pd.read_csv(output / "preanalysis_features_zscore.csv"))),
            "feature_columns": feature_schema["feature_columns"],
            "scaler_fit_split": feature_schema["standardization"]["fit_source"],
            "features_used_by_ssl_encoder": False,
        }
    artifact_hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            rel = path.relative_to(output).as_posix()
            artifact_hashes[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    run_manifest["artifacts"] = artifact_hashes
    write_json(output / "run_manifest.json", run_manifest)
    return output


def _unique_group_count_from_ids(values) -> int:
    return len({str(value) for value in values if pd.notna(value)})


def _metrics(model, values: torch.Tensor, masks: torch.Tensor, batch_size: int) -> tuple[float, float]:
    model.eval()
    model_total = 0.0
    baseline_total = 0.0
    n_windows = 0
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            x = values[start : start + batch_size]
            mask = masks[start : start + batch_size]
            prediction = model(x, mask)
            n = len(x)
            model_total += masked_reconstruction_loss(prediction, x, mask).item() * n
            baseline_total += masked_reconstruction_loss(torch.zeros_like(x), x, mask).item() * n
            n_windows += n
    if n_windows == 0:
        raise ValueError("cannot evaluate an empty split")
    return model_total / n_windows, baseline_total / n_windows


def _make_values(rows: list[dict], mean: float, scale: float) -> torch.Tensor:
    matrix = np.stack([row["signal"] for row in rows]).astype(np.float32, copy=False)
    normalized = (matrix - np.float32(mean)) / np.float32(scale)
    return torch.from_numpy(normalized[:, None, :].copy())


def _group_metric_rows(model, rows, mean, scale, *, seed, batch_size, role):
    metrics = []
    for group_offset, group_id in enumerate(sorted({row["group_id"] for row in rows})):
        selected = [row for row in rows if row["group_id"] == group_id]
        values = _make_values(selected, mean, scale)
        masks = random_block_mask(
            batch_size=len(values),
            n_samples=WINDOW_SAMPLES,
            block_size=MASK_BLOCK_SAMPLES,
            mask_fraction=MASK_FRACTION,
            generator=torch.Generator().manual_seed(seed + group_offset),
        )
        model_mse, baseline_mse = _metrics(model, values, masks, batch_size)
        metrics.append({
            "source_role": role,
            "group_id": group_id,
            "n_windows": len(selected),
            "masked_reconstruction_mse": model_mse,
            "zero_baseline_mse": baseline_mse,
            "improvement_over_zero_baseline_pct": 100 * (baseline_mse - model_mse) / baseline_mse if baseline_mse else None,
        })
    return metrics


def _train(args) -> Path:
    course_path = Path(args.course_archive).expanduser().resolve()
    raw_path = Path(args.raw_archive).expanduser().resolve()
    if not course_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError("both original course ZIP and raw.rar inputs are required")
    output = Path(args.output_dir).expanduser().resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("private training outputs must be stored outside the Git repository")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source"
    source_receipts = [
        _copy_verified(course_path, source_dir / "course_source.zip"),
        _copy_verified(raw_path, source_dir / "raw.rar"),
    ]
    outer_receipt = None
    if args.outer_archive:
        outer_path = Path(args.outer_archive).expanduser().resolve()
        outer_receipt = {"filename": outer_path.name, "sha256": sha256_file(outer_path), "bytes": outer_path.stat().st_size}

    if (output / "checkpoint.pt").exists():
        raise FileExistsError("checkpoint.pt already exists; choose a new run directory to preserve prior results")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu")
    cfg = StudyConfig(notch=True)

    course_records, course_rows, qc_rows, line_rows = _source_recordings(course_path, "course_zip", cfg)
    raw_records, raw_rows, raw_qc, raw_line_rows = _source_recordings(raw_path, "raw_rar", cfg)
    qc_rows.extend(raw_qc)
    line_rows.extend(raw_line_rows)
    if not course_rows:
        raise ValueError("course ZIP has no windows passing preprocessing")
    if not raw_rows:
        raise ValueError("raw.rar has no windows passing preprocessing")

    groups_with_windows = sorted({row["group_id"] for row in course_rows})
    split = split_groups(groups_with_windows, validation_fraction=args.validation_fraction, seed=args.seed)
    train_groups = set(split["train"])
    validation_groups = set(split["validation"])
    train_rows = [row for row in course_rows if row["group_id"] in train_groups]
    validation_rows = [row for row in course_rows if row["group_id"] in validation_groups]
    if not train_rows or not validation_rows:
        raise ValueError("group split produced an empty train or validation set")
    write_preanalysis_feature_outputs(output, course_rows, raw_rows, train_groups)

    # Train-only population standardization; RAR and validation signals do not fit it.
    train_samples = np.concatenate([row["signal"] for row in train_rows]).astype(np.float64)
    normalization_mean = float(np.mean(train_samples))
    normalization_scale = float(np.std(train_samples, ddof=0))
    if not np.isfinite(normalization_scale) or normalization_scale <= 1e-12:
        raise ValueError("training split has no usable signal variation")
    train_values = _make_values(train_rows, normalization_mean, normalization_scale).to(device)
    validation_values = _make_values(validation_rows, normalization_mean, normalization_scale).to(device)
    raw_values = _make_values(raw_rows, normalization_mean, normalization_scale).to(device)

    validation_masks = random_block_mask(
        batch_size=len(validation_values),
        n_samples=WINDOW_SAMPLES,
        block_size=MASK_BLOCK_SAMPLES,
        mask_fraction=MASK_FRACTION,
        generator=torch.Generator().manual_seed(args.seed + 1),
        device=device,
    )

    model = MaskedConvAutoencoder().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    order_generator = torch.Generator().manual_seed(args.seed + 2)
    mask_generator = torch.Generator().manual_seed(args.seed + 3)
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_values), generator=order_generator)
        training_total = 0.0
        training_windows = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch = train_values[indices].to(device)
            mask = random_block_mask(
                batch_size=len(batch),
                n_samples=WINDOW_SAMPLES,
                block_size=MASK_BLOCK_SAMPLES,
                mask_fraction=MASK_FRACTION,
                generator=mask_generator,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = masked_reconstruction_loss(model(batch, mask), batch, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            training_total += loss.detach().item() * len(batch)
            training_windows += len(batch)
        train_mse = training_total / training_windows
        validation_mse, zero_baseline_mse = _metrics(model, validation_values, validation_masks, args.batch_size)
        improvement = 100 * (zero_baseline_mse - validation_mse) / zero_baseline_mse if zero_baseline_mse else 0.0
        row = {
            "epoch": epoch,
            "train_masked_mse": train_mse,
            "validation_masked_mse": validation_mse,
            "validation_zero_baseline_mse": zero_baseline_mse,
            "validation_improvement_over_zero_baseline_pct": improvement,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(f"epoch={epoch:03d} train_masked_mse={train_mse:.6f} val_masked_mse={validation_mse:.6f} baseline={zero_baseline_mse:.6f}", flush=True)
        if validation_mse < best_loss - args.min_delta:
            best_loss = validation_mse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training finished without a finite validation checkpoint")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": {
            "name": "MaskedConvAutoencoder",
            "input_channels": 1,
            "mask_indicator_channels": 1,
            "encoder_channels": [16, 32, 64, 64],
            "embedding_dim": model.embedding_dim,
            "parameter_count": parameter_count,
        },
        "input": {"window_samples": WINDOW_SAMPLES, "sampling_rate_hz": 1000.0, "window_seconds": 0.5},
        "normalization": {"method": "training_window_population_zscore", "mean": normalization_mean, "scale": normalization_scale},
        "best_epoch": best_epoch,
        "best_validation_masked_mse": best_loss,
        "seed": args.seed,
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    _write_csv(output / "training_history.csv", history)

    # Evaluation masks for held-out groups are generated independently from training.
    group_metrics = _group_metric_rows(
        model,
        validation_rows,
        normalization_mean,
        normalization_scale,
        seed=args.seed + 100,
        batch_size=args.batch_size,
        role="course_zip_grouped_validation",
    )
    group_metrics.extend(_group_metric_rows(
        model,
        raw_rows,
        normalization_mean,
        normalization_scale,
        seed=args.seed + 200,
        batch_size=args.batch_size,
        role="raw_rar_source_holdout_identity_crosswalk_unverified",
    ))
    _write_csv(output / "group_reconstruction_metrics.csv", group_metrics)

    window_manifest = []
    for row in course_rows:
        split_name = "train" if row["group_id"] in train_groups else "validation"
        window_manifest.append({key: value for key, value in row.items() if key != "signal"} | {"split": split_name})
    for row in raw_rows:
        window_manifest.append({key: value for key, value in row.items() if key != "signal"} | {"split": "source_holdout_descriptive"})
    _write_csv(output / "window_manifest.csv", window_manifest)
    _write_csv(output / "preprocessing_qc.csv", qc_rows)
    _write_csv(output / "line_interference_qc.csv", line_rows)

    windows_by_group = {}
    for group_id in sorted({row["group_id"] for row in course_rows + raw_rows}):
        course_count = sum(row["group_id"] == group_id for row in course_rows)
        raw_count = sum(row["group_id"] == group_id for row in raw_rows)
        windows_by_group[group_id] = {"course_zip_windows": course_count, "raw_rar_windows": raw_count}
    write_json(output / "split_manifest.json", {
        "split_method": "deterministic_group_holdout_before_windowing",
        "group_key": "filename-derived archive folder group; identity status unverified",
        "seed": args.seed,
        "validation_fraction_requested": args.validation_fraction,
        "train_groups": split["train"],
        "validation_groups": split["validation"],
        "groups_disjoint": set(split["train"]).isdisjoint(split["validation"]),
        "windows_by_group": windows_by_group,
        "raw_rar_use": "excluded from fitting and checkpoint selection; descriptive source-holdout reconstruction only",
        "cross_archive_identity_crosswalk": "unverified; raw RAR groups are not claimed as independent participants",
    })

    sos = signal.butter(4, (20.0, 400.0), btype="bandpass", fs=1000.0, output="sos")
    qc_status_counts = pd.Series([row["qc_status"] for row in qc_rows]).value_counts().to_dict()
    notch_applied = sum(bool(row["notch_applied"]) for row in line_rows)
    line50 = [row["prominence_50_hz_db"] for row in line_rows if row["prominence_50_hz_db"] is not None]
    line60 = [row["prominence_60_hz_db"] for row in line_rows if row["prominence_60_hz_db"] is not None]
    preprocessing_manifest = {
        "input_signal": "ADC_Value counts; Voltage_mV ignored because calibration is not independently verified",
        "gap_rule": "split when dt > max(0.0015 s, 1.5 * median_dt); no filtering or window crosses a split",
        "jitter_rule": "(q95(dt) - q05(dt)) / median(dt) <= 0.05",
        "minimum_sampling_rate_hz": 800.0,
        "bandpass": {
            "family": "digital Butterworth band-pass",
            "prototype_order": 4,
            "digital_order": 8,
            "cutoffs_hz": [20.0, 400.0],
            "application": "SOS zero-phase forward-backward filtering",
            "equation": "y = sosfiltfilt(SOS, x - mean(x))",
            "sos_coefficients_at_1000_hz": sos.tolist(),
        },
        "line_interference_check": {
            "method": "2 s Hann periodogram; 10*log10(mean PSD within f0 +/- 1 Hz / mean PSD at 2-5 Hz offsets)",
            "candidate_hz": [50.0, 60.0],
            "notch_trigger_db": 10.0,
            "notch_q": 30.0,
            "triggered_runs": notch_applied,
            "prominence_50_hz_db_median_p90": [float(np.median(line50)), float(np.percentile(line50, 90))] if line50 else None,
            "prominence_60_hz_db_median_p90": [float(np.median(line60)), float(np.percentile(line60, 90))] if line60 else None,
        },
        "edge_mask": {"formula": "max(3*(2*n_sos_sections+1), ceil(0.1*fs)) samples at each end", "observed_samples_at_1000_hz": 100},
        "windowing": {
            "type": "fixed-length sliding rectangular windows within valid filtered runs",
            "window_samples": WINDOW_SAMPLES,
            "window_seconds_at_1000_hz": 0.5,
            "hop_samples": HOP_SAMPLES,
            "overlap_fraction": 0.5,
            "window_acceptance": "all samples must pass edge mask and timestamps must be gap-free",
        },
        "normalization": {"method": "training-window population z-score; train split only", "mean": normalization_mean, "scale": normalization_scale},
        "qc_run_status_counts": {str(key): int(value) for key, value in qc_status_counts.items()},
    }
    write_json(output / "preprocessing_manifest.json", preprocessing_manifest)

    write_json(output / "feature_schema.json", {
        "schema": "masked_ssl_emg_representation_v1",
        "input": {"name": "filtered_adc_window", "shape": [1, WINDOW_SAMPLES], "unit": "standardized ADC counts", "sampling_rate_hz": 1000.0},
        "engineered_features": [],
        "encoder_representation": {"shape": [64], "pooling": "global average over 64-channel temporal latent map"},
        "pretext_target": "reconstruct only randomly masked time samples",
        "masked_fraction": MASK_FRACTION,
        "contiguous_mask_block_samples": MASK_BLOCK_SAMPLES,
        "contiguous_mask_block_ms": 25.0,
        "loss": "mean squared error on masked samples only",
        "fatigue_label": None,
        "movement_phase_label": None,
        "classifier_head": None,
        "interpretation": "representation-learning output; not a fatigue or extension/flexion prediction",
    })

    finalized = finalize_existing_run(args)
    print(json.dumps({
        "output_dir": str(finalized),
        "course_zip_recordings": len(course_records),
        "course_zip_windows": len(course_rows),
        "train_groups": len(split["train"]),
        "validation_groups": len(split["validation"]),
        "raw_rar_holdout_windows": len(raw_rows),
        "best_epoch": best_epoch,
        "best_validation_masked_mse": best_loss,
        "parameter_count": parameter_count,
        "fatigue_classifier_trained": False,
    }, indent=2), flush=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-archive", required=True, help="inner ZIP containing original course CSV files")
    parser.add_argument("--raw-archive", required=True, help="original raw.rar archive")
    parser.add_argument("--outer-archive", help="outer course ZIP for source-provenance hash")
    parser.add_argument("--output-dir", required=True, help="private output directory outside this repository")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--min-delta", type=float, default=0.00001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--finalize-existing", action="store_true", help="complete missing audit manifests from an existing checkpoint without retraining")
    parser.add_argument("--rebuild-preanalysis-features", action="store_true", help="recompute required window features and refresh the private run manifest")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("epochs, patience, batch-size, and cpu-threads must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        parser.error("learning-rate must be positive and weight-decay/min-delta nonnegative")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.finalize_existing:
        print(finalize_existing_run(parsed))
    else:
        _train(parsed)
