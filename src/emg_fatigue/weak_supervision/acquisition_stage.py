"""Run development-only acquisition QC, preprocessing, activity, and window audits."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .activity import detect_candidate_activity
from .contracts import ActivitySegments, ProcessedRecording, RawRecording
from .io import load_training_zip
from .preprocess import preprocess_recording
from .windows import make_windows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _json_number(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...], rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _private_output_hashes(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def _acquisition_row(recording: RawRecording, processed: ProcessedRecording) -> dict[str, object]:
    audit = processed.audit
    adc_values = np.asarray(recording.adc_value, dtype=np.float64)
    voltage_values = (
        np.asarray(recording.voltage_mv, dtype=np.float64)
        if recording.voltage_mv is not None
        else np.asarray([], dtype=np.float64)
    )
    return {
        "group_id": recording.group_id,
        "recording_id": recording.recording_id,
        "source_file_sha256": recording.source_sha256,
        "sample_count": audit.sample_count,
        "duration_s": _json_number(audit.duration_s),
        "effective_sampling_rate_hz": _json_number(audit.sampling_rate_hz),
        "median_dt_s": _json_number(audit.median_dt_s),
        "dt_p05_s": _json_number(audit.dt_p05_s),
        "dt_p95_s": _json_number(audit.dt_p95_s),
        "jitter_percent": _json_number(audit.jitter_percent),
        "gap_threshold_s": _json_number(audit.gap_threshold_s),
        "continuous_run_count": len(audit.continuous_ranges),
        "gap_count": audit.gap_count,
        "duplicate_timestamp_count": audit.duplicate_timestamp_count,
        "clock_reset_count": audit.clock_reset_count,
        "nyquist_hz": _json_number(audit.nyquist_hz),
        "adc_voltage_pair_count": audit.adc_voltage_pair_count,
        "adc_voltage_slope_mv_per_adc_unit": _json_number(audit.adc_voltage_slope_mv_per_unit),
        "adc_voltage_intercept_mv": _json_number(audit.adc_voltage_intercept_mv),
        "adc_voltage_r_squared": _json_number(audit.adc_voltage_r_squared),
        "adc_value_integer_like_fraction": _json_number(audit.adc_value_integer_like_fraction),
        "adc_value_min": _json_number(np.nanmin(adc_values)) if adc_values.size else None,
        "adc_value_max": _json_number(np.nanmax(adc_values)) if adc_values.size else None,
        "adc_negative_sample_fraction": _json_number(np.mean(adc_values < 0)) if adc_values.size else None,
        "voltage_mv_min": _json_number(np.nanmin(voltage_values)) if voltage_values.size else None,
        "voltage_mv_max": _json_number(np.nanmax(voltage_values)) if voltage_values.size else None,
        "sampling_compatible": audit.sampling_compatible,
        "in_band_20_400_hz_power_fraction": _json_number(audit.in_band_power_fraction),
        "below_20_hz_power_fraction": _json_number(audit.below_20_hz_power_fraction),
        "above_400_hz_power_fraction": _json_number(audit.above_400_hz_power_fraction),
        "line_prominence_50_hz_db": _json_number(audit.line_prominence_db["50"]),
        "line_prominence_60_hz_db": _json_number(audit.line_prominence_db["60"]),
        "notch_frequency_hz": audit.notch_frequency_hz,
        "raw_signal_compatible": audit.signal_compatible,
        "processed_segment_count": len(processed.segments),
        "qc_invalid_sample_count": int(audit.quality_invalid_mask.sum()),
        "quality_flags": ";".join(audit.quality_flags),
        "preprocessing_status": "processed" if processed.segments else "excluded_before_filtering",
    }


def _activity_rows(processed: ProcessedRecording, activity: ActivitySegments):
    by_index = {item.segment_index: item for item in activity.segments}
    if not processed.segments:
        yield {
            "group_id": processed.group_id,
            "recording_id": processed.recording_id,
            "segment_index": 0,
            "status": "indeterminate",
            "threshold_rms": None,
            "active_frame_fraction": None,
            "valid_frame_count": 0,
            "active_frame_count": 0,
            "exclusion_reason": ";".join(processed.audit.quality_flags) or "no_processable_segment",
        }
        return
    for segment in processed.segments:
        item = by_index[segment.segment_index]
        yield {
            "group_id": processed.group_id,
            "recording_id": processed.recording_id,
            "segment_index": segment.segment_index,
            "status": item.status,
            "threshold_rms": _json_number(item.threshold_rms),
            "active_frame_fraction": _json_number(item.active_fraction),
            "valid_frame_count": item.valid_frame_count,
            "active_frame_count": int(item.frame_active[item.frame_valid].sum()),
            "exclusion_reason": "" if item.status == "determinate" else "activity_outside_5_95_percent_or_insufficient_frames",
        }


def _window_rows(processed: ProcessedRecording, activity: ActivitySegments, duration_s: float, overlap: float):
    for window in make_windows(processed, activity, duration_s, overlap):
        yield {
            "window_key": (
                f"{window.group_id}:{window.recording_id}:s{window.segment_index}:"
                f"d{int(round(window.duration_s * 1000)):04d}:i{window.window_index:06d}"
            ),
            "group_id": window.group_id,
            "recording_id": window.recording_id,
            "segment_index": window.segment_index,
            "window_index": window.window_index,
            "start_sample": window.start_sample,
            "start_time_s": window.start_time_s,
            "duration_s": window.duration_s,
            "sampling_rate_hz": window.sampling_rate_hz,
            "activity_occupancy": window.activity_occupancy,
            "eligible_for_future_reference_alignment": window.eligible_for_label,
            "rejection_reason": window.rejection_reason or "",
        }


def _coefficients(config: dict, notch_frequencies: set[float]) -> dict[str, object]:
    acquisition = config["acquisition"]
    target = float(acquisition["target_sampling_rate_hz"])
    bandpass = signal.butter(
        int(acquisition["filter_order"]),
        [float(acquisition["bandpass_low_hz"]), float(acquisition["bandpass_high_hz"])],
        btype="bandpass",
        fs=target,
        output="sos",
    )
    notches = {}
    for frequency in sorted(notch_frequencies):
        b, a = signal.iirnotch(frequency, float(acquisition["notch_q"]), fs=target)
        notches[str(int(frequency))] = {"b": b.tolist(), "a": a.tolist()}
    return {
        "target_sampling_rate_hz": target,
        "detrend": "linear per continuous run before filtering",
        "resampling": "scipy.signal.resample_poly with Kaiser beta 5; only if native rate > 800 Hz",
        "bandpass": {
            "family": "Butterworth",
            "order": int(acquisition["filter_order"]),
            "cutoff_hz": [float(acquisition["bandpass_low_hz"]), float(acquisition["bandpass_high_hz"])],
            "implementation": "second-order sections, zero-phase sosfiltfilt",
            "sos": bandpass.tolist(),
        },
        "notch": {
            "family": "IIR notch",
            "q": float(acquisition["notch_q"]),
            "minimum_raw_line_prominence_db_exclusive": float(acquisition["line_prominence_notch_db"]),
            "coefficients_by_selected_frequency_hz": notches,
        },
        "filter_edge_mask_s_per_edge": float(acquisition["filter_edge_mask_s"]),
    }


def _save_plots(
    plot_dir: Path,
    acquisition_rows: list[dict[str, object]],
    window_counts: dict[str, dict[str, object]],
    example: tuple[RawRecording, ProcessedRecording] | None,
) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    rates = [float(row["effective_sampling_rate_hz"]) for row in acquisition_rows if row["effective_sampling_rate_hz"] is not None]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    if rates:
        median_rate = float(np.median(rates))
        bins = min(24, max(5, int(np.sqrt(len(rates)))))
        if float(np.ptp(rates)) < max(0.5, median_rate * 1e-3):
            axis.hist(rates, bins=bins, range=(median_rate - 0.5, median_rate + 0.5), color="#2c7fb8", edgecolor="white")
            axis.set_xlim(median_rate - 0.5, median_rate + 0.5)
        else:
            axis.hist(rates, bins=bins, color="#2c7fb8", edgecolor="white")
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.set(title="Development recording sampling rates", xlabel="Effective sampling rate (Hz)", ylabel="Recordings")
    fig.tight_layout()
    fig.savefig(plot_dir / "sampling_rate_distribution.png", dpi=160)
    plt.close(fig)
    outputs.append("plots/sampling_rate_distribution.png")

    fraction_names = ("in_band_20_400_hz_power_fraction", "below_20_hz_power_fraction", "above_400_hz_power_fraction")
    fraction_labels = ("20–400 Hz", "<20 Hz", ">400 Hz")
    values = [
        [float(row[name]) for row in acquisition_rows if row[name] is not None]
        for name in fraction_names
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    if any(values):
        axis.boxplot(values, tick_labels=fraction_labels, showmeans=True)
    axis.axhline(0.8, color="#d7301f", linestyle="--", linewidth=1, label="Configured in-band minimum")
    axis.set(title="Raw power spectral density fractions", ylabel="Fraction of integrated power", ylim=(0, 1.05))
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "raw_psd_power_fractions.png", dpi=160)
    plt.close(fig)
    outputs.append("plots/raw_psd_power_fractions.png")

    durations = sorted(window_counts, key=float)
    eligible_counts = [window_counts[key]["eligible"] for key in durations]
    rejected_counts = [window_counts[key]["rejected"] for key in durations]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(durations))
    axis.bar(positions, eligible_counts, color="#31a354", label="Eligible for future reference alignment")
    axis.bar(positions, rejected_counts, bottom=eligible_counts, color="#fd8d3c", label="Rejected by QC/activity gates")
    axis.set_xticks(positions, [f"{int(round(float(item) * 1000))} ms" for item in durations])
    axis.set(title="Development window counts", ylabel="Windows")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "window_counts_by_duration.png", dpi=160)
    plt.close(fig)
    outputs.append("plots/window_counts_by_duration.png")

    if example is not None:
        raw, processed = example
        segment = processed.segments[0]
        raw_start, raw_stop = processed.audit.continuous_ranges[0]
        raw_values = np.asarray(raw.adc_value[raw_start:raw_stop], dtype=np.float64)
        raw_times = np.asarray(raw.relative_time_s[raw_start:raw_stop], dtype=np.float64)
        display_samples = min(raw_values.size, int(round(1.5 * processed.audit.sampling_rate_hz)))
        display_processed = min(segment.signal.size, int(round(1.5 * segment.sampling_rate_hz)))
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(raw_times[:display_samples], raw_values[:display_samples], linewidth=0.7, label="Raw ADC_Value")
        axes[0].plot(
            segment.start_time_s + np.arange(display_processed) / segment.sampling_rate_hz,
            segment.signal[:display_processed],
            linewidth=0.7,
            label="Processed",
        )
        axes[0].set(xlabel="Relative time (s)", ylabel="Signal units", title="Representative raw and processed excerpt")
        axes[0].legend(loc="best")
        if raw_values.size >= 64:
            raw_fs = processed.audit.sampling_rate_hz
            f_raw, p_raw = signal.welch(raw_values, fs=raw_fs, nperseg=min(raw_values.size, int(raw_fs)))
            f_proc, p_proc = signal.welch(segment.signal, fs=segment.sampling_rate_hz, nperseg=min(segment.signal.size, int(segment.sampling_rate_hz)))
            axes[1].semilogy(f_raw, np.maximum(p_raw, np.finfo(float).tiny), label="Raw")
            axes[1].semilogy(f_proc, np.maximum(p_proc, np.finfo(float).tiny), label="Processed")
        axes[1].set(xlabel="Frequency (Hz)", ylabel="PSD density", xlim=(0, 500), title="Raw and processed power spectra")
        axes[1].legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / "representative_waveform_and_psd.png", dpi=160)
        plt.close(fig)
        outputs.append("plots/representative_waveform_and_psd.png")
    return outputs


def run_acquisition_stage(training_zip: Path, run_dir: Path, config_path: Path) -> dict[str, object]:
    """Create private development-only QC and window receipts from the original ZIP."""
    repo_root = Path(__file__).resolve().parents[3]
    training_zip = Path(training_zip).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    private_dir = run_dir / "private"
    input_manifest_path = private_dir / "input_manifest.json"
    split_manifest_path = private_dir / "split_manifest.json"
    run_manifest_path = private_dir / "run_manifest.json"
    if not all(path.is_file() for path in (input_manifest_path, split_manifest_path, run_manifest_path)):
        raise FileNotFoundError("Prepare the training ZIP and split before running acquisition QC.")
    acquisition_dir = private_dir / "acquisition"
    if acquisition_dir.exists() and any(acquisition_dir.iterdir()):
        raise FileExistsError("The private acquisition-stage directory is not empty.")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    prior_run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    archive_sha = _sha256_file(training_zip)
    if prior_run.get("input_archive_sha256") != archive_sha:
        raise ValueError("Training ZIP hash differs from the archive prepared for this private run.")
    recordings, inventory = load_training_zip(
        training_zip,
        seed=int(config["run"]["split_seed"]),
        train_fraction=float(config["run"]["train_fraction"]),
    )
    if list(inventory.split.development_groups) != split["development_group_ids"]:
        raise ValueError("Development group split differs from the prepared private run.")
    if list(inventory.split.test_groups) != split["test_group_ids"]:
        raise ValueError("Locked internal test groups differ from the prepared private run.")
    if any(item.parsed_numeric_values for item in inventory.members if item.group_id in set(split["test_group_ids"])):
        raise RuntimeError("Internal-test numeric values were parsed during acquisition processing.")
    if acquisition_dir.exists():
        acquisition_dir.rmdir()
    acquisition_dir.mkdir(parents=True)

    audit_path = acquisition_dir / "acquisition_audit.csv"
    activity_path = acquisition_dir / "activity_audit.csv"
    window_path = acquisition_dir / "window_manifest.csv"
    audit_fields = (
        "group_id", "recording_id", "source_file_sha256", "sample_count", "duration_s",
        "effective_sampling_rate_hz", "median_dt_s", "dt_p05_s", "dt_p95_s", "jitter_percent",
        "gap_threshold_s", "continuous_run_count", "gap_count", "duplicate_timestamp_count",
        "clock_reset_count", "nyquist_hz", "sampling_compatible", "in_band_20_400_hz_power_fraction",
        "adc_voltage_pair_count", "adc_voltage_slope_mv_per_adc_unit", "adc_voltage_intercept_mv",
        "adc_voltage_r_squared", "adc_value_integer_like_fraction", "adc_value_min", "adc_value_max",
        "adc_negative_sample_fraction", "voltage_mv_min", "voltage_mv_max",
        "below_20_hz_power_fraction", "above_400_hz_power_fraction", "line_prominence_50_hz_db",
        "line_prominence_60_hz_db", "notch_frequency_hz", "raw_signal_compatible",
        "processed_segment_count", "qc_invalid_sample_count", "quality_flags", "preprocessing_status",
    )
    activity_fields = (
        "group_id", "recording_id", "segment_index", "status", "threshold_rms",
        "active_frame_fraction", "valid_frame_count", "active_frame_count", "exclusion_reason",
    )
    window_fields = (
        "window_key", "group_id", "recording_id", "segment_index", "window_index", "start_sample",
        "start_time_s", "duration_s", "sampling_rate_hz", "activity_occupancy",
        "eligible_for_future_reference_alignment", "rejection_reason",
    )
    rates = Counter()
    raw_invalid_flags = Counter()
    window_counts: dict[str, dict[str, object]] = {
        str(float(duration)): {"total": 0, "eligible": 0, "rejected": 0, "rejection_reasons": {}}
        for duration in config["run"]["window_durations_s"]
    }
    notch_frequencies: set[float] = set()
    acquisition_rows: list[dict[str, object]] = []
    activity_row_count = 0
    window_row_count = 0
    example: tuple[RawRecording, ProcessedRecording] | None = None

    with audit_path.open("w", encoding="utf-8", newline="") as audit_stream, activity_path.open(
        "w", encoding="utf-8", newline=""
    ) as activity_stream, window_path.open("w", encoding="utf-8", newline="") as window_stream:
        audit_writer = csv.DictWriter(audit_stream, fieldnames=audit_fields)
        activity_writer = csv.DictWriter(activity_stream, fieldnames=activity_fields)
        window_writer = csv.DictWriter(window_stream, fieldnames=window_fields)
        audit_writer.writeheader()
        activity_writer.writeheader()
        window_writer.writeheader()
        for recording in recordings:
            processed = preprocess_recording(recording, config)
            audit_row = _acquisition_row(recording, processed)
            acquisition_rows.append(audit_row)
            audit_writer.writerow(audit_row)
            rates["sampling_compatible"] += int(processed.audit.sampling_compatible)
            rates["raw_signal_compatible"] += int(processed.audit.signal_compatible)
            rates["processed_with_segments"] += int(bool(processed.segments))
            if processed.audit.notch_frequency_hz is not None:
                notch_frequencies.add(float(processed.audit.notch_frequency_hz))
            raw_invalid_flags.update(processed.audit.quality_flags)
            activity = detect_candidate_activity(processed, config)
            for row in _activity_rows(processed, activity):
                activity_writer.writerow(row)
                activity_row_count += 1
            if example is None and processed.segments:
                example = (recording, processed)
            for duration_s in config["run"]["window_durations_s"]:
                duration_key = str(float(duration_s))
                for row in _window_rows(processed, activity, float(duration_s), float(config["run"]["window_overlap"])):
                    window_writer.writerow(row)
                    window_row_count += 1
                    stats = window_counts[duration_key]
                    stats["total"] += 1
                    if row["eligible_for_future_reference_alignment"]:
                        stats["eligible"] += 1
                    else:
                        stats["rejected"] += 1
                        reasons = stats["rejection_reasons"]
                        reasons[row["rejection_reason"]] = reasons.get(row["rejection_reason"], 0) + 1

    coefficients = _coefficients(config, notch_frequencies)
    _write_json(acquisition_dir / "preprocessing_coefficients.json", coefficients)
    plot_paths = _save_plots(acquisition_dir / "plots", acquisition_rows, window_counts, example)
    software = {"python": os.sys.version.split()[0]}
    for distribution in ("numpy", "scipy", "matplotlib"):
        try:
            software[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            software[distribution] = None

    output_hashes = _private_output_hashes(acquisition_dir)
    receipt = {
        "stage": "development_acquisition_qc_preprocessing_activity_windows",
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_archive_sha256": archive_sha,
        "config_sha256": _sha256_file(config_path),
        "git_branch": _git(repo_root, "branch", "--show-current"),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "source_files_sha256": {
            path.relative_to(repo_root).as_posix(): _sha256_file(path)
            for path in (
                repo_root / "src" / "emg_fatigue" / "weak_supervision" / "acquisition_stage.py",
                repo_root / "src" / "emg_fatigue" / "weak_supervision" / "activity.py",
                repo_root / "src" / "emg_fatigue" / "weak_supervision" / "contracts.py",
                repo_root / "src" / "emg_fatigue" / "weak_supervision" / "io.py",
                repo_root / "src" / "emg_fatigue" / "weak_supervision" / "preprocess.py",
                repo_root / "src" / "emg_fatigue" / "weak_supervision" / "runner.py",
                repo_root / "src" / "emg_fatigue" / "weak_supervision" / "windows.py",
                repo_root / "src" / "emg_fatigue" / "weak_supervision" / "__init__.py",
                repo_root / "tools" / "run_fatigue_proxy.py",
            )
        },
        "split_seed": inventory.split.seed,
        "development_group_ids": list(inventory.split.development_groups),
        "internal_test_group_ids": list(inventory.split.test_groups),
        "development_record_count": len(recordings),
        "development_group_count": len(inventory.split.development_groups),
        "internal_test_member_count": inventory.unparsed_internal_test_member_count,
        "internal_test_numeric_values_parsed": 0,
        "external_archive_opened": False,
        "per_record_original_file_hashes_receipt": "../input_manifest.json",
        "sampling_gate": {
            "minimum_source_rate_hz_exclusive": float(config["acquisition"]["minimum_source_rate_hz"]),
            "maximum_timestamp_jitter_percent": float(config["acquisition"]["maximum_jitter_percent"]),
            "gap_threshold": "max(1.5 * median positive dt, 1.5 ms)",
            "target_sampling_rate_hz": float(config["acquisition"]["target_sampling_rate_hz"]),
        },
        "raw_psd_gate": {
            "candidate_passband_hz": [float(config["acquisition"]["bandpass_low_hz"]), float(config["acquisition"]["bandpass_high_hz"])],
            "minimum_in_band_power_fraction": float(config["acquisition"]["minimum_in_band_power_fraction"]),
            "maximum_below_20_hz_power_fraction": float(config["acquisition"]["maximum_below_20_hz_power_fraction"]),
            "maximum_above_400_hz_power_fraction": float(config["acquisition"]["maximum_above_400_hz_power_fraction"]),
            "all_recordings": "see acquisition_audit.csv",
        },
        "adc_to_voltage_audit": {
            "voltage_column_assumed_units": "mV as named by the source column; physical calibration not independently verified",
            "per_recording_linear_fit": "Voltage_mV = slope * ADC_Value + intercept; see acquisition_audit.csv",
            "signed_adc_representation": "min/max and integer-like fraction are recorded; not assumed to be hardware counts",
        },
        "quality_flag_counts": dict(sorted(raw_invalid_flags.items())),
        "recording_gate_counts": dict(sorted(rates.items())),
        "activity_configuration": {
            "rms_frame_s": float(config["activity"]["rms_frame_s"]),
            "rms_hop_s": float(config["activity"]["rms_hop_s"]),
            "fill_inactive_gap_s": float(config["activity"]["fill_inactive_gap_s"]),
            "indeterminate_outside_active_fraction": [float(config["activity"]["minimum_active_frame_fraction"]), float(config["activity"]["maximum_active_frame_fraction"])],
        },
        "window_configuration": {
            "durations_s": [float(item) for item in config["run"]["window_durations_s"]],
            "overlap": float(config["run"]["window_overlap"]),
            "primary_duration_s": float(config["run"]["primary_window_s"]),
            "eligible_threshold_activity_occupancy": 0.5,
            "eligibility_is_not_a_fatigue_or_phase_label": True,
            "counts_by_duration": window_counts,
        },
        "row_counts": {
            "acquisition_audit": len(acquisition_rows),
            "activity_audit": activity_row_count,
            "window_manifest": window_row_count,
        },
        "plot_paths": plot_paths,
        "software_versions": software,
        "output_sha256_excluding_this_manifest": output_hashes,
    }
    stage_manifest = acquisition_dir / "acquisition_stage_manifest.json"
    _write_json(stage_manifest, receipt)

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["stage"] = "development_acquisition_qc_preprocessing_activity_windows"
    run_manifest["acquisition_stage_manifest_sha256"] = _sha256_file(stage_manifest)
    run_manifest["external_archive_opened_this_run"] = False
    run_manifest["git_branch"] = _git(repo_root, "branch", "--show-current")
    run_manifest["git_commit"] = _git(repo_root, "rev-parse", "HEAD")
    run_manifest["git_working_tree_status"] = _git(repo_root, "status", "--short")
    run_manifest["output_sha256"] = _private_output_hashes(private_dir, exclude={"run_manifest.json"})
    _write_json(run_manifest_path, run_manifest)
    return {
        "development_record_count": len(recordings),
        "unparsed_internal_test_member_count": inventory.unparsed_internal_test_member_count,
        "recording_gate_counts": dict(sorted(rates.items())),
        "window_counts": window_counts,
        "acquisition_manifest_path": str(stage_manifest),
    }
