"""Prepare private, reference-aligned sEMG windows from the Zenodo release."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from emg_fatigue.perceived_dataset import (
    CHANNEL_ORDER,
    TRIAL_PRIME_MOVER,
    iter_perceived_trials,
)
from emg_fatigue.perceived_preprocessing import process_perceived_trial


DATASET_ID = "10.5281/zenodo.14182446"
DATASET_VERSION = "v2"
SEED = 20261109
WINDOW_S = 0.5
OVERLAP = 0.5
BAND_HZ = (20.0, 450.0)
FILTER_ORDER = 4
EDGE_MASK_S = 0.10


def file_hashes(path: Path) -> dict[str, str]:
    hashes = {"md5": hashlib.md5(), "sha256": hashlib.sha256()}
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            for digest in hashes.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in hashes.items()}


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {
        "window_id",
        "subject_id",
        "trial_id",
        "prime_mover",
        "window_start_s",
        "window_end_s",
        "window_center_s",
        "label",
        "label_status",
    }
    return [str(column) for column in frame.columns if column not in metadata]


def prepare_dataset(emg_zip: Path, labels_zip: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    emg_hashes = file_hashes(emg_zip)
    label_hashes = file_hashes(labels_zip)
    expected_emg_md5 = "ea7338b95543ada98b1ddc65d4b6aabc"
    expected_label_md5 = "3352c990b2baf2246cfa7b331e52607a"
    if emg_hashes["md5"] != expected_emg_md5:
        raise ValueError("EMG archive MD5 does not match the official Zenodo v2 record")
    if label_hashes["md5"] != expected_label_md5:
        raise ValueError("label archive MD5 does not match the official Zenodo v2 record")

    feature_rows = []
    window_audit_rows = []
    sampling_rows = []
    qc_rows = []
    filter_rows = []
    trial_rows = []
    source_rows = []
    label_sample_counts = Counter()
    label_cleanup_counts = Counter()
    exclusions_by_trial = []
    seen = set()
    for trial in iter_perceived_trials(emg_zip, labels_zip):
        key = (trial.subject_id, trial.trial_id)
        if key in seen:
            raise ValueError(f"duplicate matched participant/trial key: {key}")
        seen.add(key)
        for state, count in zip(*np.unique(trial.perceived_level, return_counts=True)):
            label_sample_counts[str(int(state))] += int(count)
        for name, value in trial.label_cleanup.items():
            if name == "rows_dropped":
                label_cleanup_counts["rows_dropped"] += int(value)
            elif name == "terminal_backward_duplicate_dropped" and value:
                label_cleanup_counts[name] += 1

        processed = process_perceived_trial(
            trial,
            window_s=WINDOW_S,
            overlap=OVERLAP,
            band_hz=BAND_HZ,
            filter_order=FILTER_ORDER,
            edge_mask_s=EDGE_MASK_S,
        )
        feature_rows.extend(
            {
                **row["features"],
                "window_id": row["window_id"],
                "subject_id": row["subject_id"],
                "trial_id": row["trial_id"],
                "prime_mover": row["prime_mover"],
                "window_start_s": row["window_start_s"],
                "window_end_s": row["window_end_s"],
                "window_center_s": row["window_center_s"],
                "label": row["label"],
                "label_status": row["label_status"],
            }
            for row in processed.feature_rows
        )
        window_audit_rows.extend(processed.window_audit_rows)
        sampling_rows.extend(processed.sampling_rate_rows)
        qc_rows.extend(processed.qc_rows)
        filter_rows.extend(processed.filter_design_rows)
        trial_rows.append({
            "subject_id": trial.subject_id,
            "trial_id": trial.trial_id,
            "raw_member": trial.raw_member,
            "raw_member_bytes": trial.raw_member_bytes,
            "raw_member_crc32": trial.raw_member_crc32,
            "label_member": trial.label_member,
            "label_member_bytes": trial.label_member_bytes,
            "label_member_crc32": trial.label_member_crc32,
            "raw_rows_input": int(trial.signal_metadata["input_rows"]),
            "raw_terminal_reset_rows_dropped": int(trial.signal_metadata["rows_dropped"]),
            "raw_samples_processed": int(len(trial.time_s)),
            "raw_start_s": float(trial.time_s[0]),
            "raw_end_s": float(trial.time_s[-1]),
            "label_rows_after_cleanup": int(len(trial.label_time_s)),
            "label_first_s": float(trial.label_time_s[0]),
            "label_last_s": float(trial.label_time_s[-1]),
            "common_duration_s": float(
                max(0.0, min(trial.time_s[-1], trial.label_time_s[-1]) - trial.time_s[0])
            ),
            "initial_reported_state": int(trial.perceived_level[0]),
            "initial_state_from_protocol": 0,
            "terminal_rows_dropped": int(trial.label_cleanup["rows_dropped"]),
            "label_rate_hz": float(trial.label_cleanup["estimated_rate_hz"]),
            "emg_rate_hz": float(processed.estimated_fs_hz),
            "jitter_cv": float(processed.jitter_cv),
            "gap_count": int(processed.gap_count),
            "window_samples": int(processed.window_samples),
            "hop_samples": int(processed.hop_samples),
            "feature_windows": int(len(processed.feature_rows)),
            "raw_member_crc_read_to_eof": True,
            "label_member_crc_read_to_eof": True,
            "exclusion_counts": json.dumps(processed.exclusion_counts, sort_keys=True),
        })
        exclusions_by_trial.append({
            "subject_id": trial.subject_id,
            "trial_id": trial.trial_id,
            **processed.exclusion_counts,
        })
        if len(seen) % 12 == 0:
            print(
                f"Prepared {len(seen)}/156 matched trials; "
                f"eligible windows so far: {len(feature_rows)}",
                flush=True,
            )

    if len(seen) != 13 * 12:
        raise ValueError(f"expected 156 unique subject/trial pairs, found {len(seen)}")
    if {subject for subject, _ in seen} != {f"subject_{i}" for i in range(1, 14)}:
        raise ValueError("matched corpus does not contain exactly subjects 1 through 13")
    if {trial for _, trial in seen} != {f"trial_{i}" for i in range(1, 13)}:
        raise ValueError("matched corpus does not contain exactly trials 1 through 12")

    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise ValueError("no stable, reference-aligned feature windows were produced")
    feature_columns = _feature_columns(features)
    if len(feature_columns) != 4 * 15:
        raise ValueError(f"expected 60 active-muscle features, found {len(feature_columns)}")
    if not np.isfinite(features[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("feature matrix contains nonfinite values")
    if set(features["label"].unique()) != {0, 1}:
        raise ValueError("eligible binary dataset must contain both perceived-fatigue classes")
    if features["subject_id"].nunique() != 13:
        raise ValueError("all 13 independent participants must contribute eligible windows")
    features = features.sort_values(
        ["subject_id", "trial_id", "window_start_s"], kind="stable"
    ).reset_index(drop=True)
    features.to_parquet(output_dir / "features.parquet", index=False)
    pd.DataFrame(window_audit_rows).to_csv(output_dir / "window_audit.csv", index=False)
    pd.DataFrame(sampling_rows).to_csv(output_dir / "sampling_rate_audit.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(output_dir / "qc_summary.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(output_dir / "trial_coverage.csv", index=False)
    pd.DataFrame(filter_rows).to_json(
        output_dir / "filter_designs.json", orient="records", indent=2
    )
    pd.DataFrame(exclusions_by_trial).fillna(0).to_csv(
        output_dir / "exclusions_by_trial.csv", index=False
    )

    label_definition = {
        "task_id": "PERCEIVED_FATIGUE_CLASSIFICATION",
        "reference": "participant self-reported 3-level fatigue state, recorded at 50 Hz",
        "source_level_0": "no perceived fatigue; binary target 0",
        "source_level_1": "transition to fatigue; excluded from the primary binary task",
        "source_level_2": "perceived fatigue; binary target 1",
        "window_label_rule": "all 50 Hz reference events in the full 500 ms window must have the same state",
        "initial_state": "level 0 at acquisition start, per the published protocol",
        "prediction_unit": "stable 500 ms EMG window aligned to a time-stamped reference state",
        "synchronization": "EMG and rating systems used a Delsys hardware trigger; use their shared elapsed-time origin and only their common support",
        "mechanical_fatigue_reference": False,
        "movement_phase_reference": False,
        "label_version": "self_report_0_vs_2_transition_1_excluded_v1",
    }
    (output_dir / "label_definition.json").write_text(
        json.dumps(label_definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    feature_schema = {
        "schema_id": "zenodo_perceived_fatigue_sEMG_4channel_features_v1",
        "input_units": "V as identified in the raw CSV column names",
        "feature_channels": list(CHANNEL_ORDER),
        "feature_order": feature_columns,
        "formulas": {
            "rms": "sqrt(sum(x_i^2)/N)",
            "mav": "sum(abs(x_i))/N",
            "waveform_length_v": "sum(abs(x_i-x_(i-1)))",
            "waveform_length_v_per_s": "waveform_length_v/((N-1)/fs)",
            "iemg_v_s": "sum(abs(x_i))/fs",
            "periodogram": "one-sided scipy periodogram, Hann taper, constant-detrended input, PSD density",
            "mnf_hz": "sum(f_k P_k)/sum(P_k), for 20<=f<=450 Hz",
            "mdf_hz": "first f_k whose cumulative P_k reaches 0.5*sum(P_k), for 20<=f<=450 Hz",
            "bandpower_v2": "sum(P_k)*df over 20-60, 60-150, and 150-450 Hz",
            "spectral_entropy_norm": "-sum(p_k log(p_k))/log(K), p_k=P_k/sum(P_k), 20-450 Hz",
            "wavelet_energy": "db4 level-3 symmetric extension; each coefficient-set squared energy divided by total coefficient energy",
        },
        "predictor_exclusions": ["subject_id", "trial_id", "prime_mover", "window_id", "time", "label", "label_status"],
        "filters": {
            "bandpass_hz": list(BAND_HZ),
            "butterworth_prototype_order": FILTER_ORDER,
            "representation": "SOS",
            "application": "zero-phase sosfiltfilt, independently within continuous segments",
            "edge_mask_s_each_end": EDGE_MASK_S,
            "notch": "only when measured 50/60 Hz prominence exceeds 10 dB; all decisions recorded in filter_designs.json",
        },
        "windows": {
            "type": "fixed-length sliding",
            "primary_duration_s": WINDOW_S,
            "overlap_fraction": OVERLAP,
            "sample_count": "round(0.5*measured_fs_hz)",
            "hop_samples": "round(N*(1-0.5))",
            "nominal_frequency_spacing": "measured_fs_hz/N",
        },
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    audit = {
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "source_url": "https://zenodo.org/records/14182446",
        "license": "CC BY 4.0",
        "emg_archive": {"path_basename": emg_zip.name, "bytes": emg_zip.stat().st_size, **emg_hashes},
        "label_archive": {"path_basename": labels_zip.name, "bytes": labels_zip.stat().st_size, **label_hashes},
        "expected_md5_matches": {"emg": True, "labels": True},
        "participants": int(features["subject_id"].nunique()),
        "trials": int(len(seen)),
        "channels_per_trial": 4,
        "channel_names": list(CHANNEL_ORDER),
        "signal_unit": "V",
        "published_emg_sampling_hz": 1259,
        "published_label_sampling_hz": 50,
        "observed_fs_hz_range": [
            float(pd.DataFrame(sampling_rows)["effective_fs_hz"].min()),
            float(pd.DataFrame(sampling_rows)["effective_fs_hz"].max()),
        ],
        "observed_label_sample_counts_after_terminal_cleanup": dict(label_sample_counts),
        "label_cleanup": dict(label_cleanup_counts),
        "raw_terminal_reset_rows_dropped": int(
            pd.DataFrame(trial_rows)["raw_terminal_reset_rows_dropped"].sum()
        ),
        "raw_terminal_reset_trials": int(
            (pd.DataFrame(trial_rows)["raw_terminal_reset_rows_dropped"] > 0).sum()
        ),
        "raw_emg_time_origin_alignment": "shared elapsed-time origin documented by Delsys hardware trigger; labels initialize at 0 and start samples are near 0.02 s",
        "common_support_seconds_min_max": [
            float(pd.DataFrame(trial_rows)["common_duration_s"].min()),
            float(pd.DataFrame(trial_rows)["common_duration_s"].max()),
        ],
        "matched_subject_trial_pairs": len(seen),
        "verified_active_muscles": list(CHANNEL_ORDER),
        "prime_mover_by_trial_from_released_notebook": TRIAL_PRIME_MOVER,
        "label_definition_id": label_definition["label_version"],
        "prediction_unit": label_definition["prediction_unit"],
        "eligible_windows": int(len(features)),
        "eligible_windows_by_class": {
            str(int(label)): int(count)
            for label, count in features["label"].value_counts().sort_index().items()
        },
        "windows_by_subject_and_class": {
            str(subject): {str(int(label)): int(count) for label, count in rows.items()}
            for subject, rows in features.groupby("subject_id")["label"].value_counts().unstack(fill_value=0).iterrows()
        },
        "windows_by_trial_and_class": {
            f"{subject}/{trial}": {str(int(label)): int(count) for label, count in rows.items()}
            for (subject, trial), rows in features.groupby(["subject_id", "trial_id"])["label"].value_counts().unstack(fill_value=0).iterrows()
        },
        "objective_force_reference": False,
        "phase_reference": False,
        "note": "This dataset validates self-perceived fatigue classification only; it does not provide objective force-loss labels or synchronized extension/flexion labels.",
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    feature_manifest = {
        "features_file": "features.parquet",
        "rows": int(len(features)),
        "columns": list(features.columns),
        "predictor_columns": feature_columns,
        "feature_rows_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(features[feature_columns], index=False)
            .to_numpy()
            .tobytes()
        ).hexdigest(),
        "parquet_sha256": hashlib.sha256((output_dir / "features.parquet").read_bytes()).hexdigest(),
        "input_archive_hashes": {"emg": emg_hashes, "labels": label_hashes},
    }
    (output_dir / "features_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": "run_001",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "task_id": label_definition["task_id"],
        "unit_of_prediction": label_definition["prediction_unit"],
        "seed": SEED,
        "window_s": WINDOW_S,
        "overlap": OVERLAP,
        "band_hz": list(BAND_HZ),
        "filter_prototype_order": FILTER_ORDER,
        "edge_mask_s": EDGE_MASK_S,
        "code_revision": "pending_benchmark_commit",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "results_directory_private": True,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--emg-zip", type=Path, required=True)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = prepare_dataset(args.emg_zip, args.labels_zip, args.output_dir)
    print(json.dumps({
        "task_id": audit["label_definition_id"],
        "participants": audit["participants"],
        "trials": audit["trials"],
        "eligible_windows": audit["eligible_windows"],
        "class_counts": audit["eligible_windows_by_class"],
        "observed_fs_hz_range": audit["observed_fs_hz_range"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
