"""Frozen, prediction-only evaluation for the unlabeled P11/P12 RAR archive."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import activity as activity_module
from . import augmentation as augmentation_module
from . import evaluation
from . import evaluation_internal
from . import evaluation_tabular_noise as tabular_noise_module
from . import features as features_module
from . import io as io_module
from . import label_audit as label_audit_module
from . import preprocess as preprocess_module
from . import pseudo_labels as pseudo_labels_module
from . import windows as windows_module
from .activity import detect_candidate_activity
from .augmentation import CORRUPTION_FAMILIES, corrupt_window, stable_seed
from .contracts import RawRecording
from .evaluation_guardrails import OOD_FEATURES, _robust_ood_score
from .features import FEATURE_VIEWS, window_key
from .label_audit import _extract_rows_for_windows
from .preprocess import preprocess_recording
from .windows import make_windows


LABEL_RULE = "low_frequency_fraction_persistence_v1"
A1_FEATURES = tuple(FEATURE_VIEWS["A1_morphology_only"])
EXPECTED_EXTERNAL_SHA256 = "15b61b870269a95da894d3225cc43e95fb2465eb0ba80107664bbf015f799e75"
EXTERNAL_GROUPS = ("P11", "P12")
EXTERNAL_FAMILIES = tuple(CORRUPTION_FAMILIES)
EXTERNAL_LEVELS = (0.25, 0.50)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _verified_artifact_map(values: dict[str, str], label: str) -> None:
    for raw_path, expected in values.items():
        path = Path(raw_path)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"{label} checksum mismatch: {path}")


def _external_members(unrar_exe: Path, archive: Path) -> list[str]:
    completed = subprocess.run(
        [str(unrar_exe), "lb", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    expected = {
        f"P{group}_set{series:02d}_{side}.csv"
        for group in (11, 12)
        for series in range(1, 6)
        for side in ("dr", "iz")
    }
    if len(names) != 20 or set(names) != expected:
        raise ValueError("The external RAR member list differs from the documented 20 P11/P12 CSV contract")
    return sorted(names)


def _prediction_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby(
        ["condition", "family", "requested_level", "group_id", "model", "policy"],
        dropna=False,
        sort=True,
    ).agg(
        windows_scored_descriptive_only=("window_key", "count"),
        proxy_fatigue_like_prediction_count=("prediction_class_id", lambda values: int((values == 1).sum())),
        proxy_fatigue_like_prediction_rate=("prediction_class_id", lambda values: float(np.mean(values == 1))),
        abstention_count=("abstain", "sum"),
        coverage=("abstain", lambda values: float(np.mean(~values.astype(bool)))),
        mean_probability_fatigue_like=("probability_class_1", "mean"),
        median_probability_fatigue_like=("probability_class_1", "median"),
        mean_confidence=("confidence", "mean"),
        robust_mad_ood_flag_count=("ood_flag", "sum"),
        robust_mad_ood_flag_rate=("ood_flag", "mean"),
    ).reset_index()
    summary["independent_labels_available"] = False
    summary["classification_accuracy_computed"] = False
    summary["classification_macro_f1_computed"] = False
    return summary


def run_external_unlabeled_evaluation(
    external_archive: Path,
    run_dir: Path,
    config_path: Path,
    *,
    unrar_exe: Path = Path(r"C:\Program Files\WinRAR\UnRAR.exe"),
    frozen_subdir: str = "frozen_development_lowfreq_v1",
    temporal_subdir: str = "frozen_development_temporal_baseline_v4",
    internal_subdir: str = "internal_lowfreq_frozen_evaluation_v3",
    output_subdir: str = "external_raw_rar_frozen_predictions_v1",
) -> dict[str, Any]:
    """Run clean and noisy predictions only; do not invent external labels."""
    external_archive = Path(external_archive).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    unrar_exe = Path(unrar_exe).expanduser().resolve()
    private = run_dir / "private"
    if not external_archive.is_file() or not unrar_exe.is_file():
        raise FileNotFoundError("The supplied external RAR or WinRAR UnRAR executable was not found")
    if external_archive.name.casefold() != "raw.rar":
        raise ValueError("This run expects the provided raw.rar content alias for the documented P11/P12 archive")
    archive_sha = _sha256_file(external_archive)
    if archive_sha.casefold() != EXPECTED_EXTERNAL_SHA256:
        raise ValueError("External RAR checksum does not match the documented P11/P12 source archive")
    for value in (output_subdir, frozen_subdir, temporal_subdir, internal_subdir):
        if Path(value).name != value or value in {"", ".", ".."}:
            raise ValueError("Stage subdirectories must each be one path component")

    # Require a complete internal holdout evaluation before opening the external archive.
    internal_dir = private / internal_subdir
    internal_manifest_path = internal_dir / "internal_evaluation_manifest.json"
    if not internal_manifest_path.is_file():
        raise FileNotFoundError("The internal holdout stage must complete before external evaluation")
    internal_manifest = json.loads(internal_manifest_path.read_text(encoding="utf-8"))
    if internal_manifest.get("status") != "completed" or internal_manifest.get("raw2_rar_opened") is not False:
        raise ValueError("Internal holdout completion receipt is invalid or shows premature RAR access")
    _verified_artifact_map(internal_manifest.get("outputs_sha256", {}), "internal-evaluation output")

    frozen, freeze_manifest, prediction_policy, ood_policy = evaluation_internal._verify_freeze(private, frozen_subdir)
    temporal_dir = private / temporal_subdir
    temporal_manifest_path = temporal_dir / "temporal_baseline_freeze_manifest.json"
    if not temporal_manifest_path.is_file():
        raise FileNotFoundError("The temporal-only development comparator has not been frozen")
    temporal_manifest = json.loads(temporal_manifest_path.read_text(encoding="utf-8"))
    if (
        temporal_manifest.get("status") != "completed"
        or int(temporal_manifest.get("locked_internal_test_values_read", -1)) != 0
        or temporal_manifest.get("raw2_rar_opened") is not False
    ):
        raise ValueError("Temporal baseline does not have a valid development-only freeze receipt")
    _verified_artifact_map(temporal_manifest.get("outputs_sha256", {}), "temporal-baseline output")
    if not np.isclose(float(temporal_manifest.get("selected_threshold", np.nan)), 0.23361054368864045, atol=1e-12):
        raise ValueError("External temporal baseline threshold differs from the frozen development result")

    feature_label_manifest_path = private / "features_labels" / "feature_label_stage_manifest.json"
    feature_label_manifest = json.loads(feature_label_manifest_path.read_text(encoding="utf-8"))
    if feature_label_manifest.get("config_sha256") != _sha256_file(config_path):
        raise ValueError("External preprocessing configuration is not the exact development snapshot")
    if feature_label_manifest.get("training_archive_sha256") != internal_manifest.get("input_archive_sha256"):
        raise ValueError("Development and internal-stage source archive receipts disagree")
    output = private / output_subdir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite external prediction stage: {output}")

    # Integrity test and central-directory-only listing happen after internal completion.
    integrity = subprocess.run(
        [str(unrar_exe), "t", "-inul", str(external_archive)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if integrity.returncode != 0:
        raise ValueError("External RAR failed the WinRAR integrity test")
    members = _external_members(unrar_exe, external_archive)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not all(name in config.get("labels", {}) for name in ("line_exclusion_hz",)):
        raise ValueError("Frozen configuration lacks deterministic feature parameters")
    primary_duration = float(config["run"]["primary_window_s"])
    overlap = float(config["run"]["window_overlap"])
    occupancy = float(config["activity"]["primary_window_occupancy"])
    line_exclusion = float(config["labels"]["line_exclusion_hz"])

    a1_package = joblib.load(frozen / "frozen_a1_morphology_logistic.joblib")
    if (
        not isinstance(a1_package, dict)
        or a1_package.get("label_rule") != LABEL_RULE
        or tuple(a1_package.get("feature_names", ())) != A1_FEATURES
        or not np.isclose(float(a1_package.get("decision_threshold", np.nan)), float(prediction_policy["selected_decision_threshold"]))
    ):
        raise ValueError("Frozen external A1 model package disagrees with its development receipt")
    a1_model = a1_package["estimator"]
    temporal_model = joblib.load(temporal_dir / "frozen_temporal_baseline.joblib")
    a1_cut = float(prediction_policy["selected_decision_threshold"])
    fixed_cut = float(prediction_policy["fixed_threshold_comparator"])
    time_cut = float(temporal_manifest["selected_threshold"])
    confidence_cut = float(prediction_policy["abstention"]["confidence_threshold"])
    ood_center = np.asarray(ood_policy["center"], dtype=float)
    ood_scale = np.asarray(ood_policy["scale"], dtype=float)
    ood_keep = np.asarray(ood_policy["keep"], dtype=bool)
    ood_cut = float(ood_policy["threshold"])
    support_names = tuple(str(name) for name in ood_policy["feature_support_names"])
    support_indices = np.asarray([A1_FEATURES.index(name) for name in support_names], dtype=int)
    support_min = np.asarray(ood_policy["feature_support_minimum"], dtype=float)
    support_max = np.asarray(ood_policy["feature_support_maximum"], dtype=float)
    ood_indices = np.asarray([A1_FEATURES.index(name) for name in OOD_FEATURES], dtype=int)

    source_hashes: dict[str, str] = {}
    member_rows: list[dict[str, Any]] = []
    all_feature_rows: list[dict[str, Any]] = []
    signal_by_key: dict[str, np.ndarray] = {}
    rate_by_key: dict[str, float] = {}
    quality_by_key: dict[str, bool] = {}
    for member in members:
        extracted = subprocess.run(
            [str(unrar_exe), "p", "-inul", str(external_archive), member],
            capture_output=True,
            check=True,
        )
        payload = extracted.stdout
        member_sha = hashlib.sha256(payload).hexdigest()
        source_hashes[member] = member_sha
        group, stem = member.split("_", maxsplit=1)
        series = stem.split("_", maxsplit=1)[0]
        recording_id = member.removesuffix(".csv").replace("_", "_")
        recording, row_count = io_module._parse_development_record(
            payload,
            group,
            recording_id,
            member_sha,
            member,
        )
        recording.filename_metadata["series_token"] = series.upper()
        total_member_bytes = len(payload)
        processed = preprocess_recording(recording, config)
        activity = detect_candidate_activity(processed, config)
        windows = make_windows(processed, activity, primary_duration, overlap, minimum_occupancy=occupancy)
        rows, _ = _extract_rows_for_windows(
            windows,
            processed.audit.notch_frequency_hz,
            line_exclusion,
        )
        all_feature_rows.extend(rows)
        for window in windows:
            key = window_key(window)
            signal_by_key[key] = np.asarray(window.signal, dtype=np.float32)
            rate_by_key[key] = float(window.sampling_rate_hz)
            segment = next((value for value in processed.segments if value.segment_index == window.segment_index), None)
            quality_by_key[key] = bool(window.eligible_for_label and segment is not None and segment.acquisition_compatible)
        member_rows.append(
            {
                "group_id": group,
                "recording_id": recording_id,
                "member_name": member,
                "member_sha256": member_sha,
                "byte_size": total_member_bytes,
                "numeric_sample_count": int(row_count),
                "sampling_rate_hz": float(processed.audit.sampling_rate_hz),
                "signal_compatible": bool(processed.audit.signal_compatible),
                "continuous_segment_count": int(len(processed.segments)),
                "primary_window_count_descriptive_only": int(len(windows)),
                "eligible_window_count_descriptive_only": int(sum(item.eligible_for_label for item in windows)),
                "notch_frequency_hz": processed.audit.notch_frequency_hz,
                "quality_flags_json": json.dumps(list(processed.audit.quality_flags), separators=(",", ":")),
            }
        )

    features = pd.DataFrame(all_feature_rows)
    if features.empty:
        raise ValueError("The external RAR produced no windows under frozen preprocessing")
    features["relative_time_fraction"] = evaluation._relative_time_fraction(features)
    valid_mask = np.asarray(
        [quality_by_key.get(str(key), False) for key in features.window_key],
        dtype=bool,
    ) & (features.feature_valid.to_numpy(dtype=int) == 1)
    scored = features.loc[valid_mask].sort_values("window_key").reset_index(drop=True)
    if scored.empty:
        raise ValueError("No external windows passed the frozen quality and feature-validity gates")
    scored_keys = scored.window_key.astype(str).to_numpy()
    groups = scored.group_id.astype(str).to_numpy()
    recording_ids = scored.recording_id.astype(str).to_numpy()
    x_a1 = scored.loc[:, list(A1_FEATURES)].astype(float)
    x_time = scored[["relative_time_fraction"]].astype(float)
    if not np.isfinite(x_a1.to_numpy(dtype=float)).all() or not np.isfinite(x_time.to_numpy(dtype=float)).all():
        raise ValueError("An external window passing the quality gate contains non-finite model inputs")
    rates = np.asarray([rate_by_key[str(key)] for key in scored_keys], dtype=float)
    if len(set(np.round(rates, 6).tolist())) != 1:
        raise ValueError("The external RAR contains multiple processed window rates")
    clean_a1_probability = evaluation._positive_probability(a1_model, x_a1)[:, 1]
    clean_time_probability = evaluation._positive_probability(temporal_model, x_time)[:, 1]

    conditions: list[tuple[str, float, pd.DataFrame, np.ndarray | None]] = [
        ("clean", 0.0, x_a1.reset_index(drop=True), None)
    ]
    noise_receipts: list[dict[str, Any]] = []
    for family in EXTERNAL_FAMILIES:
        for level in EXTERNAL_LEVELS:
            altered = np.empty((len(scored), len(signal_by_key[scored_keys[0]])), dtype=np.float32)
            achieved: list[float] = []
            for position, key in enumerate(scored_keys):
                seed = stable_seed("emg-matched-evaluation-noise-v1", str(key), family, float(level))
                values, receipt = corrupt_window(
                    signal_by_key[str(key)],
                    family,
                    float(level),
                    seed,
                    sampling_rate_hz=float(rates[position]),
                )
                altered[position] = values
                achieved.append(float(receipt["achieved_rms_ratio"]))
                noise_receipts.append(
                    {
                        "window_key": str(key),
                        "group_id": str(groups[position]),
                        "recording_id": str(recording_ids[position]),
                        "family": family,
                        "requested_level": float(level),
                        "stress_test_gt_20_percent": True,
                        "corruption_seed": int(seed),
                        "achieved_rms_ratio": float(receipt["achieved_rms_ratio"]),
                        "sampling_rate_hz": float(receipt["sampling_rate_hz"]),
                        "label_preserving_assumption": True,
                        "label_preserving_independently_verified": False,
                    }
                )
            noise_features = tabular_noise_module._morphology_batch(altered, float(rates[0]))
            missing = set(A1_FEATURES) - set(noise_features.columns)
            if missing:
                raise ValueError(f"External matched-noise features are incomplete: {sorted(missing)}")
            noise_x = noise_features.loc[:, list(A1_FEATURES)].astype(float)
            if not np.isfinite(noise_x.to_numpy(dtype=float)).all():
                raise ValueError(f"External noise condition {family}/{level} generated invalid model inputs")
            conditions.append((family, float(level), noise_x, np.asarray(achieved, dtype=float)))

    prediction_rows: list[dict[str, Any]] = []
    for family, level, x_condition, achieved in conditions:
        condition = "clean:0" if family == "clean" else f"{family}:{level:.2f}"
        p_a1 = clean_a1_probability if family == "clean" else evaluation._positive_probability(a1_model, x_condition)[:, 1]
        p_time = clean_time_probability
        morphology = x_condition.to_numpy(dtype=float)
        ood_scores = _robust_ood_score(morphology[:, ood_indices], ood_center, ood_scale, ood_keep)
        ood_flags = ood_scores > ood_cut
        support_values = morphology[:, support_indices]
        outside = np.any((support_values < support_min) | (support_values > support_max), axis=1)
        confidence = np.maximum(p_a1, 1.0 - p_a1)
        pred_fixed = (p_a1 >= fixed_cut).astype(int)
        pred_selected = (p_a1 >= a1_cut).astype(int)
        conf_abstain = confidence < confidence_cut
        combined_abstain = conf_abstain | ood_flags | outside
        a1_policies = (
            ("fixed_0_5", pred_fixed, np.zeros(len(scored), dtype=bool), fixed_cut),
            ("development_oof_selected", pred_selected, np.zeros(len(scored), dtype=bool), a1_cut),
            ("confidence_abstention", pred_selected, conf_abstain, a1_cut),
            ("ood_abstention", pred_selected, ood_flags, a1_cut),
            ("combined_abstention", pred_selected, combined_abstain, a1_cut),
        )
        for policy, raw_pred, abstain, threshold in a1_policies:
            pred = np.where(abstain, -1, raw_pred).astype(int)
            for index in range(len(scored)):
                prediction_rows.append(
                    {
                        "window_key": str(scored_keys[index]),
                        "group_id": str(groups[index]),
                        "recording_id": str(recording_ids[index]),
                        "model": "A1_morphology_logistic",
                        "policy": policy,
                        "condition": condition,
                        "family": family,
                        "requested_level": float(level),
                        "probability_class_1": float(p_a1[index]),
                        "probability_class_0": float(1.0 - p_a1[index]),
                        "prediction_class_id": int(pred[index]),
                        "prediction_class_name": (
                            "ABSTAIN"
                            if pred[index] == -1
                            else ("FATIGUE_LIKE_PROXY" if pred[index] == 1 else "NON_FATIGUE_LIKE_PROXY")
                        ),
                        "abstain": bool(abstain[index]),
                        "decision_threshold": float(threshold),
                        "confidence": float(confidence[index]),
                        "confidence_threshold_frozen": float(confidence_cut),
                        "ood_score": float(ood_scores[index]),
                        "ood_threshold_frozen": float(ood_cut),
                        "ood_flag": bool(ood_flags[index]),
                        "outside_development_feature_support": bool(outside[index]),
                        "quality_valid": True,
                        "achieved_rms_ratio": float(achieved[index]) if achieved is not None else 0.0,
                        "independent_target_label": None,
                    }
                )
        for policy, prediction, threshold in (
            ("fixed_0_5", (p_time >= fixed_cut).astype(int), fixed_cut),
            ("development_oof_selected", (p_time >= time_cut).astype(int), time_cut),
        ):
            for index in range(len(scored)):
                prediction_rows.append(
                    {
                        "window_key": str(scored_keys[index]),
                        "group_id": str(groups[index]),
                        "recording_id": str(recording_ids[index]),
                        "model": "time_only_logistic",
                        "policy": policy,
                        "condition": condition,
                        "family": family,
                        "requested_level": float(level),
                        "probability_class_1": float(p_time[index]),
                        "probability_class_0": float(1.0 - p_time[index]),
                        "prediction_class_id": int(prediction[index]),
                        "prediction_class_name": "FATIGUE_LIKE_PROXY" if prediction[index] == 1 else "NON_FATIGUE_LIKE_PROXY",
                        "abstain": False,
                        "decision_threshold": float(threshold),
                        "confidence": float(max(p_time[index], 1.0 - p_time[index])),
                        "confidence_threshold_frozen": None,
                        "ood_score": None,
                        "ood_threshold_frozen": None,
                        "ood_flag": None,
                        "outside_development_feature_support": None,
                        "quality_valid": True,
                        "achieved_rms_ratio": float(achieved[index]) if achieved is not None else 0.0,
                        "independent_target_label": None,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    summary = _prediction_summary(predictions)
    output.mkdir(parents=True)
    predictions.to_parquet(output / "external_predictions.parquet", index=False)
    summary.to_csv(output / "prediction_distribution_by_group_condition.csv", index=False)
    pd.DataFrame(member_rows).to_csv(output / "external_member_processing_receipts.csv", index=False)
    pd.DataFrame(noise_receipts).to_parquet(output / "matched_noise_receipts.parquet", index=False)
    features.to_parquet(output / "external_window_features.parquet", index=False)

    manifest_path = output / "external_evaluation_manifest.json"
    manifest: dict[str, Any] = {
        "stage": "frozen_external_prediction_only_and_noise_sensitivity",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_archive_filename": external_archive.name,
        "external_archive_sha256": archive_sha,
        "external_archive_alias_note": "raw(2).rar was not present by filename; raw.rar SHA-256 matches the README-documented 20-file P11/P12 archive",
        "external_archive_matches_documented_sha256": archive_sha.casefold() == EXPECTED_EXTERNAL_SHA256,
        "external_integrity_test": "WinRAR UnRAR t passed",
        "external_member_count": len(member_rows),
        "external_member_sha256": source_hashes,
        "external_group_ids": list(EXTERNAL_GROUPS),
        "external_identity_independent_from_development_verified": False,
        "prior_exposure_or_selection_independence_verified": False,
        "independent_target_labels_available": False,
        "archive_has_only_signal_csv_files_no_reference_sidecar": True,
        "classification_accuracy_computed": False,
        "macro_f1_computed": False,
        "confusion_matrices_computed": False,
        "interpretation": "exploratory frozen model predictions only; class names describe agreement with the EMG-derived training proxy, not physiological fatigue",
        "development_group_ids": list(freeze_manifest.get("development_groups", [])),
        "development_label_rule": LABEL_RULE,
        "decision_thresholds": {
            "A1_fixed_comparator": fixed_cut,
            "A1_development_selected": a1_cut,
            "temporal_fixed_comparator": fixed_cut,
            "temporal_development_selected": time_cut,
        },
        "abstention_threshold": confidence_cut,
        "ood_policy": {
            "method": "development-calibrated robust MAD input-distance flag",
            "threshold": ood_cut,
            "quantile": prediction_policy["ood_quantile"],
            "interpretation": "distribution-shift flag only; no external OOD labels",
        },
        "noise_families": list(EXTERNAL_FAMILIES),
        "noise_levels": list(EXTERNAL_LEVELS),
        "noise_evaluation": "per-window matched post-preprocessing perturbations; predictions only because independent target labels are absent",
        "temporal_shift_labels_independently_verified": False,
        "model_fit_external": False,
        "calibration_fit_external": False,
        "threshold_tuning_external": False,
        "feature_normalization_fit_external": False,
        "domain_adaptation_external": False,
        "pretraining_external": False,
        "input_sha256": {
            str(path.resolve()): _sha256_file(path)
            for path in (
                external_archive,
                config_path,
                frozen / "freeze_manifest.json",
                frozen / "frozen_a1_morphology_logistic.joblib",
                temporal_manifest_path,
                temporal_dir / "frozen_temporal_baseline.joblib",
                internal_manifest_path,
            )
        },
        "source_sha256": {
            str(Path(module.__file__).resolve()): _sha256_file(Path(module.__file__).resolve())
            for module in (
                __import__(__name__, fromlist=["evaluation_external_unlabeled"]),
                evaluation,
                evaluation_internal,
                activity_module,
                augmentation_module,
                tabular_noise_module,
                features_module,
                io_module,
                label_audit_module,
                preprocess_module,
                pseudo_labels_module,
                windows_module,
            )
        },
        "software": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "archive_reader": str(unrar_exe),
        },
        "outputs_sha256": {},
    }
    manifest["outputs_sha256"] = {
        str(path.resolve()): _sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path != manifest_path
    }
    _write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "external_archive_sha256": archive_sha,
        "external_member_count": len(member_rows),
        "external_eligible_window_count_descriptive_only": int(len(scored)),
        "independent_target_labels_available": False,
        "classification_metrics_computed": False,
        "manifest_path": str(manifest_path.resolve()),
    }
