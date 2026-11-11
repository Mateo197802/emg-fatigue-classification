"""Frozen-model evaluation on the reserved local source groups.

The module deliberately separates a development-only temporal-baseline freeze
from the internal-evaluation function. The latter refuses to open the reserved
CSV payloads until both selected models and their thresholds have been frozen.
All labels remain EMG-derived proxy targets.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from . import evaluation
from . import activity as activity_module
from . import augmentation as augmentation_module
from . import evaluation_tabular_noise as tabular_noise_module
from . import features as features_module
from . import label_audit as label_audit_module
from . import preprocess as preprocess_module
from . import pseudo_labels as pseudo_labels_module
from . import windows as windows_module
from .activity import detect_candidate_activity
from .augmentation import CORRUPTION_FAMILIES, EVALUATION_LEVELS, corrupt_window, stable_seed
from .contracts import GroupSplit
from .development_freeze import _verify_outputs
from .evaluation_guardrails import OOD_FEATURES, _robust_ood_score
from .features import FEATURE_VIEWS, window_key
from .io import _parse_development_record
from .label_audit import _extract_rows_for_windows, _label_row
from .metrics import compute_metrics
from .preprocess import preprocess_recording
from .pseudo_labels import label_low_frequency_fraction_rule, technical_reference
from .evaluation_tabular_noise import _morphology_batch
from .windows import make_windows


LABEL_RULE = "low_frequency_fraction_persistence_v1"
A1_FEATURES = tuple(FEATURE_VIEWS["A1_morphology_only"])
METRIC_NAMES = (
    "accuracy",
    "sensitivity",
    "specificity",
    "precision_fatigue_like",
    "macro_f1",
    "balanced_accuracy",
    "matthews_correlation_coefficient",
    "cohen_kappa",
    "pr_auc_average_precision",
    "roc_auc",
    "brier_score",
    "selective_error",
)


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


def _verify_hashes(values: Mapping[str, str], *, name: str) -> None:
    for raw_path, expected in values.items():
        path = Path(raw_path)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"{name} checksum mismatch: {path}")


def _verify_freeze(private: Path, frozen_subdir: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if Path(frozen_subdir).name != frozen_subdir or frozen_subdir in {"", ".", ".."}:
        raise ValueError("frozen_subdir must be one directory name")
    frozen = private / frozen_subdir
    manifest_path = frozen / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("A completed development freeze is required before internal evaluation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("stage") != "development_only_selected_model_calibration_ood_abstention_freeze":
        raise ValueError("The selected development model has no completed freeze receipt")
    if int(manifest.get("locked_internal_test_values_read", -1)) != 0 or manifest.get("raw2_rar_opened") is not False:
        raise ValueError("The development freeze receipt does not prove both reserved sets stayed closed")
    _verify_outputs(manifest, "output_sha256")
    _verify_hashes(manifest.get("input_sha256", {}), name="development freeze input")
    _verify_hashes(manifest.get("source_sha256", {}), name="development freeze source")
    prediction_policy = json.loads((frozen / "frozen_prediction_policy.json").read_text(encoding="utf-8"))
    ood_policy = json.loads((frozen / "frozen_ood_policy.json").read_text(encoding="utf-8"))
    if prediction_policy.get("label_rule") != LABEL_RULE:
        raise ValueError("Frozen target differs from the accepted low-frequency proxy contract")
    checkpoint = frozen / "frozen_a1_morphology_logistic.joblib"
    expected = manifest.get("output_sha256", {}).get(str(checkpoint))
    if not expected or _sha256_file(checkpoint) != expected or _sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
        raise ValueError("Frozen A1 checkpoint checksum does not match its receipt")
    return frozen, manifest, prediction_policy, ood_policy


def freeze_temporal_baseline(
    run_dir: Path,
    *,
    frozen_subdir: str = "frozen_development_lowfreq_v1",
    a1_subdir: str = "a1_lowfreq_optimization_v1",
    output_subdir: str = "frozen_development_temporal_baseline_v1",
) -> dict[str, Any]:
    """Freeze the development-only elapsed-time baseline before opening test signals."""
    run_dir = Path(run_dir).expanduser().resolve()
    private = run_dir / "private"
    _verify_freeze(private, frozen_subdir)
    for value in (a1_subdir, output_subdir):
        if Path(value).name != value or value in {"", ".", ".."}:
            raise ValueError("stage subdirectories must be one path component")
    output = private / output_subdir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite temporal-baseline freeze: {output}")

    source = private / a1_subdir
    source_manifest_path = source / "a1_optimization_stage_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("Completed nested A1 development search is required")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("status") != "completed"
        or source_manifest.get("label_rule") != LABEL_RULE
        or int(source_manifest.get("outer_group_folds", -1)) != 4
        or int(source_manifest.get("inner_group_folds", -1)) != 3
        or int(source_manifest.get("internal_test_signal_values_parsed", -1)) != 0
        or source_manifest.get("external_rar_opened") is not False
    ):
        raise ValueError("Nested A1 parent does not satisfy the frozen 4x3 development contract")
    _verify_outputs(source_manifest, "outputs_sha256")

    split_path = private / "split_manifest.json"
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    dev_groups = tuple(str(value) for value in split_doc["development_group_ids"])
    test_groups = tuple(str(value) for value in split_doc["test_group_ids"])
    if len(dev_groups) != 12 or len(test_groups) != 3 or set(dev_groups) & set(test_groups):
        raise ValueError("The frozen local split is not the accepted 12/3 source-group partition")
    features_path = private / "features_labels" / "features.parquet"
    labels_path = private / "features_labels" / "pseudo_labels_alternatives.parquet"
    features = pd.read_parquet(features_path, filters=[("group_id", "in", list(dev_groups))])
    features = features.loc[np.isclose(features.duration_s.to_numpy(dtype=float), 0.5)].copy()
    labels = pd.read_parquet(labels_path, filters=[("group_id", "in", list(dev_groups))])
    labels = labels.loc[labels.comparison_rule.astype(str).eq(LABEL_RULE)].drop(columns=["comparison_rule"]).copy()
    if set(features.group_id.astype(str)) != set(dev_groups) or set(labels.group_id.astype(str)) != set(dev_groups):
        raise ValueError("Temporal-baseline inputs contain missing or reserved source groups")
    split = GroupSplit(
        seed=int(split_doc["seed"]),
        train_fraction=float(split_doc["train_fraction"]),
        development_groups=dev_groups,
        test_groups=test_groups,
    )
    frame = evaluation._prepare_windows_and_labels(
        features,
        labels,
        split,
        {"relative_time_only": ("relative_time_fraction",)},
    )
    active = frame.loc[frame.class_id.isin((0, 1))].sort_values("window_key").reset_index(drop=True)
    if active.window_key.astype(str).duplicated().any() or set(active.group_id.astype(str)) != set(dev_groups):
        raise ValueError("Temporal-baseline development observations are duplicated or incomplete")
    for class_id in (0, 1):
        selected = active.loc[active.class_id.eq(class_id)]
        if selected.empty or selected.group_id.nunique() < 2 or selected.recording_id.nunique() < 2:
            raise ValueError(f"Temporal-baseline development lacks independent source-group/trial support for class {class_id}")

    oof_path = source / "nested_oof_predictions.parquet"
    oof = pd.read_parquet(oof_path)
    oof = oof.loc[
        oof.feature_view.astype(str).eq("relative_time_only")
        & oof.model_name.astype(str).eq("logistic_regression")
        & oof.threshold_strategy.astype(str).eq("inner_oof_selected")
        & oof.true_class_id.isin((0, 1))
    ].copy()
    oof = oof.sort_values("window_key").reset_index(drop=True)
    if oof.window_key.astype(str).duplicated().any() or set(oof.window_key.astype(str)) != set(active.window_key.astype(str)):
        raise ValueError("Temporal-baseline development OOF does not cover the same assigned windows exactly once")
    aligned = active.set_index("window_key").loc[oof.window_key.astype(str)].reset_index()
    if not np.array_equal(aligned.class_id.to_numpy(dtype=int), oof.true_class_id.to_numpy(dtype=int)):
        raise ValueError("Temporal-baseline OOF labels differ from the accepted target table")
    if not np.array_equal(aligned.group_id.astype(str).to_numpy(), oof.group_id.astype(str).to_numpy()):
        raise ValueError("Temporal-baseline OOF groups differ from the accepted target table")
    threshold = evaluation.select_group_balanced_threshold(
        oof.true_class_id.to_numpy(dtype=int),
        oof.probability_class_1.to_numpy(dtype=float),
        oof.group_id.astype(str).tolist(),
    )

    receipts = pd.read_csv(source / "outer_model_receipts.csv")
    selected_receipts = receipts.loc[
        receipts.feature_view.astype(str).eq("relative_time_only")
        & receipts.model_name.astype(str).eq("logistic_regression")
        & receipts.fit_status.astype(str).eq("completed")
    ].sort_values("outer_fold_id")
    if len(selected_receipts) != 4 or not selected_receipts.selected_calibration_method.astype(str).eq("sigmoid").all():
        raise ValueError("Temporal baseline lacks four calibrated outer-fold selections")
    params_by_fold = [json.loads(value) for value in selected_receipts.selected_params_json.astype(str)]
    c_values = [float(params.get("C")) for params in params_by_fold]
    counts = Counter(c_values)
    max_count = max(counts.values())
    modal_c = min(value for value, count in counts.items() if count == max_count)
    params = {"C": modal_c}
    x = active[["relative_time_fraction"]].astype(float)
    y = active.class_id.to_numpy(dtype=np.int64)
    groups = active.group_id.astype(str).to_numpy()
    seed = 20261211
    model, fit_warnings, fit_seconds = evaluation._fit_estimator(
        "logistic_regression", params, x, y, groups.tolist(), seed, calibration_method="sigmoid"
    )

    output.mkdir(parents=True)
    checkpoint = output / "frozen_temporal_baseline.joblib"
    joblib.dump(model, checkpoint, compress=3)
    final_probs = evaluation._positive_probability(model, x)[:, 1]
    oof_predictions = oof[["window_key", "group_id", "recording_id", "true_class_id", "probability_class_1"]].copy()
    oof_predictions["fixed_0_5_prediction"] = (oof_predictions.probability_class_1 >= 0.5).astype(int)
    oof_predictions["development_oof_selected_prediction"] = (
        oof_predictions.probability_class_1 >= float(threshold.threshold)
    ).astype(int)
    oof_predictions.to_parquet(output / "development_temporal_oof_predictions.parquet", index=False)
    final_fit_predictions = pd.DataFrame(
        {
            "window_key": active.window_key.astype(str),
            "group_id": groups,
            "recording_id": active.recording_id.astype(str),
            "true_class_id": y,
            "in_sample_probability_class_1_descriptive_only": final_probs,
        }
    )
    final_fit_predictions.to_parquet(output / "development_full_fit_predictions_descriptive_only.parquet", index=False)

    manifest_path = output / "temporal_baseline_freeze_manifest.json"
    manifest: dict[str, Any] = {
        "stage": "development_only_temporal_baseline_calibration_threshold_freeze",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "time_only_logistic",
        "feature_view": "relative_time_only",
        "feature_names": ["relative_time_fraction"],
        "feature_definition": "(window start time - earliest window start time) / (latest window start time - earliest window start time) within recording, clipped to [0,1]",
        "label_rule": LABEL_RULE,
        "label_scope": "EMG-derived operational proxy only; not independent physiological fatigue evidence",
        "development_group_ids": list(dev_groups),
        "locked_internal_group_ids": list(test_groups),
        "locked_internal_test_values_read": 0,
        "raw2_rar_opened": False,
        "outer_group_folds": 4,
        "inner_group_folds": 3,
        "parameter_selection_source": "modal selected params in the four nested development outer-fold receipts",
        "selected_params": params,
        "outer_selected_C_values": c_values,
        "calibration_method": "group-disjoint sigmoid via three folds over development groups",
        "calibration_group_count": int(len(set(groups.tolist()))),
        "fixed_threshold_comparator": 0.5,
        "selected_threshold": float(threshold.threshold),
        "threshold_objective": "group-balanced Macro-F1 on assigned development outer-OOF temporal-only probabilities",
        "threshold_candidate_count": int(threshold.candidate_count),
        "threshold_tie_resolution": threshold.tie_resolution,
        "threshold_oof_group_count": int(oof.group_id.nunique()),
        "threshold_oof_window_count_descriptive_only": int(len(oof)),
        "full_development_fit_is_not_an_evaluation": True,
        "fit_seconds": float(fit_seconds),
        "fit_warnings": fit_warnings,
        "source_sha256": {
            str(Path(__file__).resolve()): _sha256_file(Path(__file__).resolve()),
            str(Path(evaluation.__file__).resolve()): _sha256_file(Path(evaluation.__file__).resolve()),
        },
        "input_sha256": {
            str(features_path): _sha256_file(features_path),
            str(labels_path): _sha256_file(labels_path),
            str(split_path): _sha256_file(split_path),
            str(source_manifest_path): _sha256_file(source_manifest_path),
            str(source / "nested_oof_predictions.parquet"): _sha256_file(oof_path),
            str(source / "outer_model_receipts.csv"): _sha256_file(source / "outer_model_receipts.csv"),
        },
        "checkpoint_sha256": _sha256_file(checkpoint),
        "outputs_sha256": {},
    }
    for cut, strategy in ((0.5, "fixed_0_5"), (float(threshold.threshold), "development_oof_selected")):
        pred = (oof.probability_class_1.to_numpy(dtype=float) >= cut).astype(int)
        record = compute_metrics(
            oof.true_class_id.to_numpy(dtype=int),
            oof.probability_class_1.to_numpy(dtype=float),
            pred,
            np.zeros(len(oof), dtype=bool),
        )
        manifest.setdefault("oof_metrics", {})[strategy] = record.to_dict()
    manifest["outputs_sha256"] = {
        str(path.resolve()): _sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path != manifest_path
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def _class_support(labels: pd.DataFrame, groups: Sequence[str]) -> dict[str, Any]:
    scoped = labels.loc[labels.group_id.astype(str).isin(set(map(str, groups)))].copy()
    support: dict[str, Any] = {
        "window_rows_descriptive_only": int(len(scoped)),
        "assigned_rows_descriptive_only": int(scoped.class_id.isin((0, 1)).sum()),
        "class_counts_descriptive_only": {str(key): int(value) for key, value in scoped.class_id.value_counts().sort_index().items()},
        "source_group_count": int(scoped.group_id.nunique()),
        "recording_trial_count": int(scoped.recording_id.nunique()),
        "class_support": {},
    }
    for class_id in (0, 1):
        rows = scoped.loc[scoped.class_id.eq(class_id)]
        support["class_support"][str(class_id)] = {
            "assigned_window_count_descriptive_only": int(len(rows)),
            "source_group_count": int(rows.group_id.nunique()),
            "recording_trial_count": int(rows.recording_id.nunique()),
        }
    return support


def _metric_row(
    y: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    abstain: np.ndarray,
    *,
    model: str,
    policy: str,
    condition: str,
    family: str,
    level: float,
    group_id: str | None = None,
    recording_id: str | None = None,
) -> dict[str, Any]:
    record = compute_metrics(y, probability, prediction, abstain)
    values = record.to_dict()
    p1 = values["per_class"]["1"]
    return {
        "model": model,
        "policy": policy,
        "condition": condition,
        "family": family,
        "requested_level": float(level),
        "stress_test_gt_20_percent": bool(level > 0.20),
        "group_id": group_id,
        "recording_id": recording_id,
        "both_classes_present_in_truth": bool(set(np.unique(y).tolist()) == {0, 1}),
        "class_0_support_descriptive_only": int(np.sum(y == 0)),
        "class_1_support_descriptive_only": int(np.sum(y == 1)),
        "n_samples_descriptive_only": int(len(y)),
        "n_predicted": int(values["n_predicted"]),
        "n_abstained": int(values["n_abstained"]),
        "coverage": values["coverage"],
        "abstention_rate": values["abstention_rate"],
        "accuracy": values["accuracy"],
        "sensitivity": values["sensitivity"],
        "specificity": values["specificity"],
        "precision_fatigue_like": p1["precision"],
        "recall_fatigue_like": p1["recall"],
        "macro_f1": values["macro_f1"],
        "balanced_accuracy": values["balanced_accuracy"],
        "matthews_correlation_coefficient": values["matthews_correlation_coefficient"],
        "cohen_kappa": values["cohen_kappa"],
        "pr_auc_average_precision": values["pr_auc_average_precision"],
        "roc_auc": values["roc_auc"],
        "brier_score": values["calibration"].get("brier_score"),
        "log_loss": values["log_loss"],
        "selective_error": values["selective_error"],
        "confusion_matrix_true_rows_0_1_pred_cols_0_1_abstain": json.dumps(values["confusion_matrix"], separators=(",", ":")),
        "confusion_matrix_normalized_by_true": json.dumps(values["confusion_matrix_normalized_by_true"], separators=(",", ":")),
        "undefined_reasons_json": json.dumps(values["undefined_reasons"], sort_keys=True, separators=(",", ":")),
    }


def _risk_coverage_curve(
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    groups: np.ndarray,
    fixed_guardrail_abstain: np.ndarray,
    *,
    model: str,
    condition: str,
    family: str,
    level: float,
    confidence_only: bool,
) -> tuple[list[dict[str, Any]], float]:
    confidence = np.maximum(probability, 1.0 - probability)
    base_prediction = (probability >= float(threshold)).astype(int)
    group_values = sorted(set(groups.astype(str).tolist()))
    candidates = np.unique(np.concatenate(([float(np.nextafter(1.0, 2.0))], confidence, [0.0])))
    rows: list[dict[str, Any]] = []
    for cut in np.sort(candidates)[::-1]:
        accepted = confidence >= cut
        if not confidence_only:
            accepted &= ~fixed_guardrail_abstain
        total_mass = 0.0
        error_mass = 0.0
        for group in group_values:
            selected = groups.astype(str) == group
            if not np.any(selected):
                continue
            weights = 1.0 / int(selected.sum())
            total_mass += float(np.sum(accepted[selected])) * weights / len(group_values)
            error_mass += float(np.sum(accepted[selected] & (base_prediction[selected] != truth[selected]))) * weights / len(group_values)
        risk = error_mass / total_mass if total_mass > 0.0 else 0.0
        rows.append(
            {
                "model": model,
                "condition": condition,
                "family": family,
                "requested_level": float(level),
                "selection_score": "max_probability_class",
                "curve": "confidence_only" if confidence_only else "confidence_plus_frozen_guardrails",
                "confidence_cutoff_diagnostic_only": float(cut),
                "group_balanced_coverage": float(total_mass),
                "group_balanced_selective_error": float(risk),
            }
        )
    curve = pd.DataFrame(rows).sort_values("group_balanced_coverage").drop_duplicates("group_balanced_coverage")
    aurc = float(np.trapezoid(curve.group_balanced_selective_error, curve.group_balanced_coverage)) if len(curve) > 1 else 0.0
    return rows, aurc


def _metric_lookup(metrics: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    grouped = metrics.loc[metrics.group_id.notna() & metrics.recording_id.isna()]
    for row in grouped.to_dict(orient="records"):
        result[(str(row["condition"]), str(row["model"]), str(row["policy"]), str(row["group_id"]))] = row
    return result


def _paired_group_intervals(metrics: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    lookup = _metric_lookup(metrics)
    conditions = sorted(set(metrics.condition.astype(str)))
    comparisons = (
        ("A1_morphology_logistic", "development_oof_selected", "time_only_logistic", "development_oof_selected", "selected_A1_minus_temporal_baseline"),
        ("A1_morphology_logistic", "development_oof_selected", "A1_morphology_logistic", "fixed_0_5", "selected_A1_minus_fixed_0_5"),
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for condition in conditions:
        for left_model, left_policy, right_model, right_policy, name in comparisons:
            groups = sorted(
                key[3]
                for key in lookup
                if key[0] == condition
                and key[1] == left_model
                and key[2] == left_policy
                and (condition, right_model, right_policy, key[3]) in lookup
            )
            for metric in METRIC_NAMES:
                values: list[tuple[float, float]] = []
                valid_groups: list[str] = []
                for group in groups:
                    a = lookup[(condition, left_model, left_policy, group)].get(metric)
                    b = lookup[(condition, right_model, right_policy, group)].get(metric)
                    if a is not None and b is not None and np.isfinite(float(a)) and np.isfinite(float(b)):
                        values.append((float(a), float(b)))
                        valid_groups.append(group)
                if not values:
                    rows.append(
                        {
                            "condition": condition,
                            "comparison": name,
                            "metric": metric,
                            "paired_source_group_count": 0,
                            "delta_left_minus_right": None,
                            "ci_95_low": None,
                            "ci_95_high": None,
                            "ci_status": "undefined_no_groups_with_valid_metric_support",
                        }
                    )
                    continue
                deltas = np.asarray([left - right for left, right in values], dtype=float)
                if len(deltas) >= 2:
                    draws = rng.integers(0, len(deltas), size=(int(replicates), len(deltas)))
                    bootstrap = deltas[draws].mean(axis=1)
                    low, high = np.quantile(bootstrap, [0.025, 0.975])
                    status = "paired_source_group_percentile_bootstrap; very_imprecise_with_fewer_than_5_groups"
                else:
                    low = high = None
                    status = "ci_not_computed_with_only_one_valid_source_group"
                rows.append(
                    {
                        "condition": condition,
                        "comparison": name,
                        "metric": metric,
                        "paired_source_group_count": len(deltas),
                        "paired_source_groups": json.dumps(valid_groups, separators=(",", ":")),
                        "mean_paired_group_delta": float(np.mean(deltas)),
                        "delta_left_minus_right": float(np.mean(deltas)),
                        "ci_95_low": float(low) if low is not None else None,
                        "ci_95_high": float(high) if high is not None else None,
                        "ci_status": status,
                        "bootstrap_replicates": int(replicates) if len(deltas) >= 2 else 0,
                        "bootstrap_unit": "source group; paired rows and labels; source identity is unverified",
                    }
                )
    return pd.DataFrame(rows)


def run_internal_evaluation(
    training_zip: Path,
    run_dir: Path,
    config_path: Path,
    *,
    frozen_subdir: str = "frozen_development_lowfreq_v1",
    temporal_subdir: str = "frozen_development_temporal_baseline_v1",
    output_subdir: str = "internal_lowfreq_frozen_evaluation_v1",
    bootstrap_replicates: int = 10000,
) -> dict[str, Any]:
    """Evaluate frozen selected A1 and temporal-control models on reserved groups.

    Only the 30 manifest-listed internal CSV payloads are read. No threshold,
    model, calibration parameter, or OOD/abstention policy is refit here.
    """
    training_zip = Path(training_zip).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    private = run_dir / "private"
    frozen, freeze_manifest, prediction_policy, ood_policy = _verify_freeze(private, frozen_subdir)
    temporal_dir = private / temporal_subdir
    temporal_manifest_path = temporal_dir / "temporal_baseline_freeze_manifest.json"
    if not temporal_manifest_path.is_file():
        raise FileNotFoundError("Freeze the temporal-only development baseline before opening the internal test")
    temporal_manifest = json.loads(temporal_manifest_path.read_text(encoding="utf-8"))
    if (
        temporal_manifest.get("status") != "completed"
        or int(temporal_manifest.get("locked_internal_test_values_read", -1)) != 0
        or temporal_manifest.get("raw2_rar_opened") is not False
        or temporal_manifest.get("label_rule") != LABEL_RULE
    ):
        raise ValueError("Temporal baseline is not safely frozen on development only")
    _verify_hashes(temporal_manifest.get("input_sha256", {}), name="temporal-baseline input")
    _verify_hashes(temporal_manifest.get("source_sha256", {}), name="temporal-baseline source")
    for output_path, expected in temporal_manifest.get("outputs_sha256", {}).items():
        if not Path(output_path).is_file() or _sha256_file(Path(output_path)) != expected:
            raise ValueError(f"Temporal-baseline artifact checksum mismatch: {output_path}")

    for value in (output_subdir,):
        if Path(value).name != value or value in {"", ".", ".."}:
            raise ValueError("output_subdir must be one path component")
    output = private / output_subdir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite internal evaluation stage: {output}")

    input_manifest_path = private / "input_manifest.json"
    split_path = private / "split_manifest.json"
    feature_label_manifest_path = private / "features_labels" / "feature_label_stage_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    feature_label_manifest = json.loads(feature_label_manifest_path.read_text(encoding="utf-8"))
    expected_archive_sha = str(input_manifest.get("input_archive_sha256", ""))
    if not expected_archive_sha or _sha256_file(training_zip) != expected_archive_sha:
        raise ValueError("Training ZIP checksum does not match the prepared private receipt")
    if str(input_manifest.get("input_archive_name")) != training_zip.name:
        raise ValueError("Training ZIP name differs from the prepared private receipt")
    if (
        feature_label_manifest.get("config_sha256") != _sha256_file(config_path)
        or feature_label_manifest.get("training_archive_sha256") != expected_archive_sha
        or feature_label_manifest.get("label_scope") != "EMG-derived operational proxy only; not clinical, perceived, or mechanically verified fatigue"
    ):
        raise ValueError("Internal preprocessing/label configuration differs from the development label receipt")
    dev_groups = tuple(str(value) for value in split_doc["development_group_ids"])
    test_groups = tuple(str(value) for value in split_doc["test_group_ids"])
    if (
        tuple(temporal_manifest.get("development_group_ids", [])) != dev_groups
        or tuple(temporal_manifest.get("locked_internal_group_ids", [])) != test_groups
        or tuple(freeze_manifest.get("development_groups", [])) != dev_groups
        or len(test_groups) != 3
    ):
        raise ValueError("Frozen model, temporal control, and internal partition groups disagree")
    if input_manifest.get("unparsed_internal_test_member_count") != 30 or split_doc.get("unparsed_test_member_count") != 30:
        raise ValueError("Prepared split no longer records exactly 30 unopened internal CSV files")

    expected_test = {str(row["member_name"]): row for row in split_doc["test_members"]}
    source_members = {str(row["member_name"]): row for row in input_manifest["members"]}
    if len(expected_test) != 30 or set(expected_test) != {
        str(row["member_name"]) for row in input_manifest["members"] if str(row["group_id"]) in set(test_groups)
    }:
        raise ValueError("Test-member allowlist is not an exact match for the reserved groups")
    for name, row in expected_test.items():
        receipt = source_members.get(name)
        if (
            receipt is None
            or str(receipt["group_id"]) != str(row["group_id"])
            or str(receipt["recording_id"]) != str(row["recording_id"])
            or str(receipt["sha256"]) != str(row["sha256"])
            or bool(receipt.get("parsed_numeric_values"))
            or bool(row.get("numeric_values_parsed"))
        ):
            raise ValueError(f"Internal member metadata or unopened-state receipt mismatch: {name}")

    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not all(
        name in config.get("labels", {})
        for name in (
            "reference_window_count",
            "reference_instability_percent",
            "minimum_scaled_mad_multiple",
            "minimum_low_frequency_fraction_increase_when_mad_zero",
            "entry_window_count",
            "entry_rolling_window_count",
            "minimum_entry_span_s",
            "recovery_consecutive_window_count",
        )
    ):
        raise ValueError("Frozen development configuration lacks parameters for the accepted low-frequency proxy rule")
    primary_duration = float(config["run"]["primary_window_s"])
    overlap = float(config["run"]["window_overlap"])
    occupancy = float(config["activity"]["primary_window_occupancy"])
    line_exclusion = float(config["labels"]["line_exclusion_hz"])
    output.mkdir(parents=True)
    all_feature_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    all_member_rows: list[dict[str, Any]] = []
    signal_by_key: dict[str, np.ndarray] = {}
    rate_by_key: dict[str, float] = {}
    quality_by_key: dict[str, bool] = {}
    source_hashes: dict[str, str] = {}
    total_parsed_numeric_rows = 0

    # This is the first access to numeric values in the reserved test CSVs.
    with zipfile.ZipFile(training_zip, "r") as archive:
        zip_names = set(archive.namelist())
        missing_members = set(expected_test) - zip_names
        if missing_members:
            raise ValueError(f"The authorized ZIP is missing reserved test members: {sorted(missing_members)[:3]}")
        for name in sorted(expected_test):
            payload = archive.read(name)
            receipt = expected_test[name]
            digest = hashlib.sha256(payload).hexdigest()
            if digest != str(receipt["sha256"]):
                raise ValueError(f"Reserved test member checksum mismatch: {name}")
            if len(payload) != int(source_members[name]["byte_size"]):
                raise ValueError(f"Reserved test member byte size mismatch: {name}")
            source_hashes[name] = digest
            recording, row_count = _parse_development_record(
                payload,
                str(receipt["group_id"]),
                str(receipt["recording_id"]),
                digest,
                name,
            )
            total_parsed_numeric_rows += int(row_count)
            processed = preprocess_recording(recording, config)
            activity = detect_candidate_activity(processed, config)
            primary_windows = make_windows(processed, activity, primary_duration, overlap, minimum_occupancy=occupancy)
            reference_windows = make_windows(processed, activity, primary_duration, 0.0, minimum_occupancy=occupancy)
            feature_rows, feature_map = _extract_rows_for_windows(
                primary_windows,
                processed.audit.notch_frequency_hz,
                line_exclusion,
            )
            all_feature_rows.extend(feature_rows)
            windows_by_segment: dict[int, list[Any]] = {}
            refs_by_segment: dict[int, list[Any]] = {}
            for item in primary_windows:
                windows_by_segment.setdefault(int(item.segment_index), []).append(item)
                signal_by_key[window_key(item)] = np.asarray(item.signal, dtype=np.float32)
                rate_by_key[window_key(item)] = float(item.sampling_rate_hz)
                segment = next((entry for entry in processed.segments if entry.segment_index == item.segment_index), None)
                quality_by_key[window_key(item)] = bool(
                    item.eligible_for_label
                    and segment is not None
                    and segment.acquisition_compatible
                )
            for item in reference_windows:
                refs_by_segment.setdefault(int(item.segment_index), []).append(item)
            for segment in processed.segments:
                segment_windows = windows_by_segment.get(int(segment.segment_index), [])
                if not segment_windows:
                    continue
                reference = technical_reference(
                    refs_by_segment.get(int(segment.segment_index), []),
                    feature_map,
                    config,
                    stability_components=("low_fraction",),
                )
                labels = label_low_frequency_fraction_rule(segment_windows, feature_map, reference, config)
                all_label_rows.extend(_label_row(label) for label in labels)
            all_member_rows.append(
                {
                    "group_id": str(receipt["group_id"]),
                    "recording_id": str(receipt["recording_id"]),
                    "member_name_private": name,
                    "member_sha256": digest,
                    "byte_size": len(payload),
                    "numeric_row_count": int(row_count),
                    "numeric_values_parsed": True,
                    "sampling_rate_hz": float(processed.audit.sampling_rate_hz),
                    "signal_compatible": bool(processed.audit.signal_compatible),
                    "continuous_segment_count": int(len(processed.segments)),
                    "window_count_descriptive_only": int(len(primary_windows)),
                    "notch_frequency_hz": processed.audit.notch_frequency_hz,
                    "quality_flags": json.dumps(list(processed.audit.quality_flags), separators=(",", ":")),
                }
            )

    features = pd.DataFrame(all_feature_rows)
    labels = pd.DataFrame(all_label_rows)
    if features.empty or labels.empty or features.window_key.astype(str).duplicated().any() or labels.window_key.astype(str).duplicated().any():
        raise ValueError("Internal feature/label materialization is empty or contains duplicate primary windows")
    if set(features.group_id.astype(str)) - set(test_groups) or set(labels.group_id.astype(str)) - set(test_groups):
        raise ValueError("Internal test features or targets include a group outside the reserved partition")
    if not labels.rule_version.astype(str).eq(LABEL_RULE).all():
        raise ValueError("Internal labels do not match the accepted operational proxy version")

    # Model features and time baseline are computed over all windows in each recording,
    # then metrics are restricted to assigned targets; ambiguous labels stay explicit.
    features["relative_time_fraction"] = evaluation._relative_time_fraction(features)
    merged = features.merge(
        labels[["window_key", "group_id", "recording_id", "class_id", "label_confidence", "label_reason", "state_transition"]],
        on="window_key",
        how="inner",
        suffixes=("", "_target"),
        validate="one_to_one",
    )
    if len(merged) != len(labels) or not merged.group_id.astype(str).eq(merged.group_id_target.astype(str)).all():
        raise ValueError("Internal feature/target join is incomplete or has mismatched groups")
    support = _class_support(labels, test_groups)
    support_rows: list[dict[str, Any]] = []
    for group in test_groups:
        group_labels = labels.loc[labels.group_id.astype(str).eq(group)]
        counts = group_labels.class_id.value_counts().to_dict()
        support_rows.append(
            {
                "group_id": group,
                "window_rows_descriptive_only": int(len(group_labels)),
                "ambiguous_windows_descriptive_only": int((group_labels.class_id == -1).sum()),
                "class_0_windows_descriptive_only": int(counts.get(0, 0)),
                "class_1_windows_descriptive_only": int(counts.get(1, 0)),
                "has_both_assigned_classes": bool(counts.get(0, 0) > 0 and counts.get(1, 0) > 0),
                "assigned_recording_trials": int(group_labels.loc[group_labels.class_id.isin((0, 1)), "recording_id"].nunique()),
                "window_counts_are_independent_n": False,
            }
        )
    trial_support_rows = []
    for (group, recording), trial in labels.groupby(["group_id", "recording_id"], sort=True):
        counts = trial.class_id.value_counts().to_dict()
        trial_support_rows.append(
            {
                "group_id": str(group),
                "recording_id": str(recording),
                "window_rows_descriptive_only": int(len(trial)),
                "ambiguous_windows_descriptive_only": int((trial.class_id == -1).sum()),
                "class_0_windows_descriptive_only": int(counts.get(0, 0)),
                "class_1_windows_descriptive_only": int(counts.get(1, 0)),
                "has_both_assigned_classes": bool(counts.get(0, 0) > 0 and counts.get(1, 0) > 0),
                "window_counts_are_independent_n": False,
            }
        )

    assigned = merged.loc[merged.class_id.isin((0, 1))].sort_values("window_key").reset_index(drop=True)
    if assigned.empty:
        raise ValueError("The frozen internal split has no assigned targets under the accepted proxy rule")
    if not assigned.window_key.astype(str).isin(signal_by_key).all():
        raise ValueError("An assigned target has no corresponding processed signal window")
    x_a1 = assigned.loc[:, list(A1_FEATURES)].astype(float)
    if not np.isfinite(x_a1.to_numpy(dtype=float)).all():
        raise ValueError("Assigned internal targets include invalid A1 model features; metrics will not be forced")
    quality_valid = np.asarray([quality_by_key.get(str(key), False) for key in assigned.window_key], dtype=bool)
    if not quality_valid.all():
        raise ValueError("Assigned internal targets failed the frozen acquisition/quality gate")
    y = assigned.class_id.to_numpy(dtype=np.int64)
    groups = assigned.group_id.astype(str).to_numpy()
    recording_ids = assigned.recording_id.astype(str).to_numpy()
    windows = assigned.window_key.astype(str).to_numpy()
    signals = np.stack([signal_by_key[str(key)] for key in windows]).astype(np.float32)
    rates = np.asarray([rate_by_key[str(key)] for key in windows], dtype=float)
    if len(set(np.round(rates, 6).tolist())) != 1:
        raise ValueError("Internal assigned windows have mixed sampling rates; noise pipeline requires a rate per row")
    x_time = assigned[["relative_time_fraction"]].astype(float)
    a1_package = joblib.load(frozen / "frozen_a1_morphology_logistic.joblib")
    if not isinstance(a1_package, dict) or "estimator" not in a1_package:
        raise ValueError("Frozen A1 checkpoint package does not contain its fitted estimator")
    if (
        a1_package.get("label_rule") != LABEL_RULE
        or tuple(a1_package.get("feature_names", ())) != A1_FEATURES
        or not np.isclose(float(a1_package.get("decision_threshold", np.nan)), float(prediction_policy["selected_decision_threshold"]))
    ):
        raise ValueError("Frozen A1 checkpoint metadata disagrees with the development decision policy")
    a1_model = a1_package["estimator"]
    temporal_model = joblib.load(temporal_dir / "frozen_temporal_baseline.joblib")
    probability_a1_clean = evaluation._positive_probability(a1_model, x_a1)[:, 1]
    probability_time = evaluation._positive_probability(temporal_model, x_time)[:, 1]
    a1_cut = float(prediction_policy["selected_decision_threshold"])
    time_cut = float(temporal_manifest["selected_threshold"])
    fixed_cut = float(prediction_policy["fixed_threshold_comparator"])
    confidence_cut = float(prediction_policy["abstention"]["confidence_threshold"])
    ood_center = np.asarray(ood_policy["center"], dtype=float)
    ood_scale = np.asarray(ood_policy["scale"], dtype=float)
    ood_keep = np.asarray(ood_policy["keep"], dtype=bool)
    ood_cut = float(ood_policy["threshold"])
    support_names = tuple(str(name) for name in ood_policy["feature_support_names"])
    if len(support_names) != len(ood_policy["feature_support_minimum"]) or len(support_names) != len(ood_policy["feature_support_maximum"]):
        raise ValueError("Frozen feature-support names and bounds have inconsistent dimensions")
    if any(name not in A1_FEATURES for name in support_names):
        raise ValueError("Frozen feature-support policy requests an unknown A1 feature")
    support_indices = np.asarray([A1_FEATURES.index(name) for name in support_names], dtype=int)
    support_min = np.asarray(ood_policy["feature_support_minimum"], dtype=float)
    support_max = np.asarray(ood_policy["feature_support_maximum"], dtype=float)

    all_prediction_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_ood_rows: list[dict[str, Any]] = []
    all_noise_receipts: list[dict[str, Any]] = []
    all_risk_rows: list[dict[str, Any]] = []
    aurc_lookup: dict[tuple[str, str], float] = {}
    conditions: list[tuple[str, float, pd.DataFrame, np.ndarray | None]] = [("clean", 0.0, x_a1.reset_index(drop=True), None)]

    for family in CORRUPTION_FAMILIES:
        for level in EVALUATION_LEVELS:
            corrupted = np.empty_like(signals, dtype=np.float32)
            receipts: list[dict[str, Any]] = []
            for position, key in enumerate(windows):
                seed = stable_seed("emg-matched-evaluation-noise-v1", str(key), family, float(level))
                altered, receipt = corrupt_window(
                    signals[position],
                    family,
                    float(level),
                    seed,
                    sampling_rate_hz=float(rates[position]),
                )
                corrupted[position] = altered
                receipts.append(receipt)
            noise_features = _morphology_batch(corrupted, float(rates[0]))
            missing = set(A1_FEATURES) - set(noise_features.columns)
            if missing:
                raise ValueError(f"Vectorized noise feature recomputation is missing model inputs: {sorted(missing)}")
            x_noise = noise_features.loc[:, list(A1_FEATURES)].astype(float)
            if not np.isfinite(x_noise.to_numpy(dtype=float)).all():
                raise ValueError(f"Noise condition {family}/{level} generated invalid A1 features")
            conditions.append((family, float(level), x_noise, np.asarray([r["achieved_rms_ratio"] for r in receipts], dtype=float)))
            for key, group, recording, receipt in zip(windows, groups, recording_ids, receipts, strict=True):
                all_noise_receipts.append(
                    {
                        "window_key": str(key),
                        "group_id": str(group),
                        "recording_id": str(recording),
                        "family": family,
                        "requested_level": float(level),
                        "stress_test_gt_20_percent": bool(level > 0.20),
                        "corruption_seed": int(receipt["seed"]),
                        "achieved_rms_ratio": float(receipt["achieved_rms_ratio"]),
                        "clean_rms": float(receipt["clean_rms"]),
                        "delta_rms": float(receipt["delta_rms"]),
                        "sampling_rate_hz": float(receipt["sampling_rate_hz"]),
                        "level_semantics": str(receipt["level_semantics"]),
                        "label_preserving_assumption": True,
                        "label_preserving_independently_verified": False,
                    }
                )

    for family, level, condition_x, achieved in conditions:
        condition_name = "clean:0" if family == "clean" else f"{family}:{level:.2f}"
        if family == "clean":
            probability_a1 = probability_a1_clean
        else:
            probability_a1 = evaluation._positive_probability(a1_model, condition_x)[:, 1]
        probability_time_condition = probability_time
        if family == "clean":
            morphology = condition_x.to_numpy(dtype=float)
        else:
            morphology = condition_x.to_numpy(dtype=float)
        ood_scores = _robust_ood_score(morphology[:, [list(A1_FEATURES).index(name) for name in OOD_FEATURES]], ood_center, ood_scale, ood_keep)
        ood_flags = ood_scores > ood_cut
        support_values = morphology[:, support_indices]
        outside = np.any((support_values < support_min) | (support_values > support_max), axis=1)
        confidence = np.maximum(probability_a1, 1.0 - probability_a1)
        pred_fixed = (probability_a1 >= fixed_cut).astype(int)
        pred_selected = (probability_a1 >= a1_cut).astype(int)
        conf_abstain = confidence < confidence_cut
        ood_abstain = ood_flags
        combined_abstain = conf_abstain | ood_abstain | outside | ~quality_valid
        a1_policies = (
            ("fixed_0_5", pred_fixed, np.zeros(len(y), dtype=bool), fixed_cut),
            ("development_oof_selected", pred_selected, np.zeros(len(y), dtype=bool), a1_cut),
            ("confidence_abstention", pred_selected, conf_abstain, a1_cut),
            ("ood_abstention", pred_selected, ood_abstain, a1_cut),
            ("combined_abstention", pred_selected, combined_abstain, a1_cut),
        )
        for policy_name, raw_prediction, abstain, cut in a1_policies:
            prediction = np.where(abstain, -1, raw_prediction).astype(int)
            for position in range(len(y)):
                all_prediction_rows.append(
                    {
                        "window_key": str(windows[position]),
                        "group_id": str(groups[position]),
                        "recording_id": str(recording_ids[position]),
                        "true_class_id": int(y[position]),
                        "model": "A1_morphology_logistic",
                        "policy": policy_name,
                        "condition": condition_name,
                        "family": family,
                        "requested_level": float(level),
                        "stress_test_gt_20_percent": bool(level > 0.20),
                        "probability_class_1": float(probability_a1[position]),
                        "prediction_class_id": int(prediction[position]),
                        "abstain": bool(abstain[position]),
                        "decision_threshold": float(cut),
                        "confidence": float(confidence[position]),
                        "confidence_threshold_frozen": confidence_cut,
                        "ood_score": float(ood_scores[position]),
                        "ood_threshold_frozen": ood_cut,
                        "ood_flag": bool(ood_flags[position]),
                        "outside_development_feature_support": bool(outside[position]),
                        "quality_valid": bool(quality_valid[position]),
                        "achieved_rms_ratio": float(achieved[position]) if achieved is not None else 0.0,
                    }
                )
            metrics_row = _metric_row(
                y,
                probability_a1,
                prediction,
                abstain,
                model="A1_morphology_logistic",
                policy=policy_name,
                condition=condition_name,
                family=family,
                level=level,
            )
            all_metric_rows.append(metrics_row)
            for group in sorted(set(groups.tolist())):
                mask = groups == group
                all_metric_rows.append(
                    _metric_row(
                        y[mask], probability_a1[mask], prediction[mask], abstain[mask],
                        model="A1_morphology_logistic", policy=policy_name, condition=condition_name,
                        family=family, level=level, group_id=str(group),
                    )
                )
            for recording in sorted(set(recording_ids.tolist())):
                mask = recording_ids == recording
                all_metric_rows.append(
                    _metric_row(
                        y[mask], probability_a1[mask], prediction[mask], abstain[mask],
                        model="A1_morphology_logistic", policy=policy_name, condition=condition_name,
                        family=family, level=level, group_id=str(groups[np.flatnonzero(mask)[0]]),
                        recording_id=str(recording),
                    )
                )
        time_policies = (
            ("fixed_0_5", (probability_time_condition >= fixed_cut).astype(int), fixed_cut),
            ("development_oof_selected", (probability_time_condition >= time_cut).astype(int), time_cut),
        )
        for policy_name, prediction, cut in time_policies:
            abstain = np.zeros(len(y), dtype=bool)
            for position in range(len(y)):
                all_prediction_rows.append(
                    {
                        "window_key": str(windows[position]),
                        "group_id": str(groups[position]),
                        "recording_id": str(recording_ids[position]),
                        "true_class_id": int(y[position]),
                        "model": "time_only_logistic",
                        "policy": policy_name,
                        "condition": condition_name,
                        "family": family,
                        "requested_level": float(level),
                        "stress_test_gt_20_percent": bool(level > 0.20),
                        "probability_class_1": float(probability_time_condition[position]),
                        "prediction_class_id": int(prediction[position]),
                        "abstain": False,
                        "decision_threshold": float(cut),
                        "confidence": float(max(probability_time_condition[position], 1.0 - probability_time_condition[position])),
                        "confidence_threshold_frozen": None,
                        "ood_score": None,
                        "ood_threshold_frozen": None,
                        "ood_flag": None,
                        "outside_development_feature_support": None,
                        "quality_valid": True,
                        "achieved_rms_ratio": float(achieved[position]) if achieved is not None else 0.0,
                    }
                )
            all_metric_rows.append(
                _metric_row(
                    y, probability_time_condition, prediction, abstain,
                    model="time_only_logistic", policy=policy_name, condition=condition_name,
                    family=family, level=level,
                )
            )
            for group in sorted(set(groups.tolist())):
                mask = groups == group
                all_metric_rows.append(
                    _metric_row(
                        y[mask], probability_time_condition[mask], prediction[mask], abstain[mask],
                        model="time_only_logistic", policy=policy_name, condition=condition_name,
                        family=family, level=level, group_id=str(group),
                    )
                )
            for recording in sorted(set(recording_ids.tolist())):
                mask = recording_ids == recording
                all_metric_rows.append(
                    _metric_row(
                        y[mask], probability_time_condition[mask], prediction[mask], abstain[mask],
                        model="time_only_logistic", policy=policy_name, condition=condition_name,
                        family=family, level=level, group_id=str(groups[np.flatnonzero(mask)[0]]),
                        recording_id=str(recording),
                    )
                )

        for model_name, probability, threshold, flag, family_name, requested in (
            ("A1_morphology_logistic", probability_a1, a1_cut, (ood_flags | outside | ~quality_valid), family, level),
        ):
            risk_rows, aurc_conf = _risk_coverage_curve(
                y, probability, threshold, groups, np.zeros(len(y), dtype=bool),
                model=model_name, condition=condition_name, family=family_name, level=requested,
                confidence_only=True,
            )
            all_risk_rows.extend(risk_rows)
            risk_rows, aurc_combined = _risk_coverage_curve(
                y, probability, threshold, groups, flag,
                model=model_name, condition=condition_name, family=family_name, level=requested,
                confidence_only=False,
            )
            all_risk_rows.extend(risk_rows)
            aurc_lookup[(condition_name, "confidence_only")] = aurc_conf
            aurc_lookup[(condition_name, "confidence_plus_frozen_guardrails")] = aurc_combined
        for class_id in (0, 1):
            class_mask = y == class_id
            for policy_name, abstain in (
                ("confidence_abstention", conf_abstain),
                ("ood_abstention", ood_abstain),
                ("combined_abstention", combined_abstain),
            ):
                n = int(class_mask.sum())
                accepted = int(np.sum(class_mask & ~abstain))
                all_ood_rows.append(
                    {
                        "condition": condition_name,
                        "family": family,
                        "requested_level": float(level),
                        "policy": policy_name,
                        "class_id": int(class_id),
                        "class_name": "FATIGUE_LIKE" if class_id == 1 else "NON_FATIGUE_LIKE",
                        "label_support_descriptive_only": n,
                        "accepted_count": accepted,
                        "class_coverage": accepted / n if n else None,
                        "ood_flag_rate": float(np.mean(ood_flags[class_mask])) if n else None,
                        "ood_detector_ground_truth_available": False,
                        "confidence_is_ood_score": False,
                    }
                )

    prediction_frame = pd.DataFrame(all_prediction_rows)
    metric_frame = pd.DataFrame(all_metric_rows)
    group_metric_frame = metric_frame.loc[metric_frame.group_id.notna() & metric_frame.recording_id.isna()].copy()
    trial_metric_frame = metric_frame.loc[metric_frame.recording_id.notna()].copy()
    for row in metric_frame.index[metric_frame.group_id.isna() & metric_frame.recording_id.isna()]:
        key = (str(metric_frame.at[row, "condition"]), str(metric_frame.at[row, "policy"]))
        metric_frame.at[row, "group_balanced_aurc_confidence"] = (
            aurc_lookup.get((key[0], "confidence_only"))
            if str(metric_frame.at[row, "model"]) == "A1_morphology_logistic"
            and str(metric_frame.at[row, "policy"]) == "confidence_abstention"
            else None
        )
        metric_frame.at[row, "group_balanced_aurc_confidence_plus_guardrails"] = (
            aurc_lookup.get((key[0], "confidence_plus_frozen_guardrails"))
            if str(metric_frame.at[row, "model"]) == "A1_morphology_logistic"
            and str(metric_frame.at[row, "policy"]) == "combined_abstention"
            else None
        )
    paired_frame = _paired_group_intervals(metric_frame, replicates=bootstrap_replicates, seed=20261213)

    features.to_parquet(output / "internal_test_window_features.parquet", index=False)
    labels.to_parquet(output / "internal_test_window_proxy_labels.parquet", index=False)
    prediction_frame.to_parquet(output / "internal_test_predictions.parquet", index=False)
    metric_frame.to_csv(output / "metrics_by_condition_group_trial.csv", index=False)
    group_metric_frame.to_csv(output / "metrics_by_group.csv", index=False)
    trial_metric_frame.to_csv(output / "metrics_by_trial.csv", index=False)
    paired_frame.to_csv(output / "paired_group_bootstrap.csv", index=False)
    pd.DataFrame(support_rows).to_csv(output / "label_support_by_group.csv", index=False)
    pd.DataFrame(trial_support_rows).to_csv(output / "label_support_by_trial.csv", index=False)
    pd.DataFrame(all_member_rows).to_csv(output / "reserved_member_processing_receipts.csv", index=False)
    pd.DataFrame(all_noise_receipts).to_parquet(output / "matched_noise_receipts.parquet", index=False)
    pd.DataFrame(all_ood_rows).to_csv(output / "class_coverage_and_ood_flags.csv", index=False)
    pd.DataFrame(all_risk_rows).to_csv(output / "risk_coverage_curves.csv", index=False)

    manifest_path = output / "internal_evaluation_manifest.json"
    manifest: dict[str, Any] = {
        "stage": "frozen_model_internal_80_20_evaluation_and_noise_robustness",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_archive_name": training_zip.name,
        "input_archive_sha256": expected_archive_sha,
        "input_config_sha256": _sha256_file(config_path),
        "input_sha256": {
            str(path.resolve()): _sha256_file(path)
            for path in (
                training_zip,
                config_path,
                input_manifest_path,
                split_path,
                feature_label_manifest_path,
                frozen / "frozen_a1_morphology_logistic.joblib",
                temporal_dir / "frozen_temporal_baseline.joblib",
            )
        },
        "development_freeze_manifest_sha256": _sha256_file(frozen / "freeze_manifest.json"),
        "temporal_baseline_freeze_manifest_sha256": _sha256_file(temporal_manifest_path),
        "frozen_a1_checkpoint_sha256": _sha256_file(frozen / "frozen_a1_morphology_logistic.joblib"),
        "frozen_temporal_checkpoint_sha256": _sha256_file(temporal_dir / "frozen_temporal_baseline.joblib"),
        "label_rule": LABEL_RULE,
        "label_scope": "EMG-derived weak-supervision proxy agreement only; not independent physiological fatigue validation",
        "development_group_ids": list(dev_groups),
        "internal_test_group_ids": list(test_groups),
        "internal_test_source_group_count": len(test_groups),
        "source_groups_are_verified_individuals": False,
        "internal_test_member_count": len(all_member_rows),
        "internal_test_member_sha256": source_hashes,
        "internal_test_numeric_values_parsed": len(all_member_rows),
        "internal_numeric_csv_rows_parsed": int(total_parsed_numeric_rows),
        "assigned_internal_window_count_descriptive_only": int(len(assigned)),
        "ambiguous_internal_window_count_descriptive_only": int((labels.class_id == -1).sum()),
        "window_counts_are_independent_n": False,
        "independent_unit_available": "three source groups; actual participant identity remains unverified",
        "test_class_support": support,
        "all_models_thresholds_calibration_ood_and_abstention_were_frozen_before_internal_values_were_read": True,
        "a1_thresholds": {"fixed_comparator": fixed_cut, "development_selected": a1_cut},
        "temporal_thresholds": {"fixed_comparator": fixed_cut, "development_selected": time_cut},
        "abstention_policy": prediction_policy["abstention"],
        "ood_policy": {
            "detector": prediction_policy["ood_detector"],
            "threshold": ood_cut,
            "quantile": prediction_policy["ood_quantile"],
            "interpretation": "input-distribution flag only; there are no independent OOD labels",
        },
        "noise_families": list(CORRUPTION_FAMILIES),
        "noise_levels": list(EVALUATION_LEVELS),
        "noise_definition": "deterministic matched post-preprocessing window corruption; additive levels use RMS(delta)/RMS(clean), temporal shift reports achieved RMS and is only assumed label-preserving",
        "noise_seed_contract": "sha256-based stable seed from window_key, family, requested level; identical across evaluated models",
        "internal_test_model_fitting": False,
        "internal_test_threshold_tuning": False,
        "internal_test_calibration_fitting": False,
        "internal_test_ood_or_abstention_policy_tuning": False,
        "raw2_rar_opened": False,
        "paired_confidence_intervals": "paired bootstrap resampling of source groups; only three source groups, therefore very imprecise",
        "source_sha256": {
            str(Path(module.__file__).resolve()): _sha256_file(Path(module.__file__).resolve())
            for module in (
                __import__(__name__, fromlist=["evaluation_internal"]),
                evaluation,
                activity_module,
                augmentation_module,
                tabular_noise_module,
                features_module,
                label_audit_module,
                preprocess_module,
                pseudo_labels_module,
                windows_module,
            )
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
        "label_rule": LABEL_RULE,
        "internal_test_group_ids": list(test_groups),
        "internal_test_member_count": len(all_member_rows),
        "assigned_internal_window_count_descriptive_only": int(len(assigned)),
        "test_class_support": support,
        "raw2_rar_opened": False,
        "manifest_path": str(manifest_path.resolve()),
    }
