"""Freeze the selected development-only low-frequency proxy classifier."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .evaluation import (
    _fit_estimator,
    _fold_plan_from_manifest,
    _positive_probability,
    _prepare_windows_and_labels,
    group_calibration_splits,
    select_group_balanced_threshold,
)
from .evaluation_guardrails import OOD_FEATURES, _robust_location_scale, _robust_ood_score
from .evaluation_neural_nested import _sha256_file, _weighted_confusion_metrics
from .features import FEATURE_VIEWS
from .contracts import GroupSplit
from .metrics import compute_metrics


LABEL_RULE = "low_frequency_fraction_persistence_v1"
LABEL_TYPE = "weak_supervision_proxy_not_independent_fatigue_ground_truth"
OOD_QUANTILE = 0.995
CONFIDENCE_THRESHOLD = 0.55


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str), encoding="utf-8")


def _verify_outputs(manifest: dict[str, Any], key: str) -> None:
    failures: list[str] = []
    for raw_path, expected in manifest.get(key, {}).items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != str(expected):
            failures.append(str(path))
    if failures:
        raise ValueError(f"parent stage {key} checksum failures: {failures[:8]}")


def _support(frame: pd.DataFrame, role: str, *, minimum_groups: int = 2, minimum_trials: int = 2) -> None:
    if set(frame.class_id.astype(int).unique()) != {0, 1}:
        raise ValueError(f"{role} lacks both primary proxy classes")
    for class_id in (0, 1):
        selected = frame.loc[frame.class_id.astype(int).eq(class_id)]
        groups = int(selected.group_id.astype(str).nunique())
        trials = int(selected.recording_id.astype(str).nunique())
        if groups < int(minimum_groups) or trials < int(minimum_trials):
            raise ValueError(
                f"{role} class {class_id} support is insufficient: groups={groups}, trials={trials}"
            )


def _group_confusions(y: np.ndarray, predicted: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for group in sorted(set(groups.astype(str).tolist())):
        selected = groups.astype(str) == group
        if set(np.unique(y[selected]).tolist()) != {0, 1}:
            raise ValueError(f"group {group} lacks both classes for a group-balanced comparison")
        matrix = confusion_matrix(y[selected], predicted[selected], labels=(0, 1)).astype(np.float64)
        result[group] = matrix / matrix.sum()
    return result


def _paired_group_bootstrap(
    left_y: np.ndarray,
    left_prediction: np.ndarray,
    left_groups: np.ndarray,
    right_y: np.ndarray,
    right_prediction: np.ndarray,
    right_groups: np.ndarray,
    *,
    comparison: str,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not np.array_equal(left_y, right_y) or not np.array_equal(left_groups.astype(str), right_groups.astype(str)):
        raise ValueError("paired comparison rows do not use identical labels and groups")
    left = _group_confusions(left_y, left_prediction, left_groups)
    right = _group_confusions(right_y, right_prediction, right_groups)
    group_ids = tuple(sorted(left))
    if set(right) != set(group_ids):
        raise ValueError("paired comparison group sets differ")
    point_left = _weighted_confusion_metrics(np.mean(np.stack([left[group] for group in group_ids]), axis=0))
    point_right = _weighted_confusion_metrics(np.mean(np.stack([right[group] for group in group_ids]), axis=0))
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(group_ids), size=(int(replicates), len(group_ids)))
    stats: dict[str, list[float]] = {
        name: [] for name in ("accuracy", "macro_f1", "balanced_accuracy", "sensitivity", "specificity", "precision_class_1")
    }
    for draw in draws:
        sample = [group_ids[int(index)] for index in draw]
        left_metric = _weighted_confusion_metrics(np.mean(np.stack([left[group] for group in sample]), axis=0))
        right_metric = _weighted_confusion_metrics(np.mean(np.stack([right[group] for group in sample]), axis=0))
        for name in stats:
            if left_metric[name] is not None and right_metric[name] is not None:
                stats[name].append(float(left_metric[name] - right_metric[name]))
    return [
        {
            "comparison": comparison,
            "metric": name,
            "delta_left_minus_right": (
                float(point_left[name] - point_right[name])
                if point_left[name] is not None and point_right[name] is not None
                else None
            ),
            "ci_95_low": float(np.quantile(values, 0.025)) if values else None,
            "ci_95_high": float(np.quantile(values, 0.975)) if values else None,
            "valid_replicates": len(values),
            "requested_replicates": int(replicates),
            "bootstrap_unit": "paired source group; group confusion matrices normalized to unit mass",
            "bootstrap_seed": int(seed),
        }
        for name, values in stats.items()
    ]


def _select_modal_outer_parameters(receipts: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = receipts.loc[
        receipts.feature_view.astype(str).eq("A1_morphology_only")
        & receipts.model_name.astype(str).eq("logistic_regression")
        & receipts.fit_status.astype(str).eq("completed")
    ].sort_values("outer_fold_id")
    if len(selected) != 4:
        raise ValueError("A1 development parent must contain four completed outer model selections")
    if not selected.selected_calibration_method.astype(str).eq("sigmoid").all():
        raise ValueError("A1 outer selections do not consistently support sigmoid calibration")
    parsed = [json.loads(value) for value in selected.selected_params_json.astype(str)]
    parameter_names = ("C", "class_weight", "scaler", "feature_k")
    parameters: dict[str, Any] = {}
    mode_receipt: list[dict[str, Any]] = []
    for name in parameter_names:
        values = [params.get(name) for params in parsed]
        counts = Counter(json.dumps(value, sort_keys=True) for value in values)
        max_count = max(counts.values())
        tied = [json.loads(value) for value, count in counts.items() if count == max_count]
        if name == "feature_k" and None in tied:
            choice = None
            tie_rule = "prefer all morphology features when modal counts tie"
        else:
            choice = sorted(tied, key=lambda value: (value is not None, str(value)))[0]
            tie_rule = "lexicographically first serialized value"
        parameters[name] = choice
        mode_receipt.append(
            {
                "parameter": name,
                "outer_selected_values": values,
                "modal_value": choice,
                "modal_count": max_count,
                "tie_rule": tie_rule if len(tied) > 1 else "unique mode",
            }
        )
    if parameters.get("feature_k") is None:
        parameters.pop("feature_k")
    return parameters, mode_receipt


def _calibrate_ood(values: np.ndarray, groups: np.ndarray, quantile: float) -> dict[str, Any]:
    group_scores: list[dict[str, Any]] = []
    for heldout in sorted(set(groups.astype(str).tolist())):
        fit = values[groups.astype(str) != heldout]
        query = values[groups.astype(str) == heldout]
        if len(fit) < 2 or len(query) < 1:
            raise ValueError(f"OOD leave-group-out calibration is unsupported for {heldout}")
        center, scale, keep = _robust_location_scale(fit)
        score = _robust_ood_score(query, center, scale, keep)
        group_scores.append(
            {
                "heldout_group": heldout,
                "fit_group_count": int(len(set(groups.astype(str).tolist())) - 1),
                "heldout_window_count_descriptive_only": int(len(query)),
                "heldout_score_p95": float(np.quantile(score, 0.95)),
            }
        )
    threshold = float(np.quantile([row["heldout_score_p95"] for row in group_scores], float(quantile)))
    center, scale, keep = _robust_location_scale(values)
    return {
        "method": "robust median/MAD distance; threshold calibrated from leave-one-source-group-out group p95 scores",
        "features": [name for name, retain in zip(OOD_FEATURES, keep, strict=True) if retain],
        "center": center.tolist(),
        "scale": scale.tolist(),
        "keep": keep.tolist(),
        "threshold": threshold,
        "quantile": float(quantile),
        "group_level_calibration_scores": group_scores,
        "uses_proxy_labels": False,
        "interpretation": "input-distribution flag only; independent OOD labels are unavailable",
    }


def freeze_development_model(
    run_dir: Path,
    *,
    source_subdir: str = "a1_lowfreq_optimization_v1",
    guardrail_subdir: str = "a1_lowfreq_guardrails_v1",
    noise_subdir: str = "a1_lowfreq_noise_robustness_v1",
    neural_noise_subdir: str = "neural_lowfreq_noise_robustness_v1",
    output_subdir: str = "frozen_development_lowfreq_v1",
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Refit the selected A1 proxy model on development groups and freeze policies."""
    run_dir = Path(run_dir).expanduser().resolve()
    private = run_dir / "private"
    for value in (source_subdir, guardrail_subdir, noise_subdir, neural_noise_subdir, output_subdir):
        if Path(value).name != value or value in {"", ".", ".."}:
            raise ValueError("stage subdirectories must be single path components")
    output_dir = private / output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen model stage: {output_dir}")

    source_dir = private / source_subdir
    guardrail_dir = private / guardrail_subdir
    noise_dir = private / noise_subdir
    neural_noise_dir = private / neural_noise_subdir
    source_manifest_path = source_dir / "a1_optimization_stage_manifest.json"
    guardrail_manifest_path = guardrail_dir / "a1_guardrail_stage_manifest.json"
    noise_manifest_path = noise_dir / "a1_noise_robustness_stage_manifest.json"
    neural_noise_manifest_path = neural_noise_dir / "noise_robustness_stage_manifest.json"
    for path in (source_manifest_path, guardrail_manifest_path, noise_manifest_path, neural_noise_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required completed development parent is missing: {path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    guardrail_manifest = json.loads(guardrail_manifest_path.read_text(encoding="utf-8"))
    noise_manifest = json.loads(noise_manifest_path.read_text(encoding="utf-8"))
    neural_noise_manifest = json.loads(neural_noise_manifest_path.read_text(encoding="utf-8"))
    for manifest, name in (
        (source_manifest, "A1 nested search"),
        (guardrail_manifest, "A1 OOD/abstention"),
        (noise_manifest, "A1 noise robustness"),
        (neural_noise_manifest, "neural noise robustness"),
    ):
        if manifest.get("status") != "completed" or manifest.get("label_rule") != LABEL_RULE:
            raise ValueError(f"{name} parent does not satisfy the active low-frequency development contract")
        if int(manifest.get("internal_test_signal_values_parsed", -1)) != 0 or manifest.get("external_rar_opened") is not False:
            raise ValueError(f"{name} parent receipt does not prove reserved evaluations stayed closed")
    if int(source_manifest.get("outer_group_folds", -1)) != 4 or int(source_manifest.get("inner_group_folds", -1)) != 3:
        raise ValueError("A1 parent is not the approved nested 4x3 design")
    _verify_outputs(source_manifest, "outputs_sha256")
    _verify_outputs(guardrail_manifest, "output_sha256")
    _verify_outputs(noise_manifest, "output_sha256")
    _verify_outputs(neural_noise_manifest, "output_sha256")

    split_path = private / "split_manifest.json"
    split_document = json.loads(split_path.read_text(encoding="utf-8"))
    development_groups = tuple(str(value) for value in split_document["development_group_ids"])
    reserved_groups = tuple(str(value) for value in split_document["test_group_ids"])
    if set(development_groups) & set(reserved_groups) or len(development_groups) != 12 or len(reserved_groups) != 3:
        raise ValueError("Frozen local split does not match the verified 12 development / 3 reserved group design")

    feature_path = private / "features_labels" / "features.parquet"
    labels_path = private / "features_labels" / "pseudo_labels_alternatives.parquet"
    features = pd.read_parquet(feature_path, filters=[("group_id", "in", list(development_groups))])
    features = features.loc[np.isclose(features.duration_s.to_numpy(dtype=float), 0.5)].copy()
    labels = pd.read_parquet(labels_path, filters=[("group_id", "in", list(development_groups))])
    labels = labels.loc[labels.comparison_rule.astype(str).eq(LABEL_RULE)].drop(columns=["comparison_rule"]).copy()
    if set(features.group_id.astype(str)) != set(development_groups) or set(labels.group_id.astype(str)) != set(development_groups):
        raise ValueError("development-filtered features/labels do not cover exactly the approved development groups")
    feature_names = tuple(FEATURE_VIEWS["A1_morphology_only"])
    split = GroupSplit(
        seed=int(split_document["seed"]),
        train_fraction=float(split_document["train_fraction"]),
        development_groups=development_groups,
        test_groups=reserved_groups,
    )
    frame = _prepare_windows_and_labels(features, labels, split, {"A1_morphology_only": feature_names})
    active = frame.loc[frame.class_id.isin((0, 1))].sort_values("window_key").reset_index(drop=True)
    _support(active, "all development groups", minimum_groups=12, minimum_trials=2)
    if active.window_key.astype(str).duplicated().any():
        raise ValueError("development feature/label join contains duplicate windows")
    x = active.loc[:, feature_names].astype(float)
    y = active.class_id.to_numpy(dtype=np.int64)
    groups = active.group_id.astype(str).to_numpy()

    plan = _fold_plan_from_manifest(source_dir / "fold_manifest.json")
    if tuple(sorted(plan.development_group_ids)) != tuple(sorted(development_groups)):
        raise ValueError("A1 nested fold parent and frozen development groups disagree")
    receipt_table = pd.read_csv(source_dir / "outer_model_receipts.csv")
    final_params, parameter_receipt = _select_modal_outer_parameters(receipt_table)

    # Reproduce the frozen four outer assignments with a single, selected final configuration.
    oof_rows: list[dict[str, Any]] = []
    fit_receipts: list[dict[str, Any]] = []
    for fold in plan.outer_folds:
        train_mask = np.isin(groups, np.asarray(fold.train_group_ids, dtype=object))
        valid_mask = np.isin(groups, np.asarray(fold.validation_group_ids, dtype=object))
        train_rows = active.loc[train_mask].reset_index(drop=True)
        valid_rows = active.loc[valid_mask].reset_index(drop=True)
        _support(train_rows, f"outer {fold.fold_id} final-configuration training")
        _support(valid_rows, f"outer {fold.fold_id} final-configuration validation")
        if set(train_rows.group_id.astype(str)) & set(valid_rows.group_id.astype(str)):
            raise RuntimeError("final-configuration OOF fold contains group leakage")
        model, warnings, fit_seconds = _fit_estimator(
            "logistic_regression",
            final_params,
            train_rows.loc[:, feature_names].astype(float),
            train_rows.class_id.to_numpy(dtype=np.int64),
            train_rows.group_id.astype(str).tolist(),
            seed=int(plan.seed + 77000 + int(fold.fold_id)),
            calibration_method="sigmoid",
        )
        probabilities = _positive_probability(model, valid_rows.loc[:, feature_names].astype(float))[:, 1]
        if len(probabilities) != int(valid_mask.sum()) or not np.isfinite(probabilities).all():
            raise RuntimeError("final-configuration OOF predictions are incomplete or non-finite")
        for row, probability in zip(valid_rows.itertuples(), probabilities, strict=True):
            oof_rows.append(
                {
                    "window_key": str(row.window_key),
                    "group_id": str(row.group_id),
                    "recording_id": str(row.recording_id),
                    "outer_fold_id": int(fold.fold_id),
                    "true_class_id": int(row.class_id),
                    "probability_class_0": float(1.0 - probability),
                    "probability_class_1": float(probability),
                    "fixed_0_5_prediction": int(probability >= 0.5),
                }
            )
        fit_receipts.append(
            {
                "outer_fold_id": int(fold.fold_id),
                "training_source_groups": list(fold.train_group_ids),
                "validation_source_groups": list(fold.validation_group_ids),
                "training_window_count_descriptive_only": int(len(train_rows)),
                "validation_window_count_descriptive_only": int(len(valid_rows)),
                "selected_params": final_params,
                "calibration": "sigmoid calibration fitted from three group-disjoint folds inside the outer training groups",
                "fit_seconds": float(fit_seconds),
                "warnings": warnings,
            }
        )
    oof = pd.DataFrame(oof_rows).sort_values("window_key").reset_index(drop=True)
    if len(oof) != len(active) or oof.window_key.astype(str).duplicated().any():
        raise RuntimeError("final-configuration outer OOF does not cover every assigned development window exactly once")
    if set(oof.group_id.astype(str)) != set(development_groups):
        raise RuntimeError("final-configuration OOF does not cover every development group")
    threshold = select_group_balanced_threshold(
        oof.true_class_id.to_numpy(dtype=int),
        oof.probability_class_1.to_numpy(dtype=float),
        oof.group_id.astype(str).tolist(),
    )
    oof["inner_oof_selected_prediction"] = (
        oof.probability_class_1.to_numpy(dtype=float) >= float(threshold.threshold)
    ).astype(np.int64)

    # Ensure the paired temporal comparison uses the identical OOF windows, labels, and groups.
    parent_oof_path = source_dir / "nested_oof_predictions.parquet"
    parent_oof = pd.read_parquet(parent_oof_path)
    baseline = parent_oof.loc[
        parent_oof.feature_view.astype(str).eq("relative_time_only")
        & parent_oof.model_name.astype(str).eq("logistic_regression")
        & parent_oof.threshold_strategy.astype(str).eq("inner_oof_selected")
        & parent_oof.true_class_id.isin((0, 1)),
        ["window_key", "group_id", "true_class_id", "predicted_class_id"],
    ].copy()
    paired = oof.merge(baseline, on="window_key", how="inner", suffixes=("", "_baseline"), validate="one_to_one")
    if len(paired) != len(oof):
        raise ValueError("development OOF and temporal baseline differ in assigned observations")
    if not paired.group_id.astype(str).eq(paired.group_id_baseline.astype(str)).all() or not paired.true_class_id.eq(paired.true_class_id_baseline).all():
        raise ValueError("development OOF and temporal baseline differ in groups or pseudo-labels")

    fixed_prediction = oof.fixed_0_5_prediction.to_numpy(dtype=int)
    selected_prediction = oof.inner_oof_selected_prediction.to_numpy(dtype=int)
    probability_matrix = np.column_stack((oof.probability_class_0, oof.probability_class_1))
    metric_rows: list[dict[str, Any]] = []
    for strategy, predicted in (("fixed_0_5", fixed_prediction), ("development_oof_selected", selected_prediction)):
        metric = compute_metrics(y, probability_matrix, predicted, np.zeros(len(y), dtype=bool)).to_dict()
        group_confusions = _group_confusions(y, predicted, groups)
        group_balanced = _weighted_confusion_metrics(np.mean(np.stack(list(group_confusions.values())), axis=0))
        metric_rows.append(
            {
                "model_name": "A1_morphology_logistic",
                "threshold_strategy": strategy,
                "decision_threshold": 0.5 if strategy == "fixed_0_5" else float(threshold.threshold),
                "window_count_descriptive_only": int(len(y)),
                "source_group_count": int(len(set(groups.tolist()))),
                "accuracy": metric["accuracy"],
                "precision_fatigue_like": metric["per_class"]["1"]["precision"],
                "sensitivity_fatigue_like": metric["sensitivity"],
                "specificity": metric["specificity"],
                "macro_f1": metric["macro_f1"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "mcc": metric["matthews_correlation_coefficient"],
                "cohen_kappa": metric["cohen_kappa"],
                "pr_auc_fatigue_like": metric["pr_auc_average_precision"],
                "roc_auc": metric["roc_auc"],
                "brier_score": metric["calibration"]["brier_score"],
                "confusion_matrix": json.dumps(metric["confusion_matrix"], separators=(",", ":")),
                "group_balanced_metrics": json.dumps(group_balanced, sort_keys=True, separators=(",", ":")),
                "group_balanced_confusion_matrix": json.dumps(np.mean(np.stack(list(group_confusions.values())), axis=0).tolist(), separators=(",", ":")),
                "interpretation": "development OOF agreement with EMG-derived proxy only",
            }
        )
    paired_intervals: list[dict[str, Any]] = []
    paired_intervals.extend(
        _paired_group_bootstrap(
            y, selected_prediction, groups,
            y, fixed_prediction, groups,
            comparison="selected_threshold_minus_fixed_0_5",
            replicates=int(bootstrap_replicates), seed=int(plan.seed + 8080),
        )
    )
    for name, prediction in (("fixed_0_5", fixed_prediction), ("development_oof_selected", selected_prediction)):
        paired_intervals.extend(
            _paired_group_bootstrap(
                paired.true_class_id.to_numpy(dtype=int), prediction, paired.group_id.astype(str).to_numpy(),
                paired.true_class_id_baseline.to_numpy(dtype=int), paired.predicted_class_id.to_numpy(dtype=int), paired.group_id_baseline.astype(str).to_numpy(),
                comparison=f"A1_{name}_minus_temporal_baseline_inner_oof_selected",
                replicates=int(bootstrap_replicates), seed=int(plan.seed + 9090 + (0 if name == "fixed_0_5" else 1)),
            )
        )

    # Train the deployable checkpoint on all 12 development groups; calibration remains group-cross-fitted.
    final_estimator, final_warnings, final_fit_seconds = _fit_estimator(
        "logistic_regression",
        final_params,
        x,
        y,
        groups.tolist(),
        seed=int(plan.seed + 99000),
        calibration_method="sigmoid",
    )
    calibration_split_receipt = [
        {
            "fit_group_count": int(len(set(groups[fit].tolist()))),
            "heldout_group_count": int(len(set(groups[calibration].tolist()))),
            "group_overlap": bool(set(groups[fit].tolist()) & set(groups[calibration].tolist())),
            "fit_class_support": {str(cls): int(np.sum(y[fit] == cls)) for cls in (0, 1)},
            "heldout_class_support": {str(cls): int(np.sum(y[calibration] == cls)) for cls in (0, 1)},
        }
        for fit, calibration in group_calibration_splits(groups.tolist(), y, int(plan.seed + 99000), n_splits=3)
    ]

    ood_values = active.loc[:, OOD_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(ood_values).all():
        raise ValueError("development morphology rows contain non-finite OOD inputs")
    ood_policy = _calibrate_ood(ood_values, groups, OOD_QUANTILE)
    feature_minimum = np.min(ood_values, axis=0)
    feature_maximum = np.max(ood_values, axis=0)
    guardrail_thresholds = pd.read_csv(guardrail_dir / "abstention_threshold_selection.csv")
    a1_thresholds = guardrail_thresholds.loc[
        guardrail_thresholds.model_name.astype(str).eq("A1_morphology_logistic")
    ]
    if len(a1_thresholds) != 4 or not np.allclose(a1_thresholds.selected_confidence_threshold.to_numpy(dtype=float), CONFIDENCE_THRESHOLD):
        raise ValueError("A1 inner-OOF abstention selection does not support the frozen 0.55 confidence threshold")
    abstention_policy = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "selection_source": str((guardrail_dir / "abstention_threshold_selection.csv").resolve()),
        "selection_objective": "inner OOF group-balanced Macro-F1 with abstentions counted as class errors; ties favor higher coverage",
        "outer_validation_used_for_policy": False,
        "abstain_on_robust_mad_ood_flag": True,
        "abstain_on_outside_development_feature_support": True,
        "abstain_on_failed_quality_gate": True,
        "interpretation": "selective proxy-label policy; confidence is not an OOD score and not physiological evidence",
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "frozen_a1_morphology_logistic.joblib"
    checkpoint_receipt = {
        "estimator": final_estimator,
        "model_name": "A1_morphology_logistic",
        "label_rule": LABEL_RULE,
        "feature_names": list(feature_names),
        "parameters": final_params,
        "calibration_method": "group-disjoint sigmoid via 3-fold crossfit across development source groups",
        "decision_threshold": float(threshold.threshold),
        "fixed_comparison_threshold": 0.5,
        "confidence_abstention_threshold": CONFIDENCE_THRESHOLD,
        "independent_unit": "source folder group; participant identity unverified",
    }
    joblib.dump(checkpoint_receipt, checkpoint_path, compress=3)
    reloaded = joblib.load(checkpoint_path)
    probe = x.iloc[: min(100, len(x))].copy()
    original_probability = _positive_probability(final_estimator, probe)
    reloaded_probability = _positive_probability(reloaded["estimator"], probe)
    if not np.allclose(original_probability, reloaded_probability, rtol=0.0, atol=1e-12):
        raise RuntimeError("frozen checkpoint reload changed development predictions")

    oof_path = output_dir / "development_oof_predictions.parquet"
    metrics_path = output_dir / "development_oof_metrics.csv"
    paired_path = output_dir / "paired_group_bootstrap.csv"
    oof.to_parquet(oof_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(paired_intervals).to_csv(paired_path, index=False)
    ood_path = output_dir / "frozen_ood_policy.json"
    _write_json(
        ood_path,
        {
            **ood_policy,
            "feature_support_minimum": feature_minimum.tolist(),
            "feature_support_maximum": feature_maximum.tolist(),
            "feature_support_names": list(OOD_FEATURES),
        },
    )
    policy_path = output_dir / "frozen_prediction_policy.json"
    _write_json(
        policy_path,
        {
            "model_name": "A1_morphology_logistic",
            "label_rule": LABEL_RULE,
            "fixed_threshold_comparator": 0.5,
            "selected_decision_threshold": float(threshold.threshold),
            "threshold_objective": "group-balanced Macro-F1 on calibrated development outer-OOF predictions",
            "threshold_source_groups": sorted(set(groups.tolist())),
            "threshold_candidate_count": int(threshold.candidate_count),
            "threshold_tie_resolution": threshold.tie_resolution,
            "calibration_method": checkpoint_receipt["calibration_method"],
            "ood_detector": "robust_mad",
            "ood_threshold": float(ood_policy["threshold"]),
            "ood_quantile": OOD_QUANTILE,
            "abstention": abstention_policy,
            "quality_gate": "development pipeline feature_valid and raw_signal_compatible must both pass",
            "overlap_windows_are_independent": False,
            "interpretation": "agreement with the EMG-derived operational proxy; not independent physiological fatigue validation",
        },
    )

    input_paths = [
        split_path,
        source_manifest_path,
        source_dir / "fold_manifest.json",
        source_dir / "outer_model_receipts.csv",
        parent_oof_path,
        guardrail_manifest_path,
        guardrail_dir / "abstention_threshold_selection.csv",
        guardrail_dir / "aurc_summary.csv",
        noise_manifest_path,
        neural_noise_manifest_path,
        feature_path,
        labels_path,
    ]
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("evaluation.py"),
        Path(__file__).with_name("evaluation_guardrails.py"),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("group_cv.py"),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("models_tabular.py"),
    ]
    output_paths = [checkpoint_path, oof_path, metrics_path, paired_path, ood_path, policy_path]
    receipt = {
        "stage": "development_only_selected_model_calibration_ood_abstention_freeze",
        "status": "completed",
        "label_rule": LABEL_RULE,
        "label_rule_type": LABEL_TYPE,
        "selected_model": "A1_morphology_logistic",
        "selected_parameters": final_params,
        "parameter_selection": "modal configuration among four inner-selected outer-fold winners; ties prefer no feature selection",
        "parameter_selection_receipt": parameter_receipt,
        "selection_evidence": "highest development group-balanced Macro-F1 among eligible signal models; paired CI versus temporal-only baseline includes zero",
        "calibration": "sigmoid; three source-group-disjoint crossfit folds within development training data",
        "calibration_group_split_receipt": calibration_split_receipt,
        "decision_threshold": float(threshold.threshold),
        "threshold_fixed_comparator": 0.5,
        "threshold_selection": "group-balanced Macro-F1 on calibrated 4-fold development OOF predictions only",
        "threshold_selection_candidate_count": int(threshold.candidate_count),
        "threshold_tie_resolution": threshold.tie_resolution,
        "ood_policy": "robust_mad calibrated from leave-one-development-group-out group p95 scores",
        "abstention_confidence_threshold": CONFIDENCE_THRESHOLD,
        "outer_group_folds": 4,
        "inner_group_folds_per_outer": 3,
        "development_groups": list(plan.development_group_ids),
        "locked_internal_group_count": len(reserved_groups),
        "locked_internal_test_values_read": 0,
        "raw2_rar_opened": False,
        "assigned_development_windows_descriptive_only": int(len(active)),
        "class_support_descriptive_only": {str(cls): int(np.sum(y == cls)) for cls in (0, 1)},
        "group_count": int(len(set(groups.tolist()))),
        "independent_unit": "source folder group; participant identity unverified",
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_reload_predictions_verified": True,
        "training_fit_seconds": float(final_fit_seconds),
        "training_warnings": final_warnings,
        "source_sha256": {str(path.resolve()): _sha256_file(path) for path in source_paths},
        "input_sha256": {str(path.resolve()): _sha256_file(path) for path in input_paths},
        "output_sha256": {str(path.resolve()): _sha256_file(path) for path in output_paths},
        "interpretation": "proxy-label agreement only; not physiological fatigue validation",
    }
    manifest_path = output_dir / "freeze_manifest.json"
    _write_json(manifest_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path.resolve())}
