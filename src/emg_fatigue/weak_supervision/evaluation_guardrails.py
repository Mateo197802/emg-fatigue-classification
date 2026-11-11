"""Development-only OOD and selective-prediction analysis for EMG proxy models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import tomllib

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .evaluation_neural import AVAILABLE_MODEL_NAMES, _load_fold_plan
from .evaluation_neural_nested import (
    PRIMARY_LABEL_RULE,
    _sha256_file,
    _weighted_confusion_metrics,
)
from .metrics import compute_metrics


OOD_FEATURES = (
    "rms",
    "mav",
    "iemg",
    "variance",
    "waveform_length_per_second",
    "zero_crossings",
    "slope_sign_changes",
    "willison_amplitude",
    "median_absolute_deviation",
    "interquartile_range",
    "skewness",
    "kurtosis_excess",
    "hjorth_activity",
    "hjorth_mobility_hz",
    "hjorth_complexity",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def _hash_entries(mapping: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for name, expected in mapping.items():
        path = Path(name).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != str(expected):
            errors.append(str(path))
    return errors


def _verify_neural_stage(stage_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_path = stage_dir / "neural_supervised_stage_manifest.json"
    prediction_path = stage_dir / "neural_lowfreq_oof_predictions.parquet"
    inner_path = stage_dir / "inner_oof_selection_predictions_with_model.parquet"
    if not inner_path.is_file():
        inner_path = stage_dir / "inner_oof_selection_predictions.parquet"
    label_path = stage_dir / "active_lowfreq_signal_window_manifest.parquet"
    for path in (manifest_path, prediction_path, inner_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required neural guardrail input is missing: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("label_rule") != PRIMARY_LABEL_RULE
        or manifest.get("external_rar_opened") is not False
        or int(manifest.get("internal_test_signal_values_parsed", -1)) != 0
        or int(manifest.get("outer_group_folds", -1)) != 4
        or int(manifest.get("inner_group_folds_per_outer", -1)) != 3
    ):
        raise ValueError("neural parent receipt is not a locked low-frequency 4x3 development run")
    hash_errors = _hash_entries(manifest.get("output_sha256", {}))
    if hash_errors:
        raise ValueError("neural parent output checksums failed: " + "; ".join(hash_errors))
    predictions = pd.read_parquet(prediction_path)
    inner = pd.read_parquet(inner_path)
    labels = pd.read_parquet(label_path)
    if "model_name" not in inner:
        raise ValueError("model-keyed inner OOF copy is required for development abstention selection")
    if inner.duplicated(["model_name", "outer_fold_id", "window_key"]).any():
        raise ValueError("model-keyed inner OOF copy has duplicate model/fold/window rows")
    model_names = tuple(str(value) for value in manifest.get("models", ()))
    if not model_names or not set(model_names).issubset(set(AVAILABLE_MODEL_NAMES)):
        raise ValueError("neural parent model registry is empty or unknown")
    if not set(inner.model_name.astype(str)).issubset(set(model_names)):
        raise ValueError("inner OOF model names differ from the evaluated architecture registry")
    if set(predictions.loc[predictions.feature_view.eq("raw_signal"), "model_name"].astype(str)) != set(model_names):
        raise ValueError("outer OOF model names differ from the neural stage receipt")
    return manifest, predictions, inner, labels


def _robust_location_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("OOD reference matrix must be finite with at least two rows")
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = np.where(mad > 1e-10, mad, (q75 - q25) / 1.349)
    keep = scale > 1e-10
    if not keep.any():
        raise ValueError("OOD detector found no varying robust feature dimensions")
    scale = scale[keep]
    center = center[keep]
    return center, scale, keep


def _robust_ood_score(matrix: np.ndarray, center: np.ndarray, scale: np.ndarray, keep: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    z = np.abs((values[:, keep] - center) / scale)
    return np.sqrt(np.mean(z * z, axis=1))


def _fit_fold_ood_detector(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    quantile: float,
) -> tuple[np.ndarray, float, dict[str, Any], pd.DataFrame]:
    train_groups = sorted(set(train.group_id.astype(str)))
    if len(train_groups) < 4:
        raise ValueError("group-calibrated OOD guard requires at least four outer-training source groups")
    calibration_summaries: list[dict[str, Any]] = []
    calibration_scores: list[float] = []
    for heldout_group in train_groups:
        fit = train.loc[~train.group_id.astype(str).eq(heldout_group)]
        heldout = train.loc[train.group_id.astype(str).eq(heldout_group)]
        if fit.group_id.nunique() < 2 or heldout.empty:
            raise ValueError("OOD leave-group-out calibration split lacks independent support")
        center, scale, keep = _robust_location_scale(fit[list(OOD_FEATURES)].to_numpy(dtype=float))
        scores = _robust_ood_score(heldout[list(OOD_FEATURES)].to_numpy(dtype=float), center, scale, keep)
        group_p95 = float(np.quantile(scores, 0.95))
        calibration_scores.append(group_p95)
        calibration_summaries.append(
            {
                "heldout_group": heldout_group,
                "fit_group_count": int(fit.group_id.nunique()),
                "heldout_window_count_descriptive_only": int(len(heldout)),
                "heldout_score_p95": group_p95,
            }
        )
    threshold = float(np.quantile(calibration_scores, float(quantile)))
    final_center, final_scale, final_keep = _robust_location_scale(train[list(OOD_FEATURES)].to_numpy(dtype=float))
    validation_scores = _robust_ood_score(
        validation[list(OOD_FEATURES)].to_numpy(dtype=float),
        final_center,
        final_scale,
        final_keep,
    )
    receipt = {
        "method": "fold-specific robust median/MAD distance; leave-one-source-group-out calibration",
        "features": [name for name, keep in zip(OOD_FEATURES, final_keep, strict=True) if keep],
        "feature_count": int(final_keep.sum()),
        "training_source_groups": train_groups,
        "calibration_group_count": len(calibration_scores),
        "threshold_quantile_of_group_p95": float(quantile),
        "ood_threshold": threshold,
        "group_level_calibration_scores": calibration_summaries,
        "uses_proxy_labels": False,
        "outer_validation_used_for_fit_or_threshold": False,
        "interpretation": "input-distribution flag only; no independent OOD truth is available",
    }
    result = validation[["window_key", "group_id", "recording_id", "class_id"]].copy()
    result["ood_score"] = validation_scores
    result["ood_threshold"] = threshold
    result["ood_flag"] = validation_scores > threshold
    result["quality_pass"] = validation.eligible_for_label.fillna(False).astype(bool).to_numpy()
    return validation_scores, threshold, receipt, result


OOD_METHODS = ("robust_mad", "mahalanobis_ledoit_wolf", "knn_k5", "isolation_forest")


def _fit_ood_scorer(fit_values: np.ndarray, query_values: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit one novelty score using only ``fit_values`` and score query rows."""
    from sklearn.covariance import LedoitWolf
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors

    center, scale, keep = _robust_location_scale(fit_values)
    fit_matrix = (np.asarray(fit_values, dtype=float)[:, keep] - center) / scale
    query_matrix = (np.asarray(query_values, dtype=float)[:, keep] - center) / scale
    if method == "robust_mad":
        score = _robust_ood_score(query_values, center, scale, keep)
        return score, fit_matrix, {"method": "robust median/MAD root-mean-square distance"}
    if method == "mahalanobis_ledoit_wolf":
        estimator = LedoitWolf(assume_centered=False).fit(fit_matrix)
        delta = query_matrix - estimator.location_
        score = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, estimator.precision_, delta), 0.0))
        return score, fit_matrix, {"method": "robust-scaled shrinkage Mahalanobis distance", "covariance": "Ledoit-Wolf"}
    if method == "knn_k5":
        neighbor_count = min(5, int(len(fit_matrix)))
        if neighbor_count < 2:
            raise ValueError("kNN novelty requires at least two fit windows")
        estimator = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean", algorithm="auto").fit(fit_matrix)
        distances, _ = estimator.kneighbors(query_matrix, return_distance=True)
        return distances[:, -1], fit_matrix, {"method": f"robust-scaled {neighbor_count}-nearest-neighbor distance"}
    if method == "isolation_forest":
        estimator = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            random_state=20261116,
            n_jobs=1,
        ).fit(fit_matrix)
        score = -estimator.score_samples(query_matrix)
        return score.astype(float), fit_matrix, {"method": "robust-scaled Isolation Forest negative score_samples"}
    raise ValueError(f"Unsupported OOD method: {method}")


def _fit_fold_ood_detectors(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    quantile: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Calibrate multiple fold-specific novelty detectors by leave-group-out scores."""
    train_groups = sorted(set(train.group_id.astype(str)))
    if len(train_groups) < 4:
        raise ValueError("OOD calibration requires at least four outer-training source groups")
    train_values = train[list(OOD_FEATURES)].to_numpy(dtype=float)
    valid_values = validation[list(OOD_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(train_values).all() or not np.isfinite(valid_values).all():
        raise ValueError("OOD fit and validation features must be finite")
    rows: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    # This fixed envelope is a literal support check, separate from probabilistic
    # uncertainty and the statistical novelty detectors.
    support_low = np.min(train_values, axis=0)
    support_high = np.max(train_values, axis=0)
    unsupported = np.any((valid_values < support_low) | (valid_values > support_high), axis=1)
    for method in OOD_METHODS:
        calibration_group_p95: list[float] = []
        calibration_receipts: list[dict[str, Any]] = []
        for heldout_group in train_groups:
            fit = train.loc[~train.group_id.astype(str).eq(heldout_group)]
            held = train.loc[train.group_id.astype(str).eq(heldout_group)]
            if fit.group_id.nunique() < 2 or held.empty:
                raise ValueError(f"{method} leave-group-out split lacks independent group support")
            scores, _, method_receipt = _fit_ood_scorer(
                fit[list(OOD_FEATURES)].to_numpy(dtype=float),
                held[list(OOD_FEATURES)].to_numpy(dtype=float),
                method,
            )
            group_p95 = float(np.quantile(scores, 0.95))
            calibration_group_p95.append(group_p95)
            calibration_receipts.append({
                "heldout_group": heldout_group,
                "fit_group_count": int(fit.group_id.nunique()),
                "heldout_windows_descriptive_only": int(len(held)),
                "heldout_score_p95": group_p95,
            })
        threshold = float(np.quantile(calibration_group_p95, float(quantile)))
        scores, _, method_receipt = _fit_ood_scorer(train_values, valid_values, method)
        result = validation[["window_key", "group_id", "recording_id", "class_id"]].copy()
        result["ood_detector"] = method
        result["ood_score"] = scores
        result["ood_threshold"] = threshold
        result["ood_flag"] = scores > threshold
        result["unsupported_signal"] = unsupported
        result["quality_pass"] = (
            validation.eligible_for_label.fillna(False).astype(bool).to_numpy()
            & validation.feature_valid.fillna(False).astype(bool).to_numpy()
            & validation.raw_signal_compatible.fillna(False).astype(bool).to_numpy()
        )
        reasons = []
        for q_pass, ood, unsupported_row in zip(
            result.quality_pass, result.ood_flag, result.unsupported_signal, strict=True
        ):
            items = []
            if not q_pass:
                items.append("signal_quality")
            if ood:
                items.append("ood_novelty")
            if unsupported_row:
                items.append("outside_training_feature_support")
            reasons.append("|".join(items))
        result["non_classifier_abstention_reasons"] = reasons
        rows.append(result)
        receipts.append({
            **method_receipt,
            "method_id": method,
            "features": list(OOD_FEATURES),
            "feature_count": len(OOD_FEATURES),
            "training_source_groups": train_groups,
            "calibration_group_count": len(calibration_group_p95),
            "threshold_quantile_of_group_p95": float(quantile),
            "ood_threshold": threshold,
            "group_level_calibration_scores": calibration_receipts,
            "outer_validation_used_for_fit_or_threshold": False,
            "uses_proxy_labels": False,
            "interpretation": "novelty score only; independent OOD ground truth is unavailable",
            "unsupported_signal_rule": "any selected morphology feature outside the outer-training empirical min/max; fixed support rule, no outcome labels used",
        })
    combined = pd.concat(rows, ignore_index=True)
    expected = len(validation) * len(OOD_METHODS)
    if len(combined) != expected or combined.duplicated(["ood_detector", "window_key"]).any():
        raise RuntimeError("multi-detector OOD outputs do not cover the fold windows exactly once")
    return combined, receipts


def _group_macro_f1_with_abstention(rows: pd.DataFrame, confidence_threshold: float) -> float:
    truth = rows.true_class_id.to_numpy(dtype=int)
    probability = rows.probability_class_1.to_numpy(dtype=float)
    decision_threshold = rows.decision_threshold.to_numpy(dtype=float)
    confidence = rows.confidence.to_numpy(dtype=float)
    predicted = (probability >= decision_threshold).astype(int)
    predicted[confidence < float(confidence_threshold)] = -1
    matrices: list[np.ndarray] = []
    for _, group in rows.assign(_predicted=predicted).groupby("group_id", sort=True):
        matrix = np.zeros((2, 3), dtype=np.float64)
        column = {0: 0, 1: 1, -1: 2}
        for actual, guess in zip(group.true_class_id.astype(int), group._predicted.astype(int), strict=True):
            matrix[actual, column[guess]] += 1.0
        matrices.append(matrix / matrix.sum())
    if not matrices:
        raise ValueError("abstention threshold selection requires inner OOF groups")
    cm = np.mean(np.stack(matrices), axis=0)
    # Abstained positives and negatives remain false negatives for their class.
    tp0, fp0, fn0 = cm[0, 0], cm[1, 0], cm[0, 1] + cm[0, 2]
    tp1, fp1, fn1 = cm[1, 1], cm[0, 1], cm[1, 0] + cm[1, 2]
    f10 = 2 * tp0 / (2 * tp0 + fp0 + fn0) if 2 * tp0 + fp0 + fn0 else 0.0
    f11 = 2 * tp1 / (2 * tp1 + fp1 + fn1) if 2 * tp1 + fp1 + fn1 else 0.0
    return float((f10 + f11) / 2.0)


def _select_confidence_threshold(rows: pd.DataFrame, candidates: Sequence[float]) -> dict[str, Any]:
    choices: list[dict[str, Any]] = []
    for threshold in sorted(set(float(value) for value in candidates)):
        macro_f1 = _group_macro_f1_with_abstention(rows, threshold)
        coverage = float((rows.confidence.to_numpy(dtype=float) >= threshold).mean())
        choices.append({"threshold": threshold, "inner_oof_group_balanced_macro_f1": macro_f1, "inner_oof_coverage": coverage})
    best_score = max(item["inner_oof_group_balanced_macro_f1"] for item in choices)
    tied = [item for item in choices if np.isclose(item["inner_oof_group_balanced_macro_f1"], best_score, atol=1e-12, rtol=1e-12)]
    selected = sorted(tied, key=lambda item: (-item["inner_oof_coverage"], item["threshold"]))[0]
    return {
        **selected,
        "candidate_thresholds": choices,
        "selection_objective": "maximize group-balanced Macro-F1 with abstentions counted as class errors; ties favor higher coverage",
    }


def _risk_coverage_points(
    rows: pd.DataFrame,
    thresholds: Sequence[float],
    *,
    fixed_abstain: Sequence[bool] | None = None,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    groups = tuple(sorted(rows.group_id.astype(str).unique()))
    fixed = np.zeros(len(rows), dtype=bool) if fixed_abstain is None else np.asarray(fixed_abstain, dtype=bool)
    if fixed.shape != (len(rows),):
        raise ValueError("fixed_abstain must match the risk-coverage prediction rows")
    for threshold in sorted(set(float(value) for value in thresholds)):
        accepted = (rows.confidence.to_numpy(dtype=float) >= threshold) & ~fixed
        truth = rows.true_class_id.to_numpy(dtype=int)
        predicted = rows.predicted_class_id.to_numpy(dtype=int)
        selected = accepted.sum()
        class_0 = truth == 0
        class_1 = truth == 1
        group_coverages: list[float] = []
        group_risks: list[float] = []
        for group_id in groups:
            mask = rows.group_id.astype(str).to_numpy() == group_id
            local_accepted = accepted[mask]
            local_truth = truth[mask]
            local_prediction = predicted[mask]
            group_coverages.append(float(local_accepted.mean()))
            if local_accepted.any():
                group_risks.append(float(np.mean(local_prediction[local_accepted] != local_truth[local_accepted])))
            else:
                group_risks.append(1.0)
        output.append(
            {
                "confidence_threshold": threshold,
                "coverage": float(selected / len(rows)),
                "class_0_coverage": float(accepted[class_0].mean()) if class_0.any() else None,
                "class_1_coverage": float(accepted[class_1].mean()) if class_1.any() else None,
                "selective_error": float(np.mean(predicted[accepted] != truth[accepted])) if selected else None,
                "group_balanced_coverage": float(np.mean(group_coverages)),
                "group_balanced_selective_error": float(np.mean(group_risks)),
                "accepted_window_count_descriptive_only": int(selected),
                "abstained_window_count_descriptive_only": int(len(rows) - selected),
                "fixed_quality_ood_or_unsupported_abstentions": int(fixed.sum()),
                "source_group_count": len(groups),
            }
        )
    return pd.DataFrame(output)


def _aurc(points: pd.DataFrame, coverage_column: str, risk_column: str) -> float | None:
    valid = points.dropna(subset=[coverage_column, risk_column]).sort_values(coverage_column)
    if len(valid) < 2:
        return None
    x = valid[coverage_column].to_numpy(dtype=float)
    y = valid[risk_column].to_numpy(dtype=float)
    return float(np.trapezoid(y, x))


def _group_confusion_for_selective(rows: pd.DataFrame, *, abstentions_as_errors: bool) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for group_id, group in rows.groupby("group_id", sort=True):
        truth = group.true_class_id.to_numpy(dtype=int)
        prediction = (
            group.selective_predicted_class_id.to_numpy(dtype=int, copy=True)
            if abstentions_as_errors
            else group.predicted_class_id.to_numpy(dtype=int, copy=True)
        )
        if abstentions_as_errors:
            abstained = prediction == -1
            prediction[abstained] = 1 - truth[abstained]
        matrix = confusion_matrix(truth, prediction, labels=(0, 1)).astype(np.float64)
        if matrix.sum() == 0 or set(np.unique(truth).tolist()) != {0, 1}:
            raise ValueError(f"group {group_id} lacks both classes for paired abstention uncertainty")
        result[str(group_id)] = matrix / matrix.sum()
    return result


def _paired_abstention_intervals(
    predictions: pd.DataFrame,
    replicates: int,
    seed: int,
    model_names: Sequence[str],
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for model_name in model_names:
        for detector in sorted(predictions.ood_detector.astype(str).unique()):
            rows = predictions.loc[
                predictions.model_name.astype(str).eq(model_name)
                & predictions.ood_detector.astype(str).eq(detector)
            ].copy()
            unselected = _group_confusion_for_selective(rows, abstentions_as_errors=False)
            selected = _group_confusion_for_selective(rows, abstentions_as_errors=True)
            groups = tuple(sorted(unselected))
            if set(selected) != set(groups):
                raise ValueError("abstention and non-abstention predictions differ in source groups")
            choices = rng.integers(0, len(groups), size=(int(replicates), len(groups)))
            base_point = _weighted_confusion_metrics(np.mean(np.stack([unselected[group] for group in groups]), axis=0))
            selected_point = _weighted_confusion_metrics(np.mean(np.stack([selected[group] for group in groups]), axis=0))
            deltas: list[float] = []
            for draw in choices:
                sampled = [groups[index] for index in draw]
                base_metric = _weighted_confusion_metrics(np.mean(np.stack([unselected[group] for group in sampled]), axis=0))["macro_f1"]
                selected_metric = _weighted_confusion_metrics(np.mean(np.stack([selected[group] for group in sampled]), axis=0))["macro_f1"]
                if base_metric is not None and selected_metric is not None:
                    deltas.append(float(selected_metric - base_metric))
            output.append(
                {
                    "model_name": model_name,
                    "ood_detector": detector,
                    "comparison": "combined quality, selected OOD novelty, unsupported-signal, and inner-selected classifier-uncertainty abstention minus threshold-only prediction",
                    "group_balanced_macro_f1_delta": float(selected_point["macro_f1"] - base_point["macro_f1"]),
                    "ci_95_low": float(np.quantile(deltas, 0.025)) if deltas else None,
                    "ci_95_high": float(np.quantile(deltas, 0.975)) if deltas else None,
                    "valid_replicates": len(deltas),
                    "bootstrap_replicates": int(replicates),
                    "bootstrap_seed": int(seed),
                    "bootstrap_unit": "paired source group; group confusion mass normalized to one",
                }
            )
    return pd.DataFrame(output)


def run_guardrail_stage(
    run_dir: Path,
    *,
    neural_subdir: str = "neural_lowfreq_nested_4x3_v1",
    threshold_subdir: str = "lowfreq_threshold_v2",
    output_subdir: str = "neural_lowfreq_guardrails_v2",
    confidence_thresholds: Sequence[float] | None = None,
    ood_quantile: float | None = None,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    private_dir = run_dir / "private"
    stage_dir = private_dir / neural_subdir
    threshold_dir = private_dir / threshold_subdir
    output_dir = private_dir / output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite guardrail stage: {output_dir}")
    neural_manifest, predictions, inner, labels = _verify_neural_stage(stage_dir)
    model_names = tuple(str(value) for value in neural_manifest["models"])
    prediction_path = stage_dir / "neural_lowfreq_oof_predictions.parquet"
    inner_path = stage_dir / "inner_oof_selection_predictions_with_model.parquet"
    if not inner_path.is_file():
        inner_path = stage_dir / "inner_oof_selection_predictions.parquet"
    label_path = stage_dir / "active_lowfreq_signal_window_manifest.parquet"
    fold_path = threshold_dir / "fold_manifest.json"
    feature_path = private_dir / "features_labels" / "features.parquet"
    config_path = private_dir / "fatigue_proxy_config.toml"
    for path in (fold_path, feature_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required guardrail input is missing: {path.name}")
    plan = _load_fold_plan(fold_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    guardrails = config.get("guardrails", {})
    if confidence_thresholds is None:
        confidence_thresholds = guardrails.get("confidence_thresholds", (0.55, 0.65, 0.75))
    if ood_quantile is None:
        ood_quantile = float(guardrails.get("ood_upper_quantile", 0.995))
    if not 0.5 <= float(ood_quantile) < 1.0:
        raise ValueError("ood_quantile must be at least 0.5 and less than 1")
    if not confidence_thresholds or any(not 0.5 <= float(value) <= 1.0 for value in confidence_thresholds):
        raise ValueError("confidence thresholds must be non-empty and lie in [0.5, 1]")
    confidence_thresholds = tuple(float(value) for value in confidence_thresholds)

    metadata = labels.copy()
    development_groups = sorted(metadata.group_id.astype(str).unique())
    features = pd.read_parquet(
        feature_path, filters=[("group_id", "in", development_groups)]
    )
    primary_features = features.loc[
        np.isclose(features.duration_s.to_numpy(dtype=float), 0.5),
        ["window_key", "group_id", "recording_id", "feature_valid", *OOD_FEATURES],
    ].copy()
    if primary_features.window_key.astype(str).duplicated().any():
        raise ValueError("valid 500 ms feature rows have duplicate window keys")
    primary_features["window_key"] = primary_features.window_key.astype(str)
    if not set(primary_features.window_key).issubset(set(metadata.window_key.astype(str))):
        raise ValueError("OOD feature rows include windows outside the development signal manifest")
    primary_features["group_id"] = primary_features.group_id.astype(str)
    primary_features["recording_id"] = primary_features.recording_id.astype(str)
    merged_features = metadata.merge(
        primary_features,
        on="window_key",
        how="inner",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )
    for identity in ("group_id", "recording_id"):
        right = f"{identity}_feature"
        if right in merged_features and not merged_features[identity].astype(str).eq(merged_features[right].astype(str)).all():
            raise ValueError(f"raw signal and OOD {identity} values disagree")
    acquisition = pd.read_csv(
        private_dir / "acquisition" / "acquisition_audit.csv",
        usecols=["group_id", "recording_id", "raw_signal_compatible"],
    )
    acquisition["group_id"] = acquisition.group_id.astype(str)
    acquisition["recording_id"] = acquisition.recording_id.astype(str)
    if acquisition.duplicated(["group_id", "recording_id"]).any():
        raise ValueError("acquisition compatibility table has duplicate development trials")
    merged_features = merged_features.merge(
        acquisition,
        on=["group_id", "recording_id"],
        how="left",
        validate="many_to_one",
    )
    if merged_features.raw_signal_compatible.isna().any():
        raise ValueError("some development windows lack an acquisition QC compatibility record")
    active_keys = set(metadata.loc[metadata.class_id.isin((0, 1)), "window_key"].astype(str))
    if not active_keys.issubset(set(merged_features.window_key.astype(str))):
        raise ValueError("valid OOD feature rows do not cover every assigned proxy window")
    active_feature_rows = merged_features.loc[merged_features.class_id.isin((0, 1))]
    if not active_feature_rows.feature_valid.fillna(False).astype(bool).all():
        raise ValueError("assigned proxy labels include feature-invalid OOD rows")
    if not np.isfinite(active_feature_rows[list(OOD_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("assigned proxy OOD features contain non-finite values")

    output_dir.mkdir(parents=True, exist_ok=False)
    active_predictions = predictions.loc[
        predictions.threshold_strategy.eq("inner_oof_selected")
        & predictions.feature_view.eq("raw_signal")
        & predictions.model_name.isin(model_names),
    ].copy()
    ood_receipts: list[dict[str, Any]] = []
    ood_rows: list[pd.DataFrame] = []
    for outer in plan.outer_folds:
        train = merged_features.loc[
            merged_features.group_id.astype(str).isin(set(outer.train_group_ids))
            & merged_features.eligible_for_label.fillna(False).astype(bool)
            & merged_features.feature_valid.fillna(False).astype(bool)
            & merged_features.raw_signal_compatible.fillna(False).astype(bool)
        ].copy()
        validation = merged_features.loc[
            merged_features.group_id.astype(str).isin(set(outer.validation_group_ids))
            & merged_features.class_id.isin((0, 1))
        ].copy()
        scored, detector_receipts = _fit_fold_ood_detectors(
            train,
            validation,
            quantile=float(ood_quantile),
        )
        for detector_receipt in detector_receipts:
            detector_receipt.update({
                "outer_fold_id": int(outer.fold_id),
                "held_out_groups": sorted(set(validation.group_id.astype(str))),
                "held_out_window_count_descriptive_only": int(len(validation)),
                "held_out_ood_flag_count_descriptive_only": int(
                    scored.loc[scored.ood_detector.eq(detector_receipt["method_id"]), "ood_flag"].sum()
                ),
                "held_out_ood_flag_rate": float(
                    scored.loc[scored.ood_detector.eq(detector_receipt["method_id"]), "ood_flag"].mean()
                ),
            })
            ood_receipts.append(detector_receipt)
        ood_rows.append(scored)
    ood_scores = pd.concat(ood_rows, ignore_index=True)
    if ood_scores.duplicated(["ood_detector", "window_key"]).any():
        raise RuntimeError("OOD scoring duplicated a detector/window pair")
    active_key_set = set(active_predictions.window_key.astype(str))
    for method in OOD_METHODS:
        method_rows = ood_scores.loc[ood_scores.ood_detector.eq(method)].copy()
        if set(method_rows.window_key.astype(str)) != active_key_set:
            raise ValueError(f"{method} scores do not cover the same assigned outer OOF windows")
    expanded_predictions = []
    for method in OOD_METHODS:
        method_scores = ood_scores.loc[ood_scores.ood_detector.eq(method)].drop(columns=["class_id"])
        method_predictions = active_predictions.merge(
            method_scores,
            on=["window_key", "group_id", "recording_id"],
            how="left",
            validate="many_to_one",
        )
        expanded_predictions.append(method_predictions)
    active_predictions = pd.concat(expanded_predictions, ignore_index=True)
    if active_predictions.duplicated(["model_name", "threshold_strategy", "window_key", "ood_detector"]).any():
        raise RuntimeError("multi-detector neural OOF rows contain duplicate architecture/detector/windows")
    if active_predictions.ood_score.isna().any():
        raise RuntimeError("OOD score merge left unmatched neural OOF predictions")
    active_predictions.to_parquet(output_dir / "neural_guardrail_oof_predictions.parquet", index=False)

    thresholds = pd.read_csv(stage_dir / "inner_threshold_selection.csv")
    selected_policy_rows: list[dict[str, Any]] = []
    selective_predictions: list[pd.DataFrame] = []
    for model_name in model_names:
        for outer in plan.outer_folds:
            fold_inner = inner.loc[
                inner.model_name.astype(str).eq(model_name)
                & inner.outer_fold_id.astype(int).eq(int(outer.fold_id))
            ].copy()
            fold_threshold = thresholds.loc[
                thresholds.model_name.astype(str).eq(model_name)
                & thresholds.outer_fold_id.astype(int).eq(int(outer.fold_id))
            ]
            if len(fold_threshold) != 1 or fold_inner.empty:
                raise ValueError(f"missing unique inner threshold/OOF data for {model_name}, fold {outer.fold_id}")
            fold_inner["decision_threshold"] = float(fold_threshold.threshold.iloc[0])
            fold_inner["confidence"] = fold_inner[["probability_class_0", "probability_class_1"]].max(axis=1)
            policy = _select_confidence_threshold(fold_inner, confidence_thresholds)
            selected_policy_rows.append(
                {
                    "model_name": model_name,
                    "outer_fold_id": int(outer.fold_id),
                    "selected_classifier_threshold": float(fold_threshold.threshold.iloc[0]),
                    "selected_confidence_threshold": float(policy["threshold"]),
                    "inner_oof_group_balanced_macro_f1": float(policy["inner_oof_group_balanced_macro_f1"]),
                    "inner_oof_coverage": float(policy["inner_oof_coverage"]),
                    "candidate_policy_scores": json.dumps(policy["candidate_thresholds"], separators=(",", ":")),
                    "selection_objective": policy["selection_objective"],
                    "inner_oof_groups": json.dumps(sorted(set(fold_inner.group_id.astype(str))), separators=(",", ":")),
                    "outer_validation_used_for_policy": False,
                }
            )
            fold_outer = active_predictions.loc[
                active_predictions.model_name.astype(str).eq(model_name)
                & active_predictions.fold_id.astype(int).eq(int(outer.fold_id))
            ].copy()
            confidence_cut = float(policy["threshold"])
            abstain_uncertainty = fold_outer.confidence.to_numpy(dtype=float) < confidence_cut
            abstain_quality = ~fold_outer.quality_pass.to_numpy(dtype=bool)
            abstain_ood = fold_outer.ood_flag.to_numpy(dtype=bool)
            abstain_unsupported = fold_outer.unsupported_signal.to_numpy(dtype=bool)
            abstain = abstain_uncertainty | abstain_quality | abstain_ood | abstain_unsupported
            fold_outer["selected_confidence_threshold"] = confidence_cut
            fold_outer["abstain_classifier_uncertainty"] = abstain_uncertainty
            fold_outer["abstain_signal_quality"] = abstain_quality
            fold_outer["abstain_ood_novelty"] = abstain_ood
            fold_outer["abstain_unsupported_signal"] = abstain_unsupported
            fold_outer["abstain"] = abstain
            fold_outer["abstention_reasons"] = [
                "|".join(reason for flag, reason in (
                    (quality, "signal_quality"),
                    (ood, "ood_novelty"),
                    (uncertain, "classifier_uncertainty"),
                    (unsupported, "outside_training_feature_support"),
                ) if flag)
                for quality, ood, uncertain, unsupported in zip(
                    abstain_quality, abstain_ood, abstain_uncertainty, abstain_unsupported, strict=True
                )
            ]
            fold_outer["selective_predicted_class_id"] = np.where(
                abstain,
                -1,
                fold_outer.predicted_class_id.to_numpy(dtype=int),
            )
            selective_predictions.append(fold_outer)
    selective = pd.concat(selective_predictions, ignore_index=True)
    selective.to_parquet(output_dir / "neural_abstention_oof_predictions.parquet", index=False)
    pd.DataFrame(selected_policy_rows).to_csv(output_dir / "abstention_threshold_selection.csv", index=False)

    pooled_metrics: list[dict[str, Any]] = []
    group_metrics: list[dict[str, Any]] = []
    risk_curve_parts: list[pd.DataFrame] = []
    aurc_rows: list[dict[str, Any]] = []
    for model_name in model_names:
      for detector in OOD_METHODS:
        rows = selective.loc[
            selective.model_name.astype(str).eq(model_name)
            & selective.ood_detector.astype(str).eq(detector)
        ].copy()
        record = compute_metrics(
            rows.true_class_id.to_numpy(dtype=int),
            rows.probability_class_1.to_numpy(dtype=float),
            rows.selective_predicted_class_id.to_numpy(dtype=int),
            rows.abstain.to_numpy(dtype=bool),
        ).to_dict()
        truth = rows.true_class_id.to_numpy(dtype=int)
        accepted = ~rows.abstain.to_numpy(dtype=bool)
        pooled_metrics.append(
            {
                "model_name": model_name,
                "ood_detector": detector,
                "threshold_strategy": "inner_oof_selected_plus_inner_oof_selected_confidence_abstention",
                "selected_confidence_thresholds_by_outer_fold": json.dumps(
                    {str(fold): sorted(set(float(value) for value in fold_rows.selected_confidence_threshold)) for fold, fold_rows in rows.groupby("fold_id", sort=True)},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "coverage": record["coverage"],
                "coverage_class_0": float(accepted[truth == 0].mean()) if np.any(truth == 0) else None,
                "coverage_class_1": float(accepted[truth == 1].mean()) if np.any(truth == 1) else None,
                "abstained_count": int((~accepted).sum()),
                "abstained_count_class_0": int(np.sum((~accepted) & (truth == 0))),
                "abstained_count_class_1": int(np.sum((~accepted) & (truth == 1))),
                "abstained_for_quality": int(rows.abstain_signal_quality.sum()),
                "abstained_for_ood": int(rows.abstain_ood_novelty.sum()),
                "abstained_for_uncertainty": int(rows.abstain_classifier_uncertainty.sum()),
                "abstained_for_unsupported_signal": int(rows.abstain_unsupported_signal.sum()),
                "abstained_total_any_reason": int((~accepted).sum()),
                "selective_error": record["selective_error"],
                "accuracy_including_abstention_as_error": record["accuracy"],
                "sensitivity": record["sensitivity"],
                "specificity": record["specificity"],
                "macro_f1_including_abstention": record["macro_f1"],
                "group_count": int(rows.group_id.nunique()),
                "confusion_matrix_with_abstain": json.dumps(record["confusion_matrix"], separators=(",", ":")),
                "undefined_reasons": json.dumps(record["undefined_reasons"], sort_keys=True, separators=(",", ":")),
            }
        )
        for (group_id, fold_id), group in rows.groupby(["group_id", "fold_id"], sort=True):
            g_truth = group.true_class_id.to_numpy(dtype=int)
            g_accept = ~group.abstain.to_numpy(dtype=bool)
            g_record = compute_metrics(
                g_truth,
                group.probability_class_1.to_numpy(dtype=float),
                group.selective_predicted_class_id.to_numpy(dtype=int),
                group.abstain.to_numpy(dtype=bool),
            ).to_dict()
            group_metrics.append(
                {
                    "model_name": model_name,
                    "ood_detector": detector,
                    "group_id": str(group_id),
                    "outer_fold_id": int(fold_id),
                    "windows_descriptive_only": int(len(group)),
                    "class_0_support": int(np.sum(g_truth == 0)),
                    "class_1_support": int(np.sum(g_truth == 1)),
                    "coverage": g_record["coverage"],
                    "coverage_class_0": float(g_accept[g_truth == 0].mean()) if np.any(g_truth == 0) else None,
                    "coverage_class_1": float(g_accept[g_truth == 1].mean()) if np.any(g_truth == 1) else None,
                    "selective_error": g_record["selective_error"],
                    "macro_f1_including_abstention": g_record["macro_f1"],
                    "confusion_matrix_with_abstain": json.dumps(g_record["confusion_matrix"], separators=(",", ":")),
                    "undefined_reasons": json.dumps(g_record["undefined_reasons"], sort_keys=True, separators=(",", ":")),
                }
            )
        model_risk = rows[["group_id", "true_class_id", "predicted_class_id", "confidence"]].copy()
        confidence_grid = sorted(
            set([0.5, 0.55, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
            | set(np.quantile(model_risk.confidence.to_numpy(dtype=float), np.linspace(0.0, 1.0, 51)).tolist())
        )
        fixed_abstentions = (
            rows.abstain_signal_quality.to_numpy(dtype=bool)
            | rows.abstain_ood_novelty.to_numpy(dtype=bool)
            | rows.abstain_unsupported_signal.to_numpy(dtype=bool)
        )
        points = _risk_coverage_points(model_risk, confidence_grid, fixed_abstain=fixed_abstentions)
        points["model_name"] = model_name
        points["ood_detector"] = detector
        risk_curve_parts.append(points)
        aurc_rows.append(
            {
                "model_name": model_name,
                "ood_detector": detector,
                "group_balanced_aurc": _aurc(points, "group_balanced_coverage", "group_balanced_selective_error"),
                "pooled_aurc_not_group_independent": _aurc(points, "coverage", "selective_error"),
                "risk_definition": "selective classification error at confidence cut; group-balanced mean error is primary",
                "confidence_is_ood": False,
                "source_group_count": int(rows.group_id.nunique()),
                "outer_oof_only": True,
            }
        )
    pd.DataFrame(pooled_metrics).to_csv(output_dir / "abstention_metrics.csv", index=False)
    pd.DataFrame(group_metrics).to_csv(output_dir / "abstention_group_metrics.csv", index=False)
    risk_curve = pd.concat(risk_curve_parts, ignore_index=True)
    risk_curve.to_csv(output_dir / "risk_coverage_curves.csv", index=False)
    pd.DataFrame(aurc_rows).to_csv(output_dir / "aurc_summary.csv", index=False)
    _paired_abstention_intervals(
        selective,
        int(bootstrap_replicates),
        int(plan.seed),
        model_names,
    ).to_csv(output_dir / "paired_abstention_group_bootstrap.csv", index=False)

    group_ood = (
        ood_scores.groupby(["ood_detector", "group_id", "class_id"], sort=True)
        .agg(
            window_count_descriptive_only=("window_key", "size"),
            ood_flag_count=("ood_flag", "sum"),
            ood_flag_rate=("ood_flag", "mean"),
            mean_ood_score=("ood_score", "mean"),
            threshold_min=("ood_threshold", "min"),
            threshold_max=("ood_threshold", "max"),
        )
        .reset_index()
    )
    group_ood.to_csv(output_dir / "ood_group_rates.csv", index=False)
    pd.DataFrame(ood_receipts).to_json(output_dir / "ood_detector_receipts.json", orient="records", indent=2)

    inputs = [
        stage_dir / "neural_supervised_stage_manifest.json",
        prediction_path,
        inner_path,
        label_path,
        fold_path,
        feature_path,
        config_path,
    ]
    output_paths = [
        output_dir / "neural_guardrail_oof_predictions.parquet",
        output_dir / "neural_abstention_oof_predictions.parquet",
        output_dir / "abstention_threshold_selection.csv",
        output_dir / "abstention_metrics.csv",
        output_dir / "abstention_group_metrics.csv",
        output_dir / "risk_coverage_curves.csv",
        output_dir / "aurc_summary.csv",
        output_dir / "paired_abstention_group_bootstrap.csv",
        output_dir / "ood_group_rates.csv",
        output_dir / "ood_detector_receipts.json",
    ]
    receipt = {
        "stage": "development_grouped_ood_and_selective_prediction_guardrails",
        "status": "completed",
        "label_rule": PRIMARY_LABEL_RULE,
        "outer_group_folds": len(plan.outer_folds),
        "inner_group_folds_per_outer": len(plan.inner_folds[plan.outer_folds[0].fold_id]),
        "models": list(model_names),
        "confidence_thresholds_considered": list(confidence_thresholds),
        "confidence_threshold_selection": "inner OOF group-balanced Macro-F1 with abstentions counted as class errors; ties favor coverage",
        "confidence_thresholds_use_outer_validation": False,
        "ood_detector": "outer-training-only robust median/MAD distance; threshold calibrated on leave-one-group-out group p95 scores",
        "ood_detectors_compared": list(OOD_METHODS),
        "unsupported_signal_rule": "feature values outside the outer-training empirical min/max across morphology indicators",
        "quality_gate": "development window eligibility, valid feature vector, and raw-signal-compatible acquisition record",
        "abstention_reasons": ["signal_quality", "ood_novelty", "classifier_uncertainty", "outside_training_feature_support"],
        "ood_quantile": float(ood_quantile),
        "ood_labels_available": False,
        "ood_scores_are_not_confidence": True,
        "ood_flags_are_not_validated_detection_accuracy": True,
        "risk_coverage_and_aurc_unit": "source group primary; window counts descriptive",
        "source_group_count": int(ood_scores.group_id.nunique()),
        "internal_test_signal_values_parsed": 0,
        "external_rar_opened": False,
        "paired_bootstrap_replicates": int(bootstrap_replicates),
        "input_sha256": {str(path): _sha256_file(path) for path in inputs},
        "output_sha256": {str(path): _sha256_file(path) for path in output_paths},
        "interpretation": "development proxy agreement and input-distribution diagnostics only; no physiological fatigue or OOD ground-truth validation",
    }
    manifest_path = output_dir / "guardrail_stage_manifest.json"
    _write_json(manifest_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path.resolve())}
