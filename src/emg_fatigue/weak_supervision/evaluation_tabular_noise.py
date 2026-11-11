"""Matched-noise robustness for the selected nested A1 morphology model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .augmentation import CORRUPTION_FAMILIES, EVALUATION_LEVELS, corrupt_window, stable_seed
from .evaluation import _sha256_file
from .evaluation_neural import _load_fold_plan
from .evaluation_neural_nested import join_primary_lowfreq_labels
from .evaluation_noise_robustness import _morphology_batch
from .features import TIME_FEATURES
from .metrics import compute_metrics


def _group_balanced_macro_f1_from_confusion(matrix: np.ndarray) -> float:
    cm = np.asarray(matrix, dtype=float)
    f1_values = []
    for class_id in (0, 1):
        tp = float(cm[class_id, class_id])
        fp = float(cm[:, class_id].sum() - tp)
        fn = float(cm[class_id, :].sum() - tp)
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(f1_values))


def _normalized_group_confusions(rows: pd.DataFrame, prediction_column: str) -> dict[str, np.ndarray]:
    result = {}
    for group_id, group in rows.groupby("group_id", sort=True):
        matrix = confusion_matrix(
            group.true_class_id.astype(int), group[prediction_column].astype(int), labels=[0, 1]
        ).astype(float)
        result[str(group_id)] = matrix / max(float(matrix.sum()), 1.0)
    return result


def _bootstrap_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_prediction: str,
    right_prediction: str,
    seed: int,
    replicates: int = 2000,
) -> dict[str, object]:
    groups = tuple(sorted(set(left.group_id.astype(str))))
    if groups != tuple(sorted(set(right.group_id.astype(str)))):
        raise ValueError("paired group bootstrap received different source groups")
    identity = ["window_key", "group_id", "recording_id", "fold_id", "true_class_id"]
    paired = left[identity].merge(right[identity], on="window_key", suffixes=("_left", "_right"), validate="one_to_one")
    if len(paired) != len(left) or len(right) != len(left):
        raise ValueError("paired noise evaluation has different window support")
    for name in identity[1:]:
        if not paired[f"{name}_left"].astype(str).eq(paired[f"{name}_right"].astype(str)).all():
            raise ValueError(f"paired noise evaluation differs on {name}")
    left_confusions = _normalized_group_confusions(left, left_prediction)
    right_confusions = _normalized_group_confusions(right, right_prediction)
    left_stack = np.stack([left_confusions[group] for group in groups])
    right_stack = np.stack([right_confusions[group] for group in groups])
    rng = np.random.default_rng(int(seed))
    choices = rng.integers(0, len(groups), size=(int(replicates), len(groups)))
    left_samples = left_stack[choices].mean(axis=1)
    right_samples = right_stack[choices].mean(axis=1)
    deltas = np.asarray(
        [_group_balanced_macro_f1_from_confusion(a) - _group_balanced_macro_f1_from_confusion(b)
         for a, b in zip(left_samples, right_samples, strict=True)],
        dtype=float,
    )
    point = _group_balanced_macro_f1_from_confusion(left_stack.mean(axis=0)) - _group_balanced_macro_f1_from_confusion(right_stack.mean(axis=0))
    return {
        "delta_group_balanced_macro_f1": float(point),
        "ci_95_low": float(np.quantile(deltas, 0.025)),
        "ci_95_high": float(np.quantile(deltas, 0.975)),
        "requested_replicates": int(replicates),
        "valid_replicates": int(np.isfinite(deltas).sum()),
        "bootstrap_unit": "paired source groups; group confusion matrices normalized to unit mass",
    }


def _metric_record(rows: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    truth = rows.true_class_id.astype(int).to_numpy()
    prediction = rows[prediction_column].astype(int).to_numpy()
    probability = rows.probability_class_1.astype(float).to_numpy()
    metric = compute_metrics(truth, probability, prediction, np.zeros(len(rows), dtype=bool)).to_dict()
    weights = np.zeros(len(rows), dtype=float)
    group_ids = rows.group_id.astype(str).to_numpy()
    for group in sorted(set(group_ids)):
        selected = group_ids == group
        weights[selected] = 1.0 / int(selected.sum())
    metric.update({
        "group_balanced_macro_f1": _group_balanced_macro_f1_from_confusion(
            np.mean(np.stack(list(_normalized_group_confusions(rows, prediction_column).values())), axis=0)
        ),
        "group_weighted_pr_auc_average_precision": float(average_precision_score(truth, probability, sample_weight=weights)),
        "group_weighted_roc_auc": float(roc_auc_score(truth, probability, sample_weight=weights)),
        "group_weighted_brier_score": float(np.sum(weights * np.square(probability - truth)) / np.sum(weights)),
    })
    return metric


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str), encoding="utf-8")


def run_a1_noise_robustness_stage(
    run_dir: Path,
    *,
    a1_subdir: str = "a1_lowfreq_optimization_v1",
    threshold_subdir: str = "lowfreq_threshold_v2",
    output_subdir: str = "a1_lowfreq_noise_robustness_v1",
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Evaluate clean-trained outer A1 checkpoints on deterministic matched noise."""
    run_dir = Path(run_dir).expanduser().resolve()
    private_dir = run_dir / "private"
    a1_dir = private_dir / a1_subdir
    threshold_dir = private_dir / threshold_subdir
    output_dir = private_dir / output_subdir
    if any(Path(item).name != item or item in {"", ".", ".."} for item in (a1_subdir, threshold_subdir, output_subdir)):
        raise ValueError("stage directory names must be single relative directory names")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite A1 noise stage: {output_dir}")
    a1_manifest_path = a1_dir / "a1_optimization_stage_manifest.json"
    a1_receipt = json.loads(a1_manifest_path.read_text(encoding="utf-8"))
    if (
        a1_receipt.get("status") != "completed"
        or a1_receipt.get("label_rule") != "low_frequency_fraction_persistence_v1"
        or int(a1_receipt.get("outer_group_folds", -1)) != 4
        or int(a1_receipt.get("inner_group_folds", -1)) != 3
    ):
        raise ValueError("A1 parent is not the completed low-frequency nested 4x3 stage")
    for name, expected in a1_receipt.get("outputs_sha256", {}).items():
        path = Path(name).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise ValueError(f"A1 parent output checksum mismatch: {path.name}")

    bank_dir = private_dir / "neural_data"
    bank_manifest_path = bank_dir / "signal_window_bank_manifest.json"
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    if (
        bank_manifest.get("status") != "completed"
        or bank_manifest.get("external_rar_opened") is not False
        or int(bank_manifest.get("internal_test_signal_values_parsed", -1)) != 0
        or int(bank_manifest.get("reserved_test_member_payloads_read", -1)) != 0
    ):
        raise ValueError("development signal bank does not prove reserved-data isolation")
    signals_path = bank_dir / "signals_500ms_development.npz"
    metadata_path = bank_dir / "signal_window_manifest.parquet"
    for path in (signals_path, metadata_path):
        expected = bank_manifest.get("output_sha256", {}).get(str(path))
        if not expected or _sha256_file(path) != expected:
            raise ValueError(f"signal bank checksum mismatch: {path.name}")
    split_doc = json.loads((private_dir / "split_manifest.json").read_text(encoding="utf-8"))
    development_groups = [str(value) for value in split_doc["development_group_ids"]]
    fold_path = threshold_dir / "fold_manifest.json"
    plan = _load_fold_plan(fold_path)
    bank = np.load(signals_path, allow_pickle=False)
    signals = np.asarray(bank["signals"], dtype=np.float32)
    bank_keys = bank["window_keys"].astype(str)
    metadata = pd.read_parquet(metadata_path)
    if not np.array_equal(bank_keys, metadata.window_key.astype(str).to_numpy()):
        raise ValueError("development signal values and metadata order differ")
    if set(metadata.group_id.astype(str)) != set(development_groups):
        raise ValueError("signal bank must contain only the frozen development groups")
    labels = pd.read_parquet(
        private_dir / "features_labels" / "pseudo_labels_alternatives.parquet",
        filters=[("group_id", "in", development_groups)],
    )
    metadata = join_primary_lowfreq_labels(metadata, labels)
    if len(metadata) != len(signals) or metadata.window_key.duplicated().any():
        raise ValueError("joined low-frequency labels changed the signal bank universe")
    active = metadata.loc[metadata.class_id.isin((0, 1))].copy().reset_index(drop=True)
    active_positions = metadata.index[metadata.class_id.isin((0, 1))].to_numpy(dtype=int)
    active["signal_position"] = active_positions
    group_to_fold = {
        str(group): int(outer.fold_id)
        for outer in plan.outer_folds
        for group in outer.validation_group_ids
    }
    active["fold_id"] = active.group_id.astype(str).map(group_to_fold).astype(int)
    if active.class_id.value_counts().to_dict() != {0: 5074, 1: 903}:
        raise ValueError("active pseudo-labels differ from the frozen primary low-frequency development counts")
    rates = sorted(pd.to_numeric(active.sampling_rate_hz, errors="raise").astype(float).unique().tolist())
    if len(rates) != 1:
        raise ValueError(f"A1 morphology noise stage expects a single development sample rate, got {rates}")

    model_receipts = pd.read_csv(a1_dir / "outer_model_receipts.csv")
    model_receipts = model_receipts.loc[
        model_receipts.feature_view.eq("A1_morphology_only")
        & model_receipts.model_name.eq("logistic_regression")
    ]
    if len(model_receipts) != 4 or set(model_receipts.outer_fold_id.astype(int)) != set(range(1, 5)):
        raise ValueError("A1 outer-fold checkpoint set is incomplete")
    models: dict[int, Any] = {}
    thresholds: dict[int, float] = {}
    checkpoint_paths: list[Path] = []
    for row in model_receipts.itertuples(index=False):
        path = Path(str(row.model_artifact_path)).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != str(row.model_artifact_sha256):
            raise ValueError(f"A1 checkpoint checksum mismatch for fold {row.outer_fold_id}")
        models[int(row.outer_fold_id)] = joblib.load(path)
        thresholds[int(row.outer_fold_id)] = float(row.selected_threshold)
        checkpoint_paths.append(path)

    parent_oof = pd.read_parquet(threshold_dir / "nested_oof_predictions.parquet")
    temporal = parent_oof.loc[
        parent_oof.feature_view.eq("relative_time_only")
        & parent_oof.model_name.eq("logistic_regression")
        & parent_oof.true_class_id.isin((0, 1))
    ].copy()
    if temporal.duplicated(["window_key", "threshold_strategy"]).any():
        raise ValueError("temporal control OOF contains duplicate window/threshold rows")
    expected_keys = set(active.window_key.astype(str))
    for strategy in ("fixed_0_5", "inner_oof_selected"):
        rows = temporal.loc[temporal.threshold_strategy.eq(strategy)]
        if set(rows.window_key.astype(str)) != expected_keys or len(rows) != len(active):
            raise ValueError(f"temporal control {strategy} does not match the exact A1 window set")
    output_dir.mkdir(parents=True)
    pred_parts: list[pd.DataFrame] = []
    group_metric_parts: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    condition_iter = [("clean", 0.0, False)] + [
        (family, float(level), float(level) > 0.20)
        for family in CORRUPTION_FAMILIES
        for level in EVALUATION_LEVELS
    ]
    for condition_index, (family, level, stress) in enumerate(condition_iter):
        if family == "clean":
            corrupted = signals[active_positions]
            achieved = np.zeros(len(active), dtype=float)
            seeds = np.zeros(len(active), dtype=np.uint32)
        else:
            corrupted = np.empty((len(active), signals.shape[1]), dtype=np.float32)
            achieved = np.empty(len(active), dtype=float)
            seeds = np.empty(len(active), dtype=np.uint32)
            for index, row in enumerate(active.itertuples(index=False)):
                seed = stable_seed("emg-matched-evaluation-noise-v1", str(row.window_key), family, level)
                corrupted[index], receipt = corrupt_window(
                    signals[int(row.signal_position)], family, level, seed,
                    sampling_rate_hz=float(row.sampling_rate_hz),
                )
                achieved[index] = float(receipt["achieved_rms_ratio"])
                seeds[index] = int(seed)
        feature_frame = _morphology_batch(corrupted, rates[0])
        missing = set(TIME_FEATURES) - set(feature_frame.columns)
        if missing:
            raise ValueError(f"vectorized morphology recomputation misses A1 features: {sorted(missing)}")
        feature_frame = feature_frame.loc[:, list(TIME_FEATURES)]
        for fold_id in sorted(models):
            mask = active.fold_id.astype(int).eq(fold_id).to_numpy()
            fold_rows = active.loc[mask].reset_index(drop=True)
            probabilities = np.asarray(models[fold_id].predict_proba(feature_frame.loc[mask])[:, 1], dtype=float)
            if probabilities.shape != (len(fold_rows),) or not np.isfinite(probabilities).all():
                raise ValueError("A1 noise inference returned invalid probabilities")
            threshold = thresholds[fold_id]
            for strategy, cut in (("fixed_0_5", 0.5), ("inner_oof_selected", threshold)):
                model_rows = pd.DataFrame({
                    "window_key": fold_rows.window_key.astype(str).to_numpy(),
                    "group_id": fold_rows.group_id.astype(str).to_numpy(),
                    "recording_id": fold_rows.recording_id.astype(str).to_numpy(),
                    "fold_id": fold_id,
                    "true_class_id": fold_rows.class_id.astype(int).to_numpy(),
                    "model_name": "A1_morphology_logistic",
                    "threshold_strategy": strategy,
                    "family": family,
                    "requested_level": level,
                    "stress_test": bool(stress),
                    "achieved_rms_ratio": achieved[mask],
                    "corruption_seed": seeds[mask],
                    "probability_class_1": probabilities,
                    "prediction": (probabilities >= cut).astype(int),
                    "decision_threshold": float(cut),
                })
                pred_parts.append(model_rows)
                for (group_id, fold_value), group_rows in model_rows.groupby(["group_id", "fold_id"], sort=True):
                    metric = _metric_record(group_rows, "prediction")
                    group_metric_parts.append({
                        "model_name": "A1_morphology_logistic", "threshold_strategy": strategy,
                        "family": family, "requested_level": level, "group_id": str(group_id),
                        "fold_id": int(fold_value), "window_count_descriptive_only": len(group_rows),
                        "group_macro_f1": metric["macro_f1"],
                        "group_sensitivity": metric["sensitivity"],
                        "group_specificity": metric["specificity"],
                        "group_precision_class_1": metric["per_class"]["1"]["precision"],
                    })
        # Identical temporal control rows are retained for each corruption condition.
        for strategy in ("fixed_0_5", "inner_oof_selected"):
            base = temporal.loc[temporal.threshold_strategy.eq(strategy)].copy()
            base["model_name"] = "time_only_logistic"
            base["family"] = family
            base["requested_level"] = level
            base["stress_test"] = bool(stress)
            base["achieved_rms_ratio"] = 0.0
            base["corruption_seed"] = np.uint32(0)
            base["probability_class_1"] = base.probability_class_1.astype(float)
            base["prediction"] = base.predicted_class_id.astype(int)
            base["decision_threshold"] = base.decision_threshold.astype(float)
            base = base[[
                "window_key", "group_id", "recording_id", "fold_id", "true_class_id", "model_name",
                "threshold_strategy", "family", "requested_level", "stress_test", "achieved_rms_ratio",
                "corruption_seed", "probability_class_1", "prediction", "decision_threshold",
            ]]
            pred_parts.append(base)

    predictions = pd.concat(pred_parts, ignore_index=True)
    if predictions.duplicated(["model_name", "threshold_strategy", "family", "requested_level", "window_key"]).any():
        raise RuntimeError("A1 noise OOF contains duplicate model/condition/window predictions")
    for (model, strategy, family, level), rows in predictions.groupby(
        ["model_name", "threshold_strategy", "family", "requested_level"], sort=True
    ):
        if len(rows) != len(active) or set(rows.window_key.astype(str)) != expected_keys:
            raise RuntimeError(f"incomplete matched window coverage for {model}/{strategy}/{family}/{level}")

    metric_rows: list[dict[str, object]] = []
    group_macro_lookup: dict[tuple[str, str, str, float], dict[str, float]] = {}
    for (model, strategy, family, level), rows in predictions.groupby(
        ["model_name", "threshold_strategy", "family", "requested_level"], sort=True
    ):
        metrics = _metric_record(rows, "prediction")
        groups = rows.group_id.astype(str).to_numpy()
        confusion_by_group = _normalized_group_confusions(rows, "prediction")
        group_macro = _group_balanced_macro_f1_from_confusion(np.mean(np.stack(list(confusion_by_group.values())), axis=0))
        group_macro_lookup[(str(model), str(strategy), str(family), float(level))] = {
            group: _group_balanced_macro_f1_from_confusion(matrix) for group, matrix in confusion_by_group.items()
        }
        metric_rows.append({
            "model_name": str(model), "threshold_strategy": str(strategy), "family": str(family),
            "requested_level": float(level), "stress_test": bool(level > 0.20),
            "n_oof_windows_descriptive_only": int(len(rows)), "source_group_count": int(rows.group_id.nunique()),
            "accuracy": metrics["accuracy"], "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"], "precision_fatigue_like": metrics["per_class"]["1"]["precision"],
            "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"],
            "mcc": metrics["matthews_correlation_coefficient"], "cohen_kappa": metrics["cohen_kappa"],
            "pr_auc_average_precision": metrics["pr_auc_average_precision"], "roc_auc": metrics["roc_auc"],
            "brier_score": metrics["calibration"]["brier_score"], "log_loss": metrics["log_loss"],
            "group_balanced_macro_f1": group_macro,
            "group_weighted_pr_auc_average_precision": metrics["group_weighted_pr_auc_average_precision"],
            "group_weighted_roc_auc": metrics["group_weighted_roc_auc"],
            "group_weighted_brier_score": metrics["group_weighted_brier_score"],
            "confusion_matrix": json.dumps(metrics["confusion_matrix"], separators=(",", ":")),
            "achieved_rms_ratio_median": float(rows.achieved_rms_ratio.median()),
            "window_counts_are_descriptive_only": True,
        })

    metrics_frame = pd.DataFrame(metric_rows)
    paired_rows = []
    for model in ("A1_morphology_logistic", "time_only_logistic"):
        for strategy in ("fixed_0_5", "inner_oof_selected"):
            clean = predictions.loc[
                predictions.model_name.eq(model) & predictions.threshold_strategy.eq(strategy)
                & predictions.family.eq("clean") & predictions.requested_level.eq(0.0)
            ].copy()
            for family in CORRUPTION_FAMILIES:
                for level in EVALUATION_LEVELS:
                    noisy = predictions.loc[
                        predictions.model_name.eq(model) & predictions.threshold_strategy.eq(strategy)
                        & predictions.family.eq(family) & predictions.requested_level.eq(float(level))
                    ].copy()
                    interval = _bootstrap_delta(
                        noisy, clean,
                        left_prediction="prediction", right_prediction="prediction",
                        seed=plan.seed + stable_seed("a1-noise-paired-ci", model, strategy, family, level),
                        replicates=bootstrap_replicates,
                    )
                    paired_rows.append({
                        "model_name": model, "threshold_strategy": strategy,
                        "family": family, "requested_level": float(level),
                        "comparison": "noise input minus matched clean input",
                        **interval,
                    })
                    if model == "A1_morphology_logistic":
                        temporal_rows = predictions.loc[
                            predictions.model_name.eq("time_only_logistic")
                            & predictions.threshold_strategy.eq(strategy)
                            & predictions.family.eq(family)
                            & predictions.requested_level.eq(float(level))
                        ].copy()
                        baseline_ci = _bootstrap_delta(
                            noisy, temporal_rows,
                            left_prediction="prediction", right_prediction="prediction",
                            seed=plan.seed + stable_seed("a1-noise-temporal-ci", strategy, family, level),
                            replicates=bootstrap_replicates,
                        )
                        paired_rows.append({
                            "model_name": model, "threshold_strategy": strategy,
                            "family": family, "requested_level": float(level),
                            "comparison": "A1 minus time-only logistic under matched condition",
                            **baseline_ci,
                        })

    prediction_path = output_dir / "a1_noise_oof_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    metrics_frame.to_csv(output_dir / "a1_noise_metrics.csv", index=False)
    pd.DataFrame(group_metric_parts).to_csv(output_dir / "a1_noise_group_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output_dir / "a1_noise_paired_group_bootstrap.csv", index=False)
    input_paths = [
        a1_manifest_path,
        private_dir / "features_labels" / "pseudo_labels_alternatives.parquet",
        bank_manifest_path,
        signals_path,
        metadata_path,
        threshold_dir / "fold_manifest.json",
        threshold_dir / "nested_oof_predictions.parquet",
    ] + checkpoint_paths
    output_paths = [
        prediction_path,
        output_dir / "a1_noise_metrics.csv",
        output_dir / "a1_noise_group_metrics.csv",
        output_dir / "a1_noise_paired_group_bootstrap.csv",
    ]
    receipt = {
        "stage": "development_nested_a1_matched_noise_robustness",
        "status": "completed",
        "label_rule": "low_frequency_fraction_persistence_v1",
        "label_rule_type": "weak_supervision_proxy_not_independent_fatigue_ground_truth",
        "outer_group_folds": 4,
        "inner_group_folds": 3,
        "models": ["A1_morphology_logistic", "time_only_logistic"],
        "conditions": [{"family": family, "requested_level": float(level), "stress_test": bool(level > 0.20)} for family, level in [("clean", 0.0)] + [(f, l) for f in CORRUPTION_FAMILIES for l in EVALUATION_LEVELS]],
        "noise_generation": "same deterministic corruption families, levels, and stable seeds as the neural matched-noise stage",
        "feature_recomputation": "vectorized morphology features from corrupted processed signal; model fitted on clean outer-training groups",
        "labels_changed_under_noise": False,
        "paired_groups_windows_labels_and_folds": True,
        "internal_test_signal_values_parsed": 0,
        "reserved_test_member_payloads_read": 0,
        "external_rar_opened": False,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_unit": "paired source group; overlapping windows are not treated as independent",
        "input_sha256": {str(path.resolve()): _sha256_file(path) for path in input_paths},
        "output_sha256": {str(path.resolve()): _sha256_file(path) for path in output_paths},
        "interpretation": "robustness of agreement with the weak low-frequency EMG proxy; not physiological fatigue validation",
    }
    manifest_path = output_dir / "a1_noise_robustness_stage_manifest.json"
    _write_json(manifest_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path.resolve())}
