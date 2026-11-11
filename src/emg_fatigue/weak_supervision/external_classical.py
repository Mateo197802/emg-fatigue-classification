"""Nested participant-grouped classical baselines for Cerqueira v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .external_supervision import (
    CLASS_IDS,
    _apply_temperature,
    _fit_temperature,
    _group_class_weights,
    _group_balanced_macro_f1,
    _metrics,
    _paired_group_bootstrap,
    _require_partition_support,
    _weighted_standardize_fit,
)


MODEL_CONFIGS: dict[str, tuple[dict[str, Any], ...]] = {
    "svm_rbf": (
        {"C": 0.1, "gamma": "scale"},
        {"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
    ),
    "random_forest": (
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 10},
    ),
    "hist_gradient_boosting": (
        {"max_iter": 100, "max_leaf_nodes": 7, "learning_rate": 0.05},
        {"max_iter": 250, "max_leaf_nodes": 15, "learning_rate": 0.05},
    ),
    "mlp": (
        {"hidden_layer_sizes": (32,), "alpha": 0.01},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.1},
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _fit_model(
    family: str,
    parameters: dict[str, Any],
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    weights = _group_class_weights(frame, y)
    mean, scale = _weighted_standardize_fit(x, weights)
    standardized = (x - mean) / scale
    if family == "svm_rbf":
        estimator = SVC(
            C=float(parameters["C"]),
            kernel="rbf",
            gamma=parameters["gamma"],
            decision_function_shape="ovr",
            probability=False,
            random_state=int(seed),
        )
    elif family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            class_weight=None,
            n_jobs=1,
            random_state=int(seed),
        )
    elif family == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            learning_rate=float(parameters["learning_rate"]),
            l2_regularization=1.0,
            random_state=int(seed),
        )
    elif family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(parameters["hidden_layer_sizes"]),
            alpha=float(parameters["alpha"]),
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=250,
            early_stopping=False,
            n_iter_no_change=20,
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unsupported Cerqueira model family: {family}")
    fit_started = perf_counter()
    estimator.fit(standardized, y, sample_weight=weights)
    fit_seconds = perf_counter() - fit_started
    iterations = getattr(estimator, "n_iter_", getattr(estimator, "n_estimators_", getattr(estimator, "n_estimators", None)))
    if isinstance(iterations, np.ndarray):
        iterations = iterations.tolist()
    elif isinstance(iterations, np.generic):
        iterations = iterations.item()
    return {
        "family": family,
        "parameters": parameters,
        "estimator": estimator,
        "mean": mean,
        "scale": scale,
        "fit_time_seconds": float(fit_seconds),
        "training_iterations": iterations,
    }


def _predict_probabilities(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    standardized = (x - model["mean"]) / model["scale"]
    estimator = model["estimator"]
    if model["family"] == "svm_rbf":
        decision = np.asarray(estimator.decision_function(standardized), dtype=np.float64)
        if decision.ndim != 2 or decision.shape[1] != len(CLASS_IDS):
            raise ValueError("RBF SVM did not return one OVR decision score per Cerqueira class")
        decision -= decision.max(axis=1, keepdims=True)
        probability = np.exp(decision)
        return probability / probability.sum(axis=1, keepdims=True)
    raw = np.asarray(estimator.predict_proba(standardized), dtype=np.float64)
    output = np.zeros((len(standardized), len(CLASS_IDS)), dtype=np.float64)
    for column, class_id in enumerate(estimator.classes_):
        output[:, int(class_id)] = raw[:, column]
    if not np.isfinite(output).all() or np.any(output.sum(axis=1) <= 0):
        raise FloatingPointError(f"{model['family']} produced invalid probabilities")
    return output / output.sum(axis=1, keepdims=True)


def run_cerqueira_classical_stage(
    *,
    feature_path: Path,
    baseline_predictions_path: Path,
    output_dir: Path,
    seed: int = 731,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Tune SVM, forests, boosting and MLP with identical 4x3 subject folds."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite external classical stage: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(feature_path).reset_index(drop=True)
    baseline = pd.read_parquet(baseline_predictions_path)
    if frame["subject_id"].nunique() != 13:
        raise ValueError("Cerqueira classical stage requires all 13 participant groups")
    feature_columns = [column for column in frame.columns if str(column).startswith("ch")]
    x = frame[feature_columns].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy(dtype=np.int64)
    groups = frame["subject_id"].astype(str).to_numpy()
    sample_ids = frame["sample_id"].astype(str).to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("external EMG feature matrix contains non-finite values")

    baseline = baseline[baseline["model_name"].isin(("signal_logistic", "time_only_logistic", "trial_condition_time_logistic"))].copy()
    expected_models = {"signal_logistic", "time_only_logistic", "trial_condition_time_logistic"}
    if set(baseline["model_name"].astype(str)) != expected_models:
        raise ValueError("logistic benchmark must include signal, time-only and trial-condition controls")
    for name, rows in baseline.groupby("model_name"):
        if set(rows["sample_id"].astype(str)) != set(sample_ids) or len(rows) != len(sample_ids):
            raise ValueError(f"{name} does not cover the exact Cerqueira window bank")
        aligned = frame[["sample_id", "subject_id", "recording_id", "label"]].merge(
            rows[["sample_id", "subject_id", "recording_id", "true_label"]],
            on="sample_id",
            suffixes=("", "_baseline"),
            validate="one_to_one",
        )
        if (
            not aligned.subject_id.astype(str).eq(aligned.subject_id_baseline.astype(str)).all()
            or not aligned.recording_id.astype(str).eq(aligned.recording_id_baseline.astype(str)).all()
            or not aligned.label.astype(int).eq(aligned.true_label.astype(int)).all()
        ):
            raise ValueError(f"{name} group, trial, or pseudo-label identity mismatch")

    outer_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=int(seed))
    family_oof: dict[str, list[dict[str, Any]]] = {name: [] for name in MODEL_CONFIGS}
    group_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for outer_fold, (outer_train_index, outer_validation_index) in enumerate(outer_cv.split(x, y, groups), start=1):
        train_frame = frame.iloc[outer_train_index].reset_index(drop=True)
        validation_frame = frame.iloc[outer_validation_index].reset_index(drop=True)
        train_support = _require_partition_support(train_frame, f"classical outer {outer_fold} training")
        validation_support = _require_partition_support(validation_frame, f"classical outer {outer_fold} validation")
        train_x = x[outer_train_index]
        validation_x = x[outer_validation_index]
        train_y = y[outer_train_index]
        train_groups = groups[outer_train_index]
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=int(seed + outer_fold))
        inner_splits = list(inner_cv.split(train_x, train_y, train_groups))
        selected_for_fold: dict[str, tuple[dict[str, Any], np.ndarray, float]] = {}

        for family, configs in MODEL_CONFIGS.items():
            ranked: list[tuple[float, str, dict[str, Any], np.ndarray]] = []
            for config_index, parameters in enumerate(configs):
                inner_oof = np.full((len(train_frame), len(CLASS_IDS)), np.nan, dtype=np.float64)
                fold_support_receipts = []
                inner_fit_seconds = 0.0
                for inner_fold, (inner_train_index, inner_validation_index) in enumerate(inner_splits, start=1):
                    inner_train = train_frame.iloc[inner_train_index].reset_index(drop=True)
                    inner_validation = train_frame.iloc[inner_validation_index].reset_index(drop=True)
                    inner_train_support = _require_partition_support(inner_train, f"{family} outer {outer_fold} inner {inner_fold} training")
                    inner_validation_support = _require_partition_support(inner_validation, f"{family} outer {outer_fold} inner {inner_fold} validation")
                    fitted = _fit_model(
                        family,
                        parameters,
                        inner_train,
                        train_x[inner_train_index],
                        train_y[inner_train_index],
                        seed=int(seed + outer_fold * 10000 + inner_fold * 100 + config_index),
                    )
                    inner_fit_seconds += fitted["fit_time_seconds"]
                    inner_oof[inner_validation_index] = _predict_probabilities(fitted, train_x[inner_validation_index])
                    fold_support_receipts.append({
                        "inner_fold_id": inner_fold,
                        "train_participants": sorted(inner_train.subject_id.astype(str).unique()),
                        "validation_participants": sorted(inner_validation.subject_id.astype(str).unique()),
                        "train_support": inner_train_support,
                        "validation_support": inner_validation_support,
                    })
                if not np.isfinite(inner_oof).all():
                    raise RuntimeError(f"{family} outer {outer_fold} has incomplete inner OOF probabilities")
                score = _group_balanced_macro_f1(train_y, np.argmax(inner_oof, axis=1), train_groups)
                config_key = json.dumps(parameters, sort_keys=True, default=list)
                ranked.append((score, config_key, parameters, inner_oof))
                candidate_rows.append({
                    "model_name": family,
                    "outer_fold_id": outer_fold,
                    "parameters_json": config_key,
                    "inner_oof_group_balanced_macro_f1": score,
                    "inner_oof_participant_count": int(train_frame.subject_id.nunique()),
                    "inner_fit_time_seconds_total": inner_fit_seconds,
                    "inner_fold_support_receipts_json": json.dumps(fold_support_receipts, separators=(",", ":")),
                    "selection_source": "concatenated inner-fold held-out probabilities only",
                })

            score, config_key, parameters, selected_oof = max(ranked, key=lambda item: item[0])
            temperature = _fit_temperature(selected_oof, train_y, train_groups.tolist())
            selected_for_fold[family] = (parameters, selected_oof, temperature)
            final_model = _fit_model(family, parameters, train_frame, train_x, train_y, seed=int(seed + outer_fold * 10000 + config_index + 9000))
            predict_started = perf_counter()
            probabilities = _apply_temperature(_predict_probabilities(final_model, validation_x), temperature)
            predict_seconds = perf_counter() - predict_started
            metrics = _metrics(validation_frame["label"].to_numpy(dtype=np.int64), probabilities, validation_frame["subject_id"].astype(str).tolist())
            for row in metrics.pop("per_group"):
                group_rows.append({"outer_fold_id": outer_fold, "model_name": family, **row})
            metrics.update({
                "model_name": family,
                "outer_fold_id": outer_fold,
                "selected_parameters_json": config_key,
                "inner_oof_temperature": temperature,
                "outer_fit_time_seconds": final_model["fit_time_seconds"],
                "outer_prediction_time_seconds": float(predict_seconds),
                "estimator_training_iterations": final_model["training_iterations"],
                "outer_train_support_json": json.dumps(train_support, separators=(",", ":")),
                "outer_validation_support_json": json.dumps(validation_support, separators=(",", ":")),
            })
            outer_rows.append(metrics)
            for row_index, (_, record) in enumerate(validation_frame.iterrows()):
                family_oof[family].append({
                    "sample_id": record["sample_id"],
                    "subject_id": record["subject_id"],
                    "recording_id": record["recording_id"],
                    "trial_number": int(record["trial_number"]),
                    "outer_fold_id": outer_fold,
                    "model_name": family,
                    "true_label": int(record["label"]),
                    "prediction": int(np.argmax(probabilities[row_index])),
                    **{f"probability_class_{class_id}": float(probabilities[row_index, class_id]) for class_id in CLASS_IDS},
                    "selected_parameters_json": config_key,
                    "inner_oof_calibration_temperature": temperature,
                })
            joblib.dump({
                "model": final_model,
                "feature_columns": feature_columns,
                "class_ids": CLASS_IDS,
                "class_names": {0: "NO_FATIGUE", 1: "TRANSITION", 2: "FATIGUE"},
                "selected_parameters": parameters,
                "temperature": temperature,
                "training_time_seconds": final_model["fit_time_seconds"],
                "training_iterations": final_model["training_iterations"],
                "outer_fold_id": outer_fold,
                "training_participants": sorted(train_frame.subject_id.astype(str).unique()),
                "validation_participants": sorted(validation_frame.subject_id.astype(str).unique()),
                "label_rule": "cerqueira_v2_self_perceived_fatigue_state_0_1_2",
            }, output_dir / f"cerqueira_{family}_outer_fold_{outer_fold}.joblib")
        print(f"Completed classical Cerqueira outer fold {outer_fold}/4", flush=True)

    # Check the same deterministic outer groups were used as the logistic/control benchmark.
    predictions = pd.DataFrame([record for family_rows in family_oof.values() for record in family_rows])
    baseline_rows = baseline[baseline.model_name.isin(("time_only_logistic", "trial_condition_time_logistic", "signal_logistic"))]
    outer_fold_map = baseline_rows[baseline_rows.model_name.eq("signal_logistic")].set_index("sample_id").outer_fold_id.astype(int)
    if not predictions.sample_id.map(outer_fold_map).eq(predictions.outer_fold_id).all():
        raise ValueError("classical and logistic outer participant folds differ")
    predictions.to_parquet(output_dir / "cerqueira_classical_nested_oof_predictions.parquet", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "cerqueira_classical_group_metrics.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(output_dir / "cerqueira_classical_outer_fold_metrics.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output_dir / "cerqueira_classical_inner_candidate_scores.csv", index=False)

    baseline_groups: dict[str, pd.DataFrame] = {}
    for baseline_name in expected_models:
        subset = baseline[baseline.model_name.eq(baseline_name)]
        probabilities = subset[[f"probability_class_{class_id}" for class_id in CLASS_IDS]].to_numpy(dtype=np.float64)
        baseline_metrics = _metrics(subset.true_label.to_numpy(dtype=np.int64), probabilities, subset.subject_id.astype(str).tolist())
        baseline_groups[baseline_name] = pd.DataFrame(baseline_metrics["per_group"])
    compared_group_rows = pd.DataFrame(group_rows)
    paired_rows = []
    for model_name in MODEL_CONFIGS:
        model_groups = compared_group_rows[compared_group_rows.model_name.eq(model_name)]
        for baseline_name, baseline_group_frame in baseline_groups.items():
            for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
                paired_rows.append({
                    "comparison": f"{model_name}_minus_{baseline_name}",
                    **_paired_group_bootstrap(model_groups, baseline_group_frame, metric=metric, replicates=bootstrap_replicates, seed=seed + len(paired_rows)),
                })
    pd.DataFrame(paired_rows).to_csv(output_dir / "cerqueira_classical_paired_group_bootstrap.csv", index=False)
    summary_rows = []
    for model_name in MODEL_CONFIGS:
        subset = predictions[predictions.model_name.eq(model_name)]
        metrics = _metrics(
            subset.true_label.to_numpy(dtype=np.int64),
            subset[[f"probability_class_{class_id}" for class_id in CLASS_IDS]].to_numpy(dtype=np.float64),
            subset.subject_id.astype(str).tolist(),
        )
        metrics.pop("per_group", None)
        metrics["model_name"] = model_name
        summary_rows.append(metrics)
    pd.DataFrame(summary_rows).to_csv(output_dir / "cerqueira_classical_metrics_summary.csv", index=False)

    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir(), key=lambda value: value.name.casefold())
        if path.is_file()
    }
    manifest = {
        "status": "completed",
        "dataset_key": "cerqueira_v2",
        "target": "source-reported self-perceived state 0/1/2; not physiological ground truth",
        "outer_folds": 4,
        "inner_folds": 3,
        "independent_unit": "participant",
        "models": {name: list(configs) for name, configs in MODEL_CONFIGS.items()},
        "hyperparameter_selection": "participant-grouped inner OOF group-balanced Macro-F1",
        "calibration": "participant/class balanced temperature scaling fit only to selected inner OOF probabilities",
        "baseline_predictions_sha256": _sha256(baseline_predictions_path),
        "feature_bank_sha256": _sha256(feature_path),
        "window_count_is_independent_n": False,
        "locked_course_internal_test_values_parsed": 0,
        "course_raw2_rar_opened": False,
        "artifact_sha256": artifacts,
    }
    _write_json(output_dir / "cerqueira_classical_stage_manifest.json", manifest)
    return {"status": "completed", "manifest_path": str((output_dir / "cerqueira_classical_stage_manifest.json").resolve()), "summary": pd.DataFrame(summary_rows).to_dict(orient="records")}
