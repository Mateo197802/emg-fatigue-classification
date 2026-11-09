"""Participant-held-out benchmark for time-stamped self-report fatigue labels."""

from __future__ import annotations

import json
import hashlib
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


MODEL_CONFIGS = {
    "majority": {"family": "DummyClassifier", "strategy": "prior"},
    "logistic_regression": {
        "family": "StandardScaler + LogisticRegression",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 2000,
    },
    "svm_rbf": {
        "family": "StandardScaler + SVC(kernel=rbf)",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "training_sample_cap_per_subject_trial_class": 50,
    },
    "random_forest": {
        "family": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
    },
    "hist_gradient_boosting": {
        "family": "HistGradientBoostingClassifier",
        "max_iter": 150,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    },
    "mlp_compact": {
        "family": "StandardScaler + MLPClassifier",
        "hidden_layer_sizes": [32, 16],
        "alpha": 0.001,
        "batch_size": 256,
        "max_iter": 200,
        "early_stopping": True,
    },
    "time_only_control": {
        "family": "StandardScaler + LogisticRegression",
        "features": ["time_fraction"],
        "purpose": "diagnostic shortcut control; never used in the EMG classifier",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 2000,
    },
}


def equalized_training_sample(frame, *, max_per_trial_class=50, seed=20261109):
    """Deterministically cap rows within each subject/trial/class cell.

    Even temporal spacing reduces the dominance of long, highly overlapping
    segments. This is applied only to outer-training groups. All outer-test
    windows are retained for evaluation.
    """
    required = {"subject_id", "trial_id", "label", "window_start_s"}
    if not required.issubset(frame.columns):
        raise ValueError(f"training sample lacks columns: {sorted(required - set(frame.columns))}")
    if max_per_trial_class < 1:
        raise ValueError("max_per_trial_class must be positive")
    selected = []
    for _, group in frame.groupby(["subject_id", "trial_id", "label"], sort=True):
        ordered = group.sort_values("window_start_s", kind="stable")
        if len(ordered) <= max_per_trial_class:
            selected.append(ordered)
            continue
        rng = np.random.default_rng(seed + sum(ord(char) for char in str(_)))
        edges = np.linspace(0, len(ordered), max_per_trial_class + 1).astype(int)
        positions = np.asarray(
            [rng.integers(edges[i], max(edges[i] + 1, edges[i + 1])) for i in range(max_per_trial_class)],
            dtype=int,
        )
        selected.append(ordered.iloc[positions])
    if not selected:
        raise ValueError("training sample is empty")
    result = pd.concat(selected, ignore_index=True)
    if result.duplicated("window_id").any():
        raise RuntimeError("training sample contains duplicate windows")
    return result


def binary_metric_summary(y_true, y_pred, scores):
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(scores, dtype=float)
    if y.ndim != 1 or pred.shape != y.shape or score.shape != y.shape:
        raise ValueError("truth, predictions and scores must be matching one-dimensional arrays")
    if not set(np.unique(y)).issubset({0, 1}) or not set(np.unique(pred)).issubset({0, 1}):
        raise ValueError("binary metrics require class 0 and 1 labels")
    if not np.isfinite(score).all():
        raise ValueError("scores must be finite")
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    both_classes = set(np.unique(y)) == {0, 1}
    return {
        "n_windows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_fatigue": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_fatigue": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "sensitivity_fatigue": float(sensitivity),
        "specificity_no_fatigue": float(specificity),
        "roc_auc": float(roc_auc_score(y, score)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, score)) if both_classes else float("nan"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def leave_one_participant_out(frame):
    """Yield deterministic leave-one-participant-out outer folds."""
    if "subject_id" not in frame or frame["subject_id"].isna().any():
        raise ValueError("every row needs a verified participant identifier")
    subjects = sorted(frame["subject_id"].astype(str).unique())
    if len(subjects) < 4:
        raise ValueError("at least four independent participants are required")
    groups = frame["subject_id"].astype(str).to_numpy()
    for subject in subjects:
        test_indices = np.flatnonzero(groups == subject)
        train_indices = np.flatnonzero(groups != subject)
        train_subjects = sorted(set(groups[train_indices]))
        if subject in train_subjects or not len(test_indices) or not len(train_indices):
            raise RuntimeError("participant-held-out split is invalid")
        yield {
            "test_subject": subject,
            "train_subjects": train_subjects,
            "train_indices": train_indices,
            "test_indices": test_indices,
        }


def _make_estimator(name, seed):
    if name in {"logistic_regression", "time_only_control"}:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
        )
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                cache_size=1024,
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
    if name == "mlp_compact":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=0.001,
                batch_size=256,
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=seed,
            ),
        )
    if name == "majority":
        return DummyClassifier(strategy="prior")
    raise ValueError(f"unknown model: {name}")


def _predict(estimator, X, y_pred=None):
    if y_pred is None:
        y_pred = estimator.predict(X)
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        classes = np.asarray(estimator.classes_)
        score = probabilities[:, list(classes).index(1)]
        score_type = "native_probability_uncalibrated"
    elif hasattr(estimator, "decision_function"):
        score = np.asarray(estimator.decision_function(X), dtype=float)
        score_type = "uncalibrated_svm_margin"
    else:
        raise ValueError("model must provide a probability or decision margin")
    return np.asarray(y_pred, dtype=int), score, score_type


def _participant_bootstrap(rows, *, replicates=5000, seed=20261109):
    rng = np.random.default_rng(seed)
    values = {
        subject: binary_metric_summary(
            group["y_true"], group["y_pred"], group["score"]
        )
        for subject, group in rows.groupby("subject_id", sort=True)
    }
    subjects = list(values)
    summaries = {}
    for metric in ("accuracy", "f1_macro", "balanced_accuracy", "sensitivity_fatigue", "specificity_no_fatigue"):
        vector = np.asarray([values[subject][metric] for subject in subjects], dtype=float)
        vector = vector[np.isfinite(vector)]
        if not len(vector):
            summaries[f"{metric}_participant_macro"] = float("nan")
            summaries[f"{metric}_ci95_low"] = float("nan")
            summaries[f"{metric}_ci95_high"] = float("nan")
            continue
        draws = rng.choice(vector, size=(replicates, len(vector)), replace=True).mean(axis=1)
        summaries[f"{metric}_participant_macro"] = float(vector.mean())
        summaries[f"{metric}_ci95_low"] = float(np.quantile(draws, 0.025))
        summaries[f"{metric}_ci95_high"] = float(np.quantile(draws, 0.975))
    return summaries, values


def _append_confusions(model, oof):
    output = []
    subsets = [("pooled_oof_windows", "all", oof)]
    for subject, group in oof.groupby("subject_id", sort=True):
        subsets.append(("participant_oof_windows", subject, group))
    for scope, subject, group in subsets:
        matrix = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1])
        totals = matrix.sum(axis=1)
        for actual in (0, 1):
            for predicted in (0, 1):
                count = int(matrix[actual, predicted])
                output.append({
                    "architecture": model,
                    "scope": scope,
                    "subject_id": subject,
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "count": count,
                    "row_normalized": float(count / totals[actual]) if totals[actual] else float("nan"),
                    "n_windows": int(len(group)),
                })
    return output


def run_benchmark(features_path, output_dir, *, sample_cap=50, seed=20261109):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(features_path)
    schema_path = output / "feature_schema.json"
    audit_path = output / "dataset_audit.json"
    if not schema_path.exists() or not audit_path.exists():
        raise ValueError("dataset preparation receipts are missing")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    dataset_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    feature_columns = list(schema["feature_order"])
    required = {"window_id", "subject_id", "trial_id", "label", "window_start_s", "window_center_s"}
    if not required.issubset(frame.columns):
        raise ValueError(f"feature table lacks metadata: {sorted(required - set(frame.columns))}")
    if not set(frame["label"].unique()).issubset({0, 1}) or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("perceived-fatigue target must contain both binary classes [0, 1]")
    if frame["subject_id"].nunique() != 13:
        raise ValueError("expected the 13 verified source participants")
    X_features = frame[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(X_features).all():
        raise ValueError("model input features contain nonfinite values")

    coverage_path = output / "trial_coverage.csv"
    coverage = pd.read_csv(coverage_path)
    duration = {
        (str(row.subject_id), str(row.trial_id)): float(row.raw_end_s)
        for row in coverage.itertuples()
    }
    frame = frame.copy()
    frame["time_fraction"] = [
        float(center) / duration[(str(subject), str(trial))]
        if duration[(str(subject), str(trial))] > 0
        else float("nan")
        for center, subject, trial in zip(
            frame["window_center_s"], frame["subject_id"], frame["trial_id"]
        )
    ]
    if not np.isfinite(frame["time_fraction"].to_numpy(dtype=float)).all():
        raise ValueError("time-only diagnostic feature is nonfinite")

    model_names = list(MODEL_CONFIGS)
    predictions = []
    fold_rows = []
    log_rows = []
    for fold_number, split in enumerate(leave_one_participant_out(frame), start=1):
        train = frame.iloc[split["train_indices"]].copy()
        test = frame.iloc[split["test_indices"]].copy()
        if set(train["subject_id"]) & set(test["subject_id"]):
            raise RuntimeError("outer split contains participant leakage")
        train_sample = equalized_training_sample(
            train, max_per_trial_class=sample_cap, seed=seed + fold_number
        )
        if set(train_sample["label"].unique()) != {0, 1}:
            raise ValueError(f"training fold {fold_number} lacks both perceived-fatigue classes")
        fold_id = f"LOSO_{fold_number:02d}"
        fold_rows.append({
            "fold_id": fold_id,
            "train_subjects": split["train_subjects"],
            "test_subject": split["test_subject"],
            "train_windows_full": int(len(train)),
            "train_windows_used": int(len(train_sample)),
            "test_windows": int(len(test)),
            "train_class_counts_sampled": {
                str(int(k)): int(v) for k, v in train_sample["label"].value_counts().sort_index().items()
            },
            "test_class_counts": {
                str(int(k)): int(v) for k, v in test["label"].value_counts().sort_index().items()
            },
        })
        print(
            f"Outer fold {fold_number}/13: hold out {split['test_subject']}; "
            f"train {len(train_sample)} sampled windows, test {len(test)} windows",
            flush=True,
        )
        X_test_full = test[feature_columns]
        y_test = test["label"].to_numpy(dtype=int)
        for model_index, name in enumerate(model_names):
            if name == "time_only_control":
                input_columns = ["time_fraction"]
            else:
                input_columns = feature_columns
            estimator = _make_estimator(name, seed + fold_number * 100 + model_index)
            X_train = train_sample[input_columns]
            y_train = train_sample["label"].to_numpy(dtype=int)
            X_test = test[input_columns]
            started = time.perf_counter()
            caught = []
            try:
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    estimator.fit(X_train, y_train)
                    caught = [f"{type(item.message).__name__}: {item.message}" for item in observed]
                y_pred, scores, score_type = _predict(estimator, X_test)
            except Exception as error:
                log_rows.append({
                    "fold_id": fold_id,
                    "architecture": name,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "train_rows_used": int(len(train_sample)),
                })
                print(f"  {name} failed: {type(error).__name__}: {error}", flush=True)
                continue
            elapsed = time.perf_counter() - started
            metrics = binary_metric_summary(y_test, y_pred, scores)
            log_rows.append({
                "fold_id": fold_id,
                "architecture": name,
                "status": "completed",
                "train_rows_used": int(len(train_sample)),
                "test_rows": int(len(test)),
                "seconds": float(elapsed),
                "warnings": caught,
                "metrics": metrics,
            })
            for row_index, truth, prediction, score in zip(
                test.index, y_test, y_pred, scores
            ):
                source_row = frame.loc[row_index]
                predictions.append({
                    "architecture": name,
                    "fold_id": fold_id,
                    "row_index": int(row_index),
                    "window_id": str(source_row["window_id"]),
                    "subject_id": str(source_row["subject_id"]),
                    "trial_id": str(source_row["trial_id"]),
                    "y_true": int(truth),
                    "y_pred": int(prediction),
                    "score": float(score),
                    "score_type": score_type,
                })
            print(
                f"  {name}: Accuracy={metrics['accuracy']:.4f}, "
                f"macro-F1={metrics['f1_macro']:.4f}, {elapsed:.1f}s",
                flush=True,
            )

    oof = pd.DataFrame(predictions)
    if oof.empty:
        raise RuntimeError("no models produced out-of-fold predictions")
    completed_models = sorted(oof["architecture"].unique())
    expected_rows = len(frame)
    row_counts = oof.groupby("architecture").size()
    if not (row_counts == expected_rows).all():
        raise RuntimeError(f"models do not share complete OOF support: {row_counts.to_dict()}")
    if oof.duplicated(["architecture", "window_id"]).any():
        raise RuntimeError("duplicate model/window OOF prediction")
    if len(completed_models) < 6:
        raise RuntimeError(f"fewer than six model families completed: {completed_models}")

    benchmark_rows = []
    participant_rows = []
    confusion_rows = []
    for model in completed_models:
        model_oof = oof[oof["architecture"] == model].copy()
        pooled = binary_metric_summary(model_oof["y_true"], model_oof["y_pred"], model_oof["score"])
        participant_summary, participant_metrics = _participant_bootstrap(
            model_oof, replicates=5000, seed=seed + completed_models.index(model)
        )
        benchmark_rows.append({
            "architecture": model,
            "scope": "pooled_oof_windows",
            **pooled,
            "participants": int(model_oof["subject_id"].nunique()),
            **participant_summary,
        })
        for subject, metrics in participant_metrics.items():
            participant_rows.append({"architecture": model, "subject_id": subject, **metrics})
        confusion_rows.extend(_append_confusions(model, model_oof))

    # Include a compact model-comparison table in the run metadata. The OOF
    # benchmark remains primary; selecting the best row does not alter a test fold.
    result_path = output / "benchmark_results.csv"
    pd.DataFrame(benchmark_rows).to_csv(result_path, index=False)
    pd.DataFrame(participant_rows).to_csv(output / "benchmark_by_participant.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output / "confusion_matrices.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    (output / "splits.json").write_text(
        json.dumps(fold_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "training_log.txt").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in log_rows) + "\n",
        encoding="utf-8",
    )
    model_selection = {
        "selection_method": "fixed a priori family settings; no hyperparameter selection used outer held-out participants",
        "outer_validation": "leave-one-participant-out; 13 participant-disjoint folds",
        "training_sample_strategy": "evenly spaced max 50 windows per participant x trial x class in each outer training fold; applied identically to all models; all test windows retained",
        "random_seed": int(seed),
        "feature_inputs": feature_columns,
        "time_only_control_uses_only": ["time_fraction"],
        "model_configs": MODEL_CONFIGS,
        "completed_models": completed_models,
        "failed_model_folds": [row for row in log_rows if row["status"] == "failed"],
        "model_selection_caveat": "The highest OOF score is descriptive; it is not a nested unbiased estimate for a subsequently selected model.",
    }
    (output / "model_selection.json").write_text(
        json.dumps(model_selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {
        "run_id": "run_001",
        "dataset_id": dataset_audit["dataset_id"],
        "dataset_version": dataset_audit["version"],
        "task_id": "PERCEIVED_FATIGUE_CLASSIFICATION",
        "prediction_unit": dataset_audit["prediction_unit"],
        "label_definition_id": dataset_audit["label_definition_id"],
        "seed": int(seed),
        "code_revision": code_revision or "unavailable",
        "dataset_source_hashes": {
            "emg_md5": dataset_audit["emg_archive"]["md5"],
            "emg_sha256": dataset_audit["emg_archive"]["sha256"],
            "labels_md5": dataset_audit["label_archive"]["md5"],
            "labels_sha256": dataset_audit["label_archive"]["sha256"],
        },
        "feature_table": str(Path(features_path).name),
        "feature_rows": int(len(frame)),
        "feature_table_sha256": json.loads(
            (output / "features_manifest.json").read_text(encoding="utf-8")
        )["parquet_sha256"],
        "models_completed": completed_models,
        "outer_folds": len(fold_rows),
        "predictions_sha256": hashlib.sha256(
            (output / "oof_predictions.csv").read_bytes()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "confusion_sha256": hashlib.sha256(
            (output / "confusion_matrices.csv").read_bytes()
        ).hexdigest(),
        "phase_reference_available": False,
        "mechanical_force_reference_available": False,
        "scientific_scope": "self-perceived fatigue in 13 participants performing upper-limb trials",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return pd.DataFrame(benchmark_rows), manifest
