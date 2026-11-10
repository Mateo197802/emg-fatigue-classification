"""Nested participant-held-out benchmark for task-level RF trial outcomes."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.svm import SVC

from experiments.reference_fatigue.figshare_trial_model import (
    DATASET_DOI,
    SEED,
    inner_group_splits,
    outer_leave_one_subject_out,
)


TIME_FEATURES = [
    "total_duration_s",
    "active_duration_s",
    "active_to_total_ratio",
    "active_bout_count",
    "bout_duration_median_s",
    "bout_duration_max_s",
]


def metric_summary(y_true, y_pred, scores) -> dict[str, float | int]:
    """Calculate trial-level binary metrics from one prediction per trial."""
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(scores, dtype=float)
    if y.ndim != 1 or pred.shape != y.shape or score.shape != y.shape or not len(y):
        raise ValueError("truth, predictions and scores must be matching nonempty vectors")
    if not set(np.unique(y)).issubset({0, 1}) or not set(np.unique(pred)).issubset({0, 1}):
        raise ValueError("binary metrics require labels 0 and 1")
    if not np.isfinite(score).all():
        raise ValueError("prediction scores must be finite")
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    both_classes = set(np.unique(y)) == {0, 1}
    return {
        "n_trials": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_task_failure": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_task_failure": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "sensitivity_task_failure": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity_reference": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "roc_auc": float(roc_auc_score(y, score)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, score)) if both_classes else float("nan"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame if column.endswith(("__median", "__iqr", "__slope"))]


def _rms_mdf_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("rms_log10__", "mdf_hz__")
    return [column for column in _feature_columns(frame) if column.startswith(prefixes)]


def _estimator(name: str, params: dict, seed: int):
    if name == "dummy_prior":
        return DummyClassifier(strategy="prior")
    if name in {"lr_rms_mdf", "lr_all_features", "time_only_control", "emg_plus_time_control"}:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params.get("C", 1.0)),
                class_weight="balanced",
                max_iter=4000,
                random_state=seed,
            ),
        )
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=float(params.get("C", 1.0)),
                gamma=params.get("gamma", "scale"),
                class_weight="balanced",
                cache_size=256,
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 75)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 3)),
            l2_regularization=float(params.get("l2_regularization", 1.0)),
            early_stopping=False,
            random_state=seed,
        )
    if name == "mlp_compact":
        hidden = tuple(params.get("hidden_layer_sizes", (8,)))
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                alpha=float(params.get("alpha", 0.01)),
                solver="lbfgs",
                max_iter=1000,
                early_stopping=False,
                tol=1e-4,
                random_state=seed,
            ),
        )
    raise ValueError(f"unsupported tabular model family: {name}")


def _candidate_grid(name: str) -> list[dict]:
    if name == "dummy_prior":
        return [{}]
    if name in {"lr_rms_mdf", "lr_all_features", "time_only_control", "emg_plus_time_control"}:
        return [{"C": value} for value in (0.1, 1.0, 10.0)]
    if name == "svm_rbf":
        return [
            {"C": c, "gamma": gamma}
            for c, gamma in itertools.product((0.1, 1.0, 10.0), ("scale", 0.1))
        ]
    if name == "random_forest":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth, leaf in itertools.product((2, 4, None), (1, 3))
        ]
    if name == "hist_gradient_boosting":
        return [
            {"max_iter": iterations, "max_leaf_nodes": leaves, "l2_regularization": l2}
            for iterations, leaves, l2 in itertools.product((50, 100), (3, 7), (1.0, 10.0))
        ]
    if name == "mlp_compact":
        return [
            {"hidden_layer_sizes": hidden, "alpha": alpha}
            for hidden, alpha in itertools.product(((8,), (16, 8)), (0.01, 0.1))
        ]
    raise ValueError(f"unsupported model family: {name}")


def _predict_score(estimator, X) -> tuple[np.ndarray, np.ndarray, str]:
    pred = np.asarray(estimator.predict(X), dtype=int)
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        classes = np.asarray(estimator.classes_)
        score = probabilities[:, int(np.flatnonzero(classes == 1)[0])]
        return pred, score, "uncalibrated_probability"
    if hasattr(estimator, "decision_function"):
        return pred, np.asarray(estimator.decision_function(X), dtype=float), "uncalibrated_margin"
    raise ValueError("selected model does not expose ranking scores")


def _model_features(name: str, frame: pd.DataFrame) -> list[str]:
    if name == "lr_rms_mdf":
        return _rms_mdf_columns(frame)
    if name in {"lr_all_features", "svm_rbf", "random_forest", "hist_gradient_boosting", "mlp_compact"}:
        return _feature_columns(frame)
    if name == "time_only_control":
        return TIME_FEATURES.copy()
    if name == "emg_plus_time_control":
        return _feature_columns(frame) + TIME_FEATURES
    raise ValueError(f"model family has no tabular input schema: {name}")


def _select_tabular_params(
    name: str,
    frame: pd.DataFrame,
    features: list[str],
    outer_train: np.ndarray,
    *,
    seed: int,
    outer_fold: int,
) -> tuple[dict, list[dict]]:
    candidates = _candidate_grid(name)
    splits = list(inner_group_splits(frame, outer_train, n_splits=3))
    history = []
    for candidate_index, params in enumerate(candidates):
        scores = []
        for inner_fold, (inner_train, inner_valid) in enumerate(splits, start=1):
            estimator = _estimator(name, params, seed + outer_fold * 1000 + inner_fold)
            x_train = frame.iloc[inner_train][features].to_numpy(dtype=float)
            y_train = frame.iloc[inner_train]["label"].to_numpy(dtype=int)
            x_valid = frame.iloc[inner_valid][features].to_numpy(dtype=float)
            y_valid = frame.iloc[inner_valid]["label"].to_numpy(dtype=int)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(x_train, y_train)
            prediction = np.asarray(estimator.predict(x_valid), dtype=int)
            score = float(f1_score(y_valid, prediction, labels=[0, 1], average="macro", zero_division=0))
            scores.append(score)
            history.append({
                "outer_fold": outer_fold,
                "model": name,
                "candidate_index": candidate_index,
                "params": json.dumps(params, sort_keys=True),
                "inner_fold": inner_fold,
                "inner_train_subjects": json.dumps(sorted(set(frame.iloc[inner_train]["subject_id"].astype(str)))),
                "inner_valid_subjects": json.dumps(sorted(set(frame.iloc[inner_valid]["subject_id"].astype(str)))),
                "inner_valid_n_trials": int(len(inner_valid)),
                "inner_macro_f1": score,
                "fit_warning_count": len(caught),
                "fit_warnings": json.dumps([str(item.message) for item in caught]),
            })
        mean_score = float(np.mean(scores))
        history[-len(splits)]["candidate_mean_macro_f1"] = mean_score
        for row in history[-len(splits):]:
            row["candidate_mean_macro_f1"] = mean_score
    means = [np.mean([row["inner_macro_f1"] for row in history if row["candidate_index"] == i]) for i in range(len(candidates))]
    best_index = int(np.argmax(means))
    return candidates[best_index], history


def _torch_components():
    import torch
    from torch import nn

    class TrialMILCNN(nn.Module):
        """Compact temporal CNN; trial logits are mean-pooled across windows."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(65, 16, kernel_size=1),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv1d(16, 16, kernel_size=9, stride=2, padding=4, groups=16),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv1d(16, 32, kernel_size=1),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=3, groups=32),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(p=0.35),
                nn.Linear(64, 1),
            )

        def forward(self, bags):
            if bags.ndim != 4 or bags.shape[2] != 65:
                raise ValueError("CNN input must have shape trials x windows x 65 x samples")
            batch, windows, channels, samples = bags.shape
            logits = self.features(bags.reshape(batch * windows, channels, samples))
            return logits.reshape(batch, windows)

    return torch, nn, TrialMILCNN


def _load_trial_bags(frame: pd.DataFrame, dataset_dir: Path) -> list[dict]:
    bags = []
    for row in frame.itertuples(index=False):
        path = Path(dataset_dir) / "trial_bags" / f"{row.subject_id}_{row.condition}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing private CNN trial bag: {path}")
        with np.load(path) as data:
            windows = np.asarray(data["windows"], dtype=np.float32)
            valid = np.asarray(data["channel_valid"], dtype=bool)
            channel_rms = np.asarray(data["channel_rms"], dtype=np.float32)
        if windows.ndim != 3 or windows.shape[1] != 65 or not valid.shape == (65,):
            raise ValueError(f"invalid raw trial bag: {path.name}")
        bags.append({"windows": windows, "valid": valid, "channel_rms": channel_rms})
    return bags


def _training_channel_scale(train_indices: np.ndarray, bags: list[dict]) -> np.ndarray:
    by_channel = [[] for _ in range(65)]
    for index in train_indices:
        bag = bags[int(index)]
        for channel in np.flatnonzero(bag["valid"]):
            value = float(bag["channel_rms"][channel])
            if np.isfinite(value) and value > 1e-12:
                by_channel[int(channel)].append(value)
    scale = np.ones(65, dtype=np.float32)
    for channel, values in enumerate(by_channel):
        if values:
            scale[channel] = max(float(np.median(values)), 1e-10)
    return scale


def _normalize_trial_bag(bag: dict, channel_scale: np.ndarray, *, window_cap: int = 8) -> np.ndarray:
    windows = bag["windows"]
    selection = np.linspace(0, len(windows) - 1, min(window_cap, len(windows)), dtype=int)
    selected = windows[selection].astype(np.float32, copy=True)
    selected /= channel_scale[None, :, None]
    selected[:, ~bag["valid"], :] = 0.0
    selected = np.clip(selected, -10.0, 10.0)
    selected -= selected.mean(axis=-1, keepdims=True)
    if not np.isfinite(selected).all():
        raise ValueError("normalized CNN trial bag contains nonfinite values")
    return selected


def _fit_cnn(
    frame: pd.DataFrame,
    bags: list[dict],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    *,
    epochs: int,
    seed: int,
    outer_fold: int,
    inner_fold: int | None,
) -> tuple[object, np.ndarray, np.ndarray, list[dict]]:
    torch, _, model_class = _torch_components()
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    scale = _training_channel_scale(train_indices, bags)
    normalized = {
        int(index): _normalize_trial_bag(bags[int(index)], scale)
        for index in np.concatenate((train_indices, valid_indices))
    }
    model = model_class()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    y = frame["label"].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    training_history = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        order = np.asarray(train_indices, dtype=int).copy()
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), 2):
            batch_indices = order[start : start + 2]
            x_batch = torch.from_numpy(np.stack([normalized[int(index)] for index in batch_indices]))
            y_batch = torch.from_numpy(y[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            trial_logits = model(x_batch).mean(dim=1)
            loss = loss_fn(trial_logits, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        training_history.append({
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "epoch": epoch,
            "training_bce": float(np.mean(losses)),
            "training_trial_count": int(len(train_indices)),
            "pooled_windows_per_trial": int(normalized[int(train_indices[0])].shape[0]),
        })

    model.eval()
    scores = []
    with torch.no_grad():
        for index in valid_indices:
            x = torch.from_numpy(normalized[int(index)][None, ...])
            trial_logit = model(x).mean(dim=1)
            scores.append(float(torch.sigmoid(trial_logit).cpu().item()))
    return model, scale, np.asarray(scores, dtype=float), training_history


def _select_cnn_epochs(
    frame: pd.DataFrame,
    bags: list[dict],
    outer_train: np.ndarray,
    *,
    seed: int,
    outer_fold: int,
) -> tuple[int, list[dict], list[dict]]:
    candidates = (5, 12)
    splits = list(inner_group_splits(frame, outer_train, n_splits=3))
    selection_history = []
    epoch_history = []
    for candidate_index, epochs in enumerate(candidates):
        scores = []
        for inner_fold, (inner_train, inner_valid) in enumerate(splits, start=1):
            _, _, prediction_score, history = _fit_cnn(
                frame,
                bags,
                inner_train,
                inner_valid,
                epochs=epochs,
                seed=seed + outer_fold * 1000 + inner_fold * 10 + candidate_index,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )
            pred = (prediction_score >= 0.5).astype(int)
            y_valid = frame.iloc[inner_valid]["label"].to_numpy(dtype=int)
            macro_f1 = float(f1_score(y_valid, pred, labels=[0, 1], average="macro", zero_division=0))
            scores.append(macro_f1)
            selection_history.append({
                "outer_fold": outer_fold,
                "model": "cnn1d_trial_mil",
                "candidate_index": candidate_index,
                "params": json.dumps({"epochs": epochs, "lr": 0.001, "weight_decay": 0.001, "pooling": "mean_window_logit"}, sort_keys=True),
                "inner_fold": inner_fold,
                "inner_train_subjects": json.dumps(sorted(set(frame.iloc[inner_train]["subject_id"].astype(str)))),
                "inner_valid_subjects": json.dumps(sorted(set(frame.iloc[inner_valid]["subject_id"].astype(str)))),
                "inner_valid_n_trials": int(len(inner_valid)),
                "inner_macro_f1": macro_f1,
                "fit_warning_count": 0,
                "fit_warnings": "[]",
            })
            for row in history:
                epoch_history.append({**row, "model": "cnn1d_trial_mil", "candidate_index": candidate_index})
        mean_score = float(np.mean(scores))
        for row in selection_history[-len(splits):]:
            row["candidate_mean_macro_f1"] = mean_score
    means = [
        float(np.mean([row["inner_macro_f1"] for row in selection_history if row["candidate_index"] == index]))
        for index in range(len(candidates))
    ]
    return candidates[int(np.argmax(means))], selection_history, epoch_history


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_intervals(predictions: pd.DataFrame, *, replicates: int = 5000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(predictions["subject_id"].astype(str).unique()))
    rows = []
    for model, model_rows in predictions.groupby("model", sort=True):
        subject_confusions = []
        for subject in subjects:
            group = model_rows[model_rows["subject_id"].astype(str) == subject]
            if len(group) != 2 or set(group["y_true"].astype(int)) != {0, 1}:
                raise ValueError(f"participant bootstrap requires a paired trial for {subject}")
            subject_confusions.append(
                confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1])
            )
        subject_confusions = np.asarray(subject_confusions, dtype=np.int64)
        selected = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
        matrices = subject_confusions[selected].sum(axis=1)
        tn = matrices[:, 0, 0].astype(float)
        fp = matrices[:, 0, 1].astype(float)
        fn = matrices[:, 1, 0].astype(float)
        tp = matrices[:, 1, 1].astype(float)
        accuracy = (tn + tp) / (2 * len(subjects))
        f1_positive = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        f1_negative = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) > 0)
        macro_f1 = (f1_positive + f1_negative) / 2
        rows.append({
            "model": model,
            "participant_cluster_bootstrap_replicates": replicates,
            "participant_count": int(len(subjects)),
            "accuracy_ci95_low": float(np.percentile(accuracy, 2.5)),
            "accuracy_ci95_high": float(np.percentile(accuracy, 97.5)),
            "f1_macro_ci95_low": float(np.percentile(macro_f1, 2.5)),
            "f1_macro_ci95_high": float(np.percentile(macro_f1, 97.5)),
        })
    return pd.DataFrame(rows)


def _per_subject_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, subject), group in predictions.groupby(["model", "subject_id"], sort=True):
        summary = metric_summary(group["y_true"], group["y_pred"], group["score"])
        rows.append({"model": model, "subject_id": subject, **summary})
    return pd.DataFrame(rows)


def _confusion_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        matrix = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1])
        for true_label in (0, 1):
            support = int(matrix[true_label].sum())
            for predicted_label in (0, 1):
                count = int(matrix[true_label, predicted_label])
                rows.append({
                    "model": model,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "row_normalized": float(count / support) if support else float("nan"),
                })
    return pd.DataFrame(rows)


def _paired_subject_comparisons(per_subject: pd.DataFrame, reference: str = "lr_rms_mdf") -> pd.DataFrame:
    wide_f1 = per_subject.pivot(index="subject_id", columns="model", values="f1_macro")
    wide_acc = per_subject.pivot(index="subject_id", columns="model", values="accuracy")
    rows = []
    for model in wide_f1.columns:
        if model == reference:
            continue
        for metric_name, wide in (("f1_macro", wide_f1), ("accuracy", wide_acc)):
            delta = (wide[model] - wide[reference]).dropna().to_numpy(dtype=float)
            if not len(delta):
                continue
            rng = np.random.default_rng(SEED + sum(ord(char) for char in model + metric_name))
            boots = np.mean(rng.choice(delta, size=(5000, len(delta)), replace=True), axis=1)
            rows.append({
                "model": model,
                "reference_model": reference,
                "metric": metric_name,
                "mean_paired_difference": float(np.mean(delta)),
                "participant_bootstrap_ci95_low": float(np.percentile(boots, 2.5)),
                "participant_bootstrap_ci95_high": float(np.percentile(boots, 97.5)),
                "participant_count": int(len(delta)),
            })
    return pd.DataFrame(rows)


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat([previous, frame], ignore_index=True, sort=False)
    frame.to_csv(path, index=False)


def finalize_saved_benchmark(dataset_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Aggregate already completed OOF predictions and seal the run manifest."""
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    predictions = pd.read_csv(output_dir / "oof_predictions.csv")
    selection = pd.read_csv(output_dir / "model_selection.csv")
    if predictions.empty or predictions.duplicated(["model", "trial_id"]).any():
        raise ValueError("OOF prediction ledger is empty or has duplicate model/trial rows")
    if predictions.groupby("model").size().ne(28).any():
        raise ValueError("every model must provide exactly 28 OOF trial predictions")
    if predictions.groupby("model")["subject_id"].nunique().ne(14).any():
        raise ValueError("every model must cover exactly 14 held-out participants")

    results = []
    for model, rows in predictions.groupby("model", sort=True):
        results.append({"model": model, **metric_summary(rows["y_true"], rows["y_pred"], rows["score"])})
    result_frame = pd.DataFrame(results)
    result_frame = result_frame.merge(_bootstrap_intervals(predictions), on="model", how="left")
    trained_selection = selection[selection["status"].astype(str).str.startswith("trained")]
    selection_means = (
        trained_selection.groupby("model")["inner_grouped_macro_f1"]
        .mean()
        .rename("mean_selected_inner_macro_f1")
    )
    result_frame = result_frame.merge(selection_means, on="model", how="left")
    result_frame = result_frame.sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    result_frame.to_csv(output_dir / "benchmark_results.csv", index=False)
    _confusion_rows(predictions).to_csv(output_dir / "confusion_matrix.csv", index=False)
    per_subject = _per_subject_scores(predictions)
    per_subject.to_csv(output_dir / "per_subject_metrics.csv", index=False)
    _paired_subject_comparisons(per_subject).to_csv(output_dir / "paired_subject_comparisons.csv", index=False)

    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("completed_outer_folds") != 14:
        raise ValueError("cannot finalize before all 14 outer folds complete")
    checkpoint_receipts = []
    for row in trained_selection.itertuples(index=False):
        checkpoint_receipts.append({
            "model": row.model,
            "held_out_subject": row.held_out_subject,
            "path": row.checkpoint_path,
            "sha256": row.checkpoint_sha256,
        })
    run_manifest.update({
        "status": "completed",
        "end_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_outer_folds": 14,
        "oof_prediction_rows": int(len(predictions)),
        "model_prediction_counts": {name: int(count) for name, count in predictions.groupby("model").size().items()},
        "checkpoint_receipts": checkpoint_receipts,
        "source_manifest_sha256": _file_sha256(dataset_dir / "dataset_manifest.json"),
        "code_sha256": {
            "figshare_trial_model.py": _file_sha256(Path(__file__).with_name("figshare_trial_model.py")),
            "figshare_trial_benchmark.py": _file_sha256(Path(__file__)),
        },
        "artifact_sha256": {},
    })
    for name in (
        "training_config.json",
        "feature_schema.json",
        "split_manifest.json",
        "oof_predictions.csv",
        "benchmark_results.csv",
        "confusion_matrix.csv",
        "per_subject_metrics.csv",
        "paired_subject_comparisons.csv",
        "training_history.csv",
        "model_selection.csv",
    ):
        path = output_dir / name
        if path.is_file():
            run_manifest["artifact_sha256"][name] = _file_sha256(path)
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result_frame, run_manifest


def run_benchmark(dataset_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Run nested grouped tuning, participant-held-out predictions, and controls."""
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    frame = pd.read_csv(dataset_dir / "trial_features.csv")
    if len(frame) != 28 or frame["subject_id"].nunique() != 14:
        raise ValueError("benchmark requires 28 paired task-level trials from 14 participants")
    if frame.duplicated(["subject_id", "condition"]).any():
        raise ValueError("benchmark input has duplicate participant-condition trials")
    if set(frame["label"].astype(int)) != {0, 1}:
        raise ValueError("benchmark input does not contain both trial outcome labels")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = dataset_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "oof_predictions.csv"
    for stale in (predictions_path, output_dir / "training_history.csv", output_dir / "model_selection.csv"):
        if stale.exists():
            stale.unlink()

    model_names = (
        "dummy_prior",
        "lr_rms_mdf",
        "lr_all_features",
        "svm_rbf",
        "random_forest",
        "hist_gradient_boosting",
        "mlp_compact",
        "time_only_control",
        "emg_plus_time_control",
    )
    neural_model = "cnn1d_trial_mil"
    feature_map = {name: _model_features(name, frame) for name in model_names if name != "dummy_prior"}
    feature_map["dummy_prior"] = _feature_columns(frame)
    for name, columns in feature_map.items():
        if not columns or not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"model input is empty or nonfinite: {name}")
    bags = _load_trial_bags(frame, dataset_dir)

    config = {
        "dataset_doi": DATASET_DOI,
        "prediction_unit": "trial; all inner/outer scores use one row per participant-condition trial",
        "label_0": "30% reference contraction condition",
        "label_1": "sustained TLim knee-extension task carried to task failure",
        "outer_validation": "leave one participant out; both paired trial conditions for that subject are held out",
        "inner_tuning": "3-fold GroupKFold on outer-training participants; grouped macro-F1 selection",
        "prediction_threshold": "0.5 for probabilities, zero for signed SVM margins",
        "seed": SEED,
        "models": {name: _candidate_grid(name) for name in model_names},
        "base_estimator_settings": {
            "svm_rbf": "StandardScaler + SVC(kernel=rbf, class_weight=balanced, decision margin)",
            "random_forest": "200 trees, balanced class weights, square-root feature subsampling",
            "hist_gradient_boosting": "early stopping disabled; inner participant groups select iterations/leaves/regularization",
            "mlp_compact": "StandardScaler + MLPClassifier(solver=lbfgs, max_iter=1000, tol=1e-4); compact hidden layers tuned within participant groups",
            "logistic_regression": "StandardScaler + balanced LogisticRegression, C selected within participant groups",
        },
        "cnn": {
            "architecture": "compact temporal depthwise CNN over 65-channel x 0.5-s windows",
            "trial_pooling": "mean window logit; binary cross-entropy computed once per trial bag",
            "windows_per_trial": "8 evenly spaced from the 32 preprocessed windows",
            "epochs_inner_candidates": [5, 12],
            "optimizer": "AdamW(lr=0.001, weight_decay=0.001)",
            "normalization": "median channel RMS fit from training participants only per fold",
            "candidate_selection_metric": "inner grouped macro-F1",
        },
        "pretrained_encoder_head": {
            "status": "not_run_incompatible",
            "reason": "the course SSL encoder was trained on a single-channel signal and unlabeled reconstruction; this external release has a 65-position electrode grid, a different sampling rate/protocol, and participant overlap with the course cohort is unverified. Channel adaptation and unverified identity overlap would invalidate a clean external comparison.",
        },
        "limits": [
            "the target distinguishes a reference protocol condition from a trial carried to task failure; it is not an instantaneous fatigue/no-fatigue label",
            "the external source uses isometric 30% peak-torque knee extension, unlike the course dynamic 70% 1RM task",
            "the uncalibrated force trace is used only for segmentation and is excluded from model features and labels",
            "only 14 complete paired participants are available; uncertainty is participant-clustered and expected to be wide",
        ],
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    schema_path = dataset_dir / "feature_schema.json"
    if schema_path.is_file():
        (output_dir / "feature_schema.json").write_bytes(schema_path.read_bytes())

    run_manifest = {
        "status": "running",
        "dataset_doi": DATASET_DOI,
        "trial_count": int(len(frame)),
        "participant_count": int(frame["subject_id"].nunique()),
        "model_count_including_diagnostic_controls": len(model_names) + 1,
        "completed_outer_folds": 0,
        "outer_fold_count": 14,
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": SEED,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    all_split_records = []
    all_selection_rows = []
    all_prediction_rows = []
    all_history_rows = []
    checkpoint_receipts = []
    outer_folds = list(outer_leave_one_subject_out(frame))
    for outer_fold, fold in enumerate(outer_folds, start=1):
        outer_train = fold["train_indices"]
        outer_test = fold["test_indices"]
        inner_records = []
        for inner_fold, (inner_train, inner_valid) in enumerate(
            inner_group_splits(frame, outer_train, n_splits=3), start=1
        ):
            inner_records.append({
                "inner_fold": inner_fold,
                "train_subjects": sorted(set(frame.iloc[inner_train]["subject_id"].astype(str))),
                "validation_subjects": sorted(set(frame.iloc[inner_valid]["subject_id"].astype(str))),
            })
        all_split_records.append({
            "outer_fold": outer_fold,
            "held_out_subject": fold["test_subject"],
            "training_subjects": fold["train_subjects"],
            "test_trial_ids": [
                f"{row.subject_id}_{row.condition}"
                for row in frame.iloc[outer_test].itertuples(index=False)
            ],
            "inner_folds": inner_records,
        })
        y_train = frame.iloc[outer_train]["label"].to_numpy(dtype=int)
        y_test = frame.iloc[outer_test]["label"].to_numpy(dtype=int)

        for name in model_names:
            features = feature_map[name]
            if name == "dummy_prior":
                params, tuning = {}, []
            else:
                params, tuning = _select_tabular_params(
                    name, frame, features, outer_train, seed=SEED, outer_fold=outer_fold
                )
            all_history_rows.extend(tuning)
            estimator = _estimator(name, params, SEED + outer_fold)
            with warnings.catch_warnings(record=True) as final_warnings:
                warnings.simplefilter("always")
                estimator.fit(frame.iloc[outer_train][features].to_numpy(dtype=float), y_train)
            x_test = frame.iloc[outer_test][features].to_numpy(dtype=float)
            prediction, score, score_type = _predict_score(estimator, x_test)
            checkpoint = checkpoint_dir / f"{name}_holdout_{fold['test_subject']}.joblib"
            joblib.dump(estimator, checkpoint)
            checkpoint_receipts.append({
                "model": name,
                "held_out_subject": fold["test_subject"],
                "path": str(checkpoint.relative_to(dataset_dir)),
                "sha256": _file_sha256(checkpoint),
            })
            all_selection_rows.append({
                "outer_fold": outer_fold,
                "held_out_subject": fold["test_subject"],
                "model": name,
                "status": "trained",
                "selected_params": json.dumps(params, sort_keys=True),
                "feature_columns": json.dumps(features),
                "inner_grouped_macro_f1": (
                    float(np.mean([
                        row["inner_macro_f1"]
                        for row in tuning
                        if row.get("candidate_index") == next(
                            (index for index, candidate in enumerate(_candidate_grid(name)) if candidate == params), -1
                        )
                    ]))
                    if tuning
                    else float("nan")
                ),
                "checkpoint_path": str(checkpoint.relative_to(dataset_dir)),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "final_fit_warning_count": len(final_warnings),
                "final_fit_warnings": json.dumps([str(item.message) for item in final_warnings]),
            })
            for local_row, index in enumerate(outer_test):
                trial = frame.iloc[int(index)]
                all_prediction_rows.append({
                    "model": name,
                    "outer_fold": outer_fold,
                    "subject_id": str(trial["subject_id"]),
                    "condition": str(trial["condition"]),
                    "trial_id": f"{trial['subject_id']}_{trial['condition']}",
                    "y_true": int(y_test[local_row]),
                    "y_pred": int(prediction[local_row]),
                    "score": float(score[local_row]),
                    "score_type": score_type,
                    "selected_params": json.dumps(params, sort_keys=True),
                })

        epochs, cnn_selection, cnn_epochs_history = _select_cnn_epochs(
            frame, bags, outer_train, seed=SEED, outer_fold=outer_fold
        )
        all_history_rows.extend(cnn_selection)
        all_history_rows.extend(cnn_epochs_history)
        cnn_model, scale, cnn_scores, final_history = _fit_cnn(
            frame,
            bags,
            outer_train,
            outer_test,
            epochs=epochs,
            seed=SEED + outer_fold * 1000,
            outer_fold=outer_fold,
            inner_fold=None,
        )
        all_history_rows.extend({**row, "model": neural_model, "candidate_index": "selected"} for row in final_history)
        cnn_prediction = (cnn_scores >= 0.5).astype(int)
        torch, _, _ = _torch_components()
        checkpoint = checkpoint_dir / f"{neural_model}_holdout_{fold['test_subject']}.pt"
        torch.save({
            "state_dict": {key: value.detach().cpu() for key, value in cnn_model.state_dict().items()},
            "channel_scale": scale,
            "config": {"epochs": epochs, "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.001, "pooling": "mean_window_logit", "seed": SEED + outer_fold * 1000},
        }, checkpoint)
        checkpoint_receipts.append({
            "model": neural_model,
            "held_out_subject": fold["test_subject"],
            "path": str(checkpoint.relative_to(dataset_dir)),
            "sha256": _file_sha256(checkpoint),
        })
        cnn_params = {"epochs": epochs, "lr": 0.001, "weight_decay": 0.001, "pooling": "mean_window_logit"}
        chosen_cnn_candidate = 0 if epochs == 5 else 1
        cnn_inner_mean = float(np.mean([
            row["inner_macro_f1"]
            for row in cnn_selection
            if row["candidate_index"] == chosen_cnn_candidate
        ]))
        all_selection_rows.append({
            "outer_fold": outer_fold,
            "held_out_subject": fold["test_subject"],
            "model": neural_model,
            "status": "trained_trial_level_mil",
            "selected_params": json.dumps(cnn_params, sort_keys=True),
            "feature_columns": "raw 65-channel RF windows; 8 windows/trial",
            "inner_grouped_macro_f1": cnn_inner_mean,
            "checkpoint_path": str(checkpoint.relative_to(dataset_dir)),
            "checkpoint_sha256": _file_sha256(checkpoint),
        })
        for local_row, index in enumerate(outer_test):
            trial = frame.iloc[int(index)]
            all_prediction_rows.append({
                "model": neural_model,
                "outer_fold": outer_fold,
                "subject_id": str(trial["subject_id"]),
                "condition": str(trial["condition"]),
                "trial_id": f"{trial['subject_id']}_{trial['condition']}",
                "y_true": int(y_test[local_row]),
                "y_pred": int(cnn_prediction[local_row]),
                "score": float(cnn_scores[local_row]),
                "score_type": "trial_probability_from_mean_window_logit",
                "selected_params": json.dumps(cnn_params, sort_keys=True),
            })

        all_selection_rows.append({
            "outer_fold": outer_fold,
            "held_out_subject": fold["test_subject"],
            "model": "pretrained_encoder_head",
            "status": "not_run_incompatible",
            "selected_params": "{}",
            "feature_columns": "",
            "inner_grouped_macro_f1": float("nan"),
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            "reason": config["pretrained_encoder_head"]["reason"],
        })
        pd.DataFrame(all_prediction_rows).to_csv(predictions_path, index=False)
        pd.DataFrame(all_selection_rows).to_csv(output_dir / "model_selection.csv", index=False)
        pd.DataFrame(all_history_rows).to_csv(output_dir / "training_history.csv", index=False)
        (output_dir / "split_manifest.json").write_text(
            json.dumps({"outer_folds": all_split_records}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        run_manifest["completed_outer_folds"] = outer_fold
        run_manifest["last_completed_subject"] = fold["test_subject"]
        (output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"completed outer fold {outer_fold}/{len(outer_folds)} ({fold['test_subject']} held out)",
            flush=True,
        )

    return finalize_saved_benchmark(dataset_dir, output_dir)
