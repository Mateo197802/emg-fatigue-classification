"""Nested low-frequency-proxy evaluation for raw-signal neural models.

The outer folds are copied from the frozen 4x3 tabular manifest. Epochs,
calibration and probability thresholds are chosen from outer-training groups
only. Every reported comparison is paired on identical groups, windows and
operational pseudo-labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from .evaluation import group_calibration_splits, select_group_balanced_threshold
from .evaluation_neural import (
    AVAILABLE_MODEL_NAMES,
    MODEL_NAMES,
    _group_balanced_macro_f1,
    _group_equal_weights,
    _load_fold_plan,
    _model_output,
    _new_model,
    _sha256_file,
    audit_group_fold_support,
    select_fold_pretraining_indices,
)
from .metrics import compute_metrics


PRIMARY_LABEL_RULE = "low_frequency_fraction_persistence_v1"
PROXY_LABEL_TYPE = "weak_supervision_proxy_not_independent_fatigue_ground_truth"
THRESHOLD_STRATEGIES = ("fixed_0_5", "inner_oof_selected")
GATE_BALANCE_WEIGHT = 0.05
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


def _path_hash_from_manifest(mapping: dict[str, Any], path: Path) -> str | None:
    for raw_path, digest in mapping.items():
        if Path(raw_path).expanduser().resolve() == path.resolve():
            return str(digest)
    return None


def verify_lowfreq_threshold_parent_stage(tabular_dir: Path) -> dict[str, Any]:
    """Require the corrected low-frequency 4x3 stage and matching checksums."""
    tabular_dir = Path(tabular_dir).expanduser().resolve()
    receipt_path = tabular_dir / "supervised_threshold_stage_manifest.json"
    fold_path = tabular_dir / "fold_manifest.json"
    prediction_path = tabular_dir / "nested_oof_predictions.parquet"
    for path in (receipt_path, fold_path, prediction_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen low-frequency artifact is missing: {path.name}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "completed"
        or receipt.get("label_rule") != PRIMARY_LABEL_RULE
        or receipt.get("label_rule_type") != PROXY_LABEL_TYPE
        or int(receipt.get("outer_group_folds", -1)) != 4
        or int(receipt.get("inner_group_folds", -1)) != 3
        or receipt.get("external_rar_opened") is not False
        or int(receipt.get("internal_test_signal_values_parsed", -1)) != 0
    ):
        raise ValueError("threshold parent does not match the locked low-frequency development contract")
    output_hashes = receipt.get("outputs_sha256", {})
    for path in (fold_path, prediction_path):
        expected = _path_hash_from_manifest(output_hashes, path)
        if not expected or _sha256_file(path) != expected:
            raise ValueError(f"threshold parent output checksum mismatch: {path.name}")
    plan = _load_fold_plan(fold_path)
    if len(plan.outer_folds) != 4 or any(len(plan.inner_folds[fold.fold_id]) != 3 for fold in plan.outer_folds):
        raise ValueError("threshold parent fold manifest is not the historical 4x3 plan")
    return receipt


def join_primary_lowfreq_labels(
    metadata: pd.DataFrame,
    alternative_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Replace the stale bank label columns using a strict key-checked join."""
    required_metadata = {"window_key", "group_id", "recording_id"}
    required_labels = {
        "window_key",
        "group_id",
        "recording_id",
        "class_id",
        "label_confidence",
        "rule_version",
        "label_reason",
        "comparison_rule",
    }
    if missing := required_metadata - set(metadata.columns):
        raise ValueError(f"signal metadata is missing columns: {sorted(missing)}")
    if missing := required_labels - set(alternative_labels.columns):
        raise ValueError(f"alternative labels are missing columns: {sorted(missing)}")
    if metadata.window_key.astype(str).duplicated().any():
        raise ValueError("signal metadata has duplicate window keys")
    labels = alternative_labels.loc[
        alternative_labels.comparison_rule.astype(str).eq(PRIMARY_LABEL_RULE),
        [
            "window_key",
            "group_id",
            "recording_id",
            "class_id",
            "label_confidence",
            "rule_version",
            "label_reason",
        ],
    ].copy()
    labels["window_key"] = labels.window_key.astype(str)
    if labels.window_key.duplicated().any():
        raise ValueError("primary low-frequency labels contain duplicate window keys")
    if labels.rule_version.astype(str).ne(PRIMARY_LABEL_RULE).any():
        raise ValueError("comparison rule and frozen label version disagree")
    if not labels.class_id.isin((-1, 0, 1)).all():
        raise ValueError("primary low-frequency labels use an invalid class vocabulary")
    left_keys = set(metadata.window_key.astype(str))
    right_keys = set(labels.window_key.astype(str))
    if left_keys != right_keys:
        raise ValueError(
            "signal and low-frequency label windows differ "
            f"(missing labels={len(left_keys - right_keys)}, extra labels={len(right_keys - left_keys)})"
        )
    clean_metadata = metadata.drop(
        columns=["class_id", "label_confidence", "rule_version", "label_reason"],
        errors="ignore",
    ).copy()
    clean_metadata["window_key"] = clean_metadata.window_key.astype(str)
    joined = clean_metadata.merge(
        labels,
        on="window_key",
        how="inner",
        suffixes=("", "_label"),
        validate="one_to_one",
    )
    for source_column in ("group_id", "recording_id"):
        label_column = f"{source_column}_label"
        if label_column in joined:
            if not joined[source_column].astype(str).eq(joined[label_column].astype(str)).all():
                raise ValueError(f"signal and low-frequency {source_column} values disagree")
            joined = joined.drop(columns=[label_column])
    joined["group_id"] = joined.group_id.astype(str)
    joined["recording_id"] = joined.recording_id.astype(str)
    joined["class_id"] = joined.class_id.astype(int)
    if len(joined) != len(metadata):
        raise RuntimeError("one-to-one low-frequency label join changed the signal-bank row count")
    return joined


def _class_support(rows: pd.DataFrame) -> dict[int, dict[str, int]]:
    support: dict[int, dict[str, int]] = {}
    for class_id in (0, 1):
        subset = rows[rows.class_id.eq(class_id)]
        support[class_id] = {
            "windows_descriptive_only": int(len(subset)),
            "source_groups": int(subset.group_id.nunique()),
            "recording_trials": int(subset[["group_id", "recording_id"]].drop_duplicates().shape[0]),
        }
    return support


def _require_partition_support(rows: pd.DataFrame, partition: str, minimum_groups: int = 2, minimum_trials: int = 2) -> None:
    if rows.empty or set(rows.class_id.astype(int).unique()) != {0, 1}:
        raise ValueError(f"{partition} lacks one or both pseudo-classes")
    support = _class_support(rows)
    for class_id in (0, 1):
        item = support[class_id]
        if item["source_groups"] < int(minimum_groups) or item["recording_trials"] < int(minimum_trials):
            raise ValueError(
                f"{partition} class {class_id} support is insufficient: "
                f"groups={item['source_groups']}, trials={item['recording_trials']}"
            )


def _calibration_splits(rows: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    _require_partition_support(rows, "calibration parent")
    splits = group_calibration_splits(
        rows.group_id.astype(str).tolist(),
        rows.class_id.to_numpy(dtype=int),
        seed=int(seed),
        n_splits=2,
    )
    for index, (train_index, calibration_index) in enumerate(splits, start=1):
        train = rows.iloc[train_index]
        calibration = rows.iloc[calibration_index]
        _require_partition_support(train, f"calibration train split {index}")
        _require_partition_support(calibration, f"calibration held-out split {index}")
        if set(train.group_id.astype(str)) & set(calibration.group_id.astype(str)):
            raise RuntimeError("group-aware calibration split contains source-group overlap")
    return splits


def _fit_platt_calibrator(
    logits: Sequence[float],
    truth: Sequence[int],
    groups: Sequence[str],
    recordings: Sequence[str],
) -> tuple[LogisticRegression, dict[str, Any]]:
    score = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int64)
    group_array = np.asarray([str(value) for value in groups], dtype=object)
    recording_array = np.asarray([str(value) for value in recordings], dtype=object)
    if (
        score.ndim != 1
        or labels.shape != score.shape
        or group_array.shape != score.shape
        or recording_array.shape != score.shape
        or score.size == 0
    ):
        raise ValueError("calibration logits, labels, groups, and trials must have equal non-empty lengths")
    if not np.isfinite(score).all() or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("calibration requires finite OOF logits and both proxy classes")
    support = _class_support(
        pd.DataFrame({"class_id": labels, "group_id": group_array, "recording_id": recording_array})
    )
    if any(
        support[class_id]["source_groups"] < 2 or support[class_id]["recording_trials"] < 2
        for class_id in (0, 1)
    ):
        raise ValueError("sigmoid calibration requires both classes across at least two groups and trials")
    _, inverse, counts = np.unique(group_array, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(counts) / weights.sum()
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0)
    calibrator.fit(score.reshape(-1, 1), labels, sample_weight=weights)
    receipt = {
        "method": "group_weighted_sigmoid_platt_on_group_held_out_logits",
        "source_group_count": int(np.unique(group_array).size),
        "class_0_source_group_count": int(support[0]["source_groups"]),
        "class_1_source_group_count": int(support[1]["source_groups"]),
        "class_0_recording_trial_count": int(support[0]["recording_trials"]),
        "class_1_recording_trial_count": int(support[1]["recording_trials"]),
        "class_0_window_count_descriptive_only": int(support[0]["windows_descriptive_only"]),
        "class_1_window_count_descriptive_only": int(support[1]["windows_descriptive_only"]),
        "group_weighting": "each source group has equal total calibration weight",
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    return calibrator, receipt


def _apply_platt(calibrator: LogisticRegression, logits: Sequence[float]) -> np.ndarray:
    score = np.asarray(logits, dtype=np.float64)
    if score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("calibration input must be a finite one-dimensional logit vector")
    transformed = score * float(calibrator.coef_[0, 0]) + float(calibrator.intercept_[0])
    return (1.0 / (1.0 + np.exp(-np.clip(transformed, -60.0, 60.0)))).astype(np.float64)


def _fit_outer_model(
    model_name: str,
    rows: pd.DataFrame,
    metadata: pd.DataFrame,
    signals: np.ndarray,
    groups: Sequence[str],
    *,
    epochs: int,
    seed: int,
    ssl_pretrain_epochs: int,
    validation_rows: pd.DataFrame | None = None,
    external_encoder_path: Path | None = None,
) -> tuple[Any, float, float, dict[str, Any], dict[str, list[Any]]]:
    raw_positions = select_fold_pretraining_indices(metadata, groups)
    selected_groups = set(str(value) for value in groups)
    if not set(metadata.iloc[raw_positions].group_id.astype(str)).issubset(selected_groups):
        raise RuntimeError("normalization or self-supervised pretraining admitted a held-out source group")
    scaler_mean = float(signals[raw_positions].mean())
    scaler_std = float(signals[raw_positions].std())
    if not np.isfinite(scaler_mean) or not np.isfinite(scaler_std) or scaler_std <= 1e-8:
        raise ValueError("training-fold signal normalization has zero or non-finite dispersion")
    train_x = (signals[rows.bank_position.to_numpy(dtype=int)] - scaler_mean) / scaler_std
    pretraining_x = (signals[raw_positions] - scaler_mean) / scaler_std
    validation_x = None
    if validation_rows is not None:
        validation_x = (signals[validation_rows.bank_position.to_numpy(dtype=int)] - scaler_mean) / scaler_std
    model, ssl_fit = _new_model(
        model_name,
        int(seed),
        pretraining_x if model_name in {"ssl_encoder_finetuned", "ssl_encoder_frozen"} else None,
        int(ssl_pretrain_epochs),
        external_encoder_path=external_encoder_path,
    )
    history = _train_with_validation_logits(
        model,
        train_x,
        rows.class_id.to_numpy(dtype=int),
        rows.group_id.astype(str).tolist(),
        epochs=int(epochs),
        validation_signals=validation_x,
        validation_labels=None if validation_rows is None else validation_rows.class_id.to_numpy(dtype=int),
        validation_groups=None if validation_rows is None else validation_rows.group_id.astype(str).tolist(),
    )
    fit_receipt = {
        "normalization_mean": scaler_mean,
        "normalization_std": scaler_std,
        "normalization_window_count": int(len(raw_positions)),
        "normalization_source_groups": sorted(selected_groups),
        "ssl_pretraining_window_count": 0 if ssl_fit is None else int(ssl_fit.train_window_count),
        "ssl_pretraining_epochs": 0 if ssl_fit is None else int(len(ssl_fit.reconstruction_losses)),
        "ssl_reconstruction_losses": [] if ssl_fit is None else list(ssl_fit.reconstruction_losses),
        "ssl_pretraining_optimizer": None if ssl_fit is None else ssl_fit.optimizer,
        "ssl_encoder_frozen_during_supervised_fit": model_name == "ssl_encoder_frozen",
        "external_transfer_initialization": getattr(model, "_external_transfer_receipt", None),
        "training_groups": sorted(set(rows.group_id.astype(str))),
    }
    return model, scaler_mean, scaler_std, fit_receipt, history


def _train_with_validation_logits(
    model: Any,
    train_signals: np.ndarray,
    train_labels: Sequence[int],
    train_groups: Sequence[str],
    *,
    epochs: int,
    validation_signals: np.ndarray | None,
    validation_labels: Sequence[int] | None,
    validation_groups: Sequence[str] | None,
) -> dict[str, list[Any]]:
    from .models_neural import gate_load_balance_penalty

    x_train = torch.from_numpy(np.asarray(train_signals, dtype=np.float32)[:, None, :])
    y_train = torch.from_numpy(np.asarray(train_labels, dtype=np.float32))
    weights = _group_equal_weights(train_groups)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("neural model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    has_validation = validation_signals is not None
    if has_validation and (validation_labels is None or validation_groups is None):
        raise ValueError("validation labels and groups are required with validation signals")
    x_valid = (
        torch.from_numpy(np.asarray(validation_signals, dtype=np.float32)[:, None, :])
        if has_validation
        else None
    )
    history: dict[str, list[Any]] = {
        "training_loss": [],
        "validation_logits": [],
        "validation_group_balanced_macro_f1_at_0_5": [],
    }
    for _ in range(int(epochs)):
        model.train()
        logits, gates = _model_output(model, x_train)
        element_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
        loss = torch.sum(element_loss * weights) / weights.sum().clamp_min(1e-8)
        if gates is not None:
            loss = loss + GATE_BALANCE_WEIGHT * gate_load_balance_penalty(gates)
        if not torch.isfinite(loss):
            raise FloatingPointError("supervised neural loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["training_loss"].append(float(loss.detach()))
        if has_validation:
            model.eval()
            with torch.no_grad():
                validation_logits, _ = _model_output(model, x_valid)
                validation_logits_np = validation_logits.cpu().numpy().astype(np.float64)
            history["validation_logits"].append(validation_logits_np)
            probability = 1.0 / (1.0 + np.exp(-np.clip(validation_logits_np, -60.0, 60.0)))
            prediction = (probability >= 0.5).astype(np.int64)
            history["validation_group_balanced_macro_f1_at_0_5"].append(
                _group_balanced_macro_f1(
                    np.asarray(validation_labels, dtype=np.int64),
                    prediction,
                    validation_groups,
                )
            )
    return history


def _predict_logits(model: Any, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    x = torch.from_numpy(np.asarray(signals, dtype=np.float32)[:, None, :])
    with torch.no_grad():
        logits, gates = _model_output(model, x)
    return (
        logits.cpu().numpy().astype(np.float64),
        gates.cpu().numpy().astype(np.float64) if gates is not None else None,
    )


def _labelled_rows(metadata: pd.DataFrame, group_ids: Sequence[str]) -> pd.DataFrame:
    return metadata.loc[
        metadata.group_id.astype(str).isin({str(value) for value in group_ids})
        & metadata.class_id.isin((0, 1))
    ].sort_values("window_key").reset_index(drop=True)


def _select_epoch(
    per_inner_history: dict[int, dict[str, list[Any]]],
    inner_validation_rows: dict[int, pd.DataFrame],
) -> tuple[int, list[dict[str, Any]]]:
    fold_ids = sorted(per_inner_history)
    epoch_count = len(per_inner_history[fold_ids[0]]["validation_logits"])
    score_rows: list[dict[str, Any]] = []
    mean_scores: list[float] = []
    for epoch_index in range(epoch_count):
        scores: list[float] = []
        for fold_id in fold_ids:
            rows = inner_validation_rows[fold_id]
            logits = per_inner_history[fold_id]["validation_logits"][epoch_index]
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
            prediction = (probabilities >= 0.5).astype(np.int64)
            score = _group_balanced_macro_f1(
                rows.class_id.to_numpy(dtype=int),
                prediction,
                rows.group_id.astype(str).tolist(),
            )
            scores.append(float(score))
            score_rows.append(
                {
                    "inner_fold_id": int(fold_id),
                    "epoch": int(epoch_index + 1),
                    "validation_group_balanced_macro_f1_at_0_5": float(score),
                    "selection_note": "epoch chosen from inner-held-out groups; threshold selection is separate",
                }
            )
        mean_scores.append(float(np.mean(scores)))
    selected = int(np.argmax(np.asarray(mean_scores, dtype=np.float64))) + 1
    for row in score_rows:
        row["inner_mean_group_balanced_macro_f1_at_0_5"] = mean_scores[int(row["epoch"]) - 1]
        row["selected_epoch"] = int(row["epoch"]) == selected
    return selected, score_rows


def _inner_calibrated_oof(
    *,
    model_name: str,
    outer_fold: Any,
    inner_fold: Any,
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    valid_logits: np.ndarray,
    metadata: pd.DataFrame,
    signals: np.ndarray,
    selected_epoch: int,
    plan_seed: int,
    model_index: int,
    outer_index: int,
    ssl_pretrain_epochs: int,
    external_encoder_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    calibration_logits: list[np.ndarray] = []
    calibration_truth: list[np.ndarray] = []
    calibration_groups: list[np.ndarray] = []
    calibration_recordings: list[np.ndarray] = []
    calibration_receipts: list[dict[str, Any]] = []
    splits = _calibration_splits(
        train_rows,
        int(plan_seed + outer_index * 100000 + model_index * 10000 + int(inner_fold.fold_id) * 1000),
    )
    for calibration_index, (fit_indices, heldout_indices) in enumerate(splits, start=1):
        fit_rows = train_rows.iloc[fit_indices].sort_values("window_key").reset_index(drop=True)
        heldout_rows = train_rows.iloc[heldout_indices].sort_values("window_key").reset_index(drop=True)
        calibration_seed = int(
            plan_seed
            + outer_index * 100000
            + model_index * 10000
            + int(inner_fold.fold_id) * 1000
            + calibration_index * 100
        )
        model, mean, std, fit_receipt, fit_history = _fit_outer_model(
            model_name,
            fit_rows,
            metadata,
            signals,
            sorted(set(fit_rows.group_id.astype(str))),
            epochs=selected_epoch,
            seed=calibration_seed,
            ssl_pretrain_epochs=ssl_pretrain_epochs,
            external_encoder_path=external_encoder_path,
        )
        del fit_history
        heldout_x = (signals[heldout_rows.bank_position.to_numpy(dtype=int)] - mean) / std
        logits, _ = _predict_logits(model, heldout_x)
        calibration_logits.append(logits)
        calibration_truth.append(heldout_rows.class_id.to_numpy(dtype=int))
        calibration_groups.append(heldout_rows.group_id.astype(str).to_numpy())
        calibration_recordings.append(heldout_rows.recording_id.astype(str).to_numpy())
        calibration_receipts.append(
            {
                "stage": "inner_calibration_crossfit",
                "model_name": model_name,
                "outer_fold_id": int(outer_fold.fold_id),
                "inner_fold_id": int(inner_fold.fold_id),
                "calibration_fold_id": int(calibration_index),
                "selected_epochs": int(selected_epoch),
                "fit_source_groups": sorted(set(fit_rows.group_id.astype(str))),
                "heldout_calibration_groups": sorted(set(heldout_rows.group_id.astype(str))),
                "heldout_group_overlap": False,
                "fit_receipt": fit_receipt,
            }
        )
    pooled_logits = np.concatenate(calibration_logits)
    pooled_truth = np.concatenate(calibration_truth)
    pooled_groups = np.concatenate(calibration_groups)
    pooled_recordings = np.concatenate(calibration_recordings)
    if len(set(pooled_groups.tolist())) != len(set(train_rows.group_id.astype(str))):
        raise RuntimeError("inner calibration OOF predictions do not cover every inner-training group")
    calibrator, calibrator_receipt = _fit_platt_calibrator(
        pooled_logits, pooled_truth, pooled_groups, pooled_recordings
    )
    probabilities = _apply_platt(calibrator, valid_logits)
    if probabilities.size != len(valid_rows):
        raise RuntimeError("inner calibrated OOF prediction count does not match validation rows")
    result = pd.DataFrame(
        {
            "model_name": model_name,
            "window_key": valid_rows.window_key.astype(str).to_numpy(),
            "group_id": valid_rows.group_id.astype(str).to_numpy(),
            "recording_id": valid_rows.recording_id.astype(str).to_numpy(),
            "outer_fold_id": int(outer_fold.fold_id),
            "inner_fold_id": int(inner_fold.fold_id),
            "true_class_id": valid_rows.class_id.to_numpy(dtype=int),
            "raw_logit": valid_logits.astype(np.float64),
            "probability_class_0": 1.0 - probabilities,
            "probability_class_1": probabilities,
            "calibration_source_groups": json.dumps(sorted(set(pooled_groups.tolist())), separators=(",", ":")),
            "calibration_excluded_validation_groups": json.dumps(sorted(set(valid_rows.group_id.astype(str))), separators=(",", ":")),
        }
    )
    support = _class_support(train_rows)
    receipt = {
        "stage": "inner_oof_calibration",
        "model_name": model_name,
        "outer_fold_id": int(outer_fold.fold_id),
        "inner_fold_id": int(inner_fold.fold_id),
        "selected_epochs": int(selected_epoch),
        "calibration_method": "group-weighted sigmoid Platt on inner-training group-held-out logits",
        "calibration_training_groups": sorted(set(pooled_groups.tolist())),
        "calibration_validation_groups": sorted(set(valid_rows.group_id.astype(str))),
        "class_support_in_inner_training": support,
        "calibrator": calibrator_receipt,
    }
    return result, receipt, calibration_receipts


def _weighted_confusion_metrics(matrix: np.ndarray) -> dict[str, float | None]:
    cm = np.asarray(matrix, dtype=np.float64)
    total = float(cm.sum())
    if total <= 0.0:
        return {name: None for name in ("accuracy", "sensitivity", "specificity", "macro_f1", "balanced_accuracy", "precision_class_1", "mcc", "cohen_kappa")}
    tn, fp = float(cm[0, 0]), float(cm[0, 1])
    fn, tp = float(cm[1, 0]), float(cm[1, 1])
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision_1 = tp / (tp + fp) if tp + fp else None
    precision_0 = tn / (tn + fn) if tn + fn else None
    f1_1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    accuracy = (tn + tp) / total
    balanced = (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / np.sqrt(denominator)) if denominator > 0 else None
    expected = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / (total * total)
    kappa = (accuracy - expected) / (1.0 - expected) if not np.isclose(expected, 1.0) else None
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "macro_f1": float((f1_0 + f1_1) / 2.0),
        "balanced_accuracy": balanced,
        "precision_class_1": precision_1,
        "precision_class_0": precision_0,
        "mcc": float(mcc) if mcc is not None else None,
        "cohen_kappa": float(kappa) if kappa is not None else None,
    }


def _group_normalized_confusions(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    confusions: dict[str, np.ndarray] = {}
    for group_id, group_rows in rows.groupby("group_id", sort=True):
        if set(group_rows.true_class_id.astype(int).unique()) != {0, 1}:
            raise ValueError(f"OOF group {group_id} lacks valid support for both proxy classes")
        matrix = confusion_matrix(
            group_rows.true_class_id.astype(int),
            group_rows.predicted_class_id.astype(int),
            labels=(0, 1),
        ).astype(np.float64)
        confusions[str(group_id)] = matrix / matrix.sum()
    return confusions


def _nested_metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for (model_name, strategy), rows in predictions.groupby(["model_name", "threshold_strategy"], sort=True):
        truth = rows.true_class_id.to_numpy(dtype=int)
        probability = rows.probability_class_1.to_numpy(dtype=float)
        predicted = rows.predicted_class_id.to_numpy(dtype=int)
        metrics = compute_metrics(truth, probability, predicted, np.zeros(len(rows), dtype=bool)).to_dict()
        confusions = _group_normalized_confusions(rows)
        group_confusion = np.mean(np.stack(list(confusions.values())), axis=0)
        pooled_rows.append(
            {
                "feature_view": str(rows.feature_view.iloc[0]),
                "model_name": str(model_name),
                "threshold_strategy": str(strategy),
                "decision_thresholds_by_outer_fold": json.dumps(
                    {
                        str(fold): sorted(set(float(value) for value in fold_rows.decision_threshold))
                        for fold, fold_rows in rows.groupby("fold_id", sort=True)
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "n_oof_windows_descriptive_only": int(len(rows)),
                "source_group_count": int(rows.group_id.nunique()),
                "independent_unit": "source folder group; participant identity unverified",
                **{key: metrics[key] for key in (
                    "accuracy", "sensitivity", "specificity", "macro_f1", "balanced_accuracy",
                    "matthews_correlation_coefficient", "cohen_kappa", "pr_auc_average_precision",
                    "roc_auc", "log_loss", "calibration", "confusion_matrix",
                    "confusion_matrix_normalized_by_true", "undefined_reasons",
                )},
                **{
                    f"group_balanced_{key}": value
                    for key, value in _weighted_confusion_metrics(group_confusion).items()
                },
                "group_balanced_confusion_matrix_normalized": json.dumps(group_confusion.tolist(), separators=(",", ":")),
                "interpretation": "agreement with weak-supervision proxy labels; overlapping windows are not independent",
            }
        )
        for group_id, group in rows.groupby("group_id", sort=True):
            group_metrics = compute_metrics(
                group.true_class_id.to_numpy(dtype=int),
                group.probability_class_1.to_numpy(dtype=float),
                group.predicted_class_id.to_numpy(dtype=int),
                np.zeros(len(group), dtype=bool),
            ).to_dict()
            group_rows.append(
                {
                    "feature_view": str(group.feature_view.iloc[0]),
                    "model_name": str(model_name),
                    "threshold_strategy": str(strategy),
                    "group_id": str(group_id),
                    "outer_fold_id": int(group.fold_id.iloc[0]),
                    "n_oof_windows_descriptive_only": int(len(group)),
                    "class_0_support": int(group.true_class_id.eq(0).sum()),
                    "class_1_support": int(group.true_class_id.eq(1).sum()),
                    "accuracy": group_metrics["accuracy"],
                    "sensitivity": group_metrics["sensitivity"],
                    "specificity": group_metrics["specificity"],
                    "macro_f1": group_metrics["macro_f1"],
                    "balanced_accuracy": group_metrics["balanced_accuracy"],
                    "mcc": group_metrics["matthews_correlation_coefficient"],
                    "cohen_kappa": group_metrics["cohen_kappa"],
                    "pr_auc_average_precision": group_metrics["pr_auc_average_precision"],
                    "roc_auc": group_metrics["roc_auc"],
                    "brier_score": group_metrics["calibration"]["brier_score"],
                    "ece_10_bins": group_metrics["calibration"]["expected_calibration_error_10_bins"],
                    "confusion_matrix": json.dumps(group_metrics["confusion_matrix"], separators=(",", ":")),
                    "undefined_reasons": json.dumps(group_metrics["undefined_reasons"], sort_keys=True, separators=(",", ":")),
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(group_rows)


def _paired_group_bootstrap(predictions: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    baseline_model = "time_only_logistic"
    baseline_strategies = {
        "fixed_0_5": "fixed_0_5",
        "inner_oof_selected": "inner_oof_selected",
    }
    groups = tuple(sorted(predictions.group_id.astype(str).unique()))
    if len(groups) < 2:
        raise ValueError("paired group uncertainty requires at least two source groups")
    confusions = {
        (model, strategy): _group_normalized_confusions(rows)
        for (model, strategy), rows in predictions.groupby(["model_name", "threshold_strategy"], sort=True)
    }
    rng = np.random.default_rng(int(seed))
    choices = rng.integers(0, len(groups), size=(int(replicates), len(groups)))
    metric_names = ("accuracy", "macro_f1", "balanced_accuracy", "sensitivity", "specificity")
    comparisons: list[tuple[str, str, str, str]] = []
    model_names = tuple(sorted(set(predictions.model_name.astype(str)) - {"time_only_logistic"}))
    for model_name in model_names:
        comparisons.extend(
            [
                (model_name, "inner_oof_selected", model_name, "fixed_0_5"),
                (model_name, "inner_oof_selected", baseline_model, "inner_oof_selected"),
                (model_name, "fixed_0_5", baseline_model, "fixed_0_5"),
            ]
        )
    output: list[dict[str, Any]] = []
    for model_left, strategy_left, model_right, strategy_right in comparisons:
        left = confusions[(model_left, strategy_left)]
        right = confusions[(model_right, strategy_right)]
        if set(left) != set(groups) or set(right) != set(groups):
            raise ValueError("paired predictions do not share identical source groups")
        point_left = _weighted_confusion_metrics(np.mean(np.stack([left[group] for group in groups]), axis=0))
        point_right = _weighted_confusion_metrics(np.mean(np.stack([right[group] for group in groups]), axis=0))
        for metric_name in metric_names:
            deltas: list[float] = []
            for draw in choices:
                sampled_groups = [groups[index] for index in draw]
                left_metric = _weighted_confusion_metrics(np.mean(np.stack([left[group] for group in sampled_groups]), axis=0))[metric_name]
                right_metric = _weighted_confusion_metrics(np.mean(np.stack([right[group] for group in sampled_groups]), axis=0))[metric_name]
                if left_metric is not None and right_metric is not None and np.isfinite(left_metric) and np.isfinite(right_metric):
                    deltas.append(float(left_metric - right_metric))
            left_value, right_value = point_left[metric_name], point_right[metric_name]
            output.append(
                {
                    "model_name": model_left,
                    "threshold_strategy": strategy_left,
                    "comparison_model": model_right,
                    "comparison_threshold_strategy": strategy_right,
                    "metric": metric_name,
                    "delta": None if left_value is None or right_value is None else float(left_value - right_value),
                    "ci_95_low": float(np.quantile(deltas, 0.025)) if deltas else None,
                    "ci_95_high": float(np.quantile(deltas, 0.975)) if deltas else None,
                    "valid_bootstrap_replicates": len(deltas),
                    "bootstrap_replicates": int(replicates),
                    "bootstrap_seed": int(seed),
                    "bootstrap_unit": "paired source-group resampling with equal group confusion mass",
                }
            )
    return pd.DataFrame(output)


def _serialize_checkpoint_model(model_name: str, payload: dict[str, Any]) -> Any:
    if model_name in {"latent_moe2", "latent_moe3"}:
        from .models_neural import SoftGatedExperts

        return SoftGatedExperts(expert_count=2 if model_name == "latent_moe2" else 3)
    from .models_neural import BinarySignalClassifier

    return BinarySignalClassifier()


def run_nested_neural_proxy_stage(
    run_dir: Path,
    *,
    tabular_subdir: str = "lowfreq_threshold_v2",
    output_subdir: str = "neural_lowfreq_nested_4x3_v1",
    max_epochs: int = 20,
    ssl_pretrain_epochs: int = 6,
    bootstrap_replicates: int = 2000,
    model_names: Sequence[str] = MODEL_NAMES,
    external_encoder_path: Path | None = None,
) -> dict[str, Any]:
    """Train CNN, fine-tuned SSL encoder and latent MoE on low-frequency labels."""
    if Path(output_subdir).name != output_subdir or output_subdir in {"", ".", ".."}:
        raise ValueError("output_subdir must be one directory name")
    if Path(tabular_subdir).name != tabular_subdir or tabular_subdir in {"", ".", ".."}:
        raise ValueError("tabular_subdir must be one directory name")
    if int(max_epochs) < 1 or int(ssl_pretrain_epochs) < 1 or int(bootstrap_replicates) < 1:
        raise ValueError("epoch and bootstrap limits must be positive")
    requested_models = tuple(str(value) for value in model_names)
    if not requested_models or len(set(requested_models)) != len(requested_models):
        raise ValueError("model_names must be a non-empty unique sequence")
    unknown_models = set(requested_models) - set(AVAILABLE_MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"unsupported neural model names: {sorted(unknown_models)}")
    if "cerqueira_supervised_finetuned" in requested_models:
        if external_encoder_path is None or not Path(external_encoder_path).expanduser().is_file():
            raise FileNotFoundError("Cerqueira supervised transfer requires an existing external encoder checkpoint")
        external_encoder_path = Path(external_encoder_path).expanduser().resolve()
        external_encoder_sha256 = _sha256_file(external_encoder_path)
    else:
        external_encoder_sha256 = None
    torch.set_num_threads(1)
    run_dir = Path(run_dir).expanduser().resolve()
    private_dir = run_dir / "private"
    bank_dir = private_dir / "neural_data"
    tabular_dir = private_dir / tabular_subdir
    bank_manifest_path = bank_dir / "signal_window_bank_manifest.json"
    bank_path = bank_dir / "signals_500ms_development.npz"
    metadata_path = bank_dir / "signal_window_manifest.parquet"
    labels_path = private_dir / "features_labels" / "pseudo_labels_alternatives.parquet"
    parent_receipt_path = tabular_dir / "supervised_threshold_stage_manifest.json"
    fold_path = tabular_dir / "fold_manifest.json"
    predictions_path = tabular_dir / "nested_oof_predictions.parquet"
    for path in (bank_manifest_path, bank_path, metadata_path, labels_path, parent_receipt_path, fold_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required private development input is missing: {path.name}")
    parent_receipt = verify_lowfreq_threshold_parent_stage(tabular_dir)
    bank_receipt = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    if (
        bank_receipt.get("status") != "completed"
        or bank_receipt.get("external_rar_opened") is not False
        or int(bank_receipt.get("internal_test_signal_values_parsed", -1)) != 0
        or int(bank_receipt.get("reserved_test_member_payloads_read", -1)) != 0
    ):
        raise ValueError("signal bank receipt does not prove that reserved development inputs stayed closed")
    expected_bank_hashes = bank_receipt.get("output_sha256", {})
    if expected_bank_hashes.get(str(bank_path)) != _sha256_file(bank_path):
        raise ValueError("development signal array checksum differs from its receipt")
    if expected_bank_hashes.get(str(metadata_path)) != _sha256_file(metadata_path):
        raise ValueError("development signal metadata checksum differs from its receipt")

    bank = np.load(bank_path, allow_pickle=False)
    signals = np.asarray(bank["signals"], dtype=np.float32)
    bank_keys = bank["window_keys"].astype(str)
    raw_metadata = pd.read_parquet(metadata_path)
    if signals.ndim != 2 or signals.shape[1] != 500 or signals.shape[0] != len(raw_metadata):
        raise ValueError("signal bank shape is inconsistent with 500 ms windows")
    if not np.array_equal(bank_keys, raw_metadata.window_key.astype(str).to_numpy()):
        raise ValueError("signal array order differs from the private window manifest")
    labels = pd.read_parquet(labels_path)
    metadata = join_primary_lowfreq_labels(raw_metadata, labels)
    if metadata.group_id.nunique() != 12 or len(metadata) != 11547:
        raise ValueError("low-frequency signal bank does not match the frozen 12-group development cohort")
    plan = _load_fold_plan(fold_path)

    parent_predictions = pd.read_parquet(predictions_path)
    baseline = parent_predictions.loc[
        parent_predictions.feature_view.eq("relative_time_only")
        & parent_predictions.model_name.eq("logistic_regression")
        & parent_predictions.threshold_strategy.isin(THRESHOLD_STRATEGIES)
        & parent_predictions.true_class_id.isin((0, 1)),
        [
            "window_key", "group_id", "recording_id", "fold_id", "true_class_id",
            "predicted_class_id", "probability_class_0", "probability_class_1",
            "threshold_strategy", "decision_threshold",
        ],
    ].copy()
    if baseline.duplicated(["window_key", "threshold_strategy"]).any():
        raise ValueError("temporal baseline contains duplicate outer OOF rows")
    baseline["window_key"] = baseline.window_key.astype(str)
    active = metadata.loc[metadata.class_id.isin((0, 1))].copy()
    active["bank_position"] = np.flatnonzero(metadata.class_id.isin((0, 1)).to_numpy())
    active["group_id"] = active.group_id.astype(str)
    active["recording_id"] = active.recording_id.astype(str)
    active["class_id"] = active.class_id.astype(int)
    if active.window_key.duplicated().any():
        raise ValueError("assigned low-frequency windows are duplicated")
    expected_window_keys = set(active.window_key.astype(str))
    for strategy in THRESHOLD_STRATEGIES:
        baseline_rows = baseline[baseline.threshold_strategy.eq(strategy)]
        if set(baseline_rows.window_key.astype(str)) != expected_window_keys or len(baseline_rows) != len(active):
            raise ValueError(f"temporal baseline {strategy} does not exactly cover assigned low-frequency windows")
    selected_baseline = baseline[baseline.threshold_strategy.eq("inner_oof_selected")]
    check = active[["window_key", "group_id", "recording_id", "class_id"]].merge(
        selected_baseline[["window_key", "group_id", "recording_id", "true_class_id", "fold_id"]],
        on="window_key",
        how="left",
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    if (
        check.true_class_id.isna().any()
        or not check.group_id.astype(str).eq(check.group_id_baseline.astype(str)).all()
        or not check.recording_id.astype(str).eq(check.recording_id_baseline.astype(str)).all()
        or not check.class_id.eq(check.true_class_id.astype(int)).all()
    ):
        raise ValueError("temporal baseline groups, trials, or pseudo-labels differ from the raw signal cohort")
    group_to_fold = {
        str(group_id): int(fold.fold_id)
        for fold in plan.outer_folds
        for group_id in fold.validation_group_ids
    }
    active["fold_id"] = active.group_id.map(group_to_fold).astype(int)
    baseline_fold_map = selected_baseline.set_index("window_key").fold_id.astype(int)
    if not active.window_key.map(baseline_fold_map).eq(active.fold_id).all():
        raise ValueError("temporal baseline outer-fold IDs differ from the frozen 4x3 manifest")
    support = audit_group_fold_support(active, plan, min_groups_per_class=2, min_trials_per_class=2)
    assigned_counts = active.class_id.value_counts().to_dict()
    if int(assigned_counts.get(0, 0)) != 5074 or int(assigned_counts.get(1, 0)) != 903:
        raise ValueError("active low-frequency class counts differ from the verified development label contract")

    output_dir = private_dir / output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite neural stage: {output_dir}")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    joined_path = output_dir / "active_lowfreq_signal_window_manifest.parquet"
    metadata.to_parquet(joined_path, index=False)
    support_path = output_dir / "label_support_by_fold.csv"
    support.to_csv(support_path, index=False)

    inner_metric_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    inner_oof_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    calibration_fit_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    outer_prediction_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    for outer_index, outer in enumerate(plan.outer_folds):
        for model_index, model_name in enumerate(requested_models):
            histories: dict[int, dict[str, list[Any]]] = {}
            validation_rows_by_fold: dict[int, pd.DataFrame] = {}
            training_rows_by_fold: dict[int, pd.DataFrame] = {}
            for inner in plan.inner_folds[outer.fold_id]:
                train_rows = _labelled_rows(active, inner.train_group_ids)
                valid_rows = _labelled_rows(active, inner.validation_group_ids)
                _require_partition_support(train_rows, f"outer {outer.fold_id} inner {inner.fold_id} training")
                _require_partition_support(valid_rows, f"outer {outer.fold_id} inner {inner.fold_id} validation")
                if set(train_rows.group_id) & set(valid_rows.group_id):
                    raise RuntimeError("inner training and validation source groups overlap")
                seed = int(plan.seed + outer_index * 100000 + model_index * 10000 + int(inner.fold_id) * 100)
                model, _, _, fit_receipt, history = _fit_outer_model(
                    model_name,
                    train_rows,
                    metadata,
                    signals,
                    inner.train_group_ids,
                    epochs=int(max_epochs),
                    seed=seed,
                    ssl_pretrain_epochs=int(ssl_pretrain_epochs),
                    validation_rows=valid_rows,
                    external_encoder_path=external_encoder_path,
                )
                histories[int(inner.fold_id)] = history
                training_rows_by_fold[int(inner.fold_id)] = train_rows
                validation_rows_by_fold[int(inner.fold_id)] = valid_rows
                for epoch_index, loss in enumerate(history["training_loss"], start=1):
                    inner_metric_rows.append(
                        {
                            "model_name": model_name,
                            "outer_fold_id": int(outer.fold_id),
                            "inner_fold_id": int(inner.fold_id),
                            "epoch": int(epoch_index),
                            "training_loss_group_weighted": float(loss),
                            "validation_group_balanced_macro_f1_at_0_5": float(history["validation_group_balanced_macro_f1_at_0_5"][epoch_index - 1]),
                            "normalization_window_count": fit_receipt["normalization_window_count"],
                            "ssl_pretraining_window_count": fit_receipt["ssl_pretraining_window_count"],
                            "ssl_pretraining_epochs": fit_receipt["ssl_pretraining_epochs"],
                            "normalization_and_pretraining_groups": json.dumps(fit_receipt["normalization_source_groups"], separators=(",", ":")),
                            "validation_groups_disjoint_from_training": True,
                        }
                    )
            selected_epoch, epoch_selection_rows = _select_epoch(histories, validation_rows_by_fold)
            for row in epoch_selection_rows:
                row.update({"model_name": model_name, "outer_fold_id": int(outer.fold_id)})
            epoch_rows.extend(epoch_selection_rows)

            inner_predictions_for_outer: list[pd.DataFrame] = []
            inner_raw_for_outer: list[pd.DataFrame] = []
            for inner in plan.inner_folds[outer.fold_id]:
                fold_id = int(inner.fold_id)
                valid_rows = validation_rows_by_fold[fold_id]
                train_rows = training_rows_by_fold[fold_id]
                selected_logits = histories[fold_id]["validation_logits"][selected_epoch - 1]
                oof_frame, calibration_receipt, fit_receipts = _inner_calibrated_oof(
                    model_name=model_name,
                    outer_fold=outer,
                    inner_fold=inner,
                    train_rows=train_rows,
                    valid_rows=valid_rows,
                    valid_logits=selected_logits,
                    metadata=metadata,
                    signals=signals,
                    selected_epoch=selected_epoch,
                    plan_seed=plan.seed,
                    model_index=model_index,
                    outer_index=outer_index,
                    ssl_pretrain_epochs=int(ssl_pretrain_epochs),
                    external_encoder_path=external_encoder_path,
                )
                inner_predictions_for_outer.append(oof_frame)
                inner_raw_for_outer.append(
                    oof_frame[["window_key", "group_id", "recording_id", "outer_fold_id", "inner_fold_id", "true_class_id", "raw_logit"]].copy()
                )
                calibration_rows.append(calibration_receipt)
                calibration_fit_rows.extend(fit_receipts)
            inner_oof = pd.concat(inner_predictions_for_outer, ignore_index=True)
            if inner_oof.window_key.duplicated().any():
                raise RuntimeError("inner calibrated OOF contains duplicate windows")
            if set(inner_oof.group_id.astype(str)) != set(outer.train_group_ids):
                raise RuntimeError("inner calibrated OOF does not cover each outer-training group once")
            inner_support = _class_support(inner_oof.rename(columns={"true_class_id": "class_id"}))
            if any(inner_support[class_id]["source_groups"] < 2 for class_id in (0, 1)):
                raise ValueError("inner OOF threshold-selection scores lack class support across two source groups")
            threshold = select_group_balanced_threshold(
                inner_oof.true_class_id.to_numpy(dtype=int),
                inner_oof.probability_class_1.to_numpy(dtype=float),
                inner_oof.group_id.astype(str).tolist(),
            )
            threshold_rows.append(
                {
                    "model_name": model_name,
                    "outer_fold_id": int(outer.fold_id),
                    "selected_supervised_epochs": int(selected_epoch),
                    "threshold": float(threshold.threshold),
                    "inner_oof_group_balanced_macro_f1": float(threshold.group_balanced_macro_f1),
                    "inner_oof_group_balanced_precision_class_1": float(threshold.group_balanced_precision),
                    "inner_oof_group_balanced_recall_class_1": float(threshold.group_balanced_recall),
                    "threshold_candidate_count": int(threshold.candidate_count),
                    "tie_resolution": threshold.tie_resolution,
                    "threshold_objective": "group-balanced macro-F1 on calibrated inner OOF predictions",
                    "threshold_source_groups": json.dumps(sorted(set(inner_oof.group_id.astype(str))), separators=(",", ":")),
                    "calibration_method": "per-inner-fold group-weighted sigmoid Platt on inner-training crossfit logits",
                    "outer_validation_accessed_for_selection": False,
                }
            )
            inner_oof_rows.append(inner_oof)

            # The final outer calibrator is learned only from group-held-out
            # logits produced inside the outer-training groups.
            inner_raw = pd.concat(inner_raw_for_outer, ignore_index=True)
            outer_calibrator, outer_calibration_receipt = _fit_platt_calibrator(
                inner_raw.raw_logit.to_numpy(dtype=float),
                inner_raw.true_class_id.to_numpy(dtype=int),
                inner_raw.group_id.astype(str).tolist(),
                inner_raw.recording_id.astype(str).tolist(),
            )
            train_rows = _labelled_rows(active, outer.train_group_ids)
            valid_rows = _labelled_rows(active, outer.validation_group_ids)
            _require_partition_support(train_rows, f"outer {outer.fold_id} training")
            _require_partition_support(valid_rows, f"outer {outer.fold_id} validation")
            outer_seed = int(plan.seed + outer_index * 100000 + model_index * 10000 + 9000)
            final_model, mean, std, fit_receipt, history = _fit_outer_model(
                model_name,
                train_rows,
                metadata,
                signals,
                outer.train_group_ids,
                epochs=selected_epoch,
                seed=outer_seed,
                ssl_pretrain_epochs=int(ssl_pretrain_epochs),
                external_encoder_path=external_encoder_path,
            )
            del history
            valid_x = (signals[valid_rows.bank_position.to_numpy(dtype=int)] - mean) / std
            outer_logits, gates = _predict_logits(final_model, valid_x)
            calibrated_probability = _apply_platt(outer_calibrator, outer_logits)
            checkpoint_path = checkpoint_dir / f"{model_name}_outer_{int(outer.fold_id):02d}.pt"
            checkpoint_metadata = {
                "model_name": model_name,
                "outer_fold_id": int(outer.fold_id),
                "seed": outer_seed,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "selected_supervised_epochs": int(selected_epoch),
                "label_rule": PRIMARY_LABEL_RULE,
                "label_rule_type": PROXY_LABEL_TYPE,
                "fixed_threshold": 0.5,
                "inner_oof_selected_threshold": float(threshold.threshold),
                "training_source_groups": list(outer.train_group_ids),
                "held_out_source_groups": list(outer.validation_group_ids),
                "normalization_and_pretraining_receipt": fit_receipt,
                "outer_calibration_receipt": outer_calibration_receipt,
                "outer_calibrator_coefficient": float(outer_calibrator.coef_[0, 0]),
                "outer_calibrator_intercept": float(outer_calibrator.intercept_[0]),
                "gate_balance_weight": GATE_BALANCE_WEIGHT if model_name.startswith("latent_moe") else 0.0,
                "expert_semantics": "latent experts share one binary proxy target" if model_name.startswith("latent_moe") else None,
            }
            if checkpoint_path.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path.name}")
            torch.save(
                {
                    "model_state_dict": {name: value.detach().cpu() for name, value in final_model.state_dict().items()},
                    "metadata": checkpoint_metadata,
                },
                checkpoint_path,
            )
            checkpoint_sha = _sha256_file(checkpoint_path)
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            restored_model = _serialize_checkpoint_model(model_name, saved)
            restored_model.load_state_dict(saved["model_state_dict"], strict=True)
            if not np.allclose(_predict_logits(restored_model, valid_x)[0], outer_logits, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"checkpoint reload predictions differ for {model_name}, fold {outer.fold_id}")
            fold_gate: dict[str, Any] = {"model_name": model_name, "outer_fold_id": int(outer.fold_id)}
            if gates is not None:
                mean_gate = gates.mean(axis=0)
                from .models_neural import normalized_gate_entropy

                fold_gate.update(
                    {
                        "mean_gate_entropy_normalized": float(normalized_gate_entropy(torch.from_numpy(gates.astype(np.float32))).item()),
                        "mean_gate_utilization": json.dumps(mean_gate.tolist(), separators=(",", ":")),
                        "collapsed_gate": bool(np.any(mean_gate < 0.05) or np.any(mean_gate > 0.90)),
                        "gate_diagnostic_scope": "outer-held-out OOF windows",
                        "expert_semantics": "latent experts share one proxy-fatigue output; no movement-phase meaning",
                    }
                )
            gate_rows.append(fold_gate)
            threshold_values = {"fixed_0_5": 0.5, "inner_oof_selected": float(threshold.threshold)}
            for strategy, cut in threshold_values.items():
                predicted = (calibrated_probability >= cut).astype(np.int64)
                for index, probability in enumerate(calibrated_probability):
                    outer_prediction_rows.append(
                        {
                            "window_key": str(valid_rows.window_key.iloc[index]),
                            "group_id": str(valid_rows.group_id.iloc[index]),
                            "recording_id": str(valid_rows.recording_id.iloc[index]),
                            "feature_view": "raw_signal",
                            "model_name": model_name,
                            "threshold_strategy": strategy,
                            "decision_threshold": float(cut),
                            "fold_id": int(outer.fold_id),
                            "true_class_id": int(valid_rows.class_id.iloc[index]),
                            "predicted_class_id": int(predicted[index]),
                            "probability_class_0": float(1.0 - probability),
                            "probability_class_1": float(probability),
                            "confidence": float(max(probability, 1.0 - probability)),
                            "abstain": False,
                            "raw_logit": float(outer_logits[index]),
                            "checkpoint_sha256": checkpoint_sha,
                            "label_confidence": float(valid_rows.label_confidence.iloc[index]),
                            "label_reason": str(valid_rows.label_reason.iloc[index]),
                        }
                    )
            checkpoint_rows.append(
                {
                    "model_name": model_name,
                    "outer_fold_id": int(outer.fold_id),
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha,
                    "selected_supervised_epochs": int(selected_epoch),
                    "selected_threshold": float(threshold.threshold),
                    "training_source_groups": list(outer.train_group_ids),
                    "held_out_source_groups": list(outer.validation_group_ids),
                    "normalization_and_pretraining_source_groups": fit_receipt["normalization_source_groups"],
                    "normalization_window_count": fit_receipt["normalization_window_count"],
                    "ssl_pretraining_window_count": fit_receipt["ssl_pretraining_window_count"],
                    "checkpoint_reload_predictions_verified": True,
                }
            )

    neural_predictions = pd.DataFrame(outer_prediction_rows)
    active_key_to_fold = dict(zip(active.window_key.astype(str), active.fold_id.astype(int), strict=True))
    for model_name in requested_models:
        for strategy in THRESHOLD_STRATEGIES:
            view = neural_predictions[
                neural_predictions.model_name.eq(model_name)
                & neural_predictions.threshold_strategy.eq(strategy)
            ]
            if view.window_key.duplicated().any() or set(view.window_key.astype(str)) != expected_window_keys:
                raise RuntimeError(f"{model_name}/{strategy} does not predict each assigned low-frequency window exactly once")
            if any(active_key_to_fold[key] != int(fold) for key, fold in zip(view.window_key.astype(str), view.fold_id, strict=True)):
                raise RuntimeError(f"{model_name}/{strategy} predictions violate frozen outer group folds")
    baseline_rows = baseline.copy()
    baseline_rows["feature_view"] = "relative_time_only"
    baseline_rows["model_name"] = "time_only_logistic"
    baseline_rows["confidence"] = baseline_rows[["probability_class_0", "probability_class_1"]].max(axis=1)
    baseline_rows["abstain"] = False
    baseline_rows["raw_logit"] = np.nan
    baseline_rows["checkpoint_sha256"] = ""
    baseline_rows["label_confidence"] = baseline_rows.window_key.map(active.set_index("window_key").label_confidence)
    baseline_rows["label_reason"] = baseline_rows.window_key.map(active.set_index("window_key").label_reason)
    baseline_rows = baseline_rows.rename(columns={"fold_id": "fold_id"})
    combined = pd.concat([neural_predictions, baseline_rows], ignore_index=True, sort=False)
    joined_output = output_dir / "active_lowfreq_signal_window_manifest.parquet"
    support.to_csv(support_path, index=False)
    combined.to_parquet(output_dir / "neural_lowfreq_oof_predictions.parquet", index=False)
    pd.concat(inner_oof_rows, ignore_index=True).to_parquet(output_dir / "inner_oof_selection_predictions.parquet", index=False)
    pooled_metrics, group_metrics = _nested_metric_tables(combined)
    pooled_metrics.to_csv(output_dir / "neural_lowfreq_metrics.csv", index=False)
    group_metrics.to_csv(output_dir / "neural_lowfreq_group_metrics.csv", index=False)
    paired = _paired_group_bootstrap(combined, int(bootstrap_replicates), int(plan.seed))
    paired.to_csv(output_dir / "paired_group_bootstrap.csv", index=False)
    pd.DataFrame(inner_metric_rows).to_csv(output_dir / "inner_training_history.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(output_dir / "inner_epoch_selection.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "inner_threshold_selection.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration_receipts.csv", index=False)
    pd.DataFrame(calibration_fit_rows).to_json(output_dir / "calibration_crossfit_receipts.json", orient="records", indent=2)
    pd.DataFrame(gate_rows).to_csv(output_dir / "moe_gate_diagnostics.csv", index=False)
    _write_json(output_dir / "checkpoint_manifest.json", checkpoint_rows)

    input_paths = [
        bank_manifest_path, bank_path, metadata_path, labels_path,
        parent_receipt_path, fold_path, predictions_path,
    ]
    if external_encoder_path is not None:
        input_paths.append(external_encoder_path)
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("evaluation_neural.py"),
        Path(__file__).with_name("external_transfer.py"),
        Path(__file__).with_name("models_neural.py"),
        Path(__file__).with_name("ssl_pretraining.py"),
        Path(__file__).with_name("evaluation.py"),
        Path(__file__).with_name("metrics.py"),
    ]
    output_paths = [
        joined_output,
        support_path,
        output_dir / "neural_lowfreq_oof_predictions.parquet",
        output_dir / "inner_oof_selection_predictions.parquet",
        output_dir / "neural_lowfreq_metrics.csv",
        output_dir / "neural_lowfreq_group_metrics.csv",
        output_dir / "paired_group_bootstrap.csv",
        output_dir / "inner_training_history.csv",
        output_dir / "inner_epoch_selection.csv",
        output_dir / "inner_threshold_selection.csv",
        output_dir / "calibration_receipts.csv",
        output_dir / "calibration_crossfit_receipts.json",
        output_dir / "moe_gate_diagnostics.csv",
        output_dir / "checkpoint_manifest.json",
        *sorted(checkpoint_dir.glob("*.pt")),
    ]
    receipt = {
        "stage": "development_nested_grouped_neural_models_low_frequency_proxy",
        "status": "completed",
        "label_rule": PRIMARY_LABEL_RULE,
        "label_rule_type": PROXY_LABEL_TYPE,
        "window_count": int(len(metadata)),
        "assigned_development_windows_descriptive_only": int(len(active)),
        "class_0_assigned_windows_descriptive_only": int(assigned_counts[0]),
        "class_1_assigned_windows_descriptive_only": int(assigned_counts[1]),
        "class_0_source_groups": int(active.loc[active.class_id.eq(0), "group_id"].nunique()),
        "class_1_source_groups": int(active.loc[active.class_id.eq(1), "group_id"].nunique()),
        "assigned_label_coverage": float(len(active) / len(metadata)),
        "independent_unit": "source folder group; participant identity unverified",
        "overlapping_windows_treated_as_independent_groups": False,
        "outer_group_folds": len(plan.outer_folds),
        "inner_group_folds_per_outer": len(plan.inner_folds[plan.outer_folds[0].fold_id]),
        "fold_source": "historical 4x3 low-frequency threshold parent; SHA-256 verified",
        "coverage_minima_per_fold_role": {"source_groups_per_class": 2, "recording_trials_per_class": 2},
        "coverage_gate_passed": True,
        "coverage_file": "label_support_by_fold.csv",
        "models": list(requested_models),
        "external_encoder_checkpoint_path": str(external_encoder_path) if external_encoder_path is not None else None,
        "external_encoder_checkpoint_sha256": external_encoder_sha256,
        "max_supervised_epochs": int(max_epochs),
        "supervised_epoch_selection": "mean inner-fold group-balanced Macro-F1 at fixed 0.5 on validation groups",
        "calibration_method": "group-weighted sigmoid Platt; calibration fit uses group-held-out logits inside training groups",
        "threshold_comparison": ["fixed 0.5", "inner OOF selected group-balanced Macro-F1"],
        "threshold_selection_objective": "group-balanced Macro-F1 on calibrated inner OOF probabilities only",
        "training_loss_group_weighted": True,
        "normalization_fit_training_groups_only": True,
        "ssl_pretraining": "fresh masked-span reconstruction on eligible non-overlap development windows from training groups only",
        "frozen_ssl_variant": "encoder parameters frozen during supervised head fit" if "ssl_encoder_frozen" in requested_models else None,
        "moe_expert_counts": {name: (2 if name == "latent_moe2" else 3) for name in requested_models if name.startswith("latent_moe")},
        "ssl_pretraining_epochs": int(ssl_pretrain_epochs),
        "model_outputs": "single weak-fatigue-proxy logit; MoE experts remain latent",
        "temporal_baseline": "relative-time-only logistic; identical outer folds, windows and low-frequency labels",
        "paired_uncertainty": f"{int(bootstrap_replicates)} paired source-group bootstrap replicates",
        "checkpoint_count": len(checkpoint_rows),
        "checkpoint_hashes_verified_after_reload": all(bool(row["checkpoint_reload_predictions_verified"]) for row in checkpoint_rows),
        "internal_test_signal_values_parsed": 0,
        "reserved_test_member_payloads_read": 0,
        "external_rar_opened": False,
        "source_bank_label_rule_replaced_by_checked_join": True,
        "source_bank_original_rule_metadata": bank_receipt.get("pseudo_label_rule"),
        "spectral_sensitivity_training_status": "not run: sparse rule fails historical 4x3 inner-fold support",
        "input_sha256": {str(path): _sha256_file(path) for path in input_paths},
        "source_sha256": {str(path): _sha256_file(path) for path in source_paths},
        "output_sha256": {str(path): _sha256_file(path) for path in output_paths},
        "group_support_rows": int(len(support)),
        "interpretation": "proxy-label agreement only; not independent physiological fatigue validation",
    }
    manifest_path = output_dir / "neural_supervised_stage_manifest.json"
    _write_json(manifest_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path.resolve())}
