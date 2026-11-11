"""Nested development-only noise augmentation and matched robustness analysis."""

from __future__ import annotations

import hashlib
import json
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import signal as scipy_signal
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from .augmentation import (
    CORRUPTION_FAMILIES,
    EVALUATION_LEVELS,
    MAX_SHIFT_FRACTION,
    TRAINING_LEVELS,
    corrupt_window,
    stable_seed,
)
from .contracts import SignalWindow
from .evaluation import select_group_balanced_threshold
from .evaluation_guardrails import (
    OOD_FEATURES,
    _robust_location_scale,
    _robust_ood_score,
    _select_confidence_threshold,
    _verify_neural_stage,
)
from .evaluation_neural import (
    MODEL_NAMES,
    _load_fold_plan,
    _model_output,
    _new_model,
    _sha256_file,
    audit_group_fold_support,
    select_fold_pretraining_indices,
)
from .evaluation_neural_nested import (
    PRIMARY_LABEL_RULE,
    PROXY_LABEL_TYPE,
    _apply_platt,
    _calibration_splits,
    _class_support,
    _fit_platt_calibrator,
    _group_balanced_macro_f1,
    _predict_logits,
    _select_epoch,
    _serialize_checkpoint_model,
    _weighted_confusion_metrics,
    join_primary_lowfreq_labels,
    verify_lowfreq_threshold_parent_stage,
)
from .features import window_key
from .metrics import compute_metrics
from .pseudo_labels import label_low_frequency_fraction_rule, technical_reference


OOD_QUANTILE = 0.995
CONFIDENCE_CANDIDATES_DEFAULT = (0.55, 0.65, 0.75)
RANDOM_SEED = 20261115


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


def _hash_map(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha256_file(path) for path in paths}


def _verify_hash_map(mapping: dict[str, str], label: str) -> None:
    failures: list[str] = []
    for name, expected in mapping.items():
        path = Path(name).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != str(expected):
            failures.append(str(path))
    if failures:
        raise ValueError(f"{label} checksum verification failed: " + "; ".join(failures))


def _verify_private_hash_map(mapping: dict[str, str], private_dir: Path, label: str) -> None:
    private_root = Path(private_dir).expanduser().resolve()
    scoped: dict[str, str] = {}
    for name, digest in mapping.items():
        path = Path(name).expanduser().resolve()
        try:
            path.relative_to(private_root)
        except ValueError:
            # Parent receipts sometimes bundle source-file hashes with data hashes.
            # Source hashes remain in their original receipts; this stage separately
            # records the exact current source code it executes.
            continue
        scoped[str(path)] = str(digest)
    if not scoped:
        raise ValueError(f"{label} has no private data inputs to verify")
    _verify_hash_map(scoped, label)


def _support(rows: pd.DataFrame, partition: str) -> dict[str, Any]:
    if set(rows.class_id.astype(int).unique()) != {0, 1}:
        raise ValueError(f"{partition} lacks one or both operational proxy classes")
    result: dict[str, Any] = {}
    for class_id in (0, 1):
        subset = rows.loc[rows.class_id.astype(int).eq(class_id)]
        groups = int(subset.group_id.astype(str).nunique())
        trials = int(subset[["group_id", "recording_id"]].astype(str).drop_duplicates().shape[0])
        if groups < 2 or trials < 2:
            raise ValueError(
                f"{partition} class {class_id} support is insufficient: "
                f"source_groups={groups}, recording_trials={trials}"
            )
        result[str(class_id)] = {
            "assigned_windows_descriptive_only": int(len(subset)),
            "source_groups": groups,
            "recording_trials": trials,
        }
    return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)))).astype(np.float64)


def _assign_training_schedule(rows: pd.DataFrame) -> pd.DataFrame:
    """Assign one balanced, deterministic family/severity to each clean training window."""
    schedule_rows: list[dict[str, Any]] = []
    family_count = len(CORRUPTION_FAMILIES)
    level_count = len(TRAINING_LEVELS)
    for (group_id, class_id), subset in rows.groupby(["group_id", "class_id"], sort=True):
        ordered = subset.sort_values("window_key")
        offset = stable_seed("emg-training-schedule-v1", group_id, int(class_id)) % (family_count * level_count)
        for rank, row in enumerate(ordered.itertuples(index=False)):
            choice = (int(rank) + int(offset)) % (family_count * level_count)
            family_index = choice % family_count
            level_index = choice // family_count
            schedule_rows.append(
                {
                    "window_key": str(row.window_key),
                    "group_id": str(group_id),
                    "class_id": int(class_id),
                    "family": CORRUPTION_FAMILIES[family_index],
                    "level": float(TRAINING_LEVELS[level_index]),
                    "seed": stable_seed(
                        "emg-training-corruption-v1",
                        str(row.window_key),
                        CORRUPTION_FAMILIES[family_index],
                        TRAINING_LEVELS[level_index],
                    ),
                }
            )
    result = pd.DataFrame(schedule_rows)
    if len(result) != len(rows) or result.window_key.duplicated().any():
        raise RuntimeError("training augmentation schedule does not map one-to-one to clean training windows")
    return result


def _build_training_augmentation_bank(
    metadata: pd.DataFrame,
    signals: np.ndarray,
    notch_by_window: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, set[str]]:
    active = metadata.loc[metadata.class_id.isin((0, 1))].copy()
    schedule = _assign_training_schedule(active)
    key_to_position = dict(zip(metadata.window_key.astype(str), range(len(metadata)), strict=True))
    corrupted: dict[str, np.ndarray] = {}
    augmented_full = np.asarray(signals, dtype=np.float32).copy()
    receipts: list[dict[str, Any]] = []
    for row in schedule.itertuples(index=False):
        position = int(key_to_position[str(row.window_key)])
        out, receipt = corrupt_window(
            signals[position],
            str(row.family),
            float(row.level),
            int(row.seed),
            sampling_rate_hz=float(metadata.iloc[position].sampling_rate_hz),
        )
        corrupted[str(row.window_key)] = out
        augmented_full[position] = out
        receipts.append(
            {
                "window_key": str(row.window_key),
                "group_id": str(row.group_id),
                "class_id": int(row.class_id),
                **receipt,
            }
        )
    receipt_frame = pd.DataFrame(receipts)
    receipt_frame["corruption_parameters_json"] = receipt_frame.parameters.map(
        lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    )
    receipt_frame = receipt_frame.drop(columns=["parameters"])
    replayed = _recompute_lowfreq_rule(metadata, augmented_full, notch_by_window, config)
    receipt_frame = receipt_frame.merge(
        replayed[["window_key", "recomputed_class_id", "recomputed_label_reason"]],
        on="window_key",
        how="left",
        validate="one_to_one",
    )
    if receipt_frame.recomputed_class_id.isna().any() or receipt_frame.class_id.isna().any():
        raise RuntimeError("training augmentation label replay omitted an assigned window")
    receipt_frame["label_preserved_by_frozen_rule"] = receipt_frame.recomputed_class_id.astype(int).eq(
        receipt_frame.class_id.astype(int)
    )
    preserved = set(
        receipt_frame.loc[receipt_frame.label_preserved_by_frozen_rule, "window_key"].astype(str)
    )
    return corrupted, receipt_frame, preserved


def _fit_augmented_model(
    model_name: str,
    rows: pd.DataFrame,
    metadata: pd.DataFrame,
    signals: np.ndarray,
    augmented_by_key: dict[str, np.ndarray],
    preserved_augmentation_keys: set[str],
    groups: Sequence[str],
    *,
    epochs: int,
    seed: int,
    ssl_pretrain_epochs: int,
    validation_rows: pd.DataFrame | None = None,
) -> tuple[Any, float, float, dict[str, Any], dict[str, list[Any]]]:
    """Fit on clean plus augmented training copies; validation and SSL remain clean."""
    selected_groups = {str(value) for value in groups}
    raw_positions = select_fold_pretraining_indices(metadata, sorted(selected_groups))
    if not set(metadata.iloc[raw_positions].group_id.astype(str)).issubset(selected_groups):
        raise RuntimeError("normalization or SSL pretraining admitted a held-out source group")
    scaler_mean = float(signals[raw_positions].mean())
    scaler_std = float(signals[raw_positions].std())
    if not np.isfinite(scaler_mean) or not np.isfinite(scaler_std) or scaler_std <= 1e-8:
        raise ValueError("training-fold clean signal normalization has invalid dispersion")
    positions = rows.bank_position.to_numpy(dtype=int)
    clean_train = signals[positions]
    keys = rows.window_key.astype(str).tolist()
    stable_rows = rows.loc[rows.window_key.astype(str).isin(preserved_augmentation_keys)].copy()
    _support(stable_rows, "rule-preserving augmented training examples")
    stable_keys = stable_rows.window_key.astype(str).tolist()
    try:
        augmented_train = np.stack([augmented_by_key[key] for key in stable_keys]).astype(np.float32, copy=False)
    except KeyError as error:
        raise RuntimeError("rule-preserving augmentation bank omitted a selected training key") from error
    train_x = np.concatenate((clean_train, augmented_train), axis=0)
    train_x = (train_x - scaler_mean) / scaler_std
    train_y = np.concatenate((rows.class_id.to_numpy(dtype=int), stable_rows.class_id.to_numpy(dtype=int)))
    train_groups = rows.group_id.astype(str).tolist() + stable_rows.group_id.astype(str).tolist()
    pretraining_x = (signals[raw_positions] - scaler_mean) / scaler_std
    validation_x = None
    if validation_rows is not None:
        validation_x = (
            signals[validation_rows.bank_position.to_numpy(dtype=int)] - scaler_mean
        ) / scaler_std
    model, ssl_fit = _new_model(
        model_name,
        int(seed),
        pretraining_x if model_name == "ssl_encoder_finetuned" else None,
        int(ssl_pretrain_epochs),
    )
    from .evaluation_neural_nested import _train_with_validation_logits

    history = _train_with_validation_logits(
        model,
        train_x,
        train_y,
        train_groups,
        epochs=int(epochs),
        validation_signals=validation_x,
        validation_labels=None if validation_rows is None else validation_rows.class_id.to_numpy(dtype=int),
        validation_groups=None if validation_rows is None else validation_rows.group_id.astype(str).tolist(),
    )
    fit_receipt = {
        "normalization_mean_from_clean_training_windows": scaler_mean,
        "normalization_std_from_clean_training_windows": scaler_std,
        "normalization_and_ssl_window_count": int(len(raw_positions)),
        "normalization_and_ssl_source_groups": sorted(selected_groups),
        "ssl_pretraining_window_count": 0 if ssl_fit is None else int(ssl_fit.train_window_count),
        "ssl_pretraining_epochs": 0 if ssl_fit is None else int(len(ssl_fit.reconstruction_losses)),
        "ssl_reconstruction_losses": [] if ssl_fit is None else list(ssl_fit.reconstruction_losses),
        "training_clean_window_keys_descriptive_only": int(len(rows)),
        "training_augmented_window_keys_descriptive_only": int(len(stable_rows)),
        "training_unique_clean_window_keys": int(len(set(keys))),
        "augmentation_label_preserving_rule": "recomputed low_frequency_fraction_persistence_v1 must assign the same class on the mixed deterministic corruption schedule",
        "augmentation_label_preserving_window_keys_descriptive_only": int(len(stable_rows)),
        "training_groups": sorted(set(rows.group_id.astype(str))),
        "training_only_augmentation": True,
        "augmented_labels": "the original clean pseudo-label is retained without recomputation",
    }
    return model, scaler_mean, scaler_std, fit_receipt, history


def _crossfit_augmented_calibrator(
    *,
    model_name: str,
    train_rows: pd.DataFrame,
    metadata: pd.DataFrame,
    signals: np.ndarray,
    augmented_by_key: dict[str, np.ndarray],
    preserved_augmentation_keys: set[str],
    selected_epoch: int,
    split_seed: int,
    ssl_pretrain_epochs: int,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logits_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    recording_parts: list[np.ndarray] = []
    fit_receipts: list[dict[str, Any]] = []
    splits = _calibration_splits(train_rows, int(split_seed))
    groups = sorted(set(train_rows.group_id.astype(str)))
    for calibration_index, (fit_indices, heldout_indices) in enumerate(splits, start=1):
        fit_rows = train_rows.iloc[fit_indices].sort_values("window_key").reset_index(drop=True)
        heldout_rows = train_rows.iloc[heldout_indices].sort_values("window_key").reset_index(drop=True)
        seed = stable_seed("emg-aug-calibration-model-v1", split_seed, model_name, calibration_index)
        model, mean, std, fit_receipt, _ = _fit_augmented_model(
            model_name,
            fit_rows,
            metadata,
            signals,
            augmented_by_key,
            preserved_augmentation_keys,
            sorted(set(fit_rows.group_id.astype(str))),
            epochs=selected_epoch,
            seed=seed,
            ssl_pretrain_epochs=ssl_pretrain_epochs,
        )
        heldout_x = (signals[heldout_rows.bank_position.to_numpy(dtype=int)] - mean) / std
        heldout_logits, _ = _predict_logits(model, heldout_x)
        logits_parts.append(heldout_logits)
        truth_parts.append(heldout_rows.class_id.to_numpy(dtype=int))
        group_parts.append(heldout_rows.group_id.astype(str).to_numpy())
        recording_parts.append(heldout_rows.recording_id.astype(str).to_numpy())
        fit_receipts.append(
            {
                "calibration_split_id": int(calibration_index),
                "fit_source_groups": sorted(set(fit_rows.group_id.astype(str))),
                "heldout_calibration_groups": sorted(set(heldout_rows.group_id.astype(str))),
                "group_overlap": False,
                "selected_epochs": int(selected_epoch),
                "fit_receipt": fit_receipt,
            }
        )
    pooled_logits = np.concatenate(logits_parts)
    pooled_truth = np.concatenate(truth_parts)
    pooled_groups = np.concatenate(group_parts)
    pooled_recordings = np.concatenate(recording_parts)
    if set(pooled_groups.tolist()) != set(groups):
        raise RuntimeError("augmented calibration crossfit did not cover each outer-training group")
    calibrator, calibrator_receipt = _fit_platt_calibrator(
        pooled_logits, pooled_truth, pooled_groups, pooled_recordings
    )
    return (
        calibrator,
        calibrator_receipt,
        fit_receipts,
        pooled_logits,
        pooled_truth,
        pooled_groups,
        pooled_recordings,
    )


def _augmented_inner_oof(
    *,
    model_name: str,
    outer_fold: Any,
    inner_fold: Any,
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    valid_logits: np.ndarray,
    metadata: pd.DataFrame,
    signals: np.ndarray,
    augmented_by_key: dict[str, np.ndarray],
    preserved_augmentation_keys: set[str],
    selected_epoch: int,
    plan_seed: int,
    model_index: int,
    outer_index: int,
    ssl_pretrain_epochs: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    split_seed = int(plan_seed + outer_index * 100000 + model_index * 10000 + int(inner_fold.fold_id) * 1000)
    calibrator, calibrator_receipt, fit_receipts, _, _, pooled_groups, _ = _crossfit_augmented_calibrator(
        model_name=model_name,
        train_rows=train_rows,
        metadata=metadata,
        signals=signals,
        augmented_by_key=augmented_by_key,
        preserved_augmentation_keys=preserved_augmentation_keys,
        selected_epoch=selected_epoch,
        split_seed=split_seed,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
    )
    probability = _apply_platt(calibrator, valid_logits)
    result = pd.DataFrame(
        {
            "window_key": valid_rows.window_key.astype(str).to_numpy(),
            "group_id": valid_rows.group_id.astype(str).to_numpy(),
            "recording_id": valid_rows.recording_id.astype(str).to_numpy(),
            "model_name": model_name,
            "outer_fold_id": int(outer_fold.fold_id),
            "inner_fold_id": int(inner_fold.fold_id),
            "true_class_id": valid_rows.class_id.to_numpy(dtype=int),
            "raw_logit": valid_logits.astype(np.float64),
            "probability_class_0": 1.0 - probability,
            "probability_class_1": probability,
        }
    )
    receipt = {
        "model_name": model_name,
        "outer_fold_id": int(outer_fold.fold_id),
        "inner_fold_id": int(inner_fold.fold_id),
        "selected_supervised_epochs": int(selected_epoch),
        "calibration_training_groups": sorted(set(pooled_groups.tolist())),
        "calibration_validation_groups": sorted(set(valid_rows.group_id.astype(str))),
        "validation_excluded_from_calibration": True,
        "calibration_method": "group-weighted sigmoid Platt on group-held-out logits from augmented models",
        "calibrator": calibrator_receipt,
    }
    return result, receipt, fit_receipts


def _morphology_batch(values: np.ndarray, sampling_rate_hz: float) -> pd.DataFrame:
    """Vectorized equivalent of the morphology-only OOD features."""
    x = np.asarray(values, dtype=np.float64)
    fs = float(sampling_rate_hz)
    if x.ndim != 2 or x.shape[1] < 64 or not np.isfinite(x).all():
        raise ValueError("morphology feature input must be a finite two-dimensional window matrix")
    mean = x.mean(axis=1)
    median = np.median(x, axis=1)
    centered = x - mean[:, None]
    variance = np.mean(centered * centered, axis=1)
    rms = np.sqrt(np.mean(x * x, axis=1))
    threshold = 0.01 * rms
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)
    diff_variance = np.var(dx, axis=1)
    second_diff_variance = np.var(ddx, axis=1)
    mobility = np.divide(
        fs * np.sqrt(np.maximum(diff_variance, 0.0)),
        np.sqrt(np.maximum(variance, 0.0)),
        out=np.full(len(x), np.nan),
        where=variance > 0,
    )
    derivative_mobility = np.divide(
        fs * np.sqrt(np.maximum(second_diff_variance, 0.0)),
        np.sqrt(np.maximum(diff_variance, 0.0)),
        out=np.full(len(x), np.nan),
        where=diff_variance > 0,
    )
    complexity = np.divide(
        derivative_mobility,
        mobility,
        out=np.full(len(x), np.nan),
        where=np.isfinite(mobility) & (mobility > 0),
    )
    standardized = np.divide(
        centered,
        np.sqrt(np.maximum(variance, 0.0))[:, None],
        out=np.zeros_like(centered),
        where=variance[:, None] > 0,
    )
    crossing = (
        ((x[:, :-1] > threshold[:, None]) & (x[:, 1:] < -threshold[:, None]))
        | ((x[:, :-1] < -threshold[:, None]) & (x[:, 1:] > threshold[:, None]))
    )
    slope_changes = (
        (dx[:, :-1] * dx[:, 1:] < 0)
        & (np.abs(dx[:, :-1]) > threshold[:, None])
        & (np.abs(dx[:, 1:]) > threshold[:, None])
    )
    return pd.DataFrame(
        {
            "rms": rms,
            "mav": np.mean(np.abs(x), axis=1),
            "iemg": np.sum(np.abs(x), axis=1) / fs,
            "variance": variance,
            "waveform_length": np.sum(np.abs(dx), axis=1),
            "waveform_length_per_second": np.sum(np.abs(dx), axis=1) * fs / x.shape[1],
            "zero_crossings": crossing.sum(axis=1).astype(float),
            "slope_sign_changes": slope_changes.sum(axis=1).astype(float),
            "willison_amplitude": np.sum(np.abs(dx) > threshold[:, None], axis=1).astype(float),
            "median_absolute_deviation": np.median(np.abs(x - median[:, None]), axis=1),
            "interquartile_range": np.quantile(x, 0.75, axis=1) - np.quantile(x, 0.25, axis=1),
            "skewness": np.mean(standardized**3, axis=1),
            "kurtosis_excess": np.mean(standardized**4, axis=1) - 3.0,
            "hjorth_activity": variance,
            "hjorth_mobility_hz": mobility,
            "hjorth_complexity": complexity,
        }
    )


def _label_spectral_batch(
    values: np.ndarray,
    sampling_rate_hz: float,
    notch_by_window: np.ndarray,
    line_exclusion_hz: float,
) -> pd.DataFrame:
    """Vectorized spectral indicators used by the frozen low-frequency rule."""
    x = np.asarray(values, dtype=np.float64)
    fs = float(sampling_rate_hz)
    notch = np.asarray(notch_by_window, dtype=np.float64)
    if x.ndim != 2 or notch.shape != (len(x),):
        raise ValueError("spectral indicator input dimensions are inconsistent")
    frequency, psd = scipy_signal.welch(
        x,
        fs=fs,
        nperseg=min(512, x.shape[1]),
        scaling="density",
        axis=-1,
    )
    high_frequency = min(400.0, fs / 2.0)
    supported = (frequency >= 20.0) & (frequency <= high_frequency)
    supported_frequency = frequency[supported]
    supported_psd = np.maximum(psd[:, supported], 0.0)
    total = supported_psd.sum(axis=1)
    probabilities = np.divide(
        supported_psd,
        total[:, None],
        out=np.zeros_like(supported_psd),
        where=total[:, None] > np.finfo(float).eps,
    )
    mean_frequency = probabilities @ supported_frequency
    median_index = np.argmax(np.cumsum(probabilities, axis=1) >= 0.5, axis=1)
    median_frequency = supported_frequency[median_index]
    bin_width = float(frequency[1] - frequency[0]) if frequency.size > 1 else 0.0
    fraction = np.full(len(x), np.nan, dtype=np.float64)
    log_ratio = np.full(len(x), np.nan, dtype=np.float64)
    for line_frequency in np.unique(notch[np.isfinite(notch)]):
        rows = np.isclose(notch, float(line_frequency), rtol=0.0, atol=1e-9)
        keep = np.abs(frequency - float(line_frequency)) > float(line_exclusion_hz)
        low_mask = (frequency >= 20.0) & (frequency < 80.0) & keep
        high_mask = (frequency >= 80.0) & (frequency < high_frequency + np.finfo(float).eps) & keep
        low = np.maximum(psd[rows][:, low_mask], 0.0).sum(axis=1) * bin_width
        high = np.maximum(psd[rows][:, high_mask], 0.0).sum(axis=1) * bin_width
        fraction[rows] = np.divide(
            low,
            low + high,
            out=np.full(len(low), np.nan),
            where=(low + high) > 0,
        )
        log_ratio[rows] = np.log(np.divide(low, high, out=np.full(len(low), np.nan), where=(low > 0) & (high > 0)))
    rows_without_notch = ~np.isfinite(notch)
    if rows_without_notch.any():
        low_mask = (frequency >= 20.0) & (frequency < 80.0)
        high_mask = (frequency >= 80.0) & (frequency < high_frequency + np.finfo(float).eps)
        low = np.maximum(psd[rows_without_notch][:, low_mask], 0.0).sum(axis=1) * bin_width
        high = np.maximum(psd[rows_without_notch][:, high_mask], 0.0).sum(axis=1) * bin_width
        fraction[rows_without_notch] = np.divide(
            low,
            low + high,
            out=np.full(len(low), np.nan),
            where=(low + high) > 0,
        )
        log_ratio[rows_without_notch] = np.log(
            np.divide(low, high, out=np.full(len(low), np.nan), where=(low > 0) & (high > 0))
        )
    rms = np.sqrt(np.mean(x * x, axis=1))
    valid = (
        (np.var(x, axis=1) > np.finfo(float).eps)
        & (total > np.finfo(float).eps)
        & np.isfinite(mean_frequency)
        & np.isfinite(median_frequency)
        & np.isfinite(fraction)
        & np.isfinite(log_ratio)
        & np.isfinite(rms)
    )
    return pd.DataFrame(
        {
            "feature_valid": valid.astype(int),
            "rms": rms,
            "mean_frequency_hz": mean_frequency,
            "median_frequency_hz": median_frequency,
            "line_excluded_low_high_log_ratio": log_ratio,
            "line_excluded_low_frequency_fraction_20_80": fraction,
        }
    )


def _recompute_lowfreq_rule(
    metadata: pd.DataFrame,
    corrupted: np.ndarray,
    notch_by_window: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    fs_values = metadata.sampling_rate_hz.to_numpy(dtype=float)
    if not np.allclose(fs_values, fs_values[0], rtol=0.0, atol=1e-6):
        raise ValueError("the private 500 ms signal bank has inconsistent sampling rates")
    feature_frame = _label_spectral_batch(
        corrupted,
        float(fs_values[0]),
        notch_by_window,
        float(config["labels"]["line_exclusion_hz"]),
    )
    keys = metadata.window_key.astype(str).to_numpy()
    feature_by_key = {
        key: feature_frame.iloc[index].to_dict()
        for index, key in enumerate(keys)
    }
    windows: list[SignalWindow] = []
    for position, row in enumerate(metadata.itertuples(index=False)):
        windows.append(
            SignalWindow(
                group_id=str(row.group_id),
                recording_id=str(row.recording_id),
                segment_index=int(row.segment_index),
                window_index=int(row.window_index),
                start_sample=int(row.start_sample),
                start_time_s=float(row.start_time_s),
                duration_s=float(row.duration_s),
                sampling_rate_hz=float(row.sampling_rate_hz),
                signal=corrupted[position],
                activity_occupancy=float(row.activity_occupancy),
                eligible_for_label=bool(row.eligible_for_label),
                rejection_reason=(str(row.rejection_reason) or None),
            )
        )
    result_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[SignalWindow]] = defaultdict(list)
    for window in windows:
        grouped[(window.group_id, window.recording_id, int(window.segment_index))].append(window)
    for identity in sorted(grouped):
        segment_windows = sorted(grouped[identity], key=lambda item: (item.start_time_s, item.start_sample))
        reference_windows = [item for item in segment_windows if int(item.window_index) % 2 == 0]
        reference = technical_reference(
            reference_windows,
            feature_by_key,
            config,
            stability_components=("low_fraction",),
        )
        labels = label_low_frequency_fraction_rule(segment_windows, feature_by_key, reference, config)
        if len(labels) != len(segment_windows):
            raise RuntimeError("frozen low-frequency label recomputation changed a segment window count")
        result_rows.extend(
            {
                "window_key": window_key(window),
                "group_id": str(window.group_id),
                "recording_id": str(window.recording_id),
                "segment_index": int(window.segment_index),
                "recomputed_class_id": int(label.class_id),
                "recomputed_label_reason": str(label.label_reason),
            }
            for window, label in zip(segment_windows, labels, strict=True)
        )
    result = pd.DataFrame(result_rows)
    if result.window_key.duplicated().any() or set(result.window_key) != set(keys):
        raise RuntimeError("frozen low-frequency recomputation failed exact window-key coverage")
    return result


def _label_consistency_rows(
    original: pd.DataFrame,
    recomputed: pd.DataFrame,
    family: str,
    level: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    joined = original.loc[original.class_id.isin((0, 1)), ["window_key", "group_id", "recording_id", "class_id"]].merge(
        recomputed[["window_key", "recomputed_class_id"]],
        on="window_key",
        how="left",
        validate="one_to_one",
    )
    if joined.recomputed_class_id.isna().any():
        raise RuntimeError("label-consistency comparison omitted original assigned windows")
    truth = joined.class_id.astype(int).to_numpy()
    changed = joined.recomputed_class_id.astype(int).to_numpy()
    summary = {
        "family": family,
        "requested_level": float(level),
        "stress_test": bool(level > 0.20),
        "originally_assigned_window_count_descriptive_only": int(len(joined)),
        "source_group_count": int(joined.group_id.nunique()),
        "assigned_coverage_after_corruption": float(np.mean(np.isin(changed, (0, 1)))),
        "exact_class_agreement_including_ambiguity_as_disagreement": float(np.mean(changed == truth)),
        "newly_ambiguous_fraction": float(np.mean(changed == -1)),
        "class_0_retention": float(np.mean(changed[truth == 0] == 0)) if np.any(truth == 0) else None,
        "class_1_retention": float(np.mean(changed[truth == 1] == 1)) if np.any(truth == 1) else None,
        "class_0_to_1_flip_fraction": float(np.mean(changed[truth == 0] == 1)) if np.any(truth == 0) else None,
        "class_1_to_0_flip_fraction": float(np.mean(changed[truth == 1] == 0)) if np.any(truth == 1) else None,
        "interpretation": "rule agreement after signal perturbation; clean training pseudo-labels remain unchanged",
    }
    group_rows: list[dict[str, Any]] = []
    for (group_id, class_id), rows in joined.groupby(["group_id", "class_id"], sort=True):
        y = rows.class_id.to_numpy(dtype=int)
        y_new = rows.recomputed_class_id.to_numpy(dtype=int)
        group_rows.append(
            {
                "family": family,
                "requested_level": float(level),
                "stress_test": bool(level > 0.20),
                "group_id": str(group_id),
                "original_class_id": int(class_id),
                "originally_assigned_window_count_descriptive_only": int(len(rows)),
                "retention_fraction": float(np.mean(y_new == y)),
                "ambiguity_fraction": float(np.mean(y_new == -1)),
                "flip_fraction": float(np.mean(np.isin(y_new, (0, 1)) & (y_new != y))),
            }
        )
    return summary, group_rows


def _fit_fold_ood_reference(
    clean_features: pd.DataFrame,
    train_group_ids: Sequence[str],
    *,
    quantile: float,
    outer_fold_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    train = clean_features.loc[
        clean_features.group_id.astype(str).isin({str(value) for value in train_group_ids})
        & clean_features.eligible_for_label.fillna(False).astype(bool)
    ].copy()
    groups = sorted(set(train.group_id.astype(str)))
    if len(groups) < 4:
        raise ValueError("group-calibrated OOD reference requires at least four outer-training groups")
    calibration_group_p95: list[float] = []
    calibration_receipts: list[dict[str, Any]] = []
    for heldout_group in groups:
        fit = train.loc[~train.group_id.astype(str).eq(heldout_group)]
        heldout = train.loc[train.group_id.astype(str).eq(heldout_group)]
        center, scale, keep = _robust_location_scale(fit[list(OOD_FEATURES)].to_numpy(dtype=float))
        heldout_scores = _robust_ood_score(heldout[list(OOD_FEATURES)].to_numpy(dtype=float), center, scale, keep)
        p95 = float(np.quantile(heldout_scores, 0.95))
        calibration_group_p95.append(p95)
        calibration_receipts.append(
            {"heldout_group": heldout_group, "window_count_descriptive_only": int(len(heldout)), "score_p95": p95}
        )
    threshold = float(np.quantile(calibration_group_p95, float(quantile)))
    center, scale, keep = _robust_location_scale(train[list(OOD_FEATURES)].to_numpy(dtype=float))
    receipt = {
        "outer_fold_id": int(outer_fold_id),
        "training_source_groups": groups,
        "features": [name for name, retained in zip(OOD_FEATURES, keep, strict=True) if retained],
        "threshold_quantile_of_leave_group_out_p95": float(quantile),
        "threshold": threshold,
        "uses_labels": False,
        "held_out_groups_used_for_fit_or_threshold": False,
        "group_level_calibration": calibration_receipts,
    }
    return center, scale, keep, threshold, receipt


def _checkpoint_model(path: Path, expected_name: str) -> tuple[Any, dict[str, Any]]:
    document = torch.load(Path(path), map_location="cpu", weights_only=True)
    metadata = document.get("metadata", {})
    if str(metadata.get("model_name")) != str(expected_name):
        raise ValueError("checkpoint model name disagrees with the expected architecture")
    model = _serialize_checkpoint_model(expected_name, document)
    model.load_state_dict(document["model_state_dict"], strict=True)
    model.eval()
    return model, metadata


def _predict_logits_chunked(model: Any, values: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray | None]:
    logits_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    have_gates: bool | None = None
    for start in range(0, len(values), int(batch_size)):
        logits, gates = _predict_logits(model, values[start : start + int(batch_size)])
        logits_parts.append(logits)
        if gates is not None:
            gate_parts.append(gates)
            have_gates = True
        elif have_gates is None:
            have_gates = False
    logits = np.concatenate(logits_parts) if logits_parts else np.empty(0, dtype=np.float64)
    gates_out = np.concatenate(gate_parts) if gate_parts else None
    return logits, gates_out


def _gate_receipt(gates: np.ndarray | None, model_name: str, **identity: Any) -> dict[str, Any] | None:
    if gates is None:
        return None
    values = np.asarray(gates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("latent MoE gates must be a non-empty n-by-3 matrix")
    mean_load = values.mean(axis=0)
    entropy = -np.sum(np.clip(values, 1e-12, 1.0) * np.log(np.clip(values, 1e-12, 1.0)), axis=1)
    normalized_entropy = entropy / np.log(3.0)
    return {
        **identity,
        "model_name": str(model_name),
        "mean_expert_share_1": float(mean_load[0]),
        "mean_expert_share_2": float(mean_load[1]),
        "mean_expert_share_3": float(mean_load[2]),
        "normalized_gate_entropy": float(np.mean(normalized_entropy)),
        "collapsed_gate": bool(np.max(mean_load) > 0.95 or np.mean(normalized_entropy) < 0.001),
        "window_count_descriptive_only": int(len(values)),
    }


def _group_normalized_confusion(rows: pd.DataFrame, prediction_column: str) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for group_id, group_rows in rows.groupby("group_id", sort=True):
        truth = group_rows.true_class_id.to_numpy(dtype=int)
        prediction = group_rows[prediction_column].to_numpy(dtype=int)
        if set(np.unique(truth).tolist()) != {0, 1}:
            raise ValueError(f"source group {group_id} lacks both proxy classes for paired metrics")
        matrix = confusion_matrix(truth, prediction, labels=(0, 1)).astype(np.float64)
        output[str(group_id)] = matrix / matrix.sum()
    return output


def _confusion_metric_batch(matrices: np.ndarray) -> dict[str, np.ndarray]:
    cm = np.asarray(matrices, dtype=np.float64)
    if cm.ndim != 3 or cm.shape[1:] != (2, 2):
        raise ValueError("group confusion batches must have shape (draws, 2, 2)")
    tn = cm[:, 0, 0]
    fp = cm[:, 0, 1]
    fn = cm[:, 1, 0]
    tp = cm[:, 1, 1]
    total = cm.sum(axis=(1, 2))
    sensitivity = np.divide(tp, tp + fn, out=np.full(len(cm), np.nan), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.full(len(cm), np.nan), where=(tn + fp) > 0)
    precision = np.divide(tp, tp + fp, out=np.full(len(cm), np.nan), where=(tp + fp) > 0)
    recall_0 = specificity
    recall_1 = sensitivity
    f1_0 = np.divide(2.0 * tn, 2.0 * tn + fp + fn, out=np.zeros(len(cm)), where=(2.0 * tn + fp + fn) > 0)
    f1_1 = np.divide(2.0 * tp, 2.0 * tp + fp + fn, out=np.zeros(len(cm)), where=(2.0 * tp + fp + fn) > 0)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = np.divide(tp * tn - fp * fn, np.sqrt(denominator), out=np.full(len(cm), np.nan), where=denominator > 0)
    observed = np.divide(tn + tp, total, out=np.full(len(cm), np.nan), where=total > 0)
    expected = np.divide(
        (tn + fp) * (tn + fn) + (fn + tp) * (fp + tp),
        total * total,
        out=np.full(len(cm), np.nan),
        where=total > 0,
    )
    kappa = np.divide(observed - expected, 1.0 - expected, out=np.full(len(cm), np.nan), where=~np.isclose(expected, 1.0))
    return {
        "accuracy": observed,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_class_1": precision,
        "macro_f1": 0.5 * (f1_0 + f1_1),
        "balanced_accuracy": 0.5 * (recall_0 + recall_1),
        "mcc": mcc,
        "cohen_kappa": kappa,
    }


def _paired_group_bootstrap(
    predictions: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Paired source-group CIs for augmented-vs-clean and noise-vs-clean deltas."""
    if int(replicates) < 1:
        raise ValueError("bootstrap replicates must be positive")
    groups = tuple(sorted(predictions.group_id.astype(str).unique()))
    if len(groups) < 2:
        raise ValueError("paired uncertainty needs at least two independent source groups")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(groups), size=(int(replicates), len(groups)))

    def group_matrices(rows: pd.DataFrame) -> np.ndarray:
        mapping = _group_normalized_confusion(rows, "pred_selected")
        if set(mapping) != set(groups):
            raise ValueError("paired robust predictions do not share identical independent groups")
        return np.stack([mapping[group] for group in groups])

    def compare(
        left_rows: pd.DataFrame,
        right_rows: pd.DataFrame,
        *,
        comparison_type: str,
        model_name: str,
        family: str,
        level: float,
        left_condition: str,
        right_condition: str,
    ) -> list[dict[str, Any]]:
        left = group_matrices(left_rows)
        right = group_matrices(right_rows)
        left_draws = left[draws].mean(axis=1)
        right_draws = right[draws].mean(axis=1)
        left_metric = _confusion_metric_batch(left_draws)
        right_metric = _confusion_metric_batch(right_draws)
        left_point = _confusion_metric_batch(left.mean(axis=0, keepdims=True))
        right_point = _confusion_metric_batch(right.mean(axis=0, keepdims=True))
        result: list[dict[str, Any]] = []
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "sensitivity", "specificity", "precision_class_1", "mcc", "cohen_kappa"):
            deltas = left_metric[metric] - right_metric[metric]
            finite = deltas[np.isfinite(deltas)]
            result.append(
                {
                    "comparison_type": comparison_type,
                    "model_name": model_name,
                    "family": family,
                    "requested_level": float(level),
                    "left_condition": left_condition,
                    "right_condition": right_condition,
                    "metric": metric,
                    "group_balanced_delta": float(left_point[metric][0] - right_point[metric][0]),
                    "ci_95_low": float(np.quantile(finite, 0.025)) if len(finite) else None,
                    "ci_95_high": float(np.quantile(finite, 0.975)) if len(finite) else None,
                    "valid_bootstrap_replicates": int(len(finite)),
                    "bootstrap_replicates": int(replicates),
                    "bootstrap_seed": int(seed),
                    "bootstrap_unit": "paired source-group resampling with equal group confusion mass",
                    "source_group_count": int(len(groups)),
                }
            )
        return result

    result_rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        subset = predictions.loc[
            predictions.model_name.astype(str).eq(model_name)
        ]
        for family in CORRUPTION_FAMILIES:
            for level in EVALUATION_LEVELS:
                clean_training = subset.loc[
                    subset.training_condition.eq("clean")
                    & subset.family.eq(family)
                    & np.isclose(subset.requested_level.astype(float), float(level))
                ]
                augmented_training = subset.loc[
                    subset.training_condition.eq("augmented")
                    & subset.family.eq(family)
                    & np.isclose(subset.requested_level.astype(float), float(level))
                ]
                if clean_training.empty or augmented_training.empty:
                    continue
                result_rows.extend(
                    compare(
                        augmented_training,
                        clean_training,
                        comparison_type="augmented_training_minus_clean_training",
                        model_name=model_name,
                        family=family,
                        level=level,
                        left_condition="augmented_training",
                        right_condition="clean_training",
                    )
                )
                for training_condition in ("clean", "augmented"):
                    noisy = subset.loc[
                        subset.training_condition.eq(training_condition)
                        & subset.family.eq(family)
                        & np.isclose(subset.requested_level.astype(float), float(level))
                    ]
                    clean_input = subset.loc[
                        subset.training_condition.eq(training_condition)
                        & subset.family.eq("clean")
                        & np.isclose(subset.requested_level.astype(float), 0.0)
                    ]
                    if noisy.empty or clean_input.empty:
                        continue
                    result_rows.extend(
                        compare(
                            noisy,
                            clean_input,
                            comparison_type=f"{training_condition}_noise_minus_clean_input",
                            model_name=model_name,
                            family=family,
                            level=level,
                            left_condition=f"{family}_{level:g}",
                            right_condition="clean_0",
                        )
                    )
    return pd.DataFrame(result_rows)


def _load_development_inputs(
    run_dir: Path,
    *,
    tabular_subdir: str,
    neural_subdir: str,
    guardrail_subdir: str,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    private_dir = run_dir / "private"
    tabular_dir = private_dir / tabular_subdir
    neural_dir = private_dir / neural_subdir
    guardrail_dir = private_dir / guardrail_subdir
    required = (
        private_dir / "fatigue_proxy_config.toml",
        private_dir / "split_manifest.json",
        private_dir / "features_labels" / "pseudo_labels_alternatives.parquet",
        private_dir / "features_labels" / "features.parquet",
        private_dir / "acquisition" / "acquisition_audit.csv",
        private_dir / "acquisition" / "acquisition_stage_manifest.json",
        private_dir / "neural_data" / "signal_window_bank_manifest.json",
        private_dir / "neural_data" / "signals_500ms_development.npz",
        private_dir / "neural_data" / "signal_window_manifest.parquet",
        tabular_dir / "fold_manifest.json",
        tabular_dir / "nested_oof_predictions.parquet",
        tabular_dir / "supervised_threshold_stage_manifest.json",
        neural_dir / "neural_supervised_stage_manifest.json",
        neural_dir / "checkpoint_manifest.json",
        neural_dir / "neural_lowfreq_oof_predictions.parquet",
        neural_dir / "inner_oof_selection_predictions_with_model.parquet",
        neural_dir / "active_lowfreq_signal_window_manifest.parquet",
        guardrail_dir / "guardrail_stage_manifest.json",
        guardrail_dir / "ood_detector_receipts.json",
        guardrail_dir / "abstention_threshold_selection.csv",
        guardrail_dir / "neural_guardrail_oof_predictions.parquet",
        guardrail_dir / "neural_abstention_oof_predictions.parquet",
    )
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError("required frozen development inputs are missing: " + "; ".join(absent))
    if Path(tabular_subdir).name != tabular_subdir or Path(neural_subdir).name != neural_subdir or Path(guardrail_subdir).name != guardrail_subdir:
        raise ValueError("parent stage names must be single relative directory names")

    threshold_receipt = verify_lowfreq_threshold_parent_stage(tabular_dir)
    if threshold_receipt.get("external_rar_opened") is not False or int(threshold_receipt.get("internal_test_signal_values_parsed", -1)) != 0:
        raise ValueError("low-frequency fold parent does not prove reserved input isolation")
    if threshold_receipt.get("inputs_sha256"):
        _verify_private_hash_map(threshold_receipt["inputs_sha256"], private_dir, "low-frequency threshold parent private input")
    elif threshold_receipt.get("input_sha256"):
        _verify_private_hash_map(threshold_receipt["input_sha256"], private_dir, "low-frequency threshold parent private input")
    fold_plan = _load_fold_plan(tabular_dir / "fold_manifest.json")
    if len(fold_plan.outer_folds) != 4 or any(len(fold_plan.inner_folds[fold.fold_id]) != 3 for fold in fold_plan.outer_folds):
        raise ValueError("noise development stage requires the supported historical 4x3 group plan")

    neural_manifest, neural_predictions, inner_predictions, _ = _verify_neural_stage(neural_dir)
    if (
        neural_manifest.get("external_rar_opened") is not False
        or int(neural_manifest.get("internal_test_signal_values_parsed", -1)) != 0
        or int(neural_manifest.get("reserved_test_member_payloads_read", -1)) != 0
    ):
        raise ValueError("clean neural parent does not prove that reserved test data remained closed")
    if neural_manifest.get("input_sha256"):
        _verify_private_hash_map(neural_manifest["input_sha256"], private_dir, "clean neural parent private input")
    guardrail_manifest = json.loads((guardrail_dir / "guardrail_stage_manifest.json").read_text(encoding="utf-8"))
    if (
        guardrail_manifest.get("status") != "completed"
        or guardrail_manifest.get("external_rar_opened") is not False
        or int(guardrail_manifest.get("internal_test_signal_values_parsed", -1)) != 0
        or guardrail_manifest.get("reserved_test_member_payloads_read", 0) not in (0, None)
    ):
        raise ValueError("guardrail parent does not prove development-only OOD and abstention selection")
    guardrail_outputs = guardrail_manifest.get("output_sha256", {})
    if guardrail_outputs:
        _verify_hash_map(guardrail_outputs, "guardrail parent output")
    else:
        raise ValueError("guardrail parent is missing its output checksum receipt")
    if guardrail_manifest.get("input_sha256"):
        _verify_private_hash_map(guardrail_manifest["input_sha256"], private_dir, "guardrail parent private input")

    acquisition_dir = private_dir / "acquisition"
    acquisition_manifest = json.loads((acquisition_dir / "acquisition_stage_manifest.json").read_text(encoding="utf-8"))
    if (
        acquisition_manifest.get("stage") != "development_acquisition_qc_preprocessing_activity_windows"
        or acquisition_manifest.get("external_archive_opened") is not False
        or int(acquisition_manifest.get("internal_test_numeric_values_parsed", -1)) != 0
    ):
        raise ValueError("acquisition parent does not prove development-only source processing")
    if acquisition_manifest.get("config_sha256") != _sha256_file(private_dir / "fatigue_proxy_config.toml"):
        raise ValueError("acquisition and current frozen TOML config checksums differ")
    acquisition_hashes = acquisition_manifest.get("output_sha256_excluding_this_manifest", {})
    if not acquisition_hashes:
        raise ValueError("acquisition parent is missing its output hash receipt")
    for relative, expected in acquisition_hashes.items():
        path = (acquisition_dir / str(relative)).resolve()
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise ValueError(f"acquisition parent output checksum mismatch: {relative}")

    bank_dir = private_dir / "neural_data"
    bank_manifest_path = bank_dir / "signal_window_bank_manifest.json"
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    if (
        bank_manifest.get("status") != "completed"
        or bank_manifest.get("external_rar_opened") is not False
        or int(bank_manifest.get("internal_test_signal_values_parsed", -1)) != 0
        or int(bank_manifest.get("reserved_test_member_payloads_read", -1)) != 0
    ):
        raise ValueError("development signal-bank receipt does not prove reserved-source isolation")
    _verify_hash_map(bank_manifest.get("output_sha256", {}), "development signal-bank output")

    config = tomllib.loads((private_dir / "fatigue_proxy_config.toml").read_text(encoding="utf-8"))
    split_manifest = json.loads((private_dir / "split_manifest.json").read_text(encoding="utf-8"))
    development_groups = {str(value) for value in split_manifest.get("development_group_ids", ())}
    test_groups = {str(value) for value in split_manifest.get("test_group_ids", ())}
    if not development_groups or development_groups & test_groups:
        raise ValueError("split receipt has missing development groups or overlap with reserved test groups")

    bank_path = bank_dir / "signals_500ms_development.npz"
    metadata_path = bank_dir / "signal_window_manifest.parquet"
    with np.load(bank_path, allow_pickle=False) as bank:
        signals = np.asarray(bank["signals"], dtype=np.float32)
        bank_keys = bank["window_keys"].astype(str)
    raw_metadata = pd.read_parquet(metadata_path)
    if signals.ndim != 2 or signals.shape[1] != 500 or len(signals) != len(raw_metadata):
        raise ValueError("frozen development signal matrix has unexpected dimensions")
    if not np.array_equal(bank_keys, raw_metadata.window_key.astype(str).to_numpy()):
        raise ValueError("signal matrix and manifest row order differ")
    if raw_metadata.window_key.astype(str).duplicated().any():
        raise ValueError("development signal manifest contains duplicate window keys")
    if not set(raw_metadata.group_id.astype(str)).issubset(development_groups):
        raise ValueError("signal manifest contains a non-development group")

    labels_path = private_dir / "features_labels" / "pseudo_labels_alternatives.parquet"
    labels = pd.read_parquet(labels_path)
    metadata = join_primary_lowfreq_labels(raw_metadata, labels)
    if len(metadata) != len(raw_metadata) or metadata.window_key.astype(str).duplicated().any():
        raise ValueError("low-frequency label join changed the complete raw-signal window universe")
    bank_positions = {str(key): position for position, key in enumerate(bank_keys)}
    metadata["bank_position"] = metadata.window_key.astype(str).map(bank_positions).astype(int)
    if not metadata.bank_position.eq(np.arange(len(metadata), dtype=int)).all():
        raise ValueError("frozen labels do not preserve the raw signal bank's exact window order")
    active = metadata.loc[metadata.class_id.isin((0, 1))].copy()
    active["group_id"] = active.group_id.astype(str)
    active["recording_id"] = active.recording_id.astype(str)
    active["class_id"] = active.class_id.astype(int)
    if len(active) != 5977 or active.class_id.value_counts().to_dict() != {0: 5074, 1: 903}:
        raise ValueError("active target is not the verified 5,977-window low-frequency pseudo-label cohort")
    if active.group_id.nunique() != 12:
        raise ValueError("active target does not cover the 12 development groups")
    fold_support = audit_group_fold_support(active, fold_plan, min_groups_per_class=2, min_trials_per_class=2)
    if not fold_support.passed.all():
        raise ValueError("a nested train or validation partition lacks the frozen minimum class support")
    for class_id in (0, 1):
        if active.loc[active.class_id.eq(class_id), "group_id"].nunique() < 4:
            raise ValueError("an operational class lacks source-group support for 4 outer folds")

    feature_table = pd.read_parquet(private_dir / "features_labels" / "features.parquet")
    primary_features = feature_table.loc[
        np.isclose(feature_table.duration_s.to_numpy(dtype=float), float(config["run"]["primary_window_s"]))
        & feature_table.feature_valid.fillna(False).astype(bool),
        ["window_key", "group_id", "recording_id", *OOD_FEATURES],
    ].copy()
    if primary_features.window_key.astype(str).duplicated().any():
        raise ValueError("clean OOD feature table has duplicate primary-window keys")
    primary_features["window_key"] = primary_features.window_key.astype(str)
    primary_features["group_id"] = primary_features.group_id.astype(str)
    primary_features["recording_id"] = primary_features.recording_id.astype(str)
    clean_features = metadata.merge(
        primary_features,
        on="window_key",
        how="inner",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )
    for identity in ("group_id", "recording_id"):
        feature_identity = f"{identity}_feature"
        if feature_identity in clean_features:
            if not clean_features[identity].astype(str).eq(clean_features[feature_identity].astype(str)).all():
                raise ValueError(f"clean features disagree with signal-bank {identity}")
            clean_features = clean_features.drop(columns=[feature_identity])
    if not set(active.window_key.astype(str)).issubset(set(clean_features.window_key.astype(str))):
        raise ValueError("clean morphology features do not cover every assigned proxy window")

    acquisition = pd.read_csv(private_dir / "acquisition" / "acquisition_audit.csv")
    acquisition["group_id"] = acquisition.group_id.astype(str)
    acquisition["recording_id"] = acquisition.recording_id.astype(str)
    if acquisition.duplicated(["group_id", "recording_id"]).any():
        raise ValueError("acquisition audit contains duplicate recording identities")
    notch_lookup = {
        (str(row.group_id), str(row.recording_id)): (
            None if pd.isna(row.notch_frequency_hz) else float(row.notch_frequency_hz)
        )
        for row in acquisition.itertuples(index=False)
    }
    notch = np.asarray(
        [
            np.nan if notch_lookup.get((str(group), str(recording))) is None
            else float(notch_lookup[(str(group), str(recording))])
            for group, recording in zip(metadata.group_id, metadata.recording_id, strict=True)
        ],
        dtype=np.float64,
    )

    checkpoint_manifest = json.loads((neural_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(checkpoint_manifest, list) or len(checkpoint_manifest) != len(MODEL_NAMES) * len(fold_plan.outer_folds):
        raise ValueError("clean neural checkpoint manifest does not enumerate all architecture-fold fits")
    checkpoints: dict[tuple[str, int], Path] = {}
    for item in checkpoint_manifest:
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        model_name = str(item.get("model_name"))
        fold_id = int(item.get("outer_fold_id", -1))
        if model_name not in MODEL_NAMES or not path.is_file() or _sha256_file(path) != str(item.get("sha256", "")):
            raise ValueError("clean checkpoint path or checksum differs from its receipt")
        if (model_name, fold_id) in checkpoints:
            raise ValueError("clean checkpoint manifest contains duplicate architecture-fold entries")
        checkpoints[(model_name, fold_id)] = path

    outer_clean = neural_predictions.loc[
        neural_predictions.feature_view.astype(str).eq("raw_signal")
        & neural_predictions.threshold_strategy.astype(str).eq("inner_oof_selected"),
    ].copy()
    fixed_clean = neural_predictions.loc[
        neural_predictions.feature_view.astype(str).eq("raw_signal")
        & neural_predictions.threshold_strategy.astype(str).eq("fixed_0_5"),
        ["window_key", "model_name", "fold_id", "predicted_class_id", "decision_threshold"],
    ].copy()
    if outer_clean.duplicated(["model_name", "fold_id", "window_key"]).any():
        raise ValueError("clean outer OOF table has duplicate model-fold-window records")
    outer_clean = outer_clean.merge(
        fixed_clean,
        on=["window_key", "model_name", "fold_id"],
        suffixes=("_selected", "_fixed"),
        validate="one_to_one",
    )
    outer_clean = outer_clean.rename(
        columns={
            "predicted_class_id_selected": "pred_selected",
            "predicted_class_id_fixed": "pred_fixed",
            "decision_threshold_selected": "selected_threshold",
        }
    )
    guardrail_predictions = pd.read_parquet(guardrail_dir / "neural_guardrail_oof_predictions.parquet")
    guardrail_abstention = pd.read_parquet(guardrail_dir / "neural_abstention_oof_predictions.parquet")
    confidence_cut = pd.read_csv(guardrail_dir / "abstention_threshold_selection.csv")
    guardrail_selected = guardrail_predictions.loc[
        guardrail_predictions.threshold_strategy.astype(str).eq("inner_oof_selected")
        if "threshold_strategy" in guardrail_predictions
        else guardrail_predictions.index == guardrail_predictions.index
    ].copy()
    guardrail_receipts = json.loads((guardrail_dir / "ood_detector_receipts.json").read_text(encoding="utf-8"))
    if not isinstance(guardrail_receipts, list) or len(guardrail_receipts) != len(fold_plan.outer_folds):
        raise ValueError("guardrail OOD receipt does not contain one outer-training detector per fold")
    ood_receipts = {int(item["outer_fold_id"]): item for item in guardrail_receipts}
    if set(ood_receipts) != {int(item.fold_id) for item in fold_plan.outer_folds}:
        raise ValueError("guardrail OOD receipts do not match the frozen outer folds")
    clean_policy = confidence_cut[["model_name", "outer_fold_id", "selected_confidence_threshold"]].copy()
    if clean_policy.duplicated(["model_name", "outer_fold_id"]).any():
        raise ValueError("clean confidence policy contains duplicate model-fold thresholds")
    if guardrail_abstention.duplicated(["model_name", "fold_id", "window_key"]).any():
        raise ValueError("clean abstention output contains duplicate architecture-fold-window records")
    clean_ood = guardrail_selected[
        ["window_key", "model_name", "fold_id", "ood_score", "ood_threshold", "ood_flag"]
    ].drop_duplicates(["window_key", "model_name", "fold_id"])
    clean_oof = outer_clean.merge(
        clean_ood,
        on=["window_key", "model_name", "fold_id"],
        how="left",
        validate="one_to_one",
    )
    if clean_oof.ood_flag.isna().any():
        raise ValueError("clean neural OOF and frozen OOD flags do not have exact paired coverage")
    clean_oof = clean_oof.merge(
        clean_policy.rename(columns={"outer_fold_id": "fold_id"}),
        on=["model_name", "fold_id"],
        how="left",
        validate="many_to_one",
    )
    if clean_oof.selected_confidence_threshold.isna().any():
        raise ValueError("clean neural OOF lacks an inner-selected confidence threshold")
    clean_oof = clean_oof.rename(
        columns={
            "probability_class_1": "probability_class_1",
            "true_class_id": "true_class_id",
            "group_id": "group_id",
            "recording_id": "recording_id",
        }
    )
    active_quality = active[["window_key", "group_id", "recording_id", "class_id", "eligible_for_label", "bank_position"]].copy()
    clean_oof = clean_oof.merge(
        active_quality,
        on=["window_key", "group_id", "recording_id"],
        how="left",
        validate="many_to_one",
    )
    if (
        clean_oof.class_id.isna().any()
        or not clean_oof.class_id.astype(int).eq(clean_oof.true_class_id.astype(int)).all()
        or clean_oof.eligible_for_label.isna().any()
    ):
        raise ValueError("clean neural predictions disagree with the frozen class or QC eligibility manifest")
    clean_oof["training_condition"] = "clean"
    clean_oof["family"] = "clean"
    clean_oof["requested_level"] = 0.0
    clean_oof["achieved_rms_ratio"] = 0.0
    clean_oof["stress_test"] = False
    clean_oof["raw_logit"] = clean_oof.raw_logit.astype(float)
    clean_oof["probability_class_1"] = clean_oof.probability_class_1.astype(float)
    clean_oof["confidence"] = clean_oof[["probability_class_1"]].assign(
        probability_class_0=1.0 - clean_oof.probability_class_1
    ).max(axis=1)
    clean_oof["ood_flag"] = clean_oof.ood_flag.astype(bool)
    clean_oof["pred_fixed"] = clean_oof.pred_fixed.astype(int)
    clean_oof["pred_selected"] = clean_oof.pred_selected.astype(int)
    if clean_oof.duplicated(["model_name", "fold_id", "window_key"]).any():
        raise ValueError("clean OOF reference contains duplicate matched key records")
    if len(clean_oof) != int(active.shape[0]) * len(MODEL_NAMES):
        raise ValueError("clean OOF reference does not cover each model and assigned window exactly once")

    temporal_predictions = pd.read_parquet(tabular_dir / "nested_oof_predictions.parquet")
    active["fold_id"] = active.group_id.astype(str).map(
        {
            str(group_id): int(outer.fold_id)
            for outer in fold_plan.outer_folds
            for group_id in outer.validation_group_ids
        }
    ).astype(int)
    expected_active_window_keys = set(active.window_key.astype(str))
    temporal = temporal_predictions.loc[
        temporal_predictions.feature_view.astype(str).eq("relative_time_only")
        & temporal_predictions.model_name.astype(str).eq("logistic_regression")
        & temporal_predictions.threshold_strategy.astype(str).isin(("fixed_0_5", "inner_oof_selected"))
        & temporal_predictions.true_class_id.isin((0, 1)),
    ].copy()
    fixed_temporal = temporal.loc[
        temporal.threshold_strategy.astype(str).eq("fixed_0_5"),
        ["window_key", "group_id", "recording_id", "fold_id", "true_class_id", "predicted_class_id"],
    ].rename(columns={"predicted_class_id": "pred_fixed"})
    selected_temporal = temporal.loc[
        temporal.threshold_strategy.astype(str).eq("inner_oof_selected"),
        ["window_key", "group_id", "recording_id", "fold_id", "true_class_id", "predicted_class_id", "decision_threshold", "probability_class_1"],
    ].rename(
        columns={
            "predicted_class_id": "pred_selected",
            "decision_threshold": "selected_threshold",
        }
    )
    if fixed_temporal.duplicated("window_key").any() or selected_temporal.duplicated("window_key").any():
        raise ValueError("temporal baseline contains duplicate fixed/selected OOF windows")
    temporal_baseline = selected_temporal.merge(
        fixed_temporal,
        on=["window_key", "group_id", "recording_id", "fold_id", "true_class_id"],
        how="inner",
        validate="one_to_one",
    )
    if set(temporal_baseline.window_key.astype(str)) != expected_active_window_keys or len(temporal_baseline) != len(active):
        raise ValueError("temporal baseline does not exactly cover the same assigned development windows")
    temporal_baseline = temporal_baseline.merge(
        active_quality,
        on=["window_key", "group_id", "recording_id"],
        how="left",
        validate="one_to_one",
    )
    if (
        temporal_baseline.class_id.isna().any()
        or not temporal_baseline.true_class_id.astype(int).eq(temporal_baseline.class_id.astype(int)).all()
        or not temporal_baseline.fold_id.astype(int).eq(
            temporal_baseline.window_key.map(active.set_index("window_key").fold_id).astype(int)
        ).all()
    ):
        raise ValueError("temporal baseline labels or folds disagree with the frozen active target")
    temporal_baseline["model_name"] = "time_only_logistic"
    temporal_baseline["raw_logit"] = np.nan
    temporal_baseline["confidence"] = np.maximum(
        temporal_baseline.probability_class_1.to_numpy(dtype=float),
        1.0 - temporal_baseline.probability_class_1.to_numpy(dtype=float),
    )
    temporal_baseline["selected_confidence_threshold"] = 0.5
    temporal_baseline["training_condition"] = "temporal_baseline"
    temporal_baseline["family"] = "clean"
    temporal_baseline["requested_level"] = 0.0
    temporal_baseline["achieved_rms_ratio"] = 0.0
    temporal_baseline["corruption_seed"] = np.uint32(0)
    temporal_baseline["stress_test"] = False

    return {
        "run_dir": run_dir,
        "private_dir": private_dir,
        "tabular_dir": tabular_dir,
        "neural_dir": neural_dir,
        "guardrail_dir": guardrail_dir,
        "config": config,
        "split_manifest": split_manifest,
        "development_groups": development_groups,
        "fold_plan": fold_plan,
        "signals": signals,
        "metadata": metadata,
        "active": active,
        "clean_features": clean_features,
        "notch_by_window": notch,
        "checkpoints": checkpoints,
        "clean_oof": clean_oof,
        "temporal_baseline": temporal_baseline,
        "neural_manifest": neural_manifest,
        "guardrail_manifest": guardrail_manifest,
        "acquisition_manifest": acquisition_manifest,
        "ood_receipts": ood_receipts,
        "clean_policy": clean_policy,
        "inner_clean": inner_predictions,
        "input_paths": list(required) + [bank_manifest_path],
    }


def _metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    identity_columns = ("training_condition", "model_name", "family", "requested_level")
    for identity, rows in predictions.groupby(list(identity_columns), sort=True, observed=True):
        training_condition, model_name, family, level = identity
        for strategy, prediction_column in (
            ("fixed_0_5", "pred_fixed"),
            ("inner_oof_selected", "pred_selected"),
        ):
            truth = rows.true_class_id.to_numpy(dtype=int)
            predicted = rows[prediction_column].to_numpy(dtype=int)
            probability = rows.probability_class_1.to_numpy(dtype=float)
            record = compute_metrics(truth, probability, predicted, np.zeros(len(rows), dtype=bool)).to_dict()
            matrices: list[np.ndarray] = []
            by_group: dict[str, pd.DataFrame] = {}
            for group_id, group in rows.groupby("group_id", sort=True):
                group_truth = group.true_class_id.to_numpy(dtype=int)
                group_prediction = group[prediction_column].to_numpy(dtype=int)
                if set(np.unique(group_truth).tolist()) != {0, 1}:
                    raise ValueError(f"{group_id} lacks both pseudo-classes in a matched OOF condition")
                matrix = confusion_matrix(group_truth, group_prediction, labels=(0, 1)).astype(np.float64)
                matrices.append(matrix / matrix.sum())
                by_group[str(group_id)] = group
                group_record = compute_metrics(
                    group_truth,
                    group.probability_class_1.to_numpy(dtype=float),
                    group_prediction,
                    np.zeros(len(group), dtype=bool),
                ).to_dict()
                group_rows.append(
                    {
                        "training_condition": training_condition,
                        "model_name": model_name,
                        "family": family,
                        "requested_level": float(level),
                        "stress_test": bool(float(level) > 0.20),
                        "threshold_strategy": strategy,
                        "group_id": str(group_id),
                        "outer_fold_id": int(group.fold_id.iloc[0]),
                        "n_oof_windows_descriptive_only": int(len(group)),
                        "class_0_support": int(np.sum(group_truth == 0)),
                        "class_1_support": int(np.sum(group_truth == 1)),
                        "accuracy": group_record["accuracy"],
                        "sensitivity": group_record["sensitivity"],
                        "specificity": group_record["specificity"],
                        "precision_class_1": group_record["per_class"]["1"]["precision"],
                        "recall_class_1": group_record["per_class"]["1"]["recall"],
                        "macro_f1": group_record["macro_f1"],
                        "balanced_accuracy": group_record["balanced_accuracy"],
                        "mcc": group_record["matthews_correlation_coefficient"],
                        "cohen_kappa": group_record["cohen_kappa"],
                        "pr_auc_average_precision": group_record["pr_auc_average_precision"],
                        "roc_auc": group_record["roc_auc"],
                        "brier_score": group_record["calibration"]["brier_score"],
                        "ece_10_bins": group_record["calibration"]["expected_calibration_error_10_bins"],
                        "confusion_matrix": json.dumps(group_record["confusion_matrix"], separators=(",", ":")),
                    }
                )
            group_matrix = np.mean(np.stack(matrices), axis=0)
            group_weighted = _weighted_confusion_metrics(group_matrix)
            thresholds = (
                {str(int(fold)): sorted(set(float(value) for value in part.selected_threshold)) for fold, part in rows.groupby("fold_id", sort=True)}
                if "selected_threshold" in rows
                else {}
            )
            pooled_rows.append(
                {
                    "training_condition": training_condition,
                    "model_name": model_name,
                    "family": family,
                    "requested_level": float(level),
                    "stress_test": bool(float(level) > 0.20),
                    "threshold_strategy": strategy,
                    "decision_thresholds_by_outer_fold": json.dumps(thresholds, sort_keys=True, separators=(",", ":")),
                    "n_oof_windows_descriptive_only": int(len(rows)),
                    "source_group_count": int(rows.group_id.nunique()),
                    "accuracy": record["accuracy"],
                    "sensitivity": record["sensitivity"],
                    "specificity": record["specificity"],
                    "precision_class_1": record["per_class"]["1"]["precision"],
                    "recall_class_1": record["per_class"]["1"]["recall"],
                    "macro_f1": record["macro_f1"],
                    "balanced_accuracy": record["balanced_accuracy"],
                    "mcc": record["matthews_correlation_coefficient"],
                    "cohen_kappa": record["cohen_kappa"],
                    "pr_auc_average_precision": record["pr_auc_average_precision"],
                    "roc_auc": record["roc_auc"],
                    "brier_score": record["calibration"]["brier_score"],
                    "ece_10_bins": record["calibration"]["expected_calibration_error_10_bins"],
                    "confusion_matrix": json.dumps(record["confusion_matrix"], separators=(",", ":")),
                    "group_balanced_accuracy": group_weighted["accuracy"],
                    "group_balanced_sensitivity": group_weighted["sensitivity"],
                    "group_balanced_specificity": group_weighted["specificity"],
                    "group_balanced_precision_class_1": group_weighted["precision_class_1"],
                    "group_balanced_macro_f1": group_weighted["macro_f1"],
                    "group_balanced_balanced_accuracy": group_weighted["balanced_accuracy"],
                    "group_balanced_mcc": group_weighted["mcc"],
                    "group_balanced_cohen_kappa": group_weighted["cohen_kappa"],
                    "group_balanced_confusion_matrix_normalized": json.dumps(group_matrix.tolist(), separators=(",", ":")),
                    "independent_unit": "source folder group; participant identity unverified",
                    "interpretation": "agreement with the fixed weak-supervision pseudo-label rule; overlap windows are not independent",
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(group_rows)


def _paired_group_bootstrap_from_confusions(
    confusions: dict[tuple[str, str, str, float], dict[str, np.ndarray]],
    *,
    groups: Sequence[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    group_order = tuple(sorted(str(value) for value in groups))
    if len(group_order) < 2:
        raise ValueError("paired group bootstrap requires at least two source groups")
    rng = np.random.default_rng(int(seed))
    choices = rng.integers(0, len(group_order), size=(int(replicates), len(group_order)))
    result: list[dict[str, Any]] = []

    def add(left_key: tuple[str, str, str, float], right_key: tuple[str, str, str, float], kind: str) -> None:
        left_map = confusions.get(left_key)
        right_map = confusions.get(right_key)
        if left_map is None or right_map is None:
            return
        if set(left_map) != set(group_order) or set(right_map) != set(group_order):
            raise ValueError("paired noise predictions do not share the same source groups")
        left = np.stack([left_map[group] for group in group_order])
        right = np.stack([right_map[group] for group in group_order])
        left_draws = _confusion_metric_batch(left[choices].mean(axis=1))
        right_draws = _confusion_metric_batch(right[choices].mean(axis=1))
        left_point = _confusion_metric_batch(left.mean(axis=0, keepdims=True))
        right_point = _confusion_metric_batch(right.mean(axis=0, keepdims=True))
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "sensitivity", "specificity", "precision_class_1", "mcc", "cohen_kappa"):
            difference = left_draws[metric] - right_draws[metric]
            finite = difference[np.isfinite(difference)]
            result.append(
                {
                    "comparison_type": kind,
                    "model_name": left_key[1],
                    "family": left_key[2],
                    "requested_level": float(left_key[3]),
                    "left_training_condition": left_key[0],
                    "right_training_condition": right_key[0],
                    "left_input_condition": f"{left_key[2]}_{left_key[3]:g}",
                    "right_input_condition": f"{right_key[2]}_{right_key[3]:g}",
                    "metric": metric,
                    "group_balanced_delta": float(left_point[metric][0] - right_point[metric][0]),
                    "ci_95_low": float(np.quantile(finite, 0.025)) if len(finite) else None,
                    "ci_95_high": float(np.quantile(finite, 0.975)) if len(finite) else None,
                    "valid_bootstrap_replicates": int(len(finite)),
                    "bootstrap_replicates": int(replicates),
                    "bootstrap_seed": int(seed),
                    "bootstrap_unit": "paired source-group resampling with equal group confusion mass",
                    "source_group_count": int(len(group_order)),
                }
            )

    for model_name in MODEL_NAMES:
        add(
            ("augmented", model_name, "clean", 0.0),
            ("clean", model_name, "clean", 0.0),
            "augmented_training_minus_clean_training_on_clean_input",
        )
        for family in CORRUPTION_FAMILIES:
            for level in EVALUATION_LEVELS:
                clean_condition = ("clean", model_name, family, float(level))
                augmented_condition = ("augmented", model_name, family, float(level))
                add(augmented_condition, clean_condition, "augmented_training_minus_clean_training")
                clean_input_level = ("clean", model_name, "clean", 0.0)
                augmented_input_level = ("augmented", model_name, "clean", 0.0)
                add(clean_condition, clean_input_level, "clean_training_noisy_input_minus_clean_input")
                add(augmented_condition, augmented_input_level, "augmented_training_noisy_input_minus_clean_input")
                temporal_key = ("temporal_baseline", "time_only_logistic", family, float(level))
                add(clean_condition, temporal_key, "clean_neural_minus_temporal_baseline")
                add(augmented_condition, temporal_key, "augmented_neural_minus_temporal_baseline")
    return pd.DataFrame(result)


def _build_matched_corruption(
    signals: np.ndarray,
    metadata: pd.DataFrame,
    family: str,
    level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    corrupted = np.empty_like(signals, dtype=np.float32)
    achieved = np.empty(len(signals), dtype=np.float64)
    seeds = np.empty(len(signals), dtype=np.uint32)
    receipt_rows: list[dict[str, Any]] = []
    for position, row in enumerate(metadata.itertuples(index=False)):
        seed = stable_seed("emg-matched-evaluation-noise-v1", str(row.window_key), family, float(level))
        noisy, receipt = corrupt_window(
            signals[position],
            family,
            float(level),
            seed,
            sampling_rate_hz=float(row.sampling_rate_hz),
        )
        corrupted[position] = noisy
        achieved[position] = float(receipt["achieved_rms_ratio"])
        seeds[position] = int(seed)
        receipt_rows.append(
            {
                "window_key": str(row.window_key),
                "group_id": str(row.group_id),
                "recording_id": str(row.recording_id),
                "family": str(family),
                "requested_level": float(level),
                "stress_test": bool(float(level) > 0.20),
                "corruption_seed": int(seed),
                "clean_rms": float(receipt["clean_rms"]),
                "delta_rms": float(receipt["delta_rms"]),
                "achieved_rms_ratio": float(receipt["achieved_rms_ratio"]),
                "sampling_rate_hz": float(receipt["sampling_rate_hz"]),
                "level_semantics": str(receipt["level_semantics"]),
                "corruption_parameters_json": json.dumps(receipt["parameters"], sort_keys=True, separators=(",", ":")),
            }
        )
    return corrupted, achieved, seeds, pd.DataFrame(receipt_rows)


def _score_probabilities(logits: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    coefficient = float(metadata["outer_calibrator_coefficient"])
    intercept = float(metadata["outer_calibrator_intercept"])
    return _sigmoid(np.asarray(logits, dtype=np.float64) * coefficient + intercept)


def _canonical_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "window_key", "group_id", "recording_id", "fold_id", "model_name", "true_class_id", "raw_logit",
        "probability_class_1", "pred_fixed", "pred_selected", "selected_threshold", "confidence",
        "selected_confidence_threshold", "ood_score", "ood_threshold", "ood_flag", "training_condition",
        "family", "requested_level", "achieved_rms_ratio", "corruption_seed", "stress_test", "eligible_for_label",
    )
    output = frame.copy()
    missing = set(columns) - set(output.columns)
    if missing:
        raise ValueError(f"robustness prediction frame is missing fields: {sorted(missing)}")
    output = output.loc[:, columns].copy()
    for name in ("window_key", "group_id", "recording_id", "model_name", "training_condition", "family"):
        output[name] = output[name].astype(str)
    for name in ("fold_id", "true_class_id", "pred_fixed", "pred_selected"):
        output[name] = output[name].astype(np.int64)
    for name in (
        "raw_logit", "probability_class_1", "selected_threshold", "confidence", "selected_confidence_threshold",
        "ood_score", "ood_threshold", "requested_level", "achieved_rms_ratio",
    ):
        output[name] = output[name].astype(np.float64)
    output["ood_flag"] = output.ood_flag.astype(bool)
    output["corruption_seed"] = output.corruption_seed.astype(np.uint32)
    output["stress_test"] = output.stress_test.astype(bool)
    output["eligible_for_label"] = output.eligible_for_label.astype(bool)
    return output


def _ood_rate_tables(
    rows: pd.DataFrame,
    *,
    family: str,
    level: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flagged = rows.ood_flag.to_numpy(dtype=bool)
    truth = rows.class_id.to_numpy(dtype=int)
    summary = {
        "family": family,
        "requested_level": float(level),
        "stress_test": bool(float(level) > 0.20),
        "window_count_descriptive_only": int(len(rows)),
        "source_group_count": int(rows.group_id.nunique()),
        "ood_flag_rate": float(np.mean(flagged)) if len(flagged) else None,
        "ood_flag_rate_class_0": float(np.mean(flagged[truth == 0])) if np.any(truth == 0) else None,
        "ood_flag_rate_class_1": float(np.mean(flagged[truth == 1])) if np.any(truth == 1) else None,
        "mean_ood_score": float(np.mean(rows.ood_score.to_numpy(dtype=float))) if len(rows) else None,
        "mean_ood_threshold": float(np.mean(rows.ood_threshold.to_numpy(dtype=float))) if len(rows) else None,
        "detector_ground_truth_available": False,
        "interpretation": "distribution-shift flag rate only; no OOD labels or detector accuracy are available",
    }
    group_rows: list[dict[str, Any]] = []
    for (group_id, class_id), group in rows.groupby(["group_id", "class_id"], sort=True):
        group_rows.append(
            {
                "family": family,
                "requested_level": float(level),
                "stress_test": bool(float(level) > 0.20),
                "group_id": str(group_id),
                "class_id": int(class_id),
                "window_count_descriptive_only": int(len(group)),
                "ood_flag_rate": float(group.ood_flag.astype(bool).mean()),
                "mean_ood_score": float(group.ood_score.astype(float).mean()),
                "mean_ood_threshold": float(group.ood_threshold.astype(float).mean()),
            }
        )
    return summary, group_rows


def _synthetic_ood_detection_metrics(
    clean_ood: pd.DataFrame,
    noisy_ood: pd.DataFrame,
    *,
    family: str,
    level: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean = clean_ood[["window_key", "group_id", "ood_score"]].copy()
    noisy = noisy_ood[["window_key", "group_id", "ood_score"]].copy()
    clean["window_key"] = clean.window_key.astype(str)
    noisy["window_key"] = noisy.window_key.astype(str)
    clean["group_id"] = clean.group_id.astype(str)
    noisy["group_id"] = noisy.group_id.astype(str)
    if clean.window_key.duplicated().any() or noisy.window_key.duplicated().any():
        raise ValueError("synthetic OOD comparison contains duplicate matched window keys")
    if set(clean.window_key) != set(noisy.window_key) or not clean.set_index("window_key").group_id.sort_index().equals(
        noisy.set_index("window_key").group_id.sort_index()
    ):
        raise ValueError("synthetic OOD clean/noisy conditions are not paired on the same windows and groups")
    scores = np.concatenate((clean.ood_score.to_numpy(dtype=float), noisy.ood_score.to_numpy(dtype=float)))
    truth = np.concatenate((np.zeros(len(clean), dtype=int), np.ones(len(noisy), dtype=int)))
    if not np.isfinite(scores).all():
        raise ValueError("synthetic OOD scores contain non-finite values")
    fpr, tpr, _ = roc_curve(truth, scores, drop_intermediate=False)
    valid95 = fpr[tpr >= 0.95]
    summary = {
        "family": family,
        "requested_level": float(level),
        "stress_test": bool(float(level) > 0.20),
        "clean_in_distribution_window_count_descriptive_only": int(len(clean)),
        "synthetically_corrupted_window_count_descriptive_only": int(len(noisy)),
        "matched_source_group_count": int(clean.group_id.nunique()),
        "synthetic_ood_auroc": float(roc_auc_score(truth, scores)),
        "synthetic_ood_auprc_average_precision": float(average_precision_score(truth, scores)),
        "synthetic_ood_fpr_at_95_tpr": float(np.min(valid95)) if len(valid95) else None,
        "synthetic_ood_prevalence_in_detection_benchmark": 0.5,
        "ood_definition": "the same development OOF window before versus after the named synthetic perturbation",
        "general_ood_claim": False,
        "interpretation": "detects only this operational synthetic shift; no clinical, acquisition, or general OOD labels are available",
    }
    group_rows: list[dict[str, Any]] = []
    clean_by_group = {str(group): rows for group, rows in clean.groupby("group_id", sort=True)}
    noisy_by_group = {str(group): rows for group, rows in noisy.groupby("group_id", sort=True)}
    if set(clean_by_group) != set(noisy_by_group):
        raise ValueError("synthetic OOD comparison has mismatched source groups")
    for group_id in sorted(clean_by_group):
        clean_scores = clean_by_group[group_id].ood_score.to_numpy(dtype=float)
        noisy_scores = noisy_by_group[group_id].ood_score.to_numpy(dtype=float)
        target = np.concatenate((np.zeros(len(clean_scores), dtype=int), np.ones(len(noisy_scores), dtype=int)))
        local_scores = np.concatenate((clean_scores, noisy_scores))
        local_fpr, local_tpr, _ = roc_curve(target, local_scores, drop_intermediate=False)
        local_fpr95 = local_fpr[local_tpr >= 0.95]
        group_rows.append(
            {
                "family": family,
                "requested_level": float(level),
                "stress_test": bool(float(level) > 0.20),
                "group_id": group_id,
                "clean_window_count_descriptive_only": int(len(clean_scores)),
                "corrupted_window_count_descriptive_only": int(len(noisy_scores)),
                "synthetic_ood_auroc": float(roc_auc_score(target, local_scores)),
                "synthetic_ood_auprc_average_precision": float(average_precision_score(target, local_scores)),
                "synthetic_ood_fpr_at_95_tpr": float(np.min(local_fpr95)) if len(local_fpr95) else None,
            }
        )
    return summary, group_rows


def run_noise_robustness_stage(
    run_dir: Path,
    *,
    tabular_subdir: str = "lowfreq_threshold_v2",
    neural_subdir: str = "neural_lowfreq_nested_4x3_v1",
    guardrail_subdir: str = "neural_lowfreq_guardrails_v2",
    augmented_subdir: str = "neural_lowfreq_augmented_nested_4x3_v1",
    output_subdir: str = "neural_lowfreq_noise_robustness_v1",
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Score matched fixed perturbations without changing fitted models or thresholds."""
    if int(bootstrap_replicates) < 1:
        raise ValueError("bootstrap_replicates must be positive")
    for name in (tabular_subdir, neural_subdir, guardrail_subdir, augmented_subdir, output_subdir):
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("stage directory names must be simple private directory names")
    inputs = _load_development_inputs(
        run_dir,
        tabular_subdir=tabular_subdir,
        neural_subdir=neural_subdir,
        guardrail_subdir=guardrail_subdir,
    )
    augmented_dir = inputs["private_dir"] / augmented_subdir
    augmented_manifest_path = augmented_dir / "augmentation_training_stage_manifest.json"
    augmented_manifest = json.loads(augmented_manifest_path.read_text(encoding="utf-8"))
    if (
        augmented_manifest.get("status") != "completed"
        or augmented_manifest.get("label_rule") != PRIMARY_LABEL_RULE
        or int(augmented_manifest.get("outer_group_folds", -1)) != 4
        or int(augmented_manifest.get("inner_group_folds_per_outer", -1)) != 3
        or augmented_manifest.get("external_rar_opened") is not False
        or int(augmented_manifest.get("internal_test_signal_values_parsed", -1)) != 0
        or int(augmented_manifest.get("reserved_test_member_payloads_read", -1)) != 0
    ):
        raise ValueError("augmented neural parent is not a completed low-frequency development-only 4x3 stage")
    _verify_hash_map(augmented_manifest.get("output_sha256", {}), "augmented neural parent output")
    _verify_hash_map(augmented_manifest.get("input_sha256", {}), "augmented neural parent input")
    if int(augmented_manifest.get("checkpoint_count", -1)) != len(MODEL_NAMES) * 4:
        raise ValueError("augmented neural parent does not contain twelve outer checkpoints")
    output_dir = inputs["private_dir"] / output_subdir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite noise-robustness stage: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    config = inputs["config"]
    metadata = inputs["metadata"].copy()
    active = inputs["active"].copy()
    fold_plan = inputs["fold_plan"]
    signals = inputs["signals"]
    active["fold_id"] = active.group_id.astype(str).map(
        {
            str(group_id): int(outer.fold_id)
            for outer in fold_plan.outer_folds
            for group_id in outer.validation_group_ids
        }
    ).astype(int)
    if active.fold_id.isna().any():
        raise ValueError("outer fold mapping failed for an assigned development source group")

    base_path = augmented_dir / "clean_vs_augmented_outer_oof_predictions.parquet"
    augmented_checkpoint_path = augmented_dir / "augmented_checkpoint_manifest.json"
    for path in (base_path, augmented_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"augmented training output is missing: {path.name}")
    base = pd.read_parquet(base_path)
    active_keys = active[["window_key", "group_id", "recording_id", "class_id", "fold_id", "eligible_for_label"]].copy()
    base = base.merge(active_keys, on=["window_key", "group_id", "recording_id"], how="left", suffixes=("", "_active"), validate="many_to_one")
    for identity in ("class_id",):
        right = f"{identity}_active"
        if right in base:
            if base[identity].isna().any() or not base[identity].astype(int).eq(base[right].astype(int)).all():
                raise ValueError("augmented clean OOF labels disagree with the active frozen low-frequency labels")
            base = base.drop(columns=[right])
    if "fold_id_active" not in base or base.fold_id.isna().any() or not base.fold_id.astype(int).eq(base.fold_id_active.astype(int)).all():
        raise ValueError("augmented clean OOF assignments disagree with the frozen outer group folds")
    base = base.drop(columns=["fold_id_active"])
    if base.eligible_for_label.isna().any():
        raise ValueError("augmented clean OOF predictions do not exactly join the assigned clean-quality rows")
    base["fold_id"] = base.fold_id.astype(int)
    if not base.true_class_id.astype(int).eq(base.class_id.astype(int)).all():
        raise ValueError("clean and augmented OOF true labels differ from the frozen low-frequency labels")
    base["true_class_id"] = base.class_id.astype(int)
    base = base.drop(columns=["class_id"])
    # The completed augmented parent predates corruption-seed storage in its
    # compact OOF parquet. Clean rows have no corruption, so seed zero is the
    # defined value; label eligibility was joined from frozen metadata above.
    if "corruption_seed" not in base:
        base["corruption_seed"] = np.uint32(0)
    else:
        base["corruption_seed"] = base.corruption_seed.fillna(0).astype(np.uint32)
    base["stress_test"] = False
    base = _canonical_prediction_frame(base)
    clean_ood_for_baseline = base.drop_duplicates(["window_key", "fold_id"])[
        ["window_key", "fold_id", "ood_score", "ood_threshold", "ood_flag"]
    ]
    temporal_clean = inputs["temporal_baseline"].merge(
        clean_ood_for_baseline,
        on=["window_key", "fold_id"],
        how="left",
        validate="one_to_one",
    )
    if temporal_clean.ood_flag.isna().any():
        raise ValueError("temporal baseline does not match the frozen clean OOD score for every OOF window")
    temporal_clean["true_class_id"] = temporal_clean.class_id.astype(int)
    temporal_clean = temporal_clean.drop(columns=["class_id"])
    temporal_clean = _canonical_prediction_frame(temporal_clean)
    base = pd.concat([base, temporal_clean], ignore_index=True)
    expected_base_keys = set(active.window_key.astype(str))
    for (condition, model_name), rows in base.groupby(["training_condition", "model_name"], sort=True):
        if set(rows.window_key.astype(str)) != expected_base_keys or len(rows) != len(active):
            raise ValueError(f"clean matched baseline for {condition}/{model_name} has incomplete key coverage")

    augmented_manifest_rows = json.loads(augmented_checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(augmented_manifest_rows, list) or len(augmented_manifest_rows) != 12:
        raise ValueError("augmented checkpoint manifest does not enumerate all model-fold fits")
    augmented_checkpoints: dict[tuple[str, int], Path] = {}
    for item in augmented_manifest_rows:
        path = Path(str(item["path"])).expanduser().resolve()
        model_name, fold_id = str(item["model_name"]), int(item["outer_fold_id"])
        if model_name not in MODEL_NAMES or not path.is_file() or _sha256_file(path) != str(item["sha256"]):
            raise ValueError("augmented checkpoint checksum or model identity failed")
        if (model_name, fold_id) in augmented_checkpoints:
            raise ValueError("augmented checkpoint manifest has duplicate architecture-fold entries")
        augmented_checkpoints[(model_name, fold_id)] = path

    model_cache: dict[tuple[str, str, int], tuple[Any, dict[str, Any]]] = {}
    for model_name in MODEL_NAMES:
        for outer in fold_plan.outer_folds:
            fold_id = int(outer.fold_id)
            clean_path = inputs["checkpoints"][(model_name, fold_id)]
            augmented_path = augmented_checkpoints[(model_name, fold_id)]
            model_cache[("clean", model_name, fold_id)] = _checkpoint_model(clean_path, model_name)
            model_cache[("augmented", model_name, fold_id)] = _checkpoint_model(augmented_path, model_name)

    augmented_outer = pd.read_parquet(augmented_dir / "augmented_outer_clean_oof_predictions.parquet")
    if augmented_outer.duplicated(["model_name", "fold_id", "window_key"]).any():
        raise ValueError("augmented clean OOF has duplicate architecture-fold-window predictions")
    # Check frozen model thresholds and confidence policies against their own OOF receipts.
    threshold_receipt = pd.read_csv(augmented_dir / "augmented_inner_threshold_selection.csv")
    confidence_receipt = pd.read_csv(augmented_dir / "augmented_inner_confidence_selection.csv")
    if len(threshold_receipt) != 12 or len(confidence_receipt) != 12:
        raise ValueError("augmented nested OOF selection does not cover every architecture-fold pair")
    if confidence_receipt.duplicated(["model_name", "outer_fold_id"]).any():
        raise ValueError("augmented confidence receipt contains duplicate architecture-fold policies")
    confidence_threshold_by_condition_model_fold: dict[tuple[str, str, int], float] = {}
    clean_policy = inputs["clean_policy"]
    if len(clean_policy) != 12 or clean_policy.duplicated(["model_name", "outer_fold_id"]).any():
        raise ValueError("clean guardrail receipt does not contain one confidence policy per architecture-fold fit")
    for row in clean_policy.itertuples(index=False):
        key = ("clean", str(row.model_name), int(row.outer_fold_id))
        confidence_threshold_by_condition_model_fold[key] = float(row.selected_confidence_threshold)
    for row in confidence_receipt.itertuples(index=False):
        key = ("augmented", str(row.model_name), int(row.outer_fold_id))
        if key in confidence_threshold_by_condition_model_fold:
            raise ValueError("augmented confidence receipt contains a duplicate model-fold policy")
        confidence_threshold_by_condition_model_fold[key] = float(row.selected_confidence_threshold)
    for (condition, model_name, fold_id), (_, checkpoint_meta) in model_cache.items():
        key = (str(condition), str(model_name), int(fold_id))
        if condition == "clean":
            rows = inputs["clean_oof"].loc[
                inputs["clean_oof"].model_name.astype(str).eq(model_name)
                & inputs["clean_oof"].fold_id.astype(int).eq(int(fold_id)),
                "selected_confidence_threshold",
            ]
            if rows.empty or not np.allclose(rows.to_numpy(dtype=float), confidence_threshold_by_condition_model_fold[key]):
                raise ValueError("clean OOF confidence policy disagrees with the frozen guardrail receipt")
            continue
        selected = threshold_receipt.loc[
            threshold_receipt.model_name.astype(str).eq(model_name)
            & threshold_receipt.outer_fold_id.astype(int).eq(int(fold_id))
        ]
        conf = confidence_receipt.loc[
            confidence_receipt.model_name.astype(str).eq(model_name)
            & confidence_receipt.outer_fold_id.astype(int).eq(int(fold_id))
        ]
        if len(selected) != 1 or len(conf) != 1:
            raise ValueError("augmented threshold or confidence receipt lacks a unique model-fold row")
        if not np.isclose(float(selected.threshold.iloc[0]), float(checkpoint_meta["inner_oof_selected_threshold"])):
            raise ValueError("augmented checkpoint threshold disagrees with its inner OOF receipt")
        if not np.isclose(float(conf.selected_confidence_threshold.iloc[0]), confidence_threshold_by_condition_model_fold[key]):
            raise ValueError("augmented confidence policy disagrees with its frozen inner OOF receipt")
        if not np.isclose(float(conf.selected_confidence_threshold.iloc[0]), float(checkpoint_meta["inner_oof_selected_confidence_threshold"])):
            raise ValueError("augmented checkpoint abstention threshold disagrees with its inner OOF receipt")
    if set(confidence_threshold_by_condition_model_fold) != {
        (condition, model_name, int(fold.fold_id))
        for condition in ("clean", "augmented")
        for model_name in MODEL_NAMES
        for fold in fold_plan.outer_folds
    }:
        raise ValueError("frozen clean and augmented confidence policies do not cover all model-fold fits")

    ood_parameters: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    ood_fit_receipts: list[dict[str, Any]] = []
    quantile = float(config.get("guardrails", {}).get("ood_upper_quantile", OOD_QUANTILE))
    for outer in fold_plan.outer_folds:
        center, scale, keep, threshold, receipt = _fit_fold_ood_reference(
            inputs["clean_features"],
            outer.train_group_ids,
            quantile=quantile,
            outer_fold_id=int(outer.fold_id),
        )
        previous = float(inputs["ood_receipts"][int(outer.fold_id)]["ood_threshold"])
        if not np.isclose(threshold, previous, rtol=1e-9, atol=1e-10):
            raise ValueError("development OOD threshold differs from the frozen guardrail stage")
        ood_parameters[int(outer.fold_id)] = (center, scale, keep, threshold)
        ood_fit_receipts.append(receipt)

    base_metrics, base_group_metrics = _metric_tables(base)
    pooled_metric_rows = base_metrics.to_dict(orient="records")
    group_metric_rows = base_group_metrics.to_dict(orient="records")
    selective_metric_rows: list[dict[str, Any]] = []
    risk_curve_rows: list[dict[str, Any]] = []
    risk_summary_rows: list[dict[str, Any]] = []
    group_confusions: dict[tuple[str, str, str, float], dict[str, np.ndarray]] = {}
    gate_rows: list[dict[str, Any]] = []
    inference_timing_rows: list[dict[str, Any]] = []
    noise_window_receipts: list[pd.DataFrame] = []
    paired_groups = tuple(sorted(active.group_id.astype(str).unique()))
    for (condition, model_name), rows in base.groupby(["training_condition", "model_name"], sort=True):
        group_confusions[(str(condition), str(model_name), "clean", 0.0)] = _group_normalized_confusion(rows, "pred_selected")
        cuts = rows.selected_confidence_threshold.to_numpy(dtype=float)
        selective = _selective_summary(rows, cuts)
        cut_by_fold = {
            str(int(fold)): sorted(set(float(value) for value in part.selected_confidence_threshold))
            for fold, part in rows.groupby("fold_id", sort=True)
        }
        for policy, metrics in selective.items():
            selective_metric_rows.append(
                {
                    "training_condition": condition,
                    "model_name": model_name,
                    "family": "clean",
                    "requested_level": 0.0,
                    "policy": policy,
                    "selected_confidence_thresholds_by_outer_fold": json.dumps(cut_by_fold, sort_keys=True, separators=(",", ":")),
                    **metrics,
                }
            )
        curve, summary = _risk_coverage(rows, np.linspace(0.5, 1.0, 51))
        curve.insert(0, "requested_level", 0.0)
        curve.insert(0, "family", "clean")
        curve.insert(0, "model_name", model_name)
        curve.insert(0, "training_condition", condition)
        risk_curve_rows.extend(curve.to_dict(orient="records"))
        risk_summary_rows.append(
            {
                "training_condition": condition,
                "model_name": model_name,
                "family": "clean",
                "requested_level": 0.0,
                **summary,
            }
        )

    output_prediction_path = output_dir / "noise_robustness_oof_predictions.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    prediction_schema = None

    def write_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
        nonlocal writer, prediction_schema
        canonical = _canonical_prediction_frame(frame)
        table = pa.Table.from_pandas(canonical, preserve_index=False)
        if writer is None:
            prediction_schema = table.schema
            writer = pq.ParquetWriter(output_prediction_path, prediction_schema, compression="zstd")
        elif table.schema != prediction_schema:
            raise TypeError("robustness parquet chunks do not share a stable schema")
        writer.write_table(table, row_group_size=15000)
        return canonical

    try:
        for (condition, model_name), rows in base.groupby(["training_condition", "model_name"], sort=True):
            frame = write_prediction_frame(rows)
            _ = frame

        label_summary_rows: list[dict[str, Any]] = []
        label_group_rows: list[dict[str, Any]] = []
        severity_rows: list[dict[str, Any]] = []
        quality_rows: list[dict[str, Any]] = []
        ood_summary_rows: list[dict[str, Any]] = []
        ood_group_rows: list[dict[str, Any]] = []
        ood_detection_rows: list[dict[str, Any]] = []
        ood_detection_group_rows: list[dict[str, Any]] = []
        # Recompute the clean proxy rule first to verify reconstruction equivalence.
        clean_labels = _recompute_lowfreq_rule(metadata, signals, inputs["notch_by_window"], config)
        clean_consistency, clean_group_consistency = _label_consistency_rows(metadata, clean_labels, "clean", 0.0)
        if not np.isclose(clean_consistency["exact_class_agreement_including_ambiguity_as_disagreement"], 1.0):
            raise RuntimeError("clean low-frequency rule replay does not reproduce all original assigned pseudo-labels")
        label_summary_rows.append(clean_consistency)
        label_group_rows.extend(clean_group_consistency)
        clean_active_ood = base.drop_duplicates(["window_key", "fold_id"])[
            ["window_key", "group_id", "recording_id", "true_class_id", "fold_id", "ood_score", "ood_threshold", "ood_flag"]
        ].rename(columns={"true_class_id": "class_id"})
        clean_ood_summary, clean_ood_groups = _ood_rate_tables(clean_active_ood, family="clean", level=0.0)
        ood_summary_rows.append(clean_ood_summary)
        ood_group_rows.extend(clean_ood_groups)

        fs = float(metadata.sampling_rate_hz.iloc[0])
        active_positions = active.bank_position.to_numpy(dtype=int)
        active_index_by_key = {str(key): index for index, key in enumerate(active.window_key.astype(str))}
        for family in CORRUPTION_FAMILIES:
            for level in EVALUATION_LEVELS:
                noisy_full, achieved_full, seeds_full, corruption_receipt = _build_matched_corruption(
                    signals,
                    metadata,
                    family,
                    float(level),
                )
                noise_window_receipts.append(corruption_receipt)
                active_achieved = achieved_full[active_positions]
                severity_rows.append(
                    {
                        "family": family,
                        "requested_level": float(level),
                        "stress_test": bool(float(level) > 0.20),
                        "window_count_descriptive_only": int(len(achieved_full)),
                        "assigned_oof_window_count_descriptive_only": int(len(active_achieved)),
                        "achieved_rms_ratio_mean_all_windows": float(np.mean(achieved_full)),
                        "achieved_rms_ratio_median_all_windows": float(np.median(achieved_full)),
                        "achieved_rms_ratio_p95_all_windows": float(np.quantile(achieved_full, 0.95)),
                        "achieved_rms_ratio_min_all_windows": float(np.min(achieved_full)),
                        "achieved_rms_ratio_max_all_windows": float(np.max(achieved_full)),
                        "achieved_rms_ratio_mean_assigned_oof_windows": float(np.mean(active_achieved)),
                        "achieved_rms_ratio_median_assigned_oof_windows": float(np.median(active_achieved)),
                        "achieved_rms_ratio_p95_assigned_oof_windows": float(np.quantile(active_achieved, 0.95)),
                        "severity_semantics": "RMS(delta)/RMS(clean), except shift where achieved difference is signal-dependent",
                        "seed_scheme": "stable SHA-256 from window_key, family, and requested level",
                    }
                )
                recomputed = _recompute_lowfreq_rule(metadata, noisy_full, inputs["notch_by_window"], config)
                label_summary, label_groups = _label_consistency_rows(metadata, recomputed, family, float(level))
                label_summary_rows.append(label_summary)
                label_group_rows.extend(label_groups)

                active_noisy = noisy_full[active_positions]
                morphology = _morphology_batch(active_noisy, fs)
                if not np.isfinite(morphology[list(OOD_FEATURES)].to_numpy(dtype=float)).all():
                    raise ValueError(f"OOD morphology features became non-finite for {family} at {level:g}")
                ood_scores = np.full(len(active), np.nan, dtype=np.float64)
                ood_thresholds = np.full(len(active), np.nan, dtype=np.float64)
                ood_flags = np.zeros(len(active), dtype=bool)
                for outer in fold_plan.outer_folds:
                    fold_id = int(outer.fold_id)
                    val_mask = active.group_id.astype(str).isin(set(outer.validation_group_ids)).to_numpy()
                    val_indices = np.flatnonzero(val_mask)
                    center, scale, keep, threshold = ood_parameters[fold_id]
                    scores = _robust_ood_score(
                        morphology.iloc[val_indices][list(OOD_FEATURES)].to_numpy(dtype=float),
                        center,
                        scale,
                        keep,
                    )
                    ood_scores[val_indices] = scores
                    ood_thresholds[val_indices] = threshold
                    ood_flags[val_indices] = scores > threshold
                if not np.isfinite(ood_scores).all() or not np.isfinite(ood_thresholds).all():
                    raise RuntimeError("noise OOD scores did not cover every assigned outer OOF window")
                ood_frame = active[["window_key", "group_id", "recording_id", "class_id", "fold_id"]].copy()
                ood_frame["ood_score"] = ood_scores
                ood_frame["ood_threshold"] = ood_thresholds
                ood_frame["ood_flag"] = ood_flags
                ood_summary, ood_groups = _ood_rate_tables(ood_frame, family=family, level=float(level))
                ood_summary_rows.append(ood_summary)
                ood_group_rows.extend(ood_groups)
                detection_row, detection_groups = _synthetic_ood_detection_metrics(
                    clean_active_ood,
                    ood_frame,
                    family=family,
                    level=float(level),
                )
                ood_detection_rows.append(detection_row)
                ood_detection_group_rows.extend(detection_groups)

                active_variance = np.var(active_noisy, axis=1)
                quality_rows.append(
                    {
                        "family": family,
                        "requested_level": float(level),
                        "stress_test": bool(float(level) > 0.20),
                        "assigned_window_count_descriptive_only": int(len(active_noisy)),
                        "finite_corrupted_window_fraction": float(np.mean(np.isfinite(active_noisy).all(axis=1))),
                        "positive_variance_fraction": float(np.mean(active_variance > np.finfo(float).eps)),
                        "original_clean_qc_eligible_fraction": float(active.eligible_for_label.astype(bool).mean()),
                        "qc_recomputed_after_corruption": False,
                        "interpretation": "clean preprocessing eligibility is retained; these summaries do not certify noisy-signal QC",
                    }
                )

                active_frames: list[pd.DataFrame] = []
                for condition in ("clean", "augmented"):
                    for model_name in MODEL_NAMES:
                        fold_parts: list[pd.DataFrame] = []
                        for outer in fold_plan.outer_folds:
                            fold_id = int(outer.fold_id)
                            valid_rows = active.loc[
                                active.group_id.astype(str).isin(set(outer.validation_group_ids))
                            ].sort_values("window_key").reset_index(drop=True)
                            checkpoint, checkpoint_meta = model_cache[(condition, model_name, fold_id)]
                            fit = checkpoint_meta["normalization_and_pretraining_receipt"]
                            mean = float(fit.get("normalization_mean", fit.get("normalization_mean_from_clean_training_windows")))
                            std = float(fit.get("normalization_std", fit.get("normalization_std_from_clean_training_windows")))
                            positions = valid_rows.bank_position.to_numpy(dtype=int)
                            normalized = (noisy_full[positions] - mean) / std
                            inference_started = time.perf_counter()
                            logits, gates = _predict_logits_chunked(checkpoint, normalized)
                            inference_seconds = float(time.perf_counter() - inference_started)
                            inference_timing_rows.append(
                                {
                                    "training_condition": condition,
                                    "model_name": model_name,
                                    "family": family,
                                    "requested_level": float(level),
                                    "stress_test": bool(float(level) > 0.20),
                                    "outer_fold_id": fold_id,
                                    "window_count_descriptive_only": int(len(valid_rows)),
                                    "inference_seconds": inference_seconds,
                                    "inference_milliseconds_per_window": 1000.0 * inference_seconds / len(valid_rows),
                                    "batch_size": 512,
                                    "device": "cpu" if not torch.cuda.is_available() else "cuda",
                                }
                            )
                            probability = _score_probabilities(logits, checkpoint_meta)
                            threshold = float(checkpoint_meta["inner_oof_selected_threshold"])
                            confidence_threshold = confidence_threshold_by_condition_model_fold[
                                (condition, model_name, fold_id)
                            ]
                            local_active_positions = np.asarray(
                                [active_index_by_key[str(key)] for key in valid_rows.window_key.astype(str)],
                                dtype=np.int64,
                            )
                            local_ood_score = ood_scores[local_active_positions]
                            ood_threshold = float(ood_parameters[fold_id][3])
                            fold_frame = pd.DataFrame(
                                {
                                    "window_key": valid_rows.window_key.astype(str).to_numpy(),
                                    "group_id": valid_rows.group_id.astype(str).to_numpy(),
                                    "recording_id": valid_rows.recording_id.astype(str).to_numpy(),
                                    "fold_id": fold_id,
                                    "model_name": model_name,
                                    "true_class_id": valid_rows.class_id.to_numpy(dtype=int),
                                    "raw_logit": logits,
                                    "probability_class_1": probability,
                                    "pred_fixed": (probability >= 0.5).astype(int),
                                    "pred_selected": (probability >= threshold).astype(int),
                                    "selected_threshold": threshold,
                                    "confidence": np.maximum(probability, 1.0 - probability),
                                    "selected_confidence_threshold": confidence_threshold,
                                    "ood_score": local_ood_score,
                                    "ood_threshold": ood_threshold,
                                    "ood_flag": local_ood_score > ood_threshold,
                                    "training_condition": condition,
                                    "family": family,
                                    "requested_level": float(level),
                                    "achieved_rms_ratio": achieved_full[positions],
                                    "corruption_seed": seeds_full[positions],
                                    "stress_test": bool(float(level) > 0.20),
                                    "eligible_for_label": valid_rows.eligible_for_label.fillna(False).astype(bool).to_numpy(),
                                }
                            )
                            fold_parts.append(fold_frame)
                            gate_row = _gate_receipt(
                                gates,
                                model_name,
                                training_condition=condition,
                                family=family,
                                requested_level=float(level),
                                outer_fold_id=fold_id,
                            )
                            if gate_row is not None:
                                gate_rows.append(gate_row)
                        combined = pd.concat(fold_parts, ignore_index=True)
                        if combined.duplicated("window_key").any() or set(combined.window_key.astype(str)) != expected_base_keys:
                            raise RuntimeError(f"matched robustness keys differ for {condition}/{model_name}/{family}/{level:g}")
                        canonical = write_prediction_frame(combined)
                        active_frames.append(canonical)
                        metric, group_metric = _metric_tables(canonical)
                        pooled_metric_rows.extend(metric.to_dict(orient="records"))
                        group_metric_rows.extend(group_metric.to_dict(orient="records"))
                        group_confusions[(condition, model_name, family, float(level))] = _group_normalized_confusion(
                            canonical,
                            "pred_selected",
                        )
                        cuts = canonical.selected_confidence_threshold.to_numpy(dtype=float)
                        selective = _selective_summary(canonical, cuts)
                        cut_by_fold = {
                            str(int(fold)): sorted(set(float(value) for value in part.selected_confidence_threshold))
                            for fold, part in canonical.groupby("fold_id", sort=True)
                        }
                        for policy, metrics in selective.items():
                            selective_metric_rows.append(
                                {
                                    "training_condition": condition,
                                    "model_name": model_name,
                                    "family": family,
                                    "requested_level": float(level),
                                    "policy": policy,
                                    "selected_confidence_thresholds_by_outer_fold": json.dumps(cut_by_fold, sort_keys=True, separators=(",", ":")),
                                    **metrics,
                                }
                            )
                        curve, summary = _risk_coverage(canonical, np.linspace(0.5, 1.0, 51))
                        curve.insert(0, "requested_level", float(level))
                        curve.insert(0, "family", family)
                        curve.insert(0, "model_name", model_name)
                        curve.insert(0, "training_condition", condition)
                        risk_curve_rows.extend(curve.to_dict(orient="records"))
                        risk_summary_rows.append(
                            {
                                "training_condition": condition,
                                "model_name": model_name,
                                "family": family,
                                "requested_level": float(level),
                                **summary,
                            }
                        )
                    del fold_parts
                temporal_noisy = inputs["temporal_baseline"].merge(
                    ood_frame[["window_key", "fold_id", "ood_score", "ood_threshold", "ood_flag"]],
                    on=["window_key", "fold_id"],
                    how="left",
                    validate="one_to_one",
                )
                if temporal_noisy.ood_flag.isna().any():
                    raise ValueError("temporal baseline OOD scores do not cover the matched noisy condition")
                temporal_noisy["true_class_id"] = temporal_noisy.class_id.astype(int)
                temporal_noisy = temporal_noisy.drop(columns=["class_id"])
                temporal_noisy["family"] = family
                temporal_noisy["requested_level"] = float(level)
                temporal_noisy["achieved_rms_ratio"] = achieved_full[temporal_noisy.bank_position.to_numpy(dtype=int)]
                temporal_noisy["corruption_seed"] = seeds_full[temporal_noisy.bank_position.to_numpy(dtype=int)]
                temporal_noisy["stress_test"] = bool(float(level) > 0.20)
                temporal_noisy["training_condition"] = "temporal_baseline"
                temporal_noisy = _canonical_prediction_frame(temporal_noisy)
                if set(temporal_noisy.window_key.astype(str)) != expected_base_keys:
                    raise RuntimeError("temporal baseline omitted windows in the matched noise condition")
                write_prediction_frame(temporal_noisy)
                temporal_metrics, temporal_group_metrics = _metric_tables(temporal_noisy)
                pooled_metric_rows.extend(temporal_metrics.to_dict(orient="records"))
                group_metric_rows.extend(temporal_group_metrics.to_dict(orient="records"))
                group_confusions[("temporal_baseline", "time_only_logistic", family, float(level))] = (
                    _group_normalized_confusion(temporal_noisy, "pred_selected")
                )
                temporal_selective = _selective_summary(
                    temporal_noisy,
                    temporal_noisy.selected_confidence_threshold.to_numpy(dtype=float),
                )
                for policy, metric_summary in temporal_selective.items():
                    selective_metric_rows.append(
                        {
                            "training_condition": "temporal_baseline",
                            "model_name": "time_only_logistic",
                            "family": family,
                            "requested_level": float(level),
                            "policy": policy,
                            "selected_confidence_thresholds_by_outer_fold": json.dumps({"all": 0.5}, separators=(",", ":")),
                            **metric_summary,
                        }
                    )
                temporal_curve, temporal_aurc = _risk_coverage(
                    temporal_noisy,
                    np.linspace(0.5, 1.0, 51),
                )
                temporal_curve.insert(0, "requested_level", float(level))
                temporal_curve.insert(0, "family", family)
                temporal_curve.insert(0, "model_name", "time_only_logistic")
                temporal_curve.insert(0, "training_condition", "temporal_baseline")
                risk_curve_rows.extend(temporal_curve.to_dict(orient="records"))
                risk_summary_rows.append(
                    {
                        "training_condition": "temporal_baseline",
                        "model_name": "time_only_logistic",
                        "family": family,
                        "requested_level": float(level),
                        **temporal_aurc,
                    }
                )
                print(
                    f"Matched robustness condition complete: family={family} level={level:g} "
                    f"(stress={float(level) > 0.20})",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    paired = _paired_group_bootstrap_from_confusions(
        group_confusions,
        groups=paired_groups,
        replicates=int(bootstrap_replicates),
        seed=RANDOM_SEED,
    )
    pd.DataFrame(pooled_metric_rows).to_csv(output_dir / "noise_robustness_metrics.csv", index=False)
    pd.DataFrame(group_metric_rows).to_csv(output_dir / "noise_robustness_group_metrics.csv", index=False)
    pd.DataFrame(selective_metric_rows).to_csv(output_dir / "noise_selective_prediction_metrics.csv", index=False)
    pd.DataFrame(risk_curve_rows).to_csv(output_dir / "noise_risk_coverage_curves.csv", index=False)
    pd.DataFrame(risk_summary_rows).to_csv(output_dir / "noise_risk_coverage_summary.csv", index=False)
    pd.DataFrame(label_summary_rows).to_csv(output_dir / "noise_label_rule_consistency.csv", index=False)
    pd.DataFrame(label_group_rows).to_csv(output_dir / "noise_label_rule_consistency_by_group.csv", index=False)
    pd.DataFrame(severity_rows).to_csv(output_dir / "noise_severity_receipts.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(output_dir / "noise_signal_integrity_summary.csv", index=False)
    pd.DataFrame(ood_summary_rows).to_csv(output_dir / "noise_ood_flag_rates.csv", index=False)
    pd.DataFrame(ood_group_rows).to_csv(output_dir / "noise_ood_group_flag_rates.csv", index=False)
    pd.DataFrame(ood_detection_rows).to_csv(output_dir / "synthetic_ood_detection_metrics.csv", index=False)
    pd.DataFrame(ood_detection_group_rows).to_csv(output_dir / "synthetic_ood_detection_by_group.csv", index=False)
    paired.to_csv(output_dir / "noise_paired_group_bootstrap.csv", index=False)
    pd.DataFrame(inference_timing_rows).to_csv(output_dir / "noise_inference_timing.csv", index=False)
    pd.concat(noise_window_receipts, ignore_index=True).to_parquet(
        output_dir / "noise_corruption_window_receipts.parquet",
        index=False,
    )
    if gate_rows:
        pd.DataFrame(gate_rows).to_csv(output_dir / "noise_moe_gate_diagnostics.csv", index=False)

    source_hashes = _source_hashes()
    parent_paths = list(inputs["input_paths"]) + list(inputs["checkpoints"].values())
    parent_paths.extend(Path(name).expanduser().resolve() for name in augmented_manifest.get("output_sha256", {}))
    for parent_manifest in (
        inputs["tabular_dir"] / "supervised_threshold_stage_manifest.json",
        inputs["neural_dir"] / "neural_supervised_stage_manifest.json",
        inputs["guardrail_dir"] / "guardrail_stage_manifest.json",
        augmented_manifest_path,
    ):
        parent_paths.append(parent_manifest)
    manifest = {
        "stage": "development_only_matched_noise_robustness_and_label_rule_consistency",
        "status": "completed",
        "run_id": inputs["run_dir"].name,
        "label_rule": PRIMARY_LABEL_RULE,
        "label_rule_type": PROXY_LABEL_TYPE,
        "interpretation": "all classifier results measure agreement with a weak-supervision operational proxy, not physiological fatigue validation",
        "outer_group_folds": 4,
        "inner_group_folds_per_outer": 3,
        "source_group_count": 12,
        "class_support": _support(active, "full development cohort"),
        "independent_unit": "source folder group; overlapping windows are descriptive and not independent observations",
        "models": list(MODEL_NAMES),
        "training_conditions": ["clean", "augmented"],
        "noise_families": list(CORRUPTION_FAMILIES),
        "evaluation_levels": list(EVALUATION_LEVELS),
        "stress_test_levels": [0.25, 0.50],
        "severity_definition": "requested additive RMS(delta)/RMS(clean); temporal shifts record achieved relative RMS difference and requested level maps to up to 5% duration",
        "noise_seed_scheme": "per-window SHA-256-derived seeds from window_key, family and requested level; identical corrupted arrays shared across architectures and training conditions",
        "robustness_scoring": "frozen outer-fold checkpoint, calibration, fixed 0.5 threshold, selected inner OOF threshold, confidence policy and OOD policy; no retraining or retuning on corruption results",
        "label_consistency": {
            "rule": PRIMARY_LABEL_RULE,
            "reference_windows": "even-index 500 ms windows corresponding to the original non-overlapping reference stream",
            "eligibility": "frozen clean preprocessing/QC eligibility retained",
            "training_labels_changed": False,
            "clean_rule_replay_exact_assigned_label_agreement": float(clean_consistency["exact_class_agreement_including_ambiguity_as_disagreement"]),
            "interpretation": "post-filter corruption sensitivity of the frozen proxy rule only; not evidence of physiology",
        },
        "ood_detector": {
            "method": "frozen outer-training robust median/MAD distance and leave-one-group-out group-p95 threshold",
            "threshold_quantile": quantile,
            "uses_no_corruption_labels_for_threshold_or_fit": True,
            "detector_ground_truth_available": False,
            "confidence_is_not_ood": True,
        },
        "abstention": {
            "confidence_policy": "inner OOF group-balanced Macro-F1 with abstentions as class errors; no outer validation use",
            "reported_policies": ["confidence_only", "ood_only", "confidence_or_ood"],
            "risk_coverage_aurc": "pooled and group-balanced curves on fixed confidence cuts; policy thresholds stay fold-specific",
        },
        "paired_group_bootstrap_replicates": int(bootstrap_replicates),
        "paired_group_bootstrap_comparisons": [
            "augmented-training minus clean-training under identical corruption and pseudo-label windows",
            "noisy-input minus clean-input within each training condition and architecture",
        ],
        "checkpoint_count": int(len(model_cache)),
        "all_clean_and_augmented_checkpoints_hash_verified": True,
        "internal_test_signal_values_parsed": 0,
        "reserved_test_member_payloads_read": 0,
        "external_rar_opened": False,
        "raw2_rar_opened": False,
        "noise_qc_recomputed": False,
        "noise_qc_limitation": "original clean preprocessing eligibility is retained; finite and nonzero-variance rates are descriptive checks, not a re-run of all artifact or filter gates",
        "input_sha256": _hash_map(sorted(set(Path(path).resolve() for path in parent_paths if Path(path).is_file()))),
        "source_sha256": source_hashes,
        "output_sha256": _output_hashes(output_dir),
    }
    manifest_path = output_dir / "noise_robustness_stage_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Completed matched development robustness analysis: {manifest_path}", flush=True)
    return {
        "status": "completed",
        "manifest_path": str(manifest_path),
        "models": list(MODEL_NAMES),
        "noise_conditions": int(len(CORRUPTION_FAMILIES) * len(EVALUATION_LEVELS)),
        "paired_comparisons": int(len(paired)),
    }


def _selective_summary(rows: pd.DataFrame, confidence_threshold: float | Sequence[float]) -> dict[str, Any]:
    truth = rows.true_class_id.to_numpy(dtype=int)
    predicted = rows.pred_selected.to_numpy(dtype=int)
    probability = rows.probability_class_1.to_numpy(dtype=float)
    confidence = rows.confidence.to_numpy(dtype=float)
    ood = rows.ood_flag.to_numpy(dtype=bool)
    cut = np.asarray(confidence_threshold, dtype=np.float64)
    if cut.ndim == 0:
        cut = np.full(len(rows), float(cut), dtype=np.float64)
    if cut.shape != (len(rows),) or not np.isfinite(cut).all():
        raise ValueError("confidence abstention thresholds must be finite and scalar or row-aligned")
    confidence_abstain = confidence < cut
    policies = {
        "confidence_only": confidence_abstain,
        "ood_only": ood,
        "confidence_or_ood": confidence_abstain | ood,
    }
    output: dict[str, Any] = {}
    for policy_name, abstain in policies.items():
        selective_prediction = np.where(abstain, -1, predicted)
        record = compute_metrics(truth, probability, selective_prediction, abstain).to_dict()
        accepted = ~abstain
        output[policy_name] = {
            "coverage": float(np.mean(accepted)) if len(accepted) else None,
            "coverage_class_0": float(np.mean(accepted[truth == 0])) if np.any(truth == 0) else None,
            "coverage_class_1": float(np.mean(accepted[truth == 1])) if np.any(truth == 1) else None,
            "abstained_count": int(np.sum(abstain)),
            "abstained_count_class_0": int(np.sum(abstain & (truth == 0))),
            "abstained_count_class_1": int(np.sum(abstain & (truth == 1))),
            "selective_error": record["selective_error"],
            "accuracy_including_abstention_as_error": record["accuracy"],
            "macro_f1_including_abstention": record["macro_f1"],
            "confusion_matrix_with_abstention": json.dumps(record["confusion_matrix"], separators=(",", ":")),
        }
    return output


def _risk_coverage(rows: pd.DataFrame, cuts: Sequence[float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = rows.true_class_id.to_numpy(dtype=int)
    prediction = rows.pred_selected.to_numpy(dtype=int)
    confidence = rows.confidence.to_numpy(dtype=float)
    per_point: list[dict[str, Any]] = []
    for cut in sorted(set(float(value) for value in cuts)):
        accepted = confidence >= float(cut)
        group_coverage: list[float] = []
        group_risk: list[float] = []
        for _, group in rows.assign(_accepted=accepted, _error=(prediction != truth)).groupby("group_id", sort=True):
            local_accept = group._accepted.to_numpy(dtype=bool)
            group_coverage.append(float(np.mean(local_accept)))
            if np.any(local_accept):
                group_risk.append(float(np.mean(group._error.to_numpy(dtype=bool)[local_accept])))
        pooled_coverage = float(np.mean(accepted)) if len(accepted) else 0.0
        pooled_risk = float(np.mean((prediction[accepted] != truth[accepted]))) if np.any(accepted) else None
        per_point.append(
            {
                "confidence_cut": float(cut),
                "pooled_coverage": pooled_coverage,
                "pooled_selective_error": pooled_risk,
                "group_balanced_coverage": float(np.mean(group_coverage)) if group_coverage else None,
                "group_balanced_selective_error": float(np.mean(group_risk)) if group_risk else None,
                "groups_with_accepted_predictions": int(len(group_risk)),
            }
        )
    curve = pd.DataFrame(per_point)
    summary: dict[str, Any] = {}
    for coverage_column, risk_column, output_name in (
        ("pooled_coverage", "pooled_selective_error", "pooled_aurc"),
        ("group_balanced_coverage", "group_balanced_selective_error", "group_balanced_aurc"),
    ):
        valid = curve.loc[curve[risk_column].notna()].sort_values(coverage_column)
        if len(valid) >= 2:
            x = valid[coverage_column].to_numpy(dtype=float)
            y = valid[risk_column].to_numpy(dtype=float)
            unique = np.concatenate(([True], np.diff(x) > 1e-12))
            summary[output_name] = float(np.trapezoid(y[unique], x[unique])) if unique.sum() >= 2 else None
        else:
            summary[output_name] = None
    summary["risk_coverage_curve_points"] = int(len(curve))
    summary["confidence_score_is_not_ood"] = True
    return curve, summary


def _source_hashes() -> dict[str, str]:
    module_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[3]
    candidates = (
        Path(__file__).resolve(),
        module_dir / "augmentation.py",
        module_dir / "evaluation.py",
        module_dir / "evaluation_guardrails.py",
        module_dir / "evaluation_neural.py",
        module_dir / "evaluation_neural_nested.py",
        module_dir / "models_neural.py",
        module_dir / "ssl_pretraining.py",
        module_dir / "contracts.py",
        repo_root / "tools" / "run_fatigue_proxy.py",
    )
    return _hash_map([path for path in candidates if path.is_file()])


def _output_hashes(output_dir: Path) -> dict[str, str]:
    return _hash_map(
        sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "augmentation_training_stage_manifest.json"
        )
    )


def run_augmented_nested_training_stage(
    run_dir: Path,
    *,
    tabular_subdir: str = "lowfreq_threshold_v2",
    neural_subdir: str = "neural_lowfreq_nested_4x3_v1",
    guardrail_subdir: str = "neural_lowfreq_guardrails_v2",
    output_subdir: str = "neural_lowfreq_augmented_nested_4x3_v1",
    max_epochs: int | None = None,
    ssl_pretrain_epochs: int = 6,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Train augmented CNN, fine-tuned SSL encoder and latent MoE inside the frozen 4x3 design."""
    if Path(output_subdir).name != output_subdir or output_subdir in {"", ".", ".."}:
        raise ValueError("output_subdir must be one private directory name")
    if int(ssl_pretrain_epochs) < 1 or int(bootstrap_replicates) < 1:
        raise ValueError("SSL epoch count and bootstrap replicates must be positive")
    inputs = _load_development_inputs(
        run_dir,
        tabular_subdir=tabular_subdir,
        neural_subdir=neural_subdir,
        guardrail_subdir=guardrail_subdir,
    )
    config = inputs["config"]
    epoch_limit = int(max_epochs if max_epochs is not None else config.get("models", {}).get("neural_epochs", 20))
    if epoch_limit < 1:
        raise ValueError("max_epochs must be positive")
    confidence_candidates = tuple(
        float(value)
        for value in config.get("guardrails", {}).get("confidence_thresholds", CONFIDENCE_CANDIDATES_DEFAULT)
    )
    if not confidence_candidates or any(value < 0.5 or value > 1.0 for value in confidence_candidates):
        raise ValueError("confidence thresholds must be non-empty and lie in [0.5, 1]")
    output_dir = inputs["private_dir"] / output_subdir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite augmented neural stage: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    training = _run_augmented_nested_training(
        inputs,
        output_dir,
        max_epochs=epoch_limit,
        ssl_pretrain_epochs=int(ssl_pretrain_epochs),
        confidence_candidates=confidence_candidates,
    )
    clean = inputs["clean_oof"].copy()
    clean = clean[
        [
            "window_key", "group_id", "recording_id", "fold_id", "model_name", "true_class_id",
            "raw_logit", "probability_class_1", "pred_fixed", "pred_selected", "selected_threshold",
            "confidence", "selected_confidence_threshold", "ood_score", "ood_threshold", "ood_flag",
            "eligible_for_label",
        ]
    ].copy()
    clean["training_condition"] = "clean"
    clean["family"] = "clean"
    clean["requested_level"] = 0.0
    clean["achieved_rms_ratio"] = 0.0
    clean["stress_test"] = False
    augmented = training["augmented_outer_clean_oof"].copy()
    clean_ood = clean.drop_duplicates(["window_key", "fold_id"])[
        ["window_key", "fold_id", "ood_score", "ood_threshold", "ood_flag"]
    ]
    augmented = augmented.merge(clean_ood, on=["window_key", "fold_id"], how="left", validate="many_to_one")
    if augmented.ood_flag.isna().any():
        raise ValueError("augmented clean predictions do not match the frozen outer-fold OOD scores")
    augmented["training_condition"] = "augmented"
    augmented["family"] = "clean"
    augmented["requested_level"] = 0.0
    augmented["achieved_rms_ratio"] = 0.0
    augmented["stress_test"] = False
    clean["corruption_seed"] = np.uint32(0)
    augmented["corruption_seed"] = np.uint32(0)
    common = [
        "window_key", "group_id", "recording_id", "fold_id", "model_name", "true_class_id", "raw_logit",
        "probability_class_1", "pred_fixed", "pred_selected", "selected_threshold", "confidence",
        "selected_confidence_threshold", "ood_score", "ood_threshold", "ood_flag", "training_condition",
        "family", "requested_level", "achieved_rms_ratio", "stress_test",
    ]
    paired = pd.concat([clean[common], augmented[common]], ignore_index=True)
    paired.to_parquet(output_dir / "clean_vs_augmented_outer_oof_predictions.parquet", index=False)
    metrics, groups = _metric_tables(paired)
    metrics.to_csv(output_dir / "clean_vs_augmented_nested_metrics.csv", index=False)
    groups.to_csv(output_dir / "clean_vs_augmented_nested_group_metrics.csv", index=False)
    group_confusions: dict[tuple[str, str, str, float], dict[str, np.ndarray]] = {}
    for (condition, model_name), rows in paired.groupby(["training_condition", "model_name"], sort=True):
        group_confusions[(str(condition), str(model_name), "clean", 0.0)] = _group_normalized_confusion(
            rows,
            "pred_selected",
        )
    bootstrap = _paired_group_bootstrap_from_confusions(
        group_confusions,
        groups=sorted(inputs["active"].group_id.astype(str).unique()),
        replicates=int(bootstrap_replicates),
        seed=RANDOM_SEED,
    )
    bootstrap.to_csv(output_dir / "clean_vs_augmented_paired_group_bootstrap.csv", index=False)

    source_hashes = _source_hashes()
    input_paths = list(inputs["input_paths"])
    input_paths.extend(inputs["checkpoints"].values())
    input_sha256 = _hash_map(sorted(set(Path(path).resolve() for path in input_paths)))
    manifest = {
        "stage": "development_only_nested_noise_augmented_supervised_training",
        "status": "completed",
        "run_id": inputs["run_dir"].name,
        "label_rule": PRIMARY_LABEL_RULE,
        "label_rule_type": PROXY_LABEL_TYPE,
        "target_interpretation": "agreement with an operational weak-supervision proxy; not independent physiological fatigue ground truth",
        "fold_design": "historical 4 outer x 3 inner folds; exact low-frequency threshold-stage group manifest",
        "outer_group_folds": 4,
        "inner_group_folds_per_outer": 3,
        "source_group_count": 12,
        "class_support": _support(inputs["active"], "full development cohort"),
        "minimum_support_policy": "each class needs at least two distinct source groups and two group-recording trials in every fit, calibration and validation role",
        "overlapping_window_policy": "window counts are descriptive; model selection, uncertainty and independent-unit summaries are grouped by source folder",
        "models": list(MODEL_NAMES),
        "training_augmentation": {
            "clean_plus_rule_preserving_deterministic_augmented_copies": True,
            "families": list(CORRUPTION_FAMILIES),
            "levels": list(TRAINING_LEVELS),
            "only_training_fold_groups_corrupted_for_each_fit": True,
            "normalization_and_ssl_use_clean_training_groups_only": True,
            "frozen_rule_recomputed_only_to_filter_semantically_changed_augmentations": True,
            "augmented_copy_included_only_if_frozen_rule_reproduces_original_class": True,
            "clean_pseudo_labels_reassigned": False,
            "seeds": "per-window SHA-256-derived seeds; fixed family x severity schedule balanced within group and class",
        },
        "training_loss_weighting": "equal total loss mass for every source-group by proxy-class cell",
        "inner_selection": {
            "epoch_metric": "mean inner-validation group-balanced macro-F1 at fixed 0.5",
            "calibration": "group-weighted sigmoid Platt fitted on group-held-out logits from augmented models inside each training partition",
            "classifier_threshold": "maximize group-balanced Macro-F1 on calibrated inner OOF predictions; compare 0.5 against selected threshold",
            "confidence_abstention": "inner OOF group-balanced Macro-F1 with abstentions counted as class errors; ties favor coverage",
            "outer_validation_used_for_selection": False,
        },
        "checkpoint_count": int(len(training["checkpoint_manifest"])),
        "checkpoint_reload_predictions_verified": all(bool(item["checkpoint_reload_predictions_verified"]) for item in training["checkpoint_manifest"]),
        "paired_baseline": "same clean windows, labels, groups, and outer folds as the frozen clean neural stage",
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_unit": "paired source-group resampling; overlapping windows are not resampled as independent units",
        "internal_test_signal_values_parsed": 0,
        "reserved_test_member_payloads_read": 0,
        "external_rar_opened": False,
        "input_sha256": input_sha256,
        "source_sha256": source_hashes,
        "output_sha256": _output_hashes(output_dir),
    }
    manifest_path = output_dir / "augmentation_training_stage_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Completed augmented nested model training: {manifest_path}", flush=True)
    return {"status": "completed", "manifest_path": str(manifest_path), "models": list(MODEL_NAMES), "checkpoint_count": len(training["checkpoint_manifest"])}


def _run_augmented_nested_training(
    inputs: dict[str, Any],
    output_dir: Path,
    *,
    max_epochs: int,
    ssl_pretrain_epochs: int,
    confidence_candidates: Sequence[float],
) -> dict[str, Any]:
    torch.set_num_threads(1)
    fold_plan = inputs["fold_plan"]
    metadata = inputs["metadata"]
    active = inputs["active"].copy()
    signals = inputs["signals"]
    config = inputs["config"]
    augmented_by_key, augmentation_receipts, preserved_augmentation_keys = _build_training_augmentation_bank(
        metadata,
        signals,
        inputs["notch_by_window"],
        config,
    )
    augmentation_receipts.to_csv(output_dir / "training_augmentation_window_receipts.csv", index=False)
    schedule_summary = (
        augmentation_receipts.groupby(["group_id", "class_id", "family", "requested_level"], sort=True)
        .agg(
            augmented_window_count_descriptive_only=("window_key", "size"),
            median_achieved_rms_ratio=("achieved_rms_ratio", "median"),
            mean_achieved_rms_ratio=("achieved_rms_ratio", "mean"),
        )
        .reset_index()
    )
    schedule_summary.to_csv(output_dir / "training_augmentation_schedule_summary.csv", index=False)

    inner_epoch_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    inner_oof_parts: list[pd.DataFrame] = []
    calibration_receipts: list[dict[str, Any]] = []
    calibration_fit_receipts: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    outer_prediction_parts: list[pd.DataFrame] = []
    gate_rows: list[dict[str, Any]] = []
    model_to_index = {name: index for index, name in enumerate(MODEL_NAMES)}

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for outer_index, outer in enumerate(fold_plan.outer_folds):
        for model_name in MODEL_NAMES:
            outer_model_started = time.perf_counter()
            model_index = model_to_index[model_name]
            histories: dict[int, dict[str, list[Any]]] = {}
            validation_rows_by_fold: dict[int, pd.DataFrame] = {}
            training_rows_by_fold: dict[int, pd.DataFrame] = {}
            for inner in fold_plan.inner_folds[outer.fold_id]:
                train_rows = active.loc[
                    active.group_id.astype(str).isin(set(inner.train_group_ids))
                ].sort_values("window_key").reset_index(drop=True)
                valid_rows = active.loc[
                    active.group_id.astype(str).isin(set(inner.validation_group_ids))
                ].sort_values("window_key").reset_index(drop=True)
                _support(train_rows, f"augmented outer {outer.fold_id} inner {inner.fold_id} training")
                _support(valid_rows, f"augmented outer {outer.fold_id} inner {inner.fold_id} validation")
                if set(train_rows.group_id.astype(str)) & set(valid_rows.group_id.astype(str)):
                    raise RuntimeError("augmented inner train/validation groups overlap")
                seed = stable_seed(
                    "emg-aug-inner-fit-v1", fold_plan.seed, outer_index, model_index, int(inner.fold_id)
                )
                model, _, _, fit_receipt, history = _fit_augmented_model(
                    model_name,
                    train_rows,
                    metadata,
                    signals,
                    augmented_by_key,
                    preserved_augmentation_keys,
                    inner.train_group_ids,
                    epochs=int(max_epochs),
                    seed=seed,
                    ssl_pretrain_epochs=int(ssl_pretrain_epochs),
                    validation_rows=valid_rows,
                )
                histories[int(inner.fold_id)] = history
                training_rows_by_fold[int(inner.fold_id)] = train_rows
                validation_rows_by_fold[int(inner.fold_id)] = valid_rows
                for epoch_index, loss in enumerate(history["training_loss"], start=1):
                    inner_epoch_rows.append(
                        {
                            "model_name": model_name,
                            "outer_fold_id": int(outer.fold_id),
                            "inner_fold_id": int(inner.fold_id),
                            "epoch": int(epoch_index),
                            "training_loss_group_class_balanced": float(loss),
                            "validation_group_balanced_macro_f1_at_0_5": float(
                                history["validation_group_balanced_macro_f1_at_0_5"][epoch_index - 1]
                            ),
                            "training_clean_window_keys_descriptive_only": int(len(train_rows)),
                            "training_augmented_window_keys_descriptive_only": int(fit_receipt["training_augmented_window_keys_descriptive_only"]),
                            "training_groups": json.dumps(sorted(set(train_rows.group_id.astype(str))), separators=(",", ":")),
                            "normalization_ssl_source_groups": json.dumps(fit_receipt["normalization_and_ssl_source_groups"], separators=(",", ":")),
                            "validation_group_overlap": False,
                        }
                    )
            selected_epoch, epoch_selection = _select_epoch(histories, validation_rows_by_fold)
            for row in epoch_selection:
                row.update({"model_name": model_name, "outer_fold_id": int(outer.fold_id)})
            inner_epoch_rows.extend(epoch_selection)

            inner_predictions: list[pd.DataFrame] = []
            raw_oof_frames: list[pd.DataFrame] = []
            for inner in fold_plan.inner_folds[outer.fold_id]:
                inner_id = int(inner.fold_id)
                validation_rows = validation_rows_by_fold[inner_id]
                training_rows = training_rows_by_fold[inner_id]
                raw_logits = histories[inner_id]["validation_logits"][selected_epoch - 1]
                oof, calibration_receipt, fit_receipts = _augmented_inner_oof(
                    model_name=model_name,
                    outer_fold=outer,
                    inner_fold=inner,
                    train_rows=training_rows,
                    valid_rows=validation_rows,
                    valid_logits=raw_logits,
                    metadata=metadata,
                    signals=signals,
                    augmented_by_key=augmented_by_key,
                    preserved_augmentation_keys=preserved_augmentation_keys,
                    selected_epoch=selected_epoch,
                    plan_seed=fold_plan.seed,
                    model_index=model_index,
                    outer_index=outer_index,
                    ssl_pretrain_epochs=int(ssl_pretrain_epochs),
                )
                inner_predictions.append(oof)
                raw_oof_frames.append(oof[[
                    "window_key", "group_id", "recording_id", "outer_fold_id", "inner_fold_id", "true_class_id", "raw_logit"
                ]].copy())
                calibration_receipts.append(calibration_receipt)
                calibration_fit_receipts.extend(
                    {"model_name": model_name, "outer_fold_id": int(outer.fold_id), "inner_fold_id": inner_id, **row}
                    for row in fit_receipts
                )
            inner_oof = pd.concat(inner_predictions, ignore_index=True)
            if inner_oof.window_key.duplicated().any() or set(inner_oof.group_id.astype(str)) != set(outer.train_group_ids):
                raise RuntimeError("augmented calibrated inner OOF does not cover each outer-training group once")
            _support(
                inner_oof.rename(columns={"true_class_id": "class_id"}),
                f"augmented outer {outer.fold_id} inner calibrated OOF",
            )
            inner_oof["confidence"] = np.maximum(
                inner_oof.probability_class_1.to_numpy(dtype=float),
                inner_oof.probability_class_0.to_numpy(dtype=float),
            )
            threshold = select_group_balanced_threshold(
                inner_oof.true_class_id.to_numpy(dtype=int),
                inner_oof.probability_class_1.to_numpy(dtype=float),
                inner_oof.group_id.astype(str).tolist(),
            )
            confidence_frame = inner_oof.copy()
            confidence_frame["decision_threshold"] = float(threshold.threshold)
            confidence_policy = _select_confidence_threshold(confidence_frame, confidence_candidates)
            threshold_rows.append(
                {
                    "model_name": model_name,
                    "outer_fold_id": int(outer.fold_id),
                    "selected_epoch": int(selected_epoch),
                    "threshold": float(threshold.threshold),
                    "inner_oof_group_balanced_macro_f1": float(threshold.group_balanced_macro_f1),
                    "inner_oof_group_balanced_precision_class_1": float(threshold.group_balanced_precision),
                    "inner_oof_group_balanced_recall_class_1": float(threshold.group_balanced_recall),
                    "candidate_count": int(threshold.candidate_count),
                    "tie_resolution": str(threshold.tie_resolution),
                    "objective": "maximize group-balanced Macro-F1 on calibrated inner OOF probabilities",
                    "inner_oof_source_groups": json.dumps(sorted(set(inner_oof.group_id.astype(str))), separators=(",", ":")),
                    "outer_validation_used_for_selection": False,
                }
            )
            confidence_rows.append(
                {
                    "model_name": model_name,
                    "outer_fold_id": int(outer.fold_id),
                    "selected_classifier_threshold": float(threshold.threshold),
                    "selected_confidence_threshold": float(confidence_policy["threshold"]),
                    "inner_oof_group_balanced_macro_f1_with_abstention": float(
                        confidence_policy["inner_oof_group_balanced_macro_f1"]
                    ),
                    "inner_oof_coverage": float(confidence_policy["inner_oof_coverage"]),
                    "candidate_scores": json.dumps(confidence_policy["candidate_thresholds"], separators=(",", ":")),
                    "selection_objective": confidence_policy["selection_objective"],
                    "outer_validation_used_for_policy": False,
                }
            )
            inner_oof["decision_threshold"] = float(threshold.threshold)
            inner_oof["predicted_class_id"] = (
                inner_oof.probability_class_1.to_numpy(dtype=float) >= float(threshold.threshold)
            ).astype(int)
            inner_oof["model_name"] = model_name
            inner_oof_rows = inner_oof.copy()
            inner_oof_rows["selected_epoch"] = int(selected_epoch)
            inner_oof_parts.append(inner_oof_rows)

            raw_oof = pd.concat(raw_oof_frames, ignore_index=True)
            outer_calibrator, outer_calibration_receipt = _fit_platt_calibrator(
                raw_oof.raw_logit.to_numpy(dtype=float),
                raw_oof.true_class_id.to_numpy(dtype=int),
                raw_oof.group_id.astype(str).tolist(),
                raw_oof.recording_id.astype(str).tolist(),
            )
            train_rows = active.loc[
                active.group_id.astype(str).isin(set(outer.train_group_ids))
            ].sort_values("window_key").reset_index(drop=True)
            valid_rows = active.loc[
                active.group_id.astype(str).isin(set(outer.validation_group_ids))
            ].sort_values("window_key").reset_index(drop=True)
            _support(train_rows, f"augmented outer {outer.fold_id} final training")
            _support(valid_rows, f"augmented outer {outer.fold_id} final validation")
            outer_seed = stable_seed("emg-aug-outer-fit-v1", fold_plan.seed, outer_index, model_index)
            final_model, mean, std, fit_receipt, _ = _fit_augmented_model(
                model_name,
                train_rows,
                metadata,
                signals,
                augmented_by_key,
                preserved_augmentation_keys,
                outer.train_group_ids,
                epochs=selected_epoch,
                seed=outer_seed,
                ssl_pretrain_epochs=int(ssl_pretrain_epochs),
            )
            validation_x = (signals[valid_rows.bank_position.to_numpy(dtype=int)] - mean) / std
            inference_started = time.perf_counter()
            validation_logits, validation_gates = _predict_logits_chunked(final_model, validation_x)
            outer_inference_seconds = float(time.perf_counter() - inference_started)
            validation_probability = _apply_platt(outer_calibrator, validation_logits)
            ood_threshold = float(inputs["ood_receipts"][int(outer.fold_id)]["ood_threshold"])
            checkpoint_path = checkpoint_dir / f"{model_name}_outer_{int(outer.fold_id):02d}.pt"
            checkpoint_metadata = {
                "model_name": model_name,
                "outer_fold_id": int(outer.fold_id),
                "seed": int(outer_seed),
                "optimizer": "AdamW",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "selected_supervised_epochs": int(selected_epoch),
                "outer_model_pipeline_training_seconds_before_checkpoint": float(time.perf_counter() - outer_model_started),
                "outer_validation_inference_seconds": outer_inference_seconds,
                "outer_validation_inference_milliseconds_per_window": (
                    1000.0 * outer_inference_seconds / len(valid_rows) if len(valid_rows) else None
                ),
                "label_rule": PRIMARY_LABEL_RULE,
                "label_rule_type": PROXY_LABEL_TYPE,
                "fixed_threshold": 0.5,
                "inner_oof_selected_threshold": float(threshold.threshold),
                "inner_oof_selected_confidence_threshold": float(confidence_policy["threshold"]),
                "outer_group_calibrated_ood_threshold": ood_threshold,
                "training_source_groups": list(outer.train_group_ids),
                "held_out_source_groups": list(outer.validation_group_ids),
                "normalization_and_pretraining_receipt": fit_receipt,
                "outer_calibration_receipt": outer_calibration_receipt,
                "outer_calibrator_coefficient": float(outer_calibrator.coef_[0, 0]),
                "outer_calibrator_intercept": float(outer_calibrator.intercept_[0]),
                "augmentation_training": {
                    "clean_plus_rule_preserving_augmented_training_copies": True,
                    "families": list(CORRUPTION_FAMILIES),
                    "levels": list(TRAINING_LEVELS),
                    "augmentation_scope": "current training groups only",
                    "normalization_scope": "clean training-group windows only",
                    "pseudo_labels_recomputed_for_consistency_filter_only": True,
                    "augmented_copy_included_only_if_frozen_rule_reproduces_original_class": True,
                    "clean_pseudo_label_reassigned": False,
                },
                "gate_balance_weight": 0.05 if model_name == "latent_moe3" else 0.0,
                "expert_semantics": "latent experts share one binary proxy target" if model_name == "latent_moe3" else None,
            }
            if checkpoint_path.exists():
                raise FileExistsError(f"refusing to overwrite augmented checkpoint: {checkpoint_path.name}")
            torch.save(
                {
                    "model_state_dict": {name: value.detach().cpu() for name, value in final_model.state_dict().items()},
                    "metadata": checkpoint_metadata,
                },
                checkpoint_path,
            )
            checkpoint_sha = _sha256_file(checkpoint_path)
            restored, restored_meta = _checkpoint_model(checkpoint_path, model_name)
            restored_logits, restored_gates = _predict_logits_chunked(restored, validation_x)
            if not np.allclose(restored_logits, validation_logits, rtol=1e-5, atol=1e-6):
                raise RuntimeError("augmented checkpoint reload predictions differ from pre-save predictions")
            checkpoint_rows.append(
                {
                    "path": str(checkpoint_path.resolve()),
                    "sha256": checkpoint_sha,
                    "model_name": model_name,
                    "outer_fold_id": int(outer.fold_id),
                    "selected_supervised_epochs": int(selected_epoch),
                    "outer_model_pipeline_training_seconds_before_checkpoint": checkpoint_metadata["outer_model_pipeline_training_seconds_before_checkpoint"],
                    "outer_validation_inference_seconds": outer_inference_seconds,
                    "outer_validation_inference_milliseconds_per_window": checkpoint_metadata["outer_validation_inference_milliseconds_per_window"],
                    "training_source_groups": list(outer.train_group_ids),
                    "held_out_source_groups": list(outer.validation_group_ids),
                    "checkpoint_reload_predictions_verified": True,
                }
            )
            outer_prediction_parts.append(
                pd.DataFrame(
                    {
                        "window_key": valid_rows.window_key.astype(str).to_numpy(),
                        "group_id": valid_rows.group_id.astype(str).to_numpy(),
                        "recording_id": valid_rows.recording_id.astype(str).to_numpy(),
                        "fold_id": int(outer.fold_id),
                        "model_name": model_name,
                        "true_class_id": valid_rows.class_id.to_numpy(dtype=int),
                        "raw_logit": restored_logits,
                        "probability_class_1": _apply_platt(outer_calibrator, restored_logits),
                        "probability_class_0": 1.0 - _apply_platt(outer_calibrator, restored_logits),
                        "pred_fixed": (_apply_platt(outer_calibrator, restored_logits) >= 0.5).astype(int),
                        "pred_selected": (
                            _apply_platt(outer_calibrator, restored_logits) >= float(threshold.threshold)
                        ).astype(int),
                        "selected_threshold": float(threshold.threshold),
                        "selected_confidence_threshold": float(confidence_policy["threshold"]),
                        "confidence": np.maximum(
                            _apply_platt(outer_calibrator, restored_logits),
                            1.0 - _apply_platt(outer_calibrator, restored_logits),
                        ),
                        "checkpoint_sha256": checkpoint_sha,
                    }
                )
            )
            gate_row = _gate_receipt(
                restored_gates,
                model_name,
                training_condition="augmented",
                family="clean",
                requested_level=0.0,
                outer_fold_id=int(outer.fold_id),
            )
            if gate_row is not None:
                gate_rows.append(gate_row)
            print(
                f"Augmented nested fit complete: outer={outer.fold_id}/4 model={model_name} "
                f"selected_epoch={selected_epoch} threshold={float(threshold.threshold):.4f}",
                flush=True,
            )

    inner_oof = pd.concat(inner_oof_parts, ignore_index=True)
    outer_augmented = pd.concat(outer_prediction_parts, ignore_index=True)
    if inner_oof.duplicated(["model_name", "outer_fold_id", "window_key"]).any():
        raise RuntimeError("augmented inner OOF contains duplicate model-fold-window keys")
    if outer_augmented.duplicated(["model_name", "fold_id", "window_key"]).any():
        raise RuntimeError("augmented outer OOF contains duplicate model-fold-window keys")
    if len(outer_augmented) != len(inputs["active"]) * len(MODEL_NAMES):
        raise RuntimeError("augmented outer predictions do not cover every assigned window for each model")
    inner_oof.to_parquet(output_dir / "augmented_inner_oof_predictions.parquet", index=False)
    outer_augmented.to_parquet(output_dir / "augmented_outer_clean_oof_predictions.parquet", index=False)
    pd.DataFrame(inner_epoch_rows).to_csv(output_dir / "augmented_inner_epoch_selection.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "augmented_inner_threshold_selection.csv", index=False)
    pd.DataFrame(confidence_rows).to_csv(output_dir / "augmented_inner_confidence_selection.csv", index=False)
    pd.DataFrame(calibration_receipts).to_json(
        output_dir / "augmented_calibration_receipts.json", orient="records", indent=2
    )
    pd.DataFrame(calibration_fit_receipts).to_json(
        output_dir / "augmented_calibration_crossfit_receipts.json", orient="records", indent=2
    )
    pd.DataFrame(checkpoint_rows).to_json(output_dir / "augmented_checkpoint_manifest.json", orient="records", indent=2)
    pd.DataFrame(checkpoint_rows)[
        [
            "model_name",
            "outer_fold_id",
            "selected_supervised_epochs",
            "outer_model_pipeline_training_seconds_before_checkpoint",
            "outer_validation_inference_seconds",
            "outer_validation_inference_milliseconds_per_window",
        ]
    ].to_csv(output_dir / "augmented_training_and_inference_timing.csv", index=False)
    if gate_rows:
        pd.DataFrame(gate_rows).to_csv(output_dir / "augmented_moe_gate_clean_diagnostics.csv", index=False)
    return {
        "augmented_inner_oof": inner_oof,
        "augmented_outer_clean_oof": outer_augmented,
        "checkpoint_manifest": checkpoint_rows,
        "checkpoint_directory": checkpoint_dir,
        "training_augmentation_receipts": augmentation_receipts,
        "schedule_summary": schedule_summary,
        "gate_rows": gate_rows,
        "confidence_candidates": tuple(float(value) for value in confidence_candidates),
        "max_epochs": int(max_epochs),
        "ssl_pretrain_epochs": int(ssl_pretrain_epochs),
    }
