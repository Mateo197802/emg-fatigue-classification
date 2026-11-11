"""Direct raw-signal CNN for participant-grouped Cerqueira supervision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F

from .external_supervision import (
    CLASS_IDS,
    _apply_temperature,
    _fit_temperature,
    _group_class_weights,
    _group_balanced_macro_f1,
    _metrics,
    _paired_group_bootstrap,
    _require_partition_support,
)
from .external_provenance import code_provenance


LEARNING_RATES = (1e-3, 3e-4)
EPOCH_CANDIDATES = (2, 4, 6, 8)
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4


class CerqueiraCNN1D(nn.Module):
    """Small four-channel temporal CNN with a three-class head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=17, stride=4, padding=8, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(32, 3)

    def forward(self, signal_window: torch.Tensor) -> torch.Tensor:
        if signal_window.ndim != 3 or signal_window.shape[1] != 4:
            raise ValueError("CerqueiraCNN1D expects (batch, 4 channels, time samples)")
        return self.classifier(self.encoder(signal_window))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _fit_channel_standardizer(bank: np.ndarray, indices: np.ndarray, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(4, dtype=np.float64)
    total_square = np.zeros(4, dtype=np.float64)
    selected = frame.iloc[indices].reset_index(drop=True)
    weights = _group_class_weights(selected, selected.label.to_numpy(dtype=np.int64)).astype(np.float64)
    weights /= weights.sum()
    sample_count = int(bank.shape[-1])
    for start in range(0, len(indices), 64):
        batch = np.asarray(bank[indices[start : start + 64]], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError("training signal bank contains non-finite values")
        batch_weights = weights[start : start + len(batch)]
        total += (batch * batch_weights[:, None, None]).sum(axis=(0, 2))
        total_square += (np.square(batch) * batch_weights[:, None, None]).sum(axis=(0, 2))
    mean = total / max(sample_count, 1)
    variance = total_square / max(sample_count, 1) - np.square(mean)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float32), scale.astype(np.float32)


def _predict(
    model: CerqueiraCNN1D,
    bank: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = np.asarray(bank[indices[start : start + batch_size]], dtype=np.float32)
            batch = (batch - mean[None, :, None]) / scale[None, :, None]
            logits = model(torch.from_numpy(np.ascontiguousarray(batch)))
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, len(CLASS_IDS)), dtype=np.float64)


def _train_through_epochs(
    bank: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    frame: pd.DataFrame,
    *,
    learning_rate: float,
    max_epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], CerqueiraCNN1D, np.ndarray, np.ndarray]:
    mean, scale = _fit_channel_standardizer(bank, train_indices, frame)
    local_train = frame.iloc[train_indices].reset_index(drop=True)
    train_y = local_train.label.to_numpy(dtype=np.int64)
    train_weight = _group_class_weights(local_train, train_y).astype(np.float32)
    torch.manual_seed(int(seed))
    model = CerqueiraCNN1D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(int(seed) + 1)
    validation_y = frame.iloc[validation_indices].label.to_numpy(dtype=np.int64)
    validation_groups = frame.iloc[validation_indices].subject_id.astype(str).tolist()
    predictions_by_epoch: dict[int, np.ndarray] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        permutation = rng.permutation(len(train_indices))
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            batch_pos = permutation[start : start + BATCH_SIZE]
            batch_indices = train_indices[batch_pos]
            batch = np.asarray(bank[batch_indices], dtype=np.float32)
            batch = (batch - mean[None, :, None]) / scale[None, :, None]
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            target = torch.from_numpy(train_y[batch_pos]).long().to(device)
            weights = torch.from_numpy(train_weight[batch_pos]).to(device)
            logits = model(tensor)
            row_loss = F.cross_entropy(logits, target, reduction="none")
            loss = torch.mean(row_loss * weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Cerqueira CNN training loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_indices)
            seen += int(len(batch_indices))
        probabilities = _predict(model, bank, validation_indices, mean, scale)
        predictions_by_epoch[epoch] = probabilities
        macro_f1 = _group_balanced_macro_f1(validation_y, np.argmax(probabilities, axis=1), validation_groups)
        history.append({
            "epoch": epoch,
            "training_group_class_trial_weighted_loss": loss_sum / max(seen, 1),
            "validation_group_balanced_macro_f1": macro_f1,
            "validation_window_count_descriptive_only": int(len(validation_indices)),
            "validation_participant_count": int(len(set(validation_groups))),
        })
    return predictions_by_epoch, history, model, mean, scale


def _fit_final_model(
    bank: np.ndarray,
    train_indices: np.ndarray,
    frame: pd.DataFrame,
    *,
    learning_rate: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[CerqueiraCNN1D, np.ndarray, np.ndarray, list[float], float]:
    mean, scale = _fit_channel_standardizer(bank, train_indices, frame)
    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    y = train_frame.label.to_numpy(dtype=np.int64)
    weights = _group_class_weights(train_frame, y).astype(np.float32)
    torch.manual_seed(int(seed))
    model = CerqueiraCNN1D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(int(seed) + 1)
    loss_history: list[float] = []
    started = perf_counter()
    for _ in range(int(epochs)):
        model.train()
        permutation = rng.permutation(len(train_indices))
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            batch_pos = permutation[start : start + BATCH_SIZE]
            batch = np.asarray(bank[train_indices[batch_pos]], dtype=np.float32)
            batch = (batch - mean[None, :, None]) / scale[None, :, None]
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            target = torch.from_numpy(y[batch_pos]).long().to(device)
            sample_weight = torch.from_numpy(weights[batch_pos]).to(device)
            logits = model(tensor)
            loss = torch.mean(F.cross_entropy(logits, target, reduction="none") * sample_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Cerqueira final CNN loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_pos)
            seen += int(len(batch_pos))
        loss_history.append(loss_sum / max(seen, 1))
    return model, mean, scale, loss_history, float(perf_counter() - started)


def run_cerqueira_cnn_stage(
    *,
    feature_path: Path,
    signal_path: Path,
    baseline_predictions_path: Path,
    output_dir: Path,
    seed: int = 731,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Nested 4x3 raw CNN with inner-OOF learning-rate/epoch selection."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite external CNN stage: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(feature_path).reset_index(drop=True)
    baseline = pd.read_parquet(baseline_predictions_path)
    bank = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    if bank.ndim != 3 or bank.shape[0] != len(frame) or bank.shape[1] != 4:
        raise ValueError("raw signal bank is not aligned with four-channel Cerqueira features")
    if frame.subject_id.nunique() != 13:
        raise ValueError("Cerqueira CNN requires all 13 independent participants")
    if not np.array_equal(frame.sample_id.astype(str).to_numpy(), np.asarray([f"CERQ_{i:06d}" for i in range(len(frame))])):
        raise ValueError("signal bank order cannot be verified against feature sample IDs")
    baseline = baseline[baseline.model_name.isin(("signal_logistic", "time_only_logistic", "trial_condition_time_logistic"))].copy()
    expected_baselines = {"signal_logistic", "time_only_logistic", "trial_condition_time_logistic"}
    if set(baseline.model_name.astype(str)) != expected_baselines:
        raise ValueError("nested logistic and control predictions are required for paired comparison")
    for _, rows in baseline.groupby("model_name"):
        if set(rows.sample_id.astype(str)) != set(frame.sample_id.astype(str)) or len(rows) != len(frame):
            raise ValueError("baseline and CNN do not use identical windows")
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
            raise ValueError("baseline and CNN group, trial or label identities differ")

    labels = frame.label.to_numpy(dtype=np.int64)
    groups = frame.subject_id.astype(str).to_numpy()
    outer_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=int(seed))
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cpu")
    history_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    final_selection_rows: list[dict[str, Any]] = []
    outer_oof_probability: list[np.ndarray] = []

    for outer_fold, (outer_train_index, outer_validation_index) in enumerate(outer_cv.split(np.zeros(len(frame)), labels, groups), start=1):
        outer_train = frame.iloc[outer_train_index].reset_index(drop=True)
        outer_validation = frame.iloc[outer_validation_index].reset_index(drop=True)
        train_support = _require_partition_support(outer_train, f"CNN outer {outer_fold} training")
        validation_support = _require_partition_support(outer_validation, f"CNN outer {outer_fold} validation")
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=int(seed + outer_fold))
        inner_splits = list(inner_cv.split(outer_train, outer_train.label, outer_train.subject_id))
        inner_oof: dict[tuple[float, int], np.ndarray] = {
            (float(lr), int(epoch)): np.full((len(outer_train), len(CLASS_IDS)), np.nan, dtype=np.float64)
            for lr in LEARNING_RATES for epoch in EPOCH_CANDIDATES
        }
        per_candidate_scores = {key: [] for key in inner_oof}
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(inner_splits, start=1):
            inner_train = outer_train.iloc[inner_train_local].reset_index(drop=True)
            inner_validation = outer_train.iloc[inner_validation_local].reset_index(drop=True)
            _require_partition_support(inner_train, f"CNN outer {outer_fold} inner {inner_fold} training")
            _require_partition_support(inner_validation, f"CNN outer {outer_fold} inner {inner_fold} validation")
            if set(inner_train.subject_id.astype(str)) & set(inner_validation.subject_id.astype(str)):
                raise RuntimeError("CNN inner train/validation participant overlap")
            absolute_train = outer_train_index[inner_train_local]
            absolute_validation = outer_train_index[inner_validation_local]
            inner_y = inner_validation.label.to_numpy(dtype=np.int64)
            inner_groups = inner_validation.subject_id.astype(str).tolist()
            for lr_index, learning_rate in enumerate(LEARNING_RATES):
                run_started = perf_counter()
                prediction_by_epoch, history, _, _, _ = _train_through_epochs(
                    bank,
                    absolute_train,
                    absolute_validation,
                    frame,
                    learning_rate=learning_rate,
                    max_epochs=max(EPOCH_CANDIDATES),
                    seed=int(seed + outer_fold * 100000 + inner_fold * 1000 + lr_index * 100),
                    device=device,
                )
                for row in history:
                    history_rows.append({
                        "outer_fold_id": outer_fold,
                        "inner_fold_id": inner_fold,
                        "learning_rate": learning_rate,
                        "weight_decay": WEIGHT_DECAY,
                        "batch_size": BATCH_SIZE,
                        **row,
                    })
                    if int(row["epoch"]) in EPOCH_CANDIDATES:
                        key = (float(learning_rate), int(row["epoch"]))
                        inner_oof[key][inner_validation_local] = prediction_by_epoch[int(row["epoch"])]
                        per_candidate_scores[key].append(float(row["validation_group_balanced_macro_f1"]))
                candidate_rows.append({
                    "outer_fold_id": outer_fold,
                    "inner_fold_id": inner_fold,
                    "learning_rate": learning_rate,
                    "training_groups": json.dumps(sorted(inner_train.subject_id.astype(str).unique())),
                    "validation_groups": json.dumps(sorted(inner_validation.subject_id.astype(str).unique())),
                    "training_support": json.dumps(_require_partition_support(inner_train, "CNN support receipt"), separators=(",", ":")),
                    "validation_support": json.dumps(_require_partition_support(inner_validation, "CNN support receipt"), separators=(",", ":")),
                    "training_runtime_seconds": float(perf_counter() - run_started),
                })

        scored = []
        for (learning_rate, epoch), probabilities in inner_oof.items():
            if not np.isfinite(probabilities).all():
                raise RuntimeError(f"CNN outer {outer_fold} inner OOF incomplete for lr={learning_rate}, epoch={epoch}")
            score = _group_balanced_macro_f1(
                outer_train.label.to_numpy(dtype=np.int64),
                np.argmax(probabilities, axis=1),
                outer_train.subject_id.astype(str).tolist(),
            )
            scored.append((score, learning_rate, epoch, probabilities))
            final_selection_rows.append({
                "outer_fold_id": outer_fold,
                "learning_rate": learning_rate,
                "epochs": epoch,
                "inner_oof_group_balanced_macro_f1": score,
                "candidate_inner_fold_mean_validation_macro_f1": float(np.mean(per_candidate_scores[(learning_rate, epoch)])),
            })
        selected_score, selected_lr, selected_epoch, selected_oof = max(scored, key=lambda value: value[0])
        temperature = _fit_temperature(selected_oof, outer_train.label.to_numpy(dtype=np.int64), outer_train.subject_id.astype(str).tolist())
        train_absolute = outer_train_index
        valid_absolute = outer_validation_index
        final_model, mean, scale, loss_history, fit_seconds = _fit_final_model(
            bank,
            train_absolute,
            frame,
            learning_rate=selected_lr,
            epochs=selected_epoch,
            seed=int(seed + outer_fold * 100000 + 90000),
            device=device,
        )
        predict_started = perf_counter()
        raw_probability = _predict(final_model, bank, valid_absolute, mean, scale)
        probabilities = _apply_temperature(raw_probability, temperature)
        predict_seconds = perf_counter() - predict_started
        outer_metrics = _metrics(outer_validation.label.to_numpy(dtype=np.int64), probabilities, outer_validation.subject_id.astype(str).tolist())
        for row in outer_metrics.pop("per_group"):
            group_rows.append({"outer_fold_id": outer_fold, "model_name": "cnn1d_supervised", **row})
        fold_rows.append({
            **outer_metrics,
            "outer_fold_id": outer_fold,
            "selected_learning_rate": selected_lr,
            "selected_epochs": selected_epoch,
            "inner_oof_group_balanced_macro_f1": selected_score,
            "inner_oof_temperature": temperature,
            "outer_fit_time_seconds": fit_seconds,
            "outer_prediction_time_seconds": predict_seconds,
            "training_loss_by_epoch": json.dumps(loss_history),
            "outer_train_support_json": json.dumps(train_support, separators=(",", ":")),
            "outer_validation_support_json": json.dumps(validation_support, separators=(",", ":")),
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in final_model.parameters() if parameter.requires_grad)),
        })
        outer_oof_probability.append(probabilities)
        for row_index, (_, record) in enumerate(outer_validation.iterrows()):
            oof_rows.append({
                "sample_id": record.sample_id,
                "subject_id": record.subject_id,
                "recording_id": record.recording_id,
                "outer_fold_id": outer_fold,
                "model_name": "cnn1d_supervised",
                "true_label": int(record.label),
                "prediction": int(np.argmax(probabilities[row_index])),
                **{f"probability_class_{class_id}": float(probabilities[row_index, class_id]) for class_id in CLASS_IDS},
                "selected_learning_rate": selected_lr,
                "selected_epochs": selected_epoch,
                "inner_oof_temperature": temperature,
            })
        torch.save({
            "state_dict": final_model.cpu().state_dict(),
            "architecture": "CerqueiraCNN1D(4 channels, 16-32-32 conv, global average pool, 32 embedding, 3-class head)",
            "channel_mean_training_only": mean,
            "channel_scale_training_only": scale,
            "class_ids": CLASS_IDS,
            "class_names": {0: "NO_FATIGUE", 1: "TRANSITION", 2: "FATIGUE"},
            "selected_learning_rate": selected_lr,
            "selected_epochs": selected_epoch,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "temperature_inner_oof": temperature,
            "outer_fold_id": outer_fold,
            "training_participants": sorted(outer_train.subject_id.astype(str).unique()),
            "validation_participants": sorted(outer_validation.subject_id.astype(str).unique()),
            "label_rule": "cerqueira_v2_self_perceived_fatigue_state_0_1_2",
            "loss_history": loss_history,
        }, output_dir / f"cerqueira_cnn1d_outer_fold_{outer_fold}.pt")
        print(f"Completed Cerqueira CNN outer fold {outer_fold}/4; lr={selected_lr:g}; epochs={selected_epoch}; group Macro-F1={outer_metrics['group_macro_f1']:.3f}", flush=True)

    predictions = pd.DataFrame(oof_rows)
    group_metrics = pd.DataFrame(group_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    predictions.to_parquet(output_dir / "cerqueira_cnn1d_nested_oof_predictions.parquet", index=False)
    group_metrics.to_csv(output_dir / "cerqueira_cnn1d_group_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / "cerqueira_cnn1d_outer_fold_metrics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_dir / "cerqueira_cnn1d_inner_training_history.csv", index=False)
    pd.DataFrame(final_selection_rows).to_csv(output_dir / "cerqueira_cnn1d_inner_oof_epoch_selection.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output_dir / "cerqueira_cnn1d_inner_fold_receipts.csv", index=False)

    baseline_groups: dict[str, pd.DataFrame] = {}
    for baseline_name in ("signal_logistic", "time_only_logistic", "trial_condition_time_logistic"):
        rows = baseline[baseline.model_name.eq(baseline_name)]
        probabilities = rows[[f"probability_class_{class_id}" for class_id in CLASS_IDS]].to_numpy(dtype=np.float64)
        baseline_groups[baseline_name] = pd.DataFrame(_metrics(rows.true_label.to_numpy(dtype=np.int64), probabilities, rows.subject_id.astype(str).tolist())["per_group"])
    paired_rows = []
    cnn_groups = group_metrics
    for baseline_name, baseline_group_rows in baseline_groups.items():
        for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
            paired_rows.append({
                "comparison": f"cnn1d_supervised_minus_{baseline_name}",
                **_paired_group_bootstrap(cnn_groups, baseline_group_rows, metric=metric, replicates=bootstrap_replicates, seed=seed + len(paired_rows)),
            })
    pd.DataFrame(paired_rows).to_csv(output_dir / "cerqueira_cnn1d_paired_group_bootstrap.csv", index=False)
    summary = _metrics(
        predictions.true_label.to_numpy(dtype=np.int64),
        predictions[[f"probability_class_{class_id}" for class_id in CLASS_IDS]].to_numpy(dtype=np.float64),
        predictions.subject_id.astype(str).tolist(),
    )
    summary.pop("per_group", None)
    _write_json(output_dir / "cerqueira_cnn1d_metrics_summary.json", summary)

    candidate_frame = pd.DataFrame(final_selection_rows)
    final_ranking = candidate_frame.groupby(["learning_rate", "epochs"], as_index=False).agg(
        mean_inner_oof_group_balanced_macro_f1=("inner_oof_group_balanced_macro_f1", "mean")
    ).sort_values(["mean_inner_oof_group_balanced_macro_f1", "learning_rate", "epochs"], ascending=[False, True, True])
    final_lr = float(final_ranking.iloc[0].learning_rate)
    final_epochs = int(final_ranking.iloc[0].epochs)
    final_model, final_mean, final_scale, final_loss_history, final_fit_seconds = _fit_final_model(
        bank,
        np.arange(len(frame)),
        frame,
        learning_rate=final_lr,
        epochs=final_epochs,
        seed=int(seed + 900000),
        device=device,
    )
    oof_probability = predictions[[f"probability_class_{class_id}" for class_id in CLASS_IDS]].to_numpy(dtype=np.float64)
    final_temperature = _fit_temperature(oof_probability, predictions.true_label.to_numpy(dtype=np.int64), predictions.subject_id.astype(str).tolist())
    torch.save({
        "state_dict": final_model.cpu().state_dict(),
        "architecture": "CerqueiraCNN1D(4 channels, 16-32-32 conv, global average pool, 32 embedding, 3-class head)",
        "channel_mean_training_only": final_mean,
        "channel_scale_training_only": final_scale,
        "class_ids": CLASS_IDS,
        "class_names": {0: "NO_FATIGUE", 1: "TRANSITION", 2: "FATIGUE"},
        "selected_learning_rate": final_lr,
        "selected_epochs": final_epochs,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "temperature_outer_oof": final_temperature,
        "training_participants": sorted(frame.subject_id.astype(str).unique()),
        "label_rule": "cerqueira_v2_self_perceived_fatigue_state_0_1_2",
        "calibration_source": "outer participant OOF predictions; every participant held out once",
        "training_loss_by_epoch": final_loss_history,
        "training_time_seconds": final_fit_seconds,
    }, output_dir / "cerqueira_cnn1d_final_development.pt")

    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir(), key=lambda value: value.name.casefold())
        if path.is_file()
    }
    manifest = {
        "status": "completed",
        "dataset_key": "cerqueira_v2",
        "target": "source-reported self-perceived level 0/1/2; not physiological fatigue ground truth",
        "independent_unit": "participant",
        "outer_folds": 4,
        "inner_folds": 3,
        "model": "four-channel raw 1D CNN with a three-class softmax head",
        "architecture": "Conv1d(4,16,k17,s4) -> Conv1d(16,32,k9,s4) -> Conv1d(32,32,k5,s2) -> global average pool -> 32-d embedding -> 3 logits",
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in final_model.parameters() if parameter.requires_grad)),
        "input_signal_shape": list(bank.shape),
        "preprocessing_fit_only_on_training_groups": "per-channel mean and standard deviation",
        "loss": "sample-weighted cross entropy with equal participant, class, trial and within-trial window mass",
        "optimizer": "AdamW",
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "learning_rate_candidates": list(LEARNING_RATES),
        "epoch_candidates": list(EPOCH_CANDIDATES),
        "selection": "group-balanced Macro-F1 from concatenated inner-fold held-out probabilities",
        "calibration": "participant/class-balanced temperature scaling fit on inner OOF per outer fold; final temperature from outer OOF",
        "feature_bank_sha256": _sha256(feature_path),
        "signal_bank_sha256": _sha256(signal_path),
        "baseline_predictions_sha256": _sha256(baseline_predictions_path),
        "window_count_is_independent_n": False,
        "locked_course_internal_test_values_parsed": 0,
        "course_raw2_rar_opened": False,
        "code_provenance": code_provenance(__file__, Path(__file__).with_name("external_supervision.py")),
        "artifact_sha256": artifacts,
    }
    _write_json(output_dir / "cerqueira_cnn1d_stage_manifest.json", manifest)
    return {"status": "completed", "manifest_path": str((output_dir / "cerqueira_cnn1d_stage_manifest.json").resolve()), "metrics": summary}
