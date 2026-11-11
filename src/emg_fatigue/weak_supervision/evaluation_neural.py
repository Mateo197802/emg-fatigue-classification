"""Nested source-group evaluation of compact supervised raw-signal models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from .group_cv import FoldPlan, GroupFold
from .models_neural import (
    BinarySignalClassifier,
    SignalEncoder,
    SoftGatedExperts,
    gate_load_balance_penalty,
    normalized_gate_entropy,
)
from .ssl_pretraining import pretrain_encoder
from .external_transfer import initialize_signal_encoder_from_cerqueira


MODEL_NAMES = ("cnn1d_supervised", "ssl_encoder_finetuned", "latent_moe3")
EXTENDED_MODEL_NAMES = ("ssl_encoder_frozen", "latent_moe2")
EXTERNAL_TRANSFER_MODEL_NAMES = ("cerqueira_supervised_finetuned",)
AVAILABLE_MODEL_NAMES = MODEL_NAMES + EXTENDED_MODEL_NAMES + EXTERNAL_TRANSFER_MODEL_NAMES
GATE_BALANCE_WEIGHT = 0.05
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def select_fold_pretraining_indices(
    metadata: pd.DataFrame,
    training_group_ids: Sequence[str],
    *,
    per_trial_limit: int = 16,
    per_group_limit: int = 64,
) -> np.ndarray:
    """Select eligible even-index windows from training groups, limiting redundancy.

    Returned integers are positional indices into ``metadata`` and the aligned
    signal bank. With 500 ms windows sampled every 250 ms, even-index windows
    have a 500 ms start-to-start hop and do not overlap.
    Sampling is deterministic and spread through each recording.
    """
    required = {"group_id", "recording_id", "window_index", "eligible_for_label"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"pretraining metadata is missing columns: {sorted(required - set(metadata.columns))}")
    trial_limit = int(per_trial_limit)
    group_limit = int(per_group_limit)
    if trial_limit < 1 or group_limit < 1:
        raise ValueError("pretraining sample limits must be positive")
    groups = {str(value) for value in training_group_ids}
    if not groups:
        raise ValueError("training_group_ids cannot be empty")
    group_column = metadata.group_id.astype(str).to_numpy()
    recording_column = metadata.recording_id.astype(str).to_numpy()
    segment_column = (
        metadata.segment_index.astype(int).to_numpy()
        if "segment_index" in metadata
        else np.zeros(len(metadata), dtype=np.int64)
    )
    indices = np.arange(len(metadata), dtype=np.int64)
    eligible = metadata.eligible_for_label.fillna(False).astype(bool).to_numpy()
    even = metadata.window_index.astype(int).mod(2).eq(0).to_numpy()
    candidates = indices[np.isin(group_column, tuple(groups)) & eligible & even]

    def evenly_spaced(values: np.ndarray, limit: int) -> np.ndarray:
        if values.size <= limit:
            return values
        positions = np.linspace(0, values.size - 1, num=limit, dtype=np.int64)
        return values[np.unique(positions)]

    per_trial: list[int] = []
    for group_id in sorted(groups):
        group_candidates = candidates[group_column[candidates] == group_id]
        trial_groups: dict[str, list[int]] = {}
        for position in group_candidates.tolist():
            trial_groups.setdefault(recording_column[position], []).append(position)
        for recording_id in sorted(trial_groups):
            positions = np.asarray(trial_groups[recording_id], dtype=np.int64)
            order = np.lexsort(
                (
                    metadata.iloc[positions].window_index.to_numpy(dtype=int),
                    segment_column[positions],
                )
            )
            per_trial.extend(evenly_spaced(positions[order], trial_limit).tolist())

    selected_by_group: list[int] = []
    selected = np.asarray(per_trial, dtype=np.int64)
    for group_id in sorted(groups):
        group_positions = selected[group_column[selected] == group_id]
        group_positions = np.sort(group_positions)
        selected_by_group.extend(evenly_spaced(group_positions, group_limit).tolist())
    result = np.asarray(sorted(set(selected_by_group)), dtype=np.int64)
    if result.size == 0:
        raise ValueError("No eligible non-overlapping raw windows exist in the supplied training groups")
    if not set(group_column[result]).issubset(groups):
        raise RuntimeError("Pretraining selector admitted a non-training source group")
    if not eligible[result].all() or not (metadata.iloc[result].window_index.astype(int).mod(2).eq(0)).all():
        raise RuntimeError("Pretraining selector admitted an ineligible or overlapping window")
    return result


def audit_group_fold_support(
    scored_metadata: pd.DataFrame,
    fold_plan: FoldPlan,
    *,
    min_groups_per_class: int = 2,
    min_trials_per_class: int = 2,
) -> pd.DataFrame:
    """Audit class support by source group and trial in every nested role."""
    required = {"group_id", "recording_id", "class_id"}
    if not required.issubset(scored_metadata.columns):
        raise ValueError(f"support metadata is missing columns: {sorted(required - set(scored_metadata.columns))}")
    rows = scored_metadata[scored_metadata.class_id.isin((0, 1))].copy()
    rows["group_id"] = rows.group_id.astype(str)
    rows["recording_id"] = rows.recording_id.astype(str)
    rows["class_id"] = rows.class_id.astype(int)
    minima_groups = int(min_groups_per_class)
    minima_trials = int(min_trials_per_class)
    if minima_groups < 1 or minima_trials < 1:
        raise ValueError("class support minima must be positive")

    def disjoint(left: Sequence[str], right: Sequence[str], label: str) -> None:
        if set(left).intersection(right):
            raise ValueError(f"{label} training and validation source groups overlap")

    support_rows: list[dict[str, Any]] = []
    issues: list[str] = []

    def add_role(
        partition: str,
        outer_id: int,
        inner_id: int | None,
        group_ids: Sequence[str],
    ) -> None:
        selected = rows[rows.group_id.isin({str(value) for value in group_ids})]
        for class_id in (0, 1):
            class_rows = selected[selected.class_id.eq(class_id)]
            group_count = int(class_rows.group_id.nunique())
            trial_count = int(class_rows[["group_id", "recording_id"]].drop_duplicates().shape[0])
            support_rows.append(
                {
                    "partition": partition,
                    "outer_fold_id": int(outer_id),
                    "inner_fold_id": None if inner_id is None else int(inner_id),
                    "class_id": class_id,
                    "assigned_window_count_descriptive_only": int(len(class_rows)),
                    "distinct_source_groups": group_count,
                    "distinct_recording_trials": trial_count,
                    "minimum_source_groups": minima_groups,
                    "minimum_recording_trials": minima_trials,
                    "passed": group_count >= minima_groups and trial_count >= minima_trials,
                }
            )
            if group_count < minima_groups or trial_count < minima_trials:
                issues.append(
                    f"{partition} outer={outer_id} inner={inner_id} class={class_id}: "
                    f"groups={group_count}, trials={trial_count}"
                )

    all_groups = set(str(value) for value in fold_plan.development_group_ids)
    for outer in fold_plan.outer_folds:
        disjoint(outer.train_group_ids, outer.validation_group_ids, f"outer fold {outer.fold_id}")
        if set(outer.train_group_ids).union(outer.validation_group_ids) != all_groups:
            raise ValueError(f"outer fold {outer.fold_id} does not partition all development groups")
        add_role("outer_training", outer.fold_id, None, outer.train_group_ids)
        add_role("outer_validation", outer.fold_id, None, outer.validation_group_ids)
        inner_folds = fold_plan.inner_folds.get(outer.fold_id, ())
        if not inner_folds:
            raise ValueError(f"outer fold {outer.fold_id} has no inner folds")
        for inner in inner_folds:
            disjoint(inner.train_group_ids, inner.validation_group_ids, f"inner fold {outer.fold_id}/{inner.fold_id}")
            if set(inner.train_group_ids).union(inner.validation_group_ids) != set(outer.train_group_ids):
                raise ValueError(f"inner fold {outer.fold_id}/{inner.fold_id} does not partition outer training groups")
            add_role("inner_training", outer.fold_id, inner.fold_id, inner.train_group_ids)
            add_role("inner_validation", outer.fold_id, inner.fold_id, inner.validation_group_ids)
    if issues:
        raise ValueError("group/trial support gate failed: " + "; ".join(issues))
    return pd.DataFrame(support_rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tabular_parent_stage(tabular_dir: Path) -> dict[str, Any]:
    """Verify the completed tabular stage before reusing its temporal OOF baseline."""
    tabular_dir = Path(tabular_dir).expanduser().resolve()
    receipt_path = tabular_dir / "supervised_model_stage_manifest.json"
    fold_path = tabular_dir / "fold_manifest.json"
    oof_path = tabular_dir / "oof_predictions.parquet"
    for path in (receipt_path, fold_path, oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen tabular artifact is missing: {path.name}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "completed"
        or receipt.get("label_rule") != "spectral_persistence_v1"
        or receipt.get("label_rule_type") != "weak_supervision_proxy_not_independent_fatigue_ground_truth"
        or receipt.get("external_rar_opened") is not False
        or int(receipt.get("internal_test_signal_values_parsed", -1)) != 0
        or receipt.get("training_evaluation_window_subset") != "even-index non-overlapping 500 ms windows"
        or receipt.get("overlap_between_selected_windows") != "none; 500 ms start-to-start hop"
        or not bool(receipt.get("coverage_gate", {}).get("fold_coverage_gate_pass", False))
    ):
        raise ValueError("tabular parent receipt does not match the completed, locked development contract")
    output_hashes = receipt.get("outputs_sha256", {})
    for path in (fold_path, oof_path):
        expected = next(
            (
                str(digest)
                for name, digest in output_hashes.items()
                if Path(name).expanduser().resolve() == path.resolve()
            ),
            "",
        )
        if not expected or _sha256_file(path) != expected:
            raise ValueError(f"tabular parent output checksum mismatch: {path.name}")
    return receipt


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def _load_fold_plan(path: Path) -> FoldPlan:
    document = json.loads(path.read_text(encoding="utf-8"))
    development_groups = document.get("development_groups", document.get("development_group_ids", ()))
    outer_folds: list[GroupFold] = []
    inner_folds: dict[int, tuple[GroupFold, ...]] = {}
    for item in document.get("outer_folds", ()):
        outer_id = int(item["fold_id"])
        outer_folds.append(
            GroupFold(
                outer_id,
                tuple(str(value) for value in item["train_group_ids"]),
                tuple(str(value) for value in item["validation_group_ids"]),
            )
        )
        inner_folds[outer_id] = tuple(
            GroupFold(
                int(inner["fold_id"]),
                tuple(str(value) for value in inner["train_group_ids"]),
                tuple(str(value) for value in inner["validation_group_ids"]),
            )
            for inner in item.get("inner_folds", ())
        )
    if not outer_folds or not development_groups:
        raise ValueError("frozen tabular fold manifest is empty or malformed")
    return FoldPlan(
        seed=int(document.get("seed", 20261113)),
        development_group_ids=tuple(str(value) for value in development_groups),
        outer_folds=tuple(outer_folds),
        inner_folds=inner_folds,
    )


def _support_for_groups(rows: pd.DataFrame, group_ids: Sequence[str]) -> dict[int, tuple[int, int, int]]:
    selected = rows[rows.group_id.astype(str).isin({str(value) for value in group_ids})]
    result: dict[int, tuple[int, int, int]] = {}
    for class_id in (0, 1):
        part = selected[selected.class_id.eq(class_id)]
        result[class_id] = (
            int(len(part)),
            int(part.group_id.nunique()),
            int(part[["group_id", "recording_id"]].drop_duplicates().shape[0]),
        )
    return result


def _confusion_metrics(matrix: np.ndarray) -> dict[str, Any]:
    cm = np.asarray(matrix, dtype=np.float64)
    total = float(cm.sum())
    accuracy = float(np.trace(cm) / total) if total else float("nan")
    supports = cm.sum(axis=1)
    recalls = np.divide(np.diag(cm), supports, out=np.full(2, np.nan, dtype=float), where=supports > 0)
    f1_values: list[float] = []
    for class_id in (0, 1):
        tp = float(cm[class_id, class_id])
        fp = float(cm[:, class_id].sum() - tp)
        fn = float(cm[class_id, :].sum() - tp)
        denominator = 2.0 * tp + fp + fn
        f1_values.append(2.0 * tp / denominator if denominator else float("nan"))
    undefined: dict[str, str] = {}
    if np.all(supports > 0):
        macro_f1: float | None = float(np.mean(f1_values))
        balanced_accuracy: float | None = float(np.mean(recalls))
    else:
        absent = [str(class_id) for class_id, count in enumerate(supports) if count <= 0]
        reason = "class " + ", ".join(absent) + " absent from truth"
        macro_f1 = balanced_accuracy = None
        undefined["macro_f1"] = reason
        undefined["balanced_accuracy"] = reason
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "undefined_reasons": undefined,
    }


def _group_confusions(predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for group_id, group_rows in predictions.groupby("group_id", sort=True):
        result[str(group_id)] = confusion_matrix(
            group_rows.true_class_id.astype(int), group_rows.predicted_class_id.astype(int), labels=[0, 1]
        ).astype(np.float64)
    return result


def _metrics_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for model_name, rows in predictions.groupby("model_name", sort=True):
        y_true = rows.true_class_id.to_numpy(dtype=int)
        y_pred = rows.predicted_class_id.to_numpy(dtype=int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics = _confusion_metrics(matrix)
        confusions = _group_confusions(rows)
        normalized = np.stack([cm / cm.sum() for cm in confusions.values() if cm.sum() > 0])
        group_balanced = _confusion_metrics(normalized.mean(axis=0))
        pooled_rows.append(
            {
                "model_name": str(model_name),
                "n_oof_windows": int(len(rows)),
                "source_group_count": int(rows.group_id.nunique()),
                "groups_with_both_classes": int(
                    sum(
                        bool(group_rows.true_class_id.eq(0).any() and group_rows.true_class_id.eq(1).any())
                        for _, group_rows in rows.groupby("group_id")
                    )
                ),
                **metrics,
                "group_balanced_accuracy": group_balanced["accuracy"],
                "group_balanced_macro_f1": group_balanced["macro_f1"],
                "group_balanced_confusion_matrix_normalized": json.dumps(
                    normalized.mean(axis=0).tolist(), separators=(",", ":")
                ),
                "confusion_matrix": json.dumps(matrix.tolist(), separators=(",", ":")),
                "undefined_reasons": json.dumps(metrics["undefined_reasons"], separators=(",", ":")),
                "confusion_true_labels": "[0,1]",
                "confusion_predicted_labels": "[0,1]",
                "interpretation": "OOF agreement with weak-supervision proxy labels; windows are not independent units",
            }
        )
        for group_id, group_matrix in confusions.items():
            group_metric = _confusion_metrics(group_matrix)
            group_rows.append(
                {
                    "model_name": str(model_name),
                    "group_id": group_id,
                    "n_oof_windows": int(group_matrix.sum()),
                    "class_0_support": int(group_matrix[0].sum()),
                    "class_1_support": int(group_matrix[1].sum()),
                    **group_metric,
                    "undefined_reasons": json.dumps(group_metric["undefined_reasons"], separators=(",", ":")),
                    "confusion_matrix": json.dumps(group_matrix.astype(int).tolist(), separators=(",", ":")),
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(group_rows)


def _paired_group_bootstrap_vs_time(
    predictions: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 20261113,
) -> pd.DataFrame:
    baseline = predictions[predictions.model_name.eq("time_only_logistic")]
    if baseline.empty:
        raise ValueError("time-only OOF predictions are required for paired comparisons")
    baseline_confusions = _group_confusions(baseline)
    groups = tuple(sorted(baseline_confusions))
    rng = np.random.default_rng(int(seed))
    choices = rng.integers(0, len(groups), size=(int(replicates), len(groups)))

    baseline_by_group = {
        group: cm / max(cm.sum(), 1.0) for group, cm in baseline_confusions.items()
    }
    baseline_metric = _confusion_metrics(np.mean(np.stack(list(baseline_by_group.values())), axis=0))
    output: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        rows = predictions[predictions.model_name.eq(model_name)]
        model_confusions = _group_confusions(rows)
        common_groups = tuple(group for group in groups if group in model_confusions)
        if common_groups != groups:
            raise ValueError(f"model {model_name} and time baseline do not cover identical source groups")
        model_by_group = {group: cm / max(cm.sum(), 1.0) for group, cm in model_confusions.items()}
        model_metric = _confusion_metrics(np.mean(np.stack(list(model_by_group.values())), axis=0))
        acc_deltas: list[float] = []
        f1_deltas: list[float] = []
        for sample in choices:
            baseline_sample = np.mean([baseline_by_group[groups[index]] for index in sample], axis=0)
            model_sample = np.mean([model_by_group[groups[index]] for index in sample], axis=0)
            baseline_sample_metrics = _confusion_metrics(baseline_sample)
            model_sample_metrics = _confusion_metrics(model_sample)
            if np.isfinite(baseline_sample_metrics["accuracy"]) and np.isfinite(model_sample_metrics["accuracy"]):
                acc_deltas.append(model_sample_metrics["accuracy"] - baseline_sample_metrics["accuracy"])
            if baseline_sample_metrics["macro_f1"] is not None and model_sample_metrics["macro_f1"] is not None:
                f1_deltas.append(model_sample_metrics["macro_f1"] - baseline_sample_metrics["macro_f1"])
        group_both = sum(
            bool(part.true_class_id.eq(0).any() and part.true_class_id.eq(1).any())
            for _, part in rows.groupby("group_id")
        )
        output.append(
            {
                "model_name": model_name,
                "comparison_model": "time_only_logistic",
                "source_group_count": len(groups),
                "groups_with_both_classes": int(group_both),
                "group_balanced_accuracy": model_metric["accuracy"],
                "time_group_balanced_accuracy": baseline_metric["accuracy"],
                "group_balanced_accuracy_delta": model_metric["accuracy"] - baseline_metric["accuracy"],
                "group_balanced_macro_f1": model_metric["macro_f1"],
                "time_group_balanced_macro_f1": baseline_metric["macro_f1"],
                "group_balanced_macro_f1_delta": model_metric["macro_f1"] - baseline_metric["macro_f1"],
                "accuracy_delta_ci_low": float(np.quantile(acc_deltas, 0.025)) if acc_deltas else None,
                "accuracy_delta_ci_high": float(np.quantile(acc_deltas, 0.975)) if acc_deltas else None,
                "accuracy_delta_valid_replicates": len(acc_deltas),
                "macro_f1_delta_ci_low": float(np.quantile(f1_deltas, 0.025)) if f1_deltas else None,
                "macro_f1_delta_ci_high": float(np.quantile(f1_deltas, 0.975)) if f1_deltas else None,
                "macro_f1_delta_valid_replicates": len(f1_deltas),
                "bootstrap_replicates": int(replicates),
                "bootstrap_seed": int(seed),
                "bootstrap_unit": "paired source group; each group's confusion matrix normalized to unit mass",
            }
        )
    return pd.DataFrame(output)


def _new_model(
    model_name: str,
    seed: int,
    pretraining_windows: np.ndarray | None,
    ssl_epochs: int,
    external_encoder_path: Path | None = None,
):
    encoder_fit = None
    if model_name in {"ssl_encoder_finetuned", "ssl_encoder_frozen"}:
        if pretraining_windows is None:
            raise ValueError("SSL model initialization requires training-fold raw windows")
        encoder_fit = pretrain_encoder(
            pretraining_windows,
            epochs=int(ssl_epochs),
            seed=int(seed) + 101,
            batch_size=32,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            mask_fraction=0.2,
        )
        torch.manual_seed(int(seed) + 202)
        model = BinarySignalClassifier(encoder=encoder_fit.encoder)
        if model_name == "ssl_encoder_frozen":
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(False)
    elif model_name == "cerqueira_supervised_finetuned":
        if external_encoder_path is None:
            raise ValueError("external Cerqueira transfer model requires an external encoder checkpoint")
        torch.manual_seed(int(seed))
        encoder = SignalEncoder()
        receipt = initialize_signal_encoder_from_cerqueira(encoder, external_encoder_path)
        model = BinarySignalClassifier(encoder=encoder)
        model._external_transfer_receipt = receipt
    else:
        torch.manual_seed(int(seed))
        expert_count = {"latent_moe2": 2, "latent_moe3": 3}.get(model_name)
        model = SoftGatedExperts(expert_count=expert_count) if expert_count is not None else BinarySignalClassifier()
    return model.cpu(), encoder_fit


def _model_output(model, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(signals)
    if isinstance(output, tuple):
        logits, gates, _ = output
        return logits, gates
    return output, None


def _group_equal_weights(group_ids: Sequence[str]) -> torch.Tensor:
    groups = np.asarray([str(value) for value in group_ids], dtype=object)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    del unique
    weights = 1.0 / counts[inverse].astype(np.float32)
    weights *= len(weights) / max(float(weights.sum()), 1e-12)
    return torch.from_numpy(weights.astype(np.float32))


def _group_balanced_macro_f1(truth: np.ndarray, prediction: np.ndarray, groups: Sequence[str]) -> float:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    weights = _group_equal_weights(groups).numpy().astype(np.float64)
    if truth.ndim != 1 or prediction.shape != truth.shape or weights.shape != truth.shape:
        raise ValueError("truth, prediction, and groups must have equal one-dimensional shapes")
    if set(np.unique(truth).tolist()) != {0, 1}:
        raise ValueError("Group-balanced macro-F1 requires both classes in the validation partition")
    f1_values: list[float] = []
    for class_id in (0, 1):
        true_class = truth == class_id
        predicted_class = prediction == class_id
        tp = float(weights[true_class & predicted_class].sum())
        fp = float(weights[~true_class & predicted_class].sum())
        fn = float(weights[true_class & ~predicted_class].sum())
        denominator = 2.0 * tp + fp + fn
        f1_values.append(2.0 * tp / denominator if denominator else 0.0)
    return float(np.mean(f1_values))


def _train_supervised(
    model,
    train_signals: np.ndarray,
    train_labels: np.ndarray,
    train_groups: Sequence[str],
    *,
    epochs: int,
    validation_signals: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    validation_groups: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    x_train = torch.from_numpy(np.asarray(train_signals, dtype=np.float32)[:, None, :])
    y_train = torch.from_numpy(np.asarray(train_labels, dtype=np.float32))
    weights = _group_equal_weights(train_groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    has_validation = validation_signals is not None
    if has_validation and (validation_labels is None or validation_groups is None):
        raise ValueError("validation labels and groups are required with validation signals")
    x_valid = (
        torch.from_numpy(np.asarray(validation_signals, dtype=np.float32)[:, None, :])
        if has_validation
        else None
    )
    history: dict[str, list[float]] = {"training_loss": [], "validation_group_balanced_macro_f1": []}
    for _ in range(int(epochs)):
        model.train()
        logits, gates = _model_output(model, x_train)
        element_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
        loss = torch.sum(element_loss * weights) / weights.sum().clamp_min(1e-8)
        if gates is not None:
            loss = loss + GATE_BALANCE_WEIGHT * gate_load_balance_penalty(gates)
        if not torch.isfinite(loss):
            raise FloatingPointError("Supervised neural loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["training_loss"].append(float(loss.detach()))
        if has_validation:
            model.eval()
            with torch.no_grad():
                validation_logits, _ = _model_output(model, x_valid)
                probabilities = torch.sigmoid(validation_logits).cpu().numpy()
            prediction = (probabilities >= 0.5).astype(np.int64)
            history["validation_group_balanced_macro_f1"].append(
                _group_balanced_macro_f1(
                    np.asarray(validation_labels, dtype=np.int64),
                    prediction,
                    validation_groups,
                )
            )
    return history


def _predict(model, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    x = torch.from_numpy(np.asarray(signals, dtype=np.float32)[:, None, :])
    with torch.no_grad():
        logits, gates = _model_output(model, x)
        probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        gate_values = gates.cpu().numpy().astype(np.float64) if gates is not None else None
    return probabilities, gate_values


def _score_outer_fold(
    model_name: str,
    predictions: pd.DataFrame,
    *,
    outer_id: int,
    truth: np.ndarray,
    signals: np.ndarray,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    gate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite neural checkpoint: {checkpoint_path.name}")
    payload = {
        "model_name": model_name,
        "fold_id": int(outer_id),
        "model_state_dict": {name: value.detach().cpu() for name, value in checkpoint_metadata.pop("model").state_dict().items()},
        "metadata": checkpoint_metadata,
    }
    torch.save(payload, checkpoint_path)
    checkpoint_sha = _sha256_file(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SoftGatedExperts(expert_count=3) if model_name == "latent_moe3" else BinarySignalClassifier()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    probabilities, gates = _predict(model, signals)
    predicted = (probabilities >= 0.5).astype(np.int64)
    output_rows: list[dict[str, Any]] = []
    for index, probability in enumerate(probabilities):
        row = predictions.iloc[index]
        record = {
            "window_key": str(row.window_key),
            "group_id": str(row.group_id),
            "recording_id": str(row.recording_id),
            "feature_view": "raw_signal",
            "model_name": model_name,
            "fold_id": int(outer_id),
            "true_class_id": int(truth[index]),
            "predicted_class_id": int(predicted[index]),
            "probability_class_0": float(1.0 - probability),
            "probability_class_1": float(probability),
            "confidence": float(max(probability, 1.0 - probability)),
            "abstain": False,
            "checkpoint_sha256": checkpoint_sha,
        }
        if "label_confidence" in row:
            record["label_confidence"] = float(row.label_confidence)
        if "label_reason" in row:
            record["label_reason"] = str(row.label_reason)
        if gates is not None:
            for expert_id, gate_value in enumerate(gates[index]):
                record[f"gate_probability_{expert_id}"] = float(gate_value)
        output_rows.append(record)

    gate_receipt: dict[str, Any] = {"model_name": model_name, "fold_id": int(outer_id)}
    if gates is not None:
        mean_gate = gates.mean(axis=0)
        entropy = float(normalized_gate_entropy(torch.from_numpy(gates.astype(np.float32))).item())
        gate_receipt.update(
            {
                "mean_gate_entropy_normalized": entropy,
                "mean_gate_utilization": json.dumps(mean_gate.tolist(), separators=(",", ":")),
                "collapsed_gate": bool(np.any(mean_gate < 0.05) or np.any(mean_gate > 0.90)),
                "gate_diagnostic_scope": "held-out outer-fold OOF windows",
            }
        )
        gate_rows.append(gate_receipt)
    return output_rows, {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha, "gate": gate_receipt}


def _validate_fold_assignment(active: pd.DataFrame, plan: FoldPlan) -> None:
    group_to_fold: dict[str, int] = {}
    for outer in plan.outer_folds:
        for group_id in outer.validation_group_ids:
            if group_id in group_to_fold:
                raise ValueError(f"source group {group_id} is assigned to multiple outer validation folds")
            group_to_fold[group_id] = outer.fold_id
    expected_groups = set(active.group_id.astype(str))
    if expected_groups - set(group_to_fold):
        raise ValueError("some labeled development groups are absent from outer validation folds")
    if any(group_to_fold.get(str(row.group_id)) != int(row.fold_id) for row in active.itertuples()):
        raise ValueError("frozen tabular OOF fold IDs disagree with the group fold manifest")


def run_neural_supervised_stage(
    run_dir: Path,
    *,
    tabular_subdir: str = "supervised_models_353_nested_3x2",
    output_subdir: str = "neural_supervised_353_nested_3x2",
    max_epochs: int = 20,
    ssl_pretrain_epochs: int = 6,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Train CNN, fold-limited SSL and latent MoE models on development groups only."""
    if Path(output_subdir).name != output_subdir or output_subdir in {"", ".", ".."}:
        raise ValueError("output_subdir must be one directory name")
    if Path(tabular_subdir).name != tabular_subdir or tabular_subdir in {"", ".", ".."}:
        raise ValueError("tabular_subdir must be one directory name")
    if int(max_epochs) < 1 or int(ssl_pretrain_epochs) < 1:
        raise ValueError("training epoch limits must be positive")
    torch.set_num_threads(1)
    run_dir = Path(run_dir).expanduser().resolve()
    private_dir = run_dir / "private"
    neural_data_dir = private_dir / "neural_data"
    tabular_dir = private_dir / tabular_subdir
    bank_manifest_path = neural_data_dir / "signal_window_bank_manifest.json"
    bank_path = neural_data_dir / "signals_500ms_development.npz"
    metadata_path = neural_data_dir / "signal_window_manifest.parquet"
    fold_path = tabular_dir / "fold_manifest.json"
    oof_path = tabular_dir / "oof_predictions.parquet"
    tabular_receipt_path = tabular_dir / "supervised_model_stage_manifest.json"
    for path in (bank_manifest_path, bank_path, metadata_path, fold_path, oof_path, tabular_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required private development artifact is missing: {path.name}")
    verify_tabular_parent_stage(tabular_dir)
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    if (
        bank_manifest.get("status") != "completed"
        or bank_manifest.get("external_rar_opened") is not False
        or int(bank_manifest.get("internal_test_signal_values_parsed", -1)) != 0
        or int(bank_manifest.get("reserved_test_member_payloads_read", -1)) != 0
    ):
        raise ValueError("development signal bank receipt does not prove that reserved inputs stayed closed")
    if bank_manifest.get("pseudo_label_rule") != "spectral_persistence_v1":
        raise ValueError("neural stage requires the confirmed spectral_persistence_v1 proxy labels")
    expected_bank_outputs = bank_manifest.get("output_sha256", {})
    if expected_bank_outputs.get(str(bank_path)) != _sha256_file(bank_path):
        raise ValueError("development signal bank checksum differs from its frozen receipt")
    if expected_bank_outputs.get(str(metadata_path)) != _sha256_file(metadata_path):
        raise ValueError("development signal metadata checksum differs from its frozen receipt")

    bank = np.load(bank_path, allow_pickle=False)
    signals = np.asarray(bank["signals"], dtype=np.float32)
    bank_keys = bank["window_keys"].astype(str)
    metadata = pd.read_parquet(metadata_path)
    if signals.ndim != 2 or signals.shape[1] != 500 or signals.shape[0] != len(metadata):
        raise ValueError("signal bank shape is inconsistent with the 500 ms window contract")
    if not np.array_equal(bank_keys, metadata.window_key.astype(str).to_numpy()):
        raise ValueError("signal bank order does not align with the private window manifest")
    if metadata.window_key.duplicated().any() or not metadata.class_id.isin((-1, 0, 1)).all():
        raise ValueError("signal window manifest contains duplicate keys or invalid proxy classes")

    plan = _load_fold_plan(fold_path)
    tabular_predictions = pd.read_parquet(oof_path)
    baseline = tabular_predictions.loc[
        tabular_predictions.feature_view.eq("relative_time_only")
        & tabular_predictions.model_name.eq("logistic_regression")
        & tabular_predictions.true_class_id.isin((0, 1)),
        ["window_key", "group_id", "recording_id", "fold_id", "true_class_id", "predicted_class_id", "probability_class_0", "probability_class_1"],
    ].copy()
    baseline["window_key"] = baseline.window_key.astype(str)
    if baseline.window_key.duplicated().any():
        raise ValueError("time-only baseline has duplicate OOF predictions")
    baseline = baseline.rename(
        columns={
            "group_id": "baseline_group_id",
            "recording_id": "baseline_recording_id",
            "fold_id": "baseline_fold_id",
            "true_class_id": "baseline_true_class_id",
            "predicted_class_id": "baseline_predicted_class_id",
            "probability_class_0": "baseline_probability_class_0",
            "probability_class_1": "baseline_probability_class_1",
        }
    )

    window_indices = metadata.window_index.astype(int).to_numpy()
    selected = (window_indices % 2 == 0) & metadata.class_id.isin((0, 1)).to_numpy()
    active = metadata.loc[selected].copy()
    active["bank_position"] = np.flatnonzero(selected)
    active["group_id"] = active.group_id.astype(str)
    active["recording_id"] = active.recording_id.astype(str)
    active["class_id"] = active.class_id.astype(int)
    active = active.merge(baseline, on="window_key", how="left", validate="one_to_one")
    if active.baseline_fold_id.isna().any() or len(active) != len(baseline):
        raise ValueError("time-only baseline does not exactly cover the non-overlapping assigned proxy windows")
    if not active.group_id.astype(str).eq(active.baseline_group_id.astype(str)).all():
        raise ValueError("baseline group identities do not match raw-signal metadata")
    if not active.recording_id.astype(str).eq(active.baseline_recording_id.astype(str)).all():
        raise ValueError("baseline trial identities do not match raw-signal metadata")
    if not active.class_id.eq(active.baseline_true_class_id.astype(int)).all():
        raise ValueError("time-only baseline labels do not match frozen active pseudo-labels")
    active["fold_id"] = active.baseline_fold_id.astype(int)
    _validate_fold_assignment(active, plan)
    support = audit_group_fold_support(active, plan, min_groups_per_class=2, min_trials_per_class=2)

    counts = active.class_id.value_counts().to_dict()
    if int(counts.get(0, 0)) == 0 or int(counts.get(1, 0)) == 0:
        raise ValueError("both pseudo-classes are required before neural model training")
    output_dir = private_dir / output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite neural model stage: {output_dir}")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    support.to_csv(output_dir / "label_support_by_fold.csv", index=False)

    inner_score_rows: list[dict[str, Any]] = []
    selected_epoch_rows: list[dict[str, Any]] = []
    outer_prediction_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    checkpoint_receipts: list[dict[str, Any]] = []

    def original_positions(groups: Sequence[str]) -> np.ndarray:
        return active.loc[active.group_id.isin({str(value) for value in groups}), "bank_position"].to_numpy(dtype=int)

    def labeled_rows(groups: Sequence[str]) -> pd.DataFrame:
        return active.loc[active.group_id.isin({str(value) for value in groups})].sort_values("window_key").reset_index(drop=True)

    for outer in plan.outer_folds:
        for model_index, model_name in enumerate(MODEL_NAMES):
            per_epoch_scores: list[list[float]] = []
            for inner in plan.inner_folds[outer.fold_id]:
                train_rows = labeled_rows(inner.train_group_ids)
                valid_rows = labeled_rows(inner.validation_group_ids)
                if set(train_rows.group_id).intersection(valid_rows.group_id):
                    raise RuntimeError("inner train and validation groups overlap")
                raw_positions = select_fold_pretraining_indices(metadata, inner.train_group_ids)
                if not set(metadata.iloc[raw_positions].group_id.astype(str)).issubset(set(inner.train_group_ids)):
                    raise RuntimeError("inner pretraining admitted an inner-validation source group")
                scaler_mean = float(signals[raw_positions].mean())
                scaler_std = float(signals[raw_positions].std())
                if not np.isfinite(scaler_std) or scaler_std <= 1e-8:
                    raise ValueError("training-fold signal standard deviation is zero or non-finite")
                train_raw = signals[train_rows.bank_position.to_numpy(dtype=int)]
                valid_raw = signals[valid_rows.bank_position.to_numpy(dtype=int)]
                train_x = (train_raw - scaler_mean) / scaler_std
                valid_x = (valid_raw - scaler_mean) / scaler_std
                pretraining_x = (signals[raw_positions] - scaler_mean) / scaler_std
                seed = int(plan.seed + outer.fold_id * 10000 + inner.fold_id * 100 + model_index * 10)
                model, ssl_fit = _new_model(
                    model_name,
                    seed,
                    pretraining_x if model_name == "ssl_encoder_finetuned" else None,
                    int(ssl_pretrain_epochs),
                )
                history = _train_supervised(
                    model,
                    train_x,
                    train_rows.class_id.to_numpy(dtype=int),
                    train_rows.group_id.astype(str).tolist(),
                    epochs=int(max_epochs),
                    validation_signals=valid_x,
                    validation_labels=valid_rows.class_id.to_numpy(dtype=int),
                    validation_groups=valid_rows.group_id.astype(str).tolist(),
                )
                per_epoch_scores.append(history["validation_group_balanced_macro_f1"])
                for epoch, (loss, score) in enumerate(
                    zip(history["training_loss"], history["validation_group_balanced_macro_f1"], strict=True),
                    start=1,
                ):
                    inner_score_rows.append(
                        {
                            "model_name": model_name,
                            "outer_fold_id": outer.fold_id,
                            "inner_fold_id": inner.fold_id,
                            "epoch": epoch,
                            "training_loss_group_balanced": loss,
                            "validation_group_balanced_macro_f1": score,
                            "normalization_window_count": int(len(raw_positions)),
                            "ssl_pretraining_window_count": 0 if ssl_fit is None else ssl_fit.train_window_count,
                            "pretraining_epochs": 0 if ssl_fit is None else len(ssl_fit.reconstruction_losses),
                            "ssl_pretraining_train_groups_only": ssl_fit is not None,
                            "pretraining_reconstruction_losses": [] if ssl_fit is None else list(ssl_fit.reconstruction_losses),
                        }
                    )
            epoch_matrix = np.asarray(per_epoch_scores, dtype=np.float64)
            mean_epoch_scores = np.nanmean(epoch_matrix, axis=0)
            selected_epoch = int(np.nanargmax(mean_epoch_scores)) + 1
            selected_epoch_rows.append(
                {
                    "model_name": model_name,
                    "outer_fold_id": outer.fold_id,
                    "selected_supervised_epochs": selected_epoch,
                    "inner_mean_group_balanced_macro_f1": float(mean_epoch_scores[selected_epoch - 1]),
                    "inner_fold_count": len(per_epoch_scores),
                    "selection_metric": "mean inner-fold group-balanced macro-F1; confusion counts weighted to equal source-group mass",
                    "decision_threshold": 0.5,
                }
            )

            train_rows = labeled_rows(outer.train_group_ids)
            valid_rows = labeled_rows(outer.validation_group_ids)
            raw_positions = select_fold_pretraining_indices(metadata, outer.train_group_ids)
            if not set(metadata.iloc[raw_positions].group_id.astype(str)).issubset(set(outer.train_group_ids)):
                raise RuntimeError("outer pretraining admitted an outer-validation source group")
            scaler_mean = float(signals[raw_positions].mean())
            scaler_std = float(signals[raw_positions].std())
            if not np.isfinite(scaler_std) or scaler_std <= 1e-8:
                raise ValueError("outer training-fold signal standard deviation is zero or non-finite")
            train_raw = signals[train_rows.bank_position.to_numpy(dtype=int)]
            valid_raw = signals[valid_rows.bank_position.to_numpy(dtype=int)]
            train_x = (train_raw - scaler_mean) / scaler_std
            valid_x = (valid_raw - scaler_mean) / scaler_std
            pretraining_x = (signals[raw_positions] - scaler_mean) / scaler_std
            seed = int(plan.seed + outer.fold_id * 10000 + model_index * 10 + 9000)
            model, ssl_fit = _new_model(
                model_name,
                seed,
                pretraining_x if model_name == "ssl_encoder_finetuned" else None,
                int(ssl_pretrain_epochs),
            )
            training_history = _train_supervised(
                model,
                train_x,
                train_rows.class_id.to_numpy(dtype=int),
                train_rows.group_id.astype(str).tolist(),
                epochs=selected_epoch,
            )
            checkpoint_path = checkpoint_dir / f"{model_name}_outer_{outer.fold_id:02d}.pt"
            checkpoint_metadata = {
                "seed": seed,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "selected_supervised_epochs": selected_epoch,
                "training_loss_group_balanced": training_history["training_loss"],
                "label_rule": "spectral_persistence_v1",
                "label_rule_type": "weak_supervision_proxy_not_independent_fatigue_ground_truth",
                "decision_threshold": 0.5,
                "training_source_groups": list(outer.train_group_ids),
                "held_out_source_groups": list(outer.validation_group_ids),
                "training_labeled_window_count": int(len(train_rows)),
                "training_raw_normalization_window_count": int(len(raw_positions)),
                "normalization_source_groups": list(outer.train_group_ids),
                "pretraining_source_groups": list(outer.train_group_ids) if ssl_fit is not None else [],
                "pretraining_window_count": 0 if ssl_fit is None else int(ssl_fit.train_window_count),
                "pretraining_epoch_count": 0 if ssl_fit is None else len(ssl_fit.reconstruction_losses),
                "pretraining_reconstruction_losses": [] if ssl_fit is None else list(ssl_fit.reconstruction_losses),
                "pretraining_optimizer": None if ssl_fit is None else ssl_fit.optimizer,
                "normalization_mean_fit_on_training_groups": scaler_mean,
                "normalization_std_fit_on_training_groups": scaler_std,
                "nonoverlap_pretraining_window_selection": "eligible even-index 500 ms windows; at most 16 per trial and 64 per source group",
                "gate_balance_weight": GATE_BALANCE_WEIGHT if model_name == "latent_moe3" else 0.0,
                "expert_semantics": "latent experts share one binary proxy target" if model_name == "latent_moe3" else None,
            }
            checkpoint_metadata["model"] = model
            fold_prediction_rows = valid_rows.reset_index(drop=True)
            fold_outputs, checkpoint_receipt = _score_outer_fold(
                model_name,
                fold_prediction_rows,
                outer_id=outer.fold_id,
                truth=valid_rows.class_id.to_numpy(dtype=int),
                signals=valid_x,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata=checkpoint_metadata,
                gate_rows=gate_rows,
            )
            outer_prediction_rows.extend(fold_outputs)
            checkpoint_receipts.append(
                {
                    "model_name": model_name,
                    "outer_fold_id": outer.fold_id,
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_receipt["checkpoint_sha256"],
                    "selected_supervised_epochs": selected_epoch,
                    "training_source_groups": list(outer.train_group_ids),
                    "held_out_source_groups": list(outer.validation_group_ids),
                    "normalization_source_groups": list(outer.train_group_ids),
                    "pretraining_source_groups": list(outer.train_group_ids) if ssl_fit is not None else [],
                    "training_labeled_window_count": int(len(train_rows)),
                    "normalization_window_count": int(len(raw_positions)),
                    "pretraining_window_count": 0 if ssl_fit is None else int(ssl_fit.train_window_count),
                }
            )

    neural_predictions = pd.DataFrame(outer_prediction_rows)
    expected_keys = set(active.window_key.astype(str))
    for model_name in MODEL_NAMES:
        model_rows = neural_predictions[neural_predictions.model_name.eq(model_name)]
        if model_rows.window_key.duplicated().any() or set(model_rows.window_key.astype(str)) != expected_keys:
            raise RuntimeError(f"{model_name} did not produce exactly one OOF prediction per assigned development window")
    baseline_rows = active[
        ["window_key", "group_id", "recording_id", "class_id", "baseline_predicted_class_id", "baseline_probability_class_0", "baseline_probability_class_1"]
    ].rename(
        columns={
            "class_id": "true_class_id",
            "baseline_predicted_class_id": "predicted_class_id",
            "baseline_probability_class_0": "probability_class_0",
            "baseline_probability_class_1": "probability_class_1",
        }
    )
    baseline_rows["feature_view"] = "time_only"
    baseline_rows["model_name"] = "time_only_logistic"
    baseline_rows["fold_id"] = active.baseline_fold_id.astype(int).to_numpy()
    baseline_rows["confidence"] = baseline_rows[["probability_class_0", "probability_class_1"]].max(axis=1)
    baseline_rows["abstain"] = False
    combined = pd.concat([neural_predictions, baseline_rows], ignore_index=True, sort=False)
    combined.to_parquet(output_dir / "neural_oof_predictions.parquet", index=False)
    pooled_metrics, group_metrics = _metrics_tables(combined)
    pooled_metrics.to_csv(output_dir / "neural_metrics.csv", index=False)
    group_metrics.to_csv(output_dir / "neural_group_metrics.csv", index=False)
    comparisons = _paired_group_bootstrap_vs_time(
        combined,
        replicates=int(bootstrap_replicates),
        seed=int(plan.seed),
    )
    comparisons.to_csv(output_dir / "paired_group_bootstrap_vs_time.csv", index=False)
    pd.DataFrame(inner_score_rows).to_csv(output_dir / "inner_training_history.csv", index=False)
    pd.DataFrame(selected_epoch_rows).to_csv(output_dir / "inner_epoch_selection.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output_dir / "moe_gate_diagnostics.csv", index=False)
    _write_json(output_dir / "checkpoint_manifest.json", checkpoint_receipts)

    input_paths = [bank_manifest_path, bank_path, metadata_path, tabular_receipt_path, fold_path, oof_path]
    source_paths = [Path(__file__), Path(__file__).with_name("models_neural.py"), Path(__file__).with_name("ssl_pretraining.py")]
    output_paths = [
        output_dir / "label_support_by_fold.csv",
        output_dir / "neural_oof_predictions.parquet",
        output_dir / "neural_metrics.csv",
        output_dir / "neural_group_metrics.csv",
        output_dir / "paired_group_bootstrap_vs_time.csv",
        output_dir / "inner_training_history.csv",
        output_dir / "inner_epoch_selection.csv",
        output_dir / "moe_gate_diagnostics.csv",
        output_dir / "checkpoint_manifest.json",
        *sorted(checkpoint_dir.glob("*.pt")),
    ]
    receipt = {
        "stage": "development_nested_grouped_neural_supervised_models",
        "status": "completed",
        "label_rule": "spectral_persistence_v1",
        "label_rule_type": "weak_supervision_proxy_not_independent_fatigue_ground_truth",
        "training_evaluation_window_subset": "even-index non-overlapping 500 ms windows (500 ms start-to-start hop)",
        "overlapping_windows_treated_as_independent_groups": False,
        "independent_unit": "source folder group; participant identity is unverified",
        "class_0_nonoverlap_assigned_windows": int(counts.get(0, 0)),
        "class_1_nonoverlap_assigned_windows": int(counts.get(1, 0)),
        "assigned_all_overlap_label_count_descriptive_only": int(metadata.class_id.isin((0, 1)).sum()),
        "ambiguous_all_overlap_label_count": int(metadata.class_id.eq(-1).sum()),
        "outer_group_folds": len(plan.outer_folds),
        "inner_group_folds_per_outer": len(plan.inner_folds[plan.outer_folds[0].fold_id]),
        "coverage_minima_per_train_and_validation_role": {"source_groups_per_class": 2, "recording_trials_per_class": 2},
        "coverage_gate_passed": True,
        "fold_support_file": "label_support_by_fold.csv",
        "models": list(MODEL_NAMES),
        "ssl_pretraining": "fresh masked-span reconstruction; eligible non-overlap development windows from training groups only",
        "ssl_pretraining_epochs": int(ssl_pretrain_epochs),
        "ssl_windows_per_trial_cap": 16,
        "ssl_windows_per_group_cap": 64,
        "max_supervised_epochs_for_inner_selection": int(max_epochs),
        "selected_epoch_metric": "mean inner-fold group-balanced macro-F1; each source-group confusion matrix has equal total mass",
        "decision_threshold": 0.5,
        "training_loss_group_balanced": True,
        "normalization_fit_groups_only": True,
        "model_outputs": "single weak-fatigue-proxy logit; MoE experts are latent",
        "temporal_baseline": "paired out-of-fold relative-time logistic model from the same frozen development folds",
        "paired_uncertainty": f"{int(bootstrap_replicates)} paired source-group bootstrap replicates",
        "checkpoint_count": len(checkpoint_receipts),
        "checkpoint_hashes_verified_after_save": True,
        "internal_test_signal_values_parsed": 0,
        "reserved_test_member_payloads_read": 0,
        "external_rar_opened": False,
        "input_sha256": {str(path): _sha256_file(path) for path in input_paths},
        "source_sha256": {str(path): _sha256_file(path) for path in source_paths},
        "output_sha256": {str(path): _sha256_file(path) for path in output_paths},
        "group_support_rows": int(len(support)),
    }
    _write_json(output_dir / "neural_supervised_stage_manifest.json", receipt)
    return {**receipt, "manifest_path": str((output_dir / "neural_supervised_stage_manifest.json").resolve())}
