"""Matched corruption analysis of frozen feature and self-supervised encoders."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .augmentation import ALL_FAMILIES, corrupt_signal
from .contracts import SignalWindow
from .features import extract_features
from .ssl import MaskedConvAutoencoder, apply_signal_scale, model_embedding, model_reconstruct
from .windows import window_quality_reason


def _median_feature_relative_change(
    clean_features: list[dict[str, float]], corrupted_features: list[dict[str, float]]
) -> float:
    if not clean_features or not corrupted_features:
        return float("nan")
    names = set(clean_features[0])
    for row in clean_features[1:] + corrupted_features:
        names.intersection_update(row)
    changes: list[float] = []
    for name in sorted(names):
        base = np.asarray([row.get(name, np.nan) for row in clean_features], dtype=np.float64)
        alt = np.asarray([row.get(name, np.nan) for row in corrupted_features], dtype=np.float64)
        valid = np.isfinite(base) & np.isfinite(alt)
        if valid.any():
            denominator = max(float(np.nanmedian(np.abs(base[valid]))), 1e-8)
            changes.append(float(np.nanmedian(np.abs(alt[valid] - base[valid])) / denominator))
    return float(np.median(changes)) if changes else float("nan")


def _select_matched_windows(windows: list[SignalWindow], maximum: int) -> list[SignalWindow]:
    grouped: dict[str, list[SignalWindow]] = {}
    for window in windows:
        grouped.setdefault(window.group_id, []).append(window)
    if len(windows) <= maximum:
        return windows
    per_group = max(1, maximum // max(1, len(grouped)))
    selected: list[SignalWindow] = []
    for group_id in sorted(grouped):
        records = sorted(grouped[group_id], key=lambda item: (item.recording_id, item.start_time_s))
        indices = np.linspace(0, len(records) - 1, min(per_group, len(records)), dtype=int)
        selected.extend(records[index] for index in np.unique(indices))
    return selected[:maximum]


def _make_feature_rows(windows: list[SignalWindow], threshold: float) -> list[dict[str, float]]:
    return [extract_features(window, threshold) for window in windows]


def evaluate_robustness(
    windows: list[SignalWindow],
    models: dict[str, MaskedConvAutoencoder],
    signal_scale: dict[str, float],
    willison_threshold: float,
    output_dir: Path,
    seed: int,
    line_hz: float | None,
    levels: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50),
    maximum_windows: int = 256,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = _select_matched_windows(windows, maximum_windows)
    if not selected:
        raise ValueError("No held-out windows are available for robustness analysis.")
    raw = np.stack([window.signal for window in selected]).astype(np.float32)
    clean_features = _make_feature_rows(selected, willison_threshold)
    clean_scaled = apply_signal_scale(raw, signal_scale)
    line_default = line_hz
    rows: list[dict[str, object]] = []
    clean_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name, model in models.items():
        clean_cache[model_name] = (model_embedding(model, clean_scaled), model_reconstruct(model, clean_scaled))

    def add_metrics(family: str, severity: float, corrupted: np.ndarray, receipts: list[dict[str, object]]) -> None:
        scaled = apply_signal_scale(corrupted, signal_scale)
        feature_windows = [replace(window, signal=corrupted[index]) for index, window in enumerate(selected)]
        feature_rows = _make_feature_rows(feature_windows, willison_threshold)
        reasons = [window_quality_reason(corrupted[index], selected[index].sampling_rate_hz) for index in range(len(selected))]
        receipt_ratios = [float(item["achieved_corruption_ratio"]) for item in receipts]
        receipt_snrs = [float(item["snr_db"]) for item in receipts if np.isfinite(float(item["snr_db"]))]
        for model_name, model in models.items():
            clean_embedding, clean_reconstruction = clean_cache[model_name]
            changed_embedding = model_embedding(model, scaled)
            changed_reconstruction = model_reconstruct(model, scaled)
            clean_norm = np.linalg.norm(clean_embedding, axis=1)
            changed_norm = np.linalg.norm(changed_embedding, axis=1)
            cosine = np.sum(clean_embedding * changed_embedding, axis=1) / np.maximum(clean_norm * changed_norm, 1e-8)
            mse = np.mean((changed_reconstruction - clean_scaled) ** 2, axis=1)
            rows.append(
                {
                    "model_condition": model_name,
                    "corruption_family": family,
                    "requested_severity": float(severity),
                    "n_matched_windows": int(len(selected)),
                    "achieved_corruption_ratio_median": float(np.median(receipt_ratios)),
                    "achieved_snr_db_median": float(np.median(receipt_snrs)) if receipt_snrs else float("nan"),
                    "median_feature_relative_absolute_change": _median_feature_relative_change(clean_features, feature_rows),
                    "median_embedding_cosine_similarity": float(np.nanmedian(cosine)),
                    "median_reconstruction_mse_to_clean": float(np.nanmedian(mse)),
                    "quality_rejection_rate": float(np.mean([reason is not None for reason in reasons])),
                }
            )

    clean_receipts = [
        {
            "achieved_corruption_ratio": 0.0,
            "snr_db": float("inf"),
        }
        for _ in selected
    ]
    add_metrics("clean", 0.0, raw, clean_receipts)

    for family_index, family in enumerate(ALL_FAMILIES):
        for level_index, severity in enumerate(levels):
            corrupted = np.empty_like(raw)
            receipts: list[dict[str, object]] = []
            for window_index, window in enumerate(selected):
                corruption_seed = seed + family_index * 10_000_019 + level_index * 1_000_003 + window_index * 97
                changed, receipt = corrupt_signal(
                    raw[window_index],
                    family,
                    severity,
                    corruption_seed,
                    window.sampling_rate_hz,
                    line_default,
                )
                corrupted[window_index] = changed
                receipts.append(receipt)
            add_metrics(family, severity, corrupted, receipts)

    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "feature_robustness.csv", index=False)

    reproducibility: dict[str, object] = {"seed": int(seed), "matched_window_count": len(selected), "models": {}}
    probe = clean_scaled[: min(8, len(clean_scaled))]
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(probe)
            first_embedding = model.encode(tensor)
            second_embedding = model.encode(tensor)
            first_reconstruction, _ = model(tensor)
            second_reconstruction, _ = model(tensor)
        exact_embedding = bool(torch.equal(first_embedding, second_embedding))
        exact_reconstruction = bool(torch.equal(first_reconstruction, second_reconstruction))
        reproducibility["models"][name] = {
            "same_input_embedding_exact": exact_embedding,
            "same_input_reconstruction_exact": exact_reconstruction,
        }
        if not exact_embedding or not exact_reconstruction:
            raise RuntimeError(f"Frozen {name} encoder is not deterministic on identical CPU inference inputs.")
    return frame, reproducibility
