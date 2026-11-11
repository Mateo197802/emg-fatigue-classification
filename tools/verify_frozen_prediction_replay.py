"""Verify deterministic replay of frozen clean external predictions.

This checks stored feature rows against the clean predictions produced during
the prediction-only external stage. It does not read raw external signals,
fit a model, tune a threshold, or require labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from emg_fatigue.weak_supervision import evaluation
from emg_fatigue.weak_supervision.features import FEATURE_VIEWS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_model_rows(
    *,
    model_name: str,
    policy: str,
    features: pd.DataFrame,
    saved: pd.DataFrame,
    checkpoint: Path,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    observed = saved.loc[
        saved.model.astype(str).eq(model_name)
        & saved.policy.astype(str).eq(policy)
        & saved.condition.astype(str).eq("clean:0")
    ].sort_values("window_key").reset_index(drop=True)
    if observed.empty or observed.window_key.astype(str).duplicated().any():
        raise ValueError(f"Saved clean prediction rows are empty or duplicated for {model_name}/{policy}")

    source = features.set_index(features.window_key.astype(str), drop=False)
    keys = observed.window_key.astype(str).tolist()
    missing = sorted(set(keys) - set(source.index.astype(str)))
    if missing:
        raise ValueError(f"Saved prediction keys are missing from the feature table: {missing[:3]}")
    aligned = source.loc[keys].reset_index(drop=True)
    if aligned.window_key.astype(str).tolist() != keys:
        raise ValueError(f"Feature rows could not be aligned exactly for {model_name}/{policy}")
    if not set(feature_names).issubset(aligned.columns):
        raise ValueError(f"Required frozen model features are missing for {model_name}: {feature_names}")

    package = joblib.load(checkpoint)
    if model_name == "A1_morphology_logistic":
        if not isinstance(package, dict) or tuple(package.get("feature_names", ())) != feature_names:
            raise ValueError("A1 checkpoint feature contract does not match the replay feature list")
        estimator = package["estimator"]
    else:
        estimator = package
    replay = evaluation._positive_probability(
        estimator, aligned.loc[:, list(feature_names)].astype(float)
    )[:, 1]
    stored = observed.probability_class_1.to_numpy(dtype=np.float64)
    exact = np.array_equal(replay, stored)
    delta = np.abs(replay - stored)
    threshold = observed.decision_threshold.to_numpy(dtype=np.float64)
    replay_class = (replay >= threshold).astype(np.int8)
    stored_class = observed.prediction_class_id.to_numpy(dtype=np.int8)
    class_mismatch_count = int(np.count_nonzero(replay_class != stored_class))
    if class_mismatch_count:
        raise ValueError(f"Replayed class decisions differ for {model_name}/{policy}")

    return {
        "model": model_name,
        "policy": policy,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "feature_count": len(feature_names),
        "window_count": len(keys),
        "probabilities_bitwise_identical": bool(exact),
        "max_absolute_probability_difference": float(delta.max(initial=0.0)),
        "class_decision_mismatch_count": class_mismatch_count,
        "decision_threshold_min": float(threshold.min()),
        "decision_threshold_max": float(threshold.max()),
    }


def verify_frozen_prediction_replay(
    *,
    features_path: Path,
    predictions_path: Path,
    a1_checkpoint: Path,
    temporal_checkpoint: Path,
    output_path: Path,
) -> dict[str, Any]:
    features_path = features_path.resolve()
    predictions_path = predictions_path.resolve()
    a1_checkpoint = a1_checkpoint.resolve()
    temporal_checkpoint = temporal_checkpoint.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite replay receipt: {output_path}")

    features = pd.read_parquet(features_path)
    predictions = pd.read_parquet(predictions_path)
    a1_feature_names = tuple(FEATURE_VIEWS["A1_morphology_only"])
    results = [
        _verify_model_rows(
            model_name="A1_morphology_logistic",
            policy="development_oof_selected",
            features=features,
            saved=predictions,
            checkpoint=a1_checkpoint,
            feature_names=a1_feature_names,
        ),
        _verify_model_rows(
            model_name="time_only_logistic",
            policy="development_oof_selected",
            features=features,
            saved=predictions,
            checkpoint=temporal_checkpoint,
            feature_names=("relative_time_fraction",),
        ),
    ]
    files = (features_path, predictions_path, a1_checkpoint, temporal_checkpoint)
    receipt: dict[str, Any] = {
        "stage": "frozen_clean_prediction_deterministic_replay",
        "status": "completed" if all(row["probabilities_bitwise_identical"] for row in results) else "completed_with_numeric_drift",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "external_labels_read": False,
        "external_raw_archive_read": False,
        "external_fit_or_tuning_performed": False,
        "input_sha256": {str(path): _sha256_file(path) for path in files},
        "models": results,
        "interpretation": "Deterministic replay of stored clean predictions only; no labeled external performance is implied.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(output_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify_frozen_prediction_replay(
        features_path=args.features,
        predictions_path=args.predictions,
        a1_checkpoint=args.a1_checkpoint,
        temporal_checkpoint=args.temporal_checkpoint,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
