"""Apply development-only OOD and selective prediction to the A1 model."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .evaluation import _sha256_file
from .evaluation_guardrails import (
    OOD_METHODS,
    _aurc,
    _hash_entries,
    _paired_abstention_intervals,
    _risk_coverage_points,
    _select_confidence_threshold,
    _write_json,
)
from .evaluation_neural import _load_fold_plan
from .metrics import compute_metrics


def run_a1_guardrail_stage(
    run_dir: Path,
    *,
    a1_subdir: str = "a1_lowfreq_optimization_v1",
    guardrail_subdir: str = "neural_lowfreq_guardrails_multi_v2",
    output_subdir: str = "a1_lowfreq_guardrails_v1",
    confidence_thresholds: Sequence[float] | None = None,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Evaluate A1 with identical fold-specific OOD flags and inner-selected abstention."""
    run_dir = Path(run_dir).expanduser().resolve()
    private_dir = run_dir / "private"
    a1_dir = private_dir / a1_subdir
    guardrail_dir = private_dir / guardrail_subdir
    output_dir = private_dir / output_subdir
    for name in (a1_subdir, guardrail_subdir, output_subdir):
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"{name!r} must be a single relative directory name")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite A1 guardrail stage: {output_dir}")

    a1_manifest_path = a1_dir / "a1_optimization_stage_manifest.json"
    a1_manifest = json.loads(a1_manifest_path.read_text(encoding="utf-8"))
    if (
        a1_manifest.get("status") != "completed"
        or a1_manifest.get("label_rule") != "low_frequency_fraction_persistence_v1"
        or int(a1_manifest.get("outer_group_folds", -1)) != 4
        or int(a1_manifest.get("inner_group_folds", -1)) != 3
    ):
        raise ValueError("A1 parent must be the completed primary-label nested 4x3 stage")
    if errors := _hash_entries(a1_manifest.get("outputs_sha256", {})):
        raise ValueError("A1 parent outputs failed checksum validation: " + "; ".join(errors))

    ood_manifest_path = guardrail_dir / "guardrail_stage_manifest.json"
    ood_manifest = json.loads(ood_manifest_path.read_text(encoding="utf-8"))
    if (
        ood_manifest.get("status") != "completed"
        or ood_manifest.get("label_rule") != a1_manifest["label_rule"]
        or int(ood_manifest.get("outer_group_folds", -1)) != 4
        or int(ood_manifest.get("inner_group_folds_per_outer", -1)) != 3
        or ood_manifest.get("ood_labels_available") is not False
        or ood_manifest.get("external_rar_opened") is not False
        or int(ood_manifest.get("internal_test_signal_values_parsed", -1)) != 0
    ):
        raise ValueError("OOD parent is not the locked development-only 4x3 guardrail stage")
    if errors := _hash_entries(ood_manifest.get("output_sha256", {})):
        raise ValueError("OOD parent outputs failed checksum validation: " + "; ".join(errors))

    threshold_dir = private_dir / "lowfreq_threshold_v2"
    fold_path = a1_dir / "fold_manifest.json"
    plan = _load_fold_plan(fold_path)
    if len(plan.outer_folds) != 4 or any(len(plan.inner_folds[fold.fold_id]) != 3 for fold in plan.outer_folds):
        raise ValueError("A1 guardrails require the existing nested 4x3 group plan")
    config_path = private_dir / "fatigue_proxy_config.toml"
    if not config_path.is_file():
        raise FileNotFoundError("Frozen development config is missing")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    guardrails = config.get("guardrails", {})
    if confidence_thresholds is None:
        confidence_thresholds = guardrails.get("confidence_thresholds", (0.55, 0.65, 0.75))
    confidence_thresholds = tuple(float(value) for value in confidence_thresholds)
    if not confidence_thresholds or any(not 0.5 <= value <= 1.0 for value in confidence_thresholds):
        raise ValueError("confidence threshold candidates must be in [0.5, 1]")

    prediction_path = a1_dir / "nested_oof_predictions.parquet"
    receipt_path = a1_dir / "outer_model_receipts.csv"
    predictions = pd.read_parquet(prediction_path)
    predictions = predictions.loc[
        predictions.feature_view.eq("A1_morphology_only")
        & predictions.model_name.eq("logistic_regression")
        & predictions.threshold_strategy.eq("inner_oof_selected")
        & predictions.true_class_id.isin((0, 1))
    ].copy()
    if predictions.empty or predictions.window_key.astype(str).duplicated().any():
        raise ValueError("A1 selected-threshold OOF predictions must contain unique active windows")
    predictions["window_key"] = predictions.window_key.astype(str)
    predictions["group_id"] = predictions.group_id.astype(str)
    predictions["recording_id"] = predictions.recording_id.astype(str)
    predictions["model_name"] = "A1_morphology_logistic"
    predictions["confidence"] = predictions[["probability_class_0", "probability_class_1"]].max(axis=1)

    model_receipts = pd.read_csv(receipt_path)
    model_receipts = model_receipts.loc[
        model_receipts.feature_view.eq("A1_morphology_only")
        & model_receipts.model_name.eq("logistic_regression")
    ].copy()
    if len(model_receipts) != 4 or set(model_receipts.outer_fold_id.astype(int)) != {1, 2, 3, 4}:
        raise ValueError("A1 must have one calibrated outer checkpoint and inner receipt for each fold")
    selected_threshold: dict[int, float] = {}
    inner_frames: dict[int, pd.DataFrame] = {}
    inner_paths: list[Path] = []
    for receipt in model_receipts.itertuples(index=False):
        fold_id = int(receipt.outer_fold_id)
        selected_threshold[fold_id] = float(receipt.selected_threshold)
        inner_path = Path(str(receipt.inner_oof_artifact_path)).expanduser().resolve()
        if not inner_path.is_file() or _sha256_file(inner_path) != str(receipt.inner_oof_artifact_sha256):
            raise ValueError(f"A1 selected-candidate inner OOF hash mismatch in outer fold {fold_id}")
        inner = pd.read_parquet(inner_path)
        required = {"window_key", "group_id", "true_class_id", "probability_class_0", "probability_class_1"}
        if required - set(inner.columns) or inner.window_key.astype(str).duplicated().any():
            raise ValueError(f"A1 inner OOF artifact is incomplete or duplicated for outer fold {fold_id}")
        inner["group_id"] = inner.group_id.astype(str)
        inner["decision_threshold"] = selected_threshold[fold_id]
        inner["confidence"] = inner[["probability_class_0", "probability_class_1"]].max(axis=1)
        if set(inner.true_class_id.astype(int)) != {0, 1} or set(inner.group_id) != set(plan.outer_folds[fold_id - 1].train_group_ids):
            raise ValueError(f"A1 inner OOF support differs from outer training groups in fold {fold_id}")
        inner_frames[fold_id] = inner
        inner_paths.append(inner_path)
    predictions["decision_threshold"] = predictions.fold_id.astype(int).map(selected_threshold)
    predictions["predicted_class_id"] = predictions.predicted_class_id.astype(int)

    ood_prediction_path = guardrail_dir / "neural_guardrail_oof_predictions.parquet"
    ood_predictions = pd.read_parquet(ood_prediction_path)
    identity = ["window_key", "group_id", "recording_id", "fold_id"]
    guard_columns = [
        *identity, "ood_detector", "ood_score", "ood_threshold", "ood_flag", "unsupported_signal",
        "quality_pass", "non_classifier_abstention_reasons",
    ]
    missing = set(guard_columns) - set(ood_predictions.columns)
    if missing:
        raise ValueError(f"OOD parent predictions miss fields needed for A1: {sorted(missing)}")
    ood_predictions = ood_predictions[guard_columns].copy()
    ood_predictions["window_key"] = ood_predictions.window_key.astype(str)
    ood_predictions["group_id"] = ood_predictions.group_id.astype(str)
    ood_predictions["recording_id"] = ood_predictions.recording_id.astype(str)
    # OOD scoring is model independent. Verify that each neural registry entry
    # carries exactly the same fold-specific detector result before collapsing it.
    equality_columns = ["ood_score", "ood_threshold", "ood_flag", "unsupported_signal", "quality_pass"]
    keyed = ood_predictions.groupby([*identity, "ood_detector"], sort=False, dropna=False)
    for column in equality_columns:
        if (keyed[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"OOD scores vary by neural model for supposedly model-independent field {column}")
    ood_predictions = ood_predictions.drop_duplicates([*identity, "ood_detector"]).copy()
    if set(ood_predictions.ood_detector.astype(str)) != set(OOD_METHODS):
        raise ValueError("OOD parent does not contain the four required detector families")
    expected_keys = set(predictions.window_key.astype(str))
    for detector in OOD_METHODS:
        rows = ood_predictions.loc[ood_predictions.ood_detector.astype(str).eq(detector)]
        if set(rows.window_key.astype(str)) != expected_keys or rows.window_key.duplicated().any():
            raise ValueError(f"{detector} OOD evidence does not match A1's exact OOF window set")

    output_dir.mkdir(parents=True, exist_ok=False)
    expanded = []
    for detector in OOD_METHODS:
        scores = ood_predictions.loc[ood_predictions.ood_detector.astype(str).eq(detector)].drop(columns="ood_detector")
        merged = predictions.merge(scores, on=identity, how="left", validate="one_to_one")
        if merged.ood_score.isna().any():
            raise ValueError(f"A1 OOF rows are missing {detector} OOD scores")
        merged["ood_detector"] = detector
        expanded.append(merged)
    scored = pd.concat(expanded, ignore_index=True)
    scored.to_parquet(output_dir / "a1_guardrail_oof_predictions.parquet", index=False)

    policies: list[dict[str, Any]] = []
    selective_parts: list[pd.DataFrame] = []
    for fold in plan.outer_folds:
        inner = inner_frames[int(fold.fold_id)].copy()
        policy = _select_confidence_threshold(inner, confidence_thresholds)
        cut = float(policy["threshold"])
        policies.append({
            "model_name": "A1_morphology_logistic",
            "outer_fold_id": int(fold.fold_id),
            "selected_classifier_threshold": selected_threshold[int(fold.fold_id)],
            "selected_confidence_threshold": cut,
            "inner_oof_group_balanced_macro_f1": float(policy["inner_oof_group_balanced_macro_f1"]),
            "inner_oof_coverage": float(policy["inner_oof_coverage"]),
            "candidate_policy_scores": json.dumps(policy["candidate_thresholds"], separators=(",", ":")),
            "selection_objective": policy["selection_objective"],
            "inner_oof_groups": json.dumps(sorted(set(inner.group_id)), separators=(",", ":")),
            "outer_validation_used_for_policy": False,
        })
        fold_rows = scored.loc[scored.fold_id.astype(int).eq(int(fold.fold_id))].copy()
        uncertainty = fold_rows.confidence.to_numpy(dtype=float) < cut
        quality = ~fold_rows.quality_pass.to_numpy(dtype=bool)
        novelty = fold_rows.ood_flag.to_numpy(dtype=bool)
        unsupported = fold_rows.unsupported_signal.to_numpy(dtype=bool)
        abstain = uncertainty | quality | novelty | unsupported
        fold_rows["selected_confidence_threshold"] = cut
        fold_rows["abstain_classifier_uncertainty"] = uncertainty
        fold_rows["abstain_signal_quality"] = quality
        fold_rows["abstain_ood_novelty"] = novelty
        fold_rows["abstain_unsupported_signal"] = unsupported
        fold_rows["abstain"] = abstain
        fold_rows["abstention_reasons"] = [
            "|".join(reason for flag, reason in (
                (q, "signal_quality"), (o, "ood_novelty"), (u, "classifier_uncertainty"), (s, "outside_training_feature_support")
            ) if flag)
            for q, o, u, s in zip(quality, novelty, uncertainty, unsupported, strict=True)
        ]
        fold_rows["selective_predicted_class_id"] = np.where(abstain, -1, fold_rows.predicted_class_id.to_numpy(dtype=int))
        selective_parts.append(fold_rows)
    selective = pd.concat(selective_parts, ignore_index=True)
    if selective.duplicated(["window_key", "ood_detector"]).any() or len(selective) != len(predictions) * len(OOD_METHODS):
        raise RuntimeError("A1 abstention table does not cover each OOF window/detector exactly once")
    selective.to_parquet(output_dir / "a1_abstention_oof_predictions.parquet", index=False)
    pd.DataFrame(policies).to_csv(output_dir / "abstention_threshold_selection.csv", index=False)

    pooled_metrics: list[dict[str, Any]] = []
    group_metrics: list[dict[str, Any]] = []
    risk_curve_parts: list[pd.DataFrame] = []
    aurc_rows: list[dict[str, Any]] = []
    for detector in OOD_METHODS:
        rows = selective.loc[selective.ood_detector.astype(str).eq(detector)].copy()
        truth = rows.true_class_id.to_numpy(dtype=int)
        accepted = ~rows.abstain.to_numpy(dtype=bool)
        record = compute_metrics(
            truth,
            rows.probability_class_1.to_numpy(dtype=float),
            rows.selective_predicted_class_id.to_numpy(dtype=int),
            rows.abstain.to_numpy(dtype=bool),
        ).to_dict()
        pooled_metrics.append({
            "model_name": "A1_morphology_logistic", "ood_detector": detector,
            "threshold_strategy": "inner_oof_selected_plus_inner_oof_selected_confidence_abstention",
            "selected_confidence_thresholds_by_outer_fold": json.dumps({str(row["outer_fold_id"]): row["selected_confidence_threshold"] for row in policies}, sort_keys=True, separators=(",", ":")),
            "coverage": record["coverage"],
            "coverage_class_0": float(accepted[truth == 0].mean()) if np.any(truth == 0) else None,
            "coverage_class_1": float(accepted[truth == 1].mean()) if np.any(truth == 1) else None,
            "abstained_count": int((~accepted).sum()),
            "abstained_count_class_0": int(np.sum((~accepted) & (truth == 0))),
            "abstained_count_class_1": int(np.sum((~accepted) & (truth == 1))),
            "abstained_for_quality": int(rows.abstain_signal_quality.sum()),
            "abstained_for_ood": int(rows.abstain_ood_novelty.sum()),
            "abstained_for_uncertainty": int(rows.abstain_classifier_uncertainty.sum()),
            "abstained_for_unsupported_signal": int(rows.abstain_unsupported_signal.sum()),
            "selective_error": record["selective_error"],
            "accuracy_including_abstention_as_error": record["accuracy"],
            "sensitivity": record["sensitivity"], "specificity": record["specificity"],
            "macro_f1_including_abstention": record["macro_f1"],
            "group_count": int(rows.group_id.nunique()),
            "confusion_matrix_with_abstain": json.dumps(record["confusion_matrix"], separators=(",", ":")),
            "undefined_reasons": json.dumps(record["undefined_reasons"], sort_keys=True, separators=(",", ":")),
        })
        for (group_id, fold_id), group in rows.groupby(["group_id", "fold_id"], sort=True):
            g_truth = group.true_class_id.to_numpy(dtype=int)
            g_accepted = ~group.abstain.to_numpy(dtype=bool)
            g_metric = compute_metrics(
                g_truth,
                group.probability_class_1.to_numpy(dtype=float),
                group.selective_predicted_class_id.to_numpy(dtype=int),
                group.abstain.to_numpy(dtype=bool),
            ).to_dict()
            group_metrics.append({
                "model_name": "A1_morphology_logistic", "ood_detector": detector,
                "group_id": str(group_id), "outer_fold_id": int(fold_id),
                "windows_descriptive_only": len(group), "class_0_support": int(np.sum(g_truth == 0)),
                "class_1_support": int(np.sum(g_truth == 1)), "coverage": g_metric["coverage"],
                "coverage_class_0": float(g_accepted[g_truth == 0].mean()) if np.any(g_truth == 0) else None,
                "coverage_class_1": float(g_accepted[g_truth == 1].mean()) if np.any(g_truth == 1) else None,
                "selective_error": g_metric["selective_error"],
                "macro_f1_including_abstention": g_metric["macro_f1"],
                "confusion_matrix_with_abstain": json.dumps(g_metric["confusion_matrix"], separators=(",", ":")),
                "undefined_reasons": json.dumps(g_metric["undefined_reasons"], sort_keys=True, separators=(",", ":")),
            })
        model_risk = rows[["group_id", "true_class_id", "predicted_class_id", "confidence"]].copy()
        grid = sorted(set([0.5, 0.55, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]) | set(np.quantile(model_risk.confidence.to_numpy(dtype=float), np.linspace(0, 1, 51)).tolist()))
        fixed = rows.abstain_signal_quality.to_numpy(dtype=bool) | rows.abstain_ood_novelty.to_numpy(dtype=bool) | rows.abstain_unsupported_signal.to_numpy(dtype=bool)
        points = _risk_coverage_points(model_risk, grid, fixed_abstain=fixed)
        points["model_name"] = "A1_morphology_logistic"
        points["ood_detector"] = detector
        risk_curve_parts.append(points)
        aurc_rows.append({
            "model_name": "A1_morphology_logistic", "ood_detector": detector,
            "group_balanced_aurc": _aurc(points, "group_balanced_coverage", "group_balanced_selective_error"),
            "pooled_aurc_not_group_independent": _aurc(points, "coverage", "selective_error"),
            "risk_definition": "selective classification error at confidence cut; source-group balanced error is primary",
            "confidence_is_ood": False, "source_group_count": int(rows.group_id.nunique()), "outer_oof_only": True,
        })

    pd.DataFrame(pooled_metrics).to_csv(output_dir / "abstention_metrics.csv", index=False)
    pd.DataFrame(group_metrics).to_csv(output_dir / "abstention_group_metrics.csv", index=False)
    pd.concat(risk_curve_parts, ignore_index=True).to_csv(output_dir / "risk_coverage_curves.csv", index=False)
    pd.DataFrame(aurc_rows).to_csv(output_dir / "aurc_summary.csv", index=False)
    _paired_abstention_intervals(selective, int(bootstrap_replicates), int(plan.seed), ("A1_morphology_logistic",)).to_csv(
        output_dir / "paired_abstention_group_bootstrap.csv", index=False
    )
    group_ood = (
        ood_predictions.groupby(["ood_detector", "group_id"], sort=True)
        .agg(
            window_count_descriptive_only=("window_key", "size"),
            ood_flag_count=("ood_flag", "sum"), ood_flag_rate=("ood_flag", "mean"),
            mean_ood_score=("ood_score", "mean"), threshold_min=("ood_threshold", "min"),
            threshold_max=("ood_threshold", "max"),
        ).reset_index()
    )
    group_ood.to_csv(output_dir / "ood_group_rates.csv", index=False)
    pd.DataFrame(json.loads((guardrail_dir / "ood_detector_receipts.json").read_text(encoding="utf-8"))).to_json(
        output_dir / "ood_detector_receipts.json", orient="records", indent=2
    )

    input_paths = [
        a1_manifest_path, prediction_path, receipt_path, fold_path, config_path,
        ood_manifest_path, ood_prediction_path, guardrail_dir / "ood_detector_receipts.json", *inner_paths,
    ]
    output_paths = [
        output_dir / name for name in (
            "a1_guardrail_oof_predictions.parquet", "a1_abstention_oof_predictions.parquet",
            "abstention_threshold_selection.csv", "abstention_metrics.csv", "abstention_group_metrics.csv",
            "risk_coverage_curves.csv", "aurc_summary.csv", "paired_abstention_group_bootstrap.csv",
            "ood_group_rates.csv", "ood_detector_receipts.json",
        )
    ]
    receipt = {
        "stage": "development_a1_grouped_ood_and_selective_prediction_guardrails",
        "status": "completed", "label_rule": a1_manifest["label_rule"],
        "outer_group_folds": 4, "inner_group_folds_per_outer": 3,
        "model": "A1_morphology_logistic", "ood_detectors": list(OOD_METHODS),
        "ood_scoring_reused_from": str(ood_manifest_path.resolve()),
        "ood_scoring_is_model_independent": True,
        "confidence_thresholds_considered": list(confidence_thresholds),
        "confidence_threshold_selection": "inner OOF group-balanced Macro-F1 with abstentions counted as class errors; ties favor coverage",
        "outer_validation_used_for_policy": False,
        "ood_labels_available": False, "ood_flags_are_not_validated_detection_accuracy": True,
        "ood_scores_are_not_confidence": True,
        "risk_coverage_and_aurc_unit": "source group primary; window counts descriptive",
        "internal_test_signal_values_parsed": 0, "external_rar_opened": False,
        "paired_bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_unit": "paired source group; overlapping windows are not independent units",
        "input_sha256": {str(path.resolve()): _sha256_file(path) for path in input_paths},
        "output_sha256": {str(path.resolve()): _sha256_file(path) for path in output_paths},
        "interpretation": "development agreement with the weak EMG proxy and input-distribution diagnostics only; no independent physiological fatigue or OOD validation",
    }
    manifest_path = output_dir / "a1_guardrail_stage_manifest.json"
    _write_json(manifest_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path.resolve())}
