"""Development-only group, trial, temporal, and OOF error audit artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


LABEL_RULE = "low_frequency_fraction_persistence_v1"
MODEL_CONFIGS = (
    ("lowfreq_threshold_v2", "historical_a1", "A1_morphology_only"),
    ("lowfreq_threshold_v2", "temporal_baseline", "relative_time_only"),
    ("a1_lowfreq_optimization_v1", "optimized_a1", "A1_morphology_only"),
    ("a1_lowfreq_optimization_v1", "temporal_baseline", "relative_time_only"),
)
MORPHOLOGY_FEATURES = (
    "rms", "mav", "iemg", "variance", "waveform_length",
    "waveform_length_per_second", "zero_crossings", "slope_sign_changes",
    "willison_amplitude", "median_absolute_deviation", "interquartile_range",
    "skewness", "kurtosis_excess", "hjorth_activity", "hjorth_mobility_hz",
    "hjorth_complexity",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str), encoding="utf-8")


def _metric_row(rows: pd.DataFrame, *, keys: dict[str, object]) -> dict[str, object]:
    truth = rows.true_class_id.astype(int).to_numpy()
    pred = rows.predicted_class_id.astype(int).to_numpy()
    scores = rows.probability_class_1.astype(float).to_numpy()
    matrix = confusion_matrix(truth, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1]))
    support0, support1 = int((truth == 0).sum()), int((truth == 1).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    sensitivity = tp / support1 if support1 else np.nan
    specificity = tn / support0 if support0 else np.nan
    f10 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else np.nan
    f11 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    result = {
        **keys,
        "n_windows_descriptive_only": int(len(rows)),
        "class_0_support_descriptive_only": support0,
        "class_1_support_descriptive_only": support1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": float((tn + tp) / len(rows)) if len(rows) else np.nan,
        "precision_fatigue_like": precision,
        "sensitivity_fatigue_like": sensitivity,
        "specificity_non_fatigue_like": specificity,
        "macro_f1": float(np.nanmean([f10, f11])) if np.isfinite(f10) and np.isfinite(f11) else np.nan,
        "confusion_matrix_tn_fp_fn_tp_json": json.dumps([[tn, fp], [fn, tp]], separators=(",", ":")),
        "probability_mean": float(np.mean(scores)) if len(scores) else np.nan,
        "probability_median": float(np.median(scores)) if len(scores) else np.nan,
        "probability_q10": float(np.quantile(scores, 0.10)) if len(scores) else np.nan,
        "probability_q90": float(np.quantile(scores, 0.90)) if len(scores) else np.nan,
        "window_level_metrics_are_independent_observations": False,
    }
    if support1 and support0:
        result["pr_auc_fatigue_like"] = float(average_precision_score(truth, scores))
        result["roc_auc"] = float(roc_auc_score(truth, scores))
        result["brier_score"] = float(brier_score_loss(truth, scores))
    else:
        result["pr_auc_fatigue_like"] = np.nan
        result["roc_auc"] = np.nan
        result["brier_score"] = float(brier_score_loss(truth, scores)) if len(truth) else np.nan
    return result


def _group_balanced_macro_f1(rows: pd.DataFrame, group_sample: np.ndarray | None = None) -> float:
    if group_sample is None:
        group_ids = sorted(rows.group_id.astype(str).unique())
    else:
        group_ids = list(group_sample)
    matrices = []
    for group in group_ids:
        selected = rows.loc[rows.group_id.astype(str).eq(str(group))]
        if selected.empty:
            continue
        matrices.append(confusion_matrix(
            selected.true_class_id.astype(int), selected.predicted_class_id.astype(int), labels=[0, 1]
        ).astype(float) / len(selected))
    if not matrices:
        return float("nan")
    matrix = np.mean(matrices, axis=0)
    f1s = []
    for class_id in (0, 1):
        tp = matrix[class_id, class_id]
        fp = matrix[:, class_id].sum() - tp
        fn = matrix[class_id, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s))


def _paired_group_bootstrap(a: pd.DataFrame, b: pd.DataFrame, *, seed: int = 20261113, reps: int = 2000) -> dict[str, object]:
    group_ids = sorted(set(a.group_id.astype(str)))
    if group_ids != sorted(set(b.group_id.astype(str))):
        raise ValueError("Paired bootstrap inputs have different source groups")
    deltas = []
    rng = np.random.default_rng(seed)
    a_by_group = {g: a.loc[a.group_id.astype(str).eq(g)] for g in group_ids}
    b_by_group = {g: b.loc[b.group_id.astype(str).eq(g)] for g in group_ids}
    for group in group_ids:
        left, right = a_by_group[group], b_by_group[group]
        if not left.window_key.astype(str).sort_values().reset_index(drop=True).equals(
            right.window_key.astype(str).sort_values().reset_index(drop=True)
        ):
            raise ValueError(f"Paired predictions do not use identical windows for {group}")
    for _ in range(int(reps)):
        chosen = rng.choice(group_ids, size=len(group_ids), replace=True)
        delta = _group_balanced_macro_f1(a, chosen) - _group_balanced_macro_f1(b, chosen)
        if np.isfinite(delta):
            deltas.append(float(delta))
    return {
        "delta_macro_f1_optimized_a1_minus_temporal": _group_balanced_macro_f1(a) - _group_balanced_macro_f1(b),
        "ci_95_low": float(np.quantile(deltas, 0.025)),
        "ci_95_high": float(np.quantile(deltas, 0.975)),
        "valid_replicates": len(deltas),
        "requested_replicates": int(reps),
        "bootstrap_unit": "source groups sampled with replacement; same group draw for both paired systems",
    }


def run_development_error_audit(
    run_dir: Path,
    *,
    output_subdir: str = "development_error_audit_v1",
    a1_subdir: str = "a1_lowfreq_optimization_v1",
) -> dict[str, object]:
    """Write private development-only diagnostics without scoring reserved groups."""
    run_dir = Path(run_dir).expanduser().resolve()
    private_dir = run_dir / "private"
    output_dir = private_dir / output_subdir
    if Path(output_subdir).name != output_subdir or output_subdir in {"", ".", ".."}:
        raise ValueError("output_subdir must be a single directory name")
    if Path(a1_subdir).name != a1_subdir or a1_subdir in {"", ".", ".."}:
        raise ValueError("a1_subdir must be a single directory name")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite development audit: {output_dir}")
    split_doc = json.loads((private_dir / "split_manifest.json").read_text(encoding="utf-8"))
    dev_groups = [str(value) for value in split_doc["development_group_ids"]]
    feature_dir = private_dir / "features_labels"
    features_path = feature_dir / "features.parquet"
    labels_path = feature_dir / "pseudo_labels_alternatives.parquet"
    # Filter at the Parquet reader so no reserved group feature rows are materialized.
    features = pd.read_parquet(features_path, filters=[("group_id", "in", dev_groups)])
    labels = pd.read_parquet(labels_path, filters=[("group_id", "in", dev_groups)])
    labels = labels.loc[labels.comparison_rule.eq(LABEL_RULE)].drop(columns=["comparison_rule"]).copy()
    primary_s = 0.5
    features = features.loc[np.isclose(features.duration_s.astype(float), primary_s)].copy()
    if set(features.group_id.astype(str)) != set(dev_groups) or set(labels.group_id.astype(str)) != set(dev_groups):
        raise ValueError("Feature/label audit input does not contain exactly the development groups")
    if labels.window_key.duplicated().any() or features.window_key.duplicated().any():
        raise ValueError("Window keys must be unique in the development audit inputs")

    joined = features.merge(
        labels[["window_key", "group_id", "recording_id", "class_id", "label_confidence", "label_reason"]],
        on="window_key", how="inner", suffixes=("", "_label"), validate="one_to_one",
    )
    if len(joined) != len(labels):
        raise ValueError("Some development labels have no 500 ms feature row")
    if not joined.group_id.astype(str).eq(joined.group_id_label.astype(str)).all():
        raise ValueError("Development feature and label group IDs differ")
    joined["relative_time_fraction"] = joined.groupby("recording_id", sort=False).start_time_s.transform(
        lambda values: (values - values.min()) / max(float(values.max() - values.min()), 1e-9)
    )
    joined["temporal_quartile"] = np.minimum(
        np.floor(joined.relative_time_fraction.fillna(0).to_numpy(dtype=float) * 4).astype(int), 3
    ) + 1

    output_dir.mkdir(parents=True)
    label_group = (
        labels.groupby(["group_id", "class_id"], as_index=False)
        .agg(window_count_descriptive_only=("window_key", "size"), recording_count=("recording_id", "nunique"))
    )
    label_trial = (
        labels.groupby(["group_id", "recording_id", "class_id"], as_index=False)
        .agg(window_count_descriptive_only=("window_key", "size"))
    )
    label_temporal = (
        joined.groupby(["group_id", "recording_id", "temporal_quartile", "class_id"], as_index=False)
        .agg(window_count_descriptive_only=("window_key", "size"))
    )
    label_group.to_csv(output_dir / "label_support_by_group_class.csv", index=False)
    label_trial.to_csv(output_dir / "label_support_by_trial_class.csv", index=False)
    label_temporal.to_csv(output_dir / "label_support_by_trial_time_quartile.csv", index=False)

    feature_rows = []
    for (group, class_id), rows in joined.loc[joined.class_id.isin((0, 1))].groupby(["group_id", "class_id"], sort=True):
        for feature in MORPHOLOGY_FEATURES:
            values = pd.to_numeric(rows[feature], errors="coerce").dropna().to_numpy(dtype=float)
            feature_rows.append({
                "group_id": str(group), "class_id": int(class_id), "feature": feature,
                "window_count_descriptive_only": int(len(values)),
                "median": float(np.median(values)) if len(values) else np.nan,
                "q25": float(np.quantile(values, .25)) if len(values) else np.nan,
                "q75": float(np.quantile(values, .75)) if len(values) else np.nan,
                "window_level_summary_is_independent": False,
            })
    pd.DataFrame(feature_rows).to_csv(output_dir / "morphology_feature_distribution_by_group_class.csv", index=False)

    corr_rows = []
    for (group, class_id), rows in joined.loc[joined.class_id.isin((0, 1))].groupby(["group_id", "class_id"], sort=True):
        if len(rows) < 10:
            continue
        for index, left in enumerate(MORPHOLOGY_FEATURES):
            for right in MORPHOLOGY_FEATURES[index + 1:]:
                x = pd.to_numeric(rows[left], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(rows[right], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                rho = spearmanr(x[valid], y[valid]).statistic if valid.sum() >= 10 else np.nan
                corr_rows.append({
                    "group_id": str(group), "class_id": int(class_id), "feature_left": left,
                    "feature_right": right, "spearman_rho": float(rho),
                    "n_windows_descriptive_only": int(valid.sum()),
                })
    pd.DataFrame(corr_rows).to_csv(output_dir / "morphology_redundancy_by_group_class.csv", index=False)

    acquisition_path = private_dir / "acquisition" / "acquisition_audit.csv"
    acquisition = pd.read_csv(acquisition_path)
    acquisition = acquisition.loc[acquisition.group_id.astype(str).isin(dev_groups)].copy()
    acquisition_summary = acquisition.groupby("group_id", as_index=False).agg(
        recording_count=("recording_id", "nunique"),
        median_sampling_rate_hz=("effective_sampling_rate_hz", "median"),
        minimum_sampling_rate_hz=("effective_sampling_rate_hz", "min"),
        maximum_sampling_rate_hz=("effective_sampling_rate_hz", "max"),
        raw_compatible_recordings=("raw_signal_compatible", "sum"),
        recordings_with_notch=("notch_frequency_hz", lambda values: int(values.notna().sum())),
        qc_invalid_samples=("qc_invalid_sample_count", "sum"),
        processed_segments=("processed_segment_count", "sum"),
    )
    acquisition_summary.to_csv(output_dir / "acquisition_summary_by_group.csv", index=False)

    error_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    cases = []
    loaded_models: dict[tuple[str, str, str], pd.DataFrame] = {}
    for stage_dir_name, model_label, view in MODEL_CONFIGS:
        prediction_path = private_dir / stage_dir_name / "nested_oof_predictions.parquet"
        if not prediction_path.is_file():
            if stage_dir_name == a1_subdir:
                prediction_path = private_dir / a1_subdir / "nested_oof_predictions.parquet"
            if not prediction_path.is_file():
                continue
        predictions = pd.read_parquet(prediction_path)
        selected = predictions.loc[
            predictions.feature_view.eq(view)
            & predictions.model_name.eq("logistic_regression")
            & predictions.true_class_id.isin((0, 1))
        ].copy()
        selected["audit_stage"] = stage_dir_name
        selected["audit_model"] = model_label
        loaded_models[(stage_dir_name, model_label, view)] = selected
        for threshold_strategy, rows in selected.groupby("threshold_strategy", sort=True):
            base_keys = {"stage": stage_dir_name, "model": model_label, "threshold_strategy": str(threshold_strategy)}
            error_rows.append(_metric_row(rows, keys=base_keys))
            for group, group_rows in rows.groupby("group_id", sort=True):
                error_rows.append(_metric_row(group_rows, keys={**base_keys, "group_id": str(group)}))
            for (group, recording), trial_rows in rows.groupby(["group_id", "recording_id"], sort=True):
                score_rows.append(_metric_row(trial_rows, keys={
                    **base_keys, "group_id": str(group), "recording_id": str(recording),
                }))
            for (group, class_id), class_rows in rows.groupby(["group_id", "true_class_id"], sort=True):
                score_rows.append({
                    **base_keys, "group_id": str(group), "recording_id": "",
                    "class_id": int(class_id), "score_distribution_only": True,
                    "window_count_descriptive_only": int(len(class_rows)),
                    "score_mean": float(class_rows.probability_class_1.mean()),
                    "score_median": float(class_rows.probability_class_1.median()),
                    "score_q10": float(class_rows.probability_class_1.quantile(.1)),
                    "score_q90": float(class_rows.probability_class_1.quantile(.9)),
                })
            wrong = rows.loc[rows.true_class_id.ne(rows.predicted_class_id)].copy()
            wrong["error_type"] = np.where(wrong.true_class_id.eq(1), "false_negative", "false_positive")
            cases.append(wrong[[
                "window_key", "group_id", "recording_id", "fold_id", "threshold_strategy",
                "decision_threshold", "true_class_id", "predicted_class_id", "probability_class_1",
                "error_type", "label_confidence", "label_reason", "audit_stage", "audit_model",
            ]])
    pd.DataFrame(error_rows).to_csv(output_dir / "oof_error_metrics_by_group.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output_dir / "oof_trial_and_score_distributions.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(output_dir / "oof_false_positive_false_negative_cases.csv", index=False)

    paired_bootstrap = {}
    optimized_a1 = loaded_models.get((a1_subdir, "optimized_a1", "A1_morphology_only"))
    temporal = loaded_models.get((a1_subdir, "temporal_baseline", "relative_time_only"))
    if optimized_a1 is not None and temporal is not None:
        for strategy in ("fixed_0_5", "inner_oof_selected"):
            left = optimized_a1.loc[optimized_a1.threshold_strategy.eq(strategy)].copy()
            right = temporal.loc[temporal.threshold_strategy.eq(strategy)].copy()
            keys = ["window_key", "group_id", "recording_id", "fold_id", "true_class_id"]
            check = left[keys].merge(right[keys], on="window_key", suffixes=("_a1", "_time"), validate="one_to_one")
            if len(check) != len(left) or len(left) != len(right) or any(
                not check[f"{key}_a1"].astype(str).eq(check[f"{key}_time"].astype(str)).all()
                for key in ("group_id", "recording_id", "fold_id", "true_class_id")
            ):
                raise ValueError("A1 and temporal OOF comparison is not exactly paired")
            paired_bootstrap[strategy] = _paired_group_bootstrap(left, right)
    _write_json(output_dir / "paired_group_bootstrap.json", paired_bootstrap)

    input_paths = [
        private_dir / "split_manifest.json", features_path, labels_path, acquisition_path,
        private_dir / "lowfreq_threshold_v2" / "nested_oof_predictions.parquet",
        private_dir / a1_subdir / "nested_oof_predictions.parquet",
    ]
    output_paths = sorted(path for path in output_dir.iterdir() if path.is_file())
    receipt = {
        "stage": "development_only_grouped_error_and_support_audit",
        "status": "completed",
        "label_rule": LABEL_RULE,
        "development_groups": sorted(dev_groups),
        "development_label_rows": int(len(labels)),
        "development_label_counts": {str(k): int(v) for k, v in labels.class_id.value_counts().sort_index().items()},
        "independent_unit": "source folder group; participant identity is unverified",
        "overlapping_windows_treated_as_independent": False,
        "window_level_artifacts": "descriptive only; inferential comparisons resample source groups",
        "reserved_test_feature_rows_materialized": 0,
        "reserved_test_member_payloads_read": 0,
        "internal_test_signal_values_parsed": 0,
        "external_rar_opened": False,
        "paired_group_bootstrap": paired_bootstrap,
        "input_sha256": {str(path.resolve()): _sha256(path) for path in input_paths},
        "output_sha256": {str(path.resolve()): _sha256(path) for path in output_paths},
        "interpretation": "all classification outcomes quantify agreement with the low-frequency EMG proxy, not independent physiological validation of fatigue",
    }
    manifest_path = output_dir / "development_error_audit_manifest.json"
    _write_json(manifest_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path.resolve())}
