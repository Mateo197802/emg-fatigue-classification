"""Grouped reference-backed fatigue and movement-phase model experiments.

The evaluator is fail-closed for empirical use. Synthetic fixtures may exercise
the architecture, but their outputs carry ``synthetic_only`` provenance and are
never written as experimental results.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .contracts import ReferenceStudyConfig
from .evaluation import (
    binary_metrics,
    confusion_matrix_table,
    paired_group_differences,
    summarize_grouped_predictions,
)


@dataclass(frozen=True)
class PhaseOOFResult:
    """Cross-fitted extension/flexion probabilities for grouped training rows."""

    probabilities: np.ndarray
    folds: tuple[dict, ...]
    oof_count: int


@dataclass(frozen=True)
class ReferenceEvaluationBundle:
    """Matched-support OOF predictions and group-level fold audit."""

    fatigue_predictions: pd.DataFrame
    phase_predictions: pd.DataFrame
    fatigue_metrics: pd.DataFrame
    phase_metrics: pd.DataFrame
    fold_audit: pd.DataFrame
    evidence_status: str


def build_reference_evidence_tables(bundle, *, seed=20261109, bootstrap_replicates=5000):
    """Create summaries, paired comparisons, and confusion tables from OOF rows."""
    if not isinstance(bundle, ReferenceEvaluationBundle):
        raise ValueError("bundle must be a ReferenceEvaluationBundle")
    tables = {
        "fatigue_group_summary": pd.DataFrame(),
        "phase_group_summary": pd.DataFrame(),
        "fatigue_paired_differences": pd.DataFrame(),
        "fatigue_confusion_matrix": pd.DataFrame(),
        "phase_confusion_matrix": pd.DataFrame(),
    }
    if not bundle.fatigue_predictions.empty:
        tables["fatigue_group_summary"] = summarize_grouped_predictions(
            bundle.fatigue_predictions,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        tables["fatigue_confusion_matrix"] = confusion_matrix_table(
            bundle.fatigue_predictions
        )
        available = set(bundle.fatigue_predictions["architecture"].unique())
        candidates = [
            name for name in ("phase_then_fatigue", "phase_experts")
            if name in available
        ]
        if "logistic_all" in available and candidates:
            tables["fatigue_paired_differences"] = paired_group_differences(
                bundle.fatigue_predictions,
                baseline="logistic_all",
                candidates=candidates,
                metrics=("accuracy", "f1_macro"),
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
    if not bundle.phase_predictions.empty:
        tables["phase_group_summary"] = summarize_grouped_predictions(
            bundle.phase_predictions,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        tables["phase_confusion_matrix"] = confusion_matrix_table(
            bundle.phase_predictions
        )
    for frame in tables.values():
        if not frame.empty:
            frame["evidence_status"] = bundle.evidence_status
    return tables


def _as_feature_matrix(features):
    if isinstance(features, pd.DataFrame):
        if features.columns.duplicated().any():
            raise ValueError("feature names must be unique")
        try:
            values = features.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("features must be numeric") from error
        if values.ndim != 2 or not values.shape[1]:
            raise ValueError("features must contain at least one column")
        columns = tuple(str(name) for name in features.columns)
        matrix = pd.DataFrame(values, columns=columns)
    else:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or not values.shape[1]:
            raise ValueError("features must be a nonempty two-dimensional matrix")
        matrix = pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("features must be finite; invalid windows must be excluded before fitting")
    return matrix


def _binary_labels(values, name):
    labels = np.asarray(values)
    if labels.ndim != 1 or not set(np.unique(labels)).issubset({0, 1}) or not len(labels):
        raise ValueError(f"{name} must contain only eligible binary labels [0, 1]")
    labels = labels.astype(int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{name} training support must contain both classes [0, 1]")
    return labels


def _groups(values, length):
    result = np.asarray(values, dtype=object)
    if result.ndim != 1 or len(result) != length:
        raise ValueError("groups must match the number of feature rows")
    if pd.isna(result).any():
        raise ValueError("group identifiers must be present and nonmissing")
    if any(
        isinstance(value, bool) or not isinstance(value, (str, int, np.integer))
        for value in result
    ):
        raise ValueError("group identifiers must be scalar strings or integers")
    kinds = {"string" if isinstance(value, str) else "integer" for value in result}
    if len(kinds) > 1:
        raise ValueError("group identifiers must use one identifier type")
    return result


def _phase_labels(values, length):
    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1 or len(labels) != length:
        raise ValueError("phase labels must match the number of feature rows")
    encoded = np.full(length, -1, dtype=int)
    for index, value in enumerate(labels):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "extension":
                encoded[index] = 0
            elif normalized == "flexion":
                encoded[index] = 1
            elif normalized in {"", "transition", "uncertain", "abstain", "missing"}:
                continue
            else:
                raise ValueError("phase labels must be extension, flexion, or an explicit abstention")
        elif value in {0, 1}:
            encoded[index] = int(value)
        elif value == -1:
            continue
        else:
            raise ValueError("numeric phase labels must use 0=extension, 1=flexion, -1=abstain")
    if not np.any(encoded >= 0):
        raise ValueError("no eligible extension/flexion labels are available")
    return encoded


def _phase_estimator(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
    )


def _probabilities(estimator, features):
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("reference architectures require estimators with predict_proba")
    raw = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = np.asarray(estimator.classes_)
    if raw.ndim != 2 or raw.shape[1] != len(classes) or set(classes.tolist()) != {0, 1}:
        raise ValueError("fitted estimator must expose both classes [0, 1]")
    return raw[:, list(classes).index(0)], raw[:, list(classes).index(1)]


def grouped_oof_phase_probabilities(
    features,
    phase_labels,
    groups,
    *,
    estimator_factory=None,
    n_splits=4,
    seed=20261109,
):
    """Fit phase models on disjoint groups and return one OOF probability pair per row."""
    X = _as_feature_matrix(features)
    y = _binary_labels(phase_labels, "phase labels")
    group_values = _groups(groups, len(X))
    unique_groups = np.unique(group_values)
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError("phase OOF requires 2..n_groups splits")
    factory = estimator_factory or _phase_estimator
    probabilities = np.full((len(X), 2), np.nan, dtype=float)
    counts = np.zeros(len(X), dtype=int)
    folds = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X, y, group_values), start=1):
        if set(group_values[train_idx]) & set(group_values[valid_idx]):
            raise RuntimeError("phase OOF split contains group leakage")
        if set(np.unique(y[train_idx])) != {0, 1}:
            raise ValueError("phase OOF training fold must contain both classes")
        estimator = factory(seed + fold_index)
        estimator.fit(X.iloc[train_idx], y[train_idx])
        p0, p1 = _probabilities(estimator, X.iloc[valid_idx])
        probabilities[valid_idx, 0] = p0
        probabilities[valid_idx, 1] = p1
        counts[valid_idx] += 1
        folds.append({
            "fold_id": f"P{fold_index:02d}",
            "train_groups": tuple(sorted(set(group_values[train_idx]))),
            "validation_groups": tuple(sorted(set(group_values[valid_idx]))),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
        })
    if not np.all(counts == 1) or not np.isfinite(probabilities).all():
        raise RuntimeError("each phase-training row must receive exactly one finite OOF prediction")
    return PhaseOOFResult(probabilities, tuple(folds), int(counts.sum()))


def _fit_family(family, features, labels, seed):
    if family == "majority":
        estimator = DummyClassifier(strategy="prior")
    elif family in {"logistic_rms_mdf", "logistic_all"}:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
        )
    elif family == "svm_rbf":
        estimator = make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
        )
    elif family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"unknown reference model family: {family}")
    estimator.fit(features, labels)
    return estimator


def _family_features(features, family):
    if family != "logistic_rms_mdf":
        return features
    if {"rms", "mdf_hz"}.issubset(features.columns):
        return features[["rms", "mdf_hz"]]
    raise ValueError("RMS+MDF baseline requires columns named rms and mdf_hz")


def _family_positive_score(estimator, features):
    if hasattr(estimator, "predict_proba"):
        _, score = _probabilities(estimator, features)
        return score, "native_probability_uncalibrated"
    if hasattr(estimator, "decision_function"):
        margin = np.asarray(estimator.decision_function(features), dtype=float)
        score = 1.0 / (1.0 + np.exp(-np.clip(margin, -700, 700)))
        return score, "sigmoid_margin_uncalibrated_score"
    raise ValueError("model must produce probabilities or a decision margin")


def _append_predictions(
    rows, architecture, fold_id, row_indices, groups, truth, scores,
    evidence_status, score_type="native_probability_uncalibrated",
):
    scores = np.asarray(scores, dtype=float)
    prediction = (scores >= 0.5).astype(int)
    for local_index, row_index in enumerate(row_indices):
        rows.append({
            "architecture": architecture,
            "fold_id": fold_id,
            "row_index": int(row_index),
            "group_id": groups[row_index],
            "y_true": int(truth[local_index]),
            "y_pred": int(prediction[local_index]),
            "score": float(scores[local_index]),
            "threshold": 0.5,
            "score_type": score_type,
            "evidence_status": evidence_status,
        })
    return prediction


def evaluate_direct_fatigue(
    features,
    fatigue_labels,
    groups,
    *,
    config: ReferenceStudyConfig,
    fatigue_reference_approved=False,
    participant_ids_verified=False,
    synthetic_fixture=False,
    seed=20261109,
):
    """Evaluate direct fatigue baselines using only the fatigue reference gate."""
    if not isinstance(config, ReferenceStudyConfig):
        raise ValueError("config must be a ReferenceStudyConfig")
    if synthetic_fixture is True:
        evidence_status = "synthetic_only"
    else:
        config.require_fatigue_training_ready(
            fatigue_reference_approved=fatigue_reference_approved,
            participant_ids_verified=participant_ids_verified,
        )
        evidence_status = "reference_gated_training"

    X = _as_feature_matrix(features)
    y = _binary_labels(fatigue_labels, "fatigue labels")
    if len(y) != len(X):
        raise ValueError("fatigue labels must match the number of feature rows")
    group_values = _groups(groups, len(X))
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 4:
        raise ValueError("grouped direct-fatigue evaluation requires at least four groups")

    families = ("majority", "logistic_rms_mdf", "logistic_all", "svm_rbf", "random_forest")
    prediction_rows = []
    metric_rows = []
    fold_rows = []
    splitter = GroupKFold(n_splits=len(unique_groups))
    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, group_values), start=1
    ):
        train_groups = set(group_values[train_idx])
        test_groups = set(group_values[test_idx])
        if train_groups & test_groups:
            raise RuntimeError("direct-fatigue split contains group leakage")
        if set(np.unique(y[train_idx])) != {0, 1}:
            raise ValueError("direct-fatigue training fold must contain both classes")

        fold_id = f"F{fold_index:02d}"
        for family in families:
            train_features = _family_features(X.iloc[train_idx], family)
            test_features = _family_features(X.iloc[test_idx], family)
            estimator = _fit_family(family, train_features, y[train_idx], seed + fold_index)
            scores, score_type = _family_positive_score(estimator, test_features)
            predictions = _append_predictions(
                prediction_rows,
                family,
                fold_id,
                test_idx,
                group_values,
                y[test_idx],
                scores,
                evidence_status,
                score_type,
            )
            metric_rows.append({
                "architecture": family,
                "fold_id": fold_id,
                "held_out_group": tuple(sorted(test_groups)),
                **binary_metrics(y[test_idx], predictions, scores),
            })
        fold_rows.append({
            "fold_id": fold_id,
            "train_groups": tuple(sorted(train_groups)),
            "test_groups": tuple(sorted(test_groups)),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "evidence_status": evidence_status,
        })

    predictions = pd.DataFrame(prediction_rows)
    if predictions.duplicated(["architecture", "row_index"]).any():
        raise RuntimeError("duplicate direct-fatigue OOF prediction")
    counts = predictions.groupby("architecture").size()
    if set(counts.index) != set(families) or not (counts == len(X)).all():
        raise RuntimeError("direct-fatigue models do not have matched complete OOF support")
    return ReferenceEvaluationBundle(
        fatigue_predictions=predictions,
        phase_predictions=pd.DataFrame(),
        fatigue_metrics=pd.DataFrame(metric_rows),
        phase_metrics=pd.DataFrame(),
        fold_audit=pd.DataFrame(fold_rows),
        evidence_status=evidence_status,
    )


def evaluate_movement_phase(
    features,
    phase_labels,
    groups,
    *,
    config: ReferenceStudyConfig,
    phase_reference_approved=False,
    participant_ids_verified=False,
    synthetic_fixture=False,
    seed=20261109,
):
    """Evaluate extension/flexion classifiers using only the phase reference gate."""
    if not isinstance(config, ReferenceStudyConfig):
        raise ValueError("config must be a ReferenceStudyConfig")
    if synthetic_fixture is True:
        evidence_status = "synthetic_only"
    else:
        config.require_phase_training_ready(
            phase_reference_approved=phase_reference_approved,
            participant_ids_verified=participant_ids_verified,
        )
        evidence_status = "reference_gated_training"

    X = _as_feature_matrix(features)
    encoded = _phase_labels(phase_labels, len(X))
    group_values = _groups(groups, len(X))
    eligible_idx = np.flatnonzero(encoded >= 0)
    y = encoded[eligible_idx]
    phase_groups = group_values[eligible_idx]
    unique_groups = np.unique(phase_groups)
    if len(unique_groups) < 4:
        raise ValueError("grouped movement-phase evaluation requires at least four labeled groups")

    families = ("majority", "logistic_all", "svm_rbf", "random_forest")
    prediction_rows = []
    metric_rows = []
    fold_rows = []
    splitter = GroupKFold(n_splits=len(unique_groups))
    X_eligible = X.iloc[eligible_idx]
    for fold_index, (train_local, test_local) in enumerate(
        splitter.split(X_eligible, y, phase_groups), start=1
    ):
        train_idx = eligible_idx[train_local]
        test_idx = eligible_idx[test_local]
        train_groups = set(phase_groups[train_local])
        test_groups = set(phase_groups[test_local])
        if train_groups & test_groups:
            raise RuntimeError("movement-phase split contains group leakage")
        if set(np.unique(y[train_local])) != {0, 1}:
            raise ValueError("movement-phase training fold must contain extension and flexion")

        fold_id = f"P{fold_index:02d}"
        for family in families:
            estimator = _fit_family(family, X.iloc[train_idx], y[train_local], seed + fold_index)
            scores, score_type = _family_positive_score(estimator, X.iloc[test_idx])
            predictions = _append_predictions(
                prediction_rows,
                family,
                fold_id,
                test_idx,
                group_values,
                y[test_local],
                scores,
                evidence_status,
                score_type,
            )
            metric_rows.append({
                "architecture": family,
                "fold_id": fold_id,
                "held_out_group": tuple(sorted(test_groups)),
                **binary_metrics(y[test_local], predictions, scores),
            })
        fold_rows.append({
            "fold_id": fold_id,
            "train_groups": tuple(sorted(train_groups)),
            "test_groups": tuple(sorted(test_groups)),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "evidence_status": evidence_status,
        })

    predictions = pd.DataFrame(prediction_rows)
    if predictions.duplicated(["architecture", "row_index"]).any():
        raise RuntimeError("duplicate movement-phase OOF prediction")
    counts = predictions.groupby("architecture").size()
    if set(counts.index) != set(families) or not (counts == len(eligible_idx)).all():
        raise RuntimeError("movement-phase models do not have matched complete labeled OOF support")
    return ReferenceEvaluationBundle(
        fatigue_predictions=pd.DataFrame(),
        phase_predictions=predictions,
        fatigue_metrics=pd.DataFrame(),
        phase_metrics=pd.DataFrame(metric_rows),
        fold_audit=pd.DataFrame(fold_rows),
        evidence_status=evidence_status,
    )


def evaluate_reference_architectures(
    features,
    fatigue_labels,
    phase_labels,
    groups,
    *,
    config: ReferenceStudyConfig,
    fatigue_reference_approved=False,
    phase_reference_approved=False,
    participant_ids_verified=False,
    synthetic_fixture=False,
    inner_splits=4,
    seed=20261109,
):
    """Compare direct baselines, an OOF phase cascade, and soft phase experts.

    Empirical execution requires all three independent-reference gates. The
    synthetic escape hatch is for unit tests only; no files are written and all
    returned rows carry ``synthetic_only`` provenance.
    """
    if not isinstance(config, ReferenceStudyConfig):
        raise ValueError("config must be a ReferenceStudyConfig")
    if synthetic_fixture is True:
        evidence_status = "synthetic_only"
    else:
        config.require_phase_conditioned_training_ready(
            fatigue_reference_approved=fatigue_reference_approved,
            phase_reference_approved=phase_reference_approved,
            participant_ids_verified=participant_ids_verified,
        )
        evidence_status = "reference_gated_training"

    X = _as_feature_matrix(features)
    y_fatigue = _binary_labels(fatigue_labels, "fatigue labels")
    y_phase = _phase_labels(phase_labels, len(X))
    group_values = _groups(groups, len(X))
    if len(y_fatigue) != len(X):
        raise ValueError("fatigue labels must match the number of feature rows")
    eligible_idx = np.flatnonzero(y_phase >= 0)
    if len(eligible_idx) == 0:
        raise ValueError("no matched fatigue and extension/flexion rows are available")
    matched_groups = group_values[eligible_idx]
    unique_groups = np.unique(matched_groups)
    if len(unique_groups) < 4:
        raise ValueError("matched outer and inner group evaluation requires at least four eligible groups")
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least two")

    fatigue_rows = []
    phase_rows = []
    fatigue_metric_rows = []
    phase_metric_rows = []
    fold_rows = []
    outer = GroupKFold(n_splits=len(unique_groups))
    families = ("majority", "logistic_rms_mdf", "logistic_all", "svm_rbf", "random_forest")

    matched_X = X.iloc[eligible_idx]
    matched_y_fatigue = y_fatigue[eligible_idx]
    matched_y_phase = y_phase[eligible_idx]
    for fold_index, (train_local, test_local) in enumerate(
        outer.split(matched_X, matched_y_fatigue, matched_groups), start=1
    ):
        train_idx = eligible_idx[train_local]
        test_idx = eligible_idx[test_local]
        fold_id = f"F{fold_index:02d}"
        train_groups = set(matched_groups[train_local])
        test_groups = set(matched_groups[test_local])
        if train_groups & test_groups:
            raise RuntimeError("outer split contains group leakage")
        if set(np.unique(y_fatigue[train_idx])) != {0, 1}:
            raise ValueError("fatigue training fold must contain both classes")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_fatigue[train_idx], y_fatigue[test_idx]

        for family in families:
            family_train = _family_features(X_train, family)
            family_test = _family_features(X_test, family)
            estimator = _fit_family(family, family_train, y_train, seed + fold_index)
            scores, score_type = _family_positive_score(estimator, family_test)
            predictions = _append_predictions(
                fatigue_rows, family, fold_id, test_idx, group_values, y_test, scores,
                evidence_status, score_type,
            )
            fatigue_metric_rows.append({
                "architecture": family,
                "fold_id": fold_id,
                "held_out_group": tuple(sorted(test_groups)),
                **binary_metrics(y_test, predictions, scores),
            })

        phase_train_idx = train_idx
        phase_test_idx = test_idx
        if len(np.unique(y_phase[phase_train_idx])) != 2:
            raise ValueError("outer phase-training fold must contain extension and flexion")
        phase_train_groups = group_values[phase_train_idx]
        phase_splits = min(inner_splits, len(np.unique(phase_train_groups)))
        phase_oof = grouped_oof_phase_probabilities(
            X.iloc[phase_train_idx],
            y_phase[phase_train_idx],
            phase_train_groups,
            n_splits=phase_splits,
            seed=seed + fold_index * 100,
        )

        phase_estimator = _phase_estimator(seed + fold_index * 1000)
        phase_estimator.fit(X.iloc[phase_train_idx], y_phase[phase_train_idx])
        p_extension_test, p_flexion_test = _probabilities(phase_estimator, X.iloc[test_idx])
        phase_scores = p_flexion_test
        phase_truth = y_phase[phase_test_idx]
        phase_pred = (phase_scores >= 0.5).astype(int)
        for local_index, row_index in enumerate(phase_test_idx):
            phase_rows.append({
                "architecture": "phase_classifier",
                "fold_id": fold_id,
                "row_index": int(row_index),
                "group_id": group_values[row_index],
                "y_true": int(phase_truth[local_index]),
                "y_pred": int(phase_pred[local_index]),
                "score": float(phase_scores[local_index]),
                "threshold": 0.5,
                "score_type": "native_probability_uncalibrated",
                "evidence_status": evidence_status,
            })
        phase_metric_rows.append({
            "architecture": "phase_classifier",
            "fold_id": fold_id,
            "held_out_group": tuple(sorted(test_groups)),
            **binary_metrics(phase_truth, phase_pred, phase_scores),
        })

        # Stage 2 learns only from cross-fitted phase probabilities in the
        # outer-training groups; test probabilities come from that outer-fit.
        phase_probability_by_row = {int(row_index): phase_oof.probabilities[i] for i, row_index in enumerate(phase_train_idx)}
        cascade_train_idx = np.asarray(sorted(phase_probability_by_row), dtype=int)
        cascade_train_probabilities = np.vstack([phase_probability_by_row[i] for i in cascade_train_idx])
        cascade_train_features = np.hstack([X.iloc[cascade_train_idx].to_numpy(), cascade_train_probabilities])
        cascade_test_features = np.hstack([X.iloc[test_idx].to_numpy(), np.column_stack([p_extension_test, p_flexion_test])])
        if set(np.unique(y_fatigue[cascade_train_idx])) != {0, 1}:
            raise ValueError("cascade training support must contain both fatigue classes")
        cascade = _phase_estimator(seed + fold_index * 2000)
        cascade.fit(cascade_train_features, y_fatigue[cascade_train_idx])
        _, cascade_scores = _probabilities(cascade, cascade_test_features)
        cascade_prediction = _append_predictions(
            fatigue_rows, "phase_then_fatigue", fold_id, test_idx, group_values, y_test, cascade_scores, evidence_status
        )
        fatigue_metric_rows.append({
            "architecture": "phase_then_fatigue",
            "fold_id": fold_id,
            "held_out_group": tuple(sorted(test_groups)),
            **binary_metrics(y_test, cascade_prediction, cascade_scores),
        })

        specialist_scores = []
        for phase_value in (0, 1):
            specialist_idx = phase_train_idx[y_phase[phase_train_idx] == phase_value]
            if set(np.unique(y_fatigue[specialist_idx])) != {0, 1}:
                name = "extension" if phase_value == 0 else "flexion"
                raise ValueError(f"{name} fatigue specialist requires both fatigue classes")
            specialist = _phase_estimator(seed + fold_index * 3000 + phase_value)
            specialist.fit(X.iloc[specialist_idx], y_fatigue[specialist_idx])
            _, specialist_probability = _probabilities(specialist, X_test)
            specialist_scores.append(specialist_probability)
        expert_scores = p_extension_test * specialist_scores[0] + p_flexion_test * specialist_scores[1]
        expert_prediction = _append_predictions(
            fatigue_rows, "phase_experts", fold_id, test_idx, group_values, y_test, expert_scores, evidence_status
        )
        fatigue_metric_rows.append({
            "architecture": "phase_experts",
            "fold_id": fold_id,
            "held_out_group": tuple(sorted(test_groups)),
            **binary_metrics(y_test, expert_prediction, expert_scores),
        })

        fold_rows.append({
            "fold_id": fold_id,
            "train_groups": tuple(sorted(train_groups)),
            "test_groups": tuple(sorted(test_groups)),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "excluded_no_valid_phase_rows": int(
                np.count_nonzero(np.isin(group_values, list(test_groups)))
                - len(test_idx)
            ),
            "phase_train_rows": int(len(phase_train_idx)),
            "cascade_train_rows": int(len(cascade_train_idx)),
            "phase_test_rows": int(len(phase_test_idx)),
            "phase_oof_rows": int(phase_oof.oof_count),
            "evidence_status": evidence_status,
        })

    fatigue_predictions = pd.DataFrame(fatigue_rows)
    phase_predictions = pd.DataFrame(phase_rows)
    if fatigue_predictions.duplicated(["architecture", "row_index"]).any():
        raise RuntimeError("duplicate outer-fold fatigue predictions")
    expected_rows = len(eligible_idx) * (len(families) + 2)
    counts = fatigue_predictions.groupby("architecture").size()
    if len(counts) != len(families) + 2 or not (counts == len(eligible_idx)).all() or len(fatigue_predictions) != expected_rows:
        raise RuntimeError("fatigue architectures do not have matched complete OOF support")
    return ReferenceEvaluationBundle(
        fatigue_predictions=fatigue_predictions,
        phase_predictions=phase_predictions,
        fatigue_metrics=pd.DataFrame(fatigue_metric_rows),
        phase_metrics=pd.DataFrame(phase_metric_rows),
        fold_audit=pd.DataFrame(fold_rows),
        evidence_status=evidence_status,
    )
