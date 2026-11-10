"""Tabular estimators and candidate grids for grouped proxy-label evaluation."""

from __future__ import annotations

from itertools import product
from typing import Mapping

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODEL_NAMES = (
    "dummy_majority",
    "logistic_regression",
    "svm_rbf",
    "random_forest",
    "hist_gradient_boosting",
    "mlp",
)


def candidate_grid(config: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    """Expand the frozen TOML model grids into explicit candidate parameters."""
    settings = config["models"]
    assert isinstance(settings, Mapping)
    return {
        "dummy_majority": [{"strategy": "prior"}],
        "logistic_regression": [
            {"C": float(value)} for value in settings["logistic_c_values"]
        ],
        "svm_rbf": [{"C": float(value)} for value in settings["svm_c_values"]],
        "random_forest": [
            {
                "n_estimators": int(settings["random_forest_trees"]),
                "max_depth": int(depth),
            }
            for depth in settings["random_forest_max_depths"]
        ],
        "hist_gradient_boosting": [
            {"learning_rate": float(rate), "max_leaf_nodes": int(leaves)}
            for rate, leaves in product(
                settings["hist_gradient_learning_rates"],
                settings["hist_gradient_max_leaf_nodes"],
            )
        ],
        "mlp": [
            {"hidden_layer_sizes": [int(width) for width in layers], "alpha": float(alpha)}
            for layers, alpha in product(
                settings["mlp_hidden_layers"],
                settings["mlp_alpha_values"],
            )
        ],
    }


def candidate_complexity_key(model_name: str, params: Mapping[str, object]) -> tuple[float, ...]:
    """Return a deterministic complexity tie-break key (lower is simpler)."""
    if model_name == "dummy_majority":
        return (0.0,)
    if model_name in {"logistic_regression", "svm_rbf"}:
        return (float(params.get("C", 1.0)),)
    if model_name == "random_forest":
        depth = int(params.get("max_depth", 0))
        return (float(depth if depth > 0 else 1_000_000),)
    if model_name == "hist_gradient_boosting":
        return (float(params.get("max_leaf_nodes", 31)), -float(params.get("learning_rate", 0.1)))
    if model_name == "mlp":
        widths = params.get("hidden_layer_sizes", (32,))
        if isinstance(widths, int):
            widths = (widths,)
        return (float(sum(int(width) for width in widths)), float(len(widths)), -float(params.get("alpha", 0.001)))
    raise ValueError(f"Unsupported tabular model: {model_name}")


def build_tabular_model(model_name: str, params: Mapping[str, object], seed: int) -> Pipeline:
    """Build a fresh estimator whose imputation/scaling is fit inside each fold."""
    model_seed = int(seed)
    scaled = model_name in {"logistic_regression", "svm_rbf", "mlp"}
    if model_name == "dummy_majority":
        estimator = DummyClassifier(strategy=str(params.get("strategy", "prior")))
    elif model_name == "logistic_regression":
        estimator = LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight="balanced",
            max_iter=int(params.get("max_iter", 1000)),
            random_state=model_seed,
        )
    elif model_name == "svm_rbf":
        svm = SVC(
            C=float(params.get("C", 1.0)),
            kernel="rbf",
            class_weight="balanced",
            random_state=model_seed,
        )
        base_estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", svm),
            ]
        )
        calibrated = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=3,
            ensemble=False,
        )
        return Pipeline([("model", calibrated)])
    elif model_name == "random_forest":
        configured_depth = int(params.get("max_depth", 0))
        estimator = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=configured_depth if configured_depth > 0 else None,
            class_weight="balanced_subsample",
            random_state=model_seed,
            n_jobs=1,
        )
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
            random_state=model_seed,
        )
    elif model_name == "mlp":
        widths = params.get("hidden_layer_sizes", (32,))
        if isinstance(widths, int):
            hidden_layer_sizes = (int(widths),)
        else:
            hidden_layer_sizes = tuple(int(width) for width in widths)
        estimator = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=float(params.get("alpha", 0.001)),
            max_iter=int(params.get("max_iter", 500)),
            early_stopping=False,
            random_state=model_seed,
        )
    else:
        raise ValueError(f"Unsupported tabular model: {model_name}")

    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)
