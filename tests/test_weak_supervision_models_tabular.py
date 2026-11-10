from __future__ import annotations

from pathlib import Path
import tomllib
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from emg_fatigue.weak_supervision.models_tabular import (
    build_tabular_model,
    candidate_grid,
    candidate_complexity_key,
)


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "fatigue_proxy.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_candidate_grid_covers_required_families_and_configured_values():
    grid = candidate_grid(_config())

    assert set(grid) == {
        "dummy_majority",
        "logistic_regression",
        "svm_rbf",
        "random_forest",
        "hist_gradient_boosting",
        "mlp",
    }
    assert [item["C"] for item in grid["logistic_regression"]] == [0.1, 1.0, 10.0]
    assert [item["C"] for item in grid["svm_rbf"]] == [0.1, 1.0, 10.0]
    assert [item["max_depth"] for item in grid["random_forest"]] == [4, 8, 0]
    assert len(grid["hist_gradient_boosting"]) == 4
    assert len(grid["mlp"]) == 4


def test_candidate_builders_fit_real_data_and_impute_using_the_training_pipeline():
    x_train = np.asarray(
        [[-3.0, 1.0], [-2.0, np.nan], [-1.0, 2.0], [1.0, 3.0], [2.0, np.nan], [3.0, 4.0]]
    )
    y_train = np.asarray([0, 0, 0, 1, 1, 1])
    x_valid = np.asarray([[-2.5, np.nan], [2.5, 3.5]])
    grid = candidate_grid(_config())

    for name, params_list in grid.items():
        params = dict(params_list[0])
        params.update({"n_estimators": 8, "max_iter": 500})
        model = build_tabular_model(name, params, seed=17)
        assert isinstance(model, Pipeline)
        if name == "svm_rbf":
            calibrated = model.named_steps["model"]
            assert isinstance(calibrated, CalibratedClassifierCV)
            assert isinstance(calibrated.estimator.named_steps["imputer"], SimpleImputer)
        else:
            assert isinstance(model.named_steps["imputer"], SimpleImputer)
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            model.fit(x_train, y_train)
        assert all(issubclass(item.category, ConvergenceWarning) for item in fit_warnings)
        probabilities = model.predict_proba(x_valid)
        assert probabilities.shape == (2, 2)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_complexity_ties_prefer_regularized_or_smaller_candidates():
    assert candidate_complexity_key("logistic_regression", {"C": 0.1}) < candidate_complexity_key(
        "logistic_regression", {"C": 10.0}
    )
    assert candidate_complexity_key("random_forest", {"max_depth": 4}) < candidate_complexity_key(
        "random_forest", {"max_depth": 8}
    )
    assert candidate_complexity_key("mlp", {"hidden_layer_sizes": [32], "alpha": 0.001}) < candidate_complexity_key(
        "mlp", {"hidden_layer_sizes": [64, 32], "alpha": 0.001}
    )


def test_rbf_svm_uses_the_supported_fold_local_probability_calibrator():
    x_train = np.asarray([[-2.0, 1.0], [-1.0, 2.0], [1.0, 3.0], [2.0, 4.0], [-3.0, 1.5], [3.0, 4.5]])
    y_train = np.asarray([0, 0, 1, 1, 0, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = build_tabular_model("svm_rbf", {"C": 1.0}, seed=4)
        model.fit(x_train, y_train)

    assert isinstance(model.named_steps["model"], CalibratedClassifierCV)
    assert model.named_steps["model"].ensemble is False
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]
