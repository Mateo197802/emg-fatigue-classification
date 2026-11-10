import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _estimator_factory(_seed):
    return make_pipeline(StandardScaler(), LogisticRegression())


def test_prime_mover_experts_route_to_known_task_specialist_without_subject_overlap():
    from experiments.reference_fatigue.prime_mover_specialists import fit_prime_mover_experts

    train = pd.DataFrame(
        [
            {"window_id": f"train-{mover}-{label}", "subject_id": f"train-{mover}", "trial_id": f"train-{mover}",
             "prime_mover": mover, "label": label, "signal_feature": float(label)}
            for mover in ("biceps", "deltoid")
            for label in (0, 1)
        ]
    )
    test = pd.DataFrame(
        [
            {"window_id": f"test-{mover}-{label}", "subject_id": "held-out", "trial_id": f"test-{mover}",
             "prime_mover": mover, "label": label, "signal_feature": float(label)}
            for mover in ("biceps", "deltoid")
            for label in (0, 1)
        ]
    )

    result = fit_prime_mover_experts(
        train, test, ["signal_feature"], estimator_factory=_estimator_factory
    )

    assert set(result.window_id) == set(test.window_id)
    assert result.subject_id.eq("held-out").all()
    assert result.model.eq("specialist_by_prime_mover").all()
    assert result.expert_id.tolist() == ["biceps", "biceps", "deltoid", "deltoid"]
    assert np.isfinite(result.score).all()


def test_prime_mover_expert_rejects_subject_overlap_and_missing_class():
    from experiments.reference_fatigue.prime_mover_specialists import fit_prime_mover_experts

    train = pd.DataFrame(
        [{"window_id": "a", "subject_id": "same", "trial_id": "train", "prime_mover": "biceps", "label": 0, "x": 0.0},
         {"window_id": "b", "subject_id": "same", "trial_id": "train", "prime_mover": "biceps", "label": 1, "x": 1.0}]
    )
    test = train.iloc[[0]].copy()
    with pytest.raises(ValueError, match="participant overlap"):
        fit_prime_mover_experts(train, test, ["x"], estimator_factory=_estimator_factory)

    test.loc[:, "subject_id"] = "held-out"
    with pytest.raises(ValueError, match="both fatigue labels"):
        fit_prime_mover_experts(
            train.iloc[[0]], test, ["x"], estimator_factory=_estimator_factory
        )
