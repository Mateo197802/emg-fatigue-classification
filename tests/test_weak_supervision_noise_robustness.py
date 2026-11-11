import numpy as np
import pytest

pytest.importorskip("torch")

from emg_fatigue.weak_supervision.evaluation_noise_robustness import _morphology_batch
from emg_fatigue.weak_supervision.features import extract_window_features
from emg_fatigue.weak_supervision.evaluation_guardrails import OOD_FEATURES


def test_vectorized_morphology_matches_single_window_features():
    time = np.arange(500, dtype=float) / 1000.0
    windows = np.stack(
        [
            np.sin(2 * np.pi * 65 * time) + 0.25 * np.sin(2 * np.pi * 3 * time),
            0.2 * np.cos(2 * np.pi * 110 * time + 0.4) + 0.04 * np.sin(2 * np.pi * 8 * time),
        ]
    )

    vectorized = _morphology_batch(windows, 1000.0)
    reference = [extract_window_features(row, 1000.0) for row in windows]

    for feature in OOD_FEATURES:
        expected = np.asarray([row[feature] for row in reference], dtype=float)
        np.testing.assert_allclose(vectorized[feature].to_numpy(dtype=float), expected, rtol=1e-11, atol=1e-11)
