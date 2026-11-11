from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emg_fatigue.weak_supervision.external_provenance import code_provenance
from emg_fatigue.weak_supervision.external_supervision import _build_one_trial


def _raw_trial(start_s: float = 0.0) -> pd.DataFrame:
    time = np.arange(0.0, 8.101, 0.001, dtype=np.float64) + float(start_s)
    values = [np.sin(2 * np.pi * (40 + 10 * channel) * time) for channel in range(4)]
    columns = []
    for channel, signal in enumerate(values):
        columns.extend([time, signal])
    return pd.DataFrame({f"column_{index}": value for index, value in enumerate(columns)})


def test_external_window_builder_verifies_rating_clock_alignment():
    label_time = np.arange(0.0, 8.1, 0.02, dtype=np.float64)
    label_values = np.where(label_time < 4.0, 0, 2)
    labels = pd.DataFrame({"time_s": label_time, "label": label_values})

    rows, windows, receipt = _build_one_trial(1, 1, _raw_trial(), labels)

    assert len(rows) == 2
    assert len(windows) == 2
    assert receipt["start_offset_s"] == pytest.approx(0.0)
    assert receipt["label_rate_hz"] == pytest.approx(50.0)

    misaligned = labels.assign(time_s=labels["time_s"] + 0.1)
    with pytest.raises(ValueError, match="do not share a verified start"):
        _build_one_trial(1, 1, _raw_trial(), misaligned)


def test_external_code_provenance_contains_hashes_and_runtime():
    provenance = code_provenance(__file__)

    assert provenance["git_commit"]
    assert provenance["python_version"]
    assert provenance["package_versions"]["numpy"]
    hashes = provenance["sha256"]
    assert "tests/test_external_provenance.py" in hashes
    assert "tools/run_external_emg.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())
