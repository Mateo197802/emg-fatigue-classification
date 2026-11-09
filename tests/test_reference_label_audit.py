import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from emg_fatigue.reference_labels import (
    ReferenceContract,
    fatigue_labels_from_torque,
    phase_from_flexion_angle,
)


def _contract(kind, level, *, angle_convention=None, repeatability_id=None, synthetic=False):
    return ReferenceContract(
        reference_kind=kind,
        reference_level=level,
        source_id=f"{kind}-source-01",
        unit="N*m" if kind == "torque" else "deg",
        time_axis_id=f"{kind}-clock",
        emg_time_axis_id="emg-clock",
        participant_id="participant-opaque-01",
        session_id="session-opaque-01",
        calibration_verified=True,
        calibration_id=f"{kind}-calibration-01",
        synchronized=True,
        synchronization_id="sync-01",
        synchronization_residual_s=0.001,
        synchronization_tolerance_s=0.01,
        label_definition_id="approved-definition-v1",
        label_definition_approved=not synthetic,
        approval_record_id=None if synthetic else "approval-record-01",
        synthetic_fixture=synthetic,
        repeatability_id=repeatability_id,
        angle_convention=angle_convention,
    )


def _results(synthetic=False):
    fatigue = fatigue_labels_from_torque(
        [100, 101, 99, 70, 70, 70],
        reference_contract=_contract(
            "torque", "repetition", repeatability_id="repeatability-01", synthetic=synthetic
        ),
        repetition_time_s=np.arange(6, dtype=float),
        sem_nm=1.0,
        valid_reps=np.ones(6, dtype=bool),
        angle_matched=np.ones(6, dtype=bool),
    )
    phase = phase_from_flexion_angle(
        [0, 1, 2, 3, 4, 5],
        [0, 5, 10, 10, 5, 0],
        velocity_deadband_deg_s=3.0,
        reference_contract=_contract(
            "knee_angle", "time_series", angle_convention="flexion_positive", synthetic=synthetic
        ),
        max_gap_s=1.1,
    )
    return fatigue, phase


def test_audit_result_schema_includes_source_timestamps_before_export():
    fatigue, phase = _results(synthetic=True)

    assert fatigue.repetition_time_s.tolist() == list(np.arange(6, dtype=float))
    assert phase.time_s.tolist() == list(np.arange(6, dtype=float))


def test_label_audit_export_is_private_and_counts_fatigue_phase_and_abstentions(tmp_path):
    module = importlib.import_module("emg_fatigue.reference_labels")
    writer = getattr(module, "write_reference_label_audit", None)
    assert callable(writer), "reference label audit export is missing"
    fatigue, phase = _results(synthetic=False)
    csv_path = Path(tmp_path) / "label_audit.csv"
    json_path = Path(tmp_path) / "label_audit.json"

    receipt = writer(
        fatigue_results=[fatigue],
        phase_results=[phase],
        csv_path=csv_path,
        json_path=json_path,
    )

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(rows) == 12
    assert summary["evidence_status"] == "reference_approved"
    assert summary["fatigue_counts"]["0"] + summary["fatigue_counts"]["1"] == 6
    assert summary["phase_counts"]["extension"] == 2
    assert summary["phase_counts"]["flexion"] == 2
    assert summary["phase_counts"]["transition"] == 2
    assert receipt["csv_sha256"]
    assert receipt["json_sha256"]


def test_synthetic_label_rows_cannot_be_exported_as_reference_evidence(tmp_path):
    module = importlib.import_module("emg_fatigue.reference_labels")
    writer = getattr(module, "write_reference_label_audit", None)
    assert callable(writer), "reference label audit export is missing"
    fatigue, phase = _results(synthetic=True)

    with pytest.raises(ValueError, match="synthetic"):
        writer(
            fatigue_results=[fatigue],
            phase_results=[phase],
            csv_path=tmp_path / "label_audit.csv",
            json_path=tmp_path / "label_audit.json",
        )
