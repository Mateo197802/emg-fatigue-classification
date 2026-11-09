"""Reference-backed labels with strict provenance and approval gates.

This module does not infer fatigue or movement phase from EMG. Candidate
thresholds may be exercised with synthetic fixtures, but those results are
explicitly non-exportable and cannot be mistaken for experimental labels.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferenceContract:
    """Auditable provenance required before a reference may supervise a model."""

    reference_kind: str
    reference_level: str
    source_id: str
    unit: str
    time_axis_id: str
    emg_time_axis_id: str
    participant_id: str
    session_id: str
    calibration_verified: bool
    calibration_id: str
    synchronized: bool
    synchronization_id: str
    synchronization_residual_s: float
    synchronization_tolerance_s: float
    label_definition_id: str
    label_definition_approved: bool
    approval_record_id: str | None = None
    synthetic_fixture: bool = False
    repeatability_id: str | None = None
    angle_convention: str | None = None


@dataclass(frozen=True)
class FatigueLabelResult:
    """Per-repetition labels; -1 means that the reference is unusable."""

    labels: np.ndarray
    repetition_time_s: np.ndarray
    onset_index: int | None
    onset_time_s: float | None
    baseline_torque_nm: float
    mdc95_nm: float
    threshold_nm: float
    baseline_indices: tuple[int, ...]
    exclusion_reasons: np.ndarray
    source_id: str
    calibration_id: str
    synchronization_id: str
    repeatability_id: str
    participant_id: str
    session_id: str
    reference_level: str
    time_axis_id: str
    emg_time_axis_id: str
    label_definition_id: str
    evidence_status: str


@dataclass(frozen=True)
class PhaseLabelResult:
    """Per-sample phase labels and their calibrated kinematic provenance."""

    labels: np.ndarray
    time_s: np.ndarray
    angular_velocity_deg_s: np.ndarray
    angle_calibration_id: str
    synchronization_id: str
    velocity_deadband_deg_s: float
    max_gap_s: float
    source_id: str
    participant_id: str
    session_id: str
    time_axis_id: str
    emg_time_axis_id: str
    label_definition_id: str
    angle_convention: str
    evidence_status: str


def _required_id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty provenance identifier")
    return value.strip()


def _validate_reference_contract(
    contract,
    *,
    reference_kind,
    reference_levels,
    expected_units,
    require_approved_definition=True,
):
    if not isinstance(contract, ReferenceContract):
        raise ValueError("reference_contract must be a ReferenceContract")
    if contract.reference_kind != reference_kind:
        raise ValueError(f"reference_kind must be {reference_kind}")
    if contract.reference_level not in reference_levels:
        expected = ", ".join(sorted(reference_levels))
        raise ValueError(f"reference_level must be one of: {expected}")
    if contract.unit.strip().lower().replace(" ", "") not in expected_units:
        raise ValueError(f"reference unit must be one of: {', '.join(sorted(expected_units))}")

    for field in (
        "source_id",
        "time_axis_id",
        "emg_time_axis_id",
        "participant_id",
        "session_id",
        "label_definition_id",
    ):
        _required_id(getattr(contract, field), field)

    if contract.calibration_verified is not True:
        raise ValueError("reference calibration must be verified")
    _required_id(contract.calibration_id, "calibration_id")
    if contract.synchronized is not True:
        raise ValueError("reference synchronization with EMG must be verified")
    _required_id(contract.synchronization_id, "synchronization_id")
    residual = contract.synchronization_residual_s
    tolerance = contract.synchronization_tolerance_s
    if not np.isfinite(residual) or residual < 0:
        raise ValueError("synchronization residual must be finite and nonnegative")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("synchronization tolerance must be finite and positive")
    if residual > tolerance:
        raise ValueError("synchronization drift exceeds the declared tolerance")

    if contract.synthetic_fixture is not True and contract.synthetic_fixture is not False:
        raise ValueError("synthetic_fixture must be boolean")
    if require_approved_definition and contract.label_definition_approved is not True:
        if not contract.synthetic_fixture:
            raise ValueError("label definition must be approved before producing experimental labels")
    if contract.label_definition_approved is True and not contract.synthetic_fixture:
        _required_id(contract.approval_record_id, "approval_record_id")
    return "synthetic_only" if contract.synthetic_fixture else "reference_approved"


def _validated_time_axis(time_s, *, length, name):
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or len(time) != length:
        raise ValueError(f"{name} must be one-dimensional and match the reference values")
    if not np.isfinite(time).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return time


def assert_reference_labels_exportable(result):
    """Reject synthetic or otherwise unapproved labels from experimental outputs."""
    if getattr(result, "evidence_status", None) != "reference_approved":
        raise ValueError("synthetic or unapproved reference labels cannot be exported")


def fatigue_labels_from_torque(
    torque_nm,
    *,
    reference_contract,
    repetition_time_s,
    sem_nm,
    valid_reps=None,
    angle_matched=None,
    baseline_reps=3,
    rolling_reps=3,
    confirmations=2,
):
    """Create approved repetition-level force-capacity labels from torque.

    The MDC95/persistence expression is a candidate operational rule. It is
    accepted for real records only when the supplied contract identifies an
    approved label definition, calibration, repeatability source and aligned
    participant/session provenance. Synthetic fixtures may exercise the rule
    in tests; their results are marked ``synthetic_only`` and cannot export.
    """
    evidence_status = _validate_reference_contract(
        reference_contract,
        reference_kind="torque",
        reference_levels={"repetition"},
        expected_units={"nm", "n*m", "n·m"},
    )
    repeatability_id = _required_id(reference_contract.repeatability_id, "repeatability_id")
    torque = np.asarray(torque_nm, dtype=float)
    if torque.ndim != 1 or len(torque) == 0:
        raise ValueError("torque_nm must be a nonempty one-dimensional sequence")
    time = _validated_time_axis(
        repetition_time_s, length=len(torque), name="repetition_time_s"
    )
    if valid_reps is None:
        raise ValueError("valid_reps mask is required")
    if angle_matched is None:
        raise ValueError("angle_matched mask is required")
    valid = np.asarray(valid_reps)
    matched = np.asarray(angle_matched)
    if valid.dtype != np.bool_ or matched.dtype != np.bool_:
        raise ValueError("valid_reps and angle_matched must be boolean masks")
    if valid.shape != torque.shape or matched.shape != torque.shape:
        raise ValueError("validity and angle masks must match torque_nm")
    eligible = valid & matched
    if np.any(eligible & ~np.isfinite(torque)):
        raise ValueError("eligible torque values must be finite")
    if np.any(eligible & (torque < 0)):
        raise ValueError("eligible extension torque must be nonnegative")
    if not np.isfinite(sem_nm) or sem_nm <= 0:
        raise ValueError("SEM must be finite and greater than zero")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
        for value in (baseline_reps, rolling_reps, confirmations)
    ):
        raise ValueError("rep counts must be positive integers")

    baseline_indices = tuple(np.flatnonzero(eligible)[:baseline_reps].tolist())
    if len(baseline_indices) < baseline_reps:
        raise ValueError("not enough valid angle-matched baseline repetitions")
    first_end = baseline_indices[-1] + rolling_reps
    if len(torque) < first_end + confirmations - 1:
        raise ValueError("not enough repetitions after baseline for rolling confirmation")

    baseline = float(np.median(torque[list(baseline_indices)]))
    mdc95 = float(1.96 * np.sqrt(2.0) * sem_nm)
    threshold = baseline - mdc95
    below_count = 0
    onset = None
    for end in range(first_end, len(torque)):
        start = end - rolling_reps + 1
        indices = np.arange(start, end + 1)
        if not eligible[indices].all():
            below_count = 0
            continue
        rolling_median = float(np.median(torque[start : end + 1]))
        if rolling_median < threshold:
            below_count += 1
            if below_count >= confirmations:
                onset = end - confirmations + 1
                break
        else:
            below_count = 0

    labels = np.full(len(torque), -1, dtype=np.int8)
    labels[eligible] = 0
    if onset is not None:
        labels[eligible & (np.arange(len(torque)) >= onset)] = 1
    exclusions = np.full(len(torque), "", dtype=object)
    exclusions[~valid] = "invalid_repetition"
    exclusions[valid & ~matched] = "outside_matched_angle_range"
    return FatigueLabelResult(
        labels=labels,
        repetition_time_s=time.copy(),
        onset_index=onset,
        onset_time_s=None if onset is None else float(time[onset]),
        baseline_torque_nm=baseline,
        mdc95_nm=mdc95,
        threshold_nm=threshold,
        baseline_indices=baseline_indices,
        exclusion_reasons=exclusions,
        source_id=reference_contract.source_id,
        calibration_id=reference_contract.calibration_id,
        synchronization_id=reference_contract.synchronization_id,
        repeatability_id=repeatability_id,
        participant_id=reference_contract.participant_id,
        session_id=reference_contract.session_id,
        reference_level=reference_contract.reference_level,
        time_axis_id=reference_contract.time_axis_id,
        emg_time_axis_id=reference_contract.emg_time_axis_id,
        label_definition_id=reference_contract.label_definition_id,
        evidence_status=evidence_status,
    )


def phase_from_flexion_angle(
    time_s,
    flexion_angle_deg,
    *,
    reference_contract,
    velocity_deadband_deg_s,
    angle_convention=None,
    max_gap_s,
):
    """Derive phases from synchronized knee angle and an explicit sign rule.

    ``flexion_positive`` means increasing angle is knee flexion. Under
    ``extension_positive`` the input angular velocity is sign-reversed before
    assigning phase. Velocities inside the sensor-noise deadband are
    transitions. The signal must already be calibrated and aligned to EMG.
    """
    evidence_status = _validate_reference_contract(
        reference_contract,
        reference_kind="knee_angle",
        reference_levels={"time_series"},
        expected_units={"deg", "degree", "degrees"},
    )
    convention = angle_convention or reference_contract.angle_convention
    if convention not in {"flexion_positive", "extension_positive"}:
        raise ValueError("angle_convention must be flexion_positive or extension_positive")
    if reference_contract.angle_convention not in {None, convention}:
        raise ValueError("angle_convention conflicts with the reference contract")

    time = np.asarray(time_s, dtype=float)
    angle = np.asarray(flexion_angle_deg, dtype=float)
    if time.ndim != 1 or angle.ndim != 1 or len(time) != len(angle) or len(time) < 2:
        raise ValueError("time and angle must be matching one-dimensional sequences")
    if not np.isfinite(time).all() or not np.isfinite(angle).all():
        raise ValueError("time and angle must contain only finite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing")
    gaps = np.diff(time)
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be provided, finite and greater than zero")
    if np.any(gaps > max_gap_s):
        raise ValueError("time gap exceeds max_gap_s; phase derivative is not valid")
    if not np.isfinite(velocity_deadband_deg_s) or velocity_deadband_deg_s <= 0:
        raise ValueError("velocity deadband must be finite and greater than zero")

    velocity = np.gradient(angle, time)
    if convention == "extension_positive":
        velocity = -velocity
    labels = np.full(len(time), "transition", dtype=object)
    labels[velocity > velocity_deadband_deg_s] = "flexion"
    labels[velocity < -velocity_deadband_deg_s] = "extension"
    return PhaseLabelResult(
        labels=labels,
        time_s=time.copy(),
        angular_velocity_deg_s=velocity,
        angle_calibration_id=reference_contract.calibration_id,
        synchronization_id=reference_contract.synchronization_id,
        velocity_deadband_deg_s=float(velocity_deadband_deg_s),
        max_gap_s=float(max_gap_s),
        source_id=reference_contract.source_id,
        participant_id=reference_contract.participant_id,
        session_id=reference_contract.session_id,
        time_axis_id=reference_contract.time_axis_id,
        emg_time_axis_id=reference_contract.emg_time_axis_id,
        label_definition_id=reference_contract.label_definition_id,
        angle_convention=convention,
        evidence_status=evidence_status,
    )


def write_reference_label_audit(*, fatigue_results=(), phase_results=(), csv_path, json_path):
    """Write a private row-level audit and aggregate counts for approved labels."""
    fatigue_results = tuple(fatigue_results)
    phase_results = tuple(phase_results)
    if not fatigue_results and not phase_results:
        raise ValueError("at least one approved reference-label result is required")
    for result in (*fatigue_results, *phase_results):
        assert_reference_labels_exportable(result)

    csv_path = Path(csv_path).expanduser().resolve()
    json_path = Path(json_path).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if any(path == repo or repo in path.parents for path in (csv_path, json_path)):
        raise ValueError("row-level reference label audits must remain outside the repository")
    if csv_path == json_path:
        raise ValueError("CSV and JSON audit paths must be different")

    fields = [
        "record_type", "reference_source_id", "participant_id", "session_id",
        "reference_level", "time_axis_id", "emg_time_axis_id", "time_s",
        "fatigue_state", "phase_state", "uncertainty_value", "uncertainty_unit",
        "exclusion_reason", "label_definition_id", "evidence_status",
    ]
    rows = []
    fatigue_counts = Counter({"0": 0, "1": 0, "abstain": 0})
    phase_counts = Counter({"extension": 0, "flexion": 0, "transition": 0, "abstain": 0})
    counts_by_source = defaultdict(Counter)
    counts_by_participant = defaultdict(Counter)
    fatigue_uncertainty = []
    phase_uncertainty = []

    for result in fatigue_results:
        for index, label in enumerate(result.labels):
            state = "" if int(label) < 0 else str(int(label))
            reason = str(result.exclusion_reasons[index])
            if state:
                fatigue_counts[state] += 1
                counts_by_source[result.source_id][f"fatigue_{state}"] += 1
                counts_by_participant[result.participant_id][f"fatigue_{state}"] += 1
            else:
                fatigue_counts["abstain"] += 1
                counts_by_source[result.source_id]["fatigue_abstain"] += 1
                counts_by_participant[result.participant_id]["fatigue_abstain"] += 1
            fatigue_uncertainty.append(float(result.mdc95_nm))
            rows.append({
                "record_type": "fatigue_repetition",
                "reference_source_id": result.source_id,
                "participant_id": result.participant_id,
                "session_id": result.session_id,
                "reference_level": result.reference_level,
                "time_axis_id": result.time_axis_id,
                "emg_time_axis_id": result.emg_time_axis_id,
                "time_s": float(result.repetition_time_s[index]),
                "fatigue_state": state,
                "phase_state": "",
                "uncertainty_value": float(result.mdc95_nm),
                "uncertainty_unit": "N*m_MDC95",
                "exclusion_reason": reason,
                "label_definition_id": result.label_definition_id,
                "evidence_status": result.evidence_status,
            })

    for result in phase_results:
        for index, label in enumerate(result.labels):
            state = str(label)
            if state not in {"extension", "flexion", "transition"}:
                state = "abstain"
            phase_counts[state] += 1
            counts_by_source[result.source_id][f"phase_{state}"] += 1
            counts_by_participant[result.participant_id][f"phase_{state}"] += 1
            phase_uncertainty.append(float(result.velocity_deadband_deg_s))
            rows.append({
                "record_type": "movement_phase_sample",
                "reference_source_id": result.source_id,
                "participant_id": result.participant_id,
                "session_id": result.session_id,
                "reference_level": "time_series",
                "time_axis_id": result.time_axis_id,
                "emg_time_axis_id": result.emg_time_axis_id,
                "time_s": float(result.time_s[index]),
                "fatigue_state": "",
                "phase_state": state,
                "uncertainty_value": float(result.velocity_deadband_deg_s),
                "uncertainty_unit": "deg_per_s_deadband",
                "exclusion_reason": "transition_deadband" if state == "transition" else "",
                "label_definition_id": result.label_definition_id,
                "evidence_status": result.evidence_status,
            })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "audit_version": "reference_label_audit_v1",
        "evidence_status": "reference_approved",
        "record_count": len(rows),
        "fatigue_counts": dict(fatigue_counts),
        "phase_counts": dict(phase_counts),
        "counts_by_reference_source": {
            key: dict(value) for key, value in sorted(counts_by_source.items())
        },
        "counts_by_participant": {
            key: dict(value) for key, value in sorted(counts_by_participant.items())
        },
        "uncertainty_summary": {
            "fatigue_mdc95_nm_values": sorted(set(fatigue_uncertainty)),
            "phase_deadband_deg_s_values": sorted(set(phase_uncertainty)),
        },
        "abstention_reasons": dict(Counter(
            row["exclusion_reason"] for row in rows if row["exclusion_reason"]
        )),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "csv_sha256": file_hash(csv_path),
        "json_sha256": file_hash(json_path),
        "record_count": len(rows),
        "evidence_status": "reference_approved",
    }
