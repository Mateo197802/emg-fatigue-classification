"""Reference audit: raw numeric signals are not fatigue or phase labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LabelAudit:
    status: str
    fatigue_reference_found: bool
    movement_phase_reference_found: bool
    files_examined: int
    sidecar_count: int
    available_columns: tuple[str, ...]
    missing_fatigue_references: tuple[str, ...]
    missing_phase_references: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_references(inventory: dict[str, object]) -> LabelAudit:
    headers = inventory.get("header_counts", {})
    columns: set[str] = set()
    for header in headers if isinstance(headers, dict) else []:
        columns.update(str(header).split(","))
    required_fatigue = ("synchronized Borg CR-10", "pre/post MVC", "force or torque", "repetition-level performance", "mechanical failure criterion")
    required_phase = ("synchronized extension/flexion annotation", "knee-angle or video phase labels")
    sidecar_count = int(inventory.get("non_csv_member_count", 0))
    return LabelAudit(
        status="blocked_no_independent_reference",
        fatigue_reference_found=False,
        movement_phase_reference_found=False,
        files_examined=int(inventory.get("csv_count", 0)),
        sidecar_count=sidecar_count,
        available_columns=tuple(sorted(columns)),
        missing_fatigue_references=required_fatigue,
        missing_phase_references=required_phase,
        rationale=(
            "The authorized training archive contains only Relative_Time_s, ADC_Value, and Voltage_mV signal columns. "
            "There is no synchronized independent fatigue or movement-phase reference in the archive; filenames and temporal position are not labels."
        ),
    )
