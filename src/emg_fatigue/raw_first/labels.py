"""Reference audit: raw numeric signals are not fatigue or phase labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LabelAudit:
    status: str
    reference_search_complete: bool
    unreviewed_non_signal_csv_count: int
    fatigue_reference_found: bool | None
    movement_phase_reference_found: bool | None
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
    csv_count = int(inventory.get("csv_count", 0))
    required_columns = {"Relative_Time_s", "ADC_Value", "Voltage_mV"}
    header_counts = headers if isinstance(headers, dict) else {}
    signal_csv_count = sum(
        int(count)
        for header, count in header_counts.items()
        if {column.strip() for column in str(header).split(",")} == required_columns
    )
    unreviewed_non_signal_csv_count = max(0, csv_count - signal_csv_count)
    search_complete = csv_count > 0 and sidecar_count == 0 and unreviewed_non_signal_csv_count == 0
    if search_complete:
        status = "blocked_no_independent_reference"
        rationale = (
            "All archive members were CSV signals with only Relative_Time_s, ADC_Value, and Voltage_mV columns. "
            "No independent synchronized fatigue or movement-phase reference was present; filenames and temporal position are not labels."
        )
        fatigue_found: bool | None = False
        phase_found: bool | None = False
    else:
        status = "blocked_reference_search_incomplete"
        rationale = (
            f"Reference search is incomplete: {sidecar_count} non-CSV sidecar(s) and "
            f"{unreviewed_non_signal_csv_count} non-signal CSV(s) were not validated for synchronized reference content. "
            "No independent label absence is claimed, and supervised outputs remain blocked."
        )
        fatigue_found = None
        phase_found = None
    return LabelAudit(
        status=status,
        reference_search_complete=search_complete,
        unreviewed_non_signal_csv_count=unreviewed_non_signal_csv_count,
        fatigue_reference_found=fatigue_found,
        movement_phase_reference_found=phase_found,
        files_examined=csv_count,
        sidecar_count=sidecar_count,
        available_columns=tuple(sorted(columns)),
        missing_fatigue_references=required_fatigue,
        missing_phase_references=required_phase,
        rationale=rationale,
    )
