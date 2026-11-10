"""Descriptive signal-QC summaries that do not infer fatigue or movement phase."""

import numpy as np
import pandas as pd


_QC_FIELDS = {
    "status",
    "constant_signal",
    "clipping_status",
    "p99_count",
    "p01_count",
    "rms_count",
    "linear_trend_count_per_s",
}


def _truth(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def summarize_preanalysis(qc: pd.DataFrame, line_prominence: pd.DataFrame, threshold_db=10.0):
    """Summarize run-level QC and mains-line prominence without time trajectories.

    ADC-count summaries are descriptive only. Line-prominence counts report
    measurements exceeding the preprocessing trigger; they do not prove a
    source of contamination or calibrated clipping.
    """
    missing = _QC_FIELDS - set(qc.columns)
    if missing:
        raise ValueError(f"missing QC fields: {sorted(missing)}")
    if not np.isfinite(float(threshold_db)) or float(threshold_db) < 0:
        raise ValueError("line-prominence threshold must be finite and nonnegative")

    status = qc["status"].astype(str)
    valid = qc.loc[status.eq("valid")]
    span = pd.to_numeric(valid["p99_count"], errors="coerce") - pd.to_numeric(
        valid["p01_count"], errors="coerce"
    )
    rms = pd.to_numeric(valid["rms_count"], errors="coerce")
    trend = pd.to_numeric(valid["linear_trend_count_per_s"], errors="coerce").abs()

    result = {
        "run_count": int(len(qc)),
        "valid_run_count": int(len(valid)),
        "excluded_run_count": int((~status.eq("valid")).sum()),
        "constant_run_count": int(qc["constant_signal"].map(_truth).sum()),
        "uncalibrated_extreme_repeat_flag_count": int(
            qc["clipping_status"].astype(str).eq("uncalibrated_extreme_repeat_flag").sum()
        ),
        "median_p99_minus_p01_count": float(span.median()) if span.notna().any() else None,
        "median_rms_count": float(rms.median()) if rms.notna().any() else None,
        "median_absolute_linear_trend_count_per_s": float(trend.median()) if trend.notna().any() else None,
        "line_prominence_threshold_db": float(threshold_db),
        "repetition_phase_trajectories_available": False,
        "repetition_phase_trajectories_note": (
            "Not computed: the supplied recordings have no verified repetition or synchronized phase markers."
        ),
    }
    for frequency in (50, 60):
        column = str(frequency)
        if column not in line_prominence.columns:
            values = pd.Series(dtype=float)
        else:
            values = pd.to_numeric(line_prominence[column], errors="coerce").dropna()
        result[f"line_{frequency}hz_above_threshold_count"] = int((values > threshold_db).sum())
        result[f"line_{frequency}hz_measured_count"] = int(values.count())
    return result
