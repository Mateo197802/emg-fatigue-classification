import pandas as pd
import importlib


def _summarize_preanalysis():
    try:
        module = importlib.import_module("emg_fatigue.preanalysis")
    except ModuleNotFoundError:
        return None
    return getattr(module, "summarize_preanalysis", None)


def test_preanalysis_reports_qc_counts_without_substituting_time_trajectories():
    qc = pd.DataFrame(
        [
            {
                "status": "valid",
                "constant_signal": False,
                "clipping_status": "no_extreme_repeat_flag",
                "p99_count": 80.0,
                "p01_count": -20.0,
                "rms_count": 50.0,
                "linear_trend_count_per_s": 2.0,
            },
            {
                "status": "valid",
                "constant_signal": False,
                "clipping_status": "uncalibrated_extreme_repeat_flag",
                "p99_count": 140.0,
                "p01_count": -60.0,
                "rms_count": 70.0,
                "linear_trend_count_per_s": -4.0,
            },
            {
                "status": "constant_signal",
                "constant_signal": True,
                "clipping_status": "no_extreme_repeat_flag",
                "p99_count": 0.0,
                "p01_count": 0.0,
                "rms_count": 0.0,
                "linear_trend_count_per_s": 0.0,
            },
            {
                "status": "jitter_exceeds_limit",
                "constant_signal": False,
                "clipping_status": "no_extreme_repeat_flag",
                "p99_count": 50.0,
                "p01_count": -50.0,
                "rms_count": 30.0,
                "linear_trend_count_per_s": 1.0,
            },
        ]
    )
    line = pd.DataFrame(
        {"50": [12.0, 7.0, 20.0, float("nan")], "60": [1.0, 11.0, 14.0, 0.0]}
    )

    summarize = _summarize_preanalysis()
    assert callable(summarize), "preanalysis summary behavior is not implemented"
    result = summarize(qc, line)

    assert result["run_count"] == 4
    assert result["valid_run_count"] == 2
    assert result["excluded_run_count"] == 2
    assert result["constant_run_count"] == 1
    assert result["uncalibrated_extreme_repeat_flag_count"] == 1
    assert result["median_p99_minus_p01_count"] == 150.0
    assert result["median_rms_count"] == 60.0
    assert result["median_absolute_linear_trend_count_per_s"] == 3.0
    assert result["line_prominence_threshold_db"] == 10.0
    assert result["line_50hz_above_threshold_count"] == 2
    assert result["line_50hz_measured_count"] == 3
    assert result["line_60hz_above_threshold_count"] == 2
    assert result["line_60hz_measured_count"] == 4
    assert result["repetition_phase_trajectories_available"] is False
    assert "trajectories" not in result


def test_preanalysis_summary_rejects_missing_required_qc_fields():
    qc = pd.DataFrame([{"status": "valid"}])

    try:
        summarize = _summarize_preanalysis()
        assert callable(summarize), "preanalysis summary behavior is not implemented"
        summarize(qc, pd.DataFrame())
    except ValueError as exc:
        assert "missing QC fields" in str(exc)
    else:
        raise AssertionError("missing signal QC fields must not be silently replaced")
