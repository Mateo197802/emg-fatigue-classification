"""Deterministic technical-reference labels for an EMG fatigue-like proxy."""

from __future__ import annotations

import math
from collections import deque
from typing import Mapping, Sequence

import numpy as np

from .contracts import ReferenceEstimate, SignalWindow, WeakLabel
from .features import window_key


_INDICATORS = {
    "mdf": "median_frequency_hz",
    "mnf": "mean_frequency_hz",
    "rms": "rms",
    "log_ratio": "line_excluded_low_high_log_ratio",
    "low_fraction": "line_excluded_low_frequency_fraction_20_80",
}


def _valid_feature(row: Mapping[str, object] | None) -> bool:
    if row is None or int(row.get("feature_valid", 0)) != 1:
        return False
    return all(
        name in row and np.isfinite(float(row[name]))
        for name in (_INDICATORS["mdf"], _INDICATORS["mnf"], _INDICATORS["rms"], _INDICATORS["log_ratio"], _INDICATORS["low_fraction"])
    )


def _scaled_mad(values: Sequence[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    median = float(np.median(data))
    return float(1.4826 * np.median(np.abs(data - median)))


def technical_reference(
    windows: Sequence[SignalWindow],
    feature_by_key: Mapping[str, Mapping[str, object]],
    config: dict,
    *,
    count: int | None = None,
    stability_percent: float | None = None,
    stability_log_ratio_limit: float = 0.20,
    stability_components: tuple[str, ...] = ("mdf", "mnf", "log_ratio"),
) -> ReferenceEstimate:
    """Choose the earliest valid non-overlapping activity windows as a technical comparator."""
    settings = config["labels"]
    requested = int(count if count is not None else settings["reference_window_count"])
    if requested < 2:
        raise ValueError("At least two windows are required for a robust technical reference.")
    stability_limit = float(
        stability_percent if stability_percent is not None else settings["reference_instability_percent"]
    ) / 100.0
    ordered = sorted(windows, key=lambda item: (item.start_time_s, item.start_sample))
    selected: list[SignalWindow] = []
    last_stop = -1
    for window in ordered:
        if not window.eligible_for_label:
            continue
        feature = feature_by_key.get(window_key(window))
        if not _valid_feature(feature):
            continue
        if window.start_sample < last_stop:
            continue
        selected.append(window)
        last_stop = window.start_sample + int(round(window.duration_s * window.sampling_rate_hz))
        if len(selected) == requested:
            break

    if len(selected) < requested:
        return ReferenceEstimate(
            group_id=ordered[0].group_id if ordered else "",
            recording_id=ordered[0].recording_id if ordered else "",
            segment_index=ordered[0].segment_index if ordered else 0,
            available=False,
            stable=False,
            window_count=len(selected),
            reference_window_keys=tuple(window_key(item) for item in selected),
            center={},
            scaled_mad={},
            stability_ratios={},
            status="insufficient_reference_windows",
            reasons=(f"need_{requested}_valid_nonoverlapping_windows_found_{len(selected)}",),
        )

    values = {
        name: np.asarray([float(feature_by_key[window_key(item)][column]) for item in selected], dtype=np.float64)
        for name, column in _INDICATORS.items()
    }
    centers = {name: float(np.median(array)) for name, array in values.items()}
    dispersions = {name: _scaled_mad(array) for name, array in values.items()}
    stability_ratios = {
        "mdf": dispersions["mdf"] / abs(centers["mdf"]) if abs(centers["mdf"]) > 0 else float("inf"),
        "mnf": dispersions["mnf"] / abs(centers["mnf"]) if abs(centers["mnf"]) > 0 else float("inf"),
        "log_ratio": dispersions["log_ratio"],
        "low_fraction": dispersions["low_fraction"],
    }
    reasons: list[str] = []
    for indicator in stability_components:
        indicator_limit = stability_limit if indicator in ("mdf", "mnf") else float(stability_log_ratio_limit)
        if not np.isfinite(stability_ratios[indicator]) or stability_ratios[indicator] > indicator_limit:
            reasons.append(f"unstable_{indicator}_reference")
    first = selected[0]
    return ReferenceEstimate(
        group_id=first.group_id,
        recording_id=first.recording_id,
        segment_index=first.segment_index,
        available=True,
        stable=not reasons,
        window_count=len(selected),
        reference_window_keys=tuple(window_key(item) for item in selected),
        center=centers,
        scaled_mad=dispersions,
        stability_ratios=stability_ratios,
        status="stable" if not reasons else "unstable_reference",
        reasons=tuple(reasons),
    )


def _ambiguity_label(
    window: SignalWindow,
    reason: str,
    reference: ReferenceEstimate,
    *,
    state_transition: str = "abstain",
    observed: dict[str, float] | None = None,
    deltas: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    persistence_candidates: int = 0,
    persistence_span_s: float = 0.0,
    state_latched: bool = False,
    rule_version: str = "spectral_persistence_v1",
) -> WeakLabel:
    return WeakLabel(
        window_key=window_key(window),
        group_id=window.group_id,
        recording_id=window.recording_id,
        segment_index=window.segment_index,
        start_time_s=float(window.start_time_s),
        duration_s=float(window.duration_s),
        class_id=-1,
        class_name="AMBIGUOUS",
        rule_version=rule_version,
        label_confidence=0.0,
        label_reason=reason,
        state_transition=state_transition,
        reference_window_keys=reference.reference_window_keys,
        reference_center=dict(reference.center),
        reference_scaled_mad=dict(reference.scaled_mad),
        observed_indicators=observed or {},
        indicator_deltas=deltas or {},
        indicator_thresholds=thresholds or {},
        persistence_candidates=persistence_candidates,
        persistence_span_s=float(persistence_span_s),
        state_latched=state_latched,
    )


def _assigned_label(
    window: SignalWindow,
    reference: ReferenceEstimate,
    class_id: int,
    confidence: float,
    reason: str,
    transition: str,
    observed: dict[str, float],
    deltas: dict[str, float],
    thresholds: dict[str, float],
    persistence_candidates: int,
    persistence_span_s: float,
    state_latched: bool,
    rule_version: str,
) -> WeakLabel:
    name = "NON_FATIGUE_LIKE" if class_id == 0 else "FATIGUE_LIKE"
    return WeakLabel(
        window_key=window_key(window),
        group_id=window.group_id,
        recording_id=window.recording_id,
        segment_index=window.segment_index,
        start_time_s=float(window.start_time_s),
        duration_s=float(window.duration_s),
        class_id=int(class_id),
        class_name=name,
        rule_version=rule_version,
        label_confidence=float(np.clip(confidence, 0.0, 1.0)),
        label_reason=reason,
        state_transition=transition,
        reference_window_keys=reference.reference_window_keys,
        reference_center=dict(reference.center),
        reference_scaled_mad=dict(reference.scaled_mad),
        observed_indicators=observed,
        indicator_deltas=deltas,
        indicator_thresholds=thresholds,
        persistence_candidates=persistence_candidates,
        persistence_span_s=float(persistence_span_s),
        state_latched=state_latched,
    )


def _entry_geometry(duration_s: float, settings: dict) -> tuple[int, int, float]:
    base_window_count = int(settings["entry_rolling_window_count"])
    base_candidate_count = int(settings["entry_window_count"])
    overlap = 0.5
    hop_s = duration_s * (1.0 - overlap)
    min_span = float(settings.get("minimum_entry_span_s", 1.25))
    windows_for_span = int(max(1, math.ceil(max(0.0, min_span - duration_s) / hop_s) + 1))
    rolling_count = max(base_window_count, windows_for_span)
    required_candidates = max(base_candidate_count, int(math.ceil(base_candidate_count * rolling_count / base_window_count)))
    return rolling_count, required_candidates, min_span


def label_recording(
    windows: Sequence[SignalWindow],
    feature_by_key: Mapping[str, Mapping[str, object]],
    reference: ReferenceEstimate,
    config: dict,
    *,
    relative_change: float = 0.0,
    scaled_mad_multiple: float | None = None,
    rule_version: str | None = None,
) -> list[WeakLabel]:
    """Assign 0/1/-1 labels by persistent spectral changes from a technical reference."""
    settings = config["labels"]
    version = rule_version or str(settings["version"])
    ordered = sorted(windows, key=lambda item: (item.start_time_s, item.start_sample))
    if not reference.available or not reference.stable:
        reason = ";".join((reference.status, *reference.reasons))
        return [
            _ambiguity_label(window, reason, reference, state_transition="reference_blocked", rule_version=version)
            for window in ordered
        ]
    if not ordered:
        return []

    mad_multiple = float(scaled_mad_multiple if scaled_mad_multiple is not None else settings["minimum_scaled_mad_multiple"])
    frequency_floor = float(settings["minimum_mdf_mnf_drop_hz"])
    reference_duration = float(config["run"]["primary_window_s"])
    ratio_floor = float(settings["minimum_log_ratio_increase_when_mad_zero"])
    rolling_count, required_candidates, minimum_span_s = _entry_geometry(ordered[0].duration_s, settings)
    history: deque[tuple[SignalWindow, bool]] = deque(maxlen=rolling_count)
    labels: list[WeakLabel] = []
    latched = False
    recovery_count = 0
    reference_keys = set(reference.reference_window_keys)
    reference_start_samples = [window.start_sample for window in ordered if window_key(window) in reference_keys]
    reference_end_sample = max(
        (
            window.start_sample + int(round(window.duration_s * window.sampling_rate_hz))
            for window in ordered
            if window_key(window) in reference_keys
        ),
        default=-1,
    )
    reference_start_sample = min(reference_start_samples, default=-1)

    for window in ordered:
        key = window_key(window)
        if key in reference_keys or (reference_start_sample >= 0 and window.start_sample < reference_end_sample):
            history.clear()
            recovery_count = 0
            reason = "technical_reference_window" if key in reference_keys else "technical_reference_period"
            labels.append(_ambiguity_label(window, reason, reference, state_transition="reference_only", rule_version=version))
            continue
        if not window.eligible_for_label:
            history.clear()
            recovery_count = 0
            labels.append(
                _ambiguity_label(
                    window,
                    window.rejection_reason or "window_ineligible",
                    reference,
                    state_transition="qc_ambiguous",
                    state_latched=latched,
                    rule_version=version,
                )
            )
            continue
        row = feature_by_key.get(key)
        if not _valid_feature(row):
            history.clear()
            recovery_count = 0
            labels.append(
                _ambiguity_label(
                    window,
                    "indicator_invalid",
                    reference,
                    state_transition="feature_ambiguous",
                    state_latched=latched,
                    rule_version=version,
                )
            )
            continue

        observed = {name: float(row[column]) for name, column in _INDICATORS.items()}
        deltas = {
            "mdf_drop_hz": float(reference.center["mdf"] - observed["mdf"]),
            "mnf_drop_hz": float(reference.center["mnf"] - observed["mnf"]),
            "mdf_relative_drop": float((reference.center["mdf"] - observed["mdf"]) / max(abs(reference.center["mdf"]), np.finfo(float).eps)),
            "mnf_relative_drop": float((reference.center["mnf"] - observed["mnf"]) / max(abs(reference.center["mnf"]), np.finfo(float).eps)),
            "log_ratio_increase": float(observed["log_ratio"] - reference.center["log_ratio"]),
            "rms_relative_change": float((observed["rms"] - reference.center["rms"]) / max(abs(reference.center["rms"]), np.finfo(float).eps)),
            "low_fraction_change": float(observed["low_fraction"] - reference.center["low_fraction"]),
        }
        frequency_thresholds = {
            metric: max(
                frequency_floor * reference_duration / window.duration_s,
                mad_multiple * reference.scaled_mad[metric],
                relative_change * abs(reference.center[metric]),
            )
            for metric in ("mdf", "mnf")
        }
        ratio_threshold = (
            mad_multiple * reference.scaled_mad["log_ratio"]
            if reference.scaled_mad["log_ratio"] > 0
            else ratio_floor
        )
        thresholds = {
            "mdf_drop_hz": float(frequency_thresholds["mdf"]),
            "mnf_drop_hz": float(frequency_thresholds["mnf"]),
            "log_ratio_increase": float(ratio_threshold),
            "recovery_mdf_abs_delta_hz": float(frequency_thresholds["mdf"] / 2.0),
            "recovery_mnf_abs_delta_hz": float(frequency_thresholds["mnf"] / 2.0),
            "recovery_log_ratio_abs_delta": float(ratio_threshold / 2.0),
        }
        candidate = (
            deltas["mdf_drop_hz"] > frequency_thresholds["mdf"]
            and deltas["mnf_drop_hz"] > frequency_thresholds["mnf"]
            and deltas["log_ratio_increase"] > ratio_threshold
        )
        stable_nonfatigue = (
            abs(deltas["mdf_drop_hz"]) <= frequency_thresholds["mdf"]
            and abs(deltas["mnf_drop_hz"]) <= frequency_thresholds["mnf"]
            and abs(deltas["log_ratio_increase"]) <= ratio_threshold
        )
        recovery = (
            abs(deltas["mdf_drop_hz"]) <= thresholds["recovery_mdf_abs_delta_hz"]
            and abs(deltas["mnf_drop_hz"]) <= thresholds["recovery_mnf_abs_delta_hz"]
            and abs(deltas["log_ratio_increase"]) <= thresholds["recovery_log_ratio_abs_delta"]
        )
        history.append((window, bool(candidate)))
        history_candidates = sum(item[1] for item in history)
        span_s = (
            history[-1][0].start_time_s + history[-1][0].duration_s - history[0][0].start_time_s
            if history
            else 0.0
        )

        if latched:
            if candidate:
                recovery_count = 0
                positive_margins = [
                    deltas["mdf_drop_hz"] / max(frequency_thresholds["mdf"], np.finfo(float).eps),
                    deltas["mnf_drop_hz"] / max(frequency_thresholds["mnf"], np.finfo(float).eps),
                    deltas["log_ratio_increase"] / max(ratio_threshold, np.finfo(float).eps),
                ]
                confidence = min(1.0, max(0.0, min(positive_margins) - 1.0))
                labels.append(
                    _assigned_label(window, reference, 1, confidence, "fatigue_like_latched", "latched_candidate", observed, deltas, thresholds, history_candidates, span_s, True, version)
                )
            elif recovery:
                recovery_count += 1
                if recovery_count >= int(settings["recovery_consecutive_window_count"]):
                    latched = False
                    recovery_count = 0
                    history.clear()
                    confidence = min(
                        1.0,
                        min(
                            thresholds["recovery_mdf_abs_delta_hz"] / max(abs(deltas["mdf_drop_hz"]), np.finfo(float).eps),
                            thresholds["recovery_mnf_abs_delta_hz"] / max(abs(deltas["mnf_drop_hz"]), np.finfo(float).eps),
                            thresholds["recovery_log_ratio_abs_delta"] / max(abs(deltas["log_ratio_increase"]), np.finfo(float).eps),
                        ) - 1.0,
                    )
                    labels.append(
                        _assigned_label(window, reference, 0, confidence, "recovery_confirmed", "fatigue_to_non_fatigue_like", observed, deltas, thresholds, history_candidates, span_s, False, version)
                    )
                else:
                    labels.append(
                        _ambiguity_label(window, f"recovery_pending_{recovery_count}", reference, state_transition="recovery_pending", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=history_candidates, persistence_span_s=span_s, state_latched=True, rule_version=version)
                    )
            else:
                recovery_count = 0
                labels.append(
                    _ambiguity_label(window, "discordant_while_fatigue_latched", reference, state_transition="latched_discordant", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=history_candidates, persistence_span_s=span_s, state_latched=True, rule_version=version)
                )
            continue

        if candidate and history_candidates >= required_candidates and span_s >= minimum_span_s:
            latched = True
            recovery_count = 0
            positive_margins = [
                deltas["mdf_drop_hz"] / max(frequency_thresholds["mdf"], np.finfo(float).eps),
                deltas["mnf_drop_hz"] / max(frequency_thresholds["mnf"], np.finfo(float).eps),
                deltas["log_ratio_increase"] / max(ratio_threshold, np.finfo(float).eps),
            ]
            confidence = min(1.0, max(0.0, min(positive_margins) - 1.0))
            labels.append(
                _assigned_label(window, reference, 1, confidence, "persistent_multi_indicator_entry", "enter_fatigue_like", observed, deltas, thresholds, history_candidates, span_s, True, version)
            )
        elif candidate:
            labels.append(
                _ambiguity_label(window, "candidate_persistence_not_met", reference, state_transition="entry_pending", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=history_candidates, persistence_span_s=span_s, state_latched=False, rule_version=version)
            )
        elif stable_nonfatigue:
            margins = [
                1.0 - abs(deltas["mdf_drop_hz"]) / max(frequency_thresholds["mdf"], np.finfo(float).eps),
                1.0 - abs(deltas["mnf_drop_hz"]) / max(frequency_thresholds["mnf"], np.finfo(float).eps),
                1.0 - abs(deltas["log_ratio_increase"]) / max(ratio_threshold, np.finfo(float).eps),
            ]
            labels.append(
                _assigned_label(window, reference, 0, max(0.0, min(margins)), "within_reference_envelope", "stable_non_fatigue_like", observed, deltas, thresholds, history_candidates, span_s, False, version)
            )
        else:
            labels.append(
                _ambiguity_label(window, "discordant_or_partial_indicators", reference, state_transition="discordant_evidence", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=history_candidates, persistence_span_s=span_s, state_latched=False, rule_version=version)
            )
    return labels


def label_low_frequency_fraction_rule(
    windows: Sequence[SignalWindow],
    feature_by_key: Mapping[str, Mapping[str, object]],
    reference: ReferenceEstimate,
    config: dict,
    *,
    scaled_mad_multiple: float | None = None,
) -> list[WeakLabel]:
    """Alternative proxy using only the line-excluded 20-80 Hz fraction change."""
    settings = config["labels"]
    version = "low_frequency_fraction_persistence_v1"
    ordered = sorted(windows, key=lambda item: (item.start_time_s, item.start_sample))
    if not reference.available or not reference.stable:
        reason = ";".join((reference.status, *reference.reasons))
        return [
            _ambiguity_label(window, reason, reference, state_transition="reference_blocked", rule_version=version)
            for window in ordered
        ]
    mad_multiple = float(scaled_mad_multiple if scaled_mad_multiple is not None else settings["minimum_scaled_mad_multiple"])
    mad = float(reference.scaled_mad["low_fraction"])
    threshold = mad_multiple * mad if mad > 0 else float(settings["minimum_low_frequency_fraction_increase_when_mad_zero"])
    rolling_count, required_candidates, minimum_span = _entry_geometry(ordered[0].duration_s, settings) if ordered else (4, 3, 1.25)
    history: deque[tuple[SignalWindow, bool]] = deque(maxlen=rolling_count)
    reference_keys = set(reference.reference_window_keys)
    ref_end = max(
        (window.start_sample + int(round(window.duration_s * window.sampling_rate_hz)) for window in ordered if window_key(window) in reference_keys),
        default=-1,
    )
    ref_start = min((window.start_sample for window in ordered if window_key(window) in reference_keys), default=-1)
    labels: list[WeakLabel] = []
    latched = False
    recovery_count = 0
    for window in ordered:
        key = window_key(window)
        if key in reference_keys or (ref_start >= 0 and window.start_sample < ref_end):
            history.clear()
            recovery_count = 0
            reason = "technical_reference_window" if key in reference_keys else "technical_reference_period"
            labels.append(_ambiguity_label(window, reason, reference, state_transition="reference_only", rule_version=version))
            continue
        if not window.eligible_for_label:
            history.clear()
            recovery_count = 0
            labels.append(_ambiguity_label(window, window.rejection_reason or "window_ineligible", reference, state_transition="qc_ambiguous", state_latched=latched, rule_version=version))
            continue
        row = feature_by_key.get(key)
        if not _valid_feature(row):
            history.clear()
            recovery_count = 0
            labels.append(_ambiguity_label(window, "indicator_invalid", reference, state_transition="feature_ambiguous", state_latched=latched, rule_version=version))
            continue
        observed = {"low_fraction": float(row["line_excluded_low_frequency_fraction_20_80"])}
        delta = observed["low_fraction"] - reference.center["low_fraction"]
        deltas = {"low_frequency_fraction_change": delta}
        thresholds = {"low_frequency_fraction_increase": threshold, "recovery_low_frequency_fraction_abs_delta": threshold / 2.0}
        candidate = delta > threshold
        stable = abs(delta) <= threshold
        recovery = abs(delta) <= threshold / 2.0
        history.append((window, bool(candidate)))
        candidate_count = sum(item[1] for item in history)
        span = history[-1][0].start_time_s + history[-1][0].duration_s - history[0][0].start_time_s if history else 0.0
        if latched:
            if candidate:
                recovery_count = 0
                confidence = min(1.0, max(0.0, delta / max(threshold, np.finfo(float).eps) - 1.0))
                labels.append(_assigned_label(window, reference, 1, confidence, "low_frequency_change_latched", "latched_candidate", observed, deltas, thresholds, candidate_count, span, True, version))
            elif recovery:
                recovery_count += 1
                if recovery_count >= int(settings["recovery_consecutive_window_count"]):
                    latched = False
                    recovery_count = 0
                    labels.append(_assigned_label(window, reference, 0, 1.0, "recovery_confirmed", "fatigue_to_non_fatigue_like", observed, deltas, thresholds, candidate_count, span, False, version))
                else:
                    labels.append(_ambiguity_label(window, f"recovery_pending_{recovery_count}", reference, state_transition="recovery_pending", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=candidate_count, persistence_span_s=span, state_latched=True, rule_version=version))
            else:
                recovery_count = 0
                labels.append(_ambiguity_label(window, "discordant_while_fatigue_latched", reference, state_transition="latched_discordant", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=candidate_count, persistence_span_s=span, state_latched=True, rule_version=version))
        elif candidate and candidate_count >= required_candidates and span >= minimum_span:
            latched = True
            confidence = min(1.0, max(0.0, delta / max(threshold, np.finfo(float).eps) - 1.0))
            labels.append(_assigned_label(window, reference, 1, confidence, "persistent_low_frequency_fraction_increase", "enter_fatigue_like", observed, deltas, thresholds, candidate_count, span, True, version))
        elif candidate:
            labels.append(_ambiguity_label(window, "candidate_persistence_not_met", reference, state_transition="entry_pending", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=candidate_count, persistence_span_s=span, rule_version=version))
        elif stable:
            confidence = max(0.0, 1.0 - abs(delta) / max(threshold, np.finfo(float).eps))
            labels.append(_assigned_label(window, reference, 0, confidence, "within_low_frequency_fraction_reference", "stable_non_fatigue_like", observed, deltas, thresholds, candidate_count, span, False, version))
        else:
            labels.append(_ambiguity_label(window, "low_frequency_fraction_discordant", reference, state_transition="discordant_evidence", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=candidate_count, persistence_span_s=span, rule_version=version))
    return labels


def label_log_mdf_slope_rule(
    windows: Sequence[SignalWindow],
    feature_by_key: Mapping[str, Mapping[str, object]],
    reference: ReferenceEstimate,
    config: dict,
    *,
    scaled_mad_multiple: float | None = None,
) -> list[WeakLabel]:
    """Alternative proxy requiring a persistent negative log-MDF slope and low-band shift."""
    settings = config["labels"]
    version = "log_mdf_slope_low_fraction_v1"
    ordered = sorted(windows, key=lambda item: (item.start_time_s, item.start_sample))
    if not reference.available or not reference.stable:
        reason = ";".join((reference.status, *reference.reasons))
        return [
            _ambiguity_label(window, reason, reference, state_transition="reference_blocked", rule_version=version)
            for window in ordered
        ]
    mad_multiple = float(scaled_mad_multiple if scaled_mad_multiple is not None else settings["minimum_scaled_mad_multiple"])
    duration = ordered[0].duration_s if ordered else float(config["run"]["primary_window_s"])
    frequency_threshold = max(
        float(settings["minimum_mdf_mnf_drop_hz"]) * float(config["run"]["primary_window_s"]) / duration,
        mad_multiple * reference.scaled_mad["mdf"],
    )
    low_mad = float(reference.scaled_mad["low_fraction"])
    low_threshold = mad_multiple * low_mad if low_mad > 0 else float(settings["minimum_low_frequency_fraction_increase_when_mad_zero"])
    rolling_count, required_candidates, minimum_span = _entry_geometry(duration, settings) if ordered else (4, 3, 1.25)
    history: deque[tuple[SignalWindow, bool, float]] = deque(maxlen=rolling_count)
    reference_keys = set(reference.reference_window_keys)
    ref_end = max(
        (window.start_sample + int(round(window.duration_s * window.sampling_rate_hz)) for window in ordered if window_key(window) in reference_keys),
        default=-1,
    )
    ref_start = min((window.start_sample for window in ordered if window_key(window) in reference_keys), default=-1)
    labels: list[WeakLabel] = []
    latched = False
    recovery_count = 0
    for window in ordered:
        key = window_key(window)
        if key in reference_keys or (ref_start >= 0 and window.start_sample < ref_end):
            history.clear()
            recovery_count = 0
            reason = "technical_reference_window" if key in reference_keys else "technical_reference_period"
            labels.append(_ambiguity_label(window, reason, reference, state_transition="reference_only", rule_version=version))
            continue
        if not window.eligible_for_label:
            history.clear()
            recovery_count = 0
            labels.append(_ambiguity_label(window, window.rejection_reason or "window_ineligible", reference, state_transition="qc_ambiguous", state_latched=latched, rule_version=version))
            continue
        row = feature_by_key.get(key)
        if not _valid_feature(row):
            history.clear()
            recovery_count = 0
            labels.append(_ambiguity_label(window, "indicator_invalid", reference, state_transition="feature_ambiguous", state_latched=latched, rule_version=version))
            continue
        mdf = float(row["median_frequency_hz"])
        low_fraction = float(row["line_excluded_low_frequency_fraction_20_80"])
        if mdf <= 0:
            history.clear()
            recovery_count = 0
            labels.append(_ambiguity_label(window, "nonpositive_mdf", reference, state_transition="feature_ambiguous", state_latched=latched, rule_version=version))
            continue
        log_mdf = float(np.log(mdf))
        history.append((window, False, log_mdf))
        times = np.asarray([item[0].start_time_s + item[0].duration_s / 2.0 for item in history], dtype=np.float64)
        logs = np.asarray([item[2] for item in history], dtype=np.float64)
        slope = float(np.polyfit(times, logs, 1)[0]) if logs.size >= 2 and np.ptp(times) > 0 else 0.0
        mdf_drop = float(reference.center["mdf"] - mdf)
        low_change = float(low_fraction - reference.center["low_fraction"])
        candidate = slope < 0 and mdf_drop > frequency_threshold and low_change > low_threshold
        history[-1] = (window, bool(candidate), log_mdf)
        candidate_count = sum(item[1] for item in history)
        span = history[-1][0].start_time_s + history[-1][0].duration_s - history[0][0].start_time_s if history else 0.0
        observed = {"median_frequency_hz": mdf, "low_frequency_fraction_20_80": low_fraction, "log_mdf_slope_per_s": slope}
        deltas = {
            "mdf_drop_hz": mdf_drop,
            "mdf_relative_drop": mdf_drop / max(abs(reference.center["mdf"]), np.finfo(float).eps),
            "low_frequency_fraction_change": low_change,
        }
        thresholds = {
            "mdf_drop_hz": frequency_threshold,
            "low_frequency_fraction_increase": low_threshold,
            "minimum_log_mdf_slope_per_s": 0.0,
            "recovery_mdf_abs_delta_hz": frequency_threshold / 2.0,
            "recovery_low_frequency_fraction_abs_delta": low_threshold / 2.0,
        }
        stable = abs(mdf_drop) <= frequency_threshold and abs(low_change) <= low_threshold and slope >= 0
        recovery = abs(mdf_drop) <= frequency_threshold / 2.0 and abs(low_change) <= low_threshold / 2.0
        if latched:
            if candidate:
                recovery_count = 0
                confidence = min(1.0, max(0.0, min(mdf_drop / max(frequency_threshold, np.finfo(float).eps), low_change / max(low_threshold, np.finfo(float).eps)) - 1.0))
                labels.append(_assigned_label(window, reference, 1, confidence, "log_mdf_slope_latched", "latched_candidate", observed, deltas, thresholds, candidate_count, span, True, version))
            elif recovery:
                recovery_count += 1
                if recovery_count >= int(settings["recovery_consecutive_window_count"]):
                    latched = False
                    recovery_count = 0
                    labels.append(_assigned_label(window, reference, 0, 1.0, "recovery_confirmed", "fatigue_to_non_fatigue_like", observed, deltas, thresholds, candidate_count, span, False, version))
                else:
                    labels.append(_ambiguity_label(window, f"recovery_pending_{recovery_count}", reference, state_transition="recovery_pending", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=candidate_count, persistence_span_s=span, state_latched=True, rule_version=version))
            else:
                recovery_count = 0
                labels.append(_ambiguity_label(window, "discordant_while_fatigue_latched", reference, state_transition="latched_discordant", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=candidate_count, persistence_span_s=span, state_latched=True, rule_version=version))
        elif candidate and candidate_count >= required_candidates and span >= minimum_span:
            latched = True
            confidence = min(1.0, max(0.0, min(mdf_drop / max(frequency_threshold, np.finfo(float).eps), low_change / max(low_threshold, np.finfo(float).eps)) - 1.0))
            labels.append(_assigned_label(window, reference, 1, confidence, "persistent_log_mdf_slope_and_low_fraction_shift", "enter_fatigue_like", observed, deltas, thresholds, candidate_count, span, True, version))
        elif candidate:
            labels.append(_ambiguity_label(window, "candidate_persistence_not_met", reference, state_transition="entry_pending", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=candidate_count, persistence_span_s=span, rule_version=version))
        elif stable:
            margins = [1.0 - abs(mdf_drop) / max(frequency_threshold, np.finfo(float).eps), 1.0 - abs(low_change) / max(low_threshold, np.finfo(float).eps)]
            labels.append(_assigned_label(window, reference, 0, max(0.0, min(margins)), "within_log_mdf_and_low_fraction_reference", "stable_non_fatigue_like", observed, deltas, thresholds, candidate_count, span, False, version))
        else:
            labels.append(_ambiguity_label(window, "discordant_or_partial_indicators", reference, state_transition="discordant_evidence", observed=observed, deltas=deltas, thresholds=thresholds, persistence_candidates=candidate_count, persistence_span_s=span, rule_version=version))
    return labels
