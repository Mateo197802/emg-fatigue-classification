"""Experimental sEMG features kept separate from the frozen primary schemas.

Inputs are already preprocessed, contiguous, finite windows in ADC counts.
These descriptors are candidate features for signal-quality/redundancy audits;
they are not approved fatigue predictors and are not consumed by the primary
or reference-backed model pipelines.
"""

import numpy as np
from scipy import signal


CANDIDATE_FEATURE_SCHEMA = "emg_candidate_features_v1"
CANDIDATE_FEATURE_COLUMNS = (
    "zero_crossings_per_s",
    "slope_sign_changes_per_s",
    "integrated_emg_counts_s",
    "variance_counts2",
    "spectral_entropy_norm",
)


def candidate_window_features(
    x,
    fs,
    *,
    amplitude_threshold=0.0,
    slope_threshold=0.0,
    band_hz=(20.0, 400.0),
):
    """Compute five exploratory descriptors without changing primary features.

    Rates use the represented fixed-window duration ``N/fs``. Zero crossings
    count adjacent samples moving across the symmetric amplitude threshold;
    slope-sign changes count adjacent first differences moving across the
    symmetric slope threshold. Spectral entropy is normalized by ``log(K)``
    for the K one-sided periodogram bins in the closed analysis band.
    """
    try:
        values = np.asarray(x, dtype=float)
        sample_rate = float(fs)
        amplitude_threshold = float(amplitude_threshold)
        slope_threshold = float(slope_threshold)
        low_hz, high_hz = (float(value) for value in band_hz)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("signal, sample rate, band, and thresholds must be numeric") from error
    if values.ndim != 1 or len(values) < 56 or not np.isfinite(values).all():
        raise ValueError("candidate features require a finite one-dimensional window of at least 56 samples")
    if not np.isfinite(sample_rate) or sample_rate <= 2 * high_hz:
        raise ValueError("sampling rate must be finite and exceed the candidate band Nyquist limit")
    if not np.isfinite(low_hz) or not np.isfinite(high_hz) or not 0 < low_hz < high_hz:
        raise ValueError("candidate analysis band must be finite and positive")
    if (
        not np.isfinite(amplitude_threshold)
        or not np.isfinite(slope_threshold)
        or amplitude_threshold < 0
        or slope_threshold < 0
    ):
        raise ValueError("candidate thresholds must be finite and nonnegative")

    centered = values - values.mean()
    duration_s = len(centered) / sample_rate
    left = centered[:-1]
    right = centered[1:]
    zero_crossings = np.count_nonzero(
        ((left >= amplitude_threshold) & (right < -amplitude_threshold))
        | ((left <= -amplitude_threshold) & (right > amplitude_threshold))
    )
    slopes = np.diff(centered)
    slope_sign_changes = np.count_nonzero(
        ((slopes[:-1] > slope_threshold) & (slopes[1:] < -slope_threshold))
        | ((slopes[:-1] < -slope_threshold) & (slopes[1:] > slope_threshold))
    )

    frequencies, density = signal.periodogram(
        centered, sample_rate, window="hann", detrend=False, scaling="density"
    )
    in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_power = np.asarray(density[in_band], dtype=float)
    power_sum = float(band_power.sum())
    if len(band_power) < 2 or not np.isfinite(power_sum) or power_sum <= 1e-20:
        raise ValueError("candidate spectral feature requires positive in-band energy")
    probabilities = band_power / power_sum
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    normalized_entropy = entropy / float(np.log(len(probabilities)))

    result = {
        "zero_crossings_per_s": float(zero_crossings / duration_s),
        "slope_sign_changes_per_s": float(slope_sign_changes / duration_s),
        "integrated_emg_counts_s": float(np.abs(centered).sum() / sample_rate),
        "variance_counts2": float(np.mean(centered**2)),
        "spectral_entropy_norm": float(normalized_entropy),
    }
    if not np.isfinite(list(result.values())).all():
        raise ValueError("candidate features are nonfinite")
    return result
