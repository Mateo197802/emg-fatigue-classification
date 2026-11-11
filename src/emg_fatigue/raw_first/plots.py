"""Private inspection plots and aggregate-only audit figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from .contracts import ProcessedSegment, RawRecording
from .preprocessing import contiguous_raw_segments


def make_sampling_histogram(audits: list[dict[str, object]], output_path: Path) -> Path:
    rates = np.asarray([row["effective_sampling_rate_hz"] for row in audits], dtype=np.float64)
    rates = rates[np.isfinite(rates)]
    # Sub-nanohertz floating-point noise around a common rate should not
    # dominate matplotlib's automatic axis offset and bin selection.
    rates = np.round(rates, 3)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.hist(rates, bins=min(20, max(5, int(np.sqrt(max(1, len(rates)))))), color="#2b6cb0", edgecolor="#1a365d")
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.set_title("Effective sampling rates across source files")
    axis.set_xlabel("Robust rate estimated from valid timestamp intervals (Hz)")
    axis.set_ylabel("Source file count")
    axis.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def make_global_signal_summary(audits: list[dict[str, object]], output_path: Path) -> Path:
    frame = pd.DataFrame(audits)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].hist(np.log10(np.maximum(frame["adc_rms"].to_numpy(dtype=float), 1e-12)), bins=24, color="#2b6cb0", edgecolor="#1a365d")
    axes[0].set_title("ADC-value RMS across files")
    axes[0].set_xlabel("log10(RMS), exported ADC-derived scale")
    axes[0].set_ylabel("Source file count")
    rates = np.round(frame["effective_sampling_rate_hz"].to_numpy(dtype=float), 3)
    histogram = axes[1].hist2d(
        rates,
        frame["spectral_median_hz"],
        bins=14,
        cmap="Blues",
    )
    figure.colorbar(histogram[3], ax=axes[1], label="Source file count per bin")
    axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[1].set_title("Aggregate sampling rate and spectral median")
    axes[1].set_xlabel("Effective sampling rate (Hz)")
    axes[1].set_ylabel("Median PSD frequency (Hz)")
    for axis in axes:
        axis.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def make_before_after_plots(
    recordings: list[RawRecording],
    processed: list[ProcessedSegment],
    output_dir: Path,
    sampling_rate_audit: dict[str, dict[str, object]],
    sampling_plan: dict[str, object],
) -> list[Path]:
    """Write one private waveform/PSD view per opaque source group."""
    by_record: dict[str, list[ProcessedSegment]] = {}
    for segment in processed:
        by_record.setdefault(segment.recording_id, []).append(segment)
    selected: dict[str, RawRecording] = {}
    for recording in sorted(recordings, key=lambda item: (item.group_id, item.recording_id)):
        selected.setdefault(recording.group_id, recording)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for group_id, recording in selected.items():
        fs = float(sampling_rate_audit[recording.recording_id]["effective_sampling_rate_hz"])
        segment = (by_record.get(recording.recording_id) or [None])[0]
        if segment is None or not np.isfinite(fs) or fs <= 0:
            continue
        raw_parts, _ = contiguous_raw_segments(recording, float(sampling_rate_audit[recording.recording_id]["gap_threshold_s"]))
        if not raw_parts:
            continue
        raw_t, raw_x = raw_parts[0]
        limit = min(len(raw_x), int(3 * fs))
        raw_t = raw_t[:limit]
        raw_x = raw_x[:limit]
        processed_limit = min(len(segment.signal), int(3 * segment.sampling_rate_hz))
        filt_t = segment.start_time_s + np.arange(processed_limit) / segment.sampling_rate_hz
        filt_x = segment.signal[:processed_limit]

        figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        if raw_t.size:
            axes[0, 0].plot(raw_t - raw_t[0], raw_x, linewidth=0.65, color="#4a5568")
        axes[0, 0].set_title(f"Raw waveform — {recording.recording_id}")
        axes[0, 0].set_xlabel("Time within first 3 seconds (s)")
        axes[0, 0].set_ylabel("ADC_Value (exported scale)")
        axes[0, 1].plot(filt_t - segment.start_time_s, filt_x, linewidth=0.65, color="#2b6cb0")
        axes[0, 1].set_title("Detrended, filtered waveform")
        axes[0, 1].set_xlabel("Time within processed segment (s)")
        axes[0, 1].set_ylabel("ADC-derived units")

        if raw_x.size >= 256:
            raw_fs = fs
            f_raw, p_raw = signal.welch(signal.detrend(raw_x), fs=raw_fs, nperseg=min(2048, raw_x.size))
            axes[1, 0].semilogy(f_raw, p_raw + 1e-20, color="#4a5568", linewidth=0.8)
        axes[1, 0].set_xlim(0, min(500, fs / 2))
        axes[1, 0].set_title("Raw PSD")
        axes[1, 0].set_xlabel("Frequency (Hz)")
        axes[1, 0].set_ylabel("Power spectral density")
        if filt_x.size >= 256:
            f_filt, p_filt = signal.welch(filt_x, fs=segment.sampling_rate_hz, nperseg=min(2048, filt_x.size))
            axes[1, 1].semilogy(f_filt, p_filt + 1e-20, color="#2b6cb0", linewidth=0.8)
        axes[1, 1].set_xlim(0, min(500, segment.sampling_rate_hz / 2))
        axes[1, 1].set_title(
            f"Processed PSD ({sampling_plan['bandpass_low_hz']:.1f}-{sampling_plan['bandpass_high_hz']:.1f} Hz; "
            f"notch={sampling_plan['notch_frequency_hz']})"
        )
        axes[1, 1].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_ylabel("Power spectral density")
        for axis in axes.ravel():
            axis.grid(alpha=0.22)
        destination = output_dir / f"private_waveform_psd_{group_id}.png"
        figure.savefig(destination, dpi=150)
        plt.close(figure)
        paths.append(destination)
    return paths
