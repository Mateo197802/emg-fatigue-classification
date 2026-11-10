"""Create an anonymized, private waveform/PSD/filter-response diagnostic from a ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from emg_fatigue.contracts import read_config
from emg_fatigue.io import load_recordings
from emg_fatigue.preprocessing import preprocess


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_valid_second(recordings, config):
    for recording in recordings:
        if recording.series == "R0":
            continue
        processed = preprocess(recording, config)
        for run in processed["runs"]:
            fs = float(run["fs_hz"])
            n = int(round(fs))
            valid = np.asarray(run["valid"], dtype=bool)
            starts = np.flatnonzero(valid)
            for start in starts:
                end = start + n
                if end <= len(valid) and valid[start:end].all():
                    idx = np.arange(start, end)
                    raw_idx = int(run["start_index"]) + idx
                    return recording, run, idx, recording.x[raw_idx], run["x"][idx]
    raise ValueError("no one-second continuous QC-valid interval was found")


def build_signal_evidence(archive: Path, config_path: Path, output_dir: Path) -> dict:
    """Render signal, PSD and exact digital filter response for a valid interval."""
    archive = Path(archive).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = read_config(config_path)
    records = load_recordings(archive, config)
    recording, run, idx, raw, filtered = _first_valid_second(records, config)
    fs = float(run["fs_hz"])
    time_s = np.asarray(run["t"])[idx]
    raw_centered = np.asarray(raw, dtype=float) - float(np.mean(raw))
    filtered = np.asarray(filtered, dtype=float)
    f_raw, p_raw = signal.periodogram(
        raw_centered, fs, window="hann", detrend=False, scaling="density"
    )
    f_filtered, p_filtered = signal.periodogram(
        filtered, fs, window="hann", detrend=False, scaling="density"
    )
    sos = signal.butter(config.filter_order, config.band_hz, btype="bandpass", fs=fs, output="sos")
    f_response, response = signal.sosfreqz(sos, worN=4096, fs=fs)
    response_db = 20 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "signal_preprocessing_evidence.png"
    fig, axes = plt.subplots(3, 1, figsize=(13.2, 10), constrained_layout=True)
    fig.suptitle("Representative sEMG processing diagnostic", fontsize=16, weight="bold")
    axes[0].plot(time_s - time_s[0], raw_centered, color="#6e7a86", lw=0.75, label="Stored signal, centered")
    axes[0].plot(time_s - time_s[0], filtered, color="#285b79", lw=0.85, label="SOS band-pass output")
    axes[0].set_xlabel("Time within selected interval (s)")
    axes[0].set_ylabel("ADC counts")
    axes[0].set_title("Same continuous one-second interval")
    axes[0].legend(frameon=False, ncols=2)
    axes[0].grid(alpha=0.2)

    axes[1].semilogy(f_raw, np.maximum(p_raw, np.finfo(float).tiny), color="#6e7a86", lw=0.85, label="Before filtering")
    axes[1].semilogy(f_filtered, np.maximum(p_filtered, np.finfo(float).tiny), color="#285b79", lw=0.85, label="After filtering")
    axes[1].set_xlim(0, min(fs / 2, 500))
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power density (ADC counts²/Hz)")
    axes[1].set_title("Hann one-sided periodogram; this is not Welch averaging")
    axes[1].legend(frameon=False, ncols=2)
    axes[1].grid(alpha=0.2, which="both")

    axes[2].plot(f_response, response_db, color="#285b79", lw=1.2)
    axes[2].axvspan(config.band_hz[0], config.band_hz[1], color="#e5f2e8", alpha=0.75, label="Configured passband")
    axes[2].set_xlim(0, min(fs / 2, 500))
    axes[2].set_ylim(-100, 5)
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Magnitude (dB)")
    axes[2].set_title(f"Order-{2 * config.filter_order} digital band-pass; fs={fs:.6f} Hz; coefficients recomputed for this fs")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.2)
    fig.text(0.5, -0.012, "Anonymized QC-selected illustration. Stored values are ADC counts; no independent fatigue or phase labels are shown.", ha="center", fontsize=9)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    receipt = {
        "status": "private_signal_diagnostic_created",
        "scope": "one second of a continuous QC-valid formal sEMG recording; no outcome-based selection",
        "archive_sha256": sha256_file(archive),
        "source_recording_sha256": recording.sha256,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "effective_sampling_rate_hz": fs,
        "filter": {
            "type": "digital Butterworth bandpass, SOS",
            "prototype_order": config.filter_order,
            "digital_order": 2 * config.filter_order,
            "band_hz": list(config.band_hz),
            "forward_backward": True,
            "notch_enabled": config.notch,
        },
        "figure_sha256": sha256_file(figure_path),
        "figure_name": figure_path.name,
        "no_participant_or_source_member_names_in_figure": True,
    }
    receipt_path = output_dir / "signal_preprocessing_evidence.receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_signal_evidence(args.archive, args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
