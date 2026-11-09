"""Spanish-language figures generated from frozen defense evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

# Figure generation is part of the headless, reproducible delivery pipeline.
# Select a non-interactive backend before importing pyplot so this module does
# not depend on a local Tk installation or an attached display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SENSITIVITY_LABELS_ES = {
    "no_overlap": "Sin solapamiento",
    "window_500ms": "Ventana 500 ms",
    "window_1000ms": "Ventana 1.000 ms",
    "activity_50pct": "Actividad ≥ 50%",
    "boundary_minus_5pct": "Límite −5%",
    "boundary_plus_5pct": "Límite +5%",
    "boundary_minus_10pct": "Límite −10%",
    "boundary_plus_10pct": "Límite +10%",
    "conditional_notch": "Notch condicional 50/60 Hz",
}

ABLATION_LABELS_ES = {
    "without_amplitude": "Sin amplitud",
    "without_spectral": "Sin espectro",
    "without_wavelet": "Sin wavelets",
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def plot_sensitivity_es(frame: pd.DataFrame):
    """Plot grouped macro-F1 sensitivity estimates with their SDs."""
    _require_columns(frame, {"analysis", "status", "macro_f1_mean", "macro_f1_sd"}, "sensitivity data")
    evaluated = frame.loc[frame["status"].eq("evaluated")].copy()
    unknown = set(evaluated["analysis"].astype(str)) - set(SENSITIVITY_LABELS_ES)
    if unknown:
        raise ValueError(f"Sensitivity labels need Spanish translations: {sorted(unknown)}")
    labels = [SENSITIVITY_LABELS_ES[str(name)] for name in evaluated["analysis"]]
    means = evaluated["macro_f1_mean"].astype(float).to_numpy()
    sds = evaluated["macro_f1_sd"].astype(float).to_numpy()
    if not np.isfinite(means).all() or not np.isfinite(sds).all() or (sds < 0).any():
        raise ValueError("Sensitivity estimates must be finite and standard deviations nonnegative")

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=sds, fmt="o", color="#285b79", ecolor="#285b79", capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 media ± DE por grupo")
    ax.grid(axis="x", color="#d9e1e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_ablation_es(frame: pd.DataFrame):
    """Plot ablation macro-F1 estimates and SDs in Spanish."""
    _require_columns(frame, {"ablation", "macro_f1_mean", "macro_f1_sd"}, "ablation data")
    unknown = set(frame["ablation"].astype(str)) - set(ABLATION_LABELS_ES)
    if unknown:
        raise ValueError(f"Ablation labels need Spanish translations: {sorted(unknown)}")
    labels = [ABLATION_LABELS_ES[str(name)] for name in frame["ablation"]]
    means = frame["macro_f1_mean"].astype(float).to_numpy()
    sds = frame["macro_f1_sd"].astype(float).to_numpy()
    if not np.isfinite(means).all() or not np.isfinite(sds).all() or (sds < 0).any():
        raise ValueError("Ablation estimates must be finite and standard deviations nonnegative")

    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=sds, color="#39735d", ecolor="#24343f", capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 media ± DE por grupo")
    ax.grid(axis="x", color="#d9e1e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_signal_es(signal: dict):
    """Plot signal, Hann periodograms and digital filter response in Spanish."""
    required = {
        "time", "stored", "filtered", "sample_rate_hz", "psd_frequency_hz",
        "psd_stored", "psd_filtered", "filter_frequency_hz", "filter_response_db",
    }
    missing = required - set(signal)
    if missing:
        raise ValueError(f"signal evidence is missing required fields: {sorted(missing)}")
    trace_lengths = {len(signal[key]) for key in ("time", "stored", "filtered")}
    psd_lengths = {len(signal[key]) for key in ("psd_frequency_hz", "psd_stored", "psd_filtered")}
    response_lengths = {len(signal[key]) for key in ("filter_frequency_hz", "filter_response_db")}
    if len(trace_lengths) != 1 or len(psd_lengths) != 1 or len(response_lengths) != 1:
        raise ValueError("Signal, PSD or filter-response arrays have mismatched lengths")
    if not float(signal["sample_rate_hz"]) > 0:
        raise ValueError("sample_rate_hz must be positive")

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(signal["time"], signal["stored"], color="#8a9aa5", linewidth=0.9, label="EMG registrada centrada")
    axes[0].plot(signal["time"], signal["filtered"], color="#285b79", linewidth=0.9, label="Salida SOS 20–400 Hz")
    axes[0].set_title("Señal registrada y salida del filtro")
    axes[0].set_xlabel("Tiempo en el intervalo seleccionado (s)")
    axes[0].set_ylabel("Cuentas ADC")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    axes[0].grid(color="#e0e6ea", linewidth=0.6)

    freq = np.asarray(signal["psd_frequency_hz"], dtype=float)
    raw = np.asarray(signal["psd_stored"], dtype=float)
    filtered = np.asarray(signal["psd_filtered"], dtype=float)
    positive = freq > 0
    axes[1].semilogy(freq[positive], np.maximum(raw[positive], np.finfo(float).tiny), color="#8a9aa5", linewidth=0.9, label="Antes del filtro")
    axes[1].semilogy(freq[positive], np.maximum(filtered[positive], np.finfo(float).tiny), color="#285b79", linewidth=0.9, label="Después del filtro")
    axes[1].set_title("Periodograma Hann unilateral")
    axes[1].set_xlabel("Frecuencia (Hz)")
    axes[1].set_ylabel("Densidad de potencia (cuentas²/Hz)")
    axes[1].legend(loc="lower left", frameon=False, ncol=2)
    axes[1].grid(color="#e0e6ea", linewidth=0.6)

    axes[2].plot(signal["filter_frequency_hz"], signal["filter_response_db"], color="#285b79", linewidth=1.2)
    axes[2].axvspan(20, 400, color="#dcebe2", alpha=0.7, label="Banda configurada")
    axes[2].set_title("Respuesta del filtro pasa banda")
    axes[2].set_xlabel("Frecuencia (Hz)")
    axes[2].set_ylabel("Magnitud (dB)")
    axes[2].set_xlim(0, float(signal["sample_rate_hz"]) / 2)
    axes[2].set_ylim(-100, 5)
    axes[2].legend(loc="lower center", frameon=False)
    axes[2].grid(color="#e0e6ea", linewidth=0.6)
    return fig


def write_spanish_defense_figures(
    root: Path,
    signal: dict,
    source_provenance: dict[str, str],
) -> dict[str, Path]:
    """Render the three Spanish defense plots from frozen stage inputs."""
    provenance_fields = {
        "archive_sha256",
        "source_member_manifest_sha256",
        "slides_input_sha256",
    }
    missing = provenance_fields - set(source_provenance)
    if missing:
        raise ValueError(f"figure source provenance is missing fields: {sorted(missing)}")
    for field in provenance_fields:
        value = str(source_provenance[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError(f"{field} must be a SHA-256 hex digest")

    root = Path(root)
    assets = root / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    specs = {
        "sensitivity_es.png": plot_sensitivity_es(pd.read_csv(root / "sensitivity_summary.csv")),
        "ablation_es.png": plot_ablation_es(pd.read_csv(root / "ablation_summary.csv")),
        "signal_preprocessing_evidence_es.png": plot_signal_es(signal),
    }
    outputs = {}
    for name, figure in specs.items():
        path = assets / name
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        outputs[name] = path

    signal_payload = json.dumps(
        signal, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    receipt = {
        "status": "generated",
        "language": "es",
        "data_scope": "same frozen sensitivity, ablation and QC-selected signal inputs as the defense build",
        "inputs": {
            **{
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("sensitivity_summary.csv", "ablation_summary.csv")
            },
            "signal_payload_sha256": hashlib.sha256(signal_payload).hexdigest(),
            "source_archive_sha256": source_provenance["archive_sha256"],
            "source_member_manifest_sha256": source_provenance["source_member_manifest_sha256"],
            "slides_input_sha256": source_provenance["slides_input_sha256"],
        },
        "outputs": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in outputs.items()},
    }
    receipt_path = assets / "defense_figures_es.receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs[receipt_path.name] = receipt_path
    return outputs
