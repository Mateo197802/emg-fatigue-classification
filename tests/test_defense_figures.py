import numpy as np
import pandas as pd
import hashlib
import json


def test_sensitivity_figure_uses_spanish_labels_and_frozen_metric_values():
    from emg_fatigue.defense_figures import plot_sensitivity_es

    data = pd.DataFrame([
        {"analysis": "no_overlap", "status": "evaluated", "macro_f1_mean": 0.62, "macro_f1_sd": 0.08},
        {"analysis": "conditional_notch", "status": "evaluated", "macro_f1_mean": 0.64, "macro_f1_sd": 0.09},
    ])

    figure = plot_sensitivity_es(data)
    axis = figure.axes[0]

    assert [tick.get_text() for tick in axis.get_yticklabels()] == [
        "Sin solapamiento",
        "Notch condicional 50/60 Hz",
    ]
    assert axis.get_xlabel() == "Macro-F1 media ± DE por grupo"
    assert any(np.allclose(line.get_xdata(), [0.62, 0.64]) for line in axis.lines)


def test_ablation_figure_uses_spanish_labels_and_keeps_supplied_values():
    from emg_fatigue.defense_figures import plot_ablation_es

    data = pd.DataFrame([
        {"ablation": "without_amplitude", "macro_f1_mean": 0.64, "macro_f1_sd": 0.11},
        {"ablation": "without_spectral", "macro_f1_mean": 0.63, "macro_f1_sd": 0.12},
        {"ablation": "without_wavelet", "macro_f1_mean": 0.62, "macro_f1_sd": 0.10},
    ])

    figure = plot_ablation_es(data)
    axis = figure.axes[0]

    assert [tick.get_text() for tick in axis.get_yticklabels()] == [
        "Sin amplitud",
        "Sin espectro",
        "Sin wavelets",
    ]
    assert axis.get_xlabel() == "Macro-F1 media ± DE por grupo"
    assert np.allclose([patch.get_width() for patch in axis.patches], [0.64, 0.63, 0.62])


def test_signal_figure_has_spanish_signal_psd_and_filter_panels():
    from emg_fatigue.defense_figures import plot_signal_es

    t = np.arange(1000) / 1000
    signal = {
        "time": t[::4].tolist(),
        "stored": np.sin(2 * np.pi * 50 * t)[::4].tolist(),
        "filtered": np.sin(2 * np.pi * 80 * t)[::4].tolist(),
        "sample_rate_hz": 1000.0,
        "psd_frequency_hz": [1.0, 20.0, 80.0, 400.0, 500.0],
        "psd_stored": [0.1, 0.2, 1.0, 0.2, 0.01],
        "psd_filtered": [0.01, 0.2, 0.9, 0.2, 0.001],
        "filter_frequency_hz": [0.0, 20.0, 200.0, 400.0, 500.0],
        "filter_response_db": [-100.0, -3.0, 0.0, -3.0, -100.0],
    }

    figure = plot_signal_es(signal)
    titles = [axis.get_title() for axis in figure.axes]
    labels = [axis.get_xlabel() for axis in figure.axes] + [axis.get_ylabel() for axis in figure.axes]

    assert len(figure.axes) == 3
    assert titles == [
        "Señal registrada y salida del filtro",
        "Periodograma Hann unilateral",
        "Respuesta del filtro pasa banda",
    ]
    assert "Tiempo" in labels[0]
    assert "Frecuencia" in labels[1]
    assert "Cuentas ADC" in labels[3]
    assert all("Representative" not in axis.get_title() for axis in figure.axes)


def test_signal_figure_receipt_binds_signal_payload_and_source_provenance(tmp_path):
    from emg_fatigue.defense_figures import write_spanish_defense_figures

    (tmp_path / "sensitivity_summary.csv").write_text(
        "analysis,status,macro_f1_mean,macro_f1_sd\nno_overlap,evaluated,0.61,0.08\n",
        encoding="utf-8",
    )
    (tmp_path / "ablation_summary.csv").write_text(
        "ablation,macro_f1_mean,macro_f1_sd\nwithout_amplitude,0.61,0.08\n",
        encoding="utf-8",
    )
    time = np.arange(1000) / 1000
    signal = {
        "time": time.tolist(),
        "stored": np.sin(2 * np.pi * 50 * time).tolist(),
        "filtered": np.sin(2 * np.pi * 80 * time).tolist(),
        "sample_rate_hz": 1000.0,
        "psd_frequency_hz": [1.0, 20.0, 80.0, 400.0, 500.0],
        "psd_stored": [0.1, 0.2, 1.0, 0.2, 0.01],
        "psd_filtered": [0.01, 0.2, 0.9, 0.2, 0.001],
        "filter_frequency_hz": [0.0, 20.0, 200.0, 400.0, 500.0],
        "filter_response_db": [-100.0, -3.0, 0.0, -3.0, -100.0],
        "selection": "synthetic test fixture",
    }
    source = {
        "archive_sha256": "a" * 64,
        "source_member_manifest_sha256": "b" * 64,
        "slides_input_sha256": "c" * 64,
    }

    outputs = write_spanish_defense_figures(tmp_path, signal, source)
    receipt = json.loads(outputs["defense_figures_es.receipt.json"].read_text(encoding="utf-8"))
    canonical_signal = json.dumps(signal, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    assert receipt["inputs"]["signal_payload_sha256"] == hashlib.sha256(canonical_signal).hexdigest()
    assert receipt["inputs"]["source_archive_sha256"] == source["archive_sha256"]
    assert receipt["inputs"]["source_member_manifest_sha256"] == source["source_member_manifest_sha256"]
    assert receipt["inputs"]["slides_input_sha256"] == source["slides_input_sha256"]


def test_confusion_grid_aggregates_fold_counts_and_labels_axes_in_spanish():
    from emg_fatigue.defense_figures import plot_confusion_grid_es

    data = pd.DataFrame([
        {"model": "logistic_rms_mdf", "cm_0_0": 20, "cm_0_1": 5, "cm_1_0": 7, "cm_1_1": 18},
        {"model": "logistic_rms_mdf", "cm_0_0": 22, "cm_0_1": 5, "cm_1_0": 1, "cm_1_1": 21},
    ])

    figure = plot_confusion_grid_es(data, ["logistic_rms_mdf"], normalized=False)
    axis = figure.axes[0]

    assert axis.get_title() == "Logística RMS + MDF"
    assert axis.get_xlabel() == "Predicción"
    assert axis.get_ylabel() == "Referencia"
    assert [text.get_text() for text in axis.texts] == ["42", "10", "8", "39"]


def test_normalized_confusion_grid_displays_reference_row_proportions():
    from emg_fatigue.defense_figures import plot_confusion_grid_es

    data = pd.DataFrame([
        {"model": "baseline_10", "cm_0_0": 80, "cm_0_1": 20, "cm_1_0": 25, "cm_1_1": 75},
    ])

    figure = plot_confusion_grid_es(data, ["baseline_10"], normalized=True)

    assert [text.get_text() for text in figure.axes[0].texts] == ["0.80", "0.20", "0.25", "0.75"]


def test_confusion_writer_renders_and_hashes_all_required_procedures(tmp_path):
    from emg_fatigue.defense_figures import write_spanish_confusion_figures

    models_primary = ["majority", "logistic_rms_mdf", "logistic_all", "svm_rbf", "random_forest", "nested_selector"]
    models_supplemental = ["baseline_10", "phase_score", "phase_conditioned", "phase_only", "random_conditioned"]
    columns = ["model", "cm_0_0", "cm_0_1", "cm_1_0", "cm_1_1"]
    primary = pd.DataFrame([[m, 40, 10, 12, 38] for m in models_primary], columns=columns)
    supplemental = pd.DataFrame([[m, 41, 9, 11, 39] for m in models_supplemental], columns=columns)
    primary_path = tmp_path / "primary.csv"
    supplemental_path = tmp_path / "supplemental.csv"
    primary.to_csv(primary_path, index=False)
    supplemental.to_csv(supplemental_path, index=False)

    outputs = write_spanish_confusion_figures(tmp_path, primary_path, supplemental_path)
    receipt = json.loads(outputs["confusion_figures_es.receipt.json"].read_text(encoding="utf-8"))

    expected_names = {
        "primary_confusions_counts_es.png",
        "primary_confusions_normalized_es.png",
        "phase_confusions_counts_es.png",
        "phase_confusions_normalized_es.png",
    }
    assert expected_names.issubset(outputs)
    assert all(outputs[name].stat().st_size > 1_000 for name in expected_names)
    assert receipt["inputs"]["primary_results_sha256"] == hashlib.sha256(primary_path.read_bytes()).hexdigest()
    assert receipt["inputs"]["supplemental_confusions_sha256"] == hashlib.sha256(supplemental_path.read_bytes()).hexdigest()
    assert all(hashlib.sha256(outputs[name].read_bytes()).hexdigest() == digest for name, digest in receipt["outputs"].items())
