from pathlib import Path

import numpy as np
import pandas as pd


def test_legacy_class_names_are_rendered_as_recording_position_proxies():
    from emg_fatigue.labels import display_proxy_label

    assert display_proxy_label("early_proxy_nonfatigue") == "early_recording_position_proxy"
    assert display_proxy_label("late_proxy_fatigue") == "late_recording_position_proxy"
    assert display_proxy_label("early_recording_position_proxy") == "early_recording_position_proxy"


def test_readme_and_data_dictionary_do_not_define_proxy_classes_as_fatigue():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    first_read = (root / "LEEME_PRIMERO.md").read_text(encoding="utf-8").lower()
    model_card = (root / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8").lower()
    dictionary = (root / "docs" / "data_dictionary.md").read_text(encoding="utf-8").lower()

    assert "weak elapsed-time approximation** of nonfatigue/fatigue" not in readme
    assert "early_recording_position_proxy" in dictionary
    assert "late_recording_position_proxy" in dictionary
    assert "early_proxy_nonfatigue" not in dictionary
    assert "late_proxy_fatigue" not in dictionary
    assert "la ventana es la observación" in first_read
    assert "no fue entrenado para distinguir fatiga de no fatiga" in first_read
    assert "each feature window is an observation" in model_card
    assert "not trained to distinguish fatigue from no fatigue" in model_card


def test_manuscript_builder_names_the_temporal_proxy_without_calling_it_fatigue():
    root = Path(__file__).resolve().parents[1]
    builder = (root / "tools" / "build_manuscript.py").read_text(encoding="utf-8").lower()
    assert "early-versus-late temporal approximation of acute quadriceps fatigue" not in builder
    assert "binary early-versus-late recording-position proxy" in builder
    assert "second-stage fatigue-proxy classifier" not in builder


def test_prediction_from_legacy_bundle_uses_corrected_proxy_display(monkeypatch):
    from dataclasses import asdict

    from emg_fatigue.contracts import Recording, StudyConfig
    from emg_fatigue import inference

    recording = Recording(
        "R", "G", "test", "left", "R1", np.array([0.0, 0.001]), np.array([1.0, -1.0]), "hash"
    )
    processed = {
        "qc": [{"status": "valid", "start": 0, "end": 2}],
    }
    feature_row = pd.DataFrame([{
        "start_s": 0.0,
        "end_s": 0.249,
        "qc_valid": True,
        "activity_occupancy": 1.0,
        "activity_eligible": True,
        "feature_status": "valid",
        "label": 0,
        "status": "labeled",
        "run_id": 0,
    }])
    monkeypatch.setattr(inference, "preprocess", lambda recording, config: processed)
    monkeypatch.setattr(inference, "detect_activity", lambda signal, config: processed)
    monkeypatch.setattr(inference, "make_windows", lambda processed, config: [])
    monkeypatch.setattr(inference, "extract_features", lambda processed, windows, config: feature_row)
    monkeypatch.setattr(
        inference,
        "predict_scores",
        lambda pipeline, frame: (np.array([0]), np.array([0.75]), "probability"),
    )

    result = inference.predict_recording(recording, {
        "config": asdict(StudyConfig()),
        "pipeline": object(),
        "label_vocabulary": {0: "early_proxy_nonfatigue", 1: "late_proxy_fatigue"},
    })

    assert result.loc[result.row_type == "window", "predicted_class"].tolist() == [
        "early_recording_position_proxy"
    ]
