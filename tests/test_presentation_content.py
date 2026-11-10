from pathlib import Path
import json


def test_defense_source_names_distinct_phase_and_fatigue_supervision_and_equation_indices():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools/build_defense_v2.mjs").read_text(encoding="utf-8")

    assert "Etiqueta: Y_fase" in source
    assert "Y_FATIGA SUPERVISA LOS TRES MODELOS DE FATIGA" in source
    assert "Y_FATIGA DE REFERENCIA SUPERVISA LAS TRES CABEZAS" not in source
    assert "x[n+1]−x[n]" in source
    assert "C_ZCR=Σ I{" in source
    assert "T=N/fs" in source


def test_fatigue_defense_uses_model_keys_from_published_summary():
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "results/reference_fatigue/perceived_fatigue_slides.json").read_text(encoding="utf-8"))
    keys = {row["architecture"] for row in summary["models"]}
    source = (root / "tools/build_defense_v2.mjs").read_text(encoding="utf-8")

    assert {"hist_gradient_boosting", "nested_hist_gradient_boosting", "specialist_by_prime_mover", "time_only_control"}.issubset(keys)
    assert "['HGB global','hist_gradient_boosting']" in source
    assert "['HGB con búsqueda anidada','nested_hist_gradient_boosting']" in source
