from pathlib import Path


def test_defense_source_names_distinct_phase_and_fatigue_supervision_and_equation_indices():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools/build_defense_v2.mjs").read_text(encoding="utf-8")

    assert "Etiqueta: Y_fase" in source
    assert "Y_FATIGA SUPERVISA LOS TRES MODELOS DE FATIGA" in source
    assert "Y_FATIGA DE REFERENCIA SUPERVISA LAS TRES CABEZAS" not in source
    assert "x[n+1]−x[n]" in source
    assert "C_ZCR=Σ I{" in source
    assert "T=N/fs" in source
