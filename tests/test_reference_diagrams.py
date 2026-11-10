import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bilingual_window_and_architecture_diagrams_are_generated(tmp_path):
    path = ROOT / "tools" / "build_reference_architecture.py"
    spec = importlib.util.spec_from_file_location("reference_diagrams", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    outputs = module.build(tmp_path)
    names = {item.name for item in outputs}
    assert "windowing_protocol_es.svg" in names
    assert "windowing_protocol_en.svg" in names
    assert "reference_models_es.svg" in names
    assert "reference_models_en.svg" in names
    assert all((tmp_path / name).is_file() for name in names)

    for language in ("es", "en"):
        windowing = (tmp_path / f"windowing_protocol_{language}.svg").read_text(encoding="utf-8")
        models = (tmp_path / f"reference_models_{language}.svg").read_text(encoding="utf-8")
        for name in names:
            if name.endswith(".svg"):
                svg = (tmp_path / name).read_text(encoding="utf-8")
                assert all(line == line.rstrip() for line in svg.splitlines())
        assert "500" in windowing and "250" in windowing
        assert "N" in windowing and "fs" in windowing
        assert ("7 ventanas completas" if language == "es" else "7 complete windows") in windowing
        assert ("3 ventanas completas" if language == "es" else "3 complete windows") in windowing
        assert "W1" in windowing and "W7" in windowing
        assert "direct" in models.lower() or "directa" in models.lower()
        assert "expert" in models.lower() or "experto" in models.lower()
        assert "OOF" in models
        fatigue_target = "Y_fatiga" if language == "es" else "Y_fatigue"
        assert models.count(fatigue_target) >= 4
        assert ("cabezas" if language == "es" else "heads") in models.lower()
        phase_target = "Objetivo: Y_fase" if language == "es" else "Target: Y_phase"
        assert phase_target in models
        footer = (
            "Modelos de fatiga: X, Y_fatiga y grupos retenidos. Clasificador de fase: Y_fase."
            if language == "es"
            else "Fatigue models: X, Y_fatigue and held-out groups. Phase classifier target: Y_phase."
        )
        assert footer in models
        assert ("Todos comparten X, Y_fatiga" if language == "es" else "All models share X, Y_fatigue") not in models
