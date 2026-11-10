from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_active_and_legacy_slide_builders_remove_pending_evidence_placeholder():
    builders = [ROOT / "tools" / "build_defense_v2.mjs", ROOT / "tools" / "build_slides.mjs"]

    for builder in builders:
        text = builder.read_text(encoding="utf-8").casefold()
        assert "evidencia pendiente" not in text
        assert "pending evidence" not in text
