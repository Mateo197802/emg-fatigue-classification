from pathlib import Path
import re


def test_read_script_keeps_metadata_out_of_appendix_narration(tmp_path: Path):
    from tools.build_speaking_script_pdfs import read_script

    lines = ["# Defense script", "", "16 principal slides. 14 appendices.", ""]
    for slide_number in range(1, 31):
        lines.extend([f"## {slide_number}. Slide {slide_number}", ""])
        if slide_number <= 16:
            lines.extend(["Scheduled time: 20 s.", ""])
        else:
            lines.extend(["Appendix for questions.", ""])
        lines.extend([f"Narration for slide {slide_number}.", "", "Source: docs/evidence.csv", ""])

    script_path = tmp_path / "Defense_Script_EN.md"
    script_path.write_text("\n".join(lines), encoding="utf-8")

    _, entries = read_script(script_path)

    assert len(entries) == 30
    for heading, time_text, narration, source in entries:
        assert source == "docs/evidence.csv"
        assert "Source:" not in narration
        assert "Scheduled time:" not in narration
    assert entries[-1][2] == "Appendix for questions.\n\nNarration for slide 30."


def test_checked_in_defense_scripts_match_all_30_storyboard_slides():
    from tools.build_speaking_script_pdfs import read_script

    root = Path(__file__).resolve().parents[1]
    script_paths = [
        root / "docs/defense-scripts/Guion_defensa_ES.md",
        root / "docs/defense-scripts/Defense_Script_EN.md",
    ]
    for script_path in script_paths:
        _, entries = read_script(script_path)
        slide_numbers = [int(re.match(r"(\d+)\.", heading).group(1)) for heading, *_ in entries]
        assert slide_numbers == list(range(1, 31))
        candidate_heading = entries[24][0].lower()
        assert ("candidate" in candidate_heading and "feature" in candidate_heading) or "características candidatas" in candidate_heading
        assert "signal" in entries[29][0].lower() or "señal" in entries[29][0].lower()
        source_text = script_path.read_text(encoding="utf-8").lower()
        assert any(
            phrase in source_text
            for phrase in ("14 appendix", "fourteen appendix", "14 anex", "catorce anex")
        )
        assert "thirteen appendix" not in source_text and "trece anex" not in source_text


def test_defense_script_readme_links_versioned_bilingual_pdfs():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "docs/defense-scripts/README.md").read_text(encoding="utf-8")
    assert "pdf/20261109_v5/Guion_defensa_ES.pdf" in readme
    assert "pdf/20261109_v5/Defense_Script_EN.pdf" in readme
