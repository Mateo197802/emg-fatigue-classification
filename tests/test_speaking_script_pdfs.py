from pathlib import Path


def test_read_script_keeps_metadata_out_of_appendix_narration(tmp_path: Path):
    from tools.build_speaking_script_pdfs import read_script

    lines = ["# Defense script", "", "16 principal slides. 13 appendices.", ""]
    for slide_number in range(1, 30):
        lines.extend([f"## {slide_number}. Slide {slide_number}", ""])
        if slide_number <= 16:
            lines.extend(["Scheduled time: 20 s.", ""])
        else:
            lines.extend(["Appendix for questions.", ""])
        lines.extend([f"Narration for slide {slide_number}.", "", "Source: docs/evidence.csv", ""])

    script_path = tmp_path / "Defense_Script_EN.md"
    script_path.write_text("\n".join(lines), encoding="utf-8")

    _, entries = read_script(script_path)

    assert len(entries) == 29
    for heading, time_text, narration, source in entries:
        assert source == "docs/evidence.csv"
        assert "Source:" not in narration
        assert "Scheduled time:" not in narration
    assert entries[-1][2] == "Appendix for questions.\n\nNarration for slide 29."
