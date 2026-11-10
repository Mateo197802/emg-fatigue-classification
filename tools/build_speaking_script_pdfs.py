"""Render Spanish and English slide-by-slide speaking scripts as readable PDFs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

def read_script(path: Path):
    title = path.stem.replace("_", " ").replace("-", " ")
    chunks = re.split(r"(?m)^##\s+", path.read_text(encoding="utf-8"))
    entries = []
    for chunk in chunks[1:]:
        lines = chunk.strip().splitlines()
        heading = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        time_match = re.search(r"^(?:Tiempo previsto|Scheduled time):\s*([^\n]+)", body, re.M)
        source_match = re.search(r"^(?:Fuente|Source):\s*(.+)$", body, re.M)
        time_text = time_match.group(1).strip() if time_match else ""
        source = source_match.group(1).strip() if source_match else ""
        narration_lines = [
            line for line in body.splitlines()
            if not re.match(r"^(?:Tiempo previsto|Scheduled time|Fuente|Source):\s*", line.strip())
        ]
        note = re.sub(r"\n{3,}", "\n\n", "\n".join(narration_lines)).strip()
        entries.append((heading, time_text, note, source))
    if len(entries) != 33:
        raise ValueError(f"expected 33 slide entries in {path}, found {len(entries)}")
    return title, entries


def render_one(source: Path, destination: Path, language: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

    title, entries = read_script(source)
    base = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("ArialUnicode", str(base / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("ArialUnicode-Bold", str(base / "arialbd.ttf")))
    pdfmetrics.registerFontFamily("ArialUnicode", normal="ArialUnicode", bold="ArialUnicode-Bold")
    styles = getSampleStyleSheet()
    page_title = ParagraphStyle(
        "ScriptTitle", parent=styles["Title"], fontName="ArialUnicode-Bold", fontSize=20,
        leading=25, textColor=colors.HexColor("#203346"), alignment=TA_LEFT, spaceAfter=6*mm,
    )
    heading_style = ParagraphStyle(
        "SlideHeading", parent=styles["Heading2"], fontName="ArialUnicode-Bold", fontSize=12.5,
        leading=16, textColor=colors.HexColor("#234f6b"), spaceBefore=4*mm, spaceAfter=1.5*mm,
    )
    time_style = ParagraphStyle(
        "SlideTime", parent=styles["Normal"], fontName="ArialUnicode-Bold", fontSize=8.5,
        leading=11, textColor=colors.HexColor("#526775"), spaceAfter=1.2*mm,
    )
    body_style = ParagraphStyle(
        "SlideNarration", parent=styles["BodyText"], fontName="ArialUnicode", fontSize=10.2,
        leading=14.3, textColor=colors.HexColor("#182a36"), spaceAfter=1.3*mm,
        splitLongWords=True, wordWrap="CJK",
    )
    source_style = ParagraphStyle(
        "SlideSource", parent=styles["BodyText"], fontName="ArialUnicode", fontSize=8.2,
        leading=11, textColor=colors.HexColor("#687984"), wordWrap="CJK",
    )
    story = [Paragraph(escape(title), page_title)]
    principal_count = 19
    if language == "es":
        intro = "19 diapositivas principales · 600 segundos · 14 anexos"
    else:
        intro = "19 principal slides · 600 seconds · 14 appendices"
    story.extend([Paragraph(escape(intro), source_style), Spacer(1, 5*mm)])
    for index, (heading, time_text, note, source_text) in enumerate(entries, 1):
        if index == principal_count + 1:
            story.append(Spacer(1, 3*mm))
            story.append(Paragraph("ANEXOS / APPENDICES", heading_style))
        block = [Paragraph(escape(heading), heading_style)]
        if time_text:
            prefix = "Tiempo previsto" if language == "es" else "Scheduled time"
            block.append(Paragraph(escape(f"{prefix}: {time_text}"), time_style))
        if note:
            block.append(Paragraph(escape(note).replace("\n", "<br/>"), body_style))
        if source_text:
            # Add break opportunities to long URLs/path strings for legible wrapping.
            printable_source = source_text.replace("/", "/\u200b").replace("\\", "\\\u200b")
            label = "Fuente" if language == "es" else "Source"
            block.append(Paragraph(f"<b>{label}:</b> {escape(printable_source)}", source_style))
        story.append(KeepTogether(block))
        story.append(Spacer(1, 1.6*mm))

    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(destination), pagesize=A4, rightMargin=19*mm, leftMargin=19*mm,
        topMargin=18*mm, bottomMargin=17*mm, title=title,
        author="Ethan Guevara; Mateo Gavilanes; Kennedy Rodriguez",
    )

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("ArialUnicode", 8)
        canvas.setFillColor(colors.HexColor("#687984"))
        canvas.drawString(19*mm, 9*mm, "Guion de defensa EMG" if language == "es" else "EMG defense script")
        canvas.drawRightString(A4[0]-19*mm, 9*mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spanish", required=True, type=Path)
    parser.add_argument("--english", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    render_one(args.spanish, args.output_dir / "Guion_defensa_ES.pdf", "es")
    render_one(args.english, args.output_dir / "Defense_Script_EN.pdf", "en")
    print("Spanish and English speaking-script PDFs written")


if __name__ == "__main__":
    main()
