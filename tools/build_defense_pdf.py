"""Package rendered 16:9 defense slides into a page-per-slide PDF."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


SLIDE_PATTERN = re.compile(r"^slide-(\d+)\.png$")
PAGE_WIDTH = 960.0
PAGE_HEIGHT = 540.0


def build_pdf(slides_dir: Path, output: Path, expected_slides: int = 33) -> int:
    numbered: dict[int, Path] = {}
    for path in slides_dir.glob("slide-*.png"):
        match = SLIDE_PATTERN.match(path.name)
        if match:
            number = int(match.group(1))
            if number in numbered:
                raise ValueError(f"duplicate slide number: {number}")
            numbered[number] = path

    expected = set(range(1, expected_slides + 1))
    found = set(numbered)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(f"slide set mismatch; missing={missing}, extra={extra}")

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1)
    pdf.setTitle("EMG Binary Defense — 9 November 2026")
    pdf.setAuthor("Ethan Guevara; Mateo Gavilanes; Kennedy Rodriguez")
    for number in range(1, expected_slides + 1):
        image = ImageReader(str(numbered[number]))
        pdf.drawImage(image, 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT, preserveAspectRatio=True, anchor="c")
        pdf.showPage()
    pdf.save()
    return expected_slides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-slides", type=int, default=33)
    args = parser.parse_args()
    print({"output": str(args.output), "slides": build_pdf(args.slides_dir, args.output, args.expected_slides)})


if __name__ == "__main__":
    main()
