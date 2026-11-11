"""Entry point for the new raw-first, label-free sEMG experiment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emg_fatigue.raw_first.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
