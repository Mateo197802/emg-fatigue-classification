"""CLI entry point for the fresh EMG fatigue-proxy experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_fatigue.weak_supervision.runner import prepare_training_run


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the raw-first EMG fatigue proxy experiment.")
    parser.add_argument("--prepare-training-zip", action="store_true", help="Inventory ZIP and parse development groups only.")
    parser.add_argument("--training-zip", type=Path, help="Original training ZIP; RAR is never accepted here.")
    parser.add_argument("--run-dir", type=Path, help="Private run directory outside the repository.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "configs" / "fatigue_proxy.toml",
        help="Frozen experiment configuration TOML.",
    )
    args = parser.parse_args()
    if not args.prepare_training_zip:
        parser.error("select --prepare-training-zip for the first, development-only stage")
    if args.training_zip is None or args.run_dir is None:
        parser.error("--training-zip and --run-dir are required with --prepare-training-zip")

    result = prepare_training_run(args.training_zip, args.run_dir, args.config)
    print(
        "Prepared raw inventory: "
        f"{result['csv_count']} CSVs across {result['source_group_count']} source groups; "
        f"{result['development_group_count']} development groups and "
        f"{result['internal_test_group_count']} locked test groups. "
        f"Parsed {result['development_record_count']} development recordings; "
        f"left {result['unparsed_internal_test_member_count']} test signal payloads numerically unparsed."
    )
    print(f"Private run receipt: {result['run_manifest_path']}")


if __name__ == "__main__":
    main()
