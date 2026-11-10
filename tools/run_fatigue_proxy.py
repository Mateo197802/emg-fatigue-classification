"""CLI entry point for the fresh EMG fatigue-proxy experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_fatigue.weak_supervision.acquisition_stage import run_acquisition_stage
from emg_fatigue.weak_supervision.evaluation import run_supervised_stage
from emg_fatigue.weak_supervision.label_audit import run_label_audit
from emg_fatigue.weak_supervision.runner import prepare_training_run


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the raw-first EMG fatigue proxy experiment.")
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--prepare-training-zip", action="store_true", help="Inventory ZIP and parse development groups only.")
    stage.add_argument("--run-acquisition-stage", action="store_true", help="Run development-only acquisition QC and window audits.")
    stage.add_argument("--run-label-stage", action="store_true", help="Extract development-only features and audit proxy labels.")
    stage.add_argument("--run-supervised-stage", action="store_true", help="Train and evaluate grouped proxy classifiers from existing private development tables.")
    parser.add_argument("--training-zip", type=Path, help="Original training ZIP; RAR is never accepted here.")
    parser.add_argument("--run-dir", type=Path, help="Private run directory outside the repository.")
    parser.add_argument(
        "--label-rule",
        choices=("spectral_persistence_v1", "log_mdf_slope_low_fraction_v1", "low_frequency_fraction_persistence_v1"),
        help="Precomputed development-only proxy-label rule for the supervised stage.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "configs" / "fatigue_proxy.toml",
        help="Frozen experiment configuration TOML.",
    )
    args = parser.parse_args()
    if args.run_dir is None:
        parser.error("--run-dir is required")

    if args.run_supervised_stage:
        if args.label_rule is None:
            parser.error("--label-rule is required with --run-supervised-stage")
        result = run_supervised_stage(args.run_dir, args.config, label_rule=args.label_rule)
        print(
            "Completed supervised development stage: "
            f"status={result['status']}; rule={result['label_rule']}; "
            f"checkpoints={result.get('checkpoint_count', 0)}; "
            f"internal-test signal values parsed={result.get('internal_test_signal_values_parsed', 0)}; "
            f"external archive opened={result.get('external_rar_opened', False)}."
        )
        print(f"Private supervised-stage receipt: {args.run_dir / 'private' / 'supervised_models' / 'supervised_model_stage_manifest.json'}")
        return

    if args.training_zip is None:
        parser.error("--training-zip is required for inventory, acquisition, and label stages")

    if args.run_acquisition_stage:
        result = run_acquisition_stage(args.training_zip, args.run_dir, args.config)
        print(
            "Completed development acquisition stage: "
            f"{result['development_record_count']} recordings; internal-test numeric values parsed=0; "
            f"external archive opened=False."
        )
        print(f"Private acquisition receipt: {result['acquisition_manifest_path']}")
        return

    if args.run_label_stage:
        result = run_label_audit(args.training_zip, args.run_dir, args.config)
        print(
            "Completed development feature-label audit: "
            f"{result['feature_row_count']} feature rows, {result['primary_label_row_count']} primary labels; "
            f"internal-test numeric values parsed=0; external archive opened=False."
        )
        print(f"Private feature-label receipt: {result['feature_label_manifest_path']}")
        return

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
