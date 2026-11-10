"""Reproduce the public Figshare RF task-outcome benchmark locally."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.reference_fatigue.figshare_task_level import extract_paired_rf_members
from experiments.reference_fatigue.figshare_trial_model import prepare_trial_dataset
from experiments.reference_fatigue.figshare_trial_benchmark import (
    finalize_saved_benchmark,
    run_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="range-extract paired RF MAT files from Figshare")
    parser.add_argument("--prepare", action="store_true", help="filter signals and create trial-level features/bags")
    parser.add_argument("--benchmark", action="store_true", help="run nested participant-held-out models")
    parser.add_argument("--finalize", action="store_true", help="aggregate saved OOF results without retraining")
    parser.add_argument("--verify", action="store_true", help="replay metrics and check split/checkpoint hashes")
    parser.add_argument("--all", action="store_true", help="run extraction, preparation, and benchmark")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "private" / "figshare_task_failure_2021" / "source",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=REPO_ROOT / "private" / "figshare_task_failure_2021" / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "reference_fatigue" / "figshare_task_level",
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run_all = args.all
    selected = run_all or args.extract or args.prepare or args.benchmark or args.finalize or args.verify
    if not selected:
        parser.error("select --extract, --prepare, --benchmark, --finalize, --verify, or --all")

    if run_all or args.extract:
        receipt = extract_paired_rf_members(args.source_dir, workers=args.workers)
        print(
            f"source pairs={receipt['complete_pair_count']}; members={receipt['extracted_member_count']}; "
            f"member CRC verified={receipt['member_crc32_verified']}; bytes received={receipt['bytes_received']}"
        )
    if run_all or args.prepare:
        frame = prepare_trial_dataset(args.source_dir, args.processed_dir)
        print(f"prepared trials={len(frame)}; participants={frame['subject_id'].nunique()}")
    if run_all or args.benchmark:
        results, manifest = run_benchmark(args.processed_dir, args.output_dir)
        print(results[["model", "n_trials", "accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]].to_string(index=False))
        print(f"status={manifest['status']}; outputs={args.output_dir}")
    elif args.finalize:
        results, manifest = finalize_saved_benchmark(args.processed_dir, args.output_dir)
        print(results[["model", "n_trials", "accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]].to_string(index=False))
        print(f"status={manifest['status']}; outputs={args.output_dir}")
    if run_all or args.verify or args.benchmark:
        from verify_figshare_trial_benchmark import verify

        receipt = verify(args.output_dir, args.processed_dir)
        print(f"verification={receipt['status']}; models={receipt['model_count']}; checkpoints={receipt['checkpoint_count_checked']}")
        if receipt["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
