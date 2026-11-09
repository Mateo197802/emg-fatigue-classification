"""Run the private participant-held-out perceived-fatigue benchmark."""

import argparse
from pathlib import Path

from emg_fatigue.perceived_benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-cap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    results, manifest = run_benchmark(
        args.output_dir / "features.parquet",
        args.output_dir,
        sample_cap=args.sample_cap,
        seed=args.seed,
    )
    print(results[[
        "architecture",
        "scope",
        "accuracy",
        "f1_macro",
        "balanced_accuracy",
        "sensitivity_fatigue",
        "specificity_no_fatigue",
    ]].to_string(index=False))
    print(f"completed_models={manifest['models_completed']}")
    print(f"manifest={args.output_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
