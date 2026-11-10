"""Create aggregate paired-Accuracy contrasts from frozen grouped result tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from emg_fatigue.evaluation import paired_metric_contrast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-results", required=True, type=Path)
    parser.add_argument("--phase-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()

    primary = pd.read_csv(args.primary_results)
    phase = pd.read_csv(args.phase_results)
    rows = [
        paired_metric_contrast(
            primary,
            candidate="nested_selector",
            baseline="logistic_rms_mdf",
            metric="accuracy",
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
        paired_metric_contrast(
            phase,
            candidate="phase_conditioned",
            baseline="baseline_10",
            metric="accuracy",
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.12g", lineterminator="\n")
    receipt_path = args.output.parent / "evidence_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        hashes = receipt.setdefault("aggregate_hashes", {})
        hashes[args.output.name] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} aggregate contrasts to {args.output}")


if __name__ == "__main__":
    main()
