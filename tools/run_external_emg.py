from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and train participant-grouped external EMG models.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-cerqueira-labels", action="store_true")
    mode.add_argument("--build-cerqueira-windows", action="store_true")
    mode.add_argument("--train-cerqueira", action="store_true")
    mode.add_argument("--train-cerqueira-classical", action="store_true")
    mode.add_argument("--train-cerqueira-cnn", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing private experiment run root")
    parser.add_argument("--dataset-dir", type=Path, help="Read-only dataset source root; defaults inside --run-dir")
    parser.add_argument("--audit-output-dir", type=Path, help="Separate output directory for a fresh label audit")
    parser.add_argument("--window-output-dir", type=Path, help="Separate output directory for a fresh synchronized window bank")
    parser.add_argument("--stage-output-dir", type=Path, help="Separate output directory for the selected training stage")
    parser.add_argument("--baseline-predictions-path", type=Path, help="Baseline OOF file for the classical or CNN comparison")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild only the external private window bank")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    from emg_fatigue.weak_supervision.external_supervision import audit_cerqueira_labels, build_cerqueira_windows, run_cerqueira_grouped_stage
    from emg_fatigue.weak_supervision.external_classical import run_cerqueira_classical_stage
    from emg_fatigue.weak_supervision.external_neural import run_cerqueira_cnn_stage

    run_dir = args.run_dir.resolve()
    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else run_dir / "private" / "external_supervision" / "cerqueira_v2"
    window_output_dir = args.window_output_dir.resolve() if args.window_output_dir else dataset_dir / "derived_windows"
    if args.stage_output_dir:
        output_dir = args.stage_output_dir.resolve()
    elif args.train_cerqueira_classical:
        output_dir = dataset_dir / "training_classical_v1"
    elif args.train_cerqueira_cnn:
        output_dir = dataset_dir / "training_cnn_v1"
    else:
        output_dir = dataset_dir / "training"
    if args.audit_cerqueira_labels:
        audit_output_dir = args.audit_output_dir.resolve() if args.audit_output_dir else dataset_dir / "label_audit"
        result = audit_cerqueira_labels(label_root=dataset_dir / "u", output_dir=audit_output_dir)
    elif args.build_cerqueira_windows:
        result = build_cerqueira_windows(
            signal_archive=dataset_dir / "originals" / "sEMG_data.zip",
            label_root=dataset_dir / "u",
            output_dir=window_output_dir,
            force=bool(args.force_rebuild),
        )
    elif args.train_cerqueira:
        result = run_cerqueira_grouped_stage(
            feature_path=window_output_dir / "cerqueira_v2_window_features.parquet",
            output_dir=output_dir,
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
    elif args.train_cerqueira_classical:
        result = run_cerqueira_classical_stage(
            feature_path=window_output_dir / "cerqueira_v2_window_features.parquet",
            baseline_predictions_path=(args.baseline_predictions_path.resolve() if args.baseline_predictions_path else dataset_dir / "training" / "cerqueira_nested_oof_predictions.parquet"),
            output_dir=output_dir,
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
    else:
        result = run_cerqueira_cnn_stage(
            feature_path=window_output_dir / "cerqueira_v2_window_features.parquet",
            signal_path=window_output_dir / "cerqueira_v2_signal_windows.npy",
            baseline_predictions_path=(args.baseline_predictions_path.resolve() if args.baseline_predictions_path else dataset_dir / "training" / "cerqueira_nested_oof_predictions.parquet"),
            output_dir=output_dir,
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
