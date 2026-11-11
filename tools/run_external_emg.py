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
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild only the external private window bank")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    from emg_fatigue.weak_supervision.external_supervision import audit_cerqueira_labels, build_cerqueira_windows, run_cerqueira_grouped_stage
    from emg_fatigue.weak_supervision.external_classical import run_cerqueira_classical_stage
    from emg_fatigue.weak_supervision.external_neural import run_cerqueira_cnn_stage

    run_dir = args.run_dir.resolve()
    dataset_dir = run_dir / "private" / "external_supervision" / "cerqueira_v2"
    output_dir = dataset_dir / "training"
    if args.audit_cerqueira_labels:
        result = audit_cerqueira_labels(label_root=dataset_dir / "u", output_dir=dataset_dir / "label_audit")
    elif args.build_cerqueira_windows:
        result = build_cerqueira_windows(
            signal_archive=dataset_dir / "originals" / "sEMG_data.zip",
            label_root=dataset_dir / "u",
            output_dir=dataset_dir / "derived_windows",
            force=bool(args.force_rebuild),
        )
    elif args.train_cerqueira:
        result = run_cerqueira_grouped_stage(
            feature_path=dataset_dir / "derived_windows" / "cerqueira_v2_window_features.parquet",
            output_dir=output_dir,
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
    elif args.train_cerqueira_classical:
        result = run_cerqueira_classical_stage(
            feature_path=dataset_dir / "derived_windows" / "cerqueira_v2_window_features.parquet",
            baseline_predictions_path=output_dir / "cerqueira_nested_oof_predictions.parquet",
            output_dir=dataset_dir / "training_classical_v1",
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
    else:
        result = run_cerqueira_cnn_stage(
            feature_path=dataset_dir / "derived_windows" / "cerqueira_v2_window_features.parquet",
            signal_path=dataset_dir / "derived_windows" / "cerqueira_v2_signal_windows.npy",
            baseline_predictions_path=output_dir / "cerqueira_nested_oof_predictions.parquet",
            output_dir=dataset_dir / "training_cnn_v1",
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
