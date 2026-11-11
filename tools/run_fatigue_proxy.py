"""CLI entry point for the fresh EMG fatigue-proxy experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_fatigue.weak_supervision.acquisition_stage import run_acquisition_stage
from emg_fatigue.weak_supervision.evaluation import (
    run_a1_optimization_stage,
    run_nested_threshold_stage,
    run_supervised_stage,
)
from emg_fatigue.weak_supervision.label_audit import run_label_audit
from emg_fatigue.weak_supervision.runner import prepare_training_run
from emg_fatigue.weak_supervision.signal_windows import build_development_signal_window_bank


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the raw-first EMG fatigue proxy experiment.")
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--prepare-training-zip", action="store_true", help="Inventory ZIP and parse development groups only.")
    stage.add_argument("--run-acquisition-stage", action="store_true", help="Run development-only acquisition QC and window audits.")
    stage.add_argument("--run-label-stage", action="store_true", help="Extract development-only features and audit proxy labels.")
    stage.add_argument("--run-supervised-stage", action="store_true", help="Train and evaluate grouped proxy classifiers from existing private development tables.")
    stage.add_argument("--run-nested-threshold-stage", action="store_true", help="Run development-only 4x3 nested selection of calibration and group-balanced F1 thresholds for the primary low-frequency proxy labels.")
    stage.add_argument("--run-a1-optimization-stage", action="store_true", help="Run inner-only 4x3 A1 logistic regularization, class-weight, scaler, feature-selection, calibration, and threshold search on low-frequency labels.")
    stage.add_argument("--run-development-error-audit", action="store_true", help="Write private development-only group, trial, temporal-label, acquisition, feature-redundancy, and OOF error diagnostics.")
    stage.add_argument("--run-neural-supervised-stage", action="store_true", help="Train nested low-frequency-proxy CNN, SSL encoder, and latent MoE on development groups only.")
    stage.add_argument("--run-neural-extension-stage", action="store_true", help="Train the pending frozen-SSL-head and two-expert latent-MoE variants on the existing nested development folds.")
    stage.add_argument("--run-external-transfer-stage", action="store_true", help="Fine-tune the externally supervised Cerqueira encoder on the primary local low-frequency proxy using development groups only.")
    stage.add_argument("--freeze-development-model", action="store_true", help="Refit the selected A1 low-frequency proxy candidate on development groups and freeze calibration, threshold, OOD and abstention policy.")
    stage.add_argument("--freeze-temporal-baseline", action="store_true", help="Fit and freeze the temporal-only development control before opening internal test signals.")
    stage.add_argument("--run-internal-evaluation", action="store_true", help="Evaluate frozen models on only the 30 reserved internal ZIP members and run matched noise/OOD/abstention analysis.")
    stage.add_argument("--run-external-evaluation", action="store_true", help="Apply already frozen models to the external P11/P12 RAR; output predictions only when independent labels are absent.")
    stage.add_argument("--run-neural-guardrails-stage", action="store_true", help="Evaluate development-only OOD scores, inner-selected abstention, risk-coverage, and AURC.")
    stage.add_argument("--run-a1-guardrails-stage", action="store_true", help="Apply fold-matched development OOD scores and inner-selected abstention to the A1 morphology candidate.")
    stage.add_argument("--run-augmented-neural-stage", action="store_true", help="Train clean-plus-rule-preserving-noise augmented neural models in the frozen nested development folds.")
    stage.add_argument("--run-noise-robustness-stage", action="store_true", help="Evaluate matched development-only noise robustness for frozen clean and augmented models.")
    stage.add_argument("--run-a1-noise-robustness-stage", action="store_true", help="Evaluate selected nested A1 morphology checkpoints on matched noisy signals with fold-specific feature recomputation.")
    stage.add_argument("--prepare-neural-data", action="store_true", help="Materialize processed 500 ms signal windows from development ZIP members only.")
    parser.add_argument("--training-zip", type=Path, help="Original training ZIP; RAR is never accepted here.")
    parser.add_argument("--external-archive", type=Path, help="The checksum-verified P11/P12 RAR, evaluated only after internal holdout completion.")
    parser.add_argument("--run-dir", type=Path, help="Private run directory outside the repository.")
    parser.add_argument(
        "--label-rule",
        choices=("spectral_persistence_v1", "log_mdf_slope_low_fraction_v1", "low_frequency_fraction_persistence_v1"),
        help="Precomputed development-only proxy-label rule for the supervised stage.",
    )
    parser.add_argument(
        "--coverage-aware-folds",
        action="store_true",
        help="Use nested folds balanced on distinct source groups and recordings per class.",
    )
    parser.add_argument(
        "--nonoverlap-only",
        action="store_true",
        help="Fit and score only even-index 500 ms windows (500 ms start-to-start hop) to remove direct window overlap.",
    )
    parser.add_argument(
        "--output-subdir",
        default="supervised_models",
        help="Private output directory name under the run's private directory.",
    )
    parser.add_argument(
        "--threshold-output-subdir",
        default="lowfreq_threshold_v2",
        help="Private output directory for the corrected inner-OOF threshold-selection stage.",
    )
    parser.add_argument(
        "--neural-output-subdir",
        default="neural_lowfreq_nested_4x3_v1",
        help="Private output directory for nested raw-signal model evaluation.",
    )
    parser.add_argument(
        "--a1-optimization-output-subdir",
        default="a1_lowfreq_optimization_v1",
        help="Private output directory for deeper nested A1 optimization.",
    )
    parser.add_argument(
        "--neural-extension-output-subdir",
        default="neural_lowfreq_extensions_4x3_v1",
        help="Private output directory for frozen-SSL and two-expert MoE development runs.",
    )
    parser.add_argument(
        "--external-transfer-output-subdir",
        default="neural_lowfreq_cerqueira_transfer_4x3_v1",
        help="Private output directory for nested development-only fine-tuning of the Cerqueira encoder.",
    )
    parser.add_argument(
        "--external-encoder-checkpoint",
        type=Path,
        help="Completed Cerqueira development checkpoint used to initialize supervised transfer.",
    )
    parser.add_argument(
        "--freeze-output-subdir",
        default="frozen_development_lowfreq_v1",
        help="Private destination for the selected model checkpoint and frozen development-only policies.",
    )
    parser.add_argument("--temporal-freeze-output-subdir", default="frozen_development_temporal_baseline_v4", help="Private destination for the frozen development-only time baseline.")
    parser.add_argument("--internal-output-subdir", default="internal_lowfreq_frozen_evaluation_v3", help="Private destination for frozen internal-test results.")
    parser.add_argument("--external-output-subdir", default="external_raw_rar_frozen_predictions_v1", help="Private destination for external frozen predictions.")
    parser.add_argument("--unrar-exe", type=Path, default=Path(r"C:\Program Files\WinRAR\UnRAR.exe"), help="WinRAR UnRAR.exe used to list and stream members without writing extracted CSVs.")
    parser.add_argument(
        "--development-audit-output-subdir",
        default="development_error_audit_v1",
        help="Private output directory for development-only error diagnostics.",
    )
    parser.add_argument("--neural-tabular-subdir", default="lowfreq_threshold_v2", help="Completed corrected low-frequency 4x3 stage providing frozen folds and temporal OOF baseline.")
    parser.add_argument("--guardrail-neural-subdir", default="neural_lowfreq_nested_4x3_v1", help="Completed low-frequency raw-signal neural stage.")
    parser.add_argument("--guardrail-output-subdir", default="neural_lowfreq_guardrails_v2", help="Private OOD/abstention results directory.")
    parser.add_argument("--a1-guardrail-output-subdir", default="a1_lowfreq_guardrails_v1", help="Private A1 OOD/abstention result directory.")
    parser.add_argument("--augmented-neural-output-subdir", default="neural_lowfreq_augmented_nested_4x3_v1", help="Private nested augmented-model output directory.")
    parser.add_argument("--noise-robustness-output-subdir", default="neural_lowfreq_noise_robustness_v1", help="Private matched-noise robustness output directory.")
    parser.add_argument("--a1-noise-output-subdir", default="a1_lowfreq_noise_robustness_v1", help="Private output directory for matched A1 morphology noise robustness.")
    parser.add_argument("--neural-max-epochs", type=int, default=20, help="Maximum supervised epochs selected using inner folds only.")
    parser.add_argument("--ssl-pretrain-epochs", type=int, default=6, help="Masked reconstruction epochs, using training-fold groups only.")
    parser.add_argument("--neural-bootstrap-replicates", type=int, default=2000, help="Paired source-group bootstrap replicates for temporal comparisons.")
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
        result = run_supervised_stage(
            args.run_dir,
            args.config,
            label_rule=args.label_rule,
            output_subdir=args.output_subdir,
            coverage_aware_folds=args.coverage_aware_folds,
            nonoverlap_only=args.nonoverlap_only,
        )
        print(
            "Completed supervised development stage: "
            f"status={result['status']}; rule={result['label_rule']}; "
            f"checkpoints={result.get('checkpoint_count', 0)}; "
            f"internal-test signal values parsed={result.get('internal_test_signal_values_parsed', 0)}; "
            f"external archive opened={result.get('external_rar_opened', False)}."
        )
        print(
            "Private supervised-stage receipt: "
            f"{args.run_dir / 'private' / args.output_subdir / 'supervised_model_stage_manifest.json'}"
        )
        return

    if args.run_nested_threshold_stage:
        result = run_nested_threshold_stage(
            args.run_dir,
            args.config,
            output_subdir=args.threshold_output_subdir,
        )
        print(
            "Completed nested threshold development stage: "
            f"status={result['status']}; rule={result['label_rule']}; "
            f"outer={result['outer_group_folds']}x inner={result['inner_group_folds']}; "
            f"checkpoints={result.get('checkpoint_count', 0)}; "
            f"internal-test values parsed={result.get('internal_test_signal_values_parsed', 0)}; "
            f"external RAR opened={result.get('external_rar_opened', False)}."
        )
        print(f"Private threshold-stage receipt: {result.get('manifest_path', args.run_dir / 'private' / args.threshold_output_subdir / 'supervised_threshold_stage_manifest.json')}")
        return

    if args.run_a1_optimization_stage:
        result = run_a1_optimization_stage(
            args.run_dir,
            args.config,
            parent_subdir=args.neural_tabular_subdir,
            output_subdir=args.a1_optimization_output_subdir,
        )
        print(
            "Completed nested A1 optimization: "
            f"status={result['status']}; candidates={result['candidate_count_per_outer_fold']}; "
            f"checkpoints={result['checkpoint_count']}; reserved test parsed="
            f"{result['internal_test_signal_values_parsed']}; external RAR opened={result['external_rar_opened']}."
        )
        print(f"Private A1 optimization receipt: {result['manifest_path']}")
        return

    if args.run_development_error_audit:
        from emg_fatigue.weak_supervision.development_audit import run_development_error_audit

        result = run_development_error_audit(
            args.run_dir,
            output_subdir=args.development_audit_output_subdir,
            a1_subdir=args.a1_optimization_output_subdir,
        )
        print(
            "Completed development-only error audit: "
            f"groups={len(result['development_groups'])}; windows={result['development_label_rows']}; "
            f"test rows materialized={result['reserved_test_feature_rows_materialized']}; "
            f"external RAR opened={result['external_rar_opened']}."
        )
        print(f"Private development audit receipt: {result['manifest_path']}")
        return

    if args.run_neural_supervised_stage:
        from emg_fatigue.weak_supervision.evaluation_neural_nested import run_nested_neural_proxy_stage

        result = run_nested_neural_proxy_stage(
            args.run_dir,
            tabular_subdir=args.neural_tabular_subdir,
            output_subdir=args.neural_output_subdir,
            max_epochs=args.neural_max_epochs,
            ssl_pretrain_epochs=args.ssl_pretrain_epochs,
            bootstrap_replicates=args.neural_bootstrap_replicates,
        )
        print(
            "Completed development-only nested neural evaluation: "
            f"status={result['status']}; models={len(result['models'])}; "
            f"checkpoints={result['checkpoint_count']}; assigned classes="
            f"{result['class_0_assigned_windows_descriptive_only']}/{result['class_1_assigned_windows_descriptive_only']}; "
            f"internal-test values parsed={result['internal_test_signal_values_parsed']}; "
            f"external RAR opened={result['external_rar_opened']}."
        )
        print(f"Private neural-stage receipt: {result['manifest_path']}")
        return

    if args.run_neural_extension_stage:
        from emg_fatigue.weak_supervision.evaluation_neural import EXTENDED_MODEL_NAMES
        from emg_fatigue.weak_supervision.evaluation_neural_nested import run_nested_neural_proxy_stage

        result = run_nested_neural_proxy_stage(
            args.run_dir,
            tabular_subdir=args.neural_tabular_subdir,
            output_subdir=args.neural_extension_output_subdir,
            max_epochs=args.neural_max_epochs,
            ssl_pretrain_epochs=args.ssl_pretrain_epochs,
            bootstrap_replicates=args.neural_bootstrap_replicates,
            model_names=EXTENDED_MODEL_NAMES,
        )
        print(
            "Completed nested neural extension: "
            f"status={result['status']}; models={result['models']}; "
            f"checkpoints={result['checkpoint_count']}; reserved test parsed="
            f"{result['internal_test_signal_values_parsed']}; external RAR opened={result['external_rar_opened']}."
        )
        print(f"Private neural extension receipt: {result['manifest_path']}")
        return

    if args.run_external_transfer_stage:
        if args.external_encoder_checkpoint is None:
            parser.error("--external-encoder-checkpoint is required with --run-external-transfer-stage")
        from emg_fatigue.weak_supervision.evaluation_neural_nested import run_nested_neural_proxy_stage

        result = run_nested_neural_proxy_stage(
            args.run_dir,
            tabular_subdir=args.neural_tabular_subdir,
            output_subdir=args.external_transfer_output_subdir,
            max_epochs=args.neural_max_epochs,
            ssl_pretrain_epochs=args.ssl_pretrain_epochs,
            bootstrap_replicates=args.neural_bootstrap_replicates,
            model_names=("cerqueira_supervised_finetuned",),
            external_encoder_path=args.external_encoder_checkpoint,
        )
        print(
            "Completed development-only Cerqueira-to-low-frequency transfer: "
            f"status={result['status']}; checkpoints={result['checkpoint_count']}; "
            f"source sha256={result['external_encoder_checkpoint_sha256']}; "
            f"internal-test values parsed={result['internal_test_signal_values_parsed']}; "
            f"external RAR opened={result['external_rar_opened']}."
        )
        print(f"Private transfer-stage receipt: {result['manifest_path']}")
        return

    if args.freeze_development_model:
        from emg_fatigue.weak_supervision.development_freeze import freeze_development_model

        result = freeze_development_model(
            args.run_dir,
            output_subdir=args.freeze_output_subdir,
            bootstrap_replicates=args.neural_bootstrap_replicates,
        )
        print(
            "Frozen development-only low-frequency proxy model: "
            f"status={result['status']}; model={result['selected_model']}; "
            f"threshold={result['decision_threshold']:.6f}; "
            f"internal-test values read={result['locked_internal_test_values_read']}; "
            f"raw(2).rar opened={result['raw2_rar_opened']}."
        )
        print(f"Private freeze receipt: {result['manifest_path']}")
        return

    if args.freeze_temporal_baseline:
        from emg_fatigue.weak_supervision.evaluation_internal import freeze_temporal_baseline

        result = freeze_temporal_baseline(
            args.run_dir,
            frozen_subdir=args.freeze_output_subdir,
            a1_subdir=args.a1_optimization_output_subdir,
            output_subdir=args.temporal_freeze_output_subdir,
        )
        print(
            "Frozen development-only temporal baseline: "
            f"status={result['status']}; threshold={result['selected_threshold']:.6f}; "
            f"internal-test values read={result['locked_internal_test_values_read']}; "
            f"raw(2).rar opened={result['raw2_rar_opened']}."
        )
        print(f"Private temporal-baseline receipt: {result['manifest_path']}")
        return

    if args.run_internal_evaluation:
        if args.training_zip is None:
            parser.error("--training-zip is required with --run-internal-evaluation")
        from emg_fatigue.weak_supervision.evaluation_internal import run_internal_evaluation

        result = run_internal_evaluation(
            args.training_zip,
            args.run_dir,
            args.config,
            frozen_subdir=args.freeze_output_subdir,
            temporal_subdir=args.temporal_freeze_output_subdir,
            output_subdir=args.internal_output_subdir,
            bootstrap_replicates=args.neural_bootstrap_replicates,
        )
        print(
            "Completed frozen internal evaluation: "
            f"status={result['status']}; reserved groups={result['internal_test_group_ids']}; "
            f"members parsed={result['internal_test_member_count']}; "
            f"assigned windows descriptive only={result['assigned_internal_window_count_descriptive_only']}; "
            f"raw(2).rar opened={result['raw2_rar_opened']}."
        )
        print(f"Private internal-evaluation receipt: {result['manifest_path']}")
        return

    if args.run_external_evaluation:
        if args.external_archive is None:
            parser.error("--external-archive is required with --run-external-evaluation")
        from emg_fatigue.weak_supervision.evaluation_external_unlabeled import run_external_unlabeled_evaluation

        result = run_external_unlabeled_evaluation(
            args.external_archive,
            args.run_dir,
            args.config,
            unrar_exe=args.unrar_exe,
            frozen_subdir=args.freeze_output_subdir,
            temporal_subdir=args.temporal_freeze_output_subdir,
            internal_subdir=args.internal_output_subdir,
            output_subdir=args.external_output_subdir,
        )
        print(
            "Completed external frozen prediction-only evaluation: "
            f"status={result['status']}; members={result['external_member_count']}; "
            f"eligible windows descriptive only={result['external_eligible_window_count_descriptive_only']}; "
            f"independent labels={result['independent_target_labels_available']}; "
            f"classification metrics computed={result['classification_metrics_computed']}."
        )
        print(f"Private external-evaluation receipt: {result['manifest_path']}")
        return

    if args.run_neural_guardrails_stage:
        from emg_fatigue.weak_supervision.evaluation_guardrails import run_guardrail_stage

        result = run_guardrail_stage(
            args.run_dir,
            neural_subdir=args.guardrail_neural_subdir,
            threshold_subdir=args.neural_tabular_subdir,
            output_subdir=args.guardrail_output_subdir,
            bootstrap_replicates=args.neural_bootstrap_replicates,
        )
        print(
            "Completed development-only OOD/abstention evaluation: "
            f"status={result['status']}; OOD thresholds={result['ood_quantile']}; "
            f"models={len(result['models'])}; internal-test values parsed="
            f"{result['internal_test_signal_values_parsed']}; external RAR opened={result['external_rar_opened']}."
        )
        print(f"Private guardrail-stage receipt: {result['manifest_path']}")
        return

    if args.run_a1_guardrails_stage:
        from emg_fatigue.weak_supervision.evaluation_a1_guardrails import run_a1_guardrail_stage

        result = run_a1_guardrail_stage(
            args.run_dir,
            a1_subdir=args.a1_optimization_output_subdir,
            guardrail_subdir=args.guardrail_output_subdir,
            output_subdir=args.a1_guardrail_output_subdir,
            bootstrap_replicates=args.neural_bootstrap_replicates,
        )
        print(
            "Completed A1 OOD/abstention evaluation: "
            f"status={result['status']}; thresholds={len(result['ood_detectors'])}; "
            f"internal-test values parsed={result['internal_test_signal_values_parsed']}; "
            f"external RAR opened={result['external_rar_opened']}."
        )
        print(f"Private A1 guardrail-stage receipt: {result['manifest_path']}")
        return

    if args.run_augmented_neural_stage:
        from emg_fatigue.weak_supervision.evaluation_noise_robustness import run_augmented_nested_training_stage

        result = run_augmented_nested_training_stage(
            args.run_dir,
            tabular_subdir=args.neural_tabular_subdir,
            neural_subdir=args.neural_output_subdir,
            guardrail_subdir=args.guardrail_output_subdir,
            output_subdir=args.augmented_neural_output_subdir,
            max_epochs=args.neural_max_epochs,
            ssl_pretrain_epochs=args.ssl_pretrain_epochs,
            bootstrap_replicates=args.neural_bootstrap_replicates,
        )
        print(
            "Completed development-only augmented neural evaluation: "
            f"status={result['status']}; models={len(result['models'])}; "
            f"checkpoints={result['checkpoint_count']}; manifest={result['manifest_path']}"
        )
        return

    if args.run_noise_robustness_stage:
        from emg_fatigue.weak_supervision.evaluation_noise_robustness import run_noise_robustness_stage

        result = run_noise_robustness_stage(
            args.run_dir,
            tabular_subdir=args.neural_tabular_subdir,
            neural_subdir=args.neural_output_subdir,
            guardrail_subdir=args.guardrail_output_subdir,
            augmented_subdir=args.augmented_neural_output_subdir,
            output_subdir=args.noise_robustness_output_subdir,
            bootstrap_replicates=args.neural_bootstrap_replicates,
        )
        print(
            "Completed matched development noise robustness: "
            f"status={result['status']}; models={len(result['models'])}; "
            f"noise_conditions={result['noise_conditions']}; manifest={result['manifest_path']}"
        )
        return

    if args.run_a1_noise_robustness_stage:
        from emg_fatigue.weak_supervision.evaluation_tabular_noise import run_a1_noise_robustness_stage

        result = run_a1_noise_robustness_stage(
            args.run_dir,
            a1_subdir=args.a1_optimization_output_subdir,
            threshold_subdir=args.neural_tabular_subdir,
            output_subdir=args.a1_noise_output_subdir,
            bootstrap_replicates=args.neural_bootstrap_replicates,
        )
        print(
            "Completed matched A1 noise robustness: "
            f"status={result['status']}; conditions={len(result['conditions'])}; "
            f"reserved test parsed={result['internal_test_signal_values_parsed']}; external RAR opened={result['external_rar_opened']}."
        )
        print(f"Private A1 noise receipt: {result['manifest_path']}")
        return

    if args.prepare_neural_data:
        if args.training_zip is None:
            parser.error("--training-zip is required with --prepare-neural-data")
        result = build_development_signal_window_bank(args.training_zip, args.run_dir / "private")
        print(
            "Completed development-only signal bank: "
            f"{result['development_record_count']} recordings; {result['development_window_count']} windows; "
            f"reserved test payloads read={result['reserved_test_member_payloads_read']}; "
            f"external RAR opened={result['external_rar_opened']}."
        )
        print(f"Private signal bank manifest: {result['manifest_path']}")
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
