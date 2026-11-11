# Nested Threshold Optimization and Continuation Design

## Intent

Continue the existing EMG proxy experiment by correcting its model-selection procedure, then proceed to the remaining architectures, OOD/abstention, and noise robustness. Optimize the prediction of the operational proxy without treating it as independent evidence of physiological fatigue.

## Active label contract

- The primary target for this experimental line is `low_frequency_fraction_persistence_v1`, preserving the previous supervised benchmark: 5,977 assigned windows (5,074 class 0; 903 class 1) across all 12 development source groups.
- Keep `spectral_persistence_v1` as a label-definition sensitivity analysis only: its 353 assigned overlapping windows have sparse class coverage and are not the primary target for this optimizer.
- Do not alter either rule or its thresholds to improve model scores. Every label and its rule version remains traceable.
- A result measures agreement with an EMG-derived weak proxy, not clinical, perceived, or mechanically verified fatigue.

## Validation and paired comparisons

- Retain the existing 4 outer × 3 inner group folds if the authoritative fold manifest and support receipts verify both classes in every fit and validation role. Report class support in distinct source groups, recording trials, and windows for every fold.
- Reuse the exact historical fold assignments and eligible windows for fixed-0.5 versus optimized-threshold comparisons. Keep paired models on the same source groups, observations, and pseudo-labels.
- If any fold lacks valid class support, do not score it or tune on its outer validation data. First report the failure and investigate a group-preserving alternative using development data only; justify any changed fold count and preserve the historical benchmark.
- Treat source folders as grouping units, not verified participants. Window-level counts are descriptive; uncertainty and model comparisons use paired source-group resampling. Add a non-overlapping-window sensitivity view to reveal effects of the 50% overlap.
- Keep the reserved internal-test groups and every external RAR closed throughout development and selection. No acquisition or feature regeneration unless a demonstrated technical fault requires it.

## Nested model, calibration, and threshold selection

Within each outer-training partition, evaluate candidate model and preprocessing configurations using only inner-fold held-out predictions. Hyperparameters and threshold are jointly selected from these inner OOF results. For each inner split, preprocessing, imputation, scaling, feature selection, class balancing, and calibration fit only on its training groups.

Compare the fixed 0.5 decision rule with an optimized threshold selected to maximize group-balanced Macro-F1, where every source group has equal total confusion-matrix weight. Use calibrated probabilities when a group-aware sigmoid/Platt calibrator has adequate support. Fit calibration from group-held-out predictions generated entirely inside the corresponding training partition; do not calibrate on the validation fold. For each hyperparameter candidate, prefer its calibrated probabilities whenever calibration support passes; use an uncalibrated sibling only when the group-aware calibrator for that candidate is unsupported or fails. Record that fallback explicitly. Do not use isotonic calibration unless the effective independent group support demonstrably meets a prespecified adequacy check; window count alone is not sufficient.

Evaluate threshold candidates only on inner OOF scores. Persist the inner precision-recall curve, per-group and group-balanced precision/recall, selected threshold, calibration method, candidate scores, support, fit warnings, and failures. Apply the chosen model, calibration, and threshold once to the untouched outer-validation fold. Do not use outer-fold metrics or PR points to revise any selection. Compare optimized and fixed-0.5 predictions pairwise on the same outer windows and groups.

## Metrics and uncertainty

Report pooled descriptive and macro-by-group results for Accuracy, Macro-F1, Weighted-F1, class-specific precision/recall/F1, positive precision and sensitivity, negative specificity, balanced accuracy, MCC, Cohen's kappa, PR-AUC/Average Precision, ROC-AUC when defined, Brier score and log loss for calibrated probabilities, calibration error with small-sample caveats, and absolute/normalized confusion matrices. Preserve NA values with reasons for undefined one-class groups.

Compute paired source-group bootstrap intervals for optimized-minus-fixed threshold changes and comparisons to the historical model and relative-time baseline. Record training and inference time for every fitted model family. Do not select on Accuracy alone.

## Follow-on work in this objective

After the corrected A1 optimization has run and its results are audited, continue with the authorized work in sequence: CNN-1D; frozen and fine-tuned fresh SSL encoders; two/three latent specialist heads and soft-gated MoE; deeper A1 feature/scaler/class-balance search; probability calibration; distinct quality, domain/OOD, and uncertainty gates; selective prediction with risk-coverage/AURC; matched clean-versus-augmented noise experiments and robustness evaluation. Use the same development groups and labels when comparisons are paired. Preserve negative results and latent-expert semantics; do not call experts movement phases.

Only after development architecture, model, calibration, OOD, abstention, noise, seeds, and thresholds are frozen may the single internal-test evaluation and subsequent external RAR pass run, under their existing limitations. Never retune on either reserved set.

## Approaches considered

1. Preserve fixed 0.5 and report PR curves only. This would diagnose the current errors but would not complete the requested threshold optimization.
2. **Nested group-held-out optimization with group-aware calibration and group-balanced Macro-F1 threshold selection. Chosen by the user.** It directly addresses low precision, retains the historical outer folds and benchmark observations, and keeps outer evaluation data untouched.
3. Choose thresholds on pooled outer predictions, tune separate thresholds by group, or alter pseudo-label thresholds. These would leak evaluation information, overfit known groups, or change the target to inflate agreement, so they are excluded.

## Acceptance checks for the first implementation stage

- Fresh checks confirm the active 4×3 fold manifest, exact group disjointness, and both-class support in all inner and outer train/validation roles for `low_frequency_fraction_persistence_v1`.
- Tests demonstrate deterministic group-balanced threshold selection, calibrated inner-only fitting, exact once-only OOF assignment, failure/NA handling for unsupported folds, and invariance to within-group window replication in uncertainty summaries.
- The optimizer emits new private checkpoints, inner selection/PR receipts, outer OOF probabilities/predictions for both 0.5 and optimized thresholds, comprehensive metrics, paired group-bootstrap changes, and SHA-256 manifests.
- Comparisons use identical fold memberships, windows, and labels. The 353-window spectral label run remains a separate sensitivity analysis.
- Final verification confirms that no reserved test signal values or external RAR payloads were read and that no prohibited report, slide, PDF, or ZIP was generated.
