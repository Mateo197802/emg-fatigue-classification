# Frozen primary results

The original results below remain unchanged. The [phase supplement](PHASE_EXPERIMENT.md) adds five exploratory variants on identical held-out support, auxiliary burst-half evaluation, architecture, absolute/row-normalized confusions and target-specific evidence. The matched direct baseline is `logistic_all` (0.635485), not `nested_selector` (0.624004). The phase-conditioned result (0.629263) does not establish a gain.

Target: weak_elapsed_thirds_binary_v1. These scores measure early/late proxy classification, not independently validated physiological fatigue. Fifteen source-folder recording groups are held out; person identity remains unverified.

Source: 140 files / 4,256,257 samples; formal: 138; labeled windows: 5,509 (early 2,329, late 3,180). Both sides stay in the same group.

| Procedure | Macro F1 mean ± SD | ROC AUC mean ± SD |
|---|---|---|
| logistic_all | 0.635485 ± 0.126179 | 0.844745 ± 0.074570 |
| logistic_rms_mdf | 0.627162 ± 0.097987 | 0.742990 ± 0.106871 |
| majority | 0.359768 ± 0.034861 | 0.500000 ± 0.000000 |
| nested_selector | 0.624004 ± 0.129191 | 0.798446 ± 0.113863 |
| random_forest | 0.636556 ± 0.125872 | 0.817036 ± 0.076901 |
| svm_rbf | 0.636259 ± 0.134276 | 0.803915 ± 0.110651 |

The nested selector is the prespecified primary selection-procedure estimate. Its observed mean does not clearly improve on the mandatory RMS+MDF logistic baseline. Family point estimates are not evidence of a unique winner.

- nested_selector_minus_majority: +0.264236; paired group-bootstrap 95% interval [+0.194035, +0.332007].
- nested_selector_minus_logistic_rms_mdf: -0.003158; paired group-bootstrap 95% interval [-0.059111, +0.050850].

Three complete runs were preserved. Final versus first: all 33,054 OOF predicted classes identical; maximum score difference 4.44e-16.

Nine prespecified sensitivities and three family ablations are completed privately. The fixed-reference sensitivity macro-F1 values are not an alternative winner search. The additional 20-file RAR is processed/inferred separately without an external-validation claim.

Public fold results are in results/results.csv; aggregate means/SD/intervals are in results/metrics_summary.csv. Full OOF, source mappings, model and report remain private. See the protocol and privacy policy.
