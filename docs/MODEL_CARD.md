# Model card

The [supplemental phase experiment](PHASE_EXPERIMENT.md) uses a separate auxiliary 100-tree RF and second-stage logistic models. It showed no clear gain over the matched direct ten-feature logistic baseline. The auxiliary RF predicts EMG burst halves, not verified extension/flexion. It is distinct from the final 300-tree RF described below. The exploratory cascade is not promoted to the primary serialized artifact.

The final private artifact is a complete preprocessing/feature-contract bundle and fitted scikit-learn pipeline. Full-cohort grouped inner selection chose a random forest with 300 trees, maximum depth 3, minimum leaf size 2 and balanced class weights. This is the choice from the frozen selection procedure, not evidence that random forest is universally superior.

**Target:** `weak_elapsed_thirds_binary_v1`. Class 0 denotes eligible first-third windows and class 1 eligible last-third windows. The central third is outside training support. Neither class is independently confirmed physiological fatigue/nonfatigue.

**Inputs:** a complete offline CSV with original increasing seconds and signed stored count values. Compatible fragments are reassembled before filtering. The same frozen 20–400 Hz filter, candidate activity rule, windowing, ten-feature schema and fitted model are used on recarga. No test-record normalization is fitted against other subjects.

**Development evidence:** 138 formal recordings, 15 folder-derived groups and 5,509 labeled windows. Nested group evaluation of the selection procedure obtained macro-F1 0.624004 ± 0.129191 and ROC-AUC 0.798446 ± 0.113863. The mandatory RMS+MDF baseline obtained macro-F1 0.627162 ± 0.097987. The paired difference interval includes zero. Full-data fitting supplies the artifact but contributes no new held-out estimate.

**Outputs:** predicted proxy class and an explicitly uncalibrated native probability score, or a reasoned abstention for rejected intervals, low activity, invalid features or transition. A no-score interval is never forced into class 0. Checksum verification and dependency/schema guards apply on loading; only locally trusted joblib artifacts should be loaded.

**Reproducibility:** three complete evaluations were preserved; final versus first predicted classes agree exactly, with maximum score difference 4.44e-16. A real complete recording and its timestamp-preserving reassembled fragments agree in class, with maximum score difference 2.22e-16. These are technical replay checks, not external fatigue validation.

**Limits:** group identity and physical channel mode/calibration remain unverified; no independent synchronized force, MVC, Borg, mechanical repetition or video reference is supplied. Evaluation covers selected valid active early/late windows, not an entire exercise session. The additional RAR has unknown person overlap and is not an external test. The pipeline uses full-record bounds, zero-phase filtering and per-recording activity thresholds; no online fatigue-onset capability is claimed.

The artifact, receipt, source hashes and granular predictions remain in the private deliverable, outside this public repository.
