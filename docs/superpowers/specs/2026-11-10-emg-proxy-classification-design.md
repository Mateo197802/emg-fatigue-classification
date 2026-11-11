# EMG Fatigue Proxy Classification Design

## Intent

Build and evaluate a supervised **`EMG_FATIGUE_PROXY_CLASSIFICATION`** workflow from the original sEMG recordings. The target is an operational EMG pattern compatible with fatigue, not clinical, perceived, or mechanical fatigue ground truth. The supplied files contain no Borg scores, MVC or torque references, synchronized video, or verified movement phases. The recorded muscle is described by the task as rectus femoris, but the CSV schema does not independently verify electrode location.

The user explicitly requested immediate execution. This design therefore records conservative implementation choices and proceeds without an approval pause. The old repository protocol and code use elapsed-time thirds; they conflict with this objective and are retired for this run. No earlier labels, result tables, feature caches, checkpoints, or fitted models will be used.

## Current Evidence and Constraints

- The supplied training ZIP has 140 CSV files in 15 source folders and no non-CSV members. Each CSV has `Relative_Time_s`, `ADC_Value`, and `Voltage_mV`; the development-only parse found no malformed numeric rows.
- A fresh deterministic split assigns 12 source folders to development and 3 to internal test. The test CSV signal values remain unparsed until the labeling method and candidate models are frozen. Folder identity is not verified participant identity.
- All 110 development recordings were sampled at 1000 Hz. Their duration ranged from 16.0 to 70.2 seconds. A raw PSD feasibility audit found a median 98.5% of power in 20–400 Hz and a median 0.19% in 400 Hz–Nyquist. Median 50/60 Hz line prominence was below 3 dB; strong line interference will be detected and handled conditionally.
- Otsu envelope segmentation produced candidate activity in the development files, but it is not a verified contraction or movement-phase detector. The development-only diagnostic yielded 4,748 high-activity 500 ms windows at an occupancy threshold of 50%. Descriptive spectral shifts were observed within many files, but those diagnostics are not labels, validation metrics, or physiological findings.
- Filename metadata contains series and side tokens (`R0`–`R5`, `dr`/`iz`): 2 files carry `R0`, 68 `dr`, and 72 `iz`. It may be retained privately for provenance and descriptive controls only; it will not create labels or enter primary predictors. No source folder is treated as a person.
- The supplied RAR was opened in an earlier project task. Its earlier results are excluded from this model-development run. Its external data cannot be considered completely unseen at the project level.

## Goals

- Reprocess the original training ZIP from raw timestamps and ADC samples; audit hashes, sampling, discontinuities, artifacts, line noise, amplitude, and spectrum.
- Use 500 ms windows with 50% overlap as the primary sampling unit; calculate 250 ms and 1000 ms sensitivity views.
- Create versioned, auditable weak labels in `{0, 1, -1}` with references, indicators, confidence, ambiguity reasons, and sensitivity analyses.
- Train grouped out-of-sample classifiers, an elapsed-time-only control, and a direct pseudo-label rule comparator.
- Separate the main non-spectral feature view from label-generating spectral indicators; separately evaluate raw-signal CNN and fresh self-supervised encoder models with circularity stated.
- Compare alternative pseudo-label definitions, clean and augmented training, direct and latent-specialist architectures, group-held-out noise performance, coverage, and errors.
- Freeze the full methodology and model artifacts before parsing the internal test signals and the RAR for final evaluation.
- Keep sample-level data, mappings, predictions, and checkpoints in private local storage. Publish only safe code and aggregate metrics.

## Non-goals and Anti-goals

- Do not claim that a pseudo-label is a clinical, mechanical, or perceived-fatigue ground truth.
- Do not label by file thirds, early/late position, or movement phase inferred from halves of a window.
- Do not use previous results, historical feature tables, old pseudo-labels, or old model checkpoints.
- Do not treat the supervised score against EMG-derived labels as independent validation of fatigue.
- Do not call latent experts extension/flexion without synchronized phase labels.
- Do not generate DOCX, PDF, slides, presentations, or a delivery ZIP.

## Approaches Considered

1. **Elapsed-time thirds or filename-series labels:** rejected because they do not establish physiological change and are explicitly prohibited by the user.
2. **One fixed MDF-drop cutoff:** rejected as the sole rule because dynamic EMG changes with contraction phase, velocity, and recruitment; amplitude direction is not stable, and the literature does not establish a universal per-window cutoff.
3. **Persistent multi-indicator change relative to a technical within-recording reference:** selected. It uses spectral indicators supported in repeated quadriceps work, measures local variability, requires temporal persistence, provides hysteresis, and leaves discordant cases ambiguous. Alternative spectral rules and threshold/persistence grids will be reported without selecting a definition by highest Accuracy.

For model views, the selected comparison is: (a) temporal/morphological features that exclude every spectral family used for labels; (b) spectral-only and combined feature diagnostics; and (c) raw-signal CNN and fresh encoder models explicitly described as reproducing a criterion derived from their input signal. Independence of feature calculation is not physiological or statistical independence.

## Raw Signal and Preprocessing Contract

1. Hash the training archive and every CSV; verify CSV schema, parseability, finite samples, timestamp monotonicity, per-recording sampling rate, duplicate content, and safe archive membership.
2. Split source folders before signal-based threshold or model selection. Preserve the 12/3 split and group aliases in the restricted run receipt.
3. Estimate individual sampling rates from positive timestamp intervals. Accept nominal rates above 800 Hz, polyphase-resample to 1000 Hz when the native rate differs, and reject nonmonotonic time or relative p95–p5 dt spread above 5%. Split at gaps greater than `max(1.5 × median dt, 1.5 ms)`; never interpolate across gaps.
4. Mark a run signal-incompatible for labels if prefilter power in 20–400 Hz is below 80%, power below 20 Hz exceeds 20%, or power above 400 Hz exceeds 5%. Record the ratios and run sensitivity at 15/25% low-frequency and 75/90% in-band gates. A window is QC-invalid if it contains a flatline run of at least 100 ms, a clipping-suspect plateau at either extreme lasting at least 5 samples with at least 0.1% of samples at extrema, or an impulsive peak greater than 10 × its local robust scale.
5. Use the development PSD audit to justify an order-4 prototype 20–400 Hz Butterworth bandpass at the observed 1000 Hz rate. Record pre/post spectra and 100 ms edge masks. Failing power/QC gates makes labels ambiguous; do not filter a signal into apparent validity.
6. Apply a 50 or 60 Hz notch only when local spectral prominence exceeds 10 dB; store the selected line and prominence. The threshold is a technical line-interference guard, not a fatigue threshold.
7. Candidate activity uses 50 ms RMS frames and 25 ms hops with per-recording Otsu thresholds and fills only inactive gaps up to 100 ms. If fewer than 5% or more than 95% of valid frames are active, activity is indeterminate. Activity remains a candidate signal state, not a confirmed contraction.
8. Build 250/500/1000 ms windows at 50% overlap. The primary 500 ms window must be wholly within one valid run, pass signal-quality checks, and have at least 50% candidate-activity occupancy. Poor-quality, non-active, or insufficient-reference windows receive `-1`, never class 0. Occupancy sensitivity uses 40%, 60%, and 75%.

## Operational Pseudo-label Rule v1

### Reference

For each continuous recording run, select the first five technically valid, non-overlapping, high-activity 500 ms windows as a **technical reference**. Their median MDF, MNF, and log low/high power ratio define the reference values. This reference is not assumed to be physiologically unfatigued. Fewer than five usable windows or failed acquisition/QC checks leaves the run unlabeled. If the reference scaled MAD divided by the absolute median exceeds 20% for MDF/MNF or 0.20 log units for the low/high ratio, the reference is unstable and the run remains unlabeled; report sensitivity at 15/25/35%. Reference windows themselves remain `-1` because they establish a comparator rather than evidence of either class.

### Indicators and thresholds

For each window, calculate the requested relative changes for MDF, MNF, and RMS. Also calculate a line-excluded low/high spectral power ratio using 20–80 Hz and 80–400 Hz power, with ±2 Hz removed around a detected 50/60 Hz line. RMS/MAV are contextual indicators and do not independently create fatigue-like labels.

For the primary rule, each MDF/MNF change must exceed both the 4 Hz two-bin resolution floor at 500 ms and `3 × 1.4826 × MAD` of the technical reference. The line-excluded low/high log-ratio must exceed three scaled MAD from reference; when its MAD is zero, use a 5% relative increase floor. A fatigue-like candidate requires both MDF and MNF to fall and the low/high ratio to rise beyond their personalized measurement thresholds. The entry condition must hold in at least 3 of the last 4 eligible windows, spanning at least 1.25 seconds. Sensitivity grids use 5%, 10%, 15%, and 20% relative changes; 2, 3, and 4 scaled-MAD thresholds; 1, 3, and 5-window persistence; and 3-versus-5-window technical references. The label audit will show how many cases remain unlabelable when a baseline dispersion estimate is zero or unstable.

### Hysteresis, classes, and confidence

- `1 = FATIGUE_LIKE`: a persistent multi-indicator change has crossed the entry rule.
- `0 = NON_FATIGUE_LIKE`: the window is technically valid and its spectral indicators remain within the reference envelope, with no unresolved contradictory signal. This means “no detected fatigue-like spectral change relative to this technical reference”; it does not mean clinically rested.
- `-1 = AMBIGUOUS / UNLABELED`: quality failure, missing/uncertain reference, discordant indicators, an unconfirmed transition, or insufficient confidence.

State machine: before entry, windows meeting all three stability bounds receive class 0; partial/discordant evidence and all candidate-but-unconfirmed entry windows receive `-1`. Class 1 begins on the third positive candidate within a rolling four eligible-window block; earlier candidates stay `-1` and are never retroactively relabeled. Once entered, class 1 remains latched. A discordant window while latched is `-1` but does not exit the state. The first two recovery candidates remain `-1`; the third consecutive eligible window satisfying all three recovery bounds at half entry magnitude is class 0 and exits the state. An ineligible window is `-1` and resets persistence counters while retaining the latched state; a data gap starts a new run with a new technical reference and unknown state. No persistence counter crosses a gap. Each label stores the rule version, source indicators, baseline values/dispersion, observed deltas, persistence state, assignment confidence, and abstention reason.

### Alternative label definitions

Compare v1 with: (a) persistent log-MDF slope/change-point plus low-frequency fraction shift; and (b) a line-excluded low-frequency energy-fraction change rule independent of MDF/MNF. Report agreement, group differences, class coverage, and threshold stability; do not choose the rule that maximizes model Accuracy.

## Data Partition and Model Evaluation

- Primary split: the restricted split receipt `private/split_manifest.json` assigns 12 development source groups and 3 internal-test source groups using seed `20261112`. Source groups are not verified participants. No random window split is allowed.
- Use four outer group folds and three inner group folds within the 12 development groups; each group appears in exactly one outer validation fold. Tune on mean per-validation-group macro-F1 with ties favoring lower complexity. Keep the 3 test groups blocked until thresholds, models, guardrails, and hyperparameters are frozen. Report each held-out group separately and avoid presenting windows as independent participants.
- In every outer fold, self-supervised encoder pretraining uses raw windows from outer-training groups only. Final encoder pretraining uses all 12 development groups only. No internal-test signal may enter unsupervised pretraining, normalization, or feature selection.
- Primary feature view A1 excludes MDF, MNF, frequency bins/ratios, spectral entropy, and any direct transformation of those label features. It includes time-domain amplitude/distribution and morphology features only. Also report spectral-only and combined views as circularity diagnostics.
- Train `DummyClassifier`, logistic regression, SVM-RBF, Random Forest, HistGradientBoosting, MLP, CNN-1D, and a fresh-from-scratch self-supervised encoder plus classification head. Train clean and augmented variants where architecture permits. Persist seeds, losses/epochs, parameters, selected features, serialized checkpoints outside Git, and out-of-sample predictions.
- Implement a latent mixture-of-experts model with an explicit gate, load/entropy monitoring, and no movement-phase names. Compare it with the direct CNN on held-out groups, macro-F1, label stability, and noise robustness.
- Include a time-relative-only classifier, majority baseline, and direct rule predictions. Investigate high time-only performance as a shortcut rather than changing splits or deleting samples.
- Report Accuracy, macro-F1, balanced accuracy, per-class recall, group-level and pooled confusion matrices, coverage, abstention, selective error, calibration/confidence, and grouped bootstrap uncertainty. Undefined metrics remain NA with a reason.
- For OOD/confidence guardrails, fit robust feature ranges and confidence limits on development data only. Report full-population and selective metrics at prespecified confidence thresholds; do not adjust them on the internal test or RAR.

## Augmentation and Robustness

Training augmentation uses Gaussian, colored, line-frequency, baseline-wander, impulsive, limited temporal-shift, moderate gain, and combined perturbations at 5%, 10%, 15%, and 20% severity measured relative to the clean window RMS. Apply augmentations only within model training. Keep the original clean pseudo-label fixed; separately recalculate the frozen rule on perturbed signals to quantify rule consistency without silently relabeling training examples.

Evaluate frozen models using identical fixed-noise arrays at 5%, 10%, 15%, 20%, 25%, and 50%. Levels 25% and 50% are stress tests. Report class metrics, degradation, per-family errors, rule consistency, signal-quality rejection, prediction coverage, and direct-versus-MoE stability. Do not retrain or recalibrate during robustness evaluation.

## External Evaluation and Provenance

The external RAR stays closed until the full preprocessing, label rules, thresholds, models, feature views, guardrails, and hyperparameters are frozen. Then apply the frozen method exactly. Report external file-analysis metrics as agreement with the operational pseudo-label rule, never independent validation of fatigue. Repeat 25%/50% robustness without retraining. Preserve the known prior project exposure and do not claim a pristine external holdout.

Private run storage holds group mapping, file hashes, windows, labels, predictions, per-group metrics, model artifacts, and any member-level external outputs. Public results contain only source-safe code, configuration, aggregated counts/metrics, and limitations. Hash the run configuration, label rule, source code, inputs, outputs, checkpoints, and final manifests. Do not commit raw signals, source names, individual rows, or trained model files.

## Primary Sources

- Beck TW, Stock MS, DeFreitas JM. *Shifts in EMG spectral power during fatiguing dynamic contractions.* Muscle & Nerve. 2014;50:95–102. https://doi.org/10.1002/mus.24098
- Croce R, et al. *Quadriceps mechano- and electromyographic time-frequency responses during muscular contractions to volitional exhaustion.* Muscle & Nerve. 2016;53:452–463. https://doi.org/10.1002/mus.24764
- Gerdle B, Larsson B, Karlsson S. *Criterion validation of surface EMG variables as fatigue indicators using peak torque: repetitive maximum isokinetic knee extensions.* J Electromyogr Kinesiol. 2000;10:225–232. https://pubmed.ncbi.nlm.nih.gov/10969195/
- Pincivero DM, et al. *Quadriceps femoris electromyogram during concentric, isometric and eccentric phases of fatiguing dynamic knee extensions.* J Biomech. 2006;39:246–254. https://doi.org/10.1016/j.jbiomech.2004.11.023
- Cifrek M, Tonković S, Medved V. *Measurement and analysis of surface myoelectric signals during fatigued cyclic dynamic contractions.* Measurement. 2000;27:85–92. https://doi.org/10.1016/S0263-2241(99)00059-7
- Pease WS, Elinski MA. *Surface and wire electromyographic recording during fatiguing exercise.* Electromyogr Clin Neurophysiol. 2003;43:267–271. https://pubmed.ncbi.nlm.nih.gov/12964253/
