# Raw First sEMG Experiment Summary

## Input audit

- Training input: the supplied ZIP archive; its exact filename, SHA-256, and original-file hashes are recorded in the restricted local run manifest.
- Parsed 140 CSV files in 15 opaque source groups. Source-group identity is not verified participant identity.
- The supplied training archive filename differs from expected `EMG_TRAIN_SOLO_CSV_ANONIMO.zip`; the actual name is retained only in the restricted manifest. Source folders were assigned opaque aliases in memory.
- Effective sampling rate across valid files: median 1000.000 Hz, range 1000.000–1000.000 Hz.
- Sampling-rate, passband, and notch choices were derived from 12 development source groups only (110 recordings); held-out groups were excluded from selection.
- Chosen common rate: 1000.000 Hz. 1000 Hz supported: median is within 1% and at least 95% of valid recordings are within 2%.
- Filter: 20.0–400.0 Hz; no notch applied because no line frequency met the prespecified prevalence threshold.
- Valid windows: 500 ms 16,629, 250 ms 33,649, 1000 ms 8,128. Primary 500 ms rejected: 0.
- `ADC_Value` is treated as an exported signed signal scale. The paired voltage column is analyzed for numerical relationship only; sensor calibration and raw ADC hardware representation are not verified.

## Labels and supervised scope

- Status: **blocked_no_independent_reference**. All archive members were CSV signals with only Relative_Time_s, ADC_Value, and Voltage_mV columns. No independent synchronized fatigue or movement-phase reference was present; filenames and temporal position are not labels.
- Missing independent fatigue references: Borg CR-10, pre/post MVC, force/torque, repetition-level performance, or a mechanical failure criterion.
- Missing independent phase references: synchronized extension/flexion or knee-angle/video annotations.
- No supervised fatigue/phase model or classification metric (accuracy, macro-F1, sensitivity, or confusion matrix) was produced.

## Self-supervised training

- Source-group split: 12 development groups and 3 held-out groups. These are not verified participants.
- Model: compact 1D CNN trained from scratch with masked reconstruction; clean and augmentation-trained versions use the same training/validation source groups.
- clean: best epoch 11; validation masked reconstruction loss 1.02543; checkpoint SHA-256 `56261450c04f8d5c76ada6555e57aff28e8c62d19a9a64139e2d0ff12eb62b07`.
- augmented: best epoch 11; validation masked reconstruction loss 1.01892; checkpoint SHA-256 `c7a4bf16b307be5c462919650dd4f15c6949479553d4079900e9fef47000c550`.

## Noise robustness

- Median across family-level embedding cosine summaries at 20% severity: clean 0.9968, augmentation-trained 0.9976; at 50% stress: 0.9764 and 0.9797. Worst 50% family: clean baseline_shift_wander at 0.743; augmentation-trained baseline_shift_wander at 0.813. These measure representation stability, not classification accuracy.
- All robustness numbers are feature, embedding, and reconstruction stability summaries under fixed matched corruptions; they do not establish a fatigue classifier's robustness.
- Filtering sensitivity is summarized on held-out source groups using a lower high-cut/no-notch alternative; window-size sensitivity is summarized at 250, 500, and 1000 ms.
- Detailed feature-family × severity × encoder tables, training histories, audit tables, waveforms, spectra, per-file hashes, and embeddings are stored in the restricted local run directory.

## External RAR

- The archive integrity check passed. The supplied archive filename differs from expected `raw(2).rar`; the actual name is kept in the restricted audit.
- Inspected 20 CSV members across 1 opaque source group; 20 matched the frozen signal schema, and 0 were excluded for sampling incompatibility.
- 20 of 20 parsed CSVs matched the frozen three-column schema. Frozen preprocessing and both encoders produced 2,763 exploratory windows after training freeze.
- A complete reference search found no independent fatigue or movement-phase labels. No predictions or supervised metrics were produced.
- Source identity independence and prior exposure remain unverified. Treat this as held-out file analysis, not independent external validation.
