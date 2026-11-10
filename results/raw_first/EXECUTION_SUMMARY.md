# Raw First sEMG Experiment Summary

## Input and acquisition audit

- The task's expected training archive (`EMG_TRAIN_SOLO_CSV_ANONIMO.zip`) was not among the supplied files, so the provided `OneDrive_1_10-10-2026.zip` was used. Its SHA-256 and original-file hashes are in the restricted local run manifest.
- Parsed 140 CSVs across 15 opaque source-folder groups. These groups are not verified participant identities; original folder names were mapped in memory and kept out of public outputs.
- All 140 files had a robust effective rate of approximately 1000 Hz (median 1000.000 Hz; observed range 999.9999999998881–1000.0000000005542 Hz). The selected common rate was 1000 Hz.
- Selected 20–400 Hz bandpass; no notch was applied because neither line frequency met the prespecified prevalence threshold. No primary-window rejections occurred.
- Valid windows: 16,629 at 500 ms, 33,649 at 250 ms, and 8,128 at 1000 ms. ADC-to-voltage linear fit median R² was 0.999999996; this is a numerical relationship, not sensor calibration or verification of hardware ADC units.

## Labels and supervised scope

- The training CSV schema contains only `Relative_Time_s`, `ADC_Value`, and `Voltage_mV`. There are no independent synchronized fatigue or movement-phase references in the authorized training archive.
- Missing references include Borg CR-10, pre/post MVC, force/torque, repetition-level performance or mechanical failure, and synchronized extension/flexion or knee-angle/video labels.
- No target was derived from time, filename, amplitude, repetition order, or phase halves. No supervised classifier, fatigue or phase score, accuracy, macro-F1, sensitivity, or confusion matrix was produced.

## Self-supervised training

- Source-group split: 12 development groups and 3 held-out groups for reconstruction validation. Participant independence is unverified.
- Trained compact 1D CNN masked-reconstruction encoders on 13,333 training windows and evaluated on 3,296 validation windows, with 12 epochs requested and best epoch 11 in both conditions.
- Clean: best validation masked-reconstruction loss 1.025430; checkpoint SHA-256 `56261450c04f8d5c76ada6555e57aff28e8c62d19a9a64139e2d0ff12eb62b07`.
- Augmentation-trained: best validation masked-reconstruction loss 1.018922; checkpoint SHA-256 `c7a4bf16b307be5c462919650dd4f15c6949479553d4079900e9fef47000c550`. Training corruptions used ten signal-noise families at 5%, 10%, 15%, and 20% of clean-window RMS.

## Corruption and sensitivity analysis

- Compared clean and augmentation-trained encoders on 255 matched held-out windows across ten corruption families, at 5%, 10%, 15%, 20%, 25%, and 50%; 25% and 50% are stress tests. No windows were quality-rejected in the matched set.
- Median across family-level median embedding cosine at 20% severity: clean 0.9968, augmentation-trained 0.9976. At 50%: clean 0.9764, augmentation-trained 0.9797.
- The most affected family at 50% was baseline shift/wander: median cosine 0.743 for the clean encoder and 0.813 for the augmentation-trained encoder. This measures representation stability, not classification robustness.
- Filtering sensitivity used a 320 Hz high-cut alternative on held-out source groups; window-size sensitivity compared 250, 500, and 1000 ms. Detailed feature tables and per-file plots remain in the restricted local run directory.

## Final-only external RAR analysis

- The supplied archive was `raw.rar`; the objective names `raw(2).rar`. The RAR integrity check passed after the training configuration and both encoders were frozen.
- Inspected 20 CSV members in one opaque source group. All 20 matched the frozen three-column schema and were sampling-compatible; frozen preprocessing and both encoders produced 2,763 exploratory windows. No non-CSV members or independent label references were found.
- No fatigue or phase predictions/metrics were produced. Source identity independence and prior project exposure cannot be ruled out, so this is held-out file analysis rather than independent external validation.
- Encoder checkpoint round-trip and repeated inference were exact for both models on the same 255-window input; embeddings and reconstructions matched exactly.
