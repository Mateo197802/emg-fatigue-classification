# Raw First sEMG Fatigue Experiment Design

## Intent

Build and execute a reproducible raw first signal pipeline for the supplied sEMG training archive. The scientific target remains physiological non-fatigue versus fatigue, but the supplied CSV schema contains only timestamps and signal measurements. Until synchronized independent fatigue references are found, the work stops at signal characterization, self-supervised representation learning, and noise robustness; it does not produce a fatigue classifier or fatigue metrics.

## Goals

- Audit every source CSV for timestamp integrity, sampling behavior, signal quality, ADC-to-voltage relationship, and spectral characteristics.
- Process each continuous segment separately, select preprocessing from measured acquisition evidence, and extract primary 500 ms windows plus supported 250 ms and 1000 ms sensitivities.
- Extract documented temporal, spectral, and wavelet features with training-only feature standardization.
- Train clean and augmentation-trained compact 1D CNN encoders from scratch using masked reconstruction.
- Evaluate reconstruction and representation/feature stability under the specified corruption families and severities, without treating that as supervised fatigue performance.
- Keep raw data, source-name mappings, per-file manifests, checkpoints, and participant-level artifacts in a private local output root. Commit only source code and safe aggregate evidence.

## Non-goals and anti-goals

- Do not create fatigue or extension/flexion labels from time, file order, repetition, signal features, or recording names.
- Do not reuse historical feature matrices, labels, checkpoints, or metrics as evidence for this experiment.
- Do not open the supplied RAR until training choices are frozen; any post-freeze analysis is held-out file analysis unless identity and provenance are verified.
- Do not generate a manuscript, course report, DOCX, PDF, slides, or delivery ZIP.

## Constraints and evidence

- The user supplied a training ZIP and a reserved final RAR. The ZIP contains 140 CSV files in 15 source folders with the required columns, but its folder names are not the expected anonymized `G01`–`G15`. Processing will replace those names with deterministic opaque group IDs and will not publish the name map.
- There are no label sidecars in the training ZIP; no independent fatigue or movement-phase reference is present in the supplied training source.
- The source folders are grouping units only. They are not verified participant identities.
- The final RAR is reserved for the post-freeze check and must not affect model development, normalization, feature selection, augmentation calibration, or training.
- GitHub reports read-only repository permissions, so code and evidence can be committed locally, while push/PR creation may be unavailable.
- The only ADC/voltage evidence in the supplied training source is the paired CSV columns. The audit will quantify their relationship, but will not infer ADC hardware bit depth or claim a calibrated physical voltage without device metadata.

## Approaches considered

1. Extend the existing early/late classification path. Rejected because its proxy target and prior results cannot establish physiological fatigue and the objective forbids importing those labels or models.
2. Add an isolated raw first experiment package beside the historical implementation. Selected because it preserves the old project while keeping new acquisition, preprocessing, feature, SSL, augmentation, and evaluation logic independent.
3. Train on the RAR or combine it with the supplied ZIP. Rejected because the RAR is reserved for final evaluation and is not to be opened before model freeze.

## Implementation choices left to the builder

- Confirm acquisition frequency, gap thresholds, usable passband, line frequency, and resampling from the measured timestamps and spectra.
- Use a deterministic source-group holdout for self-supervised validation; this does not imply participant independence.
- Use a compact 1D CNN masked-reconstruction objective, fixed seeds, and the CPU PyTorch environment recorded in the run manifest.
- Store all restricted outputs under a local application-data directory outside the checkout. Public evidence must be aggregate-only and contain no source filenames, per-group records, or raw traces.

## Corruption contract

Training-only augmentation uses 5%, 10%, 15%, and 20% of the clean-window RMS. Held-out robustness evaluation uses clean, 5%, 10%, 15%, 20%, 25%, and 50%, with 25% and 50% treated as extrapolation/stress conditions. Record achieved noise RMS and SNR for each additive corruption. Families are Gaussian white noise, band-limited correlated noise, measured/region-relevant 50/60 Hz line interference, DC shift and low-frequency baseline wander, sparse impulsive artifacts, low-frequency motion transients, small label-preserving temporal shifts, plausible gain perturbations, and a combined mixture normalized to the requested total RMS severity. Keep matched held-out windows and fixed seeds across clean/augmented encoders. Do not assume that a transformation preserves a label; in this label-free run, interpret results only as signal, feature, reconstruction, and embedding stability.

## External check contract

Only after preprocessing and model choices are frozen, inspect the final RAR for archive integrity, CSV count and schema, group/file layout, timestamp and sampling compatibility, signal and protocol correspondence, any available references and their granularity, previous exposure, and evidence of source-group independence. Do not fit or tune on its contents. If valid references are absent or independence is unverified, do not report supervised fatigue metrics or claim external validation. Any feature/embedding output is exploratory held-out file analysis, reported separately by opaque source group where safe.

## Delivery boundary

The work is complete only when the raw audit, preprocessing and feature pipeline, clean and augmented SSL training, corruption analysis, label-blocker review, private reproducibility receipts, and safe repository updates have been executed. Supervised model families, Architecture A/B performance, confusion matrices, and fatigue sensitivity are scientifically blocked until valid synchronized labels are supplied.
