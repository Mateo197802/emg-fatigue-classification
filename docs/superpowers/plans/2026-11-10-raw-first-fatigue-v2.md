# Raw First sEMG Fatigue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a reproducible raw-first sEMG audit, feature workflow, clean/augmented self-supervised training, and corruption robustness analysis without inventing fatigue labels.

**Architecture:** Keep the new experiment in a dedicated `emg_fatigue.raw_first` package and a separate runner. Stream the authorized training ZIP directly, anonymize source folders in memory, write restricted run artifacts outside the repository, and commit only safe code and aggregate summaries. Keep the historical early/late experiment untouched.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy, scikit-learn, Matplotlib, and the installed CPU PyTorch 2.12.1+cpu.

**Spec:** `docs/superpowers/specs/2026-11-10-raw-first-fatigue-design.md`

## Global Constraints

- The final RAR remains unopened until model and preprocessing choices are frozen.
- The only training input is the supplied training ZIP; the RAR is never mixed into development or fitted transformations.
- Source folder identity is not verified participant identity.
- No labels may be derived from time, filename, feature thresholds, repetition order, or phase halves.
- All learned transforms are fit on development source groups only.
- Raw signals, source names, per-file hashes, participant-level data, embeddings, and checkpoints remain private.
- No supervised fatigue or phase metric is reported without an independent synchronized reference.
- Do not add or run software tests; verify through the requested executable audit, training, and reproducibility runs.

## Review Focus

- Nonmonotonic or duplicated timestamps must split or invalidate segments rather than be silently sorted or interpolated — verify in the per-file acquisition audit.
- Filtering must not bridge gaps or discard fatigue-relevant amplitude information — verify segment-level before/after summaries and plots.
- Source group aliases must not leak original folder or filename identifiers into committed artifacts — inspect the aggregate evidence tree and repository diff.
- Noise severity must be measured against clean-window RMS and use matched fixed seeds across encoders — verify the augmentation receipt and robustness output.
- Missing references must block supervised outputs — verify the label audit states the missing measurement and granularity explicitly.

---

### Task 1: Training archive inventory and input contract

**Files:**
- Create: `src/emg_fatigue/raw_first/contracts.py`
- Create: `src/emg_fatigue/raw_first/io.py`
- Create: `configs/raw_first.toml`
- Create: `tools/run_raw_first_experiment.py`

**Interfaces:**
- `iter_training_csvs(zip_path) -> Iterable[RawRecording]`
- `RawRecording` contains only opaque `group_id`, opaque `recording_id`, internal archive member name, and parsed numeric arrays.
- The runner accepts explicit input archive and output root arguments and rejects any RAR path as a training input.

- [x] Validate ZIP CRC/structure, CSV count, group count, required columns, parseability, nonfinite counts, and deterministic anonymized aliases without writing raw files to the repository.
- [x] Save input hash and source-member hashes in the private run receipts only.
- [x] Record the actual supplied ZIP name and the expected-name mismatch in aggregate metadata without publishing source names.

**Execution check:** Run the inventory against the supplied ZIP and confirm 140 numeric CSV records, 15 opaque groups, and a uniform required schema.

### Task 2: Acquisition, sampling, and quality audit

**Files:**
- Create: `src/emg_fatigue/raw_first/audit.py`
- Create: `src/emg_fatigue/raw_first/plots.py`

**Interfaces:**
- `audit_recording(recording) -> (record_summary, interval_summary, signal_summary)`
- `make_audit_figures(sampled_records, output_dir) -> list[Path]`

- [x] Measure sample count, duration, timestamp monotonicity, interval distribution, duplicates, resets, gaps, jitter, Nyquist, and robust sampling-rate estimate per source file.
- [x] Quantify ADC/voltage linear relation, amplitude distribution, flatline and transient candidates, baseline, PSD, and 50/60 Hz contamination.
- [x] Produce restricted per-file tables and aggregate-only public tables/plots.
- [x] Select the common processing rate, passband, optional notch, and QC rules from development-group audits only; apply the frozen choices unchanged to held-out groups.

**Execution check:** Run the audit over all source files and inspect before/after waveform and PSD figures from representative recordings selected by deterministic opaque IDs.

### Task 3: Filtering, windows, and physiological features

**Files:**
- Create: `src/emg_fatigue/raw_first/preprocessing.py`
- Create: `src/emg_fatigue/raw_first/windows.py`
- Create: `src/emg_fatigue/raw_first/features.py`

**Interfaces:**
- `preprocess_segments(recording, config) -> list[ProcessedSegment]`
- `segment_windows(segment, duration_s, overlap) -> Iterable[SignalWindow]`
- `extract_features(window) -> dict[str, float]`

- [x] Detrend, filter with SOS and zero-phase processing where segment length allows, and preserve discontinuities.
- [x] Segment 500 ms windows at 50% overlap plus 250 ms and 1000 ms sensitivities when supported.
- [x] Exclude nonfinite, discontinuous, and clearly invalid windows using prespecified signal-quality criteria; report rejection counts.
- [x] Extract temporal, spectral, band-power/ratio, Hjorth, and Haar wavelet-energy features with formulas and units/limitations.
- [x] Fit feature standardization on development groups only and apply unchanged to validation groups.

**Execution check:** Run all window sizes, generate feature schema/formula receipts, and review global feature distributions, correlations, and filter sensitivity.

### Task 4: Reference search and group partition

**Files:**
- Create: `src/emg_fatigue/raw_first/labels.py`
- Create: `src/emg_fatigue/raw_first/splits.py`

**Interfaces:**
- `audit_references(training_archive) -> LabelAudit`
- `source_group_split(group_ids, seed) -> GroupPartition`

- [x] Inventory all authorized archive members and signal-schema headers. If sidecars or non-signal CSVs exist without validated content, mark reference search incomplete and keep supervised outputs blocked.
- [x] Do not read label information from filenames or timestamps.
- [x] If no labels exist, write a precise supervised blocker and do not create class targets, model metrics, or confusion matrices.
- [x] Make a deterministic 12/3 source-group split for self-supervised reconstruction validation, clearly marked as source-group rather than verified-participant holdout.

**Execution check:** Confirm the label audit and partition manifest contain only opaque groups, disclose that participant identity is unverified, and have no target labels.

### Task 5: Training-only corruptions and masked reconstruction

**Files:**
- Create: `src/emg_fatigue/raw_first/augmentation.py`
- Create: `src/emg_fatigue/raw_first/ssl.py`

**Interfaces:**
- `corrupt_signal(signal, family, severity, seed, config) -> (corrupted, corruption_receipt)`
- `train_masked_encoder(train_windows, validation_windows, augmentation, config) -> (model, history)`

- [x] Define Gaussian, colored, line-frequency, baseline, impulsive, motion, small-shift, gain, and combined corruptions relative to clean RMS.
- [x] Verify achieved RMS ratios/SNR and record parameters and seeds.
- [x] Train a compact 1D CNN masked-reconstruction encoder from scratch in clean and augmentation-trained conditions using development groups only.
- [x] Save checkpoints, software versions, training history, and hashes to the private output root.
- [x] Report reconstruction only; never call it fatigue detection.

**Execution check:** Run both training conditions with fixed seeds and compare final and best validation reconstruction loss on held-out source groups.

### Task 6: Feature and representation robustness

**Files:**
- Create: `src/emg_fatigue/raw_first/robustness.py`
- Modify: `tools/run_raw_first_experiment.py`

- [x] Freeze preprocessing, features, and encoders before robustness evaluation.
- [x] Evaluate clean, 5%, 10%, 15%, 20%, 25%, and 50% corruption using matched held-out windows and fixed seeds.
- [x] Evaluate all applicable families independently and combined; measure feature change, embedding similarity, reconstruction error, and invalid-window rate.
- [x] Label severe 25% and 50% cases as stress tests; do not interpret representation stability as classification robustness.

**Execution check:** Re-run the frozen encoders on identical clean/corrupted windows and confirm the same input produces identical saved predictions/embeddings under deterministic inference.

### Task 7: Frozen external file audit and reproducibility handoff

**Files:**
- Modify: `tools/run_raw_first_experiment.py`
- Create: `docs/raw_first/EXECUTION_SUMMARY.md`
- Create: `docs/raw_first/aggregate_audit.csv`
- Create: `docs/raw_first/feature_robustness.csv`
- Create: `docs/raw_first/ssl_training_summary.csv`

- [x] Freeze the complete training configuration and compute its hash before accessing the RAR.
- [x] Only then inspect the RAR's integrity, signal format, acquisition compatibility, labels, and identity/provenance; if references are absent or identities unverifiable, explicitly limit results to held-out file analysis.
- [x] Keep external per-file details, hashes, predictions, and model artifacts private.
- [x] Recompute and verify frozen configuration/checkpoint hashes before any RAR read; verify private manifest hashes and deterministic inference; commit only code and safe aggregate evidence.
- [x] Create a local commit on `research/raw-first-fatigue-v2`; attempt push/PR only if repository permission allows.

**Execution check:** Review repository diff for private data, then record local commit and the actual push/PR status without claiming an unavailable remote update.
