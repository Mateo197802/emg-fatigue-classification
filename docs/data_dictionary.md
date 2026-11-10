# Data and output dictionary

| Field or product | Meaning |
|---|---|
| Relative_Time_s | Original seconds; strictly increasing inside a supplied recording |
| ADC_Value | Stored signed count value, without verified physical calibration |
| Voltage_mV | Exporter conversion `round(ADC_Value × 3300 / 4095, 2)`; arithmetic matches stored values but does not verify physical calibration, sensor mode, offset or gain; ignored as a physiological voltage |
| subject_id | Private source-folder pseudonym; identity not independently verified |
| recording_id | Private deterministic recording key |
| series | R1–R5 formal; R0 control only; not a verified repetition number |
| side | Filename-derived left/right; metadata, never a primary predictor |
| qc_valid | Complete window passes run quality and filter-edge masks |
| activity_occupancy | Fraction of window covered by candidate activity, not verified contraction |
| label 0 | `early_recording_position_proxy`; eligible window fully before the first elapsed-time third |
| label 1 | `late_recording_position_proxy`; eligible window starts at or after the final elapsed-time third |
| missing label | QC/activity/feature failure or excluded transition/boundary; never recoded as 0 |
| rms, mav | Centered stored-signal amplitude in counts |
| waveform_length_per_s | Sum absolute adjacent changes divided by window duration, counts/s |
| mdf_hz, mnf_hz | Hann in-band PSD-derived median/mean frequency, Hz |
| power_ratio_low_high | PSD bin-sum ratio [20,60) Hz / [60,400] Hz |
| wavelet_A3/D3/D2/D1 | Relative coefficient energies, dimensionless |
| feature_schema_version | `emg_window_features_v1` for historical results; `emg_window_features_v2_sample_span` for corrected waveform-length calculations |
| waveform_length_per_s | Historical v1 convention: `Σ|x[n]−x[n−1]| / (N/fs)`, counts/s |
| waveform_length_per_s_sample_span | Corrected v2 convention: `Σ|x[n]−x[n−1]| / ((N−1)/fs)`, counts/s; never silently substituted for v1 |
| label_evidence | `weak_elapsed_time` for historical recording-position proxies; reference-backed rows require an approved torque/reference contract |
| reference_level | `session`, `repetition`, `time_series`, or `window`; do not expand a coarse level to windows without a validated mapping |
| evidence_status | `synthetic_only` for synthetic fixtures; `reference_approved` only after calibrated, synchronized and approved reference validation |
| abstention_reason | Explicit invalid/QC/phase-transition/onset-boundary reason; unknown or missing references are never encoded as class 0 |
| participant_id_verified | False in the supplied corpus; `subject_id` is folder-derived and is only a conditional grouping key |
| phase label | Future reference-backed state from synchronized knee angle/video; EMG burst-half labels remain proxy labels |
| fatigue label | Future reference-backed state from approved mechanical reference; elapsed-time early/late labels remain proxy labels |
| inventory.csv, input_hashes.json | Private source membership and original byte hashes |
| features.parquet, labels.csv | Private window products and eligibility reasons |
| processing_receipt.json | Frozen config, feature bytes, source hashes, filter-design manifest hash and environment |
| filter_designs.json | Private per-run map of sampling-rate-specific Butterworth SOS and any conditionally applied 50/60 Hz notch coefficients, thresholds and source run bindings |
| evaluation/results.csv | Model × outer fold metrics, private link to recording group |
| evaluation/oof_predictions.parquet | Private per-window held-out predictions, exactly once per model |
| search_results.csv, splits.json | Inner selection audit and group membership |
| model.joblib, model.receipt.json | Trusted local fitted pipeline, integrity and training provenance |
| predictions.csv | Offline classes/scores with explicit abstention reasons |

`average_precision` is the non-interpolated average-precision metric; `pr_auc_trapezoidal` is trapezoidal area under the precision-recall curve. They are separate quantities. Primary model uncertainty is across held-out recording groups. Pooled confusion counts weight windows by their actual support and therefore answer a different aggregation question.
