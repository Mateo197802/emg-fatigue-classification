# Data and output dictionary

| Field or product | Meaning |
|---|---|
| Relative_Time_s | Original seconds; strictly increasing inside a supplied recording |
| ADC_Value | Stored signed count value, without verified physical calibration |
| Voltage_mV | Optional exporter conversion; ignored as a physiological voltage |
| subject_id | Private source-folder pseudonym; identity not independently verified |
| recording_id | Private deterministic recording key |
| series | R1–R5 formal; R0 control only; not a verified repetition number |
| side | Filename-derived left/right; metadata, never a primary predictor |
| qc_valid | Complete window passes run quality and filter-edge masks |
| activity_occupancy | Fraction of window covered by candidate activity, not verified contraction |
| label 0 | early_proxy_nonfatigue, weak elapsed-time label |
| label 1 | late_proxy_fatigue, weak elapsed-time label |
| missing label | QC/activity/feature failure or excluded transition/boundary; never recoded as 0 |
| rms, mav | Centered stored-signal amplitude in counts |
| waveform_length_per_s | Sum absolute adjacent changes divided by window duration, counts/s |
| mdf_hz, mnf_hz | Hann in-band PSD-derived median/mean frequency, Hz |
| power_ratio_low_high | PSD bin-sum ratio [20,60) Hz / [60,400] Hz |
| wavelet_A3/D3/D2/D1 | Relative coefficient energies, dimensionless |
| inventory.csv, input_hashes.json | Private source membership and original byte hashes |
| features.parquet, labels.csv | Private window products and eligibility reasons |
| processing_receipt.json | Frozen config, feature bytes, source hashes and environment |
| evaluation/results.csv | Model × outer fold metrics, private link to recording group |
| evaluation/oof_predictions.parquet | Private per-window held-out predictions, exactly once per model |
| search_results.csv, splits.json | Inner selection audit and group membership |
| model.joblib, model.receipt.json | Trusted local fitted pipeline, integrity and training provenance |
| predictions.csv | Offline classes/scores with explicit abstention reasons |

`average_precision` is the non-interpolated average-precision metric; `pr_auc_trapezoidal` is trapezoidal area under the precision-recall curve. They are separate quantities. Primary model uncertainty is across held-out recording groups. Pooled confusion counts weight windows by their actual support and therefore answer a different aggregation question.
