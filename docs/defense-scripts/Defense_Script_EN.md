# EMG Defense Script

16 principal slides. Scheduled talk time: 600 seconds. Thirteen appendix slides for questions.

## 1. Temporal-proxy classification with sEMG

Scheduled time: 20 s.

This study classifies an early-versus-late proxy created from each recording’s position in time. The ZIP does not contain independent measurements to claim true muscle-fatigue detection. I will describe the corpus, signal, pre-analysis, preprocessing, models, and the auxiliary experiment, separating completed results from the proposed design.

Source: Integrative biomechanics project.

## 2. Corpus and quantitative overview

Scheduled time: 35 s.

The main ZIP contains 140 CSV files and 4,256,257 samples. The 138 R1 to R5 recordings enter formal processing; two R0 files remain controls. Fifteen groups come from folder organization. Both legs and every series stay together during grouped evaluation. The RAR remains separate because participant identity and cross-archive links are not verified.

Source: inventory.csv and coverage.csv.

## 3. Recorded signal and time diagnostics

Scheduled time: 35 s.

The CSV files contain relative time and ADC counts at approximately one thousand samples per second. The plotted trace uses one selected QC-valid run, while processing preserves every sample. We inspected clock gaps, jitter, constant segments and repeated extremes. ADC counts do not establish calibrated muscle amplitude in millivolts.

Source: Original CSV, qc.py, preprocessing.py.

## 4. Descriptive signal pre-analysis

Scheduled time: 35 s.

These curves summarize candidate-active windows by recording position and then by group. They show electrical amplitude and spectral changes in the available files. They do not establish fatigue, because there are no synchronized force or torque labels, and the curves are descriptive rather than training targets.

Source: feature_trajectories_by_group.csv.

## 5. Signal preprocessing

Scheduled time: 45 s.

The historical pipeline separates signal at acquisition gaps and rejects excessive clock jitter and constant segments. It uses a fourth-order Butterworth low-pass prototype transformed into a digital 20 to 400 hertz band-pass, applied bidirectionally in second-order sections. One hundred milliseconds at each edge are masked. The optional notch is not in the primary pipeline. Processing is offline.

Source: configs/study.yaml, qc.py and preprocessing.py.

## 6. Windows and temporal proxy labels

Scheduled time: 45 s.

Complete windows in the first elapsed-time third receive proxy class zero; windows in the final third receive class one. The central third and boundary-crossing windows are excluded. A candidate-activity criterion selects usable windows. The observation is a window; the learned target is temporal position, not measured muscle fatigue.

Source: label_audit.json and coverage.csv.

## 7. Ten features per window

Scheduled time: 35 s.

The input contains three amplitude features, three spectral features and four relative db4 wavelet energies. The two-stage proxy experiment and direct comparator use the same ten original features. Time, file identity, side and series are excluded. The scaler is learned only from training folds, and no MVC normalization is claimed.

Source: features.py and feature_schema.json.

## 8. Grouped validation method

Scheduled time: 45 s.

Each outer round holds out one entire recording-derived group. The other groups are used for grouped inner model selection. Scaling and fitting use training data only. Each eligible window receives an out-of-fold prediction. Uncertainty is summarized across groups rather than treating overlapping windows as independent subjects.

Source: evaluation.py, splits.json and evaluation_receipt.json.

## 9. Models and parameter selection

Scheduled time: 35 s.

The majority model is a trivial reference. Logistic regression on RMS and MDF is the simple baseline. Full logistic regression, an RBF support-vector machine and a random forest test more flexible mappings. Nested selection chooses a family inside grouped validation; it does not average model votes.

Source: evaluation.py and inner_search.csv.

## 10. Performance on the temporal proxy

Scheduled time: 45 s.

All procedures use the same 5,509 early/late windows and fifteen held-out groups. The nested selector has macro-F1 0.6240 versus 0.6272 for logistic RMS plus MDF. The paired difference is -0.0032, with a 95 percent group-bootstrap interval from -0.0591 to +0.0509. No complex-model gain is demonstrated. These values are not Accuracy or F1 for physiological fatigue.

Source: evaluation/metrics_summary.csv and paired_differences.csv.

## 11. Main proxy errors

Scheduled time: 35 s.

The pooled confusion matrix shows errors for the early and late temporal classes. Group-level macro-F1 varies substantially, so pooled counts do not replace across-group uncertainty. Without independent references, these errors cannot be assigned a physiological cause or interpreted as fatigue mistakes.

Source: evaluation/oof_predictions.parquet and evaluation/results.csv.

## 12. Auxiliary EMG cascade

Scheduled time: 45 s.

The first stage is a random forest that classifies first versus second halves of candidate EMG bursts. Its score enters a logistic second stage together with the original ten features and selected interactions. The auxiliary label is an EMG burst half, not observed knee extension or flexion.

Source: phase_spike.py and the frozen exploratory protocol.

## 13. Auxiliary experiment design

Scheduled time: 35 s.

The variants use identical windows, temporal labels, scaling and held-out groups. We compare the direct ten-feature logistic model, EMG plus score, conditioned interactions, score only and a random-label control. The auxiliary head was fit on 3,337 windows from 996 candidate bursts.

Source: model_registry.csv and phase protocol.

## 14. Auxiliary performance on the same proxy

Scheduled time: 40 s.

The direct model reaches 0.6355 macro-F1; the conditioned model reaches 0.6293. Their paired difference is -0.0062, with a 95 percent interval from -0.0232 to +0.0082. Six groups improve, eight decline and one ties. The result does not show an improvement on the temporal proxy and says nothing about true fatigue detection.

Source: phase_experiment/metrics_summary.csv and paired_differences.csv.

## 15. Targets for each cascade stage

Scheduled time: 35 s.

The auxiliary confusion matrix evaluates burst-half labels, while the second-stage matrix evaluates early-versus-late recording position. They use different supports. The first matrix is not extension versus flexion, and neither matrix is fatigue versus no fatigue.

Source: confusion_matrices.csv and auxiliary_phase_fold_metrics.csv.

## 16. Conclusions

Scheduled time: 35 s.

The executed experiment predicts a temporal proxy, not muscular fatigue versus no fatigue. The nested selector does not clearly improve on RMS plus MDF; the burst-half score also fails to improve the direct proxy model. This ZIP cannot produce valid physiological fatigue Accuracy or macro-F1. A future evaluation needs synchronized calibrated force or torque; phase supervision needs synchronized kinematics.

Source: Primary evaluation and auxiliary experiment.

## 17. Appendix: relationships among the models

Appendix for questions.

The project includes six principal procedures, five supplemental variants, an auxiliary head and a separate full-cohort fitted artifact. The matched ten-feature logistic baseline is distinct from the nested selection procedure. These roles should not be counted as independent evidence or independent cohorts.

Source: model_registry.csv.

## 18. Appendix: sensitivities and ablations

Appendix for questions.

Prespecified sensitivities and feature-family ablations test how proxy results change under alternate analysis choices. Some sensitivities use different retained windows. We report these scenarios as sensitivity checks and do not select the most favorable one as the headline result.

Source: sensitivity_summary.csv and ablation_summary.csv.

## 19. Appendix: principal confusion counts

Appendix for questions.

These are pooled out-of-fold confusion counts for each principal procedure on the same 5,509 windows. Class zero means the first recording third; class one means the last third.

Source: Primary confusion matrices.

## 20. Appendix: normalized principal matrices

Appendix for questions.

Each row is normalized within its observed temporal-proxy class. The matrices show class-specific error proportions, not validated physiological sensitivity or specificity.

Source: Normalized primary confusion matrices.

## 21. Appendix: auxiliary confusion counts

Appendix for questions.

These matrices compare the supplemental stage-two variants on the same early/late proxy support. The auxiliary burst-half head is shown separately because it has another target.

Source: Supplemental confusion matrices.

## 22. Appendix: normalized auxiliary matrices

Appendix for questions.

These row-normalized matrices show error proportions for each supplemental procedure. The first-stage head predicts EMG burst halves, not knee movement phase.

Source: Normalized supplemental confusion matrices.

## 23. Appendix: references for movement phase

Appendix for questions.

The cited literature motivates measuring movement phase and controlling kinematics when interpreting EMG. Those studies do not validate our burst-half labels or demonstrate improved fatigue prediction from this architecture.

Source: docs/PHASE_REFERENCES.md.

## 24. Appendix: window lengths and overlap

Appendix for questions.

The historical analysis uses 250 millisecond windows with 50 percent overlap. A separate study with independent references would use 500 milliseconds as its primary window and 250 milliseconds as a sensitivity analysis. At one kilohertz these yield nominal bin spacings of two and four hertz. Longer windows improve nominal frequency spacing; overlap increases temporal sampling but creates dependent observations.

Source: Frozen historical configuration and configs/reference_study.yaml.

## 25. Appendix: filter and window equations

Appendix for questions.

The fourth-order Butterworth analog prototype is transformed to a digital band-pass of order eight and implemented as second-order sections. Coefficients depend on verified sampling rate. Forward-backward filtering is offline. The optional notch is used only if line-noise diagnostics justify it. The historical spectrum uses a one-sided Hann periodogram, not Welch averaging.

Source: SciPy signal documentation: https://docs.scipy.org/doc/scipy/reference/signal.html.

## 26. Appendix: feature equations

Appendix for questions.

The ten input features summarize amplitude, spectrum and relative wavelet energy. The corrected waveform-length feature uses the interval between first and last samples and has a separate schema. It does not replace the frozen historical version or alter its results.

Source: features.py, feature_schema.json and docs/reference_fatigue_protocol.md.

## 27. Appendix: independent-reference architecture

Appendix for questions.

Synchronized angle or reviewed video provides extension and flexion supervision. Calibrated synchronized force or torque provides the approved fatigue label. The same feature matrix feeds the direct model, phase cascade and soft-gated experts. OOF phase probabilities are used in cascade training, and held-out participants are evaluated on matched support. This architecture has not been trained on the supplied ZIP.

Source: docs/reference_fatigue_protocol.md.

## 28. Appendix: label and model equations

Appendix for questions.

Phase labels are derived from angular velocity and an approved deadband. The candidate torque rule requires calibration, repeatability and approval. Direct, cascade and expert equations define three prospective models. The executed auxiliary formula still predicts EMG burst halves and a temporal proxy, not physiological fatigue. Fine-tuning would require compatible external labeled data; reinforcement learning is not suited to static window labels without sequential actions and rewards.

Source: docs/reference_fatigue_protocol.md and historical model registry.

## 29. Appendix: real signal and filter response

Appendix for questions.

The figure shows a one-second interval selected by the first continuous QC-valid window, without checking labels or model results. It includes centered stored ADC counts, SOS output, Hann periodograms before and after filtering, and the digital filter response at the observed rate. This is preprocessing evidence, not physiological performance or a fatigue class.

Source: Private signal_preprocessing_evidence receipt; anonymized figure.
