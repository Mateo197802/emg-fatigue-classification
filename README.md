# Binary sEMG fatigue proxy classification

## Supplemental two-stage phase experiment

The completed exploratory cascade did **not show a clear improvement**. On the same 5,509 windows and 15 held-out groups, direct ten-feature logistic regression obtained macro-F1 **0.6355 ± 0.1262**; the phase-conditioned variant obtained **0.6293 ± 0.1330**. Paired difference: **−0.0062**, 95% group-bootstrap interval **[−0.0232, +0.0082]**. The original primary model remains unchanged.

Stage 1 is a separate random forest learning **first/second EMG activity-burst halves**, not verified knee extension/flexion. Stage 2 is a logistic classifier using EMG features and the stage 1 score to predict the weak early/late target. Both stages use grouped cross-fitting inside nested validation. No independent physiological fatigue reference is introduced.

```mermaid
flowchart LR
    X[Ten EMG features] --> P[Auxiliary burst-half RF]
    X --> L[Second-stage logistic]
    P --> S[Group-out-of-fold phase score]
    S --> L
    L --> Y[Early or late proxy]
```

![Experiment architecture](docs/figures/phase_architecture.png)

See the **[full supplemental report and model relationships](docs/PHASE_EXPERIMENT.md)**, [commented references](docs/PHASE_REFERENCES.md), [all 12 target-specific confusion matrices](results/phase_experiment/confusion_matrices.csv), [aggregate/fold results](results/phase_experiment), [original experiment source and reproduction instructions](experiments/phase_proxy), and [evidence tests](tests/test_phase_evidence.py). Absolute and row-normalized matrices for all six primary procedures, five supplemental variants and the auxiliary head are shown in the report. Full OOF, manuscript, slide deck and models remain in the updated private delivery.

The supplemental `baseline_10` reproduces primary `logistic_all`; it is different from the primary nested selector (macro-F1 0.6240). The auxiliary RF, primary RF family and final serialized RF are different fits and roles, explicitly related in the model map. This experiment does not establish verified extension recognition, physiological fatigue or robustness to unseen acquisition domains.

This project implements offline binary classification of early and late candidate-active windows from the intended rectus femoris knee-extension study. Labels are a **weak elapsed-time approximation** of nonfatigue/fatigue. The supplied recordings do not include an independent synchronized fatigue reference. Consequently, the reported scores establish agreement with this operational label, not validated physiological fatigue, force loss, or fatigue onset.

The original exploratory project is retained separately: [proyecto-biomecanica-emg](https://github.com/EthanGL1/proyecto-biomecanica-emg/tree/772b7a39c18ecfd34bfbd5f20fa0a14d2ddfa003). This repository implements a new binary task; its metrics must not be substituted for the old three-class scores or for Borg/Cerqueira results from other tasks.

## What runs

- CSV/ZIP/RAR readers, input hashes, clock and signal QC, gap-preserving filtering and explicit abstention.
- Candidate activity segmentation followed by complete 250 ms windows with 50% overlap.
- Amplitude, spectral and wavelet features; mandatory majority and RMS+MDF logistic baselines.
- Outer leave-one-recording-group-out evaluation with grouped inner tuning. Both sides and all series of a group stay together. Group-to-person identity is an explicit, unverified assumption of the source folder structure.
- Model serialization, checksum verification, reloading and prediction using exactly the training feature schema.
- Prespecified sensitivity analyses, feature-family ablations, numerical tables and figures generated from frozen out-of-fold outputs.

## Installation

Python 3.12. Use the exact tested distributions and release-file SHA256 hashes:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q
```

The lock records the tested Windows environment and includes official PyPI release hashes for alternative compatible wheels. CI checks a clean Linux environment separately. `requirements.in` is the frozen distribution list used to produce this lock, not an invitation to upgrade dependencies during reproduction.

## Reproduce with authorized course data

Keep the original archive and all derived outputs **outside this checkout**. Set one input and one output root:

```powershell
$env:EMG_DATA_ZIP = 'C:\authorized-private-storage\recordings.zip'
$env:EMG_OUTPUT_ROOT = Join-Path $env:LOCALAPPDATA 'EMG_Biomecanica\binary-study'
.\.venv\Scripts\python.exe -m emg_fatigue features --config configs/study.yaml
.\.venv\Scripts\python.exe -m emg_fatigue evaluate --config configs/study.yaml
.\.venv\Scripts\python.exe -m emg_fatigue fit-final --config configs/study.yaml
.\.venv\Scripts\python.exe -m emg_fatigue sensitivities --config configs/study.yaml
.\.venv\Scripts\python.exe -m emg_fatigue report --config configs/study.yaml
```

`features` generates upstream inventory, preprocessing, labels and feature products together. The separate inventory/preprocess/labels commands are available for inspection. `evaluate` and `fit-final` reject a cache whose configuration or byte hash differs from its processing receipt. `fit-final` performs grouped inner selection on the full development cohort; its fit is not another held-out performance estimate. Use the nested selection procedure's OOF result to describe selection performance.

The primary archive layout is one folder per recording group, with CSV filenames containing an unambiguous `R0`–`R5` and `dr`/`iz`. CSV columns are exactly `Relative_Time_s,ADC_Value[,Voltage_mV]`. `Voltage_mV` is not used: its conversion does not establish physical calibration. R0 controls remain out of formal training.

The additional RAR format `Pxx_set01_dr.csv` is supported for separate processing. A different filename is not evidence of a different person. Do not pool it with the primary cohort or call it external validation without reconciling identity and session provenance.

## Prediction

```powershell
.\.venv\Scripts\python.exe -m emg_fatigue predict --config configs/study.yaml --input 'C:\private\recording.csv' --model "$env:EMG_OUTPUT_ROOT\model.joblib"
```

Or replace `--input` with `--fragments part1.csv part2.csv part3.csv`. Fragments retain original timestamps, are ordered and reassembled before filtering; overlap and clock resets are rejected. This is offline reassembly, not streaming inference.

Prediction loads the bundle's processing contract. Valid candidate-active windows in the trained early/late support receive a class and a score. Quality failures, undetected activity, invalid features and the excluded transition receive distinct abstention reasons. Native probabilities are marked uncalibrated; SVM scores are sigmoid-transformed decision margins, not calibrated probabilities. Load only a trusted locally produced joblib bundle: its checksum receipt provides integrity, not authenticity.

## Public synthetic smoke test

```powershell
.\.venv\Scripts\python.exe -m emg_fatigue demo --output-dir "$env:TEMP\emg-synthetic-demo"
```

This exercises signal processing, grouped evaluation, serialization and figures without course data. The signals are deterministic software fixtures, not a physiological EMG simulation, and their scores do not describe real fatigue detection.

## Report and evidence

The frozen numerical comparison is in [docs/RESULTS.md](docs/RESULTS.md), with the [model card](docs/MODEL_CARD.md). The nested selector's observed macro-F1 does not clearly improve on the mandatory RMS+MDF logistic baseline.

Every numerical table and figure has a source mapping in private `report_assets/table_sources.json`. `results.csv` in private output has one row per model × outer fold, all required metrics, support and confusion counts. Public `results/results.csv`, when present, contains only whitelisted fold metrics with opaque IDs; it contains no subject identity, paths, signal windows or granular predictions. The private export mapping is not published.

The complete editable manuscript follows the supplied template and remains private with the course data. The report generator takes the template locally; this repository does not redistribute the university template, participant data or the previous private manuscript. Ten references with DOI metadata and bounded reading notes are in `references/verified_references.json`.

See [the protocol](docs/protocol.md), [the data dictionary](docs/data_dictionary.md), [discrepancy resolution](docs/discrepancy_resolution.md), [privacy boundaries](docs/privacy.md) and [the execution ledger](docs/EXECUTION_LEDGER.md).
