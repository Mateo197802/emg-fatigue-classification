# Reproduce the exploratory phase experiment

This folder preserves `phase_spike.py` and its protocol verbatim from the completed private experiment. It learns EMG activity-burst halves, not knee extension/flexion. The primary numerical package and trained primary artifact are unchanged.

Use Python 3.12 and install the root hashed environment. Supply authorized data and a completed private primary output root outside this repository:

```powershell
python experiments/phase_proxy/phase_spike.py --source 'C:\private\recordings.zip' --primary 'C:\private\primary' --config configs/study.yaml --out 'C:\private\phase-run-v2'
```

The chosen output directory must be new. It receives the phase support, burst audit, outer/inner splits, head-training audit, selected parameters, 75 model×fold rows, 27,545 second-stage OOF rows and completion checksums. No values are expected to be copied from the paper into a run. The exact 5,509-window/15-group support is a guard for this study, not a generic-data promise.

The stage 1 RF is fixed at 100 trees, depth 3, minimum leaf 5 and balanced weights. Stage 2 logistic C is selected from 0.1/1/10 using grouped validation; its training scores are generated outside each group. This repeats inside inner selection. Random auxiliary labels provide one capacity control, not an exhaustive permutation test.

`evidence.py` verifies completed private inputs, baseline equivalence, target labels, duplicate-free OOF support and completion hashes before generating matrices. It accepts either the new run root directly or the original wrapper with a `run/` subdirectory. The published exporter reconstructs group-held-out stage-1 predictions from the frozen feature table, reproduces their recorded auxiliary metrics and builds the comparison figure itself. It does not require an unprovided private reconstruction script or prebuilt image, and does not retrain the second stage.

```powershell
python experiments/phase_proxy/evidence.py --primary 'C:\private\primary' --experiment 'C:\private\phase-run-v2' --private-out 'C:\private\phase-evidence' --public-dir results/phase_experiment --figures-dir docs/figures
python -m pytest -q
```

Public exports contain whitelisted fold metrics with a persistent private salt, aggregate counts and diagrams. They contain no subject keys, raw data, source-member paths, individual windows or models. The salt is reused from the primary private export. The complete private delivery supplies original source, receipts, reproduction commands, report and slides; the repository supplies code and anonymous aggregate evidence.
