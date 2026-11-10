# Execution Ledger: Raw First sEMG Fatigue

Base: `a50ba77a28ae3fb8fabad8b932f580334c717c6d` (`main`)

## Rulings

- Ruling: Use the supplied `OneDrive_1_10-10-2026.zip` as the authorized training source because the expected-named ZIP is absent and this archive has the specified 140 CSVs, 15 folders, and exact schema — this is the only training archive supplied — cost if wrong: all run results would characterize a different cohort than intended.
- Ruling: Map the non-anonymized source-folder names to opaque `G01`–`G15` aliases in memory and never write the mapping into public outputs — this preserves source-group separation while reducing identifier exposure — cost if wrong: the alias order could complicate source traceability or misrepresent a changed archive.
- Ruling: Treat source folders as grouping units only and use a 12/3 group holdout for self-supervised validation — this follows the objective while identity remains unverified — cost if wrong: group overlap with actual people could make validation less independent than described.
- Ruling: No independent fatigue or phase references are present in the supplied training ZIP, so no supervised target, classifier metric, or confusion matrix will be created — the archive has only numeric signal columns — cost if wrong: a valid external reference source might exist outside the authorized input contract and would remain unused.
- Ruling: Use the supplied `raw.rar` for the final-only stage although the objective names `raw(2).rar`; the supplied archive was the only listed RAR and passed integrity verification — cost if wrong: the external exploratory file analysis would concern a different archive than intended.
- Ruling: Treat the 20 external CSVs as one opaque source group and do not claim independent validation — their paths expose one grouping level, but no verified participant mapping or exposure history exists — cost if wrong: independence may be either stronger or weaker than the evidence supports.
- Ruling: Correct the UnRAR adapter after the first post-freeze listing returned zero members because `-inul` suppressed listing output; preserve that initial audit privately and rerun with output enabled — cost if wrong: an empty archive would have been falsely reported and no external signal analysis would occur. The training freeze and encoder hashes were unchanged.
- Ruling: Round effective sampling-rate values to 0.001 Hz for plotting and disable scientific axis offsets — the observed rates differ only at floating-point noise scale around 1000 Hz — cost if wrong: sub-millihertz variation would be visually hidden, while the exact values remain in private audits and aggregate numeric summaries.

## Task records

- Task 1: complete. The supplied training ZIP passed archive/CSV parsing; 140 records, 15 folder groups, exact three-column schema, deterministic opaque aliases, and private source hashes were recorded. The expected archive filename was not supplied.
- Task 2: complete. All 140 files had usable timestamp-derived rates centered at 1000 Hz; 20–400 Hz filtering and no notch were selected from the audit; ADC/voltage fit and contamination summaries were produced. No primary windows were rejected. Aggregate plots were visually inspected, and their floating-point axis offsets were corrected.
- Task 3: complete. Filtered segments preserved discontinuities; 500 ms, 250 ms, and 1000 ms windows produced 16,629, 33,649, and 8,128 valid windows respectively. Temporal, spectral, band-power, Hjorth, and wavelet features, train-only scaling, and sensitivity outputs were recorded.
- Task 4: complete. Independent synchronized fatigue/phase labels were absent. A deterministic 12-development/3-validation source-group split was used, with no targets or supervised metrics.
- Task 5: complete. Clean and augmentation-trained masked-reconstruction CNNs each trained on 13,333 windows and validated on 3,296. Best epoch was 11; validation losses were 1.0254296 (clean) and 1.0189225 (augmented). Checkpoint hashes are in `results/raw_first/ssl_training_summary.csv` and the private manifest.
- Task 6: complete. Ten corruption families were assessed at 5–50% on 255 matched held-out windows. Median family-level embedding cosine at 20% was 0.9968 (clean) and 0.9976 (augmented); at 50% it was 0.9764 and 0.9797. Baseline shift/wander at 50% was the most disruptive family (0.7432 and 0.8128 respectively). These are representation stability measurements, not classification robustness. Repeated inference and checkpoint round-trip were exact for both encoders.
- Task 7: complete locally. The post-freeze RAR integrity check passed; corrected UnRAR parsing found 20/20 schema-compatible CSVs in one opaque group and produced 2,763 exploratory windows. No independent references were found; participant identity independence and prior exposure remain unverified. Private hashes and outputs are recorded in the local run manifest. The branch has local commits; push/PR was not available because repository permission was read-only (`push:false`).

## Execution record

- Local run directory: `%LOCALAPPDATA%/EMG_Biomecanica/raw-first-fatigue-v2/20261110T080624Z_0723c36b27` (contains restricted inputs, per-file tables, hashes, checkpoints, and embeddings).
- Public aggregate outputs: `results/raw_first/` only; no original source-folder names, per-file hashes, per-file feature rows, checkpoints, raw signals, or participant-level embeddings were committed.
- No software tests were added or run. The requested full data audit/training, final post-freeze RAR inspection, manifest hash verification, deterministic inference check, and source compilation check were used as execution evidence.
- Commits before final review: `5488f7f` (pipeline), `81ab318` (safe-loadable checkpoint metadata), `634679c` (UnRAR adapter), `5c455a9` (preserve UnRAR output). Later summary/plot wording changes are pending a final local commit.
