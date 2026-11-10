# Execution Ledger: Raw First sEMG Fatigue

Base: `a50ba77a28ae3fb8fabad8b932f580334c717c6d` (`main`)

## Rulings

- Ruling: Use the supplied `OneDrive_1_10-10-2026.zip` as the authorized training source because the expected-named ZIP is absent and this archive has the specified 140 CSVs, 15 folders, and exact schema — this is the only training archive supplied — cost if wrong: all run results would characterize a different cohort than intended.
- Ruling: Map the non-anonymized source-folder names to opaque `G01`–`G15` aliases in memory and never write the mapping into public outputs — this preserves source-group separation while reducing identifier exposure — cost if wrong: the alias order could complicate source traceability or misrepresent a changed archive.
- Ruling: Treat source folders as grouping units only and use a 12/3 group holdout for self-supervised validation — this follows the objective while identity remains unverified — cost if wrong: group overlap with actual people could make validation less independent than described.
- Ruling: No independent fatigue or phase references are present in the supplied training ZIP, so no supervised target, classifier metric, or confusion matrix will be created — the archive has only numeric signal columns — cost if wrong: a valid external reference source might exist outside the authorized input contract and would remain unused.

## Task records

- Task 1: pending.
- Task 2: pending.
- Task 3: pending.
- Task 4: pending.
- Task 5: pending.
- Task 6: pending.
- Task 7: pending.
