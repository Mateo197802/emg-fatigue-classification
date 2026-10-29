# Execution ledger for the approved PLAN_AUDITABLE_EMG.md

Authority: user approval on 2026-10-28; plan in sibling audit_emg_2026-10-28.

Ruling: create an independent repository on codex/implement-binary-fatigue instead of a linked worktree of the old repository — the approved deliverable is a new repository; preserves the old checkout and avoids importing its historical clutter.

Ruling: execute inline using executing-plans and TDD; use a fresh reviewer at the final review as required by that skill. No per-task delegation.

Ruling: folder-derived subject keys remain explicitly unverified identities — there is no independent identity key in the supplied data. Report held-out recording groups, conditional on one person per folder; never claim independent verification of unseen persons.

Ruling: the RAR remains separate, never silently pooled or called independent external validation. Publication uses the authenticated user's GitHub namespace, not an assumed right to create under the original repository owner's account.

Pre-flight T1→T2→T3→T4: Recording keys/timestamps/counts → per-run arrays/masks → complete windows and weak labels → ordered finite feature table. Time bounds always come from the original recording, before QC.
Pre-flight T4→T5→T6→T7: Feature schema and label vocabulary [0,1] → nested OOF/results → complete pipeline/model contract → reproducible assets. All stages share a frozen configuration digest and private output root.

Ruling: sensitivity analyses use a fixed C=1 RMS+MDF logistic baseline, independently evaluated LOSO in each prespecified variation — avoids selecting a sensitivity winner from outer results, keeps primary nested selection separate and permits a complete sensitivity audit within a reasonable CPU budget. Ablations use grouped inner tuning of logistic C within each feature subset.

Ruling: use a conservative 100 ms edge mask per contiguous run (rather than merely the SOS padding length), frozen before evaluating outcomes. Diagnostic clipping remains a flag, not a fabricated calibrated rail rejection.

T0: protocol frozen in configs/study.yaml before any model results; independent physiological reference remains absent.
T1–T4: core RED 6 failures for missing implementation, then GREEN 6/6. Real ingestion reproduces 140 CSV and 4,256,257 samples; preprocessing yields 138 formal recordings and 5,509 weak-labeled active windows under the frozen primary configuration.

Ruling: jitter status takes precedence over nominal median-rate failure — alternating 0.5/1.5 ms intervals can produce a misleading median rate because the number of intervals is odd; both are invalid but jitter is the primary issue. Covered by the original failing jitter test.

T5: validation RED 5 failures for absent modules, then GREEN 11/11 suite. Group separation, test-only outlier, binary metrics, single-class guard and OOF uniqueness tested.
T6: workflow/model RED 2 failures for absent modules, then GREEN 13/13 suite. Trusted local serialization roundtrip, checksum tampering and unordered fragment reassembly tested.

Primary nested evaluation running. T7 and sensitivity/ablation execution pending.

Final review: independent read-only reviewer identified 5 Important issues. Fix pass: 8 regression cases observed RED, followed by full suite GREEN 25/25. Fixed mixed-run/stale reporting with completion-last output hashes, source-table digests and model/fold/window validation; exact/canonical signal duplicates across folders; stable rejected-interval inference schema; persisted failed inner-search audits and failed-run receipt; bounded RAR subprocess pipes and timeout/reaping. Numerical implementation revision/module hashes added to processing, evaluation and model receipts.

Reviewer did not judge physiological validity, identity, activity truth or acquisition calibration. Ruling: these require evidence absent from the supplied archive and remain explicitly unestablished; software tests cannot resolve them.

Ruling: the full-data model was fitted independently while outer evaluation ran — it was never used for OOF prediction or to choose candidates from outer scores. The final verified model will be refitted after the final evaluation. No performance estimate is assigned to that full-data fit.

Ruling: first sensitivity execution completed all nine variations and three ablations before primary OOF existed; its final side summary failed at the expected dependent read. Numerical outputs were retained, and the final verified run will re-execute under completion/identity guards. No failure was hidden.

Ruling: initial hash generation with pip-tools was replaced by official PyPI release-file hashes for the 33 exact installed distributions — avoids downloading every platform artifact merely to compute hashes. A new environment was installed using --require-hashes and passed the then-current full suite.

Ruling: public model×fold results use whitelist-only opaque fold identifiers with a private persistent salt; source recordings, names, granular OOF and models stay private. This satisfies the results.csv format without copying the private dataset into Git.

Final execution: T0–T6 complete for the declared weak-label scope. T7 numerical assets, report/deck generators and private draft are complete; final artifact hashes/visual checks are recorded privately. All 9 sensitivities and 3 ablations completed under the corrected code. Main result: 90 model×fold rows, 33,054 OOF rows, 15 groups and 5,509 labeled windows. Final/first predicted classes match exactly; maximum score difference 4.44e-16. Real complete CSV versus reassembled fragments differs by at most 2.22e-16 in score; 20 separate RAR recordings processed for inference only. G4 physiological evidence, identity reconciliation, channel calibration and contribution attestation remain explicitly unestablished.
