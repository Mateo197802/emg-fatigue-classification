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

Historical checkpoint: primary nested evaluation was running and T7/sensitivity/ablation work was pending. This checkpoint is superseded by the final execution record below.

Final review: independent read-only reviewer identified 5 Important issues. Fix pass: 8 regression cases observed RED, followed by full suite GREEN 25/25. Fixed mixed-run/stale reporting with completion-last output hashes, source-table digests and model/fold/window validation; exact/canonical signal duplicates across folders; stable rejected-interval inference schema; persisted failed inner-search audits and failed-run receipt; bounded RAR subprocess pipes and timeout/reaping. Numerical implementation revision/module hashes added to processing, evaluation and model receipts.

Reviewer did not judge physiological validity, identity, activity truth or acquisition calibration. Ruling: these require evidence absent from the supplied archive and remain explicitly unestablished; software tests cannot resolve them.

Ruling: the full-data model was fitted independently while outer evaluation ran — it was never used for OOF prediction or to choose candidates from outer scores. The final verified model will be refitted after the final evaluation. No performance estimate is assigned to that full-data fit.

Ruling: first sensitivity execution completed all nine variations and three ablations before primary OOF existed; its final side summary failed at the expected dependent read. Numerical outputs were retained, and the final verified run will re-execute under completion/identity guards. No failure was hidden.

Ruling: initial hash generation with pip-tools was replaced by official PyPI release-file hashes for the 33 exact installed distributions — avoids downloading every platform artifact merely to compute hashes. A new environment was installed using --require-hashes and passed the then-current full suite.

Ruling: public model×fold results use whitelist-only opaque fold identifiers with a private persistent salt; source recordings, names, granular OOF and models stay private. This satisfies the results.csv format without copying the private dataset into Git.

## Professor-feedback correction and reference-data audit (2026-11-09)

The historical weak elapsed-time analysis remains unchanged at 250 ms and 50% overlap. It predicts the historical temporal proxy, not independently measured fatigue. The burst-half auxiliary task predicts the first/second half of candidate EMG activity bursts, not verified extension/flexion.

The read-only source audit receipt is `audit_emg_2026-10-28/reference_data_recovery_2026-11-09.json`. It records the main ZIP SHA-256 `2c4f50e5c826b001778eb5b8fa6df0e397c0c4d210f5a257ff6820ab0cece354`, the nested EMG ZIP SHA-256 `0723c36b27a6d5b7258d5f29c26de7898a83261011b8f323ccd42e5c5a5b2776`, and RAR SHA-256 `15b61b870269a95da894d3225cc43e95fb2465eb0ba80107664bbf015f799e75`. The audit found no calibrated time-resolved force/torque, synchronized knee-angle/video, repetition events, MVC/Borg records or independent identity/session key.

A private 160-row source crosswalk was generated at `audit_emg_2026-10-28/reference_source_crosswalk_2026-11-09.csv`; its SHA-256 is `2c108ada3a9fdc1f293ce22b05daa8affabc44cb4b660347431dcf6999fccf09`. It maps 140 ZIP recordings and 20 RAR recordings to source/member hashes, marks identity verification false for every row, and leaves cross-archive links unresolved. No subject identities were inferred by comparing filename codes.

Ruling: Do not freeze an extension/flexion sign or angular-velocity deadband without the actual calibrated kinematic sensor, its sign convention, noise/resolution estimate, and synchronization residual — the archive contains no such measurements, so any fixed direction would be invented — cost if wrong: all phase labels could be reversed or displaced in time.

Ruling: Keep the reference-backed 500 ms/50% windowing configuration distinct from the frozen 250 ms historical proxy; compute sample count, hop and nominal bin spacing from each verified recording rate — the corpus rate is close to 1 kHz but not assumed exact — cost if wrong: coefficient/feature resolution and phase alignment would differ from the stated method.

Ruling: Candidate torque/MDC95/persistence labels are accepted by code only for clearly synthetic test fixtures unless a versioned label definition and written biomechanics approval receipt are supplied — repeatability, matched-angle boundaries and onset semantics cannot be verified from sEMG-only files — cost if wrong: waiting for approval could delay an otherwise valid model, while proceeding without it could label measurement noise as fatigue.

Task 0 update: G0 failed for both physiological fatigue and verified phase. True-label model fitting and associated Accuracy/F1 comparison are therefore unavailable; work continues on label-contract validation, signal/window/feature engineering, plots, formulas, documentation and delivery without producing physiological performance claims.

The 2026-11-09 correction package adds strict reference-label and window contracts with abstention, a private source crosswalk, a separately versioned corrected feature schema, a distinct 500/250 ms reference-window design, formula documentation, and bilingual proxy/target architecture diagrams. These additions do not change the frozen historical predictions or create physiological labels.

At the pre-delivery source snapshot, 89 tests passed; `git diff --check HEAD` passed; the public-tree audit scanned 125 tracked paths against the supplied archive and found no restricted artifact types, exact source archive member paths or credential patterns. At that historical checkpoint the branch had not yet been pushed; the final publication and artifact verification are recorded below.

Final execution: T0–T6 complete for the declared weak-label scope. T7 numerical assets, report/deck generators and private draft are complete; final artifact hashes/visual checks are recorded privately. All 9 sensitivities and 3 ablations completed under the corrected code. Main result: 90 model×fold rows, 33,054 OOF rows, 15 groups and 5,509 labeled windows. Final/first predicted classes match exactly; maximum score difference 4.44e-16. Real complete CSV versus reassembled fragments differs by at most 2.22e-16 in score; 20 separate RAR recordings processed for inference only. G4 physiological evidence, identity reconciliation, channel calibration and contribution attestation remain explicitly unestablished.

Final delivery state (2026-11-09): source branch `codex/implement-binary-fatigue` was published at the revision recorded in `COMPROBANTE_PAQUETE.json`; exact-head GitHub Actions passed. The 27-page report, 29-slide deck, and Spanish/English speaker-script PDFs (7/6 pages) were rebuilt against that revision, rendered and visually inspected. The private package has 990 ZIP members, CRC passed, 989 hashed members, and its source-code manifest and 43 visible root files passed cross-checks. The prior package ZIP remains byte-identical (SHA-256 `c98e8dd69dabb691da2bd3bede4cbb81a829e52017540a254f77c823e967636d`). The package verifier returned `verified`; the local suite passed 89 tests, and the public-tree audit found no restricted artifacts, source archive paths or credential patterns in 125 tracked paths. These checks establish software, provenance and delivery integrity only. G0 remains failed: no independently labeled physiological fatigue or verified extension/flexion model was fitted, and weak-label proxy metrics must not be presented as fatigue-detection performance.
