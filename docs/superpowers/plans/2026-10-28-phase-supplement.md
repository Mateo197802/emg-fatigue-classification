# Phase experiment integration implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. The user authorized integration, GitHub publication and the updated private delivery.

**Goal:** Integrate the completed phase experiment, all model relationships and auditable evidence into public documentation and private academic artifacts.

**Architecture:** Preserve primary numerical results and the completed exploratory run. Derive aggregate confusion matrices, comparative metrics and diagrams from checksum-verified OOF files. Publish code and anonymous aggregate outputs; keep raw data, models, identities and granular predictions private.

**Tech Stack:** Existing Python 3.12 numerical environment, matplotlib, bundled python-docx, artifact-tool JS, LibreOffice PDF conversion, GitHub Actions.

**Spec:** User request in this chat: supplement, report, presentation, delivery package, GitHub, README, references, tests, confusion matrices and architecture relating all models.

## Global constraints

- No numerical retraining or retrospective change to the phase protocol.
- First stage predicts EMG burst half, not verified extension/flexion. Second stage predicts the weak early/late target.
- Six primary procedures, five supplemental variants and one auxiliary phase head retain separate target and support labels.
- Preserve ten principal slides and 600 s planned narration; add clearly identified reserve evidence slides.
- Preserve course template sections and private university branding. Do not fabricate physiological evidence, identity or contribution attestations.
- Publication is explicitly authorized; no raw signals, per-window OOF, group identities or models enter Git.

## Review focus

- Confusion matrices must come from OOF counts with exact support and correct row normalization.
- Auxiliary burst-half reference must never be labeled true extension or physiological fatigue.
- Baseline_10 must match primary logistic_all; nested_selector is a different selection procedure.
- All published folds must use the existing private opaque mapping; no private file paths or identities may leak.
- Exported manuscript, slide count, hashes, archive contents and exact remote-head CI must agree.

## Tasks

- [x] T1 Add RED tests for OOF validation and confusion/count exports; implement evidence generator and preserve original experiment source/protocol.
- [x] T2 Generate private and public aggregate evidence, model map, architecture, references and README links; verify numerical equivalence and export privacy.
- [ ] T3 Integrate main report sections plus detailed appendices, and ten principal slides plus reserve evidence slides. Render and inspect all changed pages/slides.
- [ ] T4 Run complete software checks and independent branch review; resolve material findings in one tested fix pass. Commit, publish and verify exact-head CI.
- [ ] T5 Build a new complete private delivery with old numerical evidence plus the new supplement and updated code/artifacts. Verify ZIP CRC and every SHA256; preserve old releases.

## Execution ledger

Ruling: native create_worktree cannot address this nested repository because its sourceRoot allowlist accepts only the parent workspace, which is not the repository. A normal Git worktree was therefore created in the workspace on codex/integrate-phase-supplement, preserving the original checkout and published release.

Ruling: no experiment is rerun or tuned for a better score; this task integrates and verifies completed evidence. Documentation does not promote the experimental model to the primary artifact.

T1–T2: seven expected RED cases for the absent evidence exporter, then all evidence tests GREEN. Added separate RED→GREEN cases for self-referential receipts and Windows CSV line endings. Public CSVs are normalized to LF before hashing so a Git checkout preserves receipt validity. Baseline equivalence, original completion hashes and auxiliary reconstruction metrics verified. All source data and model artifacts remain private.
