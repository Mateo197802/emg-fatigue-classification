# Bilingual defense scripts

These read-aloud scripts follow the revised 33-slide defense deck in order. The 19 principal slides are scheduled for 600 seconds total; slides 20–33 are 14 evidence appendices for questions, including the target-free candidate-feature audit.

- [Spanish Markdown script](Guion_defensa_ES.md)
- [Spanish PDF script](pdf/20261109_v7/Guion_defensa_ES.pdf)
- [English Markdown script](Defense_Script_EN.md)
- [English PDF script](pdf/20261109_v7/Defense_Script_EN.pdf)

The scripts distinguish the original early/late proxy, the executed external self-reported perceived-fatigue window experiment, and a separate trial-level task-outcome benchmark. They do not present self-report as force loss, and they do not claim observed extension/flexion. The phase-to-fatigue cascade remains unevaluated because synchronized movement-phase labels are absent.

The public script PDFs are regenerated from these Markdown files with `tools/build_speaking_script_pdfs.py`. Metadata lines for timing and sources are removed from narration before rendering, so each source appears once. A parser regression test is in `tests/test_speaking_script_pdfs.py`. The private package contains the full slide deck and private signal evidence.
