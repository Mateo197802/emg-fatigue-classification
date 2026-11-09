# Bilingual defense scripts

These read-aloud scripts follow the current 30-slide defense deck in order. The 16 principal slides are scheduled for 600 seconds total; slides 17–30 are 14 evidence appendices for questions, including the target-free candidate-feature audit.

- [Spanish Markdown script](Guion_defensa_ES.md)
- [Spanish PDF script](pdf/20261109_v5/Guion_defensa_ES.pdf)
- [English Markdown script](Defense_Script_EN.md)
- [English PDF script](pdf/20261109_v5/Defense_Script_EN.pdf)

The scripts describe the executed early/late temporal-proxy experiments and the auxiliary burst-half cascade. They do not claim that either target is physiological fatigue or observed extension/flexion. The proposed reference-backed architectures are clearly described as not trained on the supplied ZIP.

The current PDFs are versioned under `pdf/20261109_v5/` and were generated from the Markdown with `tools/build_speaking_script_pdfs.py`. Metadata lines for timing and sources are removed from narration before rendering, so each source appears once. A parser regression test is in `tests/test_speaking_script_pdfs.py`. Only these two read-aloud PDFs are public; the report, slide deck and signal evidence stay in the private delivery.
