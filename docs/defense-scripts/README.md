# Bilingual defense scripts

These read-aloud scripts follow the current 29-slide defense deck in order. The 16 principal slides are scheduled for 600 seconds total; slides 17–29 are evidence appendices for questions.

- [Spanish Markdown](Guion_defensa_ES.md)
- [English Markdown](Defense_Script_EN.md)

The matching PDF exports are included in the private standalone script ZIP, not in this public source tree.

The scripts describe the executed early/late temporal-proxy experiments and the auxiliary burst-half cascade. They do not claim that either target is physiological fatigue or observed extension/flexion. The proposed reference-backed architectures are clearly described as not trained on the supplied ZIP.

The PDFs were generated from the Markdown with `tools/build_speaking_script_pdfs.py`. Metadata lines for timing and sources are removed from narration before rendering, so each source appears once. A parser regression test is in `tests/test_speaking_script_pdfs.py`.
