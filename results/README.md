# Evaluation outputs

Private `evaluation/results.csv` has one model×fold row with full numerical metrics and a private group key. Public `results.csv` is a whitelisted export with opaque fold IDs, without signals, identities or paths. See docs/privacy.md. Metrics describe weak elapsed-time labels, not independent physiological fatigue detection.

`metrics_summary.csv` gives mean, sample SD and group-bootstrap interval per model and metric. It must be read with the label definition, retained-window coverage and held-out-group assumption.
