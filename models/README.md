# Private model artifact

Run `python -m emg_fatigue fit-final --config configs/study.yaml` with EMG_OUTPUT_ROOT pointing outside this checkout. It writes model.joblib and its receipt, then verifies recarga against the original fitted scores. The artifact predicts a weak temporal label and is not an independently validated physiological fatigue detector.

No trained derivative of the course dataset is redistributed here. Load only locally trusted artifacts; a checksum is not proof that a third-party pickle is safe.
