"""Verify saved trial-level metrics, grouped splits, and checkpoint hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.reference_fatigue.figshare_trial_benchmark import metric_summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(output_dir: Path, processed_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve()
    processed_dir = Path(processed_dir).resolve()
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output_dir / "oof_predictions.csv")
    saved_metrics = pd.read_csv(output_dir / "benchmark_results.csv").set_index("model")
    saved_confusion = pd.read_csv(output_dir / "confusion_matrix.csv")
    split_manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    errors = []
    if manifest.get("status") != "completed":
        errors.append(f"run status is {manifest.get('status')!r}")
    expected_subject_count = int(manifest.get("participant_count", -1))
    if expected_subject_count != 14:
        errors.append(f"expected 14 participants, found {expected_subject_count}")
    if len(split_manifest.get("outer_folds", [])) != 14:
        errors.append("outer split manifest does not contain 14 folds")
    if predictions.duplicated(["model", "trial_id"]).any():
        errors.append("duplicate model/trial OOF rows found")
    if predictions.groupby("model").size().ne(28).any():
        errors.append("each model must have exactly 28 OOF trial predictions")
    if predictions.groupby("model")["subject_id"].nunique().ne(14).any():
        errors.append("each model must cover all 14 held-out participants")

    checked_metrics = 0
    for model, group in predictions.groupby("model", sort=True):
        if model not in saved_metrics.index:
            errors.append(f"saved metrics missing model {model}")
            continue
        recomputed = metric_summary(group["y_true"], group["y_pred"], group["score"])
        for key in ("accuracy", "f1_macro", "balanced_accuracy", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"):
            expected = float(saved_metrics.loc[model, key])
            actual = float(recomputed[key])
            if not (np.isnan(expected) and np.isnan(actual)) and not np.isclose(expected, actual, rtol=0, atol=1e-12):
                errors.append(f"metric mismatch for {model}.{key}: saved={expected} replay={actual}")
        checked_metrics += 1

        saved_cells = saved_confusion[saved_confusion["model"] == model]
        matrix = pd.crosstab(group["y_true"], group["y_pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        for row in saved_cells.itertuples(index=False):
            actual = int(matrix.loc[int(row.true_label), int(row.predicted_label)])
            if actual != int(row.count):
                errors.append(f"confusion count mismatch for {model}: {row.true_label}->{row.predicted_label}")

    for fold in split_manifest.get("outer_folds", []):
        held_out = str(fold["held_out_subject"])
        training = set(map(str, fold["training_subjects"]))
        if held_out in training:
            errors.append(f"outer leakage: {held_out} appears in its own training set")
        if len(fold.get("inner_folds", [])) != 3:
            errors.append(f"fold {held_out} does not list three inner grouped folds")
        for inner in fold.get("inner_folds", []):
            inner_train = set(map(str, inner["train_subjects"]))
            inner_valid = set(map(str, inner["validation_subjects"]))
            if inner_train & inner_valid:
                errors.append(f"inner group leakage in outer fold {held_out}")
            if held_out in inner_train | inner_valid:
                errors.append(f"outer test participant leaked into inner fold {held_out}")

    checkpoint_count = 0
    for record in manifest.get("checkpoint_receipts", []):
        checkpoint = processed_dir / record["path"]
        if not checkpoint.is_file():
            errors.append(f"checkpoint missing: {record['path']}")
            continue
        if sha256(checkpoint) != record["sha256"]:
            errors.append(f"checkpoint hash mismatch: {record['path']}")
        checkpoint_count += 1

    artifact_hashes = {}
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"artifact hash mismatch: {name}")
        else:
            artifact_hashes[name] = expected
    result = {
        "status": "passed" if not errors else "failed",
        "trial_count": int(predictions["trial_id"].nunique()),
        "participant_count": int(predictions["subject_id"].nunique()),
        "model_count": int(predictions["model"].nunique()),
        "replayed_metric_tables": checked_metrics,
        "checkpoint_count_checked": checkpoint_count,
        "artifact_hashes_checked": len(artifact_hashes),
        "errors": errors,
    }
    (output_dir / "verification_receipt.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "reference_fatigue" / "figshare_task_level",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=REPO_ROOT / "private" / "figshare_task_failure_2021" / "processed",
    )
    args = parser.parse_args()
    result = verify(args.output_dir, args.processed_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
