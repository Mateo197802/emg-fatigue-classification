"""Create a model-keyed copy of the private neural inner-OOF artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_fatigue.weak_supervision.evaluation_neural import _load_fold_plan
from emg_fatigue.weak_supervision.evaluation_neural_nested import _sha256_file


def repair(stage_dir: Path, threshold_parent_dir: Path) -> Path:
    stage_dir = Path(stage_dir).expanduser().resolve()
    threshold_parent_dir = Path(threshold_parent_dir).expanduser().resolve()
    manifest_path = stage_dir / "neural_supervised_stage_manifest.json"
    source_path = stage_dir / "inner_oof_selection_predictions.parquet"
    output_path = stage_dir / "inner_oof_selection_predictions_with_model.parquet"
    active_path = stage_dir / "active_lowfreq_signal_window_manifest.parquet"
    fold_path = threshold_parent_dir / "fold_manifest.json"
    for path in (manifest_path, source_path, active_path, fold_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_order = tuple(str(value) for value in manifest.get("models", ()))
    if not model_order or len(set(model_order)) != len(model_order):
        raise ValueError("neural stage receipt has no unique model order")
    expected_source_hash = manifest.get("output_sha256", {}).get(str(source_path))
    if not expected_source_hash or _sha256_file(source_path) != expected_source_hash:
        raise ValueError("unkeyed inner-OOF input checksum differs from its stage receipt")
    source = pd.read_parquet(source_path)
    if "model_name" in source:
        raise ValueError("source inner-OOF artifact already has model_name")
    if source.outer_fold_id.isna().any() or source.inner_fold_id.isna().any():
        raise ValueError("source inner-OOF rows contain missing fold IDs")
    active = pd.read_parquet(active_path)
    plan = _load_fold_plan(fold_path)
    model_parts: list[pd.DataFrame] = []
    for outer in plan.outer_folds:
        outer_positions = source.index[source.outer_fold_id.astype(int).eq(int(outer.fold_id))].to_numpy()
        if outer_positions.size == 0:
            raise ValueError(f"inner-OOF artifact has no rows for outer fold {outer.fold_id}")
        if not (outer_positions == np.arange(outer_positions[0], outer_positions[-1] + 1)).all():
            raise ValueError(f"inner-OOF rows for outer fold {outer.fold_id} are not contiguous")
        outer_rows = source.loc[outer_positions].reset_index(drop=True)
        expected_keys: list[str] = []
        expected_lookup: dict[str, tuple[str, str, int]] = {}
        for inner in plan.inner_folds[outer.fold_id]:
            validation = active.loc[
                active.group_id.astype(str).isin(set(inner.validation_group_ids))
                & active.class_id.isin((0, 1))
            ].sort_values("window_key")
            expected_keys.extend(validation.window_key.astype(str).tolist())
            expected_lookup.update(
                {
                    str(row.window_key): (str(row.group_id), str(row.recording_id), int(row.class_id))
                    for row in validation.itertuples()
                }
            )
        expected_count = len(expected_keys)
        if len(outer_rows) != expected_count * len(model_order):
            raise ValueError(f"outer fold {outer.fold_id} row count cannot be split into the recorded model order")
        for model_index, model_name in enumerate(model_order):
            part = outer_rows.iloc[model_index * expected_count : (model_index + 1) * expected_count].copy()
            if part.window_key.astype(str).tolist() != expected_keys:
                raise ValueError(f"outer fold {outer.fold_id} model block does not match its ordered inner OOF windows")
            if part.window_key.astype(str).duplicated().any():
                raise ValueError(f"outer fold {outer.fold_id} contains duplicate windows inside a model block")
            if part.inner_fold_id.isna().any() or not part.outer_fold_id.astype(int).eq(int(outer.fold_id)).all():
                raise ValueError(f"outer fold {outer.fold_id} model block has inconsistent fold IDs")
            for row in part.itertuples():
                expected = expected_lookup[str(row.window_key)]
                observed = (str(row.group_id), str(row.recording_id), int(row.true_class_id))
                if observed != expected:
                    raise ValueError("inner-OOF group, trial, or label differs from the frozen label join")
            part["model_name"] = model_name
            model_parts.append(part)
    keyed = pd.concat(model_parts, ignore_index=True)
    keyed.to_parquet(output_path, index=False)
    manifest.setdefault("output_sha256", {})[str(output_path)] = _sha256_file(output_path)
    if "cerqueira_supervised_finetuned" in model_order:
        transfer_adapter = Path(__file__).resolve().parents[1] / "src" / "emg_fatigue" / "weak_supervision" / "external_transfer.py"
        manifest.setdefault("source_sha256", {})[str(transfer_adapter)] = _sha256_file(transfer_adapter)
    manifest["inner_oof_model_id_repair"] = {
        "repair_script_sha256": _sha256_file(Path(__file__)),
        "source_artifact": source_path.name,
        "source_sha256": expected_source_hash,
        "keyed_copy": output_path.name,
        "keyed_copy_sha256": _sha256_file(output_path),
        "model_order_from_training_loop": list(model_order),
        "verification": "each model block matches the expected ordered inner-validation keys, groups, trials, labels, and outer fold",
        "prediction_values_changed": False,
        "reserved_inputs_opened": False,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: repair_neural_inner_oof_model_ids.py STAGE_DIR THRESHOLD_PARENT_DIR")
    result = repair(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Created verified model-keyed inner OOF copy: {result}")
