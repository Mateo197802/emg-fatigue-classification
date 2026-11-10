import pytest

torch = pytest.importorskip("torch")
import numpy as np

from emg_fatigue.original_emg_ssl import (
    MaskedConvAutoencoder,
    extract_fixed_windows,
    masked_reconstruction_loss,
    random_block_mask,
    split_groups,
)


def test_mask_uses_deterministic_contiguous_blocks():
    generator = torch.Generator().manual_seed(7)
    mask = random_block_mask(
        batch_size=2,
        n_samples=500,
        block_size=25,
        mask_fraction=0.25,
        generator=generator,
    )

    assert mask.shape == (2, 1, 500)
    assert mask.dtype == torch.bool
    assert torch.all(mask.sum(dim=(1, 2)) == 125)
    assert torch.equal(mask, random_block_mask(
        batch_size=2,
        n_samples=500,
        block_size=25,
        mask_fraction=0.25,
        generator=torch.Generator().manual_seed(7),
    ))


def test_reconstruction_loss_ignores_unmasked_samples():
    prediction = torch.full((1, 1, 4), 2.0)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[True, True, False, False]]])

    assert masked_reconstruction_loss(prediction, target, mask).item() == pytest.approx(4.0)


def test_encoder_reconstructs_signal_shape_and_returns_embedding():
    model = MaskedConvAutoencoder()
    signal = torch.randn(2, 1, 500)
    mask = torch.zeros(2, 1, 500, dtype=torch.bool)

    reconstructed = model(signal, mask)

    assert reconstructed.shape == signal.shape
    assert model.embedding(signal).shape == (2, 64)
    assert torch.isfinite(reconstructed).all()


def test_fixed_windows_never_cross_acquisition_gap():
    time_s = np.arange(1000, dtype=float) / 1000
    time_s[500:] += 1.0
    signal = np.arange(1000, dtype=float)

    windows, starts = extract_fixed_windows(
        time_s,
        signal,
        window_samples=250,
        hop_samples=125,
        max_gap_s=0.0015,
    )

    assert starts.tolist() == [0, 125, 250, 500, 625, 750]
    assert windows.shape == (6, 250)
    assert all(np.max(np.diff(time_s[start : start + 250])) <= 0.0015 for start in starts)


def test_group_split_is_repeatable_and_keeps_groups_disjoint():
    groups = ["S1", "S1", "S2", "S3", "S4", "S5"]

    split = split_groups(groups, validation_fraction=0.2, seed=19)
    repeated = split_groups(groups, validation_fraction=0.2, seed=19)

    assert split == repeated
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]) | set(split["validation"]) == {"S1", "S2", "S3", "S4", "S5"}
    assert len(split["validation"]) == 1


def test_runner_counts_filename_groups_from_recording_objects():
    from scripts.train_original_emg_ssl import _unique_group_count
    from emg_fatigue.contracts import Recording

    recordings = [
        Recording("R1", "ZIP_G01", "zip", "left", "R1", np.array([0.0, 0.001]), np.array([1.0, 2.0]), "a"),
        Recording("R2", "ZIP_G01", "zip", "right", "R1", np.array([0.0, 0.001]), np.array([1.0, 3.0]), "b"),
        Recording("R3", "ZIP_G02", "zip", "left", "R2", np.array([0.0, 0.001]), np.array([1.0, 4.0]), "c"),
    ]

    assert _unique_group_count(recordings) == 2


def test_finalize_existing_run_writes_config_and_correct_group_counts(tmp_path):
    import argparse
    import csv
    import json

    from scripts.train_original_emg_ssl import finalize_existing_run

    source_course = tmp_path / "input_course.zip"
    source_raw = tmp_path / "input_raw.rar"
    source_course.write_bytes(b"course fixture")
    source_raw.write_bytes(b"raw fixture")
    output = tmp_path / "run_001"
    output.mkdir()
    (output / "source").mkdir()
    torch.save({
        "best_epoch": 2,
        "best_validation_masked_mse": 0.25,
        "seed": 9,
        "architecture": {"parameter_count": 41585, "embedding_dim": 64},
    }, output / "checkpoint.pt")
    (output / "training_history.csv").write_text(
        "epoch,train_masked_mse,validation_masked_mse,validation_zero_baseline_mse,epoch_seconds\n"
        "1,0.5,0.3,0.4,1.0\n2,0.4,0.25,0.4,1.2\n",
        encoding="utf-8",
    )
    with (output / "window_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "recording_id", "group_id", "sampling_rate_hz", "split"])
        writer.writeheader()
        writer.writerows([
            {"source": "course_zip", "recording_id": "C1", "group_id": "ZIP_G01", "sampling_rate_hz": 1000, "split": "train"},
            {"source": "course_zip", "recording_id": "C2", "group_id": "ZIP_G02", "sampling_rate_hz": 1000, "split": "validation"},
            {"source": "raw_rar", "recording_id": "R1", "group_id": "RAR_G01", "sampling_rate_hz": 1000, "split": "source_holdout_descriptive"},
        ])
    with (output / "preprocessing_qc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "recording_id", "group_id"])
        writer.writeheader()
        writer.writerows([
            {"source": "course_zip", "recording_id": "C1", "group_id": "ZIP_G01"},
            {"source": "course_zip", "recording_id": "C2", "group_id": "ZIP_G02"},
            {"source": "raw_rar", "recording_id": "R1", "group_id": "RAR_G01"},
        ])
    (output / "split_manifest.json").write_text(json.dumps({"train_groups": ["ZIP_G01"], "validation_groups": ["ZIP_G02"]}), encoding="utf-8")
    for name in ["group_reconstruction_metrics.csv", "line_interference_qc.csv"]:
        (output / name).write_text("\n", encoding="utf-8")
    for name in ["preprocessing_manifest.json", "feature_schema.json"]:
        (output / name).write_text("{}\n", encoding="utf-8")
    (output / "preanalysis_features_raw.csv").write_text("window_id\nW1\n", encoding="utf-8")
    (output / "preanalysis_features_zscore.csv").write_text("window_id\nW1\n", encoding="utf-8")
    (output / "preanalysis_feature_schema.json").write_text(json.dumps({
        "feature_columns": ["rms"],
        "standardization": {"fit_source": "course_zip train rows only"},
    }), encoding="utf-8")

    args = argparse.Namespace(
        output_dir=str(output), course_archive=str(source_course), raw_archive=str(source_raw), outer_archive=None,
        seed=9, epochs=30, patience=7, batch_size=128, cpu_threads=4, learning_rate=0.001,
        weight_decay=0.0001, min_delta=0.00001, validation_fraction=0.2,
    )
    finalize_existing_run(args)

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data"]["course_zip_recordings"] == 2
    assert manifest["data"]["course_zip_filename_groups"] == 2
    assert manifest["data"]["raw_rar_recordings"] == 1
    assert manifest["data"]["raw_rar_filename_groups"] == 1
    assert (output / "training_config.yaml").is_file()
    assert "checkpoint.pt" in manifest["artifacts"]


def test_feature_scaler_uses_only_train_rows():
    import pandas as pd

    from scripts.train_original_emg_ssl import fit_train_only_feature_standardizer

    frame = pd.DataFrame({
        "source": ["course_zip", "course_zip", "course_zip", "raw_rar"],
        "group_id": ["ZIP_G01", "ZIP_G02", "ZIP_G03", "RAR_G01"],
        "split": ["train", "train", "validation", "source_holdout_descriptive"],
        "feature_status": ["valid"] * 4,
        "rms": [2.0, 4.0, 100.0, -100.0],
    })
    standardized, scaler = fit_train_only_feature_standardizer(frame, ["rms"])

    assert scaler["mean"]["rms"] == pytest.approx(3.0)
    assert scaler["scale"]["rms"] == pytest.approx(1.0)
    assert standardized["rms_z"].tolist() == pytest.approx([-1.0, 1.0, 97.0, -103.0])


def test_preanalysis_feature_export_contains_required_time_spectral_and_wavelet_features():
    from scripts.train_original_emg_ssl import build_preanalysis_feature_frames

    time = np.arange(500, dtype=float) / 1000
    first = 2 * np.sin(2 * np.pi * 80 * time)
    second = 3 * np.sin(2 * np.pi * 100 * time)
    rows = []
    for index, (group, values, split) in enumerate([
        ("ZIP_G01", first, "train"),
        ("ZIP_G01", second, "train"),
        ("ZIP_G02", first, "validation"),
        ("RAR_G01", second, "source_holdout_descriptive"),
    ]):
        rows.append({
            "signal": values, "source": "raw_rar" if group.startswith("RAR") else "course_zip",
            "recording_id": f"R{index}", "group_id": group, "side": "left", "series": "R1",
            "run_index": 0, "window_index_in_run": 0, "start_sample_in_run": 0,
            "start_time_s": 0.0, "end_time_s": 0.499, "duration_s": 0.499,
            "sampling_rate_hz": 1000.0, "window_id": f"W{index}",
        })
    raw, standardized, schema = build_preanalysis_feature_frames(
        [row for row in rows if row["source"] == "course_zip"],
        [row for row in rows if row["source"] == "raw_rar"],
        {"ZIP_G01"},
    )

    required = {"rms", "mav", "waveform_length_per_s_sample_span", "mdf_hz", "mnf_hz", "power_ratio_low_high", "spectral_power_20_400", "wavelet_A3", "wavelet_D3", "wavelet_D2", "wavelet_D1"}
    assert required.issubset(raw.columns)
    assert {"rms_z", "mdf_hz_z", "wavelet_A3_z"}.issubset(standardized.columns)
    assert raw["feature_status"].eq("valid").all()
    assert schema["standardization"]["fit_source"] == "course_zip train rows only"


def test_in_band_spectral_power_integrates_periodogram_density():
    from scripts.train_original_emg_ssl import spectral_power_20_400

    time = np.arange(500, dtype=float) / 1000
    signal = 2 * np.sin(2 * np.pi * 80 * time)

    assert spectral_power_20_400(signal, 1000.0) == pytest.approx(2.0, rel=0.03)


def test_training_curve_is_exported_as_a_readable_png(tmp_path):
    import pandas as pd

    from scripts.train_original_emg_ssl import write_training_curve

    history = pd.DataFrame({
        "epoch": [1, 2, 3],
        "train_masked_mse": [0.8, 0.6, 0.5],
        "validation_masked_mse": [0.5, 0.4, 0.45],
        "validation_zero_baseline_mse": [0.6, 0.6, 0.6],
    })
    path = tmp_path / "training_curve.png"

    write_training_curve(history, path)

    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.stat().st_size > 10_000
