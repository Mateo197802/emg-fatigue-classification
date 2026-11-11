import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emg_fatigue.weak_supervision.ssl_pretraining import pretrain_encoder


def test_masked_signal_pretraining_returns_finite_epoch_history_and_encoder():
    windows = np.random.default_rng(17).normal(size=(8, 500)).astype(np.float32)
    fit = pretrain_encoder(windows, epochs=2, seed=23, batch_size=4)

    assert fit.train_window_count == 8
    assert len(fit.reconstruction_losses) == 2
    assert np.isfinite(fit.reconstruction_losses).all()
    assert fit.encoder(torch.from_numpy(windows[:2, None, :])).shape == (2, 32)
