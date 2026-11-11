import pytest

torch = pytest.importorskip("torch")

from emg_fatigue.weak_supervision.models_neural import (
    BinarySignalClassifier,
    MaskedSignalAutoencoder,
    SignalEncoder,
    SoftGatedExperts,
    gate_load_balance_penalty,
    normalized_gate_entropy,
)


def test_binary_signal_classifier_returns_one_logit_per_window():
    model = BinarySignalClassifier()
    logits = model(torch.randn(3, 1, 500))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_encoder_rejects_signal_with_wrong_channel_shape():
    with pytest.raises(ValueError, match=r"shape \(batch, 1, samples\)"):
        SignalEncoder()(torch.randn(3, 2, 500))


def test_soft_moe_mixes_three_latent_experts_for_one_proxy_logit():
    logits, gates, expert_logits = SoftGatedExperts(expert_count=3)(torch.randn(4, 1, 500))

    assert logits.shape == (4,)
    assert gates.shape == (4, 3)
    assert expert_logits.shape == (4, 3)
    assert torch.allclose(gates.sum(dim=1), torch.ones(4), atol=1e-6)


def test_masked_autoencoder_reconstructs_the_input_sample_shape():
    signal = torch.randn(2, 1, 500)
    reconstruction, embedding = MaskedSignalAutoencoder()(signal)

    assert reconstruction.shape == signal.shape
    assert embedding.shape == (2, 32)


def test_gate_diagnostics_detect_collapse_and_reward_uniform_use():
    uniform = torch.full((4, 3), 1.0 / 3.0)
    collapsed = torch.tensor([[0.98, 0.01, 0.01]] * 4)

    assert gate_load_balance_penalty(uniform).item() == pytest.approx(0.0, abs=1e-7)
    assert gate_load_balance_penalty(collapsed).item() > 0.8
    assert normalized_gate_entropy(uniform).item() == pytest.approx(1.0, abs=1e-6)
    assert normalized_gate_entropy(collapsed).item() < 0.15
