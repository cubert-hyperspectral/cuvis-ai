"""Unit tests for ConcreteChannelMixer node."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai.node.channel_mixer import ConcreteChannelMixer


@torch.no_grad()
def test_concrete_mixer_basic_shape() -> None:
    """Output should have correct shape [B, H, W, output_channels]."""
    B, H, W, Cin, Cout = 2, 6, 6, 10, 3
    x = torch.rand(B, H, W, Cin)

    node = ConcreteChannelMixer(input_channels=Cin, output_channels=Cout)
    result = node.forward(data=x)

    assert result["rgb"].shape == (B, H, W, Cout)
    assert result["selection_weights"].shape == (Cout, Cin)


@torch.no_grad()
def test_concrete_mixer_output_range() -> None:
    """When input is in [0, 1], output should also be in [0, 1] (weighted sum of softmax)."""
    B, H, W, Cin, Cout = 2, 4, 4, 8, 3
    x = torch.rand(B, H, W, Cin)

    node = ConcreteChannelMixer(input_channels=Cin, output_channels=Cout)
    rgb = node.forward(data=x)["rgb"]

    assert rgb.min() >= -1e-6
    assert rgb.max() <= 1.0 + 1e-6


@torch.no_grad()
def test_concrete_mixer_hard_inference() -> None:
    """With use_hard_inference=True, val-mode weights should be one-hot."""
    Cin, Cout = 8, 3
    node = ConcreteChannelMixer(input_channels=Cin, output_channels=Cout, use_hard_inference=True)
    node.eval()

    x = torch.rand(1, 4, 4, Cin)
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)
    result = node.forward(data=x, context=ctx)

    weights = result["selection_weights"]
    # Each row should be one-hot (one 1.0, rest 0.0)
    for row in range(Cout):
        assert torch.sum(weights[row] == 1.0).item() == 1
        assert torch.sum(weights[row] == 0.0).item() == Cin - 1


@torch.no_grad()
def test_concrete_mixer_soft_inference() -> None:
    """With use_hard_inference=False, val-mode weights should be soft (sum to ~1 per row)."""
    Cin, Cout = 8, 3
    node = ConcreteChannelMixer(input_channels=Cin, output_channels=Cout, use_hard_inference=False)
    node.eval()

    x = torch.rand(1, 4, 4, Cin)
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)
    result = node.forward(data=x, context=ctx)

    weights = result["selection_weights"]
    # Each row should sum to ~1.0
    for row in range(Cout):
        assert abs(weights[row].sum().item() - 1.0) < 1e-5


def test_concrete_mixer_training_gumbel() -> None:
    """During training, Gumbel noise should be used (stochastic weights)."""
    Cin, Cout = 8, 3
    node = ConcreteChannelMixer(input_channels=Cin, output_channels=Cout)
    node.train()

    x = torch.rand(2, 4, 4, Cin)
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0)

    # Run twice — stochastic so weights should differ
    r1 = node.forward(data=x, context=ctx)
    r2 = node.forward(data=x, context=ctx)

    # Weights are stochastic in training mode
    assert not torch.allclose(r1["selection_weights"], r2["selection_weights"])


def test_concrete_mixer_gradient_flow() -> None:
    """Gradients should flow through the mixer to logits."""
    Cin, Cout = 6, 3
    node = ConcreteChannelMixer(input_channels=Cin, output_channels=Cout)
    node.train()

    x = torch.rand(1, 4, 4, Cin)
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0)
    result = node.forward(data=x, context=ctx)

    loss = result["rgb"].mean()
    loss.backward()

    assert node.logits.grad is not None
    assert node.logits.grad.shape == (Cout, Cin)


@torch.no_grad()
def test_concrete_mixer_temperature_annealing() -> None:
    """Temperature should decrease across epochs."""
    Cin, Cout = 6, 3
    node = ConcreteChannelMixer(
        input_channels=Cin, output_channels=Cout, tau_start=10.0, tau_end=0.1, max_epochs=20
    )

    ctx_early = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0)
    ctx_late = Context(stage=ExecutionStage.TRAIN, epoch=19, batch_idx=0)

    tau_early = node._current_tau(ctx_early)
    tau_late = node._current_tau(ctx_late)

    assert tau_early > tau_late
    assert abs(tau_early - 10.0) < 0.01
    assert abs(tau_late - 0.1) < 0.01


@torch.no_grad()
def test_concrete_mixer_get_selected_bands() -> None:
    """get_selected_bands should return argmax indices."""
    Cin, Cout = 6, 3
    node = ConcreteChannelMixer(input_channels=Cin, output_channels=Cout)

    # Set logits so argmax is deterministic
    with torch.no_grad():
        node.logits.data = torch.zeros(Cout, Cin)
        node.logits.data[0, 2] = 10.0
        node.logits.data[1, 4] = 10.0
        node.logits.data[2, 0] = 10.0

    bands = node.get_selected_bands()
    assert bands.tolist() == [2, 4, 0]


@torch.no_grad()
def test_concrete_mixer_get_selection_weights() -> None:
    """get_selection_weights should return valid probability distributions."""
    Cin, Cout = 6, 3
    node = ConcreteChannelMixer(input_channels=Cin, output_channels=Cout)

    weights_det = node.get_selection_weights(deterministic=True)
    weights_non = node.get_selection_weights(deterministic=False)

    assert weights_det.shape == (Cout, Cin)
    assert weights_non.shape == (Cout, Cin)
    # Each row should sum to ~1.0
    for row in range(Cout):
        assert abs(weights_det[row].sum().item() - 1.0) < 1e-5
        assert abs(weights_non[row].sum().item() - 1.0) < 1e-5


def test_concrete_mixer_invalid_params() -> None:
    """Invalid parameters should raise ValueError."""
    with pytest.raises(ValueError, match="output_channels must be positive"):
        ConcreteChannelMixer(input_channels=6, output_channels=0)

    with pytest.raises(ValueError, match="input_channels must be positive"):
        ConcreteChannelMixer(input_channels=0, output_channels=3)

    with pytest.raises(ValueError, match="tau_start and tau_end must be positive"):
        ConcreteChannelMixer(input_channels=6, output_channels=3, tau_start=-1.0)


@torch.no_grad()
def test_concrete_mixer_max_epochs_one() -> None:
    """With max_epochs=1, temperature should be tau_end."""
    node = ConcreteChannelMixer(
        input_channels=6, output_channels=3, tau_start=10.0, tau_end=0.5, max_epochs=1
    )
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0)
    assert node._current_tau(ctx) == 0.5
