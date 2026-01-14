"""Unit tests for LearnableChannelMixer node."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_mixer import LearnableChannelMixer


@torch.no_grad()
def test_learnable_channel_mixer_basic_shape_and_dtype() -> None:
    """Basic BHWC → BHWC mapping with correct channel reduction and dtype."""
    B, H, W, Cin, Cout = 2, 8, 9, 5, 3
    x = torch.randn(B, H, W, Cin, dtype=torch.float32)

    node = LearnableChannelMixer(
        input_channels=Cin,
        output_channels=Cout,
        init_method="xavier",
    )

    out = node.forward(data=x)["rgb"]

    assert out.shape == (B, H, W, Cout)
    assert out.dtype == torch.float32


@torch.no_grad()
def test_learnable_channel_mixer_invalid_input_channels_raises() -> None:
    """Mismatched input channels should raise a clear ValueError."""
    B, H, W, Cin, Cout = 1, 4, 4, 5, 3
    x = torch.randn(B, H, W, Cin + 1)

    node = LearnableChannelMixer(
        input_channels=Cin,
        output_channels=Cout,
    )

    with pytest.raises(ValueError, match="Expected 5 input channels"):
        node.forward(data=x)


@torch.no_grad()
def test_learnable_channel_mixer_reduction_scheme_validation() -> None:
    """Invalid reduction schemes should raise informative ValueErrors."""
    # Wrong first element
    with pytest.raises(ValueError, match="First element of reduction_scheme"):
        LearnableChannelMixer(
            input_channels=5,
            output_channels=3,
            reduction_scheme=[4, 3],
        )

    # Wrong last element
    with pytest.raises(ValueError, match="Last element of reduction_scheme"):
        LearnableChannelMixer(
            input_channels=5,
            output_channels=3,
            reduction_scheme=[5, 4],
        )

    # Too short
    with pytest.raises(ValueError, match="at least 2 elements"):
        LearnableChannelMixer(
            input_channels=5,
            output_channels=5,
            reduction_scheme=[5],
        )


@torch.no_grad()
def test_learnable_channel_mixer_multilayer_reduction() -> None:
    """Multi-layer reduction scheme is honored and produces correct output shape."""
    B, H, W, Cin, Cout = 2, 6, 7, 5, 2
    x = torch.randn(B, H, W, Cin)

    reduction_scheme = [Cin, 4, Cout]
    node = LearnableChannelMixer(
        input_channels=Cin,
        output_channels=Cout,
        reduction_scheme=reduction_scheme,
        use_activation=True,
        normalize_output=False,
    )

    out = node.forward(data=x)["rgb"]
    assert out.shape == (B, H, W, Cout)


@torch.no_grad()
def test_learnable_channel_mixer_normalization_unit_range() -> None:
    """When normalize_output=True, output should be in [0, 1] per-image and per-channel."""
    B, H, W, Cin, Cout = 2, 5, 6, 7, 3
    x = torch.randn(B, H, W, Cin) * 5.0 - 2.5

    node = LearnableChannelMixer(
        input_channels=Cin,
        output_channels=Cout,
        normalize_output=True,
        eps=1e-6,
    )

    out = node.forward(data=x)["rgb"]
    assert out.shape == (B, H, W, Cout)

    flat = out.view(B, -1, Cout).cpu().numpy()
    mins = flat.min(axis=1)
    maxs = flat.max(axis=1)

    # Allow small numerical tolerance around [0, 1]
    assert np.all(mins >= -1e-6)
    assert np.all(maxs <= 1.0 + 1e-6)


def test_learnable_channel_mixer_requires_initial_fit_flag() -> None:
    """requires_initial_fit is True only for init_method='pca'."""
    node_pca = LearnableChannelMixer(input_channels=5, output_channels=3, init_method="pca")
    node_xavier = LearnableChannelMixer(input_channels=5, output_channels=3, init_method="xavier")

    assert node_pca.requires_initial_fit is True
    assert node_xavier.requires_initial_fit is False


def test_learnable_channel_mixer_statistical_initialization_sets_weights() -> None:
    """statistical_initialization should set first conv weights using PCA on input."""
    Cin, Cout = 4, 3

    node = LearnableChannelMixer(
        input_channels=Cin,
        output_channels=Cout,
        init_method="pca",
        reduction_scheme=[Cin, Cout],
    )

    # Create small stream of BHWC tensors
    stream = ({"data": torch.randn(2, 4, 4, Cin)} for _ in range(3))

    # Before initialization, record initial weights
    initial_weight = node.convs[0].weight.detach().clone()

    node.statistical_initialization(stream)

    updated_weight = node.convs[0].weight.detach()
    # We expect weights to change after PCA init
    assert not torch.allclose(initial_weight, updated_weight)


def test_learnable_channel_mixer_gradients_flow() -> None:
    """unfreeze() should enable gradients and allow a backward pass through the mixer."""
    B, H, W, Cin, Cout = 2, 4, 4, 5, 3
    x = torch.randn(B, H, W, Cin, requires_grad=True)

    node = LearnableChannelMixer(
        input_channels=Cin,
        output_channels=Cout,
        normalize_output=False,
    )
    node.unfreeze()

    out = node.forward(data=x)["rgb"]
    loss = out.mean()
    loss.backward()

    # All conv parameters should have gradients
    for conv in node.convs:
        for p in conv.parameters():
            assert p.grad is not None
