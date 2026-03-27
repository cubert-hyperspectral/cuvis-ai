"""Tests for ChannelNormalizeNode."""

from __future__ import annotations

import torch

from cuvis_ai.node.preprocessors import ChannelNormalizeNode


def test_imagenet_defaults() -> None:
    """Default mean/std match ImageNet values."""
    node = ChannelNormalizeNode()
    assert node._mean_vals == (0.485, 0.456, 0.406)
    assert node._std_vals == (0.229, 0.224, 0.225)


def test_output_shape_preserved() -> None:
    """[N, C, H, W] in → [N, C, H, W] out."""
    node = ChannelNormalizeNode()
    images = torch.rand(5, 3, 256, 128)
    result = node.forward(images=images)
    assert result["normalized"].shape == (5, 3, 256, 128)


def test_custom_mean_std() -> None:
    """Non-default normalization values."""
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    node = ChannelNormalizeNode(mean=mean, std=std)
    images = torch.ones(2, 3, 4, 4)
    result = node.forward(images=images)
    # (1.0 - 0.5) / 0.5 = 1.0
    assert torch.allclose(result["normalized"], torch.ones(2, 3, 4, 4))


def test_differentiable() -> None:
    """Backward through normalize → grad is not None."""
    node = ChannelNormalizeNode()
    images = torch.rand(3, 3, 16, 16, requires_grad=True)
    result = node.forward(images=images)
    loss = result["normalized"].sum()
    loss.backward()
    assert images.grad is not None


def test_zero_batch() -> None:
    """N=0 → empty tensor passthrough."""
    node = ChannelNormalizeNode()
    images = torch.empty(0, 3, 256, 128)
    result = node.forward(images=images)
    assert result["normalized"].shape == (0, 3, 256, 128)
