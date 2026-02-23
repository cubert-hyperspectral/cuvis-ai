"""Coverage tests for channel_selector nodes not covered by existing tests.

Targets: TopKIndices, FixedWavelengthSelector, CIRSelector, HighContrastSelector,
and SoftChannelSelector edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import (
    CIRSelector,
    FixedWavelengthSelector,
    HighContrastSelector,
    SoftChannelSelector,
    TopKIndices,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# TopKIndices
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_topk_indices_basic() -> None:
    """TopKIndices should return the correct top-k indices."""
    node = TopKIndices(k=3)
    weights = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5])
    result = node.forward(weights=weights)
    indices = result["indices"]
    assert indices.shape == (3,)
    assert set(indices.tolist()) == {1, 3, 4}


@torch.no_grad()
def test_topk_indices_k_exceeds_channels() -> None:
    """When k > n_channels, should return all available indices."""
    node = TopKIndices(k=10)
    weights = torch.tensor([0.5, 0.1, 0.9])
    result = node.forward(weights=weights)
    assert result["indices"].shape == (3,)


@torch.no_grad()
def test_topk_indices_empty_weights() -> None:
    """Empty weights should return empty indices."""
    node = TopKIndices(k=3)
    weights = torch.tensor([])
    result = node.forward(weights=weights)
    assert result["indices"].shape == (0,)


# ---------------------------------------------------------------------------
# FixedWavelengthSelector
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_fixed_wavelength_selector_basic() -> None:
    """FixedWavelengthSelector should produce [B, H, W, 3] output."""
    B, H, W, C = 2, 8, 8, 20
    cube = torch.rand(B, H, W, C)
    wavelengths = np.linspace(450, 900, C).astype(np.float32)

    node = FixedWavelengthSelector(target_wavelengths=(650.0, 550.0, 450.0))
    result = node.forward(cube=cube, wavelengths=wavelengths)

    assert result["rgb_image"].shape == (B, H, W, 3)
    assert result["rgb_image"].min() >= 0.0
    assert result["rgb_image"].max() <= 1.0 + 1e-6
    assert result["band_info"]["strategy"] == "baseline_false_rgb"
    assert len(result["band_info"]["band_indices"]) == 3


# ---------------------------------------------------------------------------
# CIRSelector
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_cir_selector_basic() -> None:
    """CIRSelector should produce false-color CIR output."""
    B, H, W, C = 2, 6, 6, 30
    cube = torch.rand(B, H, W, C)
    wavelengths = np.linspace(400, 1000, C).astype(np.float32)

    node = CIRSelector(nir_nm=860.0, red_nm=670.0, green_nm=560.0)
    result = node.forward(cube=cube, wavelengths=wavelengths)

    assert result["rgb_image"].shape == (B, H, W, 3)
    assert result["band_info"]["strategy"] == "cir_false_color"
    assert result["band_info"]["channel_mapping"] == {"R": "NIR", "G": "Red", "B": "Green"}


# ---------------------------------------------------------------------------
# HighContrastSelector
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_high_contrast_selector_basic() -> None:
    """HighContrastSelector should select high-contrast bands within windows."""
    B, H, W, C = 2, 8, 8, 30
    cube = torch.rand(B, H, W, C)
    wavelengths = np.linspace(400, 800, C).astype(np.float32)

    node = HighContrastSelector(
        windows=((440, 500), (500, 580), (610, 700)),
        alpha=0.1,
    )
    result = node.forward(cube=cube, wavelengths=wavelengths)

    assert result["rgb_image"].shape == (B, H, W, 3)
    assert result["band_info"]["strategy"] == "high_contrast"
    assert len(result["band_info"]["band_indices"]) == 3


@torch.no_grad()
def test_high_contrast_selector_fallback_empty_window() -> None:
    """When a window has no bands, should fall back to nearest wavelength."""
    B, H, W, C = 1, 4, 4, 10
    cube = torch.rand(B, H, W, C)
    # Wavelengths only 400-500, so window (600,700) has no bands
    wavelengths = np.linspace(400, 500, C).astype(np.float32)

    node = HighContrastSelector(
        windows=((400, 450), (450, 500), (600, 700)),
        alpha=0.1,
    )
    result = node.forward(cube=cube, wavelengths=wavelengths)
    assert result["rgb_image"].shape == (B, H, W, 3)
    assert len(result["band_info"]["band_indices"]) == 3


# ---------------------------------------------------------------------------
# SoftChannelSelector edge cases
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_soft_channel_selector_hard_selection() -> None:
    """Hard mode should produce top-k binary weights."""
    node = SoftChannelSelector(n_select=3, input_channels=10, hard=True)
    node.eval()

    # Set known logits
    with torch.no_grad():
        node.channel_logits.data = torch.arange(10, dtype=torch.float32)

    weights = node.get_selection_weights(hard=True)
    assert weights.sum().item() == 3.0
    # Top 3 should be indices 7, 8, 9
    assert weights[7].item() == 1.0
    assert weights[8].item() == 1.0
    assert weights[9].item() == 1.0


@torch.no_grad()
def test_soft_channel_selector_update_temperature() -> None:
    """update_temperature should decay temperature correctly."""
    node = SoftChannelSelector(
        n_select=3,
        input_channels=10,
        temperature_init=5.0,
        temperature_min=0.1,
        temperature_decay=0.5,
    )

    assert node.temperature == 5.0
    node.update_temperature(epoch=1)
    assert node.temperature == pytest.approx(2.5, abs=1e-6)
    node.update_temperature(epoch=10)
    # 5.0 * 0.5^10 = 0.00488, but min is 0.1
    assert node.temperature == 0.1


def test_soft_channel_selector_invalid_n_select() -> None:
    """n_select > input_channels should raise ValueError."""
    with pytest.raises(ValueError, match="Cannot select"):
        SoftChannelSelector(n_select=20, input_channels=10)


def test_soft_channel_selector_variance_init() -> None:
    """Variance-based initialization should set logits from data statistics."""
    node = SoftChannelSelector(n_select=3, input_channels=5, init_method="variance")

    # Create stream with known variance pattern
    stream = iter([{"data": torch.randn(2, 4, 4, 5) * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}])
    node.statistical_initialization(stream)

    assert node._statistically_initialized is True
    # Higher variance channels should have higher logits
    logits = node.channel_logits.data
    assert logits[-1] > logits[0]  # channel 5 (var~25) > channel 1 (var~1)
