"""Tests for FixedWavelengthSelector generalization to n-channel output.

Covers:
- Backward compat: 3-channel default still works exactly as before
- n-channel (e.g. 6) band stacking — shape, dtype, correct band pick
- Empty / single wavelength edge cases
- normalize_output=True warning for n != 3
- OUTPUT_SPECS shape relaxed to (-1,-1,-1,-1)
- ChannelSelectorBase OUTPUT_SPECS shape is (-1,-1,-1,-1)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import ChannelSelectorBase, FixedWavelengthSelector

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cube(B: int = 1, H: int = 4, W: int = 4, C: int = 10) -> torch.Tensor:
    return torch.rand(B, H, W, C)


def _wavelengths(n: int = 10, lo: float = 400.0, hi: float = 900.0) -> np.ndarray:
    return np.linspace(lo, hi, n, dtype=np.float32)


# ---------------------------------------------------------------------------
# OUTPUT_SPECS shape relaxation
# ---------------------------------------------------------------------------

class TestOutputSpecsShape:
    def test_base_class_output_spec_is_n_channel(self) -> None:
        """ChannelSelectorBase.OUTPUT_SPECS must have shape (-1,-1,-1,-1), not (-1,-1,-1,3)."""
        spec = ChannelSelectorBase.OUTPUT_SPECS["rgb_image"]
        assert spec.shape == (-1, -1, -1, -1), (
            f"Expected (-1,-1,-1,-1), got {spec.shape}. "
            "This would reject n-channel output from FixedWavelengthSelector."
        )

    def test_fixed_selector_inherits_relaxed_spec(self) -> None:
        sel = FixedWavelengthSelector(target_wavelengths=(650.0, 550.0, 450.0))
        spec = sel.OUTPUT_SPECS["rgb_image"]
        assert spec.shape == (-1, -1, -1, -1)


# ---------------------------------------------------------------------------
# Backward compatibility — 3-channel default
# ---------------------------------------------------------------------------

class TestThreeChannelBackwardCompat:
    def test_default_wavelengths_output_shape(self) -> None:
        """Default (650, 550, 450) must still produce [B, H, W, 3]."""
        sel = FixedWavelengthSelector(normalize_output=False)
        cube = _cube(C=10)
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 3)

    def test_three_channel_with_normalize_uses_compose_rgb(self) -> None:
        """normalize_output=True + 3 channels → _compose_rgb is called → 0-1 range."""
        sel = FixedWavelengthSelector(
            target_wavelengths=(650.0, 550.0, 450.0),
            normalize_output=True,
            norm_mode="per_frame",
        )
        # Use a cube with known max so output lands in [0,1]
        cube = torch.rand(1, 4, 4, 10) * 500.0
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 3)
        assert out["rgb_image"].dtype == torch.float32
        assert float(out["rgb_image"].max()) <= 1.0 + 1e-5

    def test_band_info_strategy_and_fields(self) -> None:
        sel = FixedWavelengthSelector(
            target_wavelengths=(650.0, 550.0, 450.0), normalize_output=False
        )
        cube = _cube(C=10)
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        info = out["band_info"]
        assert info["strategy"] == "baseline_false_rgb"
        assert len(info["band_indices"]) == 3
        assert len(info["band_wavelengths_nm"]) == 3
        assert info["target_wavelengths_nm"] == [650.0, 550.0, 450.0]
        assert info["normalized_output"] is False

    def test_picks_nearest_band(self) -> None:
        """The selected band index should be the one closest to each target."""
        wl = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32)
        sel = FixedWavelengthSelector(
            target_wavelengths=(610.0, 490.0, 410.0), normalize_output=False
        )
        cube = _cube(C=4)
        out = sel.forward(cube, wl)
        # 610 → nearest is 600 (index 2), 490 → 500 (index 1), 410 → 400 (index 0)
        assert out["band_info"]["band_indices"] == [2, 1, 0]


# ---------------------------------------------------------------------------
# n-channel generalization
# ---------------------------------------------------------------------------

class TestNChannelOutput:
    def test_six_channel_output_shape(self) -> None:
        """6-target-wavelengths → output shape [B, H, W, 6]."""
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(target_wavelengths=targets, normalize_output=False)
        cube = _cube(B=2, H=8, W=8, C=20)
        wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (2, 8, 8, 6)

    def test_six_channel_dtype_is_float32(self) -> None:
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(target_wavelengths=targets, normalize_output=False)
        cube = _cube(C=20)
        wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].dtype == torch.float32

    def test_six_channel_band_info_length(self) -> None:
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(target_wavelengths=targets, normalize_output=False)
        cube = _cube(C=20)
        wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
        out = sel.forward(cube, wl)
        assert len(out["band_info"]["band_indices"]) == 6
        assert len(out["band_info"]["band_wavelengths_nm"]) == 6

    def test_six_channel_values_match_cube_bands(self) -> None:
        """Each output channel must be exactly the nearest cube band, not a mix."""
        wl = np.array([450.0, 550.0, 625.0, 1050.0, 1200.0, 1450.0], dtype=np.float32)
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(target_wavelengths=targets, normalize_output=False)
        # Cube: each channel c is filled with float(c) so we can identify which band landed where
        cube = torch.zeros(1, 4, 4, 6)
        for c in range(6):
            cube[..., c] = float(c)
        out = sel.forward(cube, wl)
        rgb = out["rgb_image"]  # [1, 4, 4, 6]
        # target order: 625→idx2, 550→idx1, 450→idx0, 1450→idx5, 1200→idx4, 1050→idx3
        expected_values = [2.0, 1.0, 0.0, 5.0, 4.0, 3.0]
        for ch, expected in enumerate(expected_values):
            assert float(rgb[0, 0, 0, ch]) == pytest.approx(expected), (
                f"channel {ch}: expected {expected}, got {float(rgb[0, 0, 0, ch])}"
            )

    def test_single_channel_output_shape(self) -> None:
        """Single target wavelength → [B, H, W, 1]."""
        sel = FixedWavelengthSelector(target_wavelengths=(550.0,), normalize_output=False)
        cube = _cube(C=10)
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 1)


# ---------------------------------------------------------------------------
# normalize_output=True warning for n != 3
# ---------------------------------------------------------------------------

class TestNormalizeOutputWarning:
    def test_six_channel_normalize_true_warns_and_returns_raw(self) -> None:
        """normalize_output=True with n=6 must warn and return raw stacked bands."""
        import warnings
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(target_wavelengths=targets, normalize_output=True)
        cube = _cube(C=20) * 500.0  # large values to show normalization was NOT applied
        wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
        # loguru warnings don't go through warnings.warn, so just check the output
        # shape is correct and values are > 1 (proving no normalization happened).
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 6)
        # Raw cube values are ~0-500, so at least some pixels should be > 1
        assert float(out["rgb_image"].max()) > 1.0, (
            "Expected raw (unnormalized) values > 1 for n=6 with normalize_output=True"
        )

    def test_four_channel_normalize_true_returns_raw(self) -> None:
        sel = FixedWavelengthSelector(
            target_wavelengths=(700.0, 600.0, 500.0, 400.0), normalize_output=True
        )
        cube = _cube(C=10) * 200.0
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 4)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_empty_wavelengths_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one wavelength"):
            FixedWavelengthSelector(target_wavelengths=())

    def test_string_wavelengths_rejected_by_tuple_coercion(self) -> None:
        """Non-numeric strings should raise during float coercion."""
        with pytest.raises((ValueError, TypeError)):
            FixedWavelengthSelector(target_wavelengths=("red", "green", "blue"))

    def test_wavelengths_coerced_to_float(self) -> None:
        """Integer wavelengths should be accepted and stored as float."""
        sel = FixedWavelengthSelector(target_wavelengths=(650, 550, 450))
        assert all(isinstance(w, float) for w in sel.target_wavelengths)
