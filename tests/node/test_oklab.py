"""Validate torch OKLab conversion against a pure-numpy reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.utils.color_spaces import linear_rgb_to_oklab, rgb_to_oklab, srgb_to_linear

# ---------------------------------------------------------------------------
# Numpy reference implementation
# ---------------------------------------------------------------------------


def _numpy_srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1.0 + a)) ** 2.4,
    )


def _numpy_linear_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    M1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    lms = rgb @ M1.T
    lms_cbrt = np.sign(lms) * np.abs(lms).clip(1e-12) ** (1.0 / 3.0)
    M2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return lms_cbrt @ M2.T


# ---------------------------------------------------------------------------
# Tests: srgb_to_linear
# ---------------------------------------------------------------------------


class TestSrgbToLinear:
    def test_below_threshold(self):
        """Values <= 0.04045 use the linear formula x/12.92."""
        x = torch.tensor([0.0, 0.01, 0.04045])
        result = srgb_to_linear(x).numpy()
        expected = np.array([0.0, 0.01, 0.04045]) / 12.92
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_above_threshold(self):
        """Values > 0.04045 use the gamma formula."""
        x = torch.tensor([0.5, 0.8, 1.0])
        result = srgb_to_linear(x).numpy()
        expected = _numpy_srgb_to_linear(np.array([0.5, 0.8, 1.0]))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_zero_and_one(self):
        """sRGB 0 maps to 0, sRGB 1 maps to 1."""
        x = torch.tensor([0.0, 1.0])
        result = srgb_to_linear(x).numpy()
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-7)


# ---------------------------------------------------------------------------
# Tests: linear_rgb_to_oklab
# ---------------------------------------------------------------------------

# Known linear RGB colors
KNOWN_COLORS = {
    "black": np.array([0.0, 0.0, 0.0]),
    "white": np.array([1.0, 1.0, 1.0]),
    "red": np.array([1.0, 0.0, 0.0]),
    "green": np.array([0.0, 1.0, 0.0]),
    "blue": np.array([0.0, 0.0, 1.0]),
    "mid_gray": np.array([0.5, 0.5, 0.5]),
}


class TestLinearRgbToOklab:
    @pytest.mark.parametrize("name,rgb", list(KNOWN_COLORS.items()))
    def test_known_colors(self, name: str, rgb: np.ndarray):
        """Torch and numpy agree for known color values."""
        torch_result = linear_rgb_to_oklab(torch.tensor(rgb, dtype=torch.float32)).numpy()
        numpy_result = _numpy_linear_rgb_to_oklab(rgb.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(
            torch_result, numpy_result, atol=1e-6, err_msg=f"Mismatch for {name}"
        )

    def test_batch_shape(self):
        """Works with [B, H, W, 3] input."""
        rng = np.random.default_rng(42)
        rgb_np = rng.random((2, 8, 8, 3)).astype(np.float32)
        rgb_torch = torch.tensor(rgb_np)

        torch_result = linear_rgb_to_oklab(rgb_torch).numpy()
        numpy_result = _numpy_linear_rgb_to_oklab(rgb_np.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(torch_result, numpy_result, atol=1e-5)

    def test_1d_input(self):
        """Works with a single [3] vector."""
        rgb = np.array([0.3, 0.6, 0.1], dtype=np.float32)
        torch_result = linear_rgb_to_oklab(torch.tensor(rgb)).numpy()
        numpy_result = _numpy_linear_rgb_to_oklab(rgb.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(torch_result, numpy_result, atol=1e-6)

    def test_white_lightness_near_one(self):
        """OKLab L channel for white should be close to 1.0."""
        result = linear_rgb_to_oklab(torch.tensor([1.0, 1.0, 1.0]))
        assert abs(result[0].item() - 1.0) < 0.01, f"White L={result[0].item()}, expected ~1.0"

    def test_black_is_near_zero(self):
        """OKLab for black should be near (0, 0, 0)."""
        result = linear_rgb_to_oklab(torch.tensor([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result.numpy(), [0.0, 0.0, 0.0], atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: rgb_to_oklab (combined pipeline)
# ---------------------------------------------------------------------------


class TestRgbToOklab:
    def test_assume_srgb_true(self):
        """With assume_srgb=True, applies sRGB→linear before OKLab."""
        srgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        linear = _numpy_srgb_to_linear(srgb.astype(np.float64))
        expected = _numpy_linear_rgb_to_oklab(linear).astype(np.float32)
        result = rgb_to_oklab(torch.tensor(srgb), assume_srgb=True).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_assume_srgb_false(self):
        """With assume_srgb=False, skips gamma and treats as linear."""
        linear = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        expected = _numpy_linear_rgb_to_oklab(linear.astype(np.float64)).astype(np.float32)
        result = rgb_to_oklab(torch.tensor(linear), assume_srgb=False).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradient_propagates(self):
        """Gradients flow through the OKLab conversion."""
        rgb = torch.rand(2, 4, 4, 3, requires_grad=True)
        oklab = linear_rgb_to_oklab(rgb)
        loss = oklab.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.isnan(rgb.grad).any()

    def test_gradient_through_srgb(self):
        """Gradients flow through sRGB→linear→OKLab pipeline."""
        rgb = torch.rand(2, 4, 4, 3, requires_grad=True)
        oklab = rgb_to_oklab(rgb, assume_srgb=True)
        loss = oklab.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.isnan(rgb.grad).any()
