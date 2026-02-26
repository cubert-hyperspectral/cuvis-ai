"""Tests for TrackingOverlayNode (multi-object coloured mask overlay)."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.visualizations import TrackingOverlayNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 16, w: int = 20) -> torch.Tensor:
    """Random float32 RGB frame [1, H, W, 3] in [0, 1]."""
    return torch.rand((1, h, w, 3), dtype=torch.float32)


def _make_mask(h: int = 16, w: int = 20, obj_ids: list[int] | None = None) -> torch.Tensor:
    """Label map [1, H, W] int32. Fills horizontal bands with object IDs."""
    mask = torch.zeros((1, h, w), dtype=torch.int32)
    if obj_ids:
        n = len(obj_ids)
        for i, oid in enumerate(obj_ids):
            row_start = (i * h) // n
            row_end = ((i + 1) * h) // n
            mask[0, row_start:row_end, :] = oid
    return mask


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype() -> None:
    h, w = 12, 16
    node = TrackingOverlayNode()
    out = node.forward(
        rgb_image=_make_frame(h, w),
        mask=_make_mask(h, w, obj_ids=[1]),
        object_ids=torch.tensor([[1]], dtype=torch.int64),
    )

    assert "rgb_with_overlay" in out
    result = out["rgb_with_overlay"]
    assert result.shape == (1, h, w, 3)
    assert result.dtype == torch.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_no_objects_output_matches_input() -> None:
    """Empty mask (all zeros) → output equals input (within uint8 rounding)."""
    node = TrackingOverlayNode()
    rgb = _make_frame()
    mask = torch.zeros((1, 16, 20), dtype=torch.int32)
    ids = torch.zeros((1, 0), dtype=torch.int64)

    out = node.forward(rgb_image=rgb, mask=mask, object_ids=ids)

    # uint8 roundtrip inside forward() introduces up to 1/255 ≈ 0.004 error.
    torch.testing.assert_close(out["rgb_with_overlay"], rgb, atol=1 / 255, rtol=0.0)


def test_object_ids_optional_inferred_from_mask() -> None:
    """Omitting object_ids produces the same result as passing them explicitly."""
    node = TrackingOverlayNode()
    rgb = _make_frame()
    mask = _make_mask(obj_ids=[2, 5])

    out_explicit = node.forward(
        rgb_image=rgb,
        mask=mask,
        object_ids=torch.tensor([[2, 5]], dtype=torch.int64),
    )
    out_inferred = node.forward(rgb_image=rgb, mask=mask, object_ids=None)

    torch.testing.assert_close(out_explicit["rgb_with_overlay"], out_inferred["rgb_with_overlay"])


def test_background_id_zero_in_object_ids_is_ignored() -> None:
    """Explicit object_ids containing 0 should render like the same list without 0."""
    node = TrackingOverlayNode(alpha=1.0, draw_contours=False, draw_ids=False)
    h, w = 10, 10
    rgb = torch.ones((1, h, w, 3), dtype=torch.float32) * 0.5
    mask = torch.zeros((1, h, w), dtype=torch.int32)
    mask[0, :5, :] = 1  # top half is object 1, bottom half background 0

    out_with_zero = node.forward(
        rgb_image=rgb,
        mask=mask,
        object_ids=torch.tensor([[0, 1]], dtype=torch.int64),
    )
    out_without_zero = node.forward(
        rgb_image=rgb,
        mask=mask,
        object_ids=torch.tensor([[1]], dtype=torch.int64),
    )

    torch.testing.assert_close(
        out_with_zero["rgb_with_overlay"],
        out_without_zero["rgb_with_overlay"],
        atol=1 / 255,
        rtol=0.0,
    )


def test_overlay_modifies_masked_pixels() -> None:
    """Pixels inside the object mask differ from the original frame after blending."""
    # draw_contours/draw_ids disabled so no pixels outside the mask region are touched.
    node = TrackingOverlayNode(alpha=1.0, draw_contours=False, draw_ids=False)
    h, w = 10, 10
    rgb = torch.ones((1, h, w, 3), dtype=torch.float32) * 0.5
    mask = torch.zeros((1, h, w), dtype=torch.int32)
    mask[0, :5, :] = 1  # top half → object 1

    out = node.forward(
        rgb_image=rgb,
        mask=mask,
        object_ids=torch.tensor([[1]], dtype=torch.int64),
    )

    result = out["rgb_with_overlay"][0]
    top_half = result[:5, :]
    bottom_half = result[5:, :]

    assert not torch.allclose(top_half, torch.full_like(top_half, 0.5), atol=1 / 255)
    torch.testing.assert_close(
        bottom_half, torch.full_like(bottom_half, 0.5), atol=1 / 255, rtol=0.0
    )


def test_multiple_objects_get_distinct_colors() -> None:
    """Two non-overlapping objects are blended to distinct average colors."""
    node = TrackingOverlayNode(alpha=1.0)
    h, w = 20, 10
    rgb = torch.ones((1, h, w, 3), dtype=torch.float32) * 0.5
    mask = torch.zeros((1, h, w), dtype=torch.int32)
    mask[0, :10, :] = 1
    mask[0, 10:, :] = 2

    out = node.forward(
        rgb_image=rgb,
        mask=mask,
        object_ids=torch.tensor([[1, 2]], dtype=torch.int64),
    )

    result = out["rgb_with_overlay"][0]
    color1 = result[:10, :].mean(dim=(0, 1))
    color2 = result[10:, :].mean(dim=(0, 1))
    assert not torch.allclose(color1, color2, atol=1e-3)


# ---------------------------------------------------------------------------
# Constructor parameters
# ---------------------------------------------------------------------------


def test_constructor_stores_alpha() -> None:
    node = TrackingOverlayNode(alpha=0.7)
    assert node.alpha == pytest.approx(0.7)


def test_constructor_stores_draw_flags() -> None:
    node = TrackingOverlayNode(draw_contours=False, draw_ids=False)
    assert node.draw_contours is False
    assert node.draw_ids is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_pixel_object() -> None:
    """One-pixel mask should not raise."""
    node = TrackingOverlayNode()
    h, w = 8, 8
    rgb = _make_frame(h, w)
    mask = torch.zeros((1, h, w), dtype=torch.int32)
    mask[0, 4, 4] = 3

    out = node.forward(
        rgb_image=rgb,
        mask=mask,
        object_ids=torch.tensor([[3]], dtype=torch.int64),
    )
    assert out["rgb_with_overlay"].shape == (1, h, w, 3)


def test_large_object_id_wraps_palette() -> None:
    """Object IDs exceeding the 12-colour palette should wrap without error."""
    node = TrackingOverlayNode()
    oid = 99
    rgb = _make_frame()
    mask = _make_mask(obj_ids=[oid])

    out = node.forward(
        rgb_image=rgb,
        mask=mask,
        object_ids=torch.tensor([[oid]], dtype=torch.int64),
    )
    assert out["rgb_with_overlay"].shape == rgb.shape


def test_forward_inside_no_grad() -> None:
    """forward() must work inside torch.no_grad() without error."""
    node = TrackingOverlayNode()
    rgb = _make_frame()
    mask = _make_mask(obj_ids=[1])

    with torch.no_grad():
        out = node.forward(
            rgb_image=rgb,
            mask=mask,
            object_ids=torch.tensor([[1]], dtype=torch.int64),
        )
    assert out["rgb_with_overlay"].shape == rgb.shape
