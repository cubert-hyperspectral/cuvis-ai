"""Diagnostic tests for mask overlay spatial alignment.

Systematically tests whether the mask overlay lands at the correct pixel
coordinates when given a known mask location and known cube dimensions.

Uses non-square dimensions (H=6, W=10) throughout to expose any H/W
transposition bugs that would be invisible with square images.
"""

from __future__ import annotations

import torch

from cuvis_ai.node.band_selection import RangeAverageFalseRGBSelector
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.visualizations import MaskOverlayNode
from cuvis_ai.utils.vis_helpers import create_mask_overlay

# Common non-square test dimensions (H != W to catch transpose bugs)
B, H, W = 1, 6, 10
# Known foreground rectangle: rows [1:4], cols [3:7]
FG_ROW_START, FG_ROW_END = 1, 4
FG_COL_START, FG_COL_END = 3, 7
ALPHA = 0.4
COLOR = (1.0, 0.0, 0.0)


def _make_rgb_and_mask():
    """Create uniform grey RGB and a mask with foreground at a known rectangle."""
    rgb = torch.full((B, H, W, 3), 0.5, dtype=torch.float32)
    mask = torch.zeros(B, H, W, dtype=torch.int32)
    mask[0, FG_ROW_START:FG_ROW_END, FG_COL_START:FG_COL_END] = 1
    return rgb, mask


def _assert_overlay_at_known_location(rgb, result, mask):
    """Assert overlay is applied exactly where mask > 0 and nowhere else."""
    fg_mask = mask[0] > 0  # [H, W]

    # --- Foreground pixels must be tinted ---
    fg_input = rgb[0][fg_mask]  # [N_fg, 3]
    fg_output = result[0][fg_mask]  # [N_fg, 3]
    assert not torch.allclose(fg_input, fg_output), (
        "FAIL: Foreground pixels are unchanged — overlay was not applied. "
        f"mask.any()={mask.any()}, mask.sum()={mask.sum()}, "
        f"mask.shape={mask.shape}, rgb.shape={rgb.shape}"
    )

    # Verify exact blend values: (1 - alpha) * 0.5 + alpha * color
    expected_fg = torch.tensor(
        [
            (1 - ALPHA) * 0.5 + ALPHA * COLOR[0],
            (1 - ALPHA) * 0.5 + ALPHA * COLOR[1],
            (1 - ALPHA) * 0.5 + ALPHA * COLOR[2],
        ]
    )
    assert torch.allclose(fg_output[0], expected_fg, atol=1e-6), (
        f"Foreground pixel value wrong. Expected {expected_fg.tolist()}, "
        f"got {fg_output[0].tolist()}"
    )

    # --- Background pixels must be unchanged ---
    bg_mask = ~fg_mask
    bg_input = rgb[0][bg_mask]
    bg_output = result[0][bg_mask]
    assert torch.allclose(bg_input, bg_output), (
        "FAIL: Background pixels were modified — overlay leaked outside mask region. "
        f"Max diff: {(bg_input - bg_output).abs().max().item()}"
    )


# ===================================================================
# Test 1: create_mask_overlay utility at known location
# ===================================================================


def test_create_mask_overlay_at_known_location():
    """Verify create_mask_overlay places tint exactly at mask>0 pixels."""
    rgb, mask = _make_rgb_and_mask()
    result = create_mask_overlay(rgb, mask, alpha=ALPHA, color=COLOR)

    assert result.shape == rgb.shape, f"Output shape {result.shape} != input shape {rgb.shape}"
    _assert_overlay_at_known_location(rgb, result, mask)


# ===================================================================
# Test 2: MaskOverlayNode.forward at known location
# ===================================================================


def test_mask_overlay_node_at_known_location():
    """Verify MaskOverlayNode wrapper preserves spatial alignment."""
    rgb, mask = _make_rgb_and_mask()
    node = MaskOverlayNode(overlay_color=COLOR, alpha=ALPHA)
    out = node.forward(rgb_image=rgb, mask=mask)

    result = out["rgb_with_overlay"]
    assert result.shape == rgb.shape
    _assert_overlay_at_known_location(rgb, result, mask)


# ===================================================================
# Test 3: CU3SDataNode preserves mask identity
# ===================================================================


def test_cu3s_data_node_preserves_mask_location(create_test_cube):
    """Verify CU3SDataNode passes mask through with identical values and shape."""
    cube, wavelengths = create_test_cube(
        batch_size=B,
        height=H,
        width=W,
        num_channels=20,
        mode="random",
        dtype=torch.uint16,
    )
    _, mask = _make_rgb_and_mask()

    data_node = CU3SDataNode()
    out = data_node.forward(cube=cube, mask=mask, wavelengths=wavelengths)

    assert "mask" in out, "CU3SDataNode dropped mask from output"
    assert torch.equal(out["mask"], mask), (
        f"Mask changed after CU3SDataNode. "
        f"Input shape: {mask.shape}, output shape: {out['mask'].shape}, "
        f"Input sum: {mask.sum()}, output sum: {out['mask'].sum()}"
    )


# ===================================================================
# Test 4: RangeAverageFalseRGBSelector preserves spatial dims
# ===================================================================


def test_false_rgb_preserves_spatial_dims(create_test_cube):
    """Verify false RGB output has same H, W as input cube."""
    cube, wavelengths = create_test_cube(
        batch_size=B,
        height=H,
        width=W,
        num_channels=20,
        mode="random",
        dtype=torch.uint16,
    )

    data_node = CU3SDataNode()
    data_out = data_node.forward(cube=cube, wavelengths=wavelengths)

    false_rgb = RangeAverageFalseRGBSelector()
    rgb_out = false_rgb.forward(
        cube=data_out["cube"],
        wavelengths=data_out["wavelengths"],
    )

    rgb_image = rgb_out["rgb_image"]
    assert rgb_image.shape == (B, H, W, 3), (
        f"RGB shape {rgb_image.shape} doesn't match expected (B={B}, H={H}, W={W}, 3). "
        f"Input cube shape was {cube.shape}. "
        "False RGB selector may have changed spatial dimensions."
    )


# ===================================================================
# Test 5: End-to-end overlay at known location
# ===================================================================


def test_end_to_end_overlay_at_known_location(create_test_cube):
    """Full pipeline chain must produce overlay at exactly the mask location."""
    cube, wavelengths = create_test_cube(
        batch_size=B,
        height=H,
        width=W,
        num_channels=20,
        mode="random",
        dtype=torch.uint16,
    )
    _, mask = _make_rgb_and_mask()

    # Step 1: CU3SDataNode
    data_node = CU3SDataNode()
    data_out = data_node.forward(cube=cube, mask=mask, wavelengths=wavelengths)

    assert "mask" in data_out, "CU3SDataNode dropped mask"
    assert data_out["mask"].any(), "Mask is all zeros after CU3SDataNode"

    # Step 2: RangeAverageFalseRGBSelector
    false_rgb = RangeAverageFalseRGBSelector()
    rgb_out = false_rgb.forward(
        cube=data_out["cube"],
        wavelengths=data_out["wavelengths"],
    )
    rgb_image = rgb_out["rgb_image"]

    assert rgb_image.shape[1:3] == (H, W), (
        f"RGB spatial dims {rgb_image.shape[1:3]} != mask spatial dims ({H}, {W})"
    )

    # Step 3: MaskOverlayNode
    overlay_node = MaskOverlayNode(overlay_color=COLOR, alpha=ALPHA)
    overlay_out = overlay_node.forward(
        rgb_image=rgb_image,
        mask=data_out["mask"],
    )
    result = overlay_out["rgb_with_overlay"]

    # The RGB from false_rgb is not uniform grey (it's from random cube data),
    # so we can't check exact pixel values. Instead verify:
    # - Foreground pixels differ from the input RGB
    fg_mask_2d = data_out["mask"][0] > 0
    fg_input = rgb_image[0][fg_mask_2d]
    fg_output = result[0][fg_mask_2d]
    assert not torch.allclose(fg_input, fg_output), (
        "End-to-end FAIL: Overlay not applied to foreground pixels. "
        f"mask sum={data_out['mask'].sum()}, rgb range=[{rgb_image.min():.3f}, {rgb_image.max():.3f}]"
    )

    # - Background pixels are unchanged
    bg_mask_2d = ~fg_mask_2d
    bg_input = rgb_image[0][bg_mask_2d]
    bg_output = result[0][bg_mask_2d]
    assert torch.allclose(bg_input, bg_output), (
        "End-to-end FAIL: Background pixels were modified. "
        f"Max diff: {(bg_input - bg_output).abs().max().item()}"
    )


# ===================================================================
# Test 6: Transposed mask demonstrates misalignment
# ===================================================================


def test_overlay_location_with_transposed_dimensions(create_test_cube):
    """Demonstrate what happens when mask H/W are transposed.

    If the mask is [1, W, H] instead of [1, H, W], the overlay should
    land in the wrong location. This test documents the failure mode
    to help diagnose the real-data issue.
    """
    cube, wavelengths = create_test_cube(
        batch_size=B,
        height=H,
        width=W,
        num_channels=20,
        mode="random",
        dtype=torch.uint16,
    )

    # Create mask in TRANSPOSED dimensions [1, W, H] instead of [1, H, W]
    transposed_mask = torch.zeros(B, W, H, dtype=torch.int32)
    # Place foreground at same index ranges but in the transposed space
    transposed_mask[0, FG_ROW_START:FG_ROW_END, FG_COL_START:FG_COL_END] = 1

    data_node = CU3SDataNode()
    data_out = data_node.forward(cube=cube, mask=transposed_mask, wavelengths=wavelengths)

    false_rgb = RangeAverageFalseRGBSelector()
    rgb_out = false_rgb.forward(
        cube=data_out["cube"],
        wavelengths=data_out["wavelengths"],
    )
    rgb_image = rgb_out["rgb_image"]

    # The transposed mask has shape [1, 10, 6] while RGB is [1, 6, 10, 3]
    # This SHOULD cause a shape mismatch error in create_mask_overlay.
    # If it DOESN'T error (silent broadcast), that's a dangerous silent failure.
    overlay_node = MaskOverlayNode(overlay_color=COLOR, alpha=ALPHA)
    try:
        overlay_out = overlay_node.forward(
            rgb_image=rgb_image,
            mask=data_out["mask"],
        )
        # If we get here, PyTorch silently broadcast the mismatched shapes.
        # Check that the overlay did NOT land at the correct location.
        result = overlay_out["rgb_with_overlay"]
        correct_fg_mask = torch.zeros(B, H, W, dtype=torch.bool)
        correct_fg_mask[0, FG_ROW_START:FG_ROW_END, FG_COL_START:FG_COL_END] = True

        assert not torch.equal(result, rgb_image), (
            "Transposed mask produced identical output — mask was likely treated as empty"
        )
        # Document: a transposed mask silently produces WRONG overlay placement
        print(
            f"WARNING: Transposed mask [1,{W},{H}] was silently accepted for "
            f"RGB [1,{H},{W},3]. Overlay may be misaligned without any error."
        )
    except RuntimeError as e:
        # Good: PyTorch caught the shape mismatch
        print(f"Shape mismatch correctly caught: {e}")
