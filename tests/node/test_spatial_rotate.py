"""Tests for SpatialRotateNode.

Uses non-square dimensions (H=6, W=10) throughout to catch H/W swap bugs.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.band_selection import RangeAverageFalseRGBSelector
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.preprocessors import SpatialRotateNode
from cuvis_ai.node.visualizations import MaskOverlayNode

B, H, W, C = 2, 6, 10, 20


# -- helpers -----------------------------------------------------------------


def _make_cube_with_marker():
    """Cube where [b, 0, 0, :] = 1.0 and rest = 0.0 — a known corner pixel."""
    cube = torch.zeros(B, H, W, C, dtype=torch.float32)
    cube[:, 0, 0, :] = 1.0
    return cube


def _make_mask_with_rect():
    """Mask with foreground at rows [1:4], cols [3:7]."""
    mask = torch.zeros(B, H, W, dtype=torch.int32)
    mask[:, 1:4, 3:7] = 1
    return mask


# -- rotation correctness ---------------------------------------------------


def test_rotate_cube_90(create_test_cube):
    cube, _ = create_test_cube(batch_size=B, height=H, width=W, num_channels=C, mode="random")
    node = SpatialRotateNode(rotation=90)
    out = node.forward(cube=cube)
    assert out["cube"].shape == (B, W, H, C)
    # rot90 k=1 on dims(1,2): out[b,r,c,ch] = cube[b, c, W-1-r, ch]
    assert torch.equal(out["cube"][:, 0, 0, :], cube[:, 0, W - 1, :])


def test_rotate_cube_minus_90(create_test_cube):
    cube, _ = create_test_cube(batch_size=B, height=H, width=W, num_channels=C, mode="random")
    node = SpatialRotateNode(rotation=-90)
    out = node.forward(cube=cube)
    assert out["cube"].shape == (B, W, H, C)
    # rot90 k=-1 on dims(1,2): out[b,r,c,ch] = cube[b, H-1-c, r, ch]
    assert torch.equal(out["cube"][:, 0, 0, :], cube[:, H - 1, 0, :])


def test_rotate_cube_180(create_test_cube):
    cube, _ = create_test_cube(batch_size=B, height=H, width=W, num_channels=C, mode="random")
    node = SpatialRotateNode(rotation=180)
    out = node.forward(cube=cube)
    assert out["cube"].shape == (B, H, W, C)
    # rot90 k=2: new[r,c] = old[H-1-r, W-1-c]
    assert torch.equal(out["cube"][:, 0, 0, :], cube[:, H - 1, W - 1, :])


def test_mask_rotates_with_cube():
    cube = _make_cube_with_marker()
    mask = _make_mask_with_rect()

    node = SpatialRotateNode(rotation=90)
    out = node.forward(cube=cube, mask=mask)

    rotated_cube = out["cube"]
    rotated_mask = out["mask"]

    # After 90 CCW: shape flips to [B, W, H]
    assert rotated_mask.shape == (B, W, H)

    # Original fg at rows [1:4], cols [3:7]
    # After rot90 k=1 on dims (1,2): new[r,c] = old[W-1-c, r]
    # Foreground should now be at rows [W-1-7 : W-1-3] = [3:7], cols [1:4]
    expected = torch.zeros(B, W, H, dtype=torch.int32)
    expected[:, 3:7, 1:4] = 1
    assert torch.equal(rotated_mask, expected)

    # Cube corner marker should also have moved
    assert torch.all(rotated_cube[:, W - 1, 0, :] == 1.0)


def test_no_wavelengths_port():
    """SpatialRotateNode only handles spatial data, not wavelengths."""
    cube = torch.zeros(B, H, W, C, dtype=torch.float32)
    node = SpatialRotateNode(rotation=90)
    out = node.forward(cube=cube)
    assert "wavelengths" not in out


# -- passthrough -------------------------------------------------------------


def test_passthrough_none_rotation():
    cube = _make_cube_with_marker()
    node = SpatialRotateNode(rotation=None)
    out = node.forward(cube=cube)
    assert out["cube"] is cube


def test_passthrough_zero_rotation():
    cube = _make_cube_with_marker()
    node = SpatialRotateNode(rotation=0)
    out = node.forward(cube=cube)
    assert out["cube"] is cube


# -- validation --------------------------------------------------------------


def test_invalid_rotation_raises():
    with pytest.raises(ValueError, match="rotation"):
        SpatialRotateNode(rotation=45)


def test_rotation_aliases():
    assert SpatialRotateNode(rotation=270).rotation == -90
    assert SpatialRotateNode(rotation=-270).rotation == 90
    assert SpatialRotateNode(rotation=-180).rotation == 180


# -- optional inputs ---------------------------------------------------------


def test_optional_inputs_omitted():
    cube = torch.zeros(B, H, W, C, dtype=torch.float32)
    node = SpatialRotateNode(rotation=90)
    out = node.forward(cube=cube)
    assert set(out.keys()) == {"cube"}


def test_cube_and_mask_only():
    cube = torch.zeros(B, H, W, C, dtype=torch.float32)
    mask = _make_mask_with_rect()
    node = SpatialRotateNode(rotation=90)
    out = node.forward(cube=cube, mask=mask)
    assert set(out.keys()) == {"cube", "mask"}


# -- end-to-end with pipeline nodes -----------------------------------------


def test_end_to_end_rotated_overlay(create_test_cube):
    """CU3SDataNode -> SpatialRotateNode -> FalseRGB -> MaskOverlay."""
    cube, wavelengths = create_test_cube(
        batch_size=1,
        height=H,
        width=W,
        num_channels=C,
        mode="random",
        dtype=torch.uint16,
    )
    mask = torch.zeros(1, H, W, dtype=torch.int32)
    mask[0, 1:4, 3:7] = 1

    # Step 1: CU3SDataNode
    data_node = CU3SDataNode()
    data_out = data_node.forward(cube=cube, mask=mask, wavelengths=wavelengths)

    # Step 2: SpatialRotateNode (90 degrees) — only spatial data
    rotate = SpatialRotateNode(rotation=90)
    rot_out = rotate.forward(cube=data_out["cube"], mask=data_out["mask"])

    assert rot_out["cube"].shape == (1, W, H, C)
    assert rot_out["mask"].shape == (1, W, H)

    # Step 3: False RGB — wavelengths bypass rotate
    false_rgb = RangeAverageFalseRGBSelector()
    rgb_out = false_rgb.forward(cube=rot_out["cube"], wavelengths=data_out["wavelengths"])
    rgb_image = rgb_out["rgb_image"]
    assert rgb_image.shape == (1, W, H, 3)

    # Step 4: Mask overlay
    overlay = MaskOverlayNode(alpha=0.4)
    overlay_out = overlay.forward(rgb_image=rgb_image, mask=rot_out["mask"])
    result = overlay_out["rgb_with_overlay"]

    # Rotated mask fg should be at rows [3:7], cols [1:4] in the [W, H] space
    fg_mask = rot_out["mask"][0] > 0
    fg_input = rgb_image[0][fg_mask]
    fg_output = result[0][fg_mask]
    assert not torch.allclose(fg_input, fg_output), "Overlay not applied to foreground"

    bg_mask = ~fg_mask
    assert torch.allclose(rgb_image[0][bg_mask], result[0][bg_mask]), "Background was modified"
