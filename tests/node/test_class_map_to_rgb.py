"""Tests for ClassMapToRGB: colourise an integer class-index map."""

from __future__ import annotations

import torch

from cuvis_ai.node.colormap import _TAB20, ClassMapToRGB

B, H, W = 1, 2, 3


def test_explicit_palette_maps_ids_to_colors():
    """Each id picks its palette colour (normalised to [0, 1]); id 0 is a valid class."""
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    class_map = torch.tensor([[[0, 1, 2], [2, 1, 0]]], dtype=torch.int32)
    out = ClassMapToRGB(palette=palette).forward(class_map=class_map)["label_rgb"]

    assert out.shape == (B, H, W, 3) and out.dtype == torch.float32
    assert torch.allclose(out[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(out[0, 0, 1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(out[0, 0, 2], torch.tensor([0.0, 0.0, 1.0]))


def test_background_value_renders_black():
    """Pixels equal to background_value are black; default -1 keeps 0 a real class."""
    class_map = torch.tensor([[[0, -1, 1]]], dtype=torch.int32)
    out = ClassMapToRGB(palette=[(255, 0, 0), (0, 255, 0)]).forward(class_map=class_map)[
        "label_rgb"
    ]
    assert torch.allclose(out[0, 0, 1], torch.zeros(3))  # background_value -> black
    assert not torch.allclose(out[0, 0, 0], torch.zeros(3))  # id 0 -> colour


def test_mask_blacks_out_non_foreground():
    """When a mask is supplied, mask==0 pixels render black even for valid classes."""
    class_map = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.int32)
    mask = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.int32)
    out = ClassMapToRGB(palette=[(255, 0, 0), (0, 255, 0)]).forward(class_map=class_map, mask=mask)[
        "label_rgb"
    ]
    assert torch.allclose(out[0, 0, 1], torch.zeros(3))
    assert torch.allclose(out[0, 1, 1], torch.zeros(3))
    assert not torch.allclose(out[0, 0, 0], torch.zeros(3))


def test_default_palette_is_tab20_and_wraps():
    """No palette -> Tableau-20, deterministic; ids beyond the palette wrap modulo."""
    class_map = torch.tensor([[[0, len(_TAB20)]]], dtype=torch.int32)  # 0 and 20 -> same colour
    out = ClassMapToRGB().forward(class_map=class_map)["label_rgb"]
    expected0 = torch.tensor([c / 255.0 for c in _TAB20[0]])
    assert torch.allclose(out[0, 0, 0], expected0, atol=1e-6)
    assert torch.allclose(out[0, 0, 1], expected0, atol=1e-6)  # wrapped
