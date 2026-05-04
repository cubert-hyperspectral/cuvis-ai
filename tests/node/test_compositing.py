from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.compositing import InsetComposer, ROIZoomNode


def test_roi_zoom_crops_resizes_and_blanks_invalid_frames() -> None:
    source = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    source[0, 1:3, 1:3] = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32)
    bbox = torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)
    valid = torch.tensor([1, 0], dtype=torch.int32)
    node = ROIZoomNode(zoom_height=8, zoom_width=8, bg_color=(0.25, 0.5, 0.75))

    out = node.forward(source=source, bbox=bbox, valid=valid)["zoom"]

    assert out.shape == (2, 8, 8, 3)
    expected_crop = torch.tensor([0.4, 0.5, 0.6], dtype=source.dtype).expand(8, 8, 3)
    assert torch.allclose(out[0], expected_crop)
    expected_bg = torch.tensor([0.25, 0.5, 0.75], dtype=source.dtype).expand(8, 8, 3)
    assert torch.allclose(out[1], expected_bg)


def test_roi_zoom_clamps_out_of_bounds_bbox() -> None:
    source = torch.ones((1, 8, 8, 3), dtype=torch.float32)
    bbox = torch.tensor([[-2.0, -2.0, 5.0, 5.0]], dtype=torch.float32)
    node = ROIZoomNode(zoom_height=8, zoom_width=8)

    out = node.forward(source=source, bbox=bbox)["zoom"]

    assert torch.allclose(out, torch.ones_like(out))


def test_roi_zoom_invalid_or_empty_bbox_returns_background() -> None:
    source = torch.ones((1, 8, 8, 3), dtype=torch.float32)
    bbox = torch.tensor([[2.0, 2.0, 1.0, 1.0]], dtype=torch.float32)
    node = ROIZoomNode(zoom_height=8, zoom_width=8, bg_color=(0.1, 0.2, 0.3))

    out = node.forward(source=source, bbox=bbox)["zoom"]

    expected = torch.tensor([0.1, 0.2, 0.3], dtype=source.dtype).expand(1, 8, 8, 3)
    assert torch.allclose(out, expected)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"zoom_height": 7}, "zoom dimensions"),
        ({"zoom_width": 7}, "zoom dimensions"),
        ({"bg_color": (0.0, 0.0)}, "bg_color"),
    ],
)
def test_roi_zoom_validates_constructor_inputs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ROIZoomNode(**kwargs)


@pytest.mark.parametrize("corner", ["top-left", "top-right", "bottom-left", "bottom-right"])
def test_inset_composer_pastes_inset_into_corner(corner: str) -> None:
    base = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    inset = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
    inset[0, :, :, 0] = 0.9
    node = InsetComposer(corner=corner, margin_px=2, border_px=0)

    out = node.forward(base=base, inset=inset)["composite"]

    expected_y0 = 2 if corner.startswith("top") else 16 - 2 - 4
    expected_x0 = 2 if corner.endswith("left") else 16 - 2 - 4
    pasted = out[0, expected_y0 : expected_y0 + 4, expected_x0 : expected_x0 + 4]
    assert torch.allclose(pasted, inset[0])

    untouched = out.clone()
    untouched[0, expected_y0 : expected_y0 + 4, expected_x0 : expected_x0 + 4] = 0.0
    assert torch.allclose(untouched, torch.zeros_like(out))


def test_inset_composer_paints_border() -> None:
    base = torch.zeros((1, 20, 20, 3), dtype=torch.float32)
    inset = torch.full((1, 4, 4, 3), 0.25, dtype=torch.float32)
    node = InsetComposer(
        corner="top-left",
        margin_px=2,
        border_px=2,
        border_color=(1.0, 0.0, 0.0),
    )

    out = node.forward(base=base, inset=inset)["composite"]

    border_top = out[0, 2, 2:8]
    expected_red = torch.tensor([1.0, 0.0, 0.0]).expand(6, 3)
    assert torch.allclose(border_top, expected_red)

    inner = out[0, 4:8, 4:8]
    assert torch.allclose(inner, inset[0])


def test_inset_composer_passes_through_when_invalid() -> None:
    base = torch.full((2, 16, 16, 3), 0.3, dtype=torch.float32)
    inset = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    valid = torch.tensor([1, 0], dtype=torch.int32)
    node = InsetComposer(corner="top-right", margin_px=2, border_px=0)

    out = node.forward(base=base, inset=inset, valid=valid)["composite"]

    assert torch.allclose(out[1], base[1])
    pasted = out[0, 2:6, 16 - 2 - 4 : 16 - 2]
    assert torch.allclose(pasted, inset[0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"corner": "middle"}, "corner must be one of"),
        ({"margin_px": -1}, "margin_px"),
        ({"border_px": -1}, "border_px"),
        ({"border_color": (1.0, 0.0)}, "border_color"),
    ],
)
def test_inset_composer_validates_constructor_inputs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        InsetComposer(**kwargs)


def test_inset_composer_rejects_oversized_inset() -> None:
    base = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    inset = torch.zeros((1, 6, 6, 3), dtype=torch.float32)
    node = InsetComposer(corner="top-left", margin_px=2, border_px=0)

    with pytest.raises(ValueError, match="does not fit"):
        node.forward(base=base, inset=inset)
