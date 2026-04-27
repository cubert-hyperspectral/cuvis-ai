from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.compositing import ROIZoomNode


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
