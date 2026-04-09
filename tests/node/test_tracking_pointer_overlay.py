"""Tests for TrackingPointerOverlayNode."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.anomaly_visualization import TrackingPointerOverlayNode


def _make_frame(h: int = 48, w: int = 60) -> torch.Tensor:
    """Create a float32 RGB frame [1, H, W, 3] in [0, 1]."""
    return torch.rand((1, h, w, 3), dtype=torch.float32)


def _make_mask(h: int = 48, w: int = 60) -> torch.Tensor:
    """Create a label map with two rectangular objects."""
    mask = torch.zeros((1, h, w), dtype=torch.int32)
    mask[0, 18:30, 4:12] = 1
    mask[0, 18:30, 36:44] = 2
    return mask


def test_output_shape_dtype_and_range() -> None:
    node = TrackingPointerOverlayNode()
    rgb = _make_frame()
    mask = _make_mask()

    out = node.forward(
        rgb_image=rgb,
        mask=mask,
        object_ids=torch.tensor([[1, 2]], dtype=torch.int64),
    )["rgb_with_overlay"]

    assert out.shape == rgb.shape
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_empty_mask_matches_input() -> None:
    node = TrackingPointerOverlayNode()
    rgb = _make_frame()
    mask = torch.zeros((1, 48, 60), dtype=torch.int32)

    out = node.forward(rgb_image=rgb, mask=mask)["rgb_with_overlay"]
    torch.testing.assert_close(out, rgb, atol=1 / 255, rtol=0.0)


def test_all_objects_get_pointers() -> None:
    node = TrackingPointerOverlayNode()
    rgb = torch.zeros((1, 48, 60, 3), dtype=torch.float32)
    mask = _make_mask()

    out = node.forward(rgb_image=rgb, mask=mask)["rgb_with_overlay"][0]
    rendered = torch.round(out * 255.0).to(torch.uint8)

    assert torch.any(rendered[0:16, 0:20, :] > 0)
    assert torch.any(rendered[0:16, 30:50, :] > 0)


def test_object_ids_input_filters_to_present() -> None:
    node = TrackingPointerOverlayNode()
    rgb = torch.zeros((1, 48, 60, 3), dtype=torch.float32)
    mask = _make_mask()

    out = node.forward(
        rgb_image=rgb,
        mask=mask,
        object_ids=torch.tensor([[2]], dtype=torch.int64),
    )["rgb_with_overlay"][0]
    rendered = torch.round(out * 255.0).to(torch.uint8)

    assert not torch.any(rendered[0:16, 0:20, :])
    assert torch.any(rendered[0:16, 30:50, :] > 0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_outputs_close() -> None:
    torch.manual_seed(13)
    node = TrackingPointerOverlayNode()
    rgb_cpu = _make_frame()
    mask_cpu = _make_mask()
    ids_cpu = torch.tensor([[1, 2]], dtype=torch.int64)

    out_cpu = node.forward(
        rgb_image=rgb_cpu,
        mask=mask_cpu,
        object_ids=ids_cpu,
    )["rgb_with_overlay"]

    out_cuda = node.forward(
        rgb_image=rgb_cpu.to("cuda"),
        mask=mask_cpu.to("cuda"),
        object_ids=ids_cpu.to("cuda"),
    )["rgb_with_overlay"].cpu()

    torch.testing.assert_close(out_cpu, out_cuda, atol=1 / 255, rtol=0.0)
