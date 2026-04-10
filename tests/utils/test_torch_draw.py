from __future__ import annotations

import pytest
import torch

from cuvis_ai.utils.torch_draw import (
    draw_box,
    draw_downward_triangle,
    draw_sparkline,
    draw_text,
    id_to_color,
    mask_edge,
    overlay_instances,
)
from cuvis_ai.utils.vis_helpers import OBJECT_PALETTE


def test_mask_edge_is_subset_of_mask() -> None:
    mask = torch.zeros((10, 12), dtype=torch.bool)
    mask[2:8, 3:9] = True
    edge = mask_edge(mask, thickness=2)
    assert torch.all(edge <= mask)


def test_mask_edge_empty_for_empty_mask() -> None:
    mask = torch.zeros((8, 8), dtype=torch.bool)
    edge = mask_edge(mask)
    assert not torch.any(edge)


def test_mask_edge_single_pixel() -> None:
    mask = torch.zeros((5, 5), dtype=torch.bool)
    mask[2, 2] = True
    edge = mask_edge(mask)
    assert edge.shape == mask.shape


def test_draw_box_places_colored_pixels() -> None:
    img = torch.zeros((12, 12, 3), dtype=torch.uint8)
    draw_box(img, (2, 3, 9, 10), (255, 0, 0), thickness=2)
    assert torch.any(img[..., 0] > 0)


def test_draw_box_clamps_to_bounds() -> None:
    img = torch.zeros((8, 8, 3), dtype=torch.uint8)
    draw_box(img, (-20, -10, 40, 30), (0, 255, 0), thickness=3)
    assert img.shape == (8, 8, 3)


def test_draw_text_modifies_image() -> None:
    img = torch.zeros((24, 48, 3), dtype=torch.uint8)
    draw_text(img, 4, 4, "123", (255, 255, 255), scale=2, bg=True)
    assert torch.any(img > 0)


def test_draw_text_out_of_bounds() -> None:
    img = torch.zeros((10, 12, 3), dtype=torch.uint8)
    draw_text(img, 10, 8, "99", (255, 255, 255), scale=2, bg=True)
    assert img.shape == (10, 12, 3)


def test_draw_downward_triangle_modifies_pixels() -> None:
    img = torch.zeros((24, 24, 3), dtype=torch.uint8)
    draw_downward_triangle(
        img,
        tip_x=12,
        tip_y=18,
        width=12,
        height=10,
        color=(255, 0, 0),
        outline_color=(0, 0, 0),
        outline_thickness=1,
    )
    assert torch.any(img[..., 0] > 0)


def test_draw_downward_triangle_clamps_to_top_left_bounds() -> None:
    img = torch.zeros((12, 12, 3), dtype=torch.uint8)
    draw_downward_triangle(
        img,
        tip_x=2,
        tip_y=5,
        width=10,
        height=8,
        color=(0, 255, 0),
        outline_color=(0, 0, 0),
        outline_thickness=1,
    )
    assert img.shape == (12, 12, 3)
    assert torch.any(img[..., 1] > 0)


def test_draw_downward_triangle_clamps_to_right_bound() -> None:
    img = torch.zeros((14, 14, 3), dtype=torch.uint8)
    draw_downward_triangle(
        img,
        tip_x=12,
        tip_y=10,
        width=12,
        height=9,
        color=(0, 0, 255),
        outline_color=(0, 0, 0),
        outline_thickness=1,
    )
    assert img.shape == (14, 14, 3)
    assert torch.any(img[..., 2] > 0)


def test_id_to_color_deterministic() -> None:
    ids = torch.tensor([0, 1, 2, 15, 101], dtype=torch.int64)
    c1 = id_to_color(ids)
    c2 = id_to_color(ids)
    torch.testing.assert_close(c1, c2)


def test_id_to_color_distinct() -> None:
    ids = torch.tensor([100, 101, 102], dtype=torch.int64)
    colors = id_to_color(ids)
    assert not torch.equal(colors[0], colors[1])
    assert not torch.equal(colors[1], colors[2])


def test_id_to_color_palette_compat() -> None:
    ids = torch.arange(len(OBJECT_PALETTE), dtype=torch.int64)
    colors = id_to_color(ids).cpu()
    palette = torch.tensor(OBJECT_PALETTE, dtype=torch.uint8)
    torch.testing.assert_close(colors, palette)


def test_id_to_color_hash_fallback_for_large_ids() -> None:
    ids = torch.tensor([len(OBJECT_PALETTE) + 1, len(OBJECT_PALETTE) + 2], dtype=torch.int64)
    colors = id_to_color(ids)
    palette = torch.tensor(OBJECT_PALETTE, dtype=torch.uint8)
    assert not torch.equal(colors[0], palette[(len(OBJECT_PALETTE) + 1) % len(OBJECT_PALETTE)])
    assert not torch.equal(colors[1], palette[(len(OBJECT_PALETTE) + 2) % len(OBJECT_PALETTE)])


def test_overlay_instances_output_shape() -> None:
    img = torch.zeros((16, 20, 3), dtype=torch.uint8)
    mask = torch.zeros((16, 20), dtype=torch.bool)
    mask[3:10, 4:12] = True
    out = overlay_instances(img, [(1, mask)])
    assert out.shape == img.shape
    assert out.dtype == torch.uint8


def test_overlay_instances_empty_masks() -> None:
    img = torch.randint(0, 255, (8, 9, 3), dtype=torch.uint8)
    out = overlay_instances(img, [])
    torch.testing.assert_close(out, img)


def test_overlay_instances_modifies_masked_region() -> None:
    img = torch.zeros((10, 10, 3), dtype=torch.uint8)
    mask = torch.zeros((10, 10), dtype=torch.bool)
    mask[:5, :] = True
    out = overlay_instances(img, [(1, mask)], alpha=1.0, draw_edges=False, draw_ids=False)
    assert torch.any(out[:5] != 0)
    assert torch.all(out[5:] == 0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_overlay_instances_cuda_output_device() -> None:
    img = torch.zeros((12, 12, 3), dtype=torch.uint8, device="cuda")
    mask = torch.zeros((12, 12), dtype=torch.bool, device="cuda")
    mask[2:8, 3:9] = True
    out = overlay_instances(img, [(1, mask)], draw_ids=False)
    assert out.device.type == "cuda"


def test_draw_sparkline_modifies_region() -> None:
    img = torch.zeros((32, 48, 3), dtype=torch.uint8)
    values = torch.linspace(0.1, 1.0, 10)
    draw_sparkline(img, 4, 4, 20, 16, values, (255, 128, 0))
    # The sparkline region should have non-zero pixels
    assert torch.any(img[4:20, 4:24, :] > 0)


def test_draw_sparkline_out_of_bounds() -> None:
    img = torch.zeros((10, 12, 3), dtype=torch.uint8)
    values = torch.linspace(0.0, 1.0, 8)
    # Sparkline extends beyond image boundaries — should not crash
    draw_sparkline(img, 8, 6, 20, 20, values, (0, 255, 0))
    assert img.shape == (10, 12, 3)


def test_draw_sparkline_empty_values() -> None:
    img = torch.zeros((16, 16, 3), dtype=torch.uint8)
    original = img.clone()
    # Single element — should be a no-op
    draw_sparkline(img, 2, 2, 10, 8, torch.tensor([0.5]), (255, 0, 0))
    assert torch.equal(img, original)
    # Empty tensor — should be a no-op
    draw_sparkline(img, 2, 2, 10, 8, torch.tensor([]), (255, 0, 0))
    assert torch.equal(img, original)


def test_draw_sparkline_flat_signal() -> None:
    img = torch.full((20, 30, 3), 128, dtype=torch.uint8)
    values = torch.ones(10) * 0.5
    # Flat signal should draw a horizontal line at mid-height without crashing
    draw_sparkline(img, 2, 2, 20, 12, values, (255, 0, 0))
    assert img.shape == (20, 30, 3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_overlay_instances_cpu_cuda_close() -> None:
    torch.manual_seed(7)
    img_cpu = torch.randint(0, 255, (14, 15, 3), dtype=torch.uint8)
    mask_cpu = torch.zeros((14, 15), dtype=torch.bool)
    mask_cpu[2:10, 4:12] = True
    cpu_out = overlay_instances(img_cpu, [(3, mask_cpu)], alpha=0.5, draw_ids=False)

    img_cuda = img_cpu.to("cuda")
    mask_cuda = mask_cpu.to("cuda")
    cuda_out = overlay_instances(img_cuda, [(3, mask_cuda)], alpha=0.5, draw_ids=False).cpu()
    torch.testing.assert_close(
        cpu_out.to(torch.float32), cuda_out.to(torch.float32), atol=1.0, rtol=0.0
    )
