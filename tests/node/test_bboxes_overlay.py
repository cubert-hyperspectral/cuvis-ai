"""Tests for BBoxesOverlayNode (torch-only bbox rendering)."""

from __future__ import annotations

import torch

import cuvis_ai.node.anomaly_visualization as anomaly_viz
from cuvis_ai.node.anomaly_visualization import BBoxesOverlayNode


def test_bboxes_overlay_specs() -> None:
    node = BBoxesOverlayNode()

    assert "rgb_image" in node.INPUT_SPECS
    assert "bboxes" in node.INPUT_SPECS
    assert "category_ids" in node.INPUT_SPECS
    assert "confidences" in node.INPUT_SPECS
    assert node.INPUT_SPECS["rgb_image"].shape == (1, -1, -1, 3)
    assert node.INPUT_SPECS["bboxes"].shape == (1, -1, 4)
    assert node.INPUT_SPECS["category_ids"].shape == (1, -1)
    assert node.INPUT_SPECS["confidences"].optional is True

    assert "rgb_with_overlay" in node.OUTPUT_SPECS
    assert node.OUTPUT_SPECS["rgb_with_overlay"].shape == (1, -1, -1, 3)


def test_bboxes_overlay_forward() -> None:
    node = BBoxesOverlayNode(line_thickness=2)
    rgb = torch.full((1, 16, 20, 3), 0.5, dtype=torch.float32)
    bboxes = torch.tensor([[[2.0, 3.0, 10.0, 12.0], [0.0, 0.0, 4.0, 4.0]]], dtype=torch.float32)
    category_ids = torch.tensor([[1, 4]], dtype=torch.int64)

    out = node.forward(rgb_image=rgb, bboxes=bboxes, category_ids=category_ids)
    rendered = out["rgb_with_overlay"]

    assert rendered.shape == rgb.shape
    assert rendered.dtype == torch.float32
    assert float(rendered.min()) >= 0.0
    assert float(rendered.max()) <= 1.0
    assert not torch.allclose(rendered, rgb)


def test_bbox_overlay_uses_torch_draw_primitives(monkeypatch) -> None:
    calls = {"id_to_color": 0, "draw_box": 0}

    def fake_id_to_color(ids: torch.Tensor) -> torch.Tensor:
        calls["id_to_color"] += 1
        return torch.full((ids.shape[0], 3), 255, dtype=torch.uint8, device=ids.device)

    def fake_draw_box(
        img: torch.Tensor,
        box_xyxy: tuple[int, int, int, int],
        color: torch.Tensor,
        thickness: int = 2,
    ) -> None:
        calls["draw_box"] += 1
        img[0, 0, :] = color

    monkeypatch.setattr(anomaly_viz, "id_to_color", fake_id_to_color)
    monkeypatch.setattr(anomaly_viz, "draw_box", fake_draw_box)

    node = BBoxesOverlayNode(line_thickness=3)
    rgb = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    bboxes = torch.tensor([[[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 6.0, 6.0]]], dtype=torch.float32)
    category_ids = torch.tensor([[7, 8]], dtype=torch.int64)

    out = node.forward(rgb_image=rgb, bboxes=bboxes, category_ids=category_ids)

    assert calls["id_to_color"] == 1
    assert calls["draw_box"] == 2
    assert out["rgb_with_overlay"].shape == rgb.shape
