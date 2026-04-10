"""Tests for BBoxRoiCropNode."""

from __future__ import annotations

import torch

from cuvis_ai.node.preprocessors import BBoxRoiCropNode


def _make_images(B: int = 1, H: int = 64, W: int = 64, C: int = 3) -> torch.Tensor:
    """Random BHWC images in [0, 1]."""
    return torch.rand(B, H, W, C)


def _make_bboxes(B: int, boxes_per_batch: list[list[list[float]]]) -> torch.Tensor:
    """Build [B, N, 4] bbox tensor from per-batch box lists."""
    max_n = max(len(b) for b in boxes_per_batch)
    t = torch.zeros(B, max_n, 4)
    for b, boxes in enumerate(boxes_per_batch):
        for i, box in enumerate(boxes):
            t[b, i] = torch.tensor(box)
    return t


def test_output_shape() -> None:
    """[1,H,W,3] + [1,N,4] → [N, 3, crop_h, crop_w]."""
    node = BBoxRoiCropNode(output_size=(256, 128))
    images = _make_images(1, 480, 640, 3)
    bboxes = torch.tensor([[[10.0, 10.0, 60.0, 80.0], [100.0, 100.0, 200.0, 250.0]]])
    result = node.forward(images=images, bboxes=bboxes)
    crops = result["crops"]
    assert crops.shape == (2, 3, 256, 128)


def test_zero_detections() -> None:
    """N=0 → [0, 3, crop_h, crop_w]."""
    node = BBoxRoiCropNode(output_size=(256, 128))
    images = _make_images(1, 64, 64, 3)
    bboxes = torch.empty(1, 0, 4)
    result = node.forward(images=images, bboxes=bboxes)
    crops = result["crops"]
    assert crops.shape == (0, 3, 256, 128)


def test_multi_batch() -> None:
    """B=2: crops from correct source images."""
    node = BBoxRoiCropNode(output_size=(32, 16))
    images = _make_images(2, 64, 64, 3)
    bboxes = _make_bboxes(
        2,
        [
            [[5.0, 5.0, 30.0, 30.0]],
            [[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]],
        ],
    )
    result = node.forward(images=images, bboxes=bboxes)
    crops = result["crops"]
    # Batch 0 has 1 valid box, batch 1 has 2 valid boxes = 3 total
    assert crops.shape[0] == 3
    assert crops.shape[1:] == (3, 32, 16)


def test_padding_filter() -> None:
    """YOLO -1 padded rows filtered out; N_out < N_padded."""
    node = BBoxRoiCropNode(output_size=(32, 16))
    images = _make_images(1, 64, 64, 3)
    # 2 valid boxes + 1 padding row (all -1)
    bboxes = torch.tensor(
        [[[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0], [-1.0, -1.0, -1.0, -1.0]]]
    )
    result = node.forward(images=images, bboxes=bboxes)
    crops = result["crops"]
    assert crops.shape[0] == 2  # padding row filtered


def test_differentiable() -> None:
    """images.requires_grad_(True) → backward through roi_align → grad is not None."""
    node = BBoxRoiCropNode(output_size=(32, 16))
    images = _make_images(1, 64, 64, 3).requires_grad_(True)
    bboxes = torch.tensor([[[5.0, 5.0, 30.0, 30.0]]])
    result = node.forward(images=images, bboxes=bboxes)
    loss = result["crops"].sum()
    loss.backward()
    assert images.grad is not None


def test_output_value_range() -> None:
    """Output values stay in [0, 1] for valid input."""
    node = BBoxRoiCropNode(output_size=(32, 16))
    images = _make_images(1, 64, 64, 3)  # values in [0, 1]
    bboxes = torch.tensor([[[5.0, 5.0, 55.0, 55.0]]])
    result = node.forward(images=images, bboxes=bboxes)
    crops = result["crops"]
    assert crops.min() >= 0.0
    assert crops.max() <= 1.0
