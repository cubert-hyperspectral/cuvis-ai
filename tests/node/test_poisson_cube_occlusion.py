"""Tests for PoissonCubeOcclusionNode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from cuvis_ai_core.data.rle import coco_rle_encode

from cuvis_ai.node.occlusion import PoissonCubeOcclusionNode


def _make_mask(h: int, w: int, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _make_tracking_json(
    tmp_path: Path,
    masks_by_frame: dict[int, list[tuple[int, np.ndarray, list]]],
    *,
    img_h: int,
    img_w: int,
    filename: str = "tracking_poisson.json",
) -> Path:
    images = []
    annotations = []
    ann_id = 1
    for frame_id, entries in sorted(masks_by_frame.items()):
        images.append({"id": frame_id, "width": img_w, "height": img_h})
        for track_id, mask_np, bbox in entries:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": frame_id,
                    "track_id": track_id,
                    "bbox": bbox,
                    "segmentation": coco_rle_encode(mask_np),
                    "area": int(mask_np.sum()),
                    "score": 0.9,
                    "iscrowd": 0,
                    "category_id": 1,
                }
            )
            ann_id += 1
    data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    json_path = tmp_path / filename
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return json_path


def _make_pattern_cube(h: int, w: int, c: int) -> torch.Tensor:
    yy, xx = np.indices((h, w))
    channels = []
    for idx in range(c):
        arr = ((xx * (idx + 3) + yy * (2 * idx + 5) + 17 * idx) % 251).astype(np.float32)
        channels.append(arr / 251.0)
    stacked = np.stack(channels, axis=-1)
    return torch.from_numpy(stacked).unsqueeze(0)


def _frame_id(idx: int) -> torch.Tensor:
    return torch.tensor([idx], dtype=torch.int64)


def test_import() -> None:
    from cuvis_ai.node import PoissonCubeOcclusionNode  # noqa: F401


def test_passthrough_outside_range(tmp_path: Path) -> None:
    h, w, c = 40, 60, 5
    mask = _make_mask(h, w, 10, 20, 20, 35)
    json_path = _make_tracking_json(
        tmp_path,
        {5: [(1, mask, [20, 10, 15, 10])]},
        img_h=h,
        img_w=w,
    )
    node = PoissonCubeOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=10,
    )
    cube = _make_pattern_cube(h, w, c)
    out = node.forward(cube=cube, frame_id=_frame_id(3))
    assert torch.equal(out["cube"], cube)


def test_bbox_inpaint_modifies_region(tmp_path: Path) -> None:
    h, w, c = 40, 60, 7
    mask = _make_mask(h, w, 10, 20, 25, 40)
    json_path = _make_tracking_json(
        tmp_path,
        {5: [(1, mask, [20, 10, 20, 15])]},
        img_h=h,
        img_w=w,
    )
    node = PoissonCubeOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=5,
        occlusion_shape="bbox",
        bbox_mode="dynamic",
        max_iter=2000,
        tol=1e-7,
    )
    cube = _make_pattern_cube(h, w, c)
    out = node.forward(cube=cube, frame_id=_frame_id(5))["cube"]

    bbox_mask = torch.zeros((h, w), dtype=torch.bool)
    bbox_mask[10:25, 20:40] = True
    assert torch.any(out[0][bbox_mask, :] != cube[0][bbox_mask, :])
    assert torch.equal(out[0][~bbox_mask, :], cube[0][~bbox_mask, :])


def test_mask_inpaint_modifies_region(tmp_path: Path) -> None:
    h, w, c = 32, 48, 6
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[9:22, 12:20] = 1
    mask[12:16, 20:30] = 1
    json_path = _make_tracking_json(
        tmp_path,
        {7: [(2, mask, [12, 9, 18, 13])]},
        img_h=h,
        img_w=w,
    )
    node = PoissonCubeOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[2],
        occlusion_start_frame=7,
        occlusion_end_frame=7,
        occlusion_shape="mask",
        max_iter=2500,
        tol=1e-7,
    )
    cube = _make_pattern_cube(h, w, c)
    out = node.forward(cube=cube, frame_id=_frame_id(7))["cube"]

    mask_bool = torch.from_numpy(mask.astype(bool))
    assert torch.any(out[0][mask_bool, :] != cube[0][mask_bool, :])
    assert torch.equal(out[0][~mask_bool, :], cube[0][~mask_bool, :])


def test_multichannel_cube(tmp_path: Path) -> None:
    h, w, c = 30, 44, 51
    mask = _make_mask(h, w, 8, 10, 20, 28)
    json_path = _make_tracking_json(
        tmp_path,
        {9: [(3, mask, [10, 8, 18, 12])]},
        img_h=h,
        img_w=w,
    )
    node = PoissonCubeOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[3],
        occlusion_start_frame=9,
        occlusion_end_frame=9,
        occlusion_shape="bbox",
        max_iter=2500,
        tol=1e-7,
    )
    cube = _make_pattern_cube(h, w, c)
    out = node.forward(cube=cube, frame_id=_frame_id(9))["cube"]
    assert out.shape == cube.shape
    assert torch.isfinite(out).all()


def test_preserves_outside_mask(tmp_path: Path) -> None:
    h, w, c = 34, 46, 9
    mask = _make_mask(h, w, 12, 14, 24, 30)
    json_path = _make_tracking_json(
        tmp_path,
        {11: [(5, mask, [14, 12, 16, 12])]},
        img_h=h,
        img_w=w,
    )
    node = PoissonCubeOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[5],
        occlusion_start_frame=11,
        occlusion_end_frame=11,
        occlusion_shape="bbox",
        bbox_mode="dynamic",
        max_iter=2500,
        tol=1e-7,
    )
    cube = _make_pattern_cube(h, w, c)
    out = node.forward(cube=cube, frame_id=_frame_id(11))["cube"]

    bbox_mask = torch.zeros((h, w), dtype=torch.bool)
    bbox_mask[12:24, 14:30] = True
    assert torch.equal(out[0][~bbox_mask, :], cube[0][~bbox_mask, :])


def test_output_shape_matches_input(tmp_path: Path) -> None:
    h, w, c = 28, 36, 13
    mask = _make_mask(h, w, 4, 5, 18, 24)
    json_path = _make_tracking_json(
        tmp_path,
        {6: [(1, mask, [5, 4, 19, 14])]},
        img_h=h,
        img_w=w,
    )
    node = PoissonCubeOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=6,
        occlusion_end_frame=6,
    )
    cube = _make_pattern_cube(h, w, c)
    out = node.forward(cube=cube, frame_id=_frame_id(6))["cube"]
    assert out.shape == cube.shape
    assert out.dtype == cube.dtype
