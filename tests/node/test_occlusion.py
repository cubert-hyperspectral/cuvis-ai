"""Tests for pure-PyTorch occlusion nodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from cuvis_ai_core.data.rle import coco_rle_encode

from cuvis_ai.node.occlusion import PoissonOcclusionNode, SolidOcclusionNode


def _make_mask(h: int, w: int, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _make_tracking_json(
    tmp_path: Path,
    masks_by_frame: dict[int, list[tuple[int, np.ndarray, list]]],
    *,
    img_h: int = 100,
    img_w: int = 200,
    filename: str = "tracking.json",
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
    path = tmp_path / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_rgb(h: int = 100, w: int = 200, value: float = 0.5) -> torch.Tensor:
    return torch.full((1, h, w, 3), value, dtype=torch.float32)


def _make_pattern_rgb(h: int = 100, w: int = 200) -> torch.Tensor:
    yy, xx = np.indices((h, w))
    arr = np.stack(
        [
            ((xx * 7 + yy * 3) % 256).astype(np.uint8),
            ((xx * 5 + yy * 11) % 256).astype(np.uint8),
            ((xx * 13 + yy * 17) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    return torch.from_numpy(arr).unsqueeze(0).to(dtype=torch.float32) / 255.0


def _make_pattern_cube(h: int = 100, w: int = 200, c: int = 51) -> torch.Tensor:
    yy, xx = np.indices((h, w))
    channels = []
    for idx in range(c):
        arr = ((xx * (idx + 3) + yy * (2 * idx + 5) + 17 * idx) % 251).astype(np.float32)
        channels.append(arr / 251.0)
    return torch.from_numpy(np.stack(channels, axis=-1)).unsqueeze(0)


def _frame_id(idx: int) -> torch.Tensor:
    return torch.tensor([idx], dtype=torch.int64)


def test_import() -> None:
    from cuvis_ai.node import (  # noqa: F401
        OcclusionNodeBase,
        PoissonCubeOcclusionNode,
        PoissonOcclusionNode,
        SolidOcclusionNode,
    )


def test_passthrough_outside_range_rgb(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 10, 20, 50, 80)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [20, 10, 60, 40])]})
    node = PoissonOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=10,
    )
    rgb = _make_rgb()
    out = node.forward(rgb_image=rgb, frame_id=_frame_id(3))
    assert torch.equal(out["rgb_image"], rgb)


def test_bbox_solid_fill_applies(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 10, 20, 50, 80)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [20, 10, 60, 40])]})
    node = PoissonOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=5,
        fill_color=(1.0, 0.0, 0.0),
        occlusion_shape="bbox",
        bbox_mode="dynamic",
    )
    rgb = _make_rgb(value=0.2)
    out = node.forward(rgb_image=rgb, frame_id=_frame_id(5))["rgb_image"]
    assert out[0, 10:50, 20:80, 0].min().item() == pytest.approx(1.0)
    assert out[0, 10:50, 20:80, 1].max().item() == pytest.approx(0.0)
    assert out[0, 10:50, 20:80, 2].max().item() == pytest.approx(0.0)
    assert out[0, 0, 0, 0].item() == pytest.approx(0.2)


def test_mask_solid_fill_applies(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 10, 20, 50, 80)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [20, 10, 60, 40])]})
    node = SolidOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=5,
        fill_color=(0.0, 1.0, 0.0),
        occlusion_shape="mask",
    )
    rgb = _make_rgb(value=0.2)
    out = node.forward(rgb_image=rgb, frame_id=_frame_id(5))["rgb_image"]
    mask_bool = torch.from_numpy(mask.astype(bool))
    assert out[0][mask_bool, 1].min().item() == pytest.approx(1.0)
    assert torch.equal(out[0][~mask_bool, :], rgb[0][~mask_bool, :])


def test_poisson_fill_bbox_modifies_only_region(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 10, 20, 50, 80)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [20, 10, 60, 40])]})
    node = PoissonOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=10,
        fill_color="poisson",
        max_iter=2000,
        tol=1e-7,
        occlusion_shape="bbox",
        bbox_mode="dynamic",
    )
    rgb = _make_pattern_rgb()
    result = node.forward(rgb_image=rgb, frame_id=_frame_id(5))["rgb_image"]

    bbox_mask = torch.zeros((100, 200), dtype=torch.bool)
    bbox_mask[10:50, 20:80] = True
    assert torch.any(result[0][bbox_mask, :] != rgb[0][bbox_mask, :])
    assert torch.equal(result[0][~bbox_mask, :], rgb[0][~bbox_mask, :])


def test_poisson_fill_mask_modifies_only_region(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 10, 20, 50, 80)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [20, 10, 60, 40])]})
    node = PoissonOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=10,
        fill_color="poisson",
        max_iter=2000,
        tol=1e-7,
        occlusion_shape="mask",
    )
    rgb = _make_pattern_rgb()
    result = node.forward(rgb_image=rgb, frame_id=_frame_id(5))["rgb_image"]

    mask_bool = torch.from_numpy(mask.astype(bool))
    assert torch.any(result[0][mask_bool, :] != rgb[0][mask_bool, :])
    assert torch.equal(result[0][~mask_bool, :], rgb[0][~mask_bool, :])


def test_cube_input_supported_poisson(tmp_path: Path) -> None:
    h, w, c = 40, 60, 51
    mask = _make_mask(h, w, 10, 20, 25, 40)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [20, 10, 20, 15])]}, img_h=h, img_w=w)
    node = PoissonOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=5,
        fill_color="poisson",
        max_iter=2000,
        tol=1e-7,
        occlusion_shape="bbox",
        bbox_mode="dynamic",
    )
    cube = _make_pattern_cube(h=h, w=w, c=c)
    out = node.forward(cube=cube, frame_id=_frame_id(5))["cube"]
    assert out.shape == cube.shape
    assert torch.isfinite(out).all()


def test_static_bbox_applies_without_annotation_on_frame(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 10, 20, 50, 80)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [20, 10, 60, 40])]})
    node = PoissonOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=10,
        fill_color=(0.0, 0.0, 0.0),
        occlusion_shape="bbox",
        bbox_mode="static",
        static_bbox_scale=1.0,
    )
    rgb = _make_rgb(value=0.5)
    out = node.forward(rgb_image=rgb, frame_id=_frame_id(7))["rgb_image"]
    assert out[0, 20, 30, 0].item() == pytest.approx(0.0)


def test_static_full_width_x_stretches_horizontal_span(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 30, 50, 40, 70)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [50, 30, 20, 10])]})
    node = PoissonOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=5,
        fill_color=(0.0, 0.0, 0.0),
        occlusion_shape="bbox",
        bbox_mode="static",
        static_bbox_scale=1.0,
        static_full_width_x=True,
    )
    rgb = _make_rgb(value=0.5)
    out = node.forward(rgb_image=rgb, frame_id=_frame_id(5))["rgb_image"]
    assert out[0, 35, :, :].abs().max().item() == 0.0
    assert out[0, 20, 0, 0].item() == pytest.approx(0.5)


def test_input_validation_requires_exactly_one_input(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 10, 20, 50, 80)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [20, 10, 60, 40])]})
    node = PoissonOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=5,
    )
    rgb = _make_rgb()
    cube = _make_pattern_cube(c=8)
    with pytest.raises(ValueError, match="exactly one input"):
        node.forward(frame_id=_frame_id(5))
    with pytest.raises(ValueError, match="either rgb_image or cube"):
        node.forward(rgb_image=rgb, cube=cube, frame_id=_frame_id(5))


def test_frame_size_mismatch_resizes_mask(tmp_path: Path) -> None:
    mask_small = _make_mask(50, 100, 5, 10, 25, 40)
    json_path = _make_tracking_json(
        tmp_path,
        {5: [(1, mask_small, [20, 10, 60, 40])]},
        img_h=50,
        img_w=100,
    )
    node = PoissonOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=5,
        occlusion_end_frame=5,
        fill_color=(0.0, 0.0, 0.0),
        occlusion_shape="mask",
    )
    rgb = _make_rgb(h=100, w=200, value=0.5)
    out = node.forward(rgb_image=rgb, frame_id=_frame_id(5))["rgb_image"]
    assert out[0, 20, 40, 0].item() == pytest.approx(0.0)
    assert out[0, 0, 0, 0].item() == pytest.approx(0.5)


def test_invalid_fill_color_string_raises(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 0, 0, 10, 10)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [0, 0, 10, 10])]})
    with pytest.raises(ValueError, match="fill_color"):
        PoissonOcclusionNode(
            tracking_json_path=str(json_path),
            track_ids=[1],
            occlusion_start_frame=5,
            occlusion_end_frame=10,
            fill_color="neighbor",
        )


def test_invalid_fill_color_tuple_raises(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 0, 0, 10, 10)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [0, 0, 10, 10])]})
    with pytest.raises(ValueError, match="exactly 3 values"):
        PoissonOcclusionNode(
            tracking_json_path=str(json_path),
            track_ids=[1],
            occlusion_start_frame=5,
            occlusion_end_frame=10,
            fill_color=(0.1, 0.2),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PoissonOcclusionNode(
            tracking_json_path=str(json_path),
            track_ids=[1],
            occlusion_start_frame=5,
            occlusion_end_frame=10,
            fill_color=(1.2, 0.0, 0.0),
        )


def test_invalid_poisson_params_raises(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 0, 0, 10, 10)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [0, 0, 10, 10])]})
    with pytest.raises(ValueError, match="max_iter"):
        PoissonOcclusionNode(
            tracking_json_path=str(json_path),
            track_ids=[1],
            occlusion_start_frame=5,
            occlusion_end_frame=10,
            fill_color="poisson",
            max_iter=0,
        )
    with pytest.raises(ValueError, match="tol"):
        PoissonOcclusionNode(
            tracking_json_path=str(json_path),
            track_ids=[1],
            occlusion_start_frame=5,
            occlusion_end_frame=10,
            fill_color="poisson",
            tol=0.0,
        )


def test_invalid_occlusion_config_raises(tmp_path: Path) -> None:
    mask = _make_mask(100, 200, 0, 0, 10, 10)
    json_path = _make_tracking_json(tmp_path, {5: [(1, mask, [0, 0, 10, 10])]})
    with pytest.raises(ValueError, match="occlusion_shape"):
        PoissonOcclusionNode(
            tracking_json_path=str(json_path),
            track_ids=[1],
            occlusion_start_frame=5,
            occlusion_end_frame=10,
            occlusion_shape="invalid",
        )
    with pytest.raises(ValueError, match="bbox_mode"):
        PoissonOcclusionNode(
            tracking_json_path=str(json_path),
            track_ids=[1],
            occlusion_start_frame=5,
            occlusion_end_frame=10,
            occlusion_shape="bbox",
            bbox_mode="invalid",
        )
    with pytest.raises(ValueError, match="static_bbox_scale"):
        PoissonOcclusionNode(
            tracking_json_path=str(json_path),
            track_ids=[1],
            occlusion_start_frame=5,
            occlusion_end_frame=10,
            static_bbox_scale=0.0,
        )
