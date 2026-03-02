"""Tests for DetectionCocoJsonNode."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch

from cuvis_ai.node.json_writer import DetectionCocoJsonNode


def _build_inputs(
    frame_idx: int,
    boxes_xyxy: list[list[float]],
    category_ids: list[int],
    scores: list[float],
    h: int = 480,
    w: int = 640,
) -> dict:
    n = len(boxes_xyxy)
    return {
        "frame_id": torch.tensor([frame_idx], dtype=torch.int64),
        "bboxes": torch.tensor([boxes_xyxy], dtype=torch.float32)
        if n > 0
        else torch.empty((1, 0, 4), dtype=torch.float32),
        "category_ids": torch.tensor([category_ids], dtype=torch.int64),
        "confidences": torch.tensor([scores], dtype=torch.float32),
        "orig_hw": torch.tensor([[h, w]], dtype=torch.int64),
    }


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_import() -> None:
    from cuvis_ai.node.json_writer import DetectionCocoJsonNode  # noqa: F401


def test_writes_valid_coco_json(tmp_path: Path) -> None:
    """Writes valid COCO detection JSON with no segmentation or track_id fields."""
    json_path = tmp_path / "detections.json"
    node = DetectionCocoJsonNode(
        output_json_path=str(json_path),
        category_id_to_name={0: "person"},
        flush_interval=1,
    )

    node.forward(
        **_build_inputs(
            frame_idx=0,
            boxes_xyxy=[[10.0, 20.0, 100.0, 200.0]],
            category_ids=[0],
            scores=[0.95],
        )
    )
    data = _read_json(json_path)

    # Top-level keys
    assert "images" in data
    assert "annotations" in data
    assert "categories" in data

    # Image entry
    assert len(data["images"]) == 1
    img = data["images"][0]
    assert img["id"] == 0
    assert img["width"] == 640
    assert img["height"] == 480

    # Annotation entry
    assert len(data["annotations"]) == 1
    ann = data["annotations"][0]
    assert ann["image_id"] == 0
    assert ann["category_id"] == 0
    # bbox must be [x, y, w, h] (COCO convention)
    assert ann["bbox"] == [10.0, 20.0, 90.0, 180.0]
    assert abs(ann["area"] - 90.0 * 180.0) < 1e-3
    assert ann["iscrowd"] == 0
    assert abs(ann["score"] - 0.95) < 1e-4
    # No segmentation or track_id
    assert "segmentation" not in ann
    assert "track_id" not in ann

    # Categories
    assert data["categories"] == [{"id": 0, "name": "person"}]

    node.close()


def test_second_frame_accumulates(tmp_path: Path) -> None:
    """Multiple frames accumulate correctly in the JSON."""
    json_path = tmp_path / "detections.json"
    node = DetectionCocoJsonNode(
        output_json_path=str(json_path),
        flush_interval=1,
    )

    node.forward(**_build_inputs(0, [[0.0, 0.0, 10.0, 10.0]], [0], [0.9]))
    node.forward(**_build_inputs(1, [[5.0, 5.0, 15.0, 15.0]], [0], [0.8]))
    data = _read_json(json_path)

    assert len(data["images"]) == 2
    assert len(data["annotations"]) == 2
    node.close()


def test_write_empty_frames_flag(tmp_path: Path) -> None:
    """When write_empty_frames=False, frames with 0 detections are excluded."""
    json_path = tmp_path / "detections.json"
    node = DetectionCocoJsonNode(
        output_json_path=str(json_path),
        write_empty_frames=False,
        flush_interval=1,
    )

    # Frame 0 with one detection
    node.forward(**_build_inputs(0, [[0.0, 0.0, 10.0, 10.0]], [0], [0.9]))
    # Frame 1 with zero detections
    node.forward(**_build_inputs(1, [], [], []))
    data = _read_json(json_path)

    image_ids = [img["id"] for img in data["images"]]
    assert 0 in image_ids
    assert 1 not in image_ids
    node.close()


def test_atomic_write(tmp_path: Path) -> None:
    """When atomic_write=True, writes via os.replace (temp file rename)."""
    json_path = tmp_path / "detections.json"
    node = DetectionCocoJsonNode(
        output_json_path=str(json_path),
        atomic_write=True,
        flush_interval=1,
    )

    with patch("cuvis_ai.node.json_writer.os.replace") as mock_replace:
        node.forward(**_build_inputs(0, [[0.0, 0.0, 10.0, 10.0]], [0], [0.9]))
        mock_replace.assert_called_once()

    node.close()
