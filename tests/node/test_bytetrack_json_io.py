"""Tests for DetectionJsonReader and CocoTrackBBoxWriter."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cuvis_ai.node.json_file import CocoTrackBBoxWriter, DetectionJsonReader


def _write_sample_json(path: Path) -> None:
    payload = {
        "info": {"description": "sample"},
        "images": [
            {"id": 0, "file_name": "f0", "height": 480, "width": 640},
            {"id": 1, "file_name": "f1", "height": 480, "width": 640},
        ],
        "annotations": [
            {"id": 1, "image_id": 0, "bbox": [10, 20, 30, 40], "score": 0.9, "category_id": 5},
            {"id": 2, "image_id": 1, "bbox": [15, 25, 35, 45], "score": 0.8, "category_id": 6},
        ],
        "categories": [{"id": 5, "name": "obj"}, {"id": 6, "name": "obj2"}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


def test_detection_json_reader_iterates_frames(tmp_path: Path) -> None:
    src = tmp_path / "detections.json"
    _write_sample_json(src)

    reader = DetectionJsonReader(json_path=str(src))

    out0 = reader.forward()
    assert int(out0["frame_id"][0].item()) == 0
    assert out0["bboxes"].shape == (1, 1, 4)
    assert out0["category_ids"][0, 0].item() == 5

    out1 = reader.forward()
    assert int(out1["frame_id"][0].item()) == 1
    assert out1["bboxes"].shape == (1, 1, 4)


def test_coco_track_bbox_writer_writes_track_ids(tmp_path: Path) -> None:
    out_path = tmp_path / "tracking.json"
    node = CocoTrackBBoxWriter(
        output_json_path=str(out_path), category_id_to_name={5: "obj"}, flush_interval=1
    )

    node.forward(
        frame_id=torch.tensor([0], dtype=torch.int64),
        bboxes=torch.tensor([[[0.0, 0.0, 10.0, 10.0]]], dtype=torch.float32),
        category_ids=torch.tensor([[5]], dtype=torch.int64),
        confidences=torch.tensor([[0.9]], dtype=torch.float32),
        track_ids=torch.tensor([[42]], dtype=torch.int64),
        orig_hw=torch.tensor([[480, 640]], dtype=torch.int64),
    )

    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["annotations"]) == 1
    ann = data["annotations"][0]
    assert ann["track_id"] == 42
    assert ann["category_id"] == 5
