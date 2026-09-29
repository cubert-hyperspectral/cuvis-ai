"""Characterization of the bbox prompt schedule built from a detection JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cuvis_ai.node.prompts import BBoxPrompt

pytestmark = pytest.mark.unit


def _write_json(tmp_path: Path, annotations: list[dict]) -> Path:
    payload = {
        "images": [{"id": 70, "height": 4, "width": 5}, {"id": 71, "height": 4, "width": 5}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ann(ann_id: int, frame_id: int, bbox: list[float], **extra) -> dict:
    return {"id": ann_id, "image_id": frame_id, "category_id": 1, "bbox": bbox, **extra}


def test_bbox_prompt_resolves_the_track_and_clamps_the_box_to_the_frame(tmp_path: Path):
    json_path = _write_json(
        tmp_path,
        [
            _ann(1, 70, [0.0, 0.0, 1.0, 1.0], track_id=1),
            _ann(2, 70, [3.0, 2.0, 5.0, 5.0], track_id=2),  # runs past the 4x5 frame
        ],
    )
    node = BBoxPrompt(json_path=str(json_path), prompt_specs=["9:2@70"])
    out = node.forward(frame_id=torch.tensor([70], dtype=torch.int64))
    assert out["bboxes"] == [
        {"element_id": 0, "object_id": 9, "x_min": 3.0, "y_min": 2.0, "x_max": 5.0, "y_max": 4.0}
    ]
    assert torch.equal(out["prompt_boxes_xyxy"], torch.tensor([[[3.0, 2.0, 5.0, 4.0]]]))
    assert torch.equal(out["prompt_object_ids"], torch.tensor([[9]], dtype=torch.int64))


def test_bbox_prompt_resolves_the_score_rank_without_track_ids(tmp_path: Path):
    json_path = _write_json(
        tmp_path,
        [
            _ann(1, 70, [0.0, 0.0, 1.0, 1.0], score=0.2),
            _ann(2, 70, [1.0, 1.0, 2.0, 2.0], score=0.9),
        ],
    )
    node = BBoxPrompt(json_path=str(json_path), prompt_specs=["3:1@70"])
    out = node.forward(frame_id=torch.tensor([70], dtype=torch.int64))
    assert out["bboxes"][0]["x_min"] == 1.0 and out["bboxes"][0]["object_id"] == 3


def test_bbox_prompt_emits_nothing_on_an_unscheduled_frame(tmp_path: Path):
    json_path = _write_json(tmp_path, [_ann(1, 70, [0.0, 0.0, 1.0, 1.0], track_id=1)])
    node = BBoxPrompt(json_path=str(json_path), prompt_specs=["9:1@70"])
    out = node.forward(frame_id=torch.tensor([71], dtype=torch.int64))
    assert out["bboxes"] == []
    assert out["prompt_boxes_xyxy"].shape == (1, 0, 4)
    assert out["prompt_object_ids"].shape == (1, 0)


@pytest.mark.parametrize(
    ("bbox", "message"),
    [([0.0, 0.0, 0.0, 1.0], "degenerate bbox"), ([1.0, 1.0, 2.0], "invalid COCO bbox")],
)
def test_bbox_prompt_rejects_unusable_boxes(tmp_path: Path, bbox, message):
    json_path = _write_json(tmp_path, [_ann(1, 70, bbox, track_id=1)])
    with pytest.raises(ValueError, match=message):
        BBoxPrompt(json_path=str(json_path), prompt_specs=["9:1@70"])


def test_bbox_prompt_rejects_an_image_without_a_positive_size_at_load(tmp_path: Path):
    payload = {"images": [{"id": 70, "height": 0, "width": 5}], "annotations": [], "categories": []}
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive height/width"):
        BBoxPrompt(json_path=str(path), prompt_specs=[])
