"""Tests for MaskPrompt schedule decoding and frame emission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from cuvis_ai_core.data.rle import coco_rle_encode

from cuvis_ai.node.static_node import (
    BBoxPrompt,
    MaskPrompt,
    parse_bbox_prompt_spec,
    parse_mask_prompt_spec,
)


def _write_detection_json(
    tmp_path: Path,
    *,
    images: dict[int, tuple[int, int]],
    annotations: list[dict],
) -> Path:
    json_path = tmp_path / "detections.json"
    payload = {
        "images": [
            {"id": frame_id, "height": hw[0], "width": hw[1]}
            for frame_id, hw in sorted(images.items())
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path


def _ann(
    *,
    ann_id: int,
    frame_id: int,
    mask: np.ndarray | None = None,
    track_id: int | None = None,
    score: float | None = None,
) -> dict:
    annotation = {
        "id": ann_id,
        "image_id": frame_id,
        "category_id": 1,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "area": float(mask.sum()) if mask is not None else 1.0,
        "iscrowd": 0,
    }
    if track_id is not None:
        annotation["track_id"] = int(track_id)
    if score is not None:
        annotation["score"] = float(score)
    if mask is not None:
        annotation["segmentation"] = coco_rle_encode(mask.astype(np.uint8))
    return annotation


def test_parse_mask_prompt_spec() -> None:
    spec = parse_mask_prompt_spec("2:7@65", order=4)

    assert spec.object_id == 2
    assert spec.detection_id == 7
    assert spec.frame_id == 65
    assert spec.order == 4


def test_parse_bbox_prompt_spec() -> None:
    spec = parse_bbox_prompt_spec("3:9@42", order=2)

    assert spec.object_id == 3
    assert spec.detection_id == 9
    assert spec.frame_id == 42
    assert spec.order == 2


def test_mask_prompt_resolves_track_id(tmp_path: Path) -> None:
    mask_track_1 = np.zeros((4, 5), dtype=np.uint8)
    mask_track_1[0:2, 0:2] = 1
    mask_track_2 = np.zeros((4, 5), dtype=np.uint8)
    mask_track_2[2:4, 3:5] = 1
    json_path = _write_detection_json(
        tmp_path,
        images={70: (4, 5)},
        annotations=[
            _ann(ann_id=1, frame_id=70, track_id=1, score=0.2, mask=mask_track_1),
            _ann(ann_id=2, frame_id=70, track_id=2, score=0.9, mask=mask_track_2),
        ],
    )

    node = MaskPrompt(json_path=str(json_path), prompt_specs=["9:2@70"])
    out = node.forward(frame_id=torch.tensor([70], dtype=torch.int64))

    assert out["mask"].shape == (1, 4, 5)
    assert torch.equal(out["mask"][0], torch.from_numpy(mask_track_2.astype(np.int32) * 9))


def test_mask_prompt_resolves_score_rank_when_track_ids_absent(tmp_path: Path) -> None:
    lower_score = np.zeros((4, 5), dtype=np.uint8)
    lower_score[0:2, 0:2] = 1
    higher_score = np.zeros((4, 5), dtype=np.uint8)
    higher_score[1:4, 2:5] = 1
    json_path = _write_detection_json(
        tmp_path,
        images={5: (4, 5)},
        annotations=[
            _ann(ann_id=1, frame_id=5, score=0.2, mask=lower_score),
            _ann(ann_id=2, frame_id=5, score=0.9, mask=higher_score),
        ],
    )

    node = MaskPrompt(json_path=str(json_path), prompt_specs=["4:1@5"])
    out = node.forward(frame_id=torch.tensor([5], dtype=torch.int64))

    assert torch.equal(out["mask"][0], torch.from_numpy(higher_score.astype(np.int32) * 4))


def test_mask_prompt_rejects_missing_segmentation(tmp_path: Path) -> None:
    json_path = _write_detection_json(
        tmp_path,
        images={12: (4, 5)},
        annotations=[_ann(ann_id=1, frame_id=12, score=0.7, mask=None)],
    )

    with pytest.raises(ValueError, match="does not contain a 'segmentation' field"):
        MaskPrompt(json_path=str(json_path), prompt_specs=["3:1@12"])


def test_mask_prompt_rejects_empty_segmentation_mask(tmp_path: Path) -> None:
    json_path = _write_detection_json(
        tmp_path,
        images={12: (4, 5)},
        annotations=[
            {
                "id": 1,
                "image_id": 12,
                "category_id": 1,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "area": 0.0,
                "iscrowd": 0,
                "track_id": 3,
                "segmentation": [],
            }
        ],
    )

    with pytest.raises(ValueError, match="has an empty segmentation mask"):
        MaskPrompt(json_path=str(json_path), prompt_specs=["3:3@12"])


def test_mask_prompt_unions_same_object_and_later_spec_wins_on_overlap(tmp_path: Path) -> None:
    mask_a = np.zeros((4, 5), dtype=np.uint8)
    mask_a[0:2, 0:2] = 1
    mask_b = np.zeros((4, 5), dtype=np.uint8)
    mask_b[2:4, 3:5] = 1
    mask_c = np.zeros((4, 5), dtype=np.uint8)
    mask_c[0, 1:4] = 1

    json_path = _write_detection_json(
        tmp_path,
        images={9: (4, 5)},
        annotations=[
            _ann(ann_id=1, frame_id=9, track_id=1, mask=mask_a),
            _ann(ann_id=2, frame_id=9, track_id=2, mask=mask_b),
            _ann(ann_id=3, frame_id=9, track_id=3, mask=mask_c),
        ],
    )

    node = MaskPrompt(
        json_path=str(json_path),
        prompt_specs=["5:1@9", "5:2@9", "9:3@9"],
    )
    out = node.forward(frame_id=torch.tensor([9], dtype=torch.int64))

    expected = np.zeros((4, 5), dtype=np.int32)
    expected[mask_a.astype(bool)] = 5
    expected[mask_b.astype(bool)] = 5
    expected[mask_c.astype(bool)] = 9

    assert torch.equal(out["mask"][0], torch.from_numpy(expected))


def test_mask_prompt_emits_zero_mask_on_unscheduled_frame(tmp_path: Path) -> None:
    scheduled = np.zeros((3, 4), dtype=np.uint8)
    scheduled[1:, 1:3] = 1
    json_path = _write_detection_json(
        tmp_path,
        images={0: (3, 4), 1: (3, 4)},
        annotations=[_ann(ann_id=1, frame_id=0, score=0.9, mask=scheduled)],
    )

    node = MaskPrompt(json_path=str(json_path), prompt_specs=["7:1@0"])
    out = node.forward(frame_id=torch.tensor([1], dtype=torch.int64))

    assert out["mask"].shape == (1, 3, 4)
    assert torch.count_nonzero(out["mask"]).item() == 0


def test_mask_prompt_prefers_segmentation_size_over_placeholder_image_size(tmp_path: Path) -> None:
    scheduled = np.zeros((3, 4), dtype=np.uint8)
    scheduled[1:, 1:3] = 1
    json_path = _write_detection_json(
        tmp_path,
        images={0: (1, 1), 1: (1, 1)},
        annotations=[_ann(ann_id=1, frame_id=0, track_id=7, mask=scheduled)],
    )

    node = MaskPrompt(json_path=str(json_path), prompt_specs=["7:7@0"])
    out = node.forward(frame_id=torch.tensor([1], dtype=torch.int64))

    assert out["mask"].shape == (1, 3, 4)
    assert torch.count_nonzero(out["mask"]).item() == 0


def test_bbox_prompt_resolves_track_id(tmp_path: Path) -> None:
    json_path = _write_detection_json(
        tmp_path,
        images={70: (4, 5)},
        annotations=[
            _ann(
                ann_id=1, frame_id=70, track_id=1, score=0.2, mask=np.ones((4, 5), dtype=np.uint8)
            ),
            {
                "id": 2,
                "image_id": 70,
                "category_id": 1,
                "bbox": [1.0, 1.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 0,
                "track_id": 2,
                "score": 0.9,
            },
        ],
    )

    node = BBoxPrompt(json_path=str(json_path), prompt_specs=["9:2@70"])
    out = node.forward(frame_id=torch.tensor([70], dtype=torch.int64))

    assert out["bboxes"] == [
        {
            "element_id": 0,
            "object_id": 9,
            "x_min": 1.0,
            "y_min": 1.0,
            "x_max": 3.0,
            "y_max": 3.0,
        }
    ]
    assert torch.equal(
        out["prompt_boxes_xyxy"],
        torch.tensor([[[1.0, 1.0, 3.0, 3.0]]], dtype=torch.float32),
    )
    assert torch.equal(out["prompt_object_ids"], torch.tensor([[9]], dtype=torch.int64))


def test_bbox_prompt_resolves_score_rank_when_track_ids_absent(tmp_path: Path) -> None:
    json_path = _write_detection_json(
        tmp_path,
        images={5: (4, 5)},
        annotations=[
            {
                "id": 1,
                "image_id": 5,
                "category_id": 1,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "area": 1.0,
                "iscrowd": 0,
                "score": 0.2,
            },
            {
                "id": 2,
                "image_id": 5,
                "category_id": 1,
                "bbox": [2.0, 1.0, 2.0, 3.0],
                "area": 6.0,
                "iscrowd": 0,
                "score": 0.9,
            },
        ],
    )

    node = BBoxPrompt(json_path=str(json_path), prompt_specs=["4:1@5"])
    out = node.forward(frame_id=torch.tensor([5], dtype=torch.int64))

    assert out["bboxes"][0]["object_id"] == 4
    assert out["bboxes"][0]["x_min"] == pytest.approx(2.0)
    assert out["bboxes"][0]["y_max"] == pytest.approx(4.0)


def test_bbox_prompt_rejects_missing_bbox(tmp_path: Path) -> None:
    json_path = _write_detection_json(
        tmp_path,
        images={12: (4, 5)},
        annotations=[
            {
                "id": 1,
                "image_id": 12,
                "category_id": 1,
                "area": 1.0,
                "iscrowd": 0,
                "track_id": 3,
            }
        ],
    )

    with pytest.raises(ValueError, match="does not contain a 'bbox' field"):
        BBoxPrompt(json_path=str(json_path), prompt_specs=["3:3@12"])


def test_bbox_prompt_later_spec_wins_for_same_object_on_same_frame(tmp_path: Path) -> None:
    json_path = _write_detection_json(
        tmp_path,
        images={9: (4, 5)},
        annotations=[
            {
                "id": 1,
                "image_id": 9,
                "category_id": 1,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "area": 1.0,
                "iscrowd": 0,
                "track_id": 1,
            },
            {
                "id": 2,
                "image_id": 9,
                "category_id": 1,
                "bbox": [1.0, 1.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 0,
                "track_id": 2,
            },
            {
                "id": 3,
                "image_id": 9,
                "category_id": 1,
                "bbox": [2.0, 0.0, 2.0, 3.0],
                "area": 6.0,
                "iscrowd": 0,
                "track_id": 3,
            },
        ],
    )

    node = BBoxPrompt(
        json_path=str(json_path),
        prompt_specs=["5:1@9", "5:2@9", "9:3@9"],
    )
    out = node.forward(frame_id=torch.tensor([9], dtype=torch.int64))

    assert out["bboxes"] == [
        {
            "element_id": 0,
            "object_id": 5,
            "x_min": 1.0,
            "y_min": 1.0,
            "x_max": 3.0,
            "y_max": 3.0,
        },
        {
            "element_id": 0,
            "object_id": 9,
            "x_min": 2.0,
            "y_min": 0.0,
            "x_max": 4.0,
            "y_max": 3.0,
        },
    ]
    assert torch.equal(
        out["prompt_object_ids"],
        torch.tensor([[5, 9]], dtype=torch.int64),
    )


def test_bbox_prompt_emits_empty_outputs_on_unscheduled_frame(tmp_path: Path) -> None:
    json_path = _write_detection_json(
        tmp_path,
        images={0: (3, 4), 1: (3, 4)},
        annotations=[
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": [1.0, 1.0, 2.0, 1.0],
                "area": 2.0,
                "iscrowd": 0,
                "track_id": 7,
            }
        ],
    )

    node = BBoxPrompt(json_path=str(json_path), prompt_specs=["7:7@0"])
    out = node.forward(frame_id=torch.tensor([1], dtype=torch.int64))

    assert out["bboxes"] == []
    assert out["prompt_boxes_xyxy"].shape == (1, 0, 4)
    assert out["prompt_object_ids"].shape == (1, 0)
