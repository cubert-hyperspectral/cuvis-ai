"""Tests for MaskPrompt schedule decoding and frame emission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from cuvis_ai_core.data.rle import coco_rle_encode

from cuvis_ai.node.prompts import (
    BBoxPrompt,
    MaskPrompt,
    TextPrompt,
    load_detection_index,
    parse_spatial_prompt_spec,
    parse_text_prompt_spec,
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


def _write_track_centric_detection_json(
    tmp_path: Path,
    *,
    frame_indices: list[int],
    annotations: list[dict],
) -> Path:
    json_path = tmp_path / "track_centric_detections.json"
    payload = {
        "videos": [{"id": 1, "name": "demo", "frame_indices": frame_indices}],
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


def _track_ann(
    *,
    ann_id: int,
    track_id: int,
    frame_masks: list[np.ndarray | None] | None = None,
    frame_bboxes: list[list[float] | None] | None = None,
    frame_scores: list[float | None] | None = None,
) -> dict:
    frame_count = 0
    if frame_masks is not None:
        frame_count = len(frame_masks)
    if frame_bboxes is not None:
        frame_count = max(frame_count, len(frame_bboxes))
    if frame_scores is not None:
        frame_count = max(frame_count, len(frame_scores))

    segmentations = None
    if frame_masks is not None:
        segmentations = [
            None if mask is None else coco_rle_encode(mask.astype(np.uint8)) for mask in frame_masks
        ]
    bboxes = frame_bboxes if frame_bboxes is not None else [None] * frame_count
    detection_scores = frame_scores if frame_scores is not None else [None] * frame_count
    areas = [
        None if mask is None else float(mask.sum())
        for mask in (frame_masks if frame_masks is not None else [None] * frame_count)
    ]
    annotation = {
        "id": ann_id,
        "track_id": track_id,
        "category_id": 1,
        "bboxes": bboxes,
        "detection_scores": detection_scores,
        "areas": areas,
    }
    if segmentations is not None:
        annotation["segmentations"] = segmentations
    return annotation


def test_parse_spatial_prompt_spec() -> None:
    spec = parse_spatial_prompt_spec("2:7@65", order=4)
    assert spec.object_id == 2
    assert spec.detection_id == 7
    assert spec.frame_id == 65
    assert spec.order == 4

    spec = parse_spatial_prompt_spec("3:9@42", order=2)
    assert spec.object_id == 3
    assert spec.detection_id == 9
    assert spec.frame_id == 42
    assert spec.order == 2


def test_parse_text_prompt_spec() -> None:
    spec = parse_text_prompt_spec("person@65", order=3)

    assert spec.text == "person"
    assert spec.frame_id == 65
    assert spec.order == 3


def test_parse_text_prompt_spec_bare_prompt_defaults_to_frame_zero() -> None:
    spec = parse_text_prompt_spec("car", order=1)

    assert spec.text == "car"
    assert spec.frame_id == 0
    assert spec.order == 1


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


def test_load_detection_index_supports_flat_coco_json(tmp_path: Path) -> None:
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    json_path = _write_detection_json(
        tmp_path,
        images={7: (4, 5)},
        annotations=[_ann(ann_id=1, frame_id=7, track_id=2, score=0.9, mask=mask)],
    )

    annotations_by_frame, frame_hw_by_id, default_hw = load_detection_index(json_path)

    assert sorted(annotations_by_frame) == [7]
    assert annotations_by_frame[7][0]["track_id"] == 2
    assert frame_hw_by_id[7] == (4, 5)
    assert default_hw == (4, 5)


def test_load_detection_index_supports_track_centric_sam3_json(tmp_path: Path) -> None:
    mask_frame_0 = np.zeros((4, 5), dtype=np.uint8)
    mask_frame_0[0:2, 0:2] = 1
    mask_frame_1 = np.zeros((4, 5), dtype=np.uint8)
    mask_frame_1[1:4, 2:5] = 1
    json_path = _write_track_centric_detection_json(
        tmp_path,
        frame_indices=[0, 1],
        annotations=[
            _track_ann(
                ann_id=1,
                track_id=2,
                frame_masks=[mask_frame_0, mask_frame_1],
                frame_bboxes=[[0.0, 0.0, 2.0, 2.0], [2.0, 1.0, 3.0, 3.0]],
                frame_scores=[0.8, 0.7],
            )
        ],
    )

    annotations_by_frame, frame_hw_by_id, default_hw = load_detection_index(json_path)

    assert sorted(annotations_by_frame) == [0, 1]
    assert annotations_by_frame[0][0]["track_id"] == 2
    assert annotations_by_frame[1][0]["bbox"] == [2.0, 1.0, 3.0, 3.0]
    assert frame_hw_by_id[0] == (4, 5)
    assert frame_hw_by_id[1] == (4, 5)
    assert default_hw == (4, 5)


def test_mask_prompt_supports_track_centric_sam3_json(tmp_path: Path) -> None:
    mask_frame_0 = np.zeros((4, 5), dtype=np.uint8)
    mask_frame_0[0:2, 1:3] = 1
    json_path = _write_track_centric_detection_json(
        tmp_path,
        frame_indices=[0, 1],
        annotations=[
            _track_ann(
                ann_id=1,
                track_id=2,
                frame_masks=[mask_frame_0, None],
                frame_bboxes=[[1.0, 0.0, 2.0, 2.0], None],
                frame_scores=[0.9, None],
            )
        ],
    )

    node = MaskPrompt(json_path=str(json_path), prompt_specs=["9:2@0"])
    out = node.forward(frame_id=torch.tensor([0], dtype=torch.int64))

    assert torch.equal(out["mask"][0], torch.from_numpy(mask_frame_0.astype(np.int32) * 9))


def test_bbox_prompt_supports_track_centric_sam3_json(tmp_path: Path) -> None:
    mask_frame_0 = np.zeros((4, 5), dtype=np.uint8)
    mask_frame_0[0:2, 1:3] = 1
    json_path = _write_track_centric_detection_json(
        tmp_path,
        frame_indices=[0, 1],
        annotations=[
            _track_ann(
                ann_id=1,
                track_id=2,
                frame_masks=[mask_frame_0, None],
                frame_bboxes=[[1.0, 0.0, 2.0, 2.0], None],
                frame_scores=[0.9, None],
            )
        ],
    )

    node = BBoxPrompt(json_path=str(json_path), prompt_specs=["9:2@0"])
    out = node.forward(frame_id=torch.tensor([0], dtype=torch.int64))

    assert out["bboxes"] == [
        {
            "element_id": 0,
            "object_id": 9,
            "x_min": 1.0,
            "y_min": 0.0,
            "x_max": 3.0,
            "y_max": 2.0,
        }
    ]


def test_track_centric_bbox_prompt_rejects_missing_frame_sizes(tmp_path: Path) -> None:
    json_path = _write_track_centric_detection_json(
        tmp_path,
        frame_indices=[0],
        annotations=[
            {
                "id": 1,
                "track_id": 2,
                "category_id": 1,
                "bboxes": [[1.0, 0.0, 2.0, 2.0]],
                "detection_scores": [0.9],
                "areas": [4.0],
            }
        ],
    )

    with pytest.raises(ValueError, match="no usable height/width"):
        BBoxPrompt(json_path=str(json_path), prompt_specs=["9:2@0"])


def test_text_prompt_emits_prompt_on_scheduled_frame_and_empty_elsewhere() -> None:
    node = TextPrompt(prompt_specs=["person@5", "car@9"], prompt_mode="scheduled")

    scheduled = node.forward(frame_id=torch.tensor([5], dtype=torch.int64))
    unscheduled = node.forward(frame_id=torch.tensor([6], dtype=torch.int64))
    later = node.forward(frame_id=torch.tensor([9], dtype=torch.int64))

    assert scheduled["text_prompt"] == "person"
    assert unscheduled["text_prompt"] == ""
    assert later["text_prompt"] == "car"


def test_text_prompt_repeat_mode_keeps_latest_prompt_active_until_replaced() -> None:
    node = TextPrompt(prompt_specs=["person@5", "car@9"], prompt_mode="repeat")

    before = node.forward(frame_id=torch.tensor([4], dtype=torch.int64))
    first = node.forward(frame_id=torch.tensor([5], dtype=torch.int64))
    carried = node.forward(frame_id=torch.tensor([6], dtype=torch.int64))
    replaced = node.forward(frame_id=torch.tensor([9], dtype=torch.int64))
    later = node.forward(frame_id=torch.tensor([10], dtype=torch.int64))

    assert before["text_prompt"] == ""
    assert first["text_prompt"] == "person"
    assert carried["text_prompt"] == "person"
    assert replaced["text_prompt"] == "car"
    assert later["text_prompt"] == "car"


def test_text_prompt_repeat_mode_keeps_bare_prompt_active_after_frame_zero() -> None:
    node = TextPrompt(prompt_specs=["person"], prompt_mode="repeat")

    later = node.forward(frame_id=torch.tensor([25], dtype=torch.int64))

    assert later["text_prompt"] == "person"


def test_text_prompt_rejects_multiple_distinct_texts_on_same_frame() -> None:
    with pytest.raises(ValueError, match="same frame"):
        TextPrompt(prompt_specs=["person@5", "car@5"])


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
