"""Tests for SAM3 example source-context helpers."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from cuvis_ai_core.data.rle import coco_rle_encode

from examples.object_tracking.sam3.sam3_source_context import (
    load_detection_annotation,
    load_detection_point_prompt,
    resolve_detection_frame_hw,
)


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


def test_load_detection_annotation_supports_track_centric_json(tmp_path: Path) -> None:
    mask_frame_0 = np.zeros((4, 5), dtype=np.uint8)
    mask_frame_0[0:2, 1:3] = 1
    json_path = _write_track_centric_detection_json(
        tmp_path,
        frame_indices=[0, 1],
        annotations=[
            {
                "id": 1,
                "track_id": 2,
                "category_id": 1,
                "segmentations": [coco_rle_encode(mask_frame_0), None],
                "bboxes": [[1.0, 0.0, 2.0, 2.0], None],
                "detection_scores": [0.9, None],
                "areas": [float(mask_frame_0.sum()), None],
            }
        ],
    )

    annotation, obj_id = load_detection_annotation(json_path, det_id=2, frame_idx=0)

    assert obj_id == 2
    assert annotation["track_id"] == 2
    assert annotation["image_id"] == 0
    assert annotation["bbox"] == [1.0, 0.0, 2.0, 2.0]


def test_load_detection_annotation_rejects_missing_track_on_frame(tmp_path: Path) -> None:
    json_path = _write_track_centric_detection_json(
        tmp_path,
        frame_indices=[0],
        annotations=[
            {
                "id": 1,
                "track_id": 2,
                "category_id": 1,
                "bboxes": [None],
                "detection_scores": [None],
                "areas": [None],
            }
        ],
    )

    try:
        load_detection_annotation(json_path, det_id=2, frame_idx=0)
    except click.ClickException as exc:
        assert "has no annotations" in str(exc)
    else:
        raise AssertionError("Expected ClickException for missing per-frame annotation.")


def test_resolve_detection_frame_hw_supports_track_centric_json(tmp_path: Path) -> None:
    mask_frame_0 = np.zeros((4, 5), dtype=np.uint8)
    mask_frame_0[0:2, 1:3] = 1
    json_path = _write_track_centric_detection_json(
        tmp_path,
        frame_indices=[0, 1],
        annotations=[
            {
                "id": 1,
                "track_id": 2,
                "category_id": 1,
                "segmentations": [coco_rle_encode(mask_frame_0), None],
                "bboxes": [[1.0, 0.0, 2.0, 2.0], None],
                "detection_scores": [0.9, None],
                "areas": [float(mask_frame_0.sum()), None],
            }
        ],
    )

    assert resolve_detection_frame_hw(json_path, frame_idx=0) == (4, 5)


def test_load_detection_point_prompt_supports_track_centric_json(tmp_path: Path) -> None:
    mask_frame_0 = np.zeros((4, 5), dtype=np.uint8)
    mask_frame_0[0:2, 1:3] = 1
    json_path = _write_track_centric_detection_json(
        tmp_path,
        frame_indices=[0, 1],
        annotations=[
            {
                "id": 1,
                "track_id": 2,
                "category_id": 1,
                "segmentations": [coco_rle_encode(mask_frame_0), None],
                "bboxes": [[1.0, 0.0, 2.0, 2.0], None],
                "detection_scores": [0.9, None],
                "areas": [float(mask_frame_0.sum()), None],
            }
        ],
    )

    points, point_labels, obj_id = load_detection_point_prompt(json_path, det_id=2, frame_idx=0)

    assert obj_id == 2
    assert point_labels == [1]
    assert points == [[0.4, 0.25]]
