"""Tests for TrackingResultsReader frame lookup mode."""

from __future__ import annotations

import torch

from cuvis_ai.node.json_file import TrackingResultsReader


def test_cursor_mode(coco_bbox_json_factory) -> None:
    """Without frame_id input, reader advances cursor sequentially."""
    json_path = coco_bbox_json_factory(
        {
            0: [{"bbox": [10, 10, 20, 20], "track_id": 1}],
            1: [{"bbox": [30, 30, 40, 40], "track_id": 2}],
        },
        filename="cursor_mode.json",
    )
    reader = TrackingResultsReader(json_path=str(json_path))

    out0 = reader.forward()
    assert out0["frame_id"].item() == 0
    assert out0["bboxes"].shape[1] == 1

    out1 = reader.forward()
    assert out1["frame_id"].item() == 1


def test_lookup_mode_frame_sync(coco_bbox_json_factory) -> None:
    """With frame_id input, reader looks up the specific frame."""
    json_path = coco_bbox_json_factory(
        {
            0: [{"bbox": [10, 10, 20, 20], "track_id": 1}],
            70: [
                {"bbox": [50, 50, 60, 60], "track_id": 10},
                {"bbox": [70, 70, 80, 80], "track_id": 20},
            ],
            100: [{"bbox": [90, 90, 95, 95], "track_id": 30}],
        },
        filename="lookup_mode.json",
    )
    reader = TrackingResultsReader(json_path=str(json_path))

    # Request frame 70 directly — should get 2 detections
    out = reader.forward(frame_id=torch.tensor([70]))
    assert out["frame_id"].item() == 70
    assert out["bboxes"].shape[1] == 2
    assert out["track_ids"][0, 0].item() == 10
    assert out["track_ids"][0, 1].item() == 20


def test_lookup_mode_missing_frame(coco_bbox_json_factory) -> None:
    """Requesting a frame not in JSON returns empty detections."""
    json_path = coco_bbox_json_factory(
        {0: [{"bbox": [10, 10, 20, 20], "track_id": 1}]},
        filename="missing_frame.json",
    )
    reader = TrackingResultsReader(json_path=str(json_path))

    out = reader.forward(frame_id=torch.tensor([999]))
    assert out["frame_id"].item() == 999
    assert out["bboxes"].shape[1] == 0


def test_lookup_mode_does_not_advance_cursor(coco_bbox_json_factory) -> None:
    """Lookup calls should not advance the internal cursor."""
    json_path = coco_bbox_json_factory(
        {
            0: [{"bbox": [10, 10, 20, 20], "track_id": 1}],
            1: [{"bbox": [30, 30, 40, 40], "track_id": 2}],
        },
        filename="cursor_no_advance.json",
    )
    reader = TrackingResultsReader(json_path=str(json_path))

    # Lookup frame 1
    reader.forward(frame_id=torch.tensor([1]))

    # Cursor should still be at 0
    out = reader.forward()
    assert out["frame_id"].item() == 0
