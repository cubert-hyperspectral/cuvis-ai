"""Roundtrip tests: TrackingCocoJsonNode writer → TrackingResultsReader reader.

The writer outputs COCO bbox+segmentation format (``images`` + ``annotations``),
which the reader detects as ``coco_bbox``.  In this mode the reader returns
bboxes/track_ids/scores (mask and object_ids are None).  These tests verify
the bbox/track roundtrip and that RLE segmentations are valid in the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from cuvis_ai_core.data.rle import coco_rle_decode

from cuvis_ai.node.json_reader import TrackingResultsReader
from cuvis_ai.node.json_writer import TrackingCocoJsonNode


def _build_inputs(
    frame_idx: int,
    mask_2d: torch.Tensor,
    object_ids: list[int],
    detection_scores: list[float],
) -> dict[str, torch.Tensor]:
    return {
        "frame_id": torch.tensor([frame_idx], dtype=torch.int64),
        "mask": mask_2d.to(dtype=torch.int32).unsqueeze(0),
        "object_ids": torch.tensor([object_ids], dtype=torch.int64),
        "detection_scores": torch.tensor([detection_scores], dtype=torch.float32),
    }


def test_roundtrip_single_object_bbox(tmp_path: Path) -> None:
    """Write one frame, read back via coco_bbox path, verify bbox and track_id."""
    json_path = tmp_path / "tracking.json"
    writer = TrackingCocoJsonNode(output_json_path=str(json_path), category_name="person")

    mask_2d = torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.int32)
    writer.forward(**_build_inputs(0, mask_2d, [1], [0.9]))
    writer.close()

    reader = TrackingResultsReader(json_path=str(json_path))
    assert reader.format == "coco_bbox"
    out = reader.forward()

    assert out["bboxes"] is not None
    assert out["track_ids"] is not None
    assert out["track_ids"][0, 0].item() == 1
    assert out["confidences"][0, 0].item() == pytest.approx(0.9)


def test_roundtrip_multi_object_bbox(tmp_path: Path) -> None:
    """Two objects roundtrip through bbox path."""
    json_path = tmp_path / "tracking.json"
    writer = TrackingCocoJsonNode(output_json_path=str(json_path), category_name="person")

    mask_2d = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 2]],
        dtype=torch.int32,
    )
    writer.forward(**_build_inputs(0, mask_2d, [1, 2], [0.9, 0.8]))
    writer.close()

    reader = TrackingResultsReader(json_path=str(json_path))
    out = reader.forward()

    track_ids = out["track_ids"][0].tolist()
    assert 1 in track_ids
    assert 2 in track_ids


def test_rle_segmentation_valid_in_json(tmp_path: Path) -> None:
    """Verify the RLE segmentation stored in JSON can be decoded back to a mask."""
    json_path = tmp_path / "tracking.json"
    writer = TrackingCocoJsonNode(output_json_path=str(json_path), category_name="person")

    mask_2d = torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.int32)
    writer.forward(**_build_inputs(0, mask_2d, [1], [0.9]))
    writer.close()

    with json_path.open() as f:
        data = json.load(f)

    ann = data["annotations"][0]
    rle = ann["segmentation"]
    decoded = coco_rle_decode(rle)

    assert decoded.shape == (2, 3)
    expected = (mask_2d == 1).numpy().astype("uint8")
    assert (decoded == expected).all()


def test_rle_non_square_mask_in_json(tmp_path: Path) -> None:
    """Non-square mask (3×7) RLE roundtrip through JSON."""
    json_path = tmp_path / "tracking.json"
    writer = TrackingCocoJsonNode(output_json_path=str(json_path), category_name="person")

    mask_2d = torch.zeros((3, 7), dtype=torch.int32)
    mask_2d[0, :] = 1

    writer.forward(**_build_inputs(0, mask_2d, [1], [0.95]))
    writer.close()

    with json_path.open() as f:
        data = json.load(f)

    rle = data["annotations"][0]["segmentation"]
    decoded = coco_rle_decode(rle)

    assert decoded.shape == (3, 7)
    assert (decoded[0, :] == 1).all()
    assert (decoded[1:, :] == 0).all()


def test_roundtrip_empty_frame(tmp_path: Path) -> None:
    """Empty frame (no objects) roundtrips through bbox path."""
    json_path = tmp_path / "tracking.json"
    writer = TrackingCocoJsonNode(
        output_json_path=str(json_path),
        category_name="person",
        write_empty_frames=True,
    )

    mask_2d = torch.zeros((4, 4), dtype=torch.int32)
    writer.forward(**_build_inputs(0, mask_2d, [], []))
    writer.close()

    reader = TrackingResultsReader(json_path=str(json_path))
    out = reader.forward()

    assert out["bboxes"].shape == (1, 0, 4)
    assert out["track_ids"].shape == (1, 0)
