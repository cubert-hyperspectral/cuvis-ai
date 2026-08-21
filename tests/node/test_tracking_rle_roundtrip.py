"""Roundtrip tests for CocoTrackMaskWriter and TrackingResultsReader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cuvis_ai.node.json_file import CocoTrackMaskWriter, TrackingResultsReader
from cuvis_ai_core.data.rle import coco_rle_decode, coco_rle_encode


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


def _write_image_dialect(tmp_path: Path, annotations: list[dict], images: list[dict]) -> Path:
    path = tmp_path / "image_dialect.json"
    payload = {
        "info": {},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rect_rle(height: int, width: int, r0: int, r1: int, c0: int, c1: int) -> dict:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    return coco_rle_encode(mask)


# ---------------------------------------------------------------------------
# Writer -> reader roundtrips
# ---------------------------------------------------------------------------


def test_roundtrip_single_object_mask(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking.json"
    writer = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

    mask_2d = torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.int32)
    writer.forward(**_build_inputs(0, mask_2d, [1], [0.9]))
    writer.close()

    reader = TrackingResultsReader(json_path=str(json_path))
    assert reader.format == "coco_bbox"
    out = reader.forward()

    assert out["object_ids"][0].tolist() == [1]
    assert out["mask"].shape == (1, 2, 3)
    assert torch.equal(out["mask"][0], mask_2d)
    assert out["track_ids"][0].tolist() == [1]
    assert out["bboxes"].shape == (1, 1, 4)


def test_roundtrip_multi_object_mask(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking.json"
    writer = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

    mask_2d = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 2]],
        dtype=torch.int32,
    )
    writer.forward(**_build_inputs(0, mask_2d, [1, 2], [0.9, 0.8]))
    writer.close()

    reader = TrackingResultsReader(json_path=str(json_path))
    out = reader.forward()

    assert sorted(out["object_ids"][0].tolist()) == [1, 2]
    assert torch.equal(out["mask"][0], mask_2d)


def test_roundtrip_video_dialect(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking.json"
    writer = CocoTrackMaskWriter(
        output_json_path=str(json_path), dialect="video", default_category_name="person"
    )

    mask_2d = torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.int32)
    writer.forward(**_build_inputs(0, mask_2d, [1], [0.9]))
    writer.close()

    reader = TrackingResultsReader(json_path=str(json_path))
    assert reader.format == "video_coco"
    out = reader.forward()

    assert out["object_ids"][0].tolist() == [1]
    assert torch.equal(out["mask"][0], mask_2d)


def test_rle_segmentation_valid_in_json(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking.json"
    writer = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

    mask_2d = torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.int32)
    writer.forward(**_build_inputs(0, mask_2d, [1], [0.9]))
    writer.close()

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    ann = data["annotations"][0]
    decoded = coco_rle_decode(ann["segmentation"])

    assert decoded.shape == (2, 3)
    expected = (mask_2d == 1).numpy().astype("uint8")
    assert (decoded == expected).all()


def test_rle_non_square_mask_in_json(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking.json"
    writer = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

    mask_2d = torch.zeros((3, 7), dtype=torch.int32)
    mask_2d[0, :] = 1

    writer.forward(**_build_inputs(0, mask_2d, [1], [0.95]))
    writer.close()

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    decoded = coco_rle_decode(data["annotations"][0]["segmentation"])

    assert decoded.shape == (3, 7)
    assert (decoded[0, :] == 1).all()
    assert (decoded[1:, :] == 0).all()


def test_roundtrip_empty_frame_in_mask_session(tmp_path: Path) -> None:
    """A frame without objects reads back as a zero label map at the frame size."""
    json_path = tmp_path / "tracking.json"
    writer = CocoTrackMaskWriter(
        output_json_path=str(json_path),
        default_category_name="person",
        write_empty_frames=True,
    )

    mask_2d = torch.zeros((4, 4), dtype=torch.int32)
    mask_2d[0, 0] = 1
    writer.forward(**_build_inputs(0, mask_2d, [1], [0.9]))
    writer.forward(**_build_inputs(1, torch.zeros((4, 4), dtype=torch.int32), [], []))
    writer.close()

    reader = TrackingResultsReader(json_path=str(json_path))
    out0 = reader.forward()
    out1 = reader.forward()

    assert torch.equal(out0["mask"][0], mask_2d)
    assert out1["object_ids"].shape == (1, 0)
    assert out1["mask"].shape == (1, 4, 4)
    assert torch.count_nonzero(out1["mask"]).item() == 0


def test_all_empty_session_keeps_legacy_empty_mask(tmp_path: Path) -> None:
    """A file with no segmentations at all keeps the legacy empty mask shape."""
    json_path = tmp_path / "tracking.json"
    writer = CocoTrackMaskWriter(
        output_json_path=str(json_path),
        default_category_name="person",
        write_empty_frames=True,
    )
    writer.forward(**_build_inputs(0, torch.zeros((4, 4), dtype=torch.int32), [], []))
    writer.close()

    reader = TrackingResultsReader(json_path=str(json_path))
    out = reader.forward()

    assert out["mask"].shape == (1, 0, 0)
    assert out["object_ids"].shape == (1, 0)


def test_required_format_matches_image_dialect_output(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking.json"
    writer = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")
    writer.forward(
        **_build_inputs(0, torch.tensor([[0, 1], [0, 1]], dtype=torch.int32), [1], [0.9])
    )
    writer.close()

    reader = TrackingResultsReader(json_path=str(json_path), required_format="coco_bbox")
    assert reader.forward()["object_ids"][0].tolist() == [1]

    mismatched = TrackingResultsReader(json_path=str(json_path), required_format="video_coco")
    with pytest.raises(ValueError, match="required_format"):
        mismatched.forward()


# ---------------------------------------------------------------------------
# Image-dialect reader specifics (hand-written files)
# ---------------------------------------------------------------------------


def test_reader_derives_bbox_from_bboxless_segmentation(tmp_path: Path) -> None:
    rle = _rect_rle(4, 6, 1, 3, 2, 5)
    path = _write_image_dialect(
        tmp_path,
        annotations=[{"id": 1, "image_id": 0, "category_id": 1, "segmentation": rle}],
        images=[{"id": 0, "file_name": "frame_000000", "height": 4, "width": 6}],
    )

    out = TrackingResultsReader(json_path=str(path)).forward()

    assert out["bboxes"][0].tolist() == [[2.0, 1.0, 5.0, 3.0]]  # xyxy from RLE
    assert (out["mask"][0][1:3, 2:5] == 1).all()  # no track_id -> annotation id
    assert out["object_ids"][0].tolist() == [1]


def test_reader_overlap_higher_annotation_id_wins(tmp_path: Path) -> None:
    base = _rect_rle(4, 4, 0, 3, 0, 3)
    overlap = _rect_rle(4, 4, 1, 4, 1, 4)
    path = _write_image_dialect(
        tmp_path,
        annotations=[
            {"id": 2, "image_id": 0, "category_id": 1, "segmentation": overlap, "track_id": 3},
            {"id": 1, "image_id": 0, "category_id": 1, "segmentation": base, "track_id": 5},
        ],
        images=[{"id": 0, "file_name": "frame_000000", "height": 4, "width": 4}],
    )

    out = TrackingResultsReader(json_path=str(path)).forward()

    # Annotation id 2 (track 3) paints last: the overlapping pixel belongs to track 3.
    assert int(out["mask"][0][1, 1]) == 3
    assert int(out["mask"][0][0, 0]) == 5
    assert out["object_ids"][0].tolist() == [3, 5]


def test_reader_rejects_rle_size_mismatch(tmp_path: Path) -> None:
    rle = _rect_rle(3, 3, 0, 2, 0, 2)
    path = _write_image_dialect(
        tmp_path,
        annotations=[{"id": 1, "image_id": 0, "category_id": 1, "segmentation": rle}],
        images=[{"id": 0, "file_name": "frame_000000", "height": 5, "width": 5}],
    )

    with pytest.raises(ValueError, match="disagrees with the image record"):
        TrackingResultsReader(json_path=str(path)).forward()


def test_reader_mixed_bbox_and_mask_annotations(tmp_path: Path) -> None:
    rle = _rect_rle(4, 4, 0, 2, 0, 2)
    path = _write_image_dialect(
        tmp_path,
        annotations=[
            {"id": 1, "image_id": 0, "category_id": 1, "segmentation": rle, "track_id": 7},
            {
                "id": 2,
                "image_id": 0,
                "category_id": 1,
                "bbox": [1.0, 1.0, 2.0, 2.0],
                "track_id": 8,
            },
        ],
        images=[{"id": 0, "file_name": "frame_000000", "height": 4, "width": 4}],
    )

    out = TrackingResultsReader(json_path=str(path)).forward()

    assert out["bboxes"].shape == (1, 2, 4)
    assert out["track_ids"][0].tolist() == [7, 8]
    # Only the segmentation-bearing annotation lands in the label map.
    assert out["object_ids"][0].tolist() == [7]
    assert int((out["mask"][0] == 7).sum()) == 4


def test_reader_rejects_annotation_without_bbox_or_segmentation(tmp_path: Path) -> None:
    path = _write_image_dialect(
        tmp_path,
        annotations=[{"id": 1, "image_id": 0, "category_id": 1}],
        images=[{"id": 0, "file_name": "frame_000000", "height": 4, "width": 4}],
    )

    with pytest.raises(ValueError, match="neither a bbox nor an RLE segmentation"):
        TrackingResultsReader(json_path=str(path)).forward()


def test_reader_pure_bbox_file_keeps_legacy_empty_mask(tmp_path: Path) -> None:
    path = _write_image_dialect(
        tmp_path,
        annotations=[
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": [1.0, 1.0, 2.0, 2.0],
                "track_id": 4,
            }
        ],
        images=[{"id": 0, "file_name": "frame_000000", "height": 4, "width": 4}],
    )

    out = TrackingResultsReader(json_path=str(path)).forward()

    assert out["mask"].shape == (1, 0, 0)
    assert out["object_ids"].shape == (1, 0)
    assert out["track_ids"][0].tolist() == [4]
