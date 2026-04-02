"""Tests for DetectionJsonReader + ByteTrackCocoJson round-trip with DeepEIoU.

These are cuvis-ai-tracking-side tests verifying the JSON I/O nodes work
correctly with DeepEIoU tracking output (same writer as ByteTrack, different
tracker).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cuvis_ai.node.json_reader import DetectionJsonReader
from cuvis_ai.node.json_writer import ByteTrackCocoJson


def _write_sample_detection_json(path: Path, n_frames: int = 3, n_dets: int = 2) -> None:
    """Create a sample COCO detection JSON for testing."""
    images = []
    annotations = []
    ann_id = 1
    for frame_id in range(n_frames):
        images.append(
            {"id": frame_id, "file_name": f"frame_{frame_id:04d}.png", "height": 480, "width": 640}
        )
        for d in range(n_dets):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": frame_id,
                    "bbox": [10.0 + d * 100, 20.0, 50.0, 60.0],
                    "score": 0.9 - d * 0.1,
                    "category_id": 1,
                }
            )
            ann_id += 1

    payload = {
        "info": {"description": "test detections"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_detection_json_reader_frame_count(tmp_path: Path) -> None:
    """DetectionJsonReader correctly iterates the expected number of frames."""
    src = tmp_path / "dets.json"
    _write_sample_detection_json(src, n_frames=3)

    reader = DetectionJsonReader(json_path=str(src))

    frames = []
    try:
        while True:
            frames.append(reader.forward())
    except StopIteration:
        pass

    assert len(frames) == 3
    for i, f in enumerate(frames):
        assert int(f["frame_id"][0].item()) == i
        assert f["bboxes"].shape[1] == 2  # 2 dets per frame


def test_tracking_json_round_trip(tmp_path: Path) -> None:
    """Read detections, write tracking JSON, verify track_ids appear."""
    src = tmp_path / "input_dets.json"
    out = tmp_path / "tracking_output.json"
    _write_sample_detection_json(src, n_frames=2, n_dets=1)

    reader = DetectionJsonReader(json_path=str(src))
    writer = ByteTrackCocoJson(
        output_json_path=str(out),
        category_id_to_name={1: "person"},
        flush_interval=1,
    )

    # Simulate tracking by assigning sequential track IDs
    for track_id_val in [10, 10]:
        det = reader.forward()
        writer.forward(
            frame_id=det["frame_id"],
            bboxes=det["bboxes"],
            category_ids=det["category_ids"],
            confidences=det["confidences"],
            track_ids=torch.tensor([[track_id_val]], dtype=torch.int64),
            orig_hw=det["orig_hw"],
        )
    writer.close()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["annotations"]) == 2
    for ann in data["annotations"]:
        assert ann["track_id"] == 10
        assert ann["category_id"] == 1
