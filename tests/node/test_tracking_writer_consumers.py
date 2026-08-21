"""Integration tests: actual CocoTrackMaskWriter output feeding downstream consumers.

The consumer tests elsewhere construct image-dialect JSON by hand, so they alone cannot
prove the writer's default output is readable. These feed real writer files into
``MaskPrompt``, the occlusion nodes, and ``append_tracking_metrics``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cuvis_ai.node.json_file import CocoTrackMaskWriter
from cuvis_ai.node.occlusion import SolidOcclusionNode
from cuvis_ai.node.prompts import MaskPrompt
from cuvis_ai.utils.cli_helpers import append_tracking_metrics


@pytest.fixture
def writer_output(tmp_path: Path) -> tuple[Path, torch.Tensor]:
    """One frame with two tracked rectangles, written by the real writer (image dialect)."""
    json_path = tmp_path / "tracking.json"
    writer = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

    label_map = torch.zeros((8, 10), dtype=torch.int32)
    label_map[2:5, 3:7] = 1
    label_map[5:8, 0:3] = 2
    writer.forward(
        frame_id=torch.tensor([0], dtype=torch.int64),
        mask=label_map.unsqueeze(0),
        object_ids=torch.tensor([[1, 2]], dtype=torch.int64),
        detection_scores=torch.tensor([[0.5, 0.25]], dtype=torch.float32),
    )
    writer.close()
    return json_path, label_map


def test_mask_prompt_consumes_writer_output(
    writer_output: tuple[Path, torch.Tensor],
) -> None:
    json_path, label_map = writer_output

    node = MaskPrompt(json_path=str(json_path), prompt_specs=["9:1@0"])
    out = node.forward(frame_id=torch.tensor([0], dtype=torch.int64))

    assert out["mask"].shape == (1, 8, 10)
    expected = (label_map == 1).to(torch.int32) * 9
    assert torch.equal(out["mask"][0], expected)


def test_occlusion_node_consumes_writer_output(
    writer_output: tuple[Path, torch.Tensor],
) -> None:
    json_path, label_map = writer_output

    node = SolidOcclusionNode(
        tracking_json_path=str(json_path),
        track_ids=[1],
        occlusion_start_frame=0,
        occlusion_end_frame=0,
        fill_color=(0.0, 1.0, 0.0),
        occlusion_shape="mask",
    )
    rgb = torch.full((1, 8, 10, 3), 0.2, dtype=torch.float32)
    out = node.forward(rgb_image=rgb, frame_id=torch.tensor([0], dtype=torch.int64))["rgb_image"]

    track_mask = label_map == 1
    assert out[0][track_mask, 1].min().item() == pytest.approx(1.0)
    assert torch.equal(out[0][~track_mask, :], rgb[0][~track_mask, :])


def test_append_tracking_metrics_consumes_writer_output(
    writer_output: tuple[Path, torch.Tensor], tmp_path: Path
) -> None:
    json_path, _ = writer_output
    info_path = tmp_path / "experiment_info.txt"
    info_path.write_text("Experiment: test\n", encoding="utf-8")

    append_tracking_metrics(info_path, json_path)

    content = info_path.read_text(encoding="utf-8")
    assert "frames: 1" in content
    assert "unique_track_ids: 2" in content
    assert "zero_track_frames: 0" in content
