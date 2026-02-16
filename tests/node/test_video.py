from __future__ import annotations

from pathlib import Path

import pytest
import torch

import cuvis_ai.node.video as video_module
from cuvis_ai.node.video import ToVideoNode


class _RecordingWriter:
    def __init__(
        self,
        path: str,
        fourcc: int,
        fps: float,
        size: tuple[int, int],
        opened: bool = True,
    ) -> None:
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.opened = opened
        self.frames: list = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def write(self, frame) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


@pytest.fixture()
def mock_cv2_video_writer(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingWriter]:
    """Monkeypatch cv2.VideoWriter with a recording stub and return the writers list."""
    created_writers: list[_RecordingWriter] = []

    def _writer_factory(
        path: str, fourcc: int, fps: float, size: tuple[int, int]
    ) -> _RecordingWriter:
        writer = _RecordingWriter(path=path, fourcc=fourcc, fps=fps, size=size, opened=True)
        created_writers.append(writer)
        return writer

    monkeypatch.setattr(video_module.cv2, "VideoWriter", _writer_factory)
    monkeypatch.setattr(video_module.cv2, "VideoWriter_fourcc", lambda *_: 42)
    return created_writers


def test_to_video_node_writes_frames_across_forward_calls(
    mock_cv2_video_writer: list[_RecordingWriter],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "video" / "false_rgb.mp4"
    node = ToVideoNode(output__video_path=str(output_path), frame_rate=12.5)

    batch = torch.tensor(
        [
            [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]]],
            [[[0.2, 0.4, 0.6], [0.8, 0.9, 1.0]]],
        ],
        dtype=torch.float32,
    )

    out_1 = node.forward(rgb_image=batch)
    out_2 = node.forward(rgb_image=batch[:1])
    node.close()

    assert out_1["video_path"] == str(output_path)
    assert out_2["video_path"] == str(output_path)
    assert len(mock_cv2_video_writer) == 1
    writer = mock_cv2_video_writer[0]
    assert writer.fps == 12.5
    assert writer.size == (2, 1)
    assert len(writer.frames) == 3
    # Input RGB [255, 0, 127] should be written as BGR [127, 0, 255].
    assert writer.frames[0][0, 0].tolist() == [127, 0, 255]
    assert writer.released is True


def test_to_video_node_applies_minus_90_rotation(
    mock_cv2_video_writer: list[_RecordingWriter],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(
        output__video_path=str(tmp_path / "rotated.mp4"),
        frame_rate=10.0,
        frame_rotation=-90,
    )
    frame = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32)
    node.forward(rgb_image=frame)
    node.close()

    writer = mock_cv2_video_writer[0]
    # Input frame is H=1,W=2; after -90 rotation it's H=2,W=1 -> size=(1,2)
    assert writer.size == (1, 2)
    # Top pixel should be red (BGR [0,0,255]), bottom pixel green (BGR [0,255,0]).
    assert writer.frames[0][0, 0].tolist() == [0, 0, 255]
    assert writer.frames[0][1, 0].tolist() == [0, 255, 0]


def test_to_video_node_applies_plus_90_rotation_anticlockwise(
    mock_cv2_video_writer: list[_RecordingWriter],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(
        output__video_path=str(tmp_path / "rotated_ccw.mp4"),
        frame_rate=10.0,
        frame_rotation=90,
    )
    frame = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32)
    node.forward(rgb_image=frame)
    node.close()

    writer = mock_cv2_video_writer[0]
    # Input frame is H=1,W=2; after +90 (anticlockwise) rotation it's H=2,W=1 -> size=(1,2)
    assert writer.size == (1, 2)
    # Top pixel should be green (BGR [0,255,0]), bottom pixel red (BGR [0,0,255]).
    assert writer.frames[0][0, 0].tolist() == [0, 255, 0]
    assert writer.frames[0][1, 0].tolist() == [0, 0, 255]


def test_to_video_node_rejects_inconsistent_frame_sizes(
    mock_cv2_video_writer: list[_RecordingWriter],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(output__video_path=str(tmp_path / "size_mismatch.mp4"), frame_rate=10.0)

    node.forward(rgb_image=torch.zeros((1, 4, 5, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="share one size"):
        node.forward(rgb_image=torch.zeros((1, 6, 5, 3), dtype=torch.float32))


def test_to_video_node_validates_frame_rate() -> None:
    with pytest.raises(ValueError, match="frame_rate"):
        ToVideoNode(output__video_path="out.mp4", frame_rate=0.0)


def test_to_video_node_validates_frame_rotation() -> None:
    with pytest.raises(ValueError, match="frame_rotation"):
        ToVideoNode(output__video_path="out.mp4", frame_rotation=45)


def test_to_video_node_raises_when_writer_fails_to_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        video_module.cv2,
        "VideoWriter",
        lambda path, fourcc, fps, size: _RecordingWriter(path, fourcc, fps, size, opened=False),
    )
    monkeypatch.setattr(video_module.cv2, "VideoWriter_fourcc", lambda *_: 42)

    node = ToVideoNode(output__video_path=str(tmp_path / "bad_writer.mp4"), frame_rate=8.0)
    with pytest.raises(RuntimeError, match="Failed to open video writer"):
        node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))
