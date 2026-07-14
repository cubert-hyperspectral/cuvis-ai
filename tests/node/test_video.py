from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch

import cuvis_ai.node.video as video_module
from cuvis_ai.node.video import ToVideoNode


class _RecordingFfmpegProc:
    """Minimal stand-in for ``subprocess.Popen`` handling ffmpeg's rgb24 pipe.

    Captures the argv and every byte written to stdin, and exposes a configurable
    returncode / stderr to simulate ffmpeg success or failure. ``stdin.write``
    raises the configured ``stdin_write_error`` (if any) to simulate
    ``BrokenPipeError`` during encoding.
    """

    def __init__(
        self,
        argv: list[str],
        *,
        returncode: int = 0,
        stderr_bytes: bytes = b"",
        stdin_write_error: BaseException | None = None,
        wait_timeout_first_call: bool = False,
    ) -> None:
        self.argv = argv
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._stderr_bytes = stderr_bytes
        self._stdin_write_error = stdin_write_error
        self._wait_timeout_first_call = wait_timeout_first_call
        self._wait_calls = 0
        self.killed = False

        self.stdin = _RecordingStdin(error=stdin_write_error)
        self.stderr = io.BytesIO(stderr_bytes)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        self._wait_calls += 1
        if self._wait_timeout_first_call and self._wait_calls == 1:
            import subprocess as _sp

            raise _sp.TimeoutExpired(cmd="ffmpeg", timeout=timeout or 0)
        if self.returncode is None:
            self.returncode = self._final_returncode
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = -9


class _RecordingStdin:
    """stdin stub that records bytes, or raises a configured error on write."""

    def __init__(self, error: BaseException | None = None) -> None:
        self._buf = bytearray()
        self._closed = False
        self._error = error

    def write(self, data: bytes) -> int:
        if self._error is not None:
            raise self._error
        if self._closed:
            raise BrokenPipeError("stdin closed")
        self._buf.extend(data)
        return len(data)

    def close(self) -> None:
        self._closed = True

    def getvalue(self) -> bytes:
        return bytes(self._buf)


@pytest.fixture()
def mock_ffmpeg_popen(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingFfmpegProc]:
    """Monkeypatch ``subprocess.Popen`` in video_module to record created ffmpeg procs."""
    created: list[_RecordingFfmpegProc] = []

    def _popen_factory(argv: list[str], **_kwargs: object) -> _RecordingFfmpegProc:
        proc = _RecordingFfmpegProc(argv=list(argv))
        created.append(proc)
        return proc

    monkeypatch.setattr(video_module.subprocess, "Popen", _popen_factory)
    return created


def _written_frames(proc: _RecordingFfmpegProc, height: int, width: int) -> np.ndarray:
    """Reshape captured stdin bytes back to [N, H, W, 3] RGB uint8."""
    raw = proc.stdin.getvalue()
    frame_size = height * width * 3
    assert len(raw) % frame_size == 0, (
        f"Total written bytes {len(raw)} is not a multiple of H*W*3={frame_size}"
    )
    count = len(raw) // frame_size
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3).copy()


def _argv_value_after(argv: list[str], flag: str) -> str:
    """Return the value that immediately follows ``flag`` in argv."""
    idx = argv.index(flag)
    return argv[idx + 1]


def test_to_video_node_writes_frames_across_forward_calls(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "video" / "false_rgb.mp4"
    node = ToVideoNode(output_video_path=str(output_path), frame_rate=12.5)

    batch = torch.tensor(
        [
            [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]]],
            [[[0.2, 0.4, 0.6], [0.8, 0.9, 1.0]]],
        ],
        dtype=torch.float32,
    )

    node.forward(rgb_image=batch)
    node.forward(rgb_image=batch[:1])
    node.close()

    assert len(mock_ffmpeg_popen) == 1
    proc = mock_ffmpeg_popen[0]

    assert _argv_value_after(proc.argv, "-r") == "12.5"
    assert _argv_value_after(proc.argv, "-s") == "2x1"
    assert str(output_path) == proc.argv[-1]

    frames = _written_frames(proc, height=1, width=2)
    assert frames.shape[0] == 3
    # Input RGB [1.0, 0.0, 0.5] -> uint8 [255, 0, 127]; written as rgb24, NOT BGR.
    assert frames[0, 0, 0].tolist() == [255, 0, 127]


def test_to_video_node_emits_default_codec_bitrate_and_pixfmt(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(output_video_path=str(tmp_path / "defaults.mp4"), frame_rate=10.0)
    node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))
    node.close()

    argv = mock_ffmpeg_popen[0].argv
    assert _argv_value_after(argv, "-c:v") == "libx264"
    assert _argv_value_after(argv, "-b:v") == "12M"
    # Two -pix_fmt occurrences: input rgb24 first, then output yuv420p.
    pix_fmt_values = [argv[i + 1] for i, a in enumerate(argv) if a == "-pix_fmt"]
    assert pix_fmt_values == ["rgb24", "yuv420p"]
    assert _argv_value_after(argv, "-vf") == "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    assert "-movflags" in argv and _argv_value_after(argv, "-movflags") == "+faststart"


def test_to_video_node_propagates_custom_codec_and_bitrate(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(
        output_video_path=str(tmp_path / "custom.mp4"),
        frame_rate=10.0,
        video_codec="libx265",
        bitrate="8M",
    )
    node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))
    node.close()

    argv = mock_ffmpeg_popen[0].argv
    assert _argv_value_after(argv, "-c:v") == "libx265"
    assert _argv_value_after(argv, "-b:v") == "8M"


def test_to_video_node_pad_filter_always_present_for_odd_dims(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    # Frame with odd height and width — pad filter is a fixed invariant.
    node = ToVideoNode(output_video_path=str(tmp_path / "odd.mp4"), frame_rate=10.0)
    node.forward(rgb_image=torch.zeros((1, 7, 5, 3), dtype=torch.float32))
    node.close()

    argv = mock_ffmpeg_popen[0].argv
    assert _argv_value_after(argv, "-vf") == "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    assert _argv_value_after(argv, "-s") == "5x7"


def test_to_video_node_applies_minus_90_rotation(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(
        output_video_path=str(tmp_path / "rotated.mp4"),
        frame_rate=10.0,
        frame_rotation=-90,
    )
    frame = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32)
    node.forward(rgb_image=frame)
    node.close()

    proc = mock_ffmpeg_popen[0]
    # Input frame H=1,W=2; after -90 rotation H=2,W=1 -> -s "1x2".
    assert _argv_value_after(proc.argv, "-s") == "1x2"
    frames = _written_frames(proc, height=2, width=1)
    # Top pixel red (RGB 255,0,0), bottom pixel green (RGB 0,255,0).
    assert frames[0, 0, 0].tolist() == [255, 0, 0]
    assert frames[0, 1, 0].tolist() == [0, 255, 0]


def test_to_video_node_applies_plus_90_rotation_anticlockwise(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(
        output_video_path=str(tmp_path / "rotated_ccw.mp4"),
        frame_rate=10.0,
        frame_rotation=90,
    )
    frame = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32)
    node.forward(rgb_image=frame)
    node.close()

    proc = mock_ffmpeg_popen[0]
    assert _argv_value_after(proc.argv, "-s") == "1x2"
    frames = _written_frames(proc, height=2, width=1)
    # Top pixel green, bottom pixel red (anticlockwise).
    assert frames[0, 0, 0].tolist() == [0, 255, 0]
    assert frames[0, 1, 0].tolist() == [255, 0, 0]


def test_to_video_node_rejects_inconsistent_frame_sizes(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(output_video_path=str(tmp_path / "size_mismatch.mp4"), frame_rate=10.0)

    node.forward(rgb_image=torch.zeros((1, 4, 5, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="share one size"):
        node.forward(rgb_image=torch.zeros((1, 6, 5, 3), dtype=torch.float32))


def test_to_video_node_validates_frame_rate() -> None:
    with pytest.raises(ValueError, match="frame_rate"):
        ToVideoNode(output_video_path="out.mp4", frame_rate=0.0)


def test_to_video_node_validates_video_codec() -> None:
    with pytest.raises(ValueError, match="video_codec"):
        ToVideoNode(output_video_path="out.mp4", video_codec="")


def test_to_video_node_validates_bitrate() -> None:
    with pytest.raises(ValueError, match="bitrate"):
        ToVideoNode(output_video_path="out.mp4", bitrate="")


def test_to_video_node_validates_frame_rotation() -> None:
    with pytest.raises(ValueError, match="frame_rotation"):
        ToVideoNode(output_video_path="out.mp4", frame_rotation=45)


def test_to_video_node_raises_when_ffmpeg_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _raise_file_not_found(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("ffmpeg not on PATH (simulated)")

    monkeypatch.setattr(video_module.subprocess, "Popen", _raise_file_not_found)

    node = ToVideoNode(output_video_path=str(tmp_path / "missing_ffmpeg.mp4"), frame_rate=8.0)
    with pytest.raises(RuntimeError, match="ffmpeg binary not found"):
        node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))


def test_to_video_node_forward_surfaces_broken_pipe_with_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[_RecordingFfmpegProc] = []

    def _popen_factory(argv: list[str], **_kwargs: object) -> _RecordingFfmpegProc:
        proc = _RecordingFfmpegProc(
            argv=list(argv),
            returncode=1,
            stderr_bytes=b"invalid input parameters",
            stdin_write_error=BrokenPipeError("ffmpeg died"),
        )
        captured.append(proc)
        return proc

    monkeypatch.setattr(video_module.subprocess, "Popen", _popen_factory)

    node = ToVideoNode(output_video_path=str(tmp_path / "broken.mp4"), frame_rate=10.0)
    with pytest.raises(RuntimeError, match="ffmpeg exited during frame write.*invalid input"):
        node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))


def test_to_video_node_close_raises_on_nonzero_ffmpeg_returncode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _popen_factory(argv: list[str], **_kwargs: object) -> _RecordingFfmpegProc:
        return _RecordingFfmpegProc(
            argv=list(argv),
            returncode=1,
            stderr_bytes=b"mux failed",
        )

    monkeypatch.setattr(video_module.subprocess, "Popen", _popen_factory)

    node = ToVideoNode(output_video_path=str(tmp_path / "bad_close.mp4"), frame_rate=10.0)
    node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))
    with pytest.raises(RuntimeError, match="non-zero return code 1.*mux failed"):
        node.close()


def test_to_video_node_close_is_idempotent(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(output_video_path=str(tmp_path / "idempotent.mp4"), frame_rate=10.0)
    node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))
    node.close()
    # Second close() must be a silent no-op.
    node.close()


def test_to_video_node_renders_title_centered_with_slim_background(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(
        output_video_path=str(tmp_path / "title_overlay.mp4"),
        frame_rate=10.0,
        overlay_title="Cubert XMR Camera in CIR View",
    )
    frame = torch.full((1, 90, 420, 3), 0.8, dtype=torch.float32)

    node.forward(rgb_image=frame)
    node.close()

    proc = mock_ffmpeg_popen[0]
    written = _written_frames(proc, height=90, width=420)[0]
    # Title block darkens the region; find the dark pixels on the R channel.
    dark_mask = written[..., 0] <= 60
    ys, xs = np.where(dark_mask)

    assert ys.size > 0
    x0, x1 = int(xs.min()), int(xs.max())
    box_center_x = (x0 + x1) / 2.0

    assert abs(box_center_x - ((written.shape[1] - 1) / 2.0)) <= 3.0
    assert x0 >= 90
    assert (written.shape[1] - 1 - x1) >= 20
    assert int(ys.min()) <= 10


def test_to_video_node_overlay_title_warns_v1_migration(tmp_path: Path) -> None:
    # overlay_title still renders via cv2; it announces the v1.0 move to the torch text renderer.
    with pytest.warns(DeprecationWarning, match="v1.0"):
        ToVideoNode(
            output_video_path=str(tmp_path / "warn.mp4"),
            frame_rate=10.0,
            overlay_title="Some Title",
        )


def test_to_video_node_without_title_does_not_warn(tmp_path: Path) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ToVideoNode(output_video_path=str(tmp_path / "ok.mp4"), frame_rate=10.0)


def test_to_video_node_keeps_frame_id_overlay_unchanged_when_title_is_added(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    frame = torch.full((1, 90, 420, 3), 0.8, dtype=torch.float32)
    frame_id = torch.tensor([42], dtype=torch.int64)

    baseline = ToVideoNode(
        output_video_path=str(tmp_path / "baseline.mp4"),
        frame_rate=10.0,
    )
    baseline.forward(rgb_image=frame.clone(), frame_id=frame_id)
    baseline.close()

    titled = ToVideoNode(
        output_video_path=str(tmp_path / "titled.mp4"),
        frame_rate=10.0,
        overlay_title="Cubert XMR Camera in CIR View",
    )
    titled.forward(rgb_image=frame.clone(), frame_id=frame_id)
    titled.close()

    baseline_frame = _written_frames(mock_ffmpeg_popen[0], height=90, width=420)[0]
    titled_frame = _written_frames(mock_ffmpeg_popen[1], height=90, width=420)[0]

    assert np.array_equal(baseline_frame[:40, :90], titled_frame[:40, :90])
