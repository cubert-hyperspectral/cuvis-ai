from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import cuvis_ai.node.video as video_module
from cuvis_ai.node.video import ToImage, ToVideoNode, _FrameRenderMixin


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


def test_to_video_node_write_mode_partial_uses_fragmented_movflags(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(output_video_path=str(tmp_path / "frag.mp4"), write_mode="partial")
    node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))
    node.close()

    argv = mock_ffmpeg_popen[0].argv
    assert _argv_value_after(argv, "-movflags") == "+frag_keyframe+empty_moov+default_base_moof"


def test_to_video_node_validates_write_mode() -> None:
    with pytest.raises(ValueError, match="write_mode"):
        ToVideoNode(output_video_path="out.mp4", write_mode="bogus")


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


def test_to_video_node_cleanup_surfaces_finalize_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A finalize failure at teardown is logged at ERROR and recorded, not swallowed.

    ``CuvisPipeline.cleanup`` downgrades any ``cleanup()`` exception to a warning, so
    the node must surface a failed trailer flush itself (ERROR log + ``_finalize_error``)
    and still re-raise, rather than leave an unplayable file while the run looks clean.
    """
    from loguru import logger

    def _popen_factory(argv: list[str], **_kwargs: object) -> _RecordingFfmpegProc:
        return _RecordingFfmpegProc(
            argv=list(argv),
            returncode=1,
            stderr_bytes=b"moov atom write failed",
        )

    monkeypatch.setattr(video_module.subprocess, "Popen", _popen_factory)

    node = ToVideoNode(output_video_path=str(tmp_path / "teardown.mp4"), frame_rate=10.0)
    node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))

    errors_seen: list[str] = []
    sink_id = logger.add(lambda m: errors_seen.append(m.record["message"]), level="ERROR")
    try:
        with pytest.raises(RuntimeError, match="non-zero return code 1.*moov atom"):
            node.cleanup()
    finally:
        logger.remove(sink_id)

    assert node._finalize_error is not None
    assert "moov atom" in node._finalize_error
    assert any("failed to finalize" in msg.lower() for msg in errors_seen)


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


# ---------------------------------------------------------------------------
# ToImage
# ---------------------------------------------------------------------------
def test_to_image_writes_one_file_per_frame_named_by_frame_id(tmp_path: Path) -> None:
    out_dir = tmp_path / "seed"
    node = ToImage(output_dir=str(out_dir))

    # Two frames (H=6, W=8) with distinct top-left pixels, away from the text overlay.
    f0 = torch.zeros((6, 8, 3), dtype=torch.float32)
    f0[0, 0] = torch.tensor([1.0, 0.0, 0.5])  # -> uint8 [255, 0, 127]
    f1 = torch.zeros((6, 8, 3), dtype=torch.float32)
    f1[0, 0] = torch.tensor([0.0, 1.0, 0.0])  # -> uint8 [0, 255, 0]
    batch = torch.stack([f0, f1])  # [2, 6, 8, 3]
    frame_id = torch.tensor([5, 9], dtype=torch.int64)

    result = node.forward(rgb_image=batch, frame_id=frame_id)
    assert result == {}

    p5 = out_dir / "frame_000005.png"
    p9 = out_dir / "frame_000009.png"
    assert p5.exists() and p9.exists()

    img5 = cv2.imread(str(p5))  # BGR, lossless PNG
    assert img5.shape == (6, 8, 3)
    # BGR -> RGB round-trips exactly through a PNG.
    assert img5[0, 0][::-1].tolist() == [255, 0, 127]
    assert cv2.imread(str(p9))[0, 0][::-1].tolist() == [0, 255, 0]


def test_to_image_uses_running_counter_without_frame_id(tmp_path: Path) -> None:
    out_dir = tmp_path / "counter"
    node = ToImage(output_dir=str(out_dir))
    batch = torch.zeros((2, 4, 4, 3), dtype=torch.float32)

    node.forward(rgb_image=batch)
    node.forward(rgb_image=batch[:1])  # counter must persist across forward calls

    names = sorted(p.name for p in out_dir.glob("*.png"))
    assert names == ["frame_000000.png", "frame_000001.png", "frame_000002.png"]


def test_to_image_infers_format_from_pattern_extension(tmp_path: Path) -> None:
    out_dir = tmp_path / "jpg"
    node = ToImage(output_dir=str(out_dir), filename_pattern="shot_{frame_id:03d}.jpg")
    node.forward(
        rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32),
        frame_id=torch.tensor([7], dtype=torch.int64),
    )
    written = out_dir / "shot_007.jpg"
    assert written.exists()
    assert cv2.imread(str(written)).shape == (4, 4, 3)


def test_to_image_applies_minus_90_rotation(tmp_path: Path) -> None:
    out_dir = tmp_path / "rot"
    node = ToImage(output_dir=str(out_dir), frame_rotation=-90)
    frame = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32)
    node.forward(rgb_image=frame)

    img = cv2.imread(str(out_dir / "frame_000000.png"))
    # Input H=1,W=2; after -90 rotation H=2,W=1 (same semantics as ToVideoNode).
    assert img.shape == (2, 1, 3)
    assert img[0, 0][::-1].tolist() == [255, 0, 0]  # top pixel red
    assert img[1, 0][::-1].tolist() == [0, 255, 0]  # bottom pixel green


def test_to_image_is_sink_with_no_outputs() -> None:
    assert ToImage.OUTPUT_SPECS == {}


def test_to_image_validates_output_dir() -> None:
    with pytest.raises(ValueError, match="output_dir"):
        ToImage(output_dir="")


def test_to_image_validates_filename_pattern(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filename_pattern"):
        ToImage(output_dir=str(tmp_path), filename_pattern="")


def test_to_image_requires_extension_in_pattern(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="extension"):
        ToImage(output_dir=str(tmp_path), filename_pattern="frame_{frame_id}")


def test_to_image_validates_frame_rotation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame_rotation"):
        ToImage(output_dir=str(tmp_path), frame_rotation=45)


def test_to_image_applies_180_rotation(tmp_path: Path) -> None:
    out_dir = tmp_path / "rot180"
    node = ToImage(output_dir=str(out_dir), frame_rotation=180)
    frame = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32)
    node.forward(rgb_image=frame)

    img = cv2.imread(str(out_dir / "frame_000000.png"))
    # Input H=1,W=2 (left red, right green); a 180 rotation reverses the row.
    assert img.shape == (1, 2, 3)
    assert img[0, 0][::-1].tolist() == [0, 255, 0]  # left pixel now green
    assert img[0, 1][::-1].tolist() == [255, 0, 0]  # right pixel now red


def test_to_image_rejects_non_bhwc_input(tmp_path: Path) -> None:
    node = ToImage(output_dir=str(tmp_path / "badshape"))
    with pytest.raises(ValueError, match=r"\[B, H, W, 3\]"):
        node.forward(rgb_image=torch.zeros((4, 4, 3), dtype=torch.float32))


def test_to_image_converts_integer_input_and_clamps(tmp_path: Path) -> None:
    out_dir = tmp_path / "int_input"
    node = ToImage(output_dir=str(out_dir))
    # Non-float, non-uint8 input takes the clamp-to-[0, 255] path (300 -> 255).
    node.forward(rgb_image=torch.full((1, 4, 4, 3), 300, dtype=torch.int32))

    img = cv2.imread(str(out_dir / "frame_000000.png"))
    assert img.shape == (4, 4, 3)
    assert img[0, 0].tolist() == [255, 255, 255]


def test_to_image_title_overlay_uses_fallback_margin_on_narrow_frame(tmp_path: Path) -> None:
    out_dir = tmp_path / "narrow_title"
    # Width 100 <= 2 * reserved_side_margin (192) exercises the fallback-margin branch.
    node = ToImage(output_dir=str(out_dir), overlay_title="Cubert XMR")
    node.forward(rgb_image=torch.full((1, 90, 100, 3), 0.8, dtype=torch.float32))

    assert (out_dir / "frame_000000.png").exists()


def test_to_image_raises_when_imwrite_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(video_module.cv2, "imwrite", lambda *_a, **_k: False)

    node = ToImage(output_dir=str(tmp_path / "imwrite_fail"))
    with pytest.raises(RuntimeError, match="cv2.imwrite failed"):
        node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))


def test_to_video_node_cleanup_finalizes_video(
    mock_ffmpeg_popen: list[_RecordingFfmpegProc],
    tmp_path: Path,
) -> None:
    node = ToVideoNode(output_video_path=str(tmp_path / "cleanup.mp4"), frame_rate=10.0)
    node.forward(rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.float32))
    node.cleanup()

    # cleanup() flushes the encoder via close(): the ffmpeg stdin is closed cleanly.
    assert mock_ffmpeg_popen[0].stdin._closed is True
    # cleanup() is close() + super().cleanup(); a second call must stay a no-op.
    node.cleanup()


# ---------------------------------------------------------------------------
# _FrameRenderMixin — direct helper coverage for degenerate / defensive paths
# ---------------------------------------------------------------------------
def test_darken_region_noop_on_degenerate_box() -> None:
    frame = np.full((6, 6, 3), 200, dtype=np.uint8)
    before = frame.copy()
    _FrameRenderMixin._darken_region(frame, x0=5, y0=5, x1=5, y1=5)  # x1 <= x0
    assert np.array_equal(frame, before)


def test_darken_region_noop_on_empty_region() -> None:
    frame = np.full((6, 6, 3), 200, dtype=np.uint8)
    before = frame.copy()
    # x0 beyond the frame width -> the slice is empty though x1 > x0 and y1 > y0.
    _FrameRenderMixin._darken_region(frame, x0=100, y0=0, x1=200, y1=5)
    assert np.array_equal(frame, before)


def test_draw_title_overlay_noop_on_zero_size_frame(tmp_path: Path) -> None:
    node = ToImage(output_dir=str(tmp_path / "zero"), overlay_title="X")
    frame = torch.zeros((0, 8, 3), dtype=torch.uint8)  # zero-height frame
    node._draw_title_overlay(frame)  # returns early, no raise
    assert frame.numel() == 0


def test_normalize_rotation_passthrough_for_unexpected_value() -> None:
    # Defensive fallthrough: both __init__ methods validate rotation first, so this
    # is only reachable by calling the helper directly with an out-of-contract value.
    assert _FrameRenderMixin._normalize_rotation(45) == 45


def test_rotate_frame_passthrough_for_unexpected_value(tmp_path: Path) -> None:
    node = ToImage(output_dir=str(tmp_path / "rot_passthrough"))  # frame_rotation None
    node.frame_rotation = 45  # force an out-of-contract value past __init__ validation
    frame = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    assert torch.equal(node._rotate_frame(frame), frame)
