from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from click.testing import CliRunner

import examples.object_tracking.export_cu3s_false_rgb_video as export_mod
from cuvis_ai.node.channel_selector import NormMode


def test_uniform_sample_positions_deterministic_and_bounded() -> None:
    positions_a = export_mod._uniform_sample_positions(total_frames=100, sample_fraction=0.05)
    positions_b = export_mod._uniform_sample_positions(total_frames=100, sample_fraction=0.05)

    assert positions_a == positions_b
    assert len(positions_a) == 5
    assert positions_a[0] == 0
    assert positions_a[-1] == 99
    assert positions_a == sorted(positions_a)
    assert len(set(positions_a)) == len(positions_a)
    assert all(0 <= pos < 100 for pos in positions_a)


def test_uniform_sample_positions_uses_ceiling_count() -> None:
    # ceil(11 * 0.05) = 1
    positions = export_mod._uniform_sample_positions(total_frames=11, sample_fraction=0.05)
    assert len(positions) == 1


def test_uniform_sample_positions_invalid_fraction_raises() -> None:
    try:
        export_mod._uniform_sample_positions(total_frames=10, sample_fraction=0.0)
        assert False, "Expected ValueError for sample_fraction=0.0"
    except ValueError as exc:
        assert "sample_fraction" in str(exc)


class _FakePredictDataset:
    def __init__(self, total_frames: int) -> None:
        self._total_frames = total_frames
        self.fps = 15.0
        self.measurement_indices = list(range(total_frames))

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        cube = np.full((2, 2, 3), fill_value=float(idx), dtype=np.float32)
        wavelengths = np.array([620.0, 530.0, 450.0], dtype=np.float32)
        return {"cube": cube, "wavelengths": wavelengths, "mesu_index": idx}


class _FakeDataModule:
    def __init__(self, **_: object) -> None:
        self.predict_ds = _FakePredictDataset(total_frames=40)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        return None


class _FakeCU3SDataNode:
    def __init__(self, **_: object) -> None:
        self.outputs = SimpleNamespace(cube="cube_out", wavelengths="wl_out", mesu_index="mesu_out")


class _FakeToVideoNode:
    def __init__(self, **_: object) -> None:
        self.rgb_image = "rgb_in"
        self.frame_id = "frame_id_in"


class _FakePipeline:
    def __init__(self, name: str) -> None:
        self.name = name

    def connect(self, *_: object) -> None:
        return None

    def visualize(self, **_: object) -> None:
        return None

    def to(self, _: object) -> None:
        return None

    def save_to_file(self, path: str) -> None:
        Path(path).write_text("pipeline: fake\n", encoding="utf-8")


class _FakePredictor:
    output_video_path: str = ""

    def __init__(self, **_: object) -> None:
        return None

    def predict(self, **_: object) -> None:
        Path(self.output_video_path).write_bytes(b"fake")


class _FakeSelector:
    def __init__(self, norm_mode: str | NormMode) -> None:
        self.norm_mode = norm_mode
        self.cube = "cube_in"
        self.wavelengths = "wl_in"
        self.rgb_image = "rgb_out"
        self.statistical_init_calls = 0
        self.sampled_count = 0

    def statistical_initialization(self, input_stream: object) -> None:
        self.statistical_init_calls += 1
        self.sampled_count = sum(1 for _ in input_stream)


def _make_local_tmp_dir() -> Path:
    root = Path(".tmp_test_export_cu3s")
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / uuid.uuid4().hex
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _run_export_with_mode(
    monkeypatch,
    tmp_dir: Path,
    normalization_mode: str,
) -> tuple[_FakeSelector, dict[str, object]]:
    created: dict[str, object] = {}

    def fake_create_false_rgb_node(method: str, **kwargs: object) -> _FakeSelector:  # noqa: ARG001
        created["kwargs"] = kwargs
        selector = _FakeSelector(norm_mode=kwargs["norm_mode"])
        created["selector"] = selector
        return selector

    out_path = tmp_dir / f"out_{normalization_mode}.mp4"
    _FakePredictor.output_video_path = str(out_path)

    monkeypatch.setattr(export_mod, "SingleCu3sDataModule", _FakeDataModule)
    monkeypatch.setattr(export_mod, "CU3SDataNode", _FakeCU3SDataNode)
    monkeypatch.setattr(export_mod, "ToVideoNode", _FakeToVideoNode)
    monkeypatch.setattr(export_mod, "CuvisPipeline", _FakePipeline)
    monkeypatch.setattr(export_mod, "Predictor", _FakePredictor)
    monkeypatch.setattr(export_mod, "_create_false_rgb_node", fake_create_false_rgb_node)

    export_mod.export_false_rgb_video(
        cu3s_file_path="dummy.cu3s",
        output_video_path=str(out_path),
        method="cie_tristimulus",
        normalization_mode=normalization_mode,
        sample_fraction=0.05,
        processing_mode="Raw",
    )
    return created["selector"], created


def test_export_sampled_fixed_calls_statistical_initialization(monkeypatch) -> None:
    selector, created = _run_export_with_mode(
        monkeypatch,
        _make_local_tmp_dir(),
        normalization_mode="sampled_fixed",
    )
    assert created["kwargs"]["norm_mode"] == NormMode.STATISTICAL
    assert selector.statistical_init_calls == 1
    # 40 frames, 5% sample => ceil(2.0) = 2 sampled frames.
    assert selector.sampled_count == 2


def test_export_running_does_not_call_statistical_initialization(monkeypatch) -> None:
    selector, created = _run_export_with_mode(
        monkeypatch,
        _make_local_tmp_dir(),
        normalization_mode="running",
    )
    assert created["kwargs"]["norm_mode"] == NormMode.RUNNING
    assert selector.statistical_init_calls == 0


def test_export_per_frame_does_not_call_statistical_initialization(monkeypatch) -> None:
    selector, created = _run_export_with_mode(
        monkeypatch,
        _make_local_tmp_dir(),
        normalization_mode="per_frame",
    )
    assert created["kwargs"]["norm_mode"] == NormMode.PER_FRAME
    assert selector.statistical_init_calls == 0


def test_cli_defaults_to_sampled_fixed_and_default_sample_fraction(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_export_false_rgb_video(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(kwargs["output_video_path"])

    monkeypatch.setattr(export_mod, "export_false_rgb_video", fake_export_false_rgb_video)

    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    out = tmp_dir / "out.mp4"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-file-path",
            str(cu3s),
            "--output-video-path",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["normalization_mode"] == "sampled_fixed"
    assert captured["sample_fraction"] == 0.05


def test_cli_rejects_invalid_sample_fraction(monkeypatch) -> None:
    # Prevent any heavy export execution; we only validate CLI argument handling.
    monkeypatch.setattr(export_mod, "export_false_rgb_video", lambda **_: Path("unused.mp4"))

    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    out = tmp_dir / "out.mp4"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-file-path",
            str(cu3s),
            "--output-video-path",
            str(out),
            "--sample-fraction",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "sample_fraction must be in (0, 1]" in result.output
