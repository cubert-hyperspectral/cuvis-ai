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


def test_parse_plugin_fast_rgb_config_resolves_evaluated_ranges() -> None:
    tmp_dir = _make_local_tmp_dir()
    xml_path = _write_plugin_xml(
        tmp_dir,
        red_wl=780.0,
        green_wl=780.0,
        blue_wl=514.0,
        width=20.0,
        normalization=0.75,
    )

    cfg = export_mod._parse_plugin_fast_rgb_config(xml_path)
    assert cfg.red_range == (770.0, 790.0)
    assert cfg.green_range == (770.0, 790.0)
    assert cfg.blue_range == (504.0, 524.0)
    assert cfg.normalization_strength == 0.75


def test_parse_plugin_fast_rgb_config_missing_reference_raises() -> None:
    tmp_dir = _make_local_tmp_dir()
    xml_path = tmp_dir / "bad_plugin.xml"
    xml_path.write_text(
        (
            '<?xml version="1.0"?>\n'
            '<userplugin xmlns="http://cubert-gmbh.de/user/plugin/userplugin.xsd">\n'
            '  <configuration name="RGB">\n'
            '    <input id="Normalize" type="scalar">0.75</input>\n'
            '    <output_image id="RGB">\n'
            '      <fast_rgb red_min="MissingRef" red_max="MissingRef" '
            'green_min="MissingRef" green_max="MissingRef" '
            'blue_min="MissingRef" blue_max="MissingRef" normalization="Normalize">\n'
            "        <cube />\n"
            "      </fast_rgb>\n"
            "    </output_image>\n"
            "  </configuration>\n"
            "</userplugin>\n"
        ),
        encoding="utf-8",
    )

    try:
        export_mod._parse_plugin_fast_rgb_config(xml_path)
        assert False, "Expected ValueError for unresolved XML reference"
    except ValueError as exc:
        assert "not found" in str(exc)


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
    def __init__(self, norm_mode: str | NormMode | None = None) -> None:
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


def _write_plugin_xml(
    tmp_dir: Path,
    *,
    red_wl: float = 780.0,
    green_wl: float = 780.0,
    blue_wl: float = 514.0,
    width: float = 20.0,
    normalization: float = 0.75,
) -> Path:
    xml_path = tmp_dir / "invisible_ink.xml"
    xml_path.write_text(
        (
            '<?xml version="1.0"?>\n'
            '<userplugin xmlns="http://cubert-gmbh.de/user/plugin/userplugin.xsd">\n'
            '  <configuration name="RGB">\n'
            f'    <input id="RedWL" type="scalar">{red_wl}</input>\n'
            f'    <input id="GreenWL" type="scalar">{green_wl}</input>\n'
            f'    <input id="BlueWL" type="scalar">{blue_wl}</input>\n'
            f'    <input id="Width" type="scalar">{width}</input>\n'
            f'    <input id="Normalize" type="scalar">{normalization}</input>\n'
            '    <evaluate id="HalfWidth">\n'
            '      <operator type="divide">\n'
            '        <variable ref="Width" />\n'
            "        <value>2</value>\n"
            "      </operator>\n"
            "    </evaluate>\n"
            '    <evaluate id="RedMin">\n'
            '      <operator type="subtract">\n'
            '        <variable ref="RedWL" />\n'
            '        <variable ref="HalfWidth" />\n'
            "      </operator>\n"
            "    </evaluate>\n"
            '    <evaluate id="RedMax">\n'
            '      <operator type="add">\n'
            '        <variable ref="RedWL" />\n'
            '        <variable ref="HalfWidth" />\n'
            "      </operator>\n"
            "    </evaluate>\n"
            '    <evaluate id="GreenMin">\n'
            '      <operator type="subtract">\n'
            '        <variable ref="GreenWL" />\n'
            '        <variable ref="HalfWidth" />\n'
            "      </operator>\n"
            "    </evaluate>\n"
            '    <evaluate id="GreenMax">\n'
            '      <operator type="add">\n'
            '        <variable ref="GreenWL" />\n'
            '        <variable ref="HalfWidth" />\n'
            "      </operator>\n"
            "    </evaluate>\n"
            '    <evaluate id="BlueMin">\n'
            '      <operator type="subtract">\n'
            '        <variable ref="BlueWL" />\n'
            '        <variable ref="HalfWidth" />\n'
            "      </operator>\n"
            "    </evaluate>\n"
            '    <evaluate id="BlueMax">\n'
            '      <operator type="add">\n'
            '        <variable ref="BlueWL" />\n'
            '        <variable ref="HalfWidth" />\n'
            "      </operator>\n"
            "    </evaluate>\n"
            '    <output_image show="true" id="RGB">\n'
            '      <fast_rgb red_min="RedMin" red_max="RedMax" green_min="GreenMin" '
            'green_max="GreenMax" blue_min="BlueMin" blue_max="BlueMax" normalization="Normalize">\n'
            "        <cube />\n"
            "      </fast_rgb>\n"
            "    </output_image>\n"
            "  </configuration>\n"
            "</userplugin>\n"
        ),
        encoding="utf-8",
    )
    return xml_path


def _run_export_with_mode(
    monkeypatch,
    tmp_dir: Path,
    normalization_mode: str,
    method: str = "cie_tristimulus",
    processing_mode: str = "Raw",
    fast_rgb_normalization_strength: float | None = None,
    plugin_xml_path: str | None = None,
    plugin_config: export_mod.PluginFastRGBConfig | None = None,
) -> tuple[_FakeSelector, dict[str, object]]:
    created: dict[str, object] = {}

    def fake_create_false_rgb_node(method: str, **kwargs: object) -> _FakeSelector:
        created["method"] = method
        created["kwargs"] = kwargs
        selector = _FakeSelector(norm_mode=kwargs.get("norm_mode"))
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
    if method == "cuvis-plugin":
        effective_xml_path = plugin_xml_path or str(tmp_dir / "plugin.xml")
        Path(effective_xml_path).write_text("<root/>", encoding="utf-8")
        plugin_xml_path = effective_xml_path
        if plugin_config is None:
            plugin_config = export_mod.PluginFastRGBConfig(
                red_range=(770.0, 790.0),
                green_range=(770.0, 790.0),
                blue_range=(504.0, 524.0),
                normalization_strength=0.75,
            )
        monkeypatch.setattr(export_mod, "_parse_plugin_fast_rgb_config", lambda _: plugin_config)

    export_mod.export_false_rgb_video(
        cu3s_file_path="dummy.cu3s",
        output_video_path=str(out_path),
        method=method,
        plugin_xml_path=plugin_xml_path,
        normalization_mode=normalization_mode,
        sample_fraction=0.05,
        processing_mode=processing_mode,
        fast_rgb_normalization_strength=fast_rgb_normalization_strength,
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


def test_export_live_running_fixed_uses_running_without_statistical_init(monkeypatch) -> None:
    selector, created = _run_export_with_mode(
        monkeypatch,
        _make_local_tmp_dir(),
        normalization_mode="live_running_fixed",
    )
    assert created["kwargs"]["norm_mode"] == NormMode.RUNNING
    assert created["kwargs"]["running_warmup_frames"] == 0
    assert created["kwargs"]["freeze_running_bounds_after_frames"] == 1
    assert selector.statistical_init_calls == 0


def test_export_fast_rgb_uses_canonical_method_and_no_statistical_init(monkeypatch) -> None:
    selector, created = _run_export_with_mode(
        monkeypatch,
        _make_local_tmp_dir(),
        normalization_mode="sampled_fixed",
        method="fastrgb",
    )
    assert created["method"] == "fast_rgb"
    assert created["kwargs"]["fast_rgb_normalization_strength"] == 0.75
    assert selector.statistical_init_calls == 0


def test_export_fast_rgb_reflectance_defaults_to_static_scaling(monkeypatch) -> None:
    _, created = _run_export_with_mode(
        monkeypatch,
        _make_local_tmp_dir(),
        normalization_mode="sampled_fixed",
        method="fast_rgb",
        processing_mode="Reflectance",
    )
    assert created["method"] == "fast_rgb"
    assert created["kwargs"]["fast_rgb_normalization_strength"] == 0.0


def test_export_fast_rgb_normalization_strength_override(monkeypatch) -> None:
    _, created = _run_export_with_mode(
        monkeypatch,
        _make_local_tmp_dir(),
        normalization_mode="sampled_fixed",
        method="fast_rgb",
        fast_rgb_normalization_strength=0.42,
    )
    assert created["method"] == "fast_rgb"
    assert created["kwargs"]["fast_rgb_normalization_strength"] == 0.42


def test_export_cuvis_plugin_uses_parsed_fast_rgb_parameters(monkeypatch) -> None:
    plugin_cfg = export_mod.PluginFastRGBConfig(
        red_range=(770.0, 790.0),
        green_range=(770.0, 790.0),
        blue_range=(504.0, 524.0),
        normalization_strength=0.75,
    )
    selector, created = _run_export_with_mode(
        monkeypatch,
        _make_local_tmp_dir(),
        normalization_mode="sampled_fixed",
        method="cuvis-plugin",
        plugin_config=plugin_cfg,
    )
    assert created["method"] == "fast_rgb"
    assert created["kwargs"]["red_low"] == 770.0
    assert created["kwargs"]["red_high"] == 790.0
    assert created["kwargs"]["green_low"] == 770.0
    assert created["kwargs"]["green_high"] == 790.0
    assert created["kwargs"]["blue_low"] == 504.0
    assert created["kwargs"]["blue_high"] == 524.0
    assert created["kwargs"]["fast_rgb_normalization_strength"] == 0.75
    assert selector.statistical_init_calls == 0


def test_export_cuvis_plugin_allows_normalization_override(monkeypatch) -> None:
    plugin_cfg = export_mod.PluginFastRGBConfig(
        red_range=(770.0, 790.0),
        green_range=(770.0, 790.0),
        blue_range=(504.0, 524.0),
        normalization_strength=0.75,
    )
    _, created = _run_export_with_mode(
        monkeypatch,
        _make_local_tmp_dir(),
        normalization_mode="sampled_fixed",
        method="cuvis-plugin",
        plugin_config=plugin_cfg,
        fast_rgb_normalization_strength=0.42,
    )
    assert created["method"] == "fast_rgb"
    assert created["kwargs"]["fast_rgb_normalization_strength"] == 0.42


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
    output_root = tmp_dir / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
        ],
    )

    expected_video = output_root / cu3s.stem / f"{cu3s.stem}.mp4"
    assert result.exit_code == 0, result.output
    assert captured["output_video_path"] == str(expected_video)
    assert captured["normalization_mode"] == "sampled_fixed"
    assert captured["sample_fraction"] == 0.05


def test_cli_rejects_invalid_sample_fraction(monkeypatch) -> None:
    # Prevent any heavy export execution; we only validate CLI argument handling.
    monkeypatch.setattr(export_mod, "export_false_rgb_video", lambda **_: Path("unused.mp4"))

    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    output_root = tmp_dir / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--sample-fraction",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "sample_fraction must be in (0, 1]" in result.output


def test_cli_allows_zero_sample_fraction_when_not_sampled_fixed(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_export_false_rgb_video(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(kwargs["output_video_path"])

    monkeypatch.setattr(export_mod, "export_false_rgb_video", fake_export_false_rgb_video)

    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    output_root = tmp_dir / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--normalization-mode",
            "running",
            "--sample-fraction",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["normalization_mode"] == "running"
    assert captured["sample_fraction"] == 0.0


def test_cli_accepts_fast_rgb_alias(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_export_false_rgb_video(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(kwargs["output_video_path"])

    monkeypatch.setattr(export_mod, "export_false_rgb_video", fake_export_false_rgb_video)

    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    output_root = tmp_dir / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--method",
            "fastrgb",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "fastrgb"


def test_cli_requires_plugin_xml_path_for_cuvis_plugin(monkeypatch) -> None:
    monkeypatch.setattr(export_mod, "export_false_rgb_video", lambda **_: Path("unused.mp4"))

    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    output_root = tmp_dir / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--method",
            "cuvis-plugin",
        ],
    )

    assert result.exit_code != 0
    assert "--plugin-xml-path is required when --method cuvis-plugin" in result.output


def test_cli_accepts_plugin_xml_path_for_cuvis_plugin(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_export_false_rgb_video(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(kwargs["output_video_path"])

    monkeypatch.setattr(export_mod, "export_false_rgb_video", fake_export_false_rgb_video)

    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    xml_path = _write_plugin_xml(tmp_dir)
    output_root = tmp_dir / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--method",
            "cuvis-plugin",
            "--plugin-xml-path",
            str(xml_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "cuvis-plugin"
    assert captured["plugin_xml_path"] == str(xml_path)


def test_cli_uses_out_basename_for_run_folder(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_export_false_rgb_video(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(kwargs["output_video_path"])

    monkeypatch.setattr(export_mod, "export_false_rgb_video", fake_export_false_rgb_video)

    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    output_root = tmp_dir / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--out-basename",
            "custom_run",
        ],
    )

    expected_video = output_root / "custom_run" / f"{cu3s.stem}.mp4"
    assert result.exit_code == 0, result.output
    assert captured["output_video_path"] == str(expected_video)


def test_cli_rejects_invalid_out_basename(monkeypatch) -> None:
    monkeypatch.setattr(export_mod, "export_false_rgb_video", lambda **_: Path("unused.mp4"))

    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    output_root = tmp_dir / "outputs"
    runner = CliRunner()

    whitespace_result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--out-basename",
            "   ",
        ],
    )
    assert whitespace_result.exit_code != 0
    assert "--out-basename must not be empty or whitespace only" in whitespace_result.output

    path_result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--out-basename",
            "bad/name",
        ],
    )
    assert path_result.exit_code != 0
    assert "--out-basename must be a folder name, not a path" in path_result.output


def test_cli_help_uses_new_surface_and_removes_legacy_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(export_mod.main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--cu3s-path" in result.output
    assert "--output-dir" in result.output
    assert "--out-basename" in result.output
    assert "--output-video-path" not in result.output
    assert "--compare-all" not in result.output


def test_cli_rejects_removed_output_video_path_flag() -> None:
    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    output_root = tmp_dir / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--output-video-path",
            str(output_root / "legacy.mp4"),
        ],
    )

    assert result.exit_code != 0
    assert "No such option: --output-video-path" in result.output


def test_cli_rejects_removed_compare_all_flag() -> None:
    tmp_dir = _make_local_tmp_dir()
    cu3s = tmp_dir / "sample.cu3s"
    cu3s.write_bytes(b"")
    output_root = tmp_dir / "outputs"
    runner = CliRunner()
    result = runner.invoke(
        export_mod.main,
        [
            "--cu3s-path",
            str(cu3s),
            "--output-dir",
            str(output_root),
            "--compare-all",
            str(output_root / "compare"),
        ],
    )

    assert result.exit_code != 0
    assert "No such option: --compare-all" in result.output
