"""compare-deploy: parity report between a canonical pipeline and its deploy folder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner

pytestmark = pytest.mark.unit
export = pytest.importorskip("cuvis_ai_core.pipeline.export")
pytest.importorskip("onnxruntime")

from scripts.compare_deploy import main  # noqa: E402
from tests.node.test_export_span_builtin import WAVELENGTHS, _build_pipeline  # noqa: E402

#: restore_pipeline (unlike CuvisPipeline.load_pipeline, which the exporter's own test
#: calls directly) resolves every node class against a plugins directory once one is
#: given, so the CLI invocations here always pass two: this repo's bundled manifests
#: (the cuvis_ai_onnx entry the deploy yaml declares) and a per-test directory holding
#: a bare manifest for the toy pipeline's test-local source node.
_BUILTIN_PLUGINS_DIR = Path(__file__).resolve().parents[2] / "cuvis_ai" / "configs" / "plugins"
_SOURCE_PLUGIN_NAME = "test_compare_deploy_source"


def _write_source_manifest(plugins_dir: Path) -> None:
    """A bare manifest so restore_pipeline's resolver can cover the toy pipeline's
    test-local source node, which ships with no manifest of its own."""
    plugins_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": _SOURCE_PLUGIN_NAME,
        "package_name": "cuvis-ai",
        "path": str(Path(__file__).resolve().parents[2]),
        "capabilities": [
            {
                "class_name": "tests.node.test_export_span_builtin._Source",
                "category": "source",
                "tags": [],
            }
        ],
    }
    (plugins_dir / f"{_SOURCE_PLUGIN_NAME}.yaml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )


def _add_plugin(yaml_path: Path, plugin_name: str) -> None:
    """Add ``plugin_name`` to a pipeline yaml's ``plugins:`` list in place."""
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    config["plugins"] = [*(config.get("plugins") or []), plugin_name]
    yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _export_toy(tmp_path: Path):
    """Build the toy pipeline, save it as the canonical yaml, and export its deploy
    folder; returns (canonical_yaml_path, deploy_yaml_path, frame).

    Both yamls get a ``plugins:`` entry for the test-local source node (the deploy
    yaml also carries ``cuvis_ai_onnx`` from the export); the manifest lives in a
    local plugins directory the caller must pass to ``compare-deploy`` as a second
    ``--plugins-dir``.
    """
    pipeline, frame = _build_pipeline(tmp_path)
    examples = [
        {
            "normalizer.data": frame,
            "selector.wavelengths": torch.tensor(WAVELENGTHS, dtype=torch.int32),
        }
    ]
    deployment = export.export_deployment(
        pipeline,
        tmp_path / "deploy",
        inputs=["normalizer.data", "selector.wavelengths"],
        outputs=["selector.rgb_image"],
        examples=examples,
        constants=["selector.wavelengths"],
    )

    canonical_path = tmp_path / "toy.yaml"
    pipeline.save_to_file(str(canonical_path))

    local_plugins_dir = tmp_path / "plugins"
    _write_source_manifest(local_plugins_dir)
    # cuvis_ai_builtin covers MinMaxNormalizer / FixedWavelengthSelector; only the
    # test-local source node needs the extra manifest.
    _add_plugin(canonical_path, "cuvis_ai_builtin")
    _add_plugin(canonical_path, _SOURCE_PLUGIN_NAME)
    _add_plugin(deployment.yaml_path, _SOURCE_PLUGIN_NAME)

    return canonical_path, deployment.yaml_path, frame, local_plugins_dir


def _write_frame_npz(path: Path, frame: torch.Tensor) -> None:
    """A frame as an .npz keyed the way the toy pipeline's ports are named."""
    np.savez(
        path,
        **{
            "normalizer.data": frame.numpy(),
            "selector.wavelengths": np.asarray(WAVELENGTHS, dtype=np.int32),
        },
    )


def test_compare_deploy_reports_near_zero_diff_on_matching_frames(tmp_path) -> None:
    """The deploy folder reproduces the canonical pipeline: a tiny max abs diff, an
    informative report written to --out."""
    canonical_path, deploy_path, frame, local_plugins_dir = _export_toy(tmp_path)
    frame_path = tmp_path / "frame0.npz"
    _write_frame_npz(frame_path, frame)
    out_path = tmp_path / "report.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--canonical",
            str(canonical_path),
            "--deploy",
            str(deploy_path),
            "--frame",
            str(frame_path),
            "--plugins-dir",
            str(_BUILTIN_PLUGINS_DIR),
            "--plugins-dir",
            str(local_plugins_dir),
            "--device",
            "cpu",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_path.is_file()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["frames"] == 1
    assert report["canonical_node"] == "selector"
    assert report["span_node"] == "exported_span"
    assert "rgb_image" in report["max_abs_diff"]
    assert report["max_abs_diff"]["rgb_image"] < 1e-4
    assert report["decider_node"] is None
    assert "Frames compared: 1" in result.output
    assert "rgb_image" in result.output


def test_compare_deploy_requires_frames_or_data_module(tmp_path) -> None:
    """Without --frame or --data-module the CLI refuses with a usage error."""
    canonical_path, deploy_path, _frame, _plugins_dir = _export_toy(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--canonical",
            str(canonical_path),
            "--deploy",
            str(deploy_path),
            "--out",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code != 0
    assert "--frame" in result.output or "--data-module" in result.output


def test_compare_deploy_derives_canonical_node_from_sidecar(tmp_path) -> None:
    """--canonical-node defaults to the node the sidecar's span.outputs names."""
    canonical_path, deploy_path, frame, local_plugins_dir = _export_toy(tmp_path)
    frame_path = tmp_path / "frame0.npz"
    _write_frame_npz(frame_path, frame)
    out_path = tmp_path / "report.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--canonical",
            str(canonical_path),
            "--deploy",
            str(deploy_path),
            "--frame",
            str(frame_path),
            "--plugins-dir",
            str(_BUILTIN_PLUGINS_DIR),
            "--plugins-dir",
            str(local_plugins_dir),
            "--device",
            "cpu",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["canonical_node"] == "selector"
