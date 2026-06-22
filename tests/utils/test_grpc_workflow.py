"""Unit tests for the gRPC example-client helpers in ``cuvis_ai.utils.grpc_workflow``.

The stub is mocked, so these exercise the request-shaping logic without a live
cuvis-ai-core server.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import yaml

from cuvis_ai.utils.grpc_workflow import (
    apply_trainrun_config,
    inline_pipeline_ref,
    load_manifest_bytes,
    normalize_pipeline_bytes,
    resolve_pipeline_ref,
)

pytestmark = pytest.mark.unit


def test_load_manifest_bytes_resolves_local_path_to_absolute(tmp_path) -> None:
    """A bare local manifest's relative path is resolved to absolute (server can't)."""
    manifest = tmp_path / "my_plugin.yaml"
    manifest.write_text(
        "name: my_plugin\npath: ../sibling\ncapabilities:\n  - class_name: pkg.mod.Node\n",
        encoding="utf-8",
    )
    payload = json.loads(load_manifest_bytes(manifest).decode("utf-8"))
    assert payload["name"] == "my_plugin"
    assert "plugins" not in payload  # bare manifest, no wrapper
    assert payload["path"] == str((tmp_path / ".." / "sibling").resolve())


def test_load_manifest_bytes_leaves_git_manifest_unchanged(tmp_path) -> None:
    """A git manifest (repo + tag) has no local path to resolve."""
    manifest = tmp_path / "sam3.yaml"
    manifest.write_text(
        "name: sam3\nrepo: https://github.com/x/sam3.git\ntag: v1.0.0\n"
        "capabilities:\n  - class_name: cuvis_ai_sam3.node.Sam3\n",
        encoding="utf-8",
    )
    payload = json.loads(load_manifest_bytes(manifest).decode("utf-8"))
    assert payload["repo"] == "https://github.com/x/sam3.git"
    assert payload["tag"] == "v1.0.0"
    assert "path" not in payload


def test_apply_trainrun_config_loads_pipeline_then_sets_config() -> None:
    """A trainrun carrying a ``pipeline`` section LoadPipelines it, then strips it."""
    stub = MagicMock()
    pipeline_section = {
        "nodes": {"n0": {"type": "FastRGBSelector"}},
        "plugins": ["cuvis_ai_builtin"],
    }
    config = {"pipeline": pipeline_section, "trainer": {"max_epochs": 1}}
    config_bytes = json.dumps(config).encode("utf-8")

    apply_trainrun_config(stub, "sess-1", config_bytes)

    # The embedded pipeline is loaded first, normalized to a PipelineConfig payload.
    stub.LoadPipeline.assert_called_once()
    load_req = stub.LoadPipeline.call_args.args[0]
    assert load_req.session_id == "sess-1"
    assert load_req.pipeline.config_bytes == normalize_pipeline_bytes(
        json.dumps(pipeline_section).encode("utf-8")
    )

    # No data section -> no data_module hint on the load request.
    assert load_req.data_module == ""

    # The trainrun config sent on is the same one with the pipeline section removed.
    stub.SetTrainRunConfig.assert_called_once()
    set_req = stub.SetTrainRunConfig.call_args.args[0]
    assert set_req.session_id == "sess-1"
    sent = json.loads(set_req.config.config_bytes.decode("utf-8"))
    assert "pipeline" not in sent
    assert sent == {"trainer": {"max_epochs": 1}}


def test_apply_trainrun_config_forwards_data_module() -> None:
    """The data section's ``data_module`` rides on the load request as a bare name."""
    stub = MagicMock()
    config = {
        "pipeline": {"nodes": {"n0": {"type": "FastRGBSelector"}}, "plugins": ["cuvis_ai_builtin"]},
        "data": {"data_module": "cu3s", "params": {"cu3s_file_path": "/x.cu3s"}},
        "trainer": {"max_epochs": 1},
    }
    config_bytes = json.dumps(config).encode("utf-8")

    apply_trainrun_config(stub, "sess-data", config_bytes)

    load_req = stub.LoadPipeline.call_args.args[0]
    # Only the module name is forwarded, not the splits/params payload.
    assert load_req.data_module == "cu3s"


def test_apply_trainrun_config_rejects_pipeline_reference() -> None:
    """A path-reference pipeline can't be inlined; the helper points at RestoreTrainRun."""
    stub = MagicMock()
    config = {"pipeline": "../pipeline/anomaly/adaclip/adaclip_baseline.yaml", "trainer": {}}
    config_bytes = json.dumps(config).encode("utf-8")

    with pytest.raises(ValueError, match="RestoreTrainRun"):
        apply_trainrun_config(stub, "sess-ref", config_bytes)

    stub.LoadPipeline.assert_not_called()
    stub.SetTrainRunConfig.assert_not_called()


def test_apply_trainrun_config_without_pipeline_skips_load() -> None:
    """A trainrun with no ``pipeline`` section goes straight to SetTrainRunConfig."""
    stub = MagicMock()
    config = {"trainer": {"max_epochs": 2}}
    config_bytes = json.dumps(config).encode("utf-8")

    apply_trainrun_config(stub, "sess-2", config_bytes)

    stub.LoadPipeline.assert_not_called()
    stub.SetTrainRunConfig.assert_called_once()
    set_req = stub.SetTrainRunConfig.call_args.args[0]
    assert set_req.session_id == "sess-2"
    assert json.loads(set_req.config.config_bytes.decode("utf-8")) == config


def test_resolve_pipeline_ref_loads_yaml_relative_to_trainrun_dir(tmp_path) -> None:
    """A path-referenced pipeline is loaded relative to the given trainrun directory."""
    pipeline = {"nodes": {"n0": {"type": "FastRGBSelector"}}, "plugins": ["cuvis_ai_builtin"]}
    (tmp_path / "pipe.yaml").write_text(yaml.safe_dump(pipeline), encoding="utf-8")

    loaded = resolve_pipeline_ref("pipe.yaml", trainrun_dir=tmp_path)
    assert loaded == pipeline


def test_inline_pipeline_ref_inlines_string_reference(tmp_path) -> None:
    """A string ``pipeline`` is replaced by the loaded mapping; the input is not mutated."""
    pipeline = {"nodes": {"n0": {"type": "FastRGBSelector"}}, "plugins": ["cuvis_ai_builtin"]}
    (tmp_path / "pipe.yaml").write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    config = {"pipeline": "pipe.yaml", "trainer": {"max_epochs": 1}}

    inlined = inline_pipeline_ref(config, trainrun_dir=tmp_path)
    assert inlined["pipeline"] == pipeline
    assert inlined["trainer"] == {"max_epochs": 1}
    assert config["pipeline"] == "pipe.yaml"  # original left untouched


def test_inline_pipeline_ref_is_noop_for_inline_or_absent_pipeline() -> None:
    """An already-inline or missing ``pipeline`` returns the same dict unchanged."""
    inline = {"pipeline": {"nodes": {"n0": {"type": "FastRGBSelector"}}}, "trainer": {}}
    assert inline_pipeline_ref(inline) is inline

    absent = {"trainer": {"max_epochs": 3}}
    assert inline_pipeline_ref(absent) is absent
