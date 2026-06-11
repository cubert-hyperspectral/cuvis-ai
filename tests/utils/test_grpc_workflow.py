"""Unit tests for the gRPC example-client helpers in ``cuvis_ai.utils.grpc_workflow``.

The stub is mocked, so these exercise the request-shaping logic without a live
cuvis-ai-core server.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cuvis_ai.utils.grpc_workflow import apply_trainrun_config, normalize_pipeline_bytes

pytestmark = pytest.mark.unit


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

    # The trainrun config sent on is the same one with the pipeline section removed.
    stub.SetTrainRunConfig.assert_called_once()
    set_req = stub.SetTrainRunConfig.call_args.args[0]
    assert set_req.session_id == "sess-1"
    sent = json.loads(set_req.config.config_bytes.decode("utf-8"))
    assert "pipeline" not in sent
    assert sent == {"trainer": {"max_epochs": 1}}


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
