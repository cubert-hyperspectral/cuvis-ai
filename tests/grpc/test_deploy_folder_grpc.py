"""In-process gRPC end-to-end check of an exported deploy folder (ALL-6819).

Exports the toy pipeline (the same one ``tests/node/test_export_span_builtin.py``
builds), points ``CUVIS_AI_DEPLOY_DIR`` at the deploy folder, then drives
cuvis-ai-core's production servicers directly: ``CreateSession``, ``LoadPlugin`` for
every manifest the deploy yaml names, ``LoadPipeline``, ``LoadPipelineWeights`` and
``Inference``. The in-memory orchestrator (``install_in_memory_orchestrator``) routes
``LoadPipeline`` / ``Inference`` through a ``RunRuntimeServicer`` in this process, the
same code path production uses minus the composed venv and the subprocess, so this is a
model for CuvisNEXT's own deploy-folder loading over gRPC.

Fast: a 4x4 cube, one frame, CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai.node.exported_span import DEPLOY_DIR_ENV

pytestmark = pytest.mark.unit
grpc_mod = pytest.importorskip("grpc")
pytest.importorskip("onnxruntime")
export = pytest.importorskip("cuvis_ai_core.pipeline.export")

from cuvis_ai_core.grpc import orchestrator_bridge  # noqa: E402
from cuvis_ai_core.grpc.helpers import proto_to_tensor  # noqa: E402
from cuvis_ai_core.grpc.service import CuvisAIService  # noqa: E402
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2  # noqa: E402
from tests.node.test_export_span_builtin import WAVELENGTHS, _build_pipeline  # noqa: E402

#: The name under which the toy pipeline's test-local source node is registered
#: as a plugin for this test only; added to the deploy yaml's ``plugins:``
#: alongside ``cuvis_ai_onnx`` so the resolver finds a provider for every node.
_SOURCE_PLUGIN_NAME = "test_deploy_folder_source"


class _FakeContext:
    """A gRPC servicer context that records the status an error path sets.

    The in-process servicers read ``context.set_code`` / ``context.set_details`` on
    their error branches and otherwise ignore the context, so a small recorder stands
    in for the real ``grpc.ServicerContext`` the production server would pass.
    """

    def __init__(self) -> None:
        self.code: grpc_mod.StatusCode | None = None
        self.details: str | None = None

    def set_code(self, code: grpc_mod.StatusCode) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _source_manifest() -> bytes:
    """A bare manifest for the toy pipeline's test-local source node.

    The in-memory orchestrator never installs anything (``_noop_composer`` is a
    stand-in), so ``path`` is required by the schema and must be absolute (the
    server refuses a client-relative one), but it is never actually read; only
    ``capabilities`` matters, since the child resolves node classes by plain
    ``importlib.import_module`` of each ``class_name``.
    """
    manifest = {
        "name": _SOURCE_PLUGIN_NAME,
        "package_name": "cuvis-ai",
        "path": str(_REPO_ROOT),
        "capabilities": [
            {
                "class_name": "tests.node.test_export_span_builtin._Source",
                "category": "source",
                "tags": [],
            }
        ],
    }
    return json.dumps(manifest).encode("utf-8")


def _onnx_manifest() -> bytes:
    """The ``cuvis_ai_onnx`` manifest the deploy yaml declares, read from this repo."""
    manifest_path = _REPO_ROOT / "cuvis_ai" / "configs" / "plugins" / "cuvis_ai_onnx.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return json.dumps(manifest).encode("utf-8")


def _deploy_config_with_source_plugin(yaml_path: Path) -> dict:
    """The deploy yaml as a dict, with the test-local source plugin added to
    ``plugins:`` alongside ``cuvis_ai_onnx`` so the resolver has a declared
    provider for every node class in the pipeline, not only the exported span."""
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    config["plugins"] = [*config.get("plugins", []), _SOURCE_PLUGIN_NAME]
    return config


@pytest.fixture
def deploy_folder(tmp_path, monkeypatch):
    """Export the toy pipeline into ``tmp_path`` / "deploy" and return (deployment, pipeline)."""
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
    return deployment, pipeline, frame


@pytest.fixture
def in_memory_orchestrator():
    """Route LoadPipeline / Inference through an in-process child, no subprocess."""
    orchestrator_bridge.install_in_memory_orchestrator()
    try:
        yield
    finally:
        orchestrator_bridge.reset_orchestrator()


def test_deploy_folder_served_over_grpc_matches_the_canonical_pipeline(
    deploy_folder, in_memory_orchestrator, monkeypatch
) -> None:
    """LoadPipeline + LoadPipelineWeights + Inference on a deploy folder reproduce
    the canonical pipeline's own forward pass, driven only through gRPC bytes."""
    deployment, pipeline, frame = deploy_folder
    monkeypatch.setenv(DEPLOY_DIR_ENV, str(deployment.yaml_path.parent))

    reference = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

    service = CuvisAIService()
    context = _FakeContext()
    session_id = service.CreateSession(cuvis_ai_pb2.CreateSessionRequest(), context).session_id
    assert session_id

    for manifest_bytes in (_source_manifest(), _onnx_manifest()):
        load_resp = service.LoadPlugin(
            cuvis_ai_pb2.LoadPluginRequest(
                session_id=session_id,
                manifest=cuvis_ai_pb2.PluginManifest(config_bytes=manifest_bytes),
            ),
            context,
        )
        assert load_resp.registered_plugin, load_resp.error

    deploy_config = _deploy_config_with_source_plugin(deployment.yaml_path)
    pipeline_resp = service.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=json.dumps(deploy_config).encode("utf-8")
            ),
        ),
        context,
    )
    assert pipeline_resp.success, context.details

    weights_resp = service.LoadPipelineWeights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id,
            weights_bytes=deployment.weights_path.read_bytes(),
        ),
        context,
    )
    assert weights_resp.success, context.details

    inference_resp = service.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(),
            output_specs=["exported_span.rgb_image"],
        ),
        context,
    )
    assert "exported_span.rgb_image" in inference_resp.outputs

    with proto_to_tensor(inference_resp.outputs["exported_span.rgb_image"]) as served:
        torch.testing.assert_close(
            served.clone(),
            reference[("selector", "rgb_image")],
            atol=1e-5,
            rtol=1e-4,
        )


def test_exported_span_without_deploy_dir_names_the_three_lookup_places(
    deploy_folder, in_memory_orchestrator, monkeypatch, tmp_path
) -> None:
    """With no CUVIS_AI_DEPLOY_DIR, a gRPC LoadPipeline (bytes, not a file path) gives
    ExportedSpan no pipeline-file folder to try either, so the only place actually
    tried is the working directory; the failure still names all three remedies
    (the env var, loading the deploy yaml from its folder, an absolute onnx_path)."""
    deployment, _pipeline, _frame = deploy_folder
    monkeypatch.delenv(DEPLOY_DIR_ENV, raising=False)
    # Run from somewhere that is not the deploy folder, so "the working directory"
    # fallback cannot accidentally find the file either.
    monkeypatch.chdir(tmp_path)

    deploy_config = _deploy_config_with_source_plugin(deployment.yaml_path)

    service = CuvisAIService()
    context = _FakeContext()
    session_id = service.CreateSession(cuvis_ai_pb2.CreateSessionRequest(), context).session_id

    for manifest_bytes in (_source_manifest(), _onnx_manifest()):
        load_resp = service.LoadPlugin(
            cuvis_ai_pb2.LoadPluginRequest(
                session_id=session_id,
                manifest=cuvis_ai_pb2.PluginManifest(config_bytes=manifest_bytes),
            ),
            context,
        )
        assert load_resp.registered_plugin, load_resp.error

    pipeline_resp = service.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=json.dumps(deploy_config).encode("utf-8")
            ),
        ),
        context,
    )

    assert not pipeline_resp.success
    assert context.details is not None
    # The only place actually tried, since gRPC supplies no pipeline file folder
    # and the env var is unset.
    assert "working directory" in context.details
    # The three remedies the error always names, regardless of what was tried.
    assert DEPLOY_DIR_ENV in context.details
    assert "deploy yaml from its folder" in context.details
    assert "absolute onnx_path" in context.details
