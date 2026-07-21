"""Shared helpers for gRPC example clients (session create / build / train / predict)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import grpc
import yaml
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def config_search_paths(extra_paths: Iterable[str | Path] | None = None) -> list[str]:
    """Return absolute search paths covering all config groups."""
    seeds = [
        CONFIG_ROOT,
        CONFIG_ROOT / "trainrun",
        CONFIG_ROOT / "pipeline",
        CONFIG_ROOT / "data",
        CONFIG_ROOT / "training",
    ]

    seen: set[Path] = set()
    paths: list[str] = []

    for path in [*seeds, *(extra_paths or [])]:
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(str(resolved))

    return paths


def build_stub(
    server_address: str = "localhost:50051", max_msg_size: int = 300 * 1024 * 1024
) -> cuvis_ai_pb2_grpc.CuvisAIServiceStub:
    """Create a gRPC stub for the CuvisAI service.

    Parameters
    ----------
    server_address : str
        Server address (default: localhost:50051)
    max_msg_size : int
        Maximum message size in bytes (default: 300MB)
    """
    options = [
        ("grpc.max_send_message_length", max_msg_size),
        ("grpc.max_receive_message_length", max_msg_size),
    ]
    channel = grpc.insecure_channel(server_address, options=options)
    return cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)


def create_session_with_search_paths(
    stub: cuvis_ai_pb2_grpc.CuvisAIServiceStub, search_paths: list[str] | None = None
) -> str:
    """Create a session and register search paths."""
    session_id = stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id
    paths = search_paths or config_search_paths()
    stub.SetSessionSearchPaths(
        cuvis_ai_pb2.SetSessionSearchPathsRequest(
            session_id=session_id,
            search_paths=paths,
            append=False,
        )
    )
    return session_id


def resolve_trainrun_config(
    stub: cuvis_ai_pb2_grpc.CuvisAIServiceStub,
    session_id: str,
    name: str,
    overrides: list[str] | None = None,
) -> tuple[cuvis_ai_pb2.ResolveConfigResponse, dict]:
    """Resolve a trainrun config via the ConfigService."""
    config_path = name if name.startswith("trainrun/") else f"trainrun/{name}"
    response = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="trainrun",
            path=config_path,
            overrides=overrides or [],
        )
    )
    config_dict = json.loads(response.config_bytes.decode("utf-8"))
    return response, config_dict


def apply_trainrun_config(
    stub: cuvis_ai_pb2_grpc.CuvisAIServiceStub,
    session_id: str,
    config_bytes: bytes,
) -> cuvis_ai_pb2.SetTrainRunConfigResponse:
    """Build the trainrun's pipeline, then attach the rest of the trainrun config.

    ``SetTrainRunConfig`` no longer builds a pipeline and rejects a trainrun
    config that still carries a ``pipeline:`` section. So this helper splits the
    resolved trainrun: it ``LoadPipeline``s the embedded ``pipeline`` section
    first (the server composes the per-run env from that pipeline's ``plugins:``
    block), then sends the remaining trainrun config to ``SetTrainRunConfig``.

    The embedded pipeline must declare a ``plugins:`` block (the standalone
    ``cuvis_ai/configs/pipeline/`` yamls already do); a trainrun whose resolved pipeline
    omits it is rejected at ``LoadPipeline`` with a ``suggest-plugins-fix`` hint.

    A trainrun whose ``pipeline`` is a path *reference* (a string) cannot be
    split this way: the pipeline lives in a separate YAML resolved relative to
    the trainrun file on the server, which only ``RestoreTrainRun`` knows how to
    locate and build. Use ``RestoreTrainRun`` for referenced trainruns (e.g.
    ``examples/grpc/core/restore_trainrun_grpc.py``); it is also the single-call
    equivalent of this helper for inline pipelines.
    """
    config = json.loads(config_bytes.decode("utf-8"))
    pipeline_section = config.pop("pipeline", None)
    data_section = config.get("data")
    if isinstance(pipeline_section, str):
        raise ValueError(
            "This trainrun references its pipeline by path "
            f"({pipeline_section!r}); apply_trainrun_config can only inline an "
            "embedded pipeline. Use RestoreTrainRun, which resolves the reference "
            "relative to the trainrun file and builds the pipeline server-side."
        )
    if pipeline_section is not None:
        load_request = cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=normalize_pipeline_bytes(json.dumps(pipeline_section).encode("utf-8"))
            ),
        )
        # Forward the data-module name so the server composes the per-run child
        # env with that module's plugin (e.g. cu3s -> cuvis-ai-dataloader).
        # Without it the compose runs with no data module and Train fails with
        # "no plugin provides data module '<name>'".
        data_module = (data_section or {}).get("data_module")
        if data_module:
            load_request.data_module = data_module
        stub.LoadPipeline(load_request)
    return stub.SetTrainRunConfig(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=cuvis_ai_pb2.TrainRunConfig(config_bytes=json.dumps(config).encode("utf-8")),
        )
    )


def format_progress(progress: cuvis_ai_pb2.TrainResponse) -> str:
    """Pretty-print training progress messages."""
    stage = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
    status = cuvis_ai_pb2.TrainStatus.Name(progress.status)

    parts = [f"[{stage}] {status}"]
    if progress.losses:
        parts.append(f"losses={dict(progress.losses)}")
    if progress.metrics:
        parts.append(f"metrics={dict(progress.metrics)}")
    if progress.message:
        parts.append(progress.message)

    return " | ".join(parts)


def load_manifest_bytes(path: Path) -> bytes:
    """Load one bare plugin manifest and return the JSON bytes for a LoadPlugin call.

    The file is a single bare manifest (``name`` + source + ``capabilities``).
    A local plugin's relative ``path`` is resolved to absolute against the
    manifest file's directory, since the server runs elsewhere and cannot
    resolve a client-relative path (``LoadPlugin`` rejects a relative local
    path). Git manifests (``repo`` + ``tag``) are returned unchanged.
    """
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(manifest, dict) and "repo" not in manifest:
        plugin_path = manifest.get("path")
        if isinstance(plugin_path, str) and plugin_path:
            resolved = Path(plugin_path)
            if not resolved.is_absolute():
                manifest["path"] = str((path.parent / resolved).resolve())
    return json.dumps(manifest).encode("utf-8")


def normalize_pipeline_bytes(config_bytes: bytes) -> bytes:
    """Unwrap Hydra group wrappers until a PipelineConfig payload with ``nodes`` is reached."""
    payload: Any = json.loads(config_bytes.decode("utf-8"))

    for _ in range(6):
        if isinstance(payload, dict) and "nodes" in payload:
            return json.dumps(payload).encode("utf-8")
        if isinstance(payload, dict) and len(payload) == 1:
            candidate = next(iter(payload.values()))
            if isinstance(candidate, dict):
                payload = candidate
                continue
        break

    raise ValueError(
        "Resolved pipeline config could not be normalized to a PipelineConfig payload."
    )


def resolve_pipeline_ref(ref: str, *, trainrun_dir: Path | None = None) -> dict:
    """Load a trainrun's path-referenced pipeline YAML into an inline dict.

    The bundled ``cuvis_ai/configs/trainrun/*.yaml`` reference their pipeline by a path
    relative to the trainrun file. Resolve it against ``trainrun_dir`` (default
    the package ``cuvis_ai/configs/trainrun``) and load the pipeline YAML directly. The
    bundled pipeline yamls are self-contained ``PipelineConfig``s (nodes +
    ``plugins:``), so a direct file load avoids the Hydra group-path wrapping a
    server-side ``ResolveConfig(config_type="pipeline")`` would add.
    """
    base = Path(trainrun_dir) if trainrun_dir is not None else (CONFIG_ROOT / "trainrun")
    pipeline_path = (base / ref).resolve()
    return yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))


def inline_pipeline_ref(config_dict: dict, *, trainrun_dir: Path | None = None) -> dict:
    """Return ``config_dict`` with a path-referenced ``pipeline`` inlined.

    No-op when ``pipeline`` is already an inline mapping or absent. Lets a
    resolved trainrun be handed to ``apply_trainrun_config`` even though the
    bundled trainruns reference their pipeline by path.
    """
    pipeline = config_dict.get("pipeline")
    if isinstance(pipeline, str):
        return {
            **config_dict,
            "pipeline": resolve_pipeline_ref(pipeline, trainrun_dir=trainrun_dir),
        }
    return config_dict


__all__ = [
    "CONFIG_ROOT",
    "apply_trainrun_config",
    "build_stub",
    "config_search_paths",
    "create_session_with_search_paths",
    "format_progress",
    "inline_pipeline_ref",
    "load_manifest_bytes",
    "normalize_pipeline_bytes",
    "resolve_pipeline_ref",
    "resolve_trainrun_config",
]
