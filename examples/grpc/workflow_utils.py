"""Shared helpers for gRPC example clients using the Phase 5 workflow."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import grpc
import yaml
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


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
    """Apply resolved trainrun config to a session."""
    return stub.SetTrainRunConfig(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=cuvis_ai_pb2.TrainRunConfig(config_bytes=config_bytes),
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
    """Load a plugin YAML manifest, resolve relative plugin paths, and return JSON bytes."""
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    plugins = manifest.get("plugins", {}) if isinstance(manifest, dict) else {}
    for plugin_config in plugins.values():
        if not isinstance(plugin_config, dict):
            continue
        plugin_path = plugin_config.get("path")
        if isinstance(plugin_path, str) and plugin_path:
            resolved = Path(plugin_path)
            if not resolved.is_absolute():
                plugin_config["path"] = str((path.parent / resolved).resolve())
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


__all__ = [
    "CONFIG_ROOT",
    "apply_trainrun_config",
    "build_stub",
    "config_search_paths",
    "create_session_with_search_paths",
    "format_progress",
    "load_manifest_bytes",
    "normalize_pipeline_bytes",
    "resolve_trainrun_config",
]
