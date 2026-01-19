"""gRPC testing fixtures."""

import json
import logging
from collections.abc import Generator
from concurrent import futures
from pathlib import Path

import grpc
import pytest
import yaml

from cuvis_ai_core.grpc import CuvisAIService, cuvis_ai_pb2, cuvis_ai_pb2_grpc
from cuvis_ai_core.grpc.session_manager import SessionManager

# Keep a handle on the service so tests can introspect the live SessionManager
SERVICE_INSTANCE: CuvisAIService | None = None


# Session-scoped gRPC server to avoid repeated startup/shutdown overhead
@pytest.fixture(scope="session")
def grpc_server() -> Generator[str, None, None]:
    """Session-scoped gRPC server fixture.

    Creates a single gRPC server that is shared across all tests in the session,
    significantly reducing startup/shutdown overhead.

    Yields:
        str: The server address (e.g., "localhost:port")
    """
    # Configure logging for gRPC server
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    # Create server with increased thread pool for better parallelism
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    global SERVICE_INSTANCE
    SERVICE_INSTANCE = CuvisAIService()
    service = SERVICE_INSTANCE
    cuvis_ai_pb2_grpc.add_CuvisAIServiceServicer_to_server(service, server)

    # Bind to available port
    port = server.add_insecure_port("localhost:0")
    if port == 0:
        raise RuntimeError("Failed to bind gRPC server to port")

    logger.info(f"Starting session-scoped gRPC server on port {port}")
    server.start()

    try:
        yield f"localhost:{port}"
    finally:
        logger.info("Stopping session-scoped gRPC server")
        server.stop(grace=None)
        SERVICE_INSTANCE = None


@pytest.fixture
def grpc_stub(grpc_server: str) -> Generator:
    """Create gRPC client stub using shared session server.

    Uses the session-scoped gRPC server to avoid creating new servers for each test.

    Args:
        grpc_server: Session-scoped server address from grpc_server fixture

    Yields:
        CuvisAIServiceStub: gRPC client stub
    """
    channel = grpc.insecure_channel(grpc_server)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        yield stub
    finally:
        channel.close()


@pytest.fixture(scope="session")
def grpc_session_manager(grpc_server: str) -> SessionManager:
    """Expose the live SessionManager used by the in-process gRPC server."""
    del grpc_server  # server is already started by the fixture dependency
    if SERVICE_INSTANCE is None:
        raise RuntimeError("CuvisAIService instance is not initialized")
    return SERVICE_INSTANCE.session_manager


# ------------------------------------------------------------------
# Shared gRPC Test Helper Functions
# ------------------------------------------------------------------


def resolve_and_load_pipeline(
    grpc_stub, session_id: str, path: str = "pipeline/channel_selector"
) -> cuvis_ai_pb2.LoadPipelineResponse:
    """Resolve and load pipeline structure via bytes-based API.

    This is a shared test helper to avoid duplicate implementations across test files.

    Args:
        grpc_stub: gRPC stub for making API calls
        session_id: Session ID to load pipeline into
        path: Pipeline path (e.g., "pipeline/channel_selector")

    Returns:
        LoadPipelineResponse from the gRPC call

    Raises:
        AssertionError: If the pipeline loading fails
    """
    config_response = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path=path,
        )
    )
    response = grpc_stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=config_response.config_bytes),
        )
    )
    assert response.success
    return response


def load_pipeline_from_file(
    grpc_stub, session_id: str, pipeline_file: str | Path
) -> cuvis_ai_pb2.LoadPipelineResponse:
    """Load a pipeline from a saved YAML by converting to JSON bytes.

    This is a shared test helper to avoid duplicate implementations across test files.

    Args:
        grpc_stub: gRPC stub for making API calls
        session_id: Session ID to load pipeline into
        pipeline_file: Path to pipeline YAML file

    Returns:
        LoadPipelineResponse from the gRPC call

    Raises:
        AssertionError: If the pipeline loading fails
    """
    pipeline_dict = yaml.safe_load(Path(pipeline_file).read_text())
    pipeline_bytes = json.dumps(pipeline_dict).encode("utf-8")
    response = grpc_stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_bytes),
        )
    )
    assert response.success
    return response
