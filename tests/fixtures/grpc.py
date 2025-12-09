"""gRPC testing fixtures."""

from concurrent import futures

import grpc
import pytest

from cuvis_ai.grpc import CuvisAIService, cuvis_ai_pb2_grpc


@pytest.fixture
def grpc_stub():
    """Create in-process gRPC client stub."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    service = CuvisAIService()
    cuvis_ai_pb2_grpc.add_CuvisAIServiceServicer_to_server(service, server)
    port = server.add_insecure_port("localhost:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
    try:
        yield stub
    finally:
        channel.close()
        server.stop(None)
