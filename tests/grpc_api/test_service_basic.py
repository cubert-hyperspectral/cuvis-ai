import json
from concurrent import futures

import grpc
import numpy as np
import pytest

from cuvis_ai.grpc import (
    CuvisAIService,
    cuvis_ai_pb2,
    cuvis_ai_pb2_grpc,
    helpers,
)


@pytest.fixture
def grpc_stub():
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


def _data_config() -> cuvis_ai_pb2.DataConfig:
    return cuvis_ai_pb2.DataConfig(
        cu3s_file_path="/tmp/data.cu3s",
        batch_size=1,
        processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
    )


def _create_session(stub, *, channels: int = 8) -> str:
    request = cuvis_ai_pb2.CreateSessionRequest(
        pipeline_type="channel_selector",
        pipeline_config=json.dumps({"input_channels": channels, "n_select": 3}),
        data_config=_data_config(),
    )
    response = stub.CreateSession(request)
    assert response.session_id  # Sanity (fails early if session not created)
    return response.session_id


class TestCreateAndClose:
    def test_create_session_returns_id(self, grpc_stub):
        response = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline_type="channel_selector",
                pipeline_config=json.dumps({"input_channels": 4}),
                data_config=_data_config(),
            )
        )
        assert response.session_id

    def test_create_session_without_data_config(self, grpc_stub):
        """Test creating an inference-only session without data_config."""
        response = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline_type="channel_selector",
                pipeline_config=json.dumps({"input_channels": 4}),
                # data_config intentionally omitted
            )
        )
        assert response.session_id

    def test_create_session_invalid_pipeline(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.CreateSession(
                cuvis_ai_pb2.CreateSessionRequest(
                    pipeline_type="unknown",
                    data_config=_data_config(),
                )
            )
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_close_session_success(self, grpc_stub):
        session_id = _create_session(grpc_stub)
        result = grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        assert result.success

    def test_close_session_not_found(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id="missing"))
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND


class TestInference:
    def test_inference_returns_outputs(self, grpc_stub):
        channels = 5
        session_id = _create_session(grpc_stub, channels=channels)
        cube = np.random.randn(1, 2, 2, channels).astype(np.float32)

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
            )
        )

        assert "selector.selected" in response.outputs
        selected = helpers.proto_to_numpy(response.outputs["selector.selected"])
        assert selected.shape == cube.shape

        # Expect deterministic key formatting (node.port)
        assert all("." in key for key in response.outputs)

    def test_inference_output_filtering(self, grpc_stub):
        channels = 4
        session_id = _create_session(grpc_stub, channels=channels)
        cube = np.random.randn(1, 3, 3, channels).astype(np.float32)

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
                output_specs=["selected"],
            )
        )

        assert set(response.outputs.keys()) == {"selector.selected"}

    def test_inference_invalid_session(self, grpc_stub):
        cube = np.random.randn(1, 1, 1, 3).astype(np.float32)
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id="invalid",
                    inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
                )
            )
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND

    def test_inference_missing_cube(self, grpc_stub):
        session_id = _create_session(grpc_stub, channels=3)
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=session_id,
                    inputs=cuvis_ai_pb2.InputBatch(),
                )
            )
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
