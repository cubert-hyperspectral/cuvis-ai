import grpc
import numpy as np
import pytest

from cuvis_ai.grpc import cuvis_ai_pb2, helpers
from tests.fixtures import create_pipeline_config_proto

DEFAULT_CHANNELS = 61


def _data_config() -> cuvis_ai_pb2.DataConfig:
    return cuvis_ai_pb2.DataConfig(
        cu3s_file_path="/tmp/data.cu3s",
        batch_size=1,
        processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
    )


def _create_session(stub, *, pipeline_path: str = "channel_selector") -> str:
    """Create a session using the new PipelineConfig pattern."""
    request = cuvis_ai_pb2.CreateSessionRequest(
        pipeline=create_pipeline_config_proto(pipeline_path)
    )
    response = stub.CreateSession(request)
    assert response.session_id  # Sanity (fails early if session not created)
    return response.session_id


class TestCreateAndClose:
    def test_create_session_returns_id(self, grpc_stub):
        """Test creating a session with new PipelineConfig pattern."""
        response = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline=create_pipeline_config_proto("channel_selector")
            )
        )
        assert response.session_id

    def test_create_session_with_weights(self, grpc_stub):
        """Test creating a session with pre-trained weights."""
        # Weights are automatically loaded if they exist alongside the YAML
        response = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline=create_pipeline_config_proto("channel_selector")
            )
        )
        assert response.session_id

    def test_create_session_invalid_pipeline(self, grpc_stub):
        """Test error handling for non-existent pipeline."""
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.CreateSession(
                cuvis_ai_pb2.CreateSessionRequest(
                    pipeline=create_pipeline_config_proto("non_existent_pipeline")
                )
            )
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND

    def test_close_session_success(self, grpc_stub):
        session_id = _create_session(grpc_stub)
        result = grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        assert result.success

    def test_close_session_not_found(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id="missing"))
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND


class TestInference:
    def test_inference_returns_outputs(self, grpc_stub, create_test_cube, trained_pipeline_session):
        # Use fixture to generate cube and wavelengths together
        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=2,
            width=2,
            num_channels=DEFAULT_CHANNELS,
            mode="random",
        )
        # Convert wavelengths to 2D format [B, C] as required by LentilsAnomalyDataNode
        wavelengths_2d = np.tile(wavelengths, (cube.shape[0], 1)).astype(np.int32)

        session_id = trained_pipeline_session(pipeline_path="channel_selector")

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.numpy_to_proto(cube.numpy()),
                    wavelengths=helpers.numpy_to_proto(wavelengths_2d),
                ),
            )
        )

        selected_key = "SoftChannelSelector.selected"

        assert selected_key in response.outputs
        selected = helpers.proto_to_numpy(response.outputs[selected_key])
        assert selected.shape == cube.shape

        # Expect deterministic key formatting (node.port)
        assert all("." in key for key in response.outputs)

    def test_inference_output_filtering(
        self, grpc_stub, create_test_cube, trained_pipeline_session
    ):
        # Use fixture to generate cube and wavelengths together
        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=3,
            width=3,
            num_channels=DEFAULT_CHANNELS,
            mode="random",
        )
        # Convert wavelengths to 2D format [B, C] as required by LentilsAnomalyDataNode
        wavelengths_2d = np.tile(wavelengths, (cube.shape[0], 1)).astype(np.int32)

        session_id = trained_pipeline_session(pipeline_path="channel_selector")

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.numpy_to_proto(cube.numpy()),
                    wavelengths=helpers.numpy_to_proto(wavelengths_2d),
                ),
                output_specs=["selected"],
            )
        )

        assert set(response.outputs.keys()) == {"SoftChannelSelector.selected"}

    def test_inference_invalid_session(self, grpc_stub, create_test_cube):
        cube, wavelengths = create_test_cube(
            batch_size=1, height=2, width=2, num_channels=DEFAULT_CHANNELS, mode="random"
        )

        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id="invalid",
                    inputs=cuvis_ai_pb2.InputBatch(
                        cube=helpers.tensor_to_proto(cube),
                        wavelengths=helpers.tensor_to_proto(wavelengths),
                    ),
                )
            )
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND

    def test_inference_missing_cube(self, grpc_stub):
        session_id = _create_session(grpc_stub)
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=session_id,
                    inputs=cuvis_ai_pb2.InputBatch(),
                )
            )
        assert exc.value.code() == grpc.StatusCode.INTERNAL
        assert "missing required inputs" in (exc.value.details() or "").lower()
