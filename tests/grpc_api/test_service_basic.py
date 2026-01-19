from pathlib import Path

import grpc
import numpy as np
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from tests.fixtures.grpc import resolve_and_load_pipeline

DEFAULT_CHANNELS = 61


class TestCreateAndClose:
    def test_create_session_returns_id(self, grpc_stub):
        """Test creating a session with new four-step workflow."""
        # Step 1: Create empty session
        response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = response.session_id
        assert session_id

        # Step 2: Load pipeline using new API
        resolve_and_load_pipeline(grpc_stub, session_id)

    def test_create_session_with_weights(self, grpc_stub):
        """Test creating a session with pre-trained weights using four-step workflow."""
        # Step 1: Create empty session
        response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = response.session_id
        assert session_id

        # Step 2: Load pipeline, then weights explicitly
        resolve_and_load_pipeline(grpc_stub, session_id)
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=str(Path("configs/pipeline/channel_selector.pt").resolve()),
                strict=True,
            )
        )

    def test_create_session_invalid_pipeline(self, grpc_stub):
        """Test error handling for non-existent pipeline in four-step workflow."""
        # Step 1: Create empty session
        response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = response.session_id
        assert session_id

        # Step 2: Try to load invalid pipeline
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.ResolveConfig(
                cuvis_ai_pb2.ResolveConfigRequest(
                    session_id=session_id,
                    config_type="pipeline",
                    path="configs/pipeline/non_existent_pipeline.yaml",
                )
            )
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND

    def test_close_session_success(self, grpc_stub, session):
        """Test closing a session successfully using session fixture."""
        session_id = session(pipeline_type="channel_selector")
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

    def test_inference_missing_cube(self, grpc_stub, session):
        """Test inference with missing cube using session fixture."""
        session_id = session(pipeline_type="channel_selector")
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=session_id,
                    inputs=cuvis_ai_pb2.InputBatch(),
                )
            )
        assert exc.value.code() == grpc.StatusCode.INTERNAL
        assert "missing required inputs" in (exc.value.details() or "").lower()
