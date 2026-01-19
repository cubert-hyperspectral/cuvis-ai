import grpc
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2


class TestGetPipelineInputs:
    def test_get_pipeline_inputs(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
        )

        assert response.input_names
        assert "cube" in response.input_names
        assert response.input_specs

    def test_input_specs_have_details(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
        )

        cube_spec = response.input_specs["cube"]
        assert cube_spec.name == "cube"
        assert cube_spec.shape
        assert cube_spec.dtype != cuvis_ai_pb2.D_TYPE_UNSPECIFIED

    def test_invalid_session(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.GetPipelineInputs(cuvis_ai_pb2.GetPipelineInputsRequest(session_id="invalid"))

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


class TestGetPipelineOutputs:
    def test_get_pipeline_outputs(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineOutputs(
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
        )

        assert response.output_names
        assert response.output_specs

    def test_output_specs_have_details(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineOutputs(
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
        )

        first_output = next(iter(response.output_specs.values()))
        assert first_output.name
        assert first_output.dtype != cuvis_ai_pb2.D_TYPE_UNSPECIFIED


class TestGetPipelineVisualization:
    def test_get_visualization_png(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineVisualization(
            cuvis_ai_pb2.GetPipelineVisualizationRequest(session_id=session_id, format="png")
        )

        assert response.image_data
        assert response.format == "png"

    def test_get_visualization_svg(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineVisualization(
            cuvis_ai_pb2.GetPipelineVisualizationRequest(session_id=session_id, format="svg")
        )

        assert response.image_data
        assert response.format == "svg"

    def test_default_format_png(self, grpc_stub, session):
        session_id = session()
        response = grpc_stub.GetPipelineVisualization(
            cuvis_ai_pb2.GetPipelineVisualizationRequest(session_id=session_id)
        )

        assert response.image_data
        assert response.format == "png"
