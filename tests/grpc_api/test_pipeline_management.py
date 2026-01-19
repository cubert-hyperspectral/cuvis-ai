"""Integration tests for pipeline management functionality (Task 5.3)."""

import json
from pathlib import Path

import grpc
import pytest
import yaml

from cuvis_ai_core.grpc import cuvis_ai_pb2

DEFAULT_CHANNELS = 61


def _pipeline_bytes_from_path(pipeline_path: str | Path) -> bytes:
    """Convert a pipeline YAML into JSON bytes for the LoadPipeline RPC."""
    pipeline_dict = yaml.safe_load(Path(pipeline_path).read_text())
    return json.dumps(pipeline_dict).encode("utf-8")


class TestSavePipeline:
    """Test the SavePipeline RPC method."""

    def test_save_pipeline_creates_yaml_and_pt(self, grpc_stub, session, tmp_path, monkeypatch):
        """Test that SavePipeline creates both .yaml and .pt files."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(tmp_path))
        session_id = session()

        pipeline_path = str(tmp_path / "saved_model.yaml")

        response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=pipeline_path,
            )
        )

        assert response.success
        assert response.pipeline_path
        assert response.weights_path

        # Verify files exist
        assert Path(response.pipeline_path).exists()
        assert Path(response.weights_path).exists()

        # Verify .pt file is co-located with .yaml
        assert Path(response.weights_path).parent == Path(response.pipeline_path).parent
        assert Path(response.weights_path).stem == Path(response.pipeline_path).stem

    def test_save_pipeline_with_metadata(self, grpc_stub, session, tmp_path, monkeypatch):
        """Test that metadata is correctly saved."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(tmp_path))
        session_id = session()

        pipeline_path = str(tmp_path / "model_with_metadata.yaml")

        metadata = cuvis_ai_pb2.PipelineMetadata(
            name="Test Model",
            description="A model for testing",
            created="2024-11-27",
            cuvis_ai_version="0.1.5",
            tags=["test", "preprocessing"],
            author="Test Author",
        )

        response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=pipeline_path,
                metadata=metadata,
            )
        )

        assert response.success

        # Read back the YAML to verify metadata was saved
        import yaml

        with open(response.pipeline_path) as f:
            saved_config = yaml.safe_load(f)

        assert saved_config["metadata"]["name"] == "Test Model"
        assert saved_config["metadata"]["description"] == "A model for testing"
        assert saved_config["metadata"]["tags"] == ["test", "preprocessing"]
        assert saved_config["metadata"]["author"] == "Test Author"

    def test_save_pipeline_invalid_session(self, grpc_stub, tmp_path, monkeypatch):
        """Test error handling for invalid session ID."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(tmp_path))

        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.SavePipeline(
                cuvis_ai_pb2.SavePipelineRequest(
                    session_id="invalid_session_id",
                    pipeline_path=str(tmp_path / "should_not_exist.yaml"),
                )
            )
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND

    def test_save_pipeline_creates_directories(self, grpc_stub, session, tmp_path, monkeypatch):
        """Test that SavePipeline creates parent directories if needed."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(tmp_path))
        session_id = session()

        # Use a nested path that doesn't exist yet
        nested_path = tmp_path / "models" / "v1" / "saved_pipeline.yaml"

        response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=str(nested_path),
            )
        )

        assert response.success
        assert Path(response.pipeline_path).exists()
        assert Path(response.weights_path).exists()


class TestLoadPipeline:
    """Test the LoadPipeline RPC method."""

    # @pytest.mark.skip(
    #     reason="Native cuvis library has thread-safety issues causing crashes during weight loading"
    # )
    def test_load_pipeline_with_weights_default(
        self, grpc_stub, session, saved_pipeline, monkeypatch
    ):
        """Test LoadPipeline loads weights when weights_path is provided."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(Path(saved_pipeline["pipeline_path"]).parent))
        session_id = session()

        response = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(
                    config_bytes=_pipeline_bytes_from_path(saved_pipeline["pipeline_path"])
                ),
            )
        )
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=saved_pipeline["weights_path"],
                strict=True,
            )
        )

        assert response.success
        assert response.metadata.name == "Test Pipeline"
        assert response.metadata.description == "Pipeline for testing"

    def test_load_pipeline_without_weights(
        self, grpc_stub, session, pipeline_yaml_only, monkeypatch
    ):
        """Test LoadPipeline without providing weights_path loads structure only."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(Path(pipeline_yaml_only).parent))
        session_id = session()

        response = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(
                    config_bytes=_pipeline_bytes_from_path(pipeline_yaml_only)
                ),
            )
        )

        assert response.success
        assert response.metadata.name == "YAML Only Pipeline"

    def test_load_pipeline_missing_pt_error(
        self, grpc_stub, session, pipeline_yaml_only, monkeypatch
    ):
        """Test error when an explicit weights_path does not exist."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(Path(pipeline_yaml_only).parent))
        session_id = session()
        missing_weights = str(Path(pipeline_yaml_only).with_suffix(".pt"))

        grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(
                    config_bytes=_pipeline_bytes_from_path(pipeline_yaml_only)
                ),
            )
        )

        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.LoadPipelineWeights(
                cuvis_ai_pb2.LoadPipelineWeightsRequest(
                    session_id=session_id,
                    weights_path=missing_weights,
                )
            )
        assert exc.value.code() in [grpc.StatusCode.NOT_FOUND, grpc.StatusCode.INVALID_ARGUMENT]

    def test_load_pipeline_strict_mode(self, grpc_stub, session, saved_pipeline, monkeypatch):
        """Test strict weight loading mode."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(Path(saved_pipeline["pipeline_path"]).parent))
        session_id = session()

        response = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(
                    config_bytes=_pipeline_bytes_from_path(saved_pipeline["pipeline_path"])
                ),
            )
        )
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=saved_pipeline["weights_path"],
                strict=True,
            )
        )

        assert response.success

    def test_load_pipeline_non_strict_mode(self, grpc_stub, session, saved_pipeline, monkeypatch):
        """Test non-strict weight loading allows missing keys."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(Path(saved_pipeline["pipeline_path"]).parent))
        session_id = session()

        response = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(
                    config_bytes=_pipeline_bytes_from_path(saved_pipeline["pipeline_path"])
                ),
            )
        )
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=saved_pipeline["weights_path"],
                strict=False,
            )
        )

        assert response.success

    def test_load_pipeline_updates_session(self, grpc_stub, session, saved_pipeline, monkeypatch):
        """Test that LoadPipeline properly updates the session pipeline."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(Path(saved_pipeline["pipeline_path"]).parent))
        session_id = session()

        # Load the pipeline
        load_response = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(
                    config_bytes=_pipeline_bytes_from_path(saved_pipeline["pipeline_path"])
                ),
            )
        )

        assert load_response.success
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=saved_pipeline["weights_path"],
                strict=True,
            )
        )

        # Verify the session reports pipeline inputs after loading the pipeline
        pipeline_response = grpc_stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
        )
        assert len(pipeline_response.input_names) > 0


class TestPipelineRoundTrip:
    """Test complete save/load cycles."""

    def test_save_load_preserves_structure(self, grpc_stub, session, tmp_path, monkeypatch):
        """Test that saving and loading preserves pipeline structure."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(tmp_path))
        session_id = session()

        # Save the pipeline
        pipeline_path = str(tmp_path / "roundtrip.yaml")
        save_response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=pipeline_path,
                metadata=cuvis_ai_pb2.PipelineMetadata(
                    name="Roundtrip Test",
                    description="Testing save/load cycle",
                ),
            )
        )

        assert save_response.success

        # Create a new session and load the saved pipeline using the new four-step workflow
        new_session_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        new_session_id = new_session_response.session_id
        assert new_session_id  # Verify session was created successfully

        # Load the saved pipeline
        load_response = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=new_session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(
                    config_bytes=_pipeline_bytes_from_path(save_response.pipeline_path)
                ),
            )
        )
        assert load_response.success
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=new_session_id,
                weights_path=save_response.weights_path,
                strict=True,
            )
        )

        # Verify the pipeline is usable by inspecting its expected inputs
        pipeline_response = grpc_stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=new_session_id)
        )
        assert len(pipeline_response.input_names) > 0  # Pipeline structure preserved

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=new_session_id))
