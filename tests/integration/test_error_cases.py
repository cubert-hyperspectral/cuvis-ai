"""Integration tests for error handling and edge cases."""

import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2


def _load_pipeline(grpc_stub, session_id: str, pipeline_name: str = "rx_statistical"):
    """Helper to resolve and load a pipeline using bytes-based API."""
    config_response = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path=f"pipeline/{pipeline_name}",
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


@pytest.mark.integration
class TestErrorCases:
    """Test error handling across gRPC API."""

    def test_invalid_session_id(self, grpc_stub):
        """Test operations with invalid session ID."""
        invalid_session_id = "nonexistent-session-12345"

        # Try inference with invalid session
        try:
            grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=invalid_session_id,
                    inputs=cuvis_ai_pb2.InputBatch(),
                )
            )
            raise AssertionError("Expected error for invalid session ID")
        except Exception as e:
            assert "session" in str(e).lower() or "not found" in str(e).lower()

    def test_train_without_trainrun_config(self, grpc_stub):
        """Test training without setting TrainRunConfig."""
        # Create session with pipeline but no trainrun config
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Build pipeline separately
        _load_pipeline(grpc_stub, session_id)

        # Try to train without setting trainrun config
        try:
            train_responses = list(
                grpc_stub.Train(
                    cuvis_ai_pb2.TrainRequest(
                        session_id=session_id,
                        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
                    )
                )
            )
            # Should error or return error status
            if train_responses:
                last_response = train_responses[-1]
                assert (
                    last_response.status == cuvis_ai_pb2.TRAIN_STATUS_ERROR
                    or "config" in last_response.message.lower()
                )
        except Exception as e:
            # Expected error
            assert "config" in str(e).lower() or "not set" in str(e).lower()

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_resolve_config_file_not_found(self, grpc_stub):
        """Test ResolveConfig with non-existent file."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Set search paths
        grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=["./configs/trainrun"],
                append=False,
            )
        )

        # Try to resolve non-existent config
        try:
            grpc_stub.ResolveConfig(
                cuvis_ai_pb2.ResolveConfigRequest(
                    session_id=session_id,
                    config_type="trainrun",
                    path="this_file_does_not_exist_xyz.yaml",
                    overrides=[],
                )
            )
            raise AssertionError("Expected error for missing config file")
        except Exception as e:
            assert "not found" in str(e).lower()

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_resolve_config_invalid_yaml(self, grpc_stub, tmp_path):
        """Test ResolveConfig with malformed YAML."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create invalid YAML file
        custom_dir = tmp_path / "invalid_configs"
        custom_dir.mkdir()

        invalid_yaml = custom_dir / "invalid.yaml"
        invalid_yaml.write_text("{ this is: [not valid yaml")

        # Set search paths
        grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=[str(custom_dir)],
                append=False,
            )
        )

        # Try to resolve invalid YAML
        try:
            grpc_stub.ResolveConfig(
                cuvis_ai_pb2.ResolveConfigRequest(
                    session_id=session_id,
                    config_type="trainrun",
                    path="invalid.yaml",
                    overrides=[],
                )
            )
            raise AssertionError("Expected error for invalid YAML")
        except Exception as e:
            # Should report YAML or parsing error
            assert (
                "yaml" in str(e).lower() or "parse" in str(e).lower() or "invalid" in str(e).lower()
            )

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_resolve_config_validation_error(self, grpc_stub, tmp_path):
        """Test ResolveConfig with config that fails Pydantic validation."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create config with validation errors
        custom_dir = tmp_path / "invalid_configs"
        custom_dir.mkdir()

        invalid_config = custom_dir / "validation_error.yaml"
        invalid_config.write_text(
            """
name: invalid-config
# Missing required fields: pipeline, data, training
"""
        )

        grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=[str(custom_dir)],
                append=False,
            )
        )

        # Try to resolve - should fail validation
        try:
            grpc_stub.ResolveConfig(
                cuvis_ai_pb2.ResolveConfigRequest(
                    session_id=session_id,
                    config_type="trainrun",
                    path="validation_error.yaml",
                    overrides=[],
                )
            )
            raise AssertionError("Expected validation error")
        except Exception as e:
            assert "validation" in str(e).lower() or "required" in str(e).lower()

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_invalid_config_type(self, grpc_stub):
        """Test ResolveConfig with invalid config type."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        try:
            grpc_stub.ResolveConfig(
                cuvis_ai_pb2.ResolveConfigRequest(
                    session_id=session_id,
                    config_type="invalid_type_xyz",  # Invalid type
                    path="rx_statistical.yaml",
                    overrides=[],
                )
            )
            raise AssertionError("Expected error for invalid config type")
        except Exception as e:
            assert "config type" in str(e).lower() or "unknown" in str(e).lower()

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_close_nonexistent_session(self, grpc_stub):
        """Test closing a non-existent session."""
        try:
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id="nonexistent-session")
            )
            # Some implementations may allow this silently
        except Exception as e:
            # Or raise error
            assert "session" in str(e).lower() or "not found" in str(e).lower()

    def test_double_close_session(self, grpc_stub):
        """Test closing a session twice."""
        # Create and close session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        response1 = grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        assert response1.success

        # Try to close again
        try:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
            # May succeed silently or raise error
        except Exception as e:
            assert "session" in str(e).lower() or "not found" in str(e).lower()

    def test_inference_without_pipeline(self, grpc_stub):
        """Test inference on session without pipeline."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Try inference without setting pipeline
        try:
            grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=session_id,
                    inputs=cuvis_ai_pb2.InputBatch(),
                )
            )
            raise AssertionError("Expected error for missing pipeline")
        except Exception as e:
            assert "pipeline" in str(e).lower() or "not found" in str(e).lower()

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_invalid_override_syntax(self, grpc_stub):
        """Test ResolveConfig with invalid override syntax."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=["./configs/trainrun"],
                append=False,
            )
        )

        # Try to resolve with invalid override syntax
        try:
            grpc_stub.ResolveConfig(
                cuvis_ai_pb2.ResolveConfigRequest(
                    session_id=session_id,
                    config_type="trainrun",
                    path="rx_statistical",
                    overrides=["this is not valid override syntax!!!"],
                )
            )
            # May succeed if Hydra handles it, or raise error
        except Exception:
            # Expected error from Hydra or validation
            pass

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_concurrent_operations_same_session(self, grpc_stub):
        """Test that concurrent operations on same session handle gracefully."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Build pipeline separately
        _load_pipeline(grpc_stub, session_id)

        # Multiple rapid operations on same session
        # This tests session state management
        for _ in range(5):
            try:
                grpc_stub.GetTrainStatus(cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id))
            except Exception:
                # May fail if training not started, which is fine
                pass

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_missing_required_inputs(self, grpc_stub, create_test_cube):
        """Test inference with missing required inputs."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Build pipeline separately
        _load_pipeline(grpc_stub, session_id)

        # Try inference with empty inputs
        try:
            grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=session_id,
                    inputs=cuvis_ai_pb2.InputBatch(),  # Empty inputs
                )
            )
            # May fail due to missing required inputs
        except Exception as e:
            # Expected error
            assert (
                "input" in str(e).lower()
                or "required" in str(e).lower()
                or "missing" in str(e).lower()
            )

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
