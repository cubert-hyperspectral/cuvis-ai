"""Integration tests for custom search path configuration."""

import json

import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2


@pytest.mark.integration
class TestSearchPaths:
    """Test custom search path configuration through gRPC API."""

    def test_set_session_search_paths(self, grpc_stub, tmp_path):
        """Test setting custom search paths for a session."""
        # Create session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create custom config directories
        custom_config_dir = tmp_path / "custom_configs"
        custom_config_dir.mkdir()

        # Set custom search paths
        response = grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=[
                    str(custom_config_dir),
                    "./configs/trainrun",
                    "./configs/pipeline",
                ],
                append=False,  # Replace existing paths
            )
        )

        assert response.success
        assert len(response.current_paths) == 3
        assert str(custom_config_dir) in response.current_paths

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_append_search_paths(self, grpc_stub, tmp_path):
        """Test appending to existing search paths."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Set initial search paths
        initial_paths = ["./configs/trainrun"]
        grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=initial_paths,
                append=False,
            )
        )

        # Append additional paths
        additional_dir = tmp_path / "additional"
        additional_dir.mkdir()

        response = grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=[str(additional_dir)],
                append=True,  # Append to existing
            )
        )

        assert response.success
        assert len(response.current_paths) >= 2
        assert str(additional_dir) in response.current_paths

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_resolve_config_from_custom_path(self, grpc_stub, tmp_path):
        """Test resolving config from custom search path."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create custom trainrun config in custom directory
        custom_dir = tmp_path / "my_configs"
        custom_dir.mkdir()

        custom_trainrun = custom_dir / "custom_trainrun.yaml"
        custom_trainrun.write_text(
            """
name: custom-trainrun
pipeline:
  metadata:
    name: custom-pipeline
  nodes: []
  connections: []
data:
  cu3s_file_path: /tmp/data.cu3s
  batch_size: 2
training:
  optimizer:
    name: adamw
    lr: 0.001
  trainer:
    max_epochs: 1
  batch_size: 2
  max_epochs: 1
metric_nodes: []
loss_nodes: []
"""
        )

        # Set search paths to include custom directory
        grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=[str(custom_dir)],
                append=False,
            )
        )

        # Resolve config from custom path
        response = grpc_stub.ResolveConfig(
            cuvis_ai_pb2.ResolveConfigRequest(
                session_id=session_id,
                config_type="trainrun",
                path="custom_trainrun.yaml",
                overrides=[],
            )
        )

        assert response.config_bytes
        config_dict = json.loads(response.config_bytes.decode("utf-8"))
        assert config_dict["name"] == "custom-trainrun"

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_search_path_precedence(self, grpc_stub, tmp_path):
        """Test that earlier search paths have precedence."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create two directories with same-named config
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # Same filename, different content
        config1 = dir1 / "test.yaml"
        config1.write_text(
            """
name: from-dir1
pipeline:
  metadata:
    name: pipeline1
  nodes: []
  connections: []
data:
  cu3s_file_path: /tmp/data.cu3s
  batch_size: 1
training:
  optimizer:
    name: adamw
    lr: 0.001
  trainer:
    max_epochs: 1
  batch_size: 1
  max_epochs: 1
metric_nodes: []
loss_nodes: []
"""
        )

        config2 = dir2 / "test.yaml"
        config2.write_text(
            """
name: from-dir2
pipeline:
  metadata:
    name: pipeline2
  nodes: []
  connections: []
data:
  cu3s_file_path: /tmp/data.cu3s
  batch_size: 1
training:
  optimizer:
    name: adamw
    lr: 0.001
  trainer:
    max_epochs: 1
  batch_size: 1
  max_epochs: 1
metric_nodes: []
loss_nodes: []
"""
        )

        # Set search paths with dir1 first
        grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=[str(dir1), str(dir2)],
                append=False,
            )
        )

        # Resolve - should get from dir1 (first in search path)
        response = grpc_stub.ResolveConfig(
            cuvis_ai_pb2.ResolveConfigRequest(
                session_id=session_id,
                config_type="trainrun",
                path="test.yaml",
                overrides=[],
            )
        )

        config_dict = json.loads(response.config_bytes.decode("utf-8"))
        assert config_dict["name"] == "from-dir1"

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_search_paths_not_found(self, grpc_stub):
        """Test error when config not found in any search path."""
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
                    path="nonexistent_config.yaml",
                    overrides=[],
                )
            )
            # Should raise error or return error status
            raise AssertionError("Expected error for missing config")
        except Exception as e:
            # Expected error
            assert "not found" in str(e).lower()

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_search_paths_persist_across_operations(self, grpc_stub, tmp_path):
        """Test that search paths persist across multiple operations in a session."""
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Create custom config directory
        custom_dir = tmp_path / "persistent"
        custom_dir.mkdir()

        config_file = custom_dir / "persistent.yaml"
        config_file.write_text(
            """
name: persistent-config
pipeline:
  metadata:
    name: persistent-pipeline
  nodes: []
  connections: []
data:
  cu3s_file_path: /tmp/data.cu3s
  batch_size: 1
training:
  optimizer:
    name: adamw
    lr: 0.001
  trainer:
    max_epochs: 1
  batch_size: 1
  max_epochs: 1
metric_nodes: []
loss_nodes: []
"""
        )

        # Set search paths once
        grpc_stub.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=[str(custom_dir)],
                append=False,
            )
        )

        # Resolve config multiple times - should work each time
        for i in range(3):
            response = grpc_stub.ResolveConfig(
                cuvis_ai_pb2.ResolveConfigRequest(
                    session_id=session_id,
                    config_type="trainrun",
                    path="persistent.yaml",
                    overrides=[f"data.batch_size={i + 1}"],
                )
            )

            config_dict = json.loads(response.config_bytes.decode("utf-8"))
            assert config_dict["name"] == "persistent-config"
            assert config_dict["data"]["batch_size"] == i + 1

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
