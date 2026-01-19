import json

import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2

pytest.importorskip("hydra")


def test_config_resolution_and_validation(grpc_stub, tmp_path):
    """Resolve, introspect, and validate configs via new RPCs."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    training_yaml = config_dir / "training.yaml"
    training_yaml.write_text(
        """seed: 7
max_epochs: 50
batch_size: 4
optimizer:
  name: adamw
  lr: 0.001
trainer:
  max_epochs: 50
"""
    )

    # Create empty session and point it at our temp configs
    session_id = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id
    paths_response = grpc_stub.SetSessionSearchPaths(
        cuvis_ai_pb2.SetSessionSearchPathsRequest(
            session_id=session_id,
            search_paths=[str(config_dir)],
            append=False,
        )
    )

    assert str(config_dir.resolve()) in paths_response.current_paths
    assert not paths_response.rejected_paths

    # Resolve and validate the training config
    resolved = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="training",
            path="training.yaml",
        )
    )

    config_payload = json.loads(resolved.config_bytes.decode("utf-8"))
    assert config_payload["optimizer"]["lr"] == 0.001
    assert config_payload["max_epochs"] == 50

    # Schema introspection
    schema_response = grpc_stub.GetParameterSchema(
        cuvis_ai_pb2.GetParameterSchemaRequest(config_type="training")
    )
    schema = json.loads(schema_response.json_schema)
    assert "optimizer" in schema.get("properties", {})

    # Validation success
    valid_response = grpc_stub.ValidateConfig(
        cuvis_ai_pb2.ValidateConfigRequest(
            config_type="training",
            config_bytes=resolved.config_bytes,
        )
    )
    assert valid_response.valid
    assert not valid_response.errors

    # Validation failure should surface errors
    invalid_response = grpc_stub.ValidateConfig(
        cuvis_ai_pb2.ValidateConfigRequest(
            config_type="training",
            config_bytes=b'{"max_epochs": 0}',
        )
    )
    assert not invalid_response.valid
    assert invalid_response.errors
