from pathlib import Path


def test_no_legacy_validate_training_config_rpc() -> None:
    """Ensure old ValidateTrainingConfig RPC is not reintroduced."""
    proto_path = Path("proto/cuvis_ai/grpc/v1/cuvis_ai.proto")
    proto_content = proto_path.read_text()

    assert "ValidateTrainingConfigRequest" not in proto_content
    assert "ValidateTrainingConfigResponse" not in proto_content
    assert "rpc ValidateTrainingConfig" not in proto_content
