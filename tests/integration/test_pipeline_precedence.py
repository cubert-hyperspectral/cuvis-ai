import copy

import grpc
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.training.config import DataConfig, PipelineConfig, TrainingConfig, TrainRunConfig


def _build_pipeline(stub, session_id: str, pipeline_dict: dict) -> None:
    """Helper to build pipeline via RPC using LoadPipeline."""

    import json

    payload = copy.deepcopy(pipeline_dict)
    payload.pop("version", None)

    response = stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=json.dumps(payload).encode("utf-8")),
        )
    )
    assert response.success


def test_pipeline_conflict_rejected(grpc_stub, minimal_pipeline_dict, tmp_path):
    """TrainRun pipeline must match already-built pipeline."""
    session_id = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id
    _build_pipeline(grpc_stub, session_id, minimal_pipeline_dict)

    conflicting = copy.deepcopy(minimal_pipeline_dict)
    conflicting.pop("version", None)
    conflicting["metadata"]["name"] = "conflict"

    trainrun = TrainRunConfig(
        name="conflict",
        pipeline=PipelineConfig.from_dict(conflicting),
        data=DataConfig(cu3s_file_path="/tmp/dummy.cu3s"),
        training=TrainingConfig(),
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.SetTrainRunConfig(
            cuvis_ai_pb2.SetTrainRunConfigRequest(
                session_id=session_id,
                config=trainrun.to_proto(),
            )
        )

    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_pipeline_built_from_trainrun_when_missing(grpc_stub, minimal_pipeline_dict):
    """TrainRunConfig should build pipeline when session has none."""
    session_id = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id

    clean_pipeline = copy.deepcopy(minimal_pipeline_dict)
    clean_pipeline.pop("version", None)

    trainrun = TrainRunConfig(
        name="build_from_config",
        pipeline=PipelineConfig.from_dict(clean_pipeline),
        data=DataConfig(cu3s_file_path="/tmp/dummy.cu3s"),
        training=TrainingConfig(),
    )

    response = grpc_stub.SetTrainRunConfig(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=trainrun.to_proto(),
        )
    )

    assert response.success
    assert response.pipeline_from_config
