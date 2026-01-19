import copy
import json
from pathlib import Path

import grpc
import pytest
import yaml

from cuvis_ai_core.grpc import cuvis_ai_pb2

pytest.importorskip("hydra")


def _write_pipeline_file(path: Path, pipeline_dict: dict) -> None:
    path.write_text(yaml.safe_dump(pipeline_dict, sort_keys=False))


def _write_trainrun_file(path: Path, trainrun_dict: dict) -> None:
    path.write_text(yaml.safe_dump(trainrun_dict, sort_keys=False))


def test_complete_four_step_flow(grpc_stub, minimal_pipeline_dict, tmp_path):
    """End-to-end happy path for the new explicit workflow."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    pipeline_definition = copy.deepcopy(minimal_pipeline_dict)
    pipeline_definition.pop("version", None)

    pipeline_path = config_dir / "pipeline.yaml"
    _write_pipeline_file(pipeline_path, pipeline_definition)

    trainrun_path = config_dir / "trainrun.yaml"
    trainrun_dict = {
        "name": "integration-trainrun",
        "pipeline": pipeline_definition,
        "data": {
            "cu3s_file_path": "/tmp/dummy.cu3s",
            "batch_size": 2,
            "processing_mode": "Reflectance",
        },
        "training": {
            "max_epochs": 2,
            "optimizer": {"name": "adamw", "lr": 0.001},
            "trainer": {"max_epochs": 2},
        },
        "loss_nodes": [],
        "metric_nodes": [],
    }
    _write_trainrun_file(trainrun_path, trainrun_dict)

    # Step 1: Create empty session and set search paths
    session_id = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id
    path_resp = grpc_stub.SetSessionSearchPaths(
        cuvis_ai_pb2.SetSessionSearchPathsRequest(
            session_id=session_id,
            search_paths=[str(config_dir)],
            append=False,
        )
    )
    assert path_resp.success

    # Step 2: Resolve + load pipeline
    pipeline_resolved = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path="pipeline.yaml",
        )
    )
    load_response = grpc_stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_resolved.config_bytes),
        )
    )
    assert load_response.success

    # Step 3: Resolve trainrun + set on session (pipeline precedence logic exercised)
    trainrun_resolved = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="trainrun",
            path="trainrun.yaml",
        )
    )
    trainrun_config_json = json.loads(trainrun_resolved.config_bytes.decode("utf-8"))
    assert trainrun_config_json["name"] == "integration-trainrun"
    assert trainrun_config_json["data"]["cu3s_file_path"] == "/tmp/dummy.cu3s"
    trainrun_config_json.pop("pipeline", None)

    set_trainrun_response = grpc_stub.SetTrainRunConfig(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=cuvis_ai_pb2.TrainRunConfig(
                config_bytes=json.dumps(trainrun_config_json).encode("utf-8")
            ),
        )
    )
    assert set_trainrun_response.success

    # Step 4: Training call should now surface missing data files rather than missing prerequisites
    train_request = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
    with pytest.raises(grpc.RpcError) as exc_info:
        # Data path is intentionally dummy; we only assert prerequisites are satisfied.
        list(grpc_stub.Train(train_request))

    assert exc_info.value.code() in {
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.INTERNAL,
    }
