from pathlib import Path

import grpc
import numpy as np
import pytest

from cuvis_ai.grpc import cuvis_ai_pb2, helpers
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


def _data_files(test_data_path: Path) -> tuple[Path, Path]:
    cu3s_file = test_data_path / "Lentils" / "Lentils_000.cu3s"
    json_file = test_data_path / "Lentils" / "Lentils_000.json"
    if not cu3s_file.exists() or not json_file.exists():
        pytest.skip(f"Test data not found under {test_data_path}")
    return cu3s_file, json_file


class TestCompleteWorkflow:
    """End-to-end integration flows for the gRPC service."""

    def test_full_statistical_workflow(self, grpc_stub, test_data_path, mock_cuvis_sdk):
        cu3s_file, json_file = _data_files(test_data_path)

        # 1. Create session
        session_resp = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline_type="statistical",
                data_config=cuvis_ai_pb2.DataConfig(
                    cu3s_file_path=str(cu3s_file),
                    annotation_json_path=str(json_file),
                    train_ids=[0, 1, 2],
                    val_ids=[3, 4],
                    test_ids=[5, 6],
                    batch_size=2,
                    processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
                ),
            )
        )
        session_id = session_resp.session_id

        # 2. Introspect pipeline
        inputs = grpc_stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
        )
        outputs = grpc_stub.GetPipelineOutputs(
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
        )
        assert inputs.input_names
        assert outputs.output_names

        # 3. Train (statistical)
        statuses = {
            update.status
            for update in grpc_stub.Train(
                cuvis_ai_pb2.TrainRequest(
                    session_id=session_id,
                    trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
                )
            )
        }
        assert cuvis_ai_pb2.TRAIN_STATUS_RUNNING in statuses
        assert cuvis_ai_pb2.TRAIN_STATUS_COMPLETE in statuses

        # 4. Inference
        cube = np.random.rand(1, 16, 16, 61).astype(np.float32)
        inference_resp = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
            )
        )
        assert inference_resp.outputs

        # 5. Cleanup
        close_resp = grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        assert close_resp.success

    def test_full_gradient_workflow(self, grpc_stub, test_data_path, mock_cuvis_sdk, tmp_path):
        cu3s_file, json_file = _data_files(test_data_path)

        # 1. Create session
        session_resp = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline_type="gradient",
                data_config=cuvis_ai_pb2.DataConfig(
                    cu3s_file_path=str(cu3s_file),
                    annotation_json_path=str(json_file),
                    train_ids=[0, 1, 2],
                    val_ids=[3, 4],
                    batch_size=2,
                    processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
                ),
            )
        )
        session_id = session_resp.session_id

        # 2. Statistical warmup
        for _ in grpc_stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        ):
            pass

        # 3. Validate config before gradient training
        config = TrainingConfig(
            trainer=TrainerConfig(max_epochs=2, accelerator="cpu"),
            optimizer=OptimizerConfig(name="adam", lr=0.001),
        )
        validation = grpc_stub.ValidateTrainingConfig(
            cuvis_ai_pb2.ValidateTrainingConfigRequest(
                config=cuvis_ai_pb2.TrainingConfig(config_json=config.to_json().encode())
            )
        )
        assert validation.valid
        assert not validation.errors

        # 4. Gradient training with progress monitoring
        progress_updates = list(
            grpc_stub.Train(
                cuvis_ai_pb2.TrainRequest(
                    session_id=session_id,
                    trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
                    config=cuvis_ai_pb2.TrainingConfig(config_json=config.to_json().encode()),
                )
            )
        )
        assert progress_updates
        assert progress_updates[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

        # 5. Save checkpoint
        checkpoint_path = tmp_path / "model.ckpt"
        save_resp = grpc_stub.SaveCheckpoint(
            cuvis_ai_pb2.SaveCheckpointRequest(
                session_id=session_id,
                checkpoint_path=str(checkpoint_path),
            )
        )
        assert save_resp.success
        assert checkpoint_path.exists()

        # 6. Inference
        cube = np.random.rand(1, 16, 16, 61).astype(np.float32)
        inference_resp = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
                output_specs=["selector.selected", "anomaly_metrics"],
            )
        )
        assert inference_resp.outputs or inference_resp.metrics

        # 7. Cleanup
        close_resp = grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        assert close_resp.success

    def test_multiple_sessions(self, grpc_stub, test_data_path, mock_cuvis_sdk):
        cu3s_file, _ = _data_files(test_data_path)
        session_ids: list[str] = []

        for _ in range(3):
            resp = grpc_stub.CreateSession(
                cuvis_ai_pb2.CreateSessionRequest(
                    pipeline_type="statistical",
                    data_config=cuvis_ai_pb2.DataConfig(
                        cu3s_file_path=str(cu3s_file),
                        batch_size=2,
                        processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
                    ),
                )
            )
            session_ids.append(resp.session_id)

        for session_id in session_ids:
            inputs = grpc_stub.GetPipelineInputs(
                cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
            )
            assert inputs.input_names

        for session_id in session_ids:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_error_recovery(self, grpc_stub, mock_cuvis_sdk):
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.GetPipelineInputs(
                cuvis_ai_pb2.GetPipelineInputsRequest(session_id="invalid-session")
            )
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

        session_resp = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline_type="statistical",
                data_config=cuvis_ai_pb2.DataConfig(
                    cu3s_file_path="/tmp/fake.cu3s",
                    batch_size=1,
                    processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
                ),
            )
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=session_resp.session_id,
                    inputs=cuvis_ai_pb2.InputBatch(),  # Missing cube
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_resp.session_id))
