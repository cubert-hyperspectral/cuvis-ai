from pathlib import Path

import grpc
import numpy as np
import pytest
import yaml

from cuvis_ai.grpc import cuvis_ai_pb2, helpers
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig
from tests.fixtures import create_pipeline_config_proto


class TestCompleteWorkflow:
    """End-to-end integration flows for the gRPC service."""

    def test_full_statistical_workflow(self, grpc_stub, test_data_files, mock_cuvis_sdk):
        cu3s_file, json_file = test_data_files

        # 1. Create session
        data_config = cuvis_ai_pb2.DataConfig(
            cu3s_file_path=str(cu3s_file),
            annotation_json_path=str(json_file),
            train_ids=[0, 1, 2],
            val_ids=[3, 4],
            test_ids=[5, 6],
            batch_size=2,
            processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
        )
        session_resp = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline=create_pipeline_config_proto("channel_selector")
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
                    data=data_config,
                )
            )
        }
        assert cuvis_ai_pb2.TRAIN_STATUS_RUNNING in statuses
        assert cuvis_ai_pb2.TRAIN_STATUS_COMPLETE in statuses

        # 4. Inference
        cube = np.random.randint(0, 65535, size=(1, 16, 16, 61), dtype=np.uint16)
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

    def test_full_gradient_workflow(self, grpc_stub, test_data_files, mock_cuvis_sdk, tmp_path):
        cu3s_file, json_file = test_data_files

        # 1. Create experiment config file with complete configuration
        experiment_path = tmp_path / "test_experiment.yaml"

        # Load the base pipeline config
        pipeline_config_path = Path("configs/pipeline/channel_selector.yaml")
        with open(pipeline_config_path) as f:
            pipeline_dict = yaml.safe_load(f)

        # Create complete experiment config with loss_nodes, metric_nodes, and unfreeze_nodes
        experiment_dict = {
            "name": "test_channel_selector_experiment",
            "pipeline": pipeline_dict,
            "data": {
                "cu3s_file_path": str(cu3s_file),
                "annotation_json_path": str(json_file),
                "train_ids": [0, 1, 2],
                "val_ids": [3, 4],
                "batch_size": 2,
                "processing_mode": "Reflectance",
            },
            "training": {
                "seed": 42,
                "trainer": {
                    "max_epochs": 2,
                    "accelerator": "cpu",
                },
                "optimizer": {
                    "name": "adam",
                    "lr": 0.001,
                },
            },
            "loss_nodes": ["bce"],
            "metric_nodes": ["metrics_anomaly"],
            "unfreeze_nodes": ["SoftChannelSelector", "RXGlobal", "RXLogitHead"],
        }

        with open(experiment_path, "w") as f:
            yaml.dump(experiment_dict, f, default_flow_style=False)

        # 2. Restore experiment (creates session with complete config)
        restore_resp = grpc_stub.RestoreExperiment(
            cuvis_ai_pb2.RestoreExperimentRequest(experiment_path=str(experiment_path))
        )
        session_id = restore_resp.session_id
        assert session_id

        # 3. Statistical warmup
        for _ in grpc_stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        ):
            pass

        # 4. Validate config before gradient training
        config = TrainingConfig(
            trainer=TrainerConfig(max_epochs=2, accelerator="cpu"),
            optimizer=OptimizerConfig(name="adam", lr=0.001),
        )
        validation = grpc_stub.ValidateTrainingConfig(
            cuvis_ai_pb2.ValidateTrainingConfigRequest(
                config=cuvis_ai_pb2.TrainingConfig(config_bytes=config.to_json().encode())
            )
        )
        assert validation.valid
        assert not validation.errors

        # 5. Gradient training with progress monitoring (uses config from experiment)
        progress_updates = list(
            grpc_stub.Train(
                cuvis_ai_pb2.TrainRequest(
                    session_id=session_id,
                    trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
                )
            )
        )
        assert progress_updates
        assert progress_updates[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

        # 6. Save pipeline
        pipeline_path = tmp_path / "model.yaml"
        save_resp = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=str(pipeline_path),
            )
        )
        assert save_resp.success
        assert Path(save_resp.pipeline_path).exists()
        assert Path(save_resp.weights_path).exists()

        # 7. Inference
        cube = np.random.randint(0, 65535, size=(1, 16, 16, 61), dtype=np.uint16)
        inference_resp = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
                output_specs=["selected", "anomaly_metrics"],
            )
        )
        assert inference_resp.outputs or inference_resp.metrics

        # 8. Cleanup
        close_resp = grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        assert close_resp.success

    def test_multiple_sessions(self, grpc_stub, test_data_files, mock_cuvis_sdk):
        cu3s_file, _ = test_data_files
        session_ids: list[str] = []

        for _ in range(3):
            resp = grpc_stub.CreateSession(
                cuvis_ai_pb2.CreateSessionRequest(
                    pipeline=create_pipeline_config_proto("channel_selector"),
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
                pipeline=create_pipeline_config_proto("rx_statistical"),
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
