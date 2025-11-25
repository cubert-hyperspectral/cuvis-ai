import grpc
import numpy as np
import pytest

from cuvis_ai.grpc import cuvis_ai_pb2
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


@pytest.fixture
def trained_session_id(grpc_stub, test_data_path, mock_cuvis_sdk):
    """Create and fully train a gradient session (statistical + gradient)."""
    cu3s_file = test_data_path / "Lentils" / "Lentils_000.cu3s"
    json_file = test_data_path / "Lentils" / "Lentils_000.json"

    if not cu3s_file.exists() or not json_file.exists():
        pytest.skip(f"Test data not found at {test_data_path}")

    create_request = cuvis_ai_pb2.CreateSessionRequest(
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
    session_id = grpc_stub.CreateSession(create_request).session_id

    # Statistical training first
    stat_request = cuvis_ai_pb2.TrainRequest(
        session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
    )
    for _ in grpc_stub.Train(stat_request):
        pass

    # Gradient training
    config = TrainingConfig(
        trainer=TrainerConfig(max_epochs=2, accelerator="cpu"),
        optimizer=OptimizerConfig(name="adam", lr=0.001),
    )
    grad_request = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        config=cuvis_ai_pb2.TrainingConfig(config_json=config.to_json().encode()),
    )
    for _ in grpc_stub.Train(grad_request):
        pass

    return session_id


class TestCheckpointManagement:
    """Checkpoint save/load RPCs."""

    def test_save_checkpoint(self, grpc_stub, trained_session_id, tmp_path):
        checkpoint_path = tmp_path / "test_checkpoint.ckpt"

        response = grpc_stub.SaveCheckpoint(
            cuvis_ai_pb2.SaveCheckpointRequest(
                session_id=trained_session_id,
                checkpoint_path=str(checkpoint_path),
            )
        )

        assert response.success
        assert checkpoint_path.exists()

    def test_load_checkpoint(self, grpc_stub, trained_session_id, tmp_path):
        checkpoint_path = tmp_path / "test_checkpoint.ckpt"
        grpc_stub.SaveCheckpoint(
            cuvis_ai_pb2.SaveCheckpointRequest(
                session_id=trained_session_id,
                checkpoint_path=str(checkpoint_path),
            )
        )

        new_session = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline_type="gradient",
                data_config=cuvis_ai_pb2.DataConfig(
                    cu3s_file_path=str(checkpoint_path),  # Placeholder path
                    batch_size=1,
                ),
            )
        )

        response = grpc_stub.LoadCheckpoint(
            cuvis_ai_pb2.LoadCheckpointRequest(
                session_id=new_session.session_id,
                checkpoint_path=str(checkpoint_path),
            )
        )

        assert response.success

    def test_save_checkpoint_invalid_session(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.SaveCheckpoint(
                cuvis_ai_pb2.SaveCheckpointRequest(
                    session_id="invalid",
                    checkpoint_path="/tmp/test.ckpt",
                )
            )

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


class TestTrainingCapabilities:
    """Training capability discovery."""

    def test_get_training_capabilities(self, grpc_stub):
        response = grpc_stub.GetTrainingCapabilities(cuvis_ai_pb2.GetTrainingCapabilitiesRequest())

        assert "adam" in response.supported_optimizers
        assert response.supported_schedulers
        assert response.supported_callbacks

    def test_callback_info_structure(self, grpc_stub):
        response = grpc_stub.GetTrainingCapabilities(cuvis_ai_pb2.GetTrainingCapabilitiesRequest())

        for callback in response.supported_callbacks:
            assert callback.type
            assert callback.description
            if callback.parameters:
                param = callback.parameters[0]
                assert param.name
                assert param.type


class TestConfigValidation:
    """Training config validation RPC."""

    def test_validate_valid_config(self, grpc_stub):
        config = TrainingConfig(
            trainer=TrainerConfig(max_epochs=5, accelerator="cpu"),
            optimizer=OptimizerConfig(name="adam", lr=0.001),
        )

        response = grpc_stub.ValidateTrainingConfig(
            cuvis_ai_pb2.ValidateTrainingConfigRequest(
                config=cuvis_ai_pb2.TrainingConfig(config_json=config.to_json().encode())
            )
        )

        assert response.valid
        assert len(response.errors) == 0

    def test_validate_invalid_optimizer(self, grpc_stub):
        config = TrainingConfig(
            trainer=TrainerConfig(max_epochs=5),
            optimizer=OptimizerConfig(name="not_an_optimizer", lr=0.001),
        )

        response = grpc_stub.ValidateTrainingConfig(
            cuvis_ai_pb2.ValidateTrainingConfigRequest(
                config=cuvis_ai_pb2.TrainingConfig(config_json=config.to_json().encode())
            )
        )

        assert not response.valid
        assert response.errors

    def test_validate_invalid_learning_rate(self, grpc_stub):
        config = TrainingConfig(
            trainer=TrainerConfig(max_epochs=5),
            optimizer=OptimizerConfig(name="adam", lr=-0.5),
        )

        response = grpc_stub.ValidateTrainingConfig(
            cuvis_ai_pb2.ValidateTrainingConfigRequest(
                config=cuvis_ai_pb2.TrainingConfig(config_json=config.to_json().encode())
            )
        )

        assert not response.valid
        assert any("learning rate" in err.lower() or "lr" in err.lower() for err in response.errors)


class TestComplexInputs:
    """Complex input parsing (bboxes, points, text prompts)."""

    def test_inference_with_bounding_boxes(self, grpc_stub, trained_session_id):
        cube = np.random.rand(1, 32, 32, 61).astype(np.float32)

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=trained_session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=cuvis_ai_pb2.Tensor(
                        shape=list(cube.shape),
                        dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                        raw_data=cube.tobytes(),
                    ),
                    bboxes=cuvis_ai_pb2.BoundingBoxes(
                        boxes=[
                            cuvis_ai_pb2.BoundingBox(
                                element_id=0,
                                x_min=5,
                                y_min=5,
                                x_max=15,
                                y_max=15,
                            )
                        ]
                    ),
                ),
            )
        )

        assert response.outputs

    def test_inference_with_points(self, grpc_stub, trained_session_id):
        cube = np.random.rand(1, 32, 32, 61).astype(np.float32)

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=trained_session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=cuvis_ai_pb2.Tensor(
                        shape=list(cube.shape),
                        dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                        raw_data=cube.tobytes(),
                    ),
                    points=cuvis_ai_pb2.Points(
                        points=[
                            cuvis_ai_pb2.Point(
                                element_id=0,
                                x=10.5,
                                y=15.5,
                                type=cuvis_ai_pb2.POINT_TYPE_POSITIVE,
                            ),
                            cuvis_ai_pb2.Point(
                                element_id=0,
                                x=20.5,
                                y=25.5,
                                type=cuvis_ai_pb2.POINT_TYPE_NEGATIVE,
                            ),
                        ]
                    ),
                ),
            )
        )

        assert response.outputs

    def test_inference_with_text_prompt(self, grpc_stub, trained_session_id):
        cube = np.random.rand(1, 32, 32, 61).astype(np.float32)

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=trained_session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=cuvis_ai_pb2.Tensor(
                        shape=list(cube.shape),
                        dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                        raw_data=cube.tobytes(),
                    ),
                    text_prompt="Find defective items",
                ),
            )
        )

        assert response.outputs

    def test_inference_with_multiple_input_types(self, grpc_stub, trained_session_id):
        cube = np.random.rand(1, 32, 32, 61).astype(np.float32)

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=trained_session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=cuvis_ai_pb2.Tensor(
                        shape=list(cube.shape),
                        dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                        raw_data=cube.tobytes(),
                    ),
                    bboxes=cuvis_ai_pb2.BoundingBoxes(
                        boxes=[
                            cuvis_ai_pb2.BoundingBox(
                                element_id=0,
                                x_min=5,
                                y_min=5,
                                x_max=15,
                                y_max=15,
                            )
                        ]
                    ),
                    points=cuvis_ai_pb2.Points(
                        points=[
                            cuvis_ai_pb2.Point(
                                element_id=0,
                                x=10.5,
                                y=15.5,
                                type=cuvis_ai_pb2.POINT_TYPE_POSITIVE,
                            )
                        ]
                    ),
                    text_prompt="Find anomalies",
                ),
            )
        )

        assert response.outputs
