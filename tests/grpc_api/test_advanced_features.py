import json

import grpc
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from cuvis_ai_core.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


class TestCheckpointManagement:
    """Checkpoint save/load via SavePipeline/LoadPipeline RPCs."""

    def test_save_checkpoint_invalid_session(self, grpc_stub, tmp_path):
        """Test saving pipeline with invalid session (checkpoint equivalent)."""
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.SavePipeline(
                cuvis_ai_pb2.SavePipelineRequest(
                    session_id="invalid",
                    pipeline_path=str(tmp_path / "test.yaml"),
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

        response = grpc_stub.ValidateConfig(
            cuvis_ai_pb2.ValidateConfigRequest(
                config_type="training", config_bytes=config.to_json().encode()
            )
        )

        assert response.valid
        assert len(response.errors) == 0

    def test_validate_invalid_optimizer(self, grpc_stub):
        config = TrainingConfig(
            trainer=TrainerConfig(max_epochs=5),
            optimizer=OptimizerConfig(name="not_an_optimizer", lr=0.001),
        )

        response = grpc_stub.ValidateConfig(
            cuvis_ai_pb2.ValidateConfigRequest(
                config_type="training", config_bytes=config.to_json().encode()
            )
        )

        assert not response.valid
        assert response.errors

    def test_validate_invalid_learning_rate(self, grpc_stub):
        # Test validation with invalid learning rate via raw JSON
        invalid_config_json = json.dumps(
            {"optimizer": {"name": "adam", "lr": -0.5}, "trainer": {"max_epochs": 5}}
        )

        response = grpc_stub.ValidateConfig(
            cuvis_ai_pb2.ValidateConfigRequest(
                config_type="training", config_bytes=invalid_config_json.encode()
            )
        )

        assert not response.valid
        assert any("learning rate" in err.lower() or "lr" in err.lower() for err in response.errors)


class TestComplexInputs:
    """Complex input parsing (bboxes, points, text prompts)."""

    def test_inference_with_bounding_boxes(self, grpc_stub, trained_session, create_test_cube):
        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=32,
            width=32,
            mode="wavelength_dependent",
            num_channels=61,
            wavelength_range=(430.0, 910.0),
        )

        session_id, _data_config = trained_session()

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
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

    def test_inference_with_points(self, grpc_stub, trained_session, create_test_cube):
        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=32,
            width=32,
            mode="wavelength_dependent",
            num_channels=61,
            wavelength_range=(430.0, 910.0),
        )
        session_id, _data_config = trained_session()

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
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

    def test_inference_with_text_prompt(self, grpc_stub, trained_session, create_test_cube):
        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=32,
            width=32,
            mode="wavelength_dependent",
            num_channels=61,
            wavelength_range=(430.0, 910.0),
        )
        session_id, _data_config = trained_session()

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
                    text_prompt="Find defective items",
                ),
            )
        )

        assert response.outputs

    def test_inference_with_multiple_input_types(
        self, grpc_stub, trained_session, create_test_cube
    ):
        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=32,
            width=32,
            mode="wavelength_dependent",
            num_channels=61,
            wavelength_range=(430.0, 910.0),
        )
        session_id, _data_config = trained_session()

        from cuvis_ai_core.grpc import helpers

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
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
