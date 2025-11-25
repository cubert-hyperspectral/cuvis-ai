import grpc
import numpy as np
import pytest

from cuvis_ai.grpc import cuvis_ai_pb2


@pytest.fixture
def session_id(grpc_stub, test_data_path, mock_cuvis_sdk):
    """Create a test session with real data"""
    cu3s_file = test_data_path / "Lentils" / "Lentils_000.cu3s"
    json_file = test_data_path / "Lentils" / "Lentils_000.json"

    # Skip if test data not available
    if not cu3s_file.exists() or not json_file.exists():
        pytest.skip(f"Test data not found at {test_data_path}")

    request = cuvis_ai_pb2.CreateSessionRequest(
        pipeline_type="statistical",
        data_config=cuvis_ai_pb2.DataConfig(
            cu3s_file_path=str(cu3s_file),
            annotation_json_path=str(json_file),
            train_ids=[0, 1, 2],
            val_ids=[3, 4],
            test_ids=[5, 6],
            batch_size=2,
            processing_mode=cuvis_ai_pb2.PROCESSING_MODE_RAW,
        ),
    )
    response = grpc_stub.CreateSession(request)
    return response.session_id


class TestStatisticalTraining:
    """Test statistical training workflow"""

    def test_train_statistical_completes(self, grpc_stub, session_id):
        """Test that statistical training completes successfully"""
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
        )

        progress_messages = []
        for progress in grpc_stub.Train(request):
            progress_messages.append(progress)

        # Should have at least one progress message
        assert len(progress_messages) > 0

        # Last message should indicate completion
        final_progress = progress_messages[-1]
        assert final_progress.status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

    def test_statistical_training_updates_canvas(self, grpc_stub, session_id):
        """Test that statistical training updates canvas nodes"""
        # Train
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
        )

        for _progress in grpc_stub.Train(request):
            pass  # Consume all progress messages

        # Verify canvas is updated by running inference
        # (Statistical training should initialize normalizers, selectors, etc.)
        cube = np.random.rand(1, 32, 32, 61).astype(np.float32)

        inference_request = cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=cuvis_ai_pb2.Tensor(
                    shape=[1, 32, 32, 61],
                    dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                    raw_data=cube.tobytes(),
                )
            ),
        )

        response = grpc_stub.Inference(inference_request)

        # Should have outputs (mask, decisions, etc.)
        assert len(response.outputs) > 0

    def test_statistical_training_status(self, grpc_stub, session_id):
        """Test progress status during statistical training"""
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
        )

        statuses = []
        for progress in grpc_stub.Train(request):
            statuses.append(progress.status)

        # Should have running and complete statuses
        assert (
            cuvis_ai_pb2.TRAIN_STATUS_RUNNING in statuses
            or cuvis_ai_pb2.TRAIN_STATUS_COMPLETE in statuses
        )
        assert statuses[-1] == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

    def test_invalid_session_training(self, grpc_stub):
        """Test error for training with invalid session"""
        request = cuvis_ai_pb2.TrainRequest(
            session_id="invalid", trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            for _progress in grpc_stub.Train(request):
                pass

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    def test_get_train_status(self, grpc_stub, session_id):
        """Test GetTrainStatus RPC"""
        # Start training (non-blocking in real implementation)
        train_request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
        )

        # In simplified test, just consume training
        for _progress in grpc_stub.Train(train_request):
            pass

        # Query status
        status_request = cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id)

        status_response = grpc_stub.GetTrainStatus(status_request)

        # Should have status
        assert status_response.status in [
            cuvis_ai_pb2.TRAIN_STATUS_RUNNING,
            cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
            cuvis_ai_pb2.TRAIN_STATUS_ERROR,
        ]

    def test_train_without_data_config_fails(self, grpc_stub):
        """Test that training fails gracefully when data_config is not provided"""
        # Create session without data_config (inference-only)
        create_request = cuvis_ai_pb2.CreateSessionRequest(
            pipeline_type="statistical",
            # data_config intentionally omitted
        )
        create_response = grpc_stub.CreateSession(create_request)
        session_id = create_response.session_id

        # Attempt to train should fail with FAILED_PRECONDITION
        train_request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            for _progress in grpc_stub.Train(train_request):
                pass

        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        details = exc_info.value.details()
        assert details is not None and "data_config" in details.lower()
