import grpc
import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers


class TestStatisticalTraining:
    """Test statistical training workflow"""

    @pytest.mark.slow
    def test_train_statistical_completes(self, grpc_stub, session, data_config_factory):
        """Test that statistical training completes successfully"""
        session_id = session()
        data_config = data_config_factory()
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config,
        )

        progress_messages = []
        for progress in grpc_stub.Train(request):
            progress_messages.append(progress)

        # Should have at least one progress message
        assert len(progress_messages) > 0

        # Last message should indicate completion
        final_progress = progress_messages[-1]
        assert final_progress.status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

    @pytest.mark.slow
    def test_statistical_training_updates_pipeline(
        self, grpc_stub, session, data_config_factory, create_test_cube
    ):
        """Test that statistical training updates pipeline nodes"""
        session_id = session()
        data_config = data_config_factory()

        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config,
        )

        for _progress in grpc_stub.Train(request):
            pass  # Consume all progress messages

        # Verify pipeline is updated by running inference
        # (Statistical training should initialize normalizers, selectors, etc.)

        cube, wavelengths = create_test_cube(batch_size=1, height=32, width=32, num_channels=61)

        inference_request = cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.tensor_to_proto(cube),
                wavelengths=helpers.tensor_to_proto(wavelengths),
            ),
        )

        response = grpc_stub.Inference(inference_request)

        # Should have outputs (mask, decisions, etc.)
        assert len(response.outputs) > 0

    @pytest.mark.slow
    def test_statistical_training_status(self, grpc_stub, session, data_config_factory):
        """Test progress status during statistical training"""
        session_id = session()
        data_config = data_config_factory()
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config,
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

    @pytest.mark.slow
    def test_invalid_session_training(self, grpc_stub):
        """Test error for training with invalid session"""
        request = cuvis_ai_pb2.TrainRequest(
            session_id="invalid", trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            for _progress in grpc_stub.Train(request):
                pass

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    @pytest.mark.slow
    def test_get_train_status(self, grpc_stub, session, data_config_factory):
        """Test GetTrainStatus RPC"""
        session_id = session()
        data_config = data_config_factory()
        train_request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config,
        )

        # In simplified test, just consume training
        for _progress in grpc_stub.Train(train_request):
            pass

        # Query status
        status_request = cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id)

        status_response = grpc_stub.GetTrainStatus(status_request)

        # Should have status
        assert status_response.latest_progress.status in [
            cuvis_ai_pb2.TRAIN_STATUS_RUNNING,
            cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
            cuvis_ai_pb2.TRAIN_STATUS_ERROR,
        ]

    @pytest.mark.slow
    def test_train_without_data_config_fails(self, grpc_stub, session):
        """Test that training fails gracefully when data_config is not provided"""
        session_id = session()

        train_request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            for _progress in grpc_stub.Train(train_request):
                pass

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        details = exc_info.value.details()
        assert details is not None and "data" in details.lower()
