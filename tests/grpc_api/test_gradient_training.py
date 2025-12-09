import pytest

from cuvis_ai.grpc import cuvis_ai_pb2


class TestGradientTraining:
    """Gradient training workflow tests."""

    @pytest.mark.slow
    def test_streams_progress(self, grpc_stub, trained_session):
        # trained_session now loads full experiment config via RestoreExperiment
        session_id, data_config = trained_session()
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )

        progress = list(grpc_stub.Train(request))
        assert len(progress) > 1
        assert progress[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

    @pytest.mark.slow
    def test_reports_losses(self, grpc_stub, trained_session):
        session_id, data_config = trained_session()
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )

        saw_loss = False
        for update in grpc_stub.Train(request):
            if update.losses:
                saw_loss = True
                # Loss keys can be "total" or have "loss" in them
                assert any(
                    key in ["total"] or "loss" in key.lower() for key in update.losses.keys()
                )
        assert saw_loss

    @pytest.mark.slow
    def test_reports_metrics(self, grpc_stub, trained_session):
        session_id, data_config = trained_session()
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )

        saw_metrics = False
        for update in grpc_stub.Train(request):
            if update.metrics:
                saw_metrics = True
                break
        assert saw_metrics

    @pytest.mark.slow
    def test_reports_stages(self, grpc_stub, trained_session):
        session_id, data_config = trained_session()
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )

        stages = {update.context.stage for update in grpc_stub.Train(request)}
        assert cuvis_ai_pb2.EXECUTION_STAGE_TRAIN in stages

    @pytest.mark.slow
    def test_epoch_progression(self, grpc_stub, trained_session):
        session_id, data_config = trained_session()
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )

        epochs = [update.context.epoch for update in grpc_stub.Train(request)]
        assert max(epochs) >= 0
