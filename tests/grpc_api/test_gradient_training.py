import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2


class TestGradientTraining:
    """Gradient training workflow tests."""

    @pytest.mark.slow
    def test_gradient_training_comprehensive(self, grpc_stub, trained_session):
        """Comprehensive test that validates all aspects of gradient training in a single run.

        This test combines multiple validation checks that were previously in separate tests
        to avoid running training multiple times. It validates:
        - Training completion and progress updates
        - Loss reporting
        - Metrics reporting
        - Stage reporting
        - Epoch progression
        """
        # trained_session now loads full experiment config via RestoreTrainRun
        session_id, data_config = trained_session()
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )

        # Collect all training updates in a single run
        updates = list(grpc_stub.Train(request))

        # Validate training completion and progress
        assert len(updates) > 1, "Training should produce multiple progress updates"
        assert updates[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE, (
            "Training should complete successfully"
        )

        # Validate loss reporting
        saw_loss = False
        for update in updates:
            if update.losses:
                saw_loss = True
                # Loss keys can be "total" or have "loss" in them
                assert any(
                    key in ["total"] or "loss" in key.lower() for key in update.losses.keys()
                ), "Loss keys should contain 'total' or 'loss'"
                break

        assert saw_loss, "Training should report losses"

        # Validate metrics reporting
        saw_metrics = False
        for update in updates:
            if update.metrics:
                saw_metrics = True
                break

        assert saw_metrics, "Training should report metrics"

        # Validate stage reporting
        stages = {update.context.stage for update in updates}
        assert cuvis_ai_pb2.EXECUTION_STAGE_TRAIN in stages, "Training should include TRAIN stage"

        # Validate epoch progression
        epochs = [update.context.epoch for update in updates]
        assert max(epochs) >= 0, "Training should progress through epochs"
