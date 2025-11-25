import grpc
import pytest

from cuvis_ai.grpc import cuvis_ai_pb2
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


@pytest.fixture
def trained_session_id(grpc_stub, test_data_path, mock_cuvis_sdk):
    """Create a session and run statistical training first."""
    cu3s_file = test_data_path / "Lentils" / "Lentils_000.cu3s"
    json_file = test_data_path / "Lentils" / "Lentils_000.json"

    if not cu3s_file.exists() or not json_file.exists():
        pytest.skip(f"Test data not found at {test_data_path}")

    create_req = cuvis_ai_pb2.CreateSessionRequest(
        pipeline_type="gradient",
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
    session_id = grpc_stub.CreateSession(create_req).session_id

    stat_req = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
    for _ in grpc_stub.Train(stat_req):
        pass

    return session_id


class TestGradientTraining:
    """Gradient training workflow tests."""

    def _build_training_config(self) -> cuvis_ai_pb2.TrainingConfig:
        cfg = TrainingConfig(
            trainer=TrainerConfig(max_epochs=2, accelerator="cpu", log_every_n_steps=1),
            optimizer=OptimizerConfig(name="adam", lr=0.001),
        )
        return cuvis_ai_pb2.TrainingConfig(config_json=cfg.to_json().encode())

    def test_requires_config(self, grpc_stub, trained_session_id):
        request = cuvis_ai_pb2.TrainRequest(
            session_id=trained_session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            list(grpc_stub.Train(request))

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_streams_progress(self, grpc_stub, trained_session_id):
        request = cuvis_ai_pb2.TrainRequest(
            session_id=trained_session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            config=self._build_training_config(),
        )

        progress = list(grpc_stub.Train(request))
        assert len(progress) > 1
        assert progress[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE

    def test_reports_losses(self, grpc_stub, trained_session_id):
        request = cuvis_ai_pb2.TrainRequest(
            session_id=trained_session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            config=self._build_training_config(),
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

    def test_reports_metrics(self, grpc_stub, trained_session_id):
        request = cuvis_ai_pb2.TrainRequest(
            session_id=trained_session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            config=self._build_training_config(),
        )

        saw_metrics = False
        for update in grpc_stub.Train(request):
            if update.metrics:
                saw_metrics = True
                break
        assert saw_metrics

    def test_reports_stages(self, grpc_stub, trained_session_id):
        request = cuvis_ai_pb2.TrainRequest(
            session_id=trained_session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            config=self._build_training_config(),
        )

        stages = {update.context.stage for update in grpc_stub.Train(request)}
        assert cuvis_ai_pb2.EXECUTION_STAGE_TRAIN in stages

    def test_epoch_progression(self, grpc_stub, trained_session_id):
        request = cuvis_ai_pb2.TrainRequest(
            session_id=trained_session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            config=self._build_training_config(),
        )

        epochs = [update.context.epoch for update in grpc_stub.Train(request)]
        assert max(epochs) >= 0

    def test_invalid_config(self, grpc_stub, trained_session_id):
        bad_request = cuvis_ai_pb2.TrainRequest(
            session_id=trained_session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            config=cuvis_ai_pb2.TrainingConfig(config_json=b"not json"),
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            list(grpc_stub.Train(bad_request))

        assert exc_info.value.code() in (grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.INTERNAL)
