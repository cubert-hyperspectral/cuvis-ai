"""Centralized session management fixtures for gRPC tests."""

import pytest

from cuvis_ai.grpc import cuvis_ai_pb2
from tests.fixtures import create_pipeline_config_proto


@pytest.fixture
def trained_pipeline_session(grpc_stub, test_data_files):
    """Factory for creating sessions with pipeline and statistical training.

    This is a simpler alternative to trained_session that creates a session
    from a pipeline (not experiment) and runs statistical training. Useful for
    basic inference tests that don't need full experiment config.

    Args:
        grpc_stub: In-process gRPC stub fixture
        test_data_files: Fixture providing validated (cu3s, json) paths

    Yields:
        Callable[[str], str]: Function returning session_id
    """
    cu3s_file, json_file = test_data_files
    created_sessions: list[str] = []

    def _create_trained_pipeline_session(pipeline_path: str = "channel_selector") -> str:
        # Create session from pipeline
        response = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(pipeline=create_pipeline_config_proto(pipeline_path))
        )
        session_id = response.session_id
        created_sessions.append(session_id)

        # Create data config for statistical training
        data_config = cuvis_ai_pb2.DataConfig(
            cu3s_file_path=str(cu3s_file),
            annotation_json_path=str(json_file),
            train_ids=[0, 1, 2],
            val_ids=[3, 4],
            test_ids=[5, 6],
            batch_size=2,
            processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
        )

        # Run statistical training
        stat_req = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config,
        )
        for _ in grpc_stub.Train(stat_req):
            pass

        return session_id

    yield _create_trained_pipeline_session

    for session_id in created_sessions:
        try:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        except Exception:
            pass


@pytest.fixture
def session(grpc_stub):
    """Factory for creating sessions with auto-cleanup.

    Creates a basic session with the specified pipeline type and tracks all
    created session IDs for cleanup after the test finishes.

    Args:
        grpc_stub: In-process gRPC stub fixture

    Yields:
        Callable[[str], str]: Function that creates a session and returns its ID
    """
    created_sessions: list[str] = []

    def _create_session(pipeline_type: str = "channel_selector") -> str:
        response = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(pipeline=create_pipeline_config_proto(pipeline_type))
        )
        session_id = response.session_id
        created_sessions.append(session_id)
        return session_id

    yield _create_session

    for session_id in created_sessions:
        try:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        except Exception:
            # Session may already be closed in the test body
            pass


@pytest.fixture
def trained_session(grpc_stub, test_data_files):
    """Factory for creating sessions with statistical training completed.

    Creates a session using RestoreExperiment to load full experiment config
    (including loss_nodes and metric_nodes), then runs statistical training.
    This ensures gradient training tests have access to all required config.

    Args:
        grpc_stub: In-process gRPC stub fixture
        test_data_files: Fixture providing validated (cu3s, json) paths

    Yields:
        Callable[[str], tuple[str, cuvis_ai_pb2.DataConfig]]:
            Function returning (session_id, data_config)
    """
    cu3s_file, json_file = test_data_files
    created_sessions: list[str] = []

    def _create_trained_session(
        experiment_path: str = "configs/experiment/deep_svdd.yaml",
    ) -> tuple[str, cuvis_ai_pb2.DataConfig]:
        # Use RestoreExperiment to load full config with loss_nodes and metric_nodes
        restore_req = cuvis_ai_pb2.RestoreExperimentRequest(experiment_path=experiment_path)
        response = grpc_stub.RestoreExperiment(restore_req)
        session_id = response.session_id
        created_sessions.append(session_id)

        # Get data config from the restored experiment
        data_config = response.experiment.data

        # Run statistical training
        stat_req = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        )
        for _ in grpc_stub.Train(stat_req):
            pass

        return session_id, data_config

    yield _create_trained_session

    for session_id in created_sessions:
        try:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        except Exception:
            pass
