import time

import pytest

from cuvis_ai_core.grpc.helpers import resolve_pipeline_path
from cuvis_ai_core.grpc.session_manager import SessionManager, SessionState
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


class TestSessionManager:
    def test_create_session_returns_unique_id(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("channel_selector")
        pipeline1 = CuvisPipeline.load_pipeline(str(pipeline_path))
        pipeline2 = CuvisPipeline.load_pipeline(str(pipeline_path))

        session_id1 = manager.create_session(pipeline=pipeline1)
        session_id2 = manager.create_session(pipeline=pipeline2)

        assert session_id1 != session_id2
        assert isinstance(session_id1, str)
        assert session_id1 in manager.list_sessions()
        assert session_id2 in manager.list_sessions()

    def test_get_session_returns_state(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("channel_selector")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))

        session_id = manager.create_session(pipeline=pipeline)
        state = manager.get_session(session_id)

        assert isinstance(state, SessionState)
        assert isinstance(state.pipeline, CuvisPipeline)
        assert isinstance(state.created_at, float)
        assert isinstance(state.last_accessed, float)
        assert state.created_at > 0
        assert state.last_accessed > 0

    def test_pipeline_config_property_derives_from_pipeline(self):
        manager = SessionManager()

        pipeline_path = resolve_pipeline_path("channel_selector")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))

        session_id = manager.create_session(pipeline=pipeline)
        state = manager.get_session(session_id)

        pipeline_config = state.pipeline_config
        assert pipeline_config.metadata is not None
        assert pipeline_config.connections is not None

    def test_get_session_nonexistent_raises_error(self):
        manager = SessionManager()
        with pytest.raises(ValueError, match="Session .* not found"):
            manager.get_session("missing")

    def test_close_session_removes_state(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("channel_selector")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id = manager.create_session(pipeline=pipeline)

        manager.close_session(session_id)
        with pytest.raises(ValueError):
            manager.get_session(session_id)

    def test_close_nonexistent_session_raises_error(self):
        manager = SessionManager()
        with pytest.raises(ValueError):
            manager.close_session("unknown")

    def test_get_session_updates_last_accessed(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("channel_selector")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id = manager.create_session(pipeline=pipeline)

        first_timestamp = manager.get_session(session_id).last_accessed
        time.sleep(0.05)
        state2 = manager.get_session(session_id)

        assert state2.last_accessed > first_timestamp

    def test_cleanup_old_sessions(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("channel_selector")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id = manager.create_session(pipeline=pipeline)

        # backdate the session
        manager._sessions[session_id].last_accessed = 0.0
        cleaned = manager.cleanup_old_sessions(max_age_hours=1)

        assert cleaned == 1
        assert session_id not in manager.list_sessions()

    def test_create_session_without_data_config(self):
        """Test creating an inference-only session (no trainrun_config)."""
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("channel_selector")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id = manager.create_session(pipeline=pipeline)
        state = manager.get_session(session_id)

        assert isinstance(state, SessionState)
        assert isinstance(state.pipeline, CuvisPipeline)
        assert state.trainrun_config is None
        assert session_id in manager.list_sessions()

    def test_session_state_with_optional_data_config(self):
        """Test that session state properly handles optional trainrun_config."""
        manager = SessionManager()
        from cuvis_ai_core.training.config import (
            DataConfig,
            TrainingConfig,
            TrainRunConfig,
        )

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("channel_selector")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))

        # Create trainrun config
        trainrun_config = TrainRunConfig(
            name="test_trainrun",
            pipeline=pipeline.serialize(),
            data=DataConfig(
                cu3s_file_path="/tmp/data.cu3s",
                annotation_json_path="/tmp/annotations.json",
                train_ids=[1, 2],
                val_ids=[3],
                test_ids=[4],
                batch_size=4,
            ),
            training=TrainingConfig(),
        )

        # Create session with trainrun_config
        session_id_with_config = manager.create_session(
            pipeline=pipeline, trainrun_config=trainrun_config
        )
        state_with_config = manager.get_session(session_id_with_config)
        assert state_with_config.trainrun_config is not None
        assert state_with_config.trainrun_config.name == "test_trainrun"

        # Create session without trainrun_config
        pipeline2 = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id_without_config = manager.create_session(pipeline=pipeline2)
        state_without_config = manager.get_session(session_id_without_config)
        assert state_without_config.trainrun_config is None
