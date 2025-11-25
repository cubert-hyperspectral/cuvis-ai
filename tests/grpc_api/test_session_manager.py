import time
from datetime import datetime

import pytest

from cuvis_ai.grpc import cuvis_ai_pb2
from cuvis_ai.grpc.session_manager import SessionManager, SessionState
from cuvis_ai.pipeline.canvas import CuvisCanvas


def _sample_data_config() -> cuvis_ai_pb2.DataConfig:
    return cuvis_ai_pb2.DataConfig(
        cu3s_file_path="/tmp/data.cu3s",
        batch_size=4,
        processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
    )


class TestSessionManager:
    def test_create_session_returns_unique_id(self):
        manager = SessionManager()
        data_config = _sample_data_config()

        session_id1 = manager.create_session("channel_selector", {}, data_config)
        session_id2 = manager.create_session("channel_selector", {}, data_config)

        assert session_id1 != session_id2
        assert isinstance(session_id1, str)
        assert session_id1 in manager.list_sessions()
        assert session_id2 in manager.list_sessions()

    def test_get_session_returns_state(self):
        manager = SessionManager()
        data_config = _sample_data_config()

        session_id = manager.create_session("channel_selector", {}, data_config)
        state = manager.get_session(session_id)

        assert isinstance(state, SessionState)
        assert isinstance(state.canvas, CuvisCanvas)
        assert state.data_config == data_config
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.last_accessed, datetime)

    def test_get_session_nonexistent_raises_error(self):
        manager = SessionManager()
        with pytest.raises(ValueError, match="Session .* not found"):
            manager.get_session("missing")

    def test_close_session_removes_state(self):
        manager = SessionManager()
        data_config = _sample_data_config()
        session_id = manager.create_session("channel_selector", {}, data_config)

        manager.close_session(session_id)
        with pytest.raises(ValueError):
            manager.get_session(session_id)

    def test_close_nonexistent_session_raises_error(self):
        manager = SessionManager()
        with pytest.raises(ValueError):
            manager.close_session("unknown")

    def test_get_session_updates_last_accessed(self):
        manager = SessionManager()
        data_config = _sample_data_config()
        session_id = manager.create_session("channel_selector", {}, data_config)

        first_timestamp = manager.get_session(session_id).last_accessed
        time.sleep(0.05)
        state2 = manager.get_session(session_id)

        assert state2.last_accessed > first_timestamp

    def test_cleanup_old_sessions(self):
        manager = SessionManager()
        data_config = _sample_data_config()
        session_id = manager.create_session("channel_selector", {}, data_config)

        # backdate the session
        manager._sessions[session_id].last_accessed = datetime(2000, 1, 1)
        cleaned = manager.cleanup_old_sessions(max_age_hours=1)

        assert cleaned == 1
        assert session_id not in manager.list_sessions()

    def test_create_session_without_data_config(self):
        """Test creating an inference-only session without data_config."""
        manager = SessionManager()

        session_id = manager.create_session("channel_selector", {}, None)
        state = manager.get_session(session_id)

        assert isinstance(state, SessionState)
        assert isinstance(state.canvas, CuvisCanvas)
        assert state.data_config is None
        assert session_id in manager.list_sessions()

    def test_session_state_with_optional_data_config(self):
        """Test that session state properly handles optional data_config."""
        manager = SessionManager()
        data_config = _sample_data_config()

        # Create session with data_config
        session_id_with_data = manager.create_session("channel_selector", {}, data_config)
        state_with_data = manager.get_session(session_id_with_data)
        assert state_with_data.data_config is not None
        assert state_with_data.data_config == data_config

        # Create session without data_config
        session_id_without_data = manager.create_session("channel_selector", {}, None)
        state_without_data = manager.get_session(session_id_without_data)
        assert state_without_data.data_config is None
