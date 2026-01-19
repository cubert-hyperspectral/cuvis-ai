import pytest

from cuvis_ai_core.grpc.session_manager import SessionManager


def test_multiple_sessions_independent():
    """Test that multiple sessions maintain independent state."""
    manager = SessionManager()

    session1_id = manager.create_session()
    session2_id = manager.create_session()

    assert session1_id != session2_id

    session1 = manager.get_session(session1_id)
    session2 = manager.get_session(session2_id)

    assert session1.search_paths == ["./configs"]
    assert session2.search_paths == ["./configs"]
    assert session1.session_id != session2.session_id


def test_session_lifecycle():
    """Test session creation and cleanup."""
    manager = SessionManager()

    session_id = manager.create_session()
    assert session_id in manager.list_sessions()

    manager.close_session(session_id)
    assert session_id not in manager.list_sessions()


def test_invalid_session_access():
    """Test error on accessing non-existent session."""
    manager = SessionManager()

    with pytest.raises(ValueError) as exc_info:
        manager.get_session("invalid-session-id")

    assert "not found" in str(exc_info.value).lower()
