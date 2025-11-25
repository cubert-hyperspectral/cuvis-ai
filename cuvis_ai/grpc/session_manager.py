"""Session lifecycle management for the gRPC API."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from cuvis_ai.pipeline.canvas import CuvisCanvas

from .canvas_builder import CanvasBuilder
from .v1 import cuvis_ai_pb2


@dataclass
class SessionState:
    """In-memory state for a single session."""

    canvas: CuvisCanvas
    data_config: cuvis_ai_pb2.DataConfig | None
    created_at: datetime
    last_accessed: datetime
    pipeline_type: str
    pipeline_config: dict
    trainer: object | None = None


class SessionManager:
    """Create, track, and retire session resources."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        pipeline_type: str,
        pipeline_config: dict | None,
        data_config: cuvis_ai_pb2.DataConfig | None,
    ) -> str:
        """Create a new session and return its ID."""
        config = pipeline_config or {}
        session_id = str(uuid.uuid4())

        canvas = CanvasBuilder.create_pipeline(pipeline_type, config)
        now = datetime.now()
        state = SessionState(
            canvas=canvas,
            data_config=data_config,
            created_at=now,
            last_accessed=now,
            pipeline_type=pipeline_type,
            pipeline_config=config,
            trainer=None,
        )
        self._sessions[session_id] = state
        return session_id

    def get_session(self, session_id: str) -> SessionState:
        """Return the session state, updating last_accessed."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        state = self._sessions[session_id]
        state.last_accessed = datetime.now()
        return state

    def close_session(self, session_id: str) -> None:
        """Close a session and drop its resources."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        state = self._sessions.pop(session_id)
        trainer = state.trainer
        if trainer is not None and hasattr(trainer, "cleanup"):
            try:
                trainer.cleanup()
            except Exception:
                # Cleanup best-effort; avoid cascading errors
                pass

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions that haven't been touched within the age window."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        expired = [sid for sid, state in self._sessions.items() if state.last_accessed < cutoff]

        for sid in expired:
            self.close_session(sid)

        return len(expired)


__all__ = ["SessionManager", "SessionState"]
