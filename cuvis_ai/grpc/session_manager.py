"""Session lifecycle management for the gRPC API."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.training.config import DataConfig, ExperimentConfig, PipelineConfig, TrainingConfig


@dataclass
class SessionState:
    """In-memory state for a single session."""

    pipeline: CuvisPipeline
    data_config: DataConfig | None
    training_config: TrainingConfig | None
    created_at: datetime
    last_accessed: datetime
    trainer: object | None = None
    experiment_config: ExperimentConfig | None = None

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return the current pipeline configuration derived from the pipeline."""
        return self.pipeline.serialize()


class SessionManager:
    """Create, track, and retire session resources."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        pipeline: CuvisPipeline,
        data_config: DataConfig | None = None,
        training_config: TrainingConfig | None = None,
        experiment_config: ExperimentConfig | None = None,
    ) -> str:
        """Create a new session with a pipeline instance.

        Args:
            pipeline: Pipeline instance
            data_config: Optional data configuration captured during training
            training_config: Optional training configuration captured during training
            experiment_config: Optional experiment configuration (for sessions created via RestoreExperiment)

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()

        state = SessionState(
            pipeline=pipeline,
            data_config=data_config,
            training_config=training_config,
            created_at=now,
            last_accessed=now,
            trainer=None,
            experiment_config=experiment_config,
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
