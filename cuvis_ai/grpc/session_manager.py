"""Session lifecycle management for the gRPC API."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.training.config import DataConfig, PipelineConfig, TrainingConfig, TrainRunConfig


@dataclass
class SessionState:
    """State for a single training session."""

    session_id: str
    pipeline: CuvisPipeline | None = None
    _pipeline_config: PipelineConfig | None = field(default=None, repr=False)
    data_config: DataConfig | None = None
    training_config: TrainingConfig | None = None
    trainrun_config: TrainRunConfig | None = None
    search_paths: list[str] = field(default_factory=lambda: ["./configs", "./configs/pipeline"])
    is_training: bool = False
    trainer: Any | None = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return pipeline config either from cache or by serializing the pipeline."""
        if self._pipeline_config is not None:
            return self._pipeline_config
        if self.pipeline is None:
            raise ValueError("Pipeline is not initialized for this session")
        return self.pipeline.serialize()

    @pipeline_config.setter
    def pipeline_config(self, value: PipelineConfig | None) -> None:
        self._pipeline_config = value


class SessionManager:
    """Create, track, and retire session resources."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        pipeline: CuvisPipeline | None = None,
        pipeline_config: PipelineConfig | None = None,
        data_config: DataConfig | None = None,
        training_config: TrainingConfig | None = None,
        trainrun_config: TrainRunConfig | None = None,
        search_paths: list[str] | None = None,
    ) -> str:
        """Create a new session with optional pipeline and configs.

        Args:
            pipeline: Optional pipeline instance
            pipeline_config: Optional pipeline configuration
            data_config: Optional data configuration captured during training
            training_config: Optional training configuration captured during training
            trainrun_config: Optional trainrun configuration (for sessions created via RestoreTrainRun)
            search_paths: Optional search paths for resolving configs/weights

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        state = SessionState(
            session_id=session_id,
            pipeline=pipeline,
            _pipeline_config=pipeline_config,
            data_config=data_config,
            training_config=training_config,
            trainrun_config=trainrun_config,
            search_paths=search_paths or ["./configs"],
        )
        self._sessions[session_id] = state
        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> SessionState:
        """Return the session state, updating last_accessed."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        state = self._sessions[session_id]
        state.last_accessed = time.time()
        return state

    def set_search_paths(
        self, session_id: str, paths: list[str], append: bool = True
    ) -> tuple[list[str], list[str]]:
        """Set or extend session search paths."""
        session = self.get_session(session_id)

        valid_paths: list[str] = []
        rejected_paths: list[str] = []

        for path in paths:
            resolved = self._validate_search_path(path)
            if resolved:
                valid_paths.append(resolved)
            else:
                rejected_paths.append(path)
                logger.warning(f"Rejected invalid search path: {path}")

        if append:
            for path in valid_paths:
                if path not in session.search_paths:
                    session.search_paths.append(path)
        else:
            session.search_paths = (
                valid_paths if valid_paths else ["./configs", "./configs/pipeline"]
            )

        logger.info(f"Session {session_id} search paths: {session.search_paths}")
        return session.search_paths, rejected_paths

    def _validate_search_path(self, path: str) -> str | None:
        """Validate search path and return resolved path if valid."""
        try:
            resolved = Path(path).resolve()
            if resolved.exists() and resolved.is_dir() and resolved.is_absolute():
                return str(resolved)
        except Exception as exc:
            logger.debug(f"Path validation failed for {path}: {exc}")
        return None

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

        logger.info(f"Closing session: {session_id}")

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions that haven't been touched within the age window."""
        cutoff = time.time() - (max_age_hours * 3600)
        expired = [sid for sid, state in self._sessions.items() if state.last_accessed < cutoff]

        for sid in expired:
            self.close_session(sid)

        return len(expired)


__all__ = ["SessionManager", "SessionState"]
