"""Session management service component."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc

from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    pass


class SessionService:
    """Session management for gRPC service."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def create_session(
        self,
        request: cuvis_ai_pb2.CreateSessionRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.CreateSessionResponse:
        """Create a new session with pipeline configuration."""
        # Phase 4: allow parameter-less creation (explicit pipeline setup via BuildPipeline)
        try:
            session_id = self.session_manager.create_session()
            return cuvis_ai_pb2.CreateSessionResponse(session_id=session_id)
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to create session: {exc}")
            return cuvis_ai_pb2.CreateSessionResponse()

    def close_session(
        self,
        request: cuvis_ai_pb2.CloseSessionRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.CloseSessionResponse:
        """Close and clean up a session."""
        try:
            self.session_manager.close_session(request.session_id)
            return cuvis_ai_pb2.CloseSessionResponse(success=True)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.CloseSessionResponse(success=False)
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to close session: {exc}")
            return cuvis_ai_pb2.CloseSessionResponse(success=False)

    def set_session_search_paths(
        self,
        request: cuvis_ai_pb2.SetSessionSearchPathsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SetSessionSearchPathsResponse:
        """Set or extend search paths for config resolution and weight loading."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.SetSessionSearchPathsResponse(success=False)

        append = request.append if request.HasField("append") else True

        try:
            current_paths, rejected_paths = self.session_manager.set_search_paths(
                session.session_id,
                list(request.search_paths),
                append=append,
            )
            return cuvis_ai_pb2.SetSessionSearchPathsResponse(
                success=True,
                current_paths=current_paths,
                rejected_paths=rejected_paths,
            )
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to update search paths: {exc}")
            return cuvis_ai_pb2.SetSessionSearchPathsResponse(success=False)


__all__ = ["SessionService"]
