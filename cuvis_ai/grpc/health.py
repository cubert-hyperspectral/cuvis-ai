"""gRPC health check service implementation."""

from __future__ import annotations

from collections.abc import Generator

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


class HealthService(health_pb2_grpc.HealthServicer):
    """Standard gRPC health checking service.

    Implements the gRPC Health Checking Protocol:
    https://github.com/grpc/grpc/blob/master/doc/health-checking.md
    """

    def __init__(self) -> None:
        self._status = health_pb2.HealthCheckResponse.SERVING

    def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        """Check health status.

        Args:
            request: Health check request
            context: gRPC context

        Returns:
            Health check response with current status
        """
        return health_pb2.HealthCheckResponse(status=self._status)

    def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> Generator[health_pb2.HealthCheckResponse, None, None]:
        """Stream health status updates.

        Args:
            request: Health check request
            context: gRPC context

        Yields:
            Health check responses as status changes
        """
        # For simplicity, just yield current status once
        # A full implementation would stream status changes
        yield health_pb2.HealthCheckResponse(status=self._status)

    def set_not_serving(self) -> None:
        """Mark service as not serving (for graceful shutdown)."""
        self._status = health_pb2.HealthCheckResponse.NOT_SERVING

    def set_serving(self) -> None:
        """Mark service as serving."""
        self._status = health_pb2.HealthCheckResponse.SERVING


__all__ = ["HealthService"]
