import logging

import pytest
from grpc_health.v1 import health_pb2

from cuvis_ai_core.grpc.production_server import (
    JSONFormatter,
    ProductionServer,
    load_tls_credentials,
    setup_logging,
)


def test_load_tls_credentials_missing_files(tmp_path) -> None:
    """TLS setup should fail fast when files are missing."""
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"

    with pytest.raises(FileNotFoundError):
        load_tls_credentials(str(cert_path), str(key_path))


def test_setup_logging_uses_json_formatter() -> None:
    """Logging helper should apply JSON formatter."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    try:
        setup_logging("INFO", "json")
        assert any(isinstance(handler.formatter, JSONFormatter) for handler in root_logger.handlers)
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)


def test_server_start_and_shutdown_updates_health_status() -> None:
    """Server should start, report healthy, then mark not serving on shutdown."""
    server = ProductionServer(port=0, max_workers=2)

    try:
        server.start()
        assert server.server is not None
        assert server.health_service is not None

        resp = server.health_service.Check(health_pb2.HealthCheckRequest(), context=None)
        assert resp.status == health_pb2.HealthCheckResponse.SERVING
    finally:
        server.shutdown()

    assert server.health_service is not None
    resp = server.health_service.Check(health_pb2.HealthCheckRequest(), context=None)
    assert resp.status == health_pb2.HealthCheckResponse.NOT_SERVING
