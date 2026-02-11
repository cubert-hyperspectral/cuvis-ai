# Production gRPC server using Cubert base image
FROM cubertgmbh/cuvis_python:3.5.0-ubuntu24.04

ENV CUVIS=/lib/cuvis

WORKDIR /app

# Install uv using standalone installer
RUN apt-get update && \
    apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files and application code
COPY pyproject.toml uv.lock ./
COPY cuvis_ai/ /app/cuvis_ai/
COPY .env.example /app/.env.example

# Install dependencies (version injected at build time for setuptools-scm)
ARG PACKAGE_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUVIS_AI=${PACKAGE_VERSION}
RUN uv sync --frozen --no-dev

# Create non-root user
ARG APP_USER=cuvisai
ARG APP_UID=1000
RUN set -eux; \
    GROUP_NAME="${APP_USER}"; \
    if getent group "${APP_UID}" >/dev/null; then \
        GROUP_NAME="$(getent group "${APP_UID}" | cut -d: -f1)"; \
    else \
        groupadd -g "${APP_UID}" "${GROUP_NAME}"; \
    fi; \
    if id -u "${APP_USER}" >/dev/null 2>&1; then \
        usermod -u "${APP_UID}" -g "${GROUP_NAME}" "${APP_USER}"; \
    elif getent passwd "${APP_UID}" >/dev/null; then \
        EXISTING_USER="$(getent passwd "${APP_UID}" | cut -d: -f1)"; \
        usermod -l "${APP_USER}" -d /home/"${APP_USER}" -m -g "${GROUP_NAME}" "${EXISTING_USER}"; \
    else \
        useradd -m -u "${APP_UID}" -g "${GROUP_NAME}" "${APP_USER}"; \
    fi; \
    chown -R "${APP_USER}":"${GROUP_NAME}" /app

USER ${APP_USER}

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD uv run python -c "import grpc; from grpc_health.v1 import health_pb2, health_pb2_grpc; \
    channel = grpc.insecure_channel('localhost:50051'); \
    stub = health_pb2_grpc.HealthStub(channel); \
    response = stub.Check(health_pb2.HealthCheckRequest()); \
    exit(0 if response.status == 1 else 1)"

# Set Python path
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Run server
CMD ["uv", "run", "python", "-m", "cuvis_ai.grpc.production_server"]
