FROM cubertgmbh/cuvis_python:3.4.0-ubuntu24.04

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && mv /root/.local/bin/uv /usr/local/bin/uv

COPY cuvis_ai cuvis_ai/
COPY entrypoint.sh entrypoint.sh
COPY build_docs.sh build_docs.sh
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
ENV CUVIS=/lib/cuvis
