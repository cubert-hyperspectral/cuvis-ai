# gRPC API Deployment Guide

This guide covers running the cuvis.ai gRPC service in local, containerized, and clustered environments.

## Prerequisites
- Python 3.10+
- PyTorch and PyTorch Lightning dependencies installed via project tooling
- gRPC runtime (`grpcio`)
- Access to `.cu3s` data and optional annotation JSON files

## Installation

### Using UV (recommended)
```bash
git clone git@gitlab.cubert.local:cubert/cuvis.ai.git
cd cuvis.ai
uv sync
cd proto && buf generate
```

### Using pip
```bash
git clone git@gitlab.cubert.local:cubert/cuvis.ai.git
cd cuvis.ai
pip install -e .
pip install grpcio grpcio-tools
cd proto && buf generate
```

## Running the Server

The reference server lives in `examples/grpc/server.py`:
```bash
uv run python examples/grpc/server.py
```

Configure message sizes or thread pool depth as needed:
```python
server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=10),
    options=[
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ],
)
```

## TLS

Generate a key/cert pair and enable TLS on the server:
```bash
openssl genrsa -out server.key 2048
openssl req -new -x509 -key server.key -out server.crt -days 365
```
```python
credentials = grpc.ssl_server_credentials([(open("server.key", "rb").read(), open("server.crt", "rb").read())])
server.add_secure_port("[::]:50051", credentials)
```

## Docker

`Dockerfile` example:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

COPY . .
RUN cd proto && buf generate

EXPOSE 50051
CMD ["uv", "run", "python", "examples/grpc/server.py"]
```

Build and run:
```bash
docker build -t cuvis-ai-grpc .
docker run -p 50051:50051 -v /data:/data cuvis-ai-grpc
```

## Kubernetes

Minimal deployment + service:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cuvis-ai-grpc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cuvis-ai-grpc
  template:
    metadata:
      labels:
        app: cuvis-ai-grpc
    spec:
      containers:
      - name: grpc-server
        image: cuvis-ai-grpc:latest
        ports:
        - containerPort: 50051
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: cuvis-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: cuvis-ai-grpc
spec:
  selector:
    app: cuvis-ai-grpc
  ports:
  - port: 50051
    targetPort: 50051
  type: LoadBalancer
```

## Monitoring and Health
- Enable structured logging with `logging.basicConfig`.
- Expose Prometheus metrics (e.g., via `prometheus-client` counters/histograms).
- Add gRPC health checks using `grpc_health.v1.health_pb2_grpc`.

## Security
- Add an authentication interceptor (token or mTLS).
- Enforce rate limiting per client to avoid resource exhaustion.
- Restrict message sizes and concurrency to fit hardware budgets.

## Troubleshooting
- `NOT_FOUND` errors: verify `session_id` exists and was not closed.
- Message size exceeded: raise `grpc.max_send_message_length` / `grpc.max_receive_message_length` on both client and server.
- Slow throughput: increase thread pool size or reduce batch sizes.
- Port conflicts: choose an available port or free the existing listener.
