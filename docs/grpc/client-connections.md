!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Client Connections & Sessions

Common patterns and best practices for building robust gRPC clients for CUVIS.AI.

---

## Overview

This guide documents proven patterns for CUVIS.AI gRPC clients, extracted from the checked-in
example client set. These patterns cover connection management, session lifecycle, configuration,
training, inference, error handling, and production deployment.

**Philosophy:** Start simple, add complexity only when needed. Most workflows use the standard 5-phase pattern with helper utilities.

---

## Connection Management

### Basic Connection Pattern

```python
from cuvis_ai_core.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc
import grpc

# Configure message size for hyperspectral data
options = [
    ("grpc.max_send_message_length", 300 * 1024 * 1024),  # 300 MB
    ("grpc.max_receive_message_length", 300 * 1024 * 1024),
]

# Create channel and stub
channel = grpc.insecure_channel("localhost:50051", options=options)
stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
```

**Using Helper Utility:**
```python
from cuvis_ai.utils.grpc_workflow import build_stub

# Simplified connection
stub = build_stub("localhost:50051", max_msg_size=600*1024*1024)
```

**Notes:**

- Message size limits are **critical** for hyperspectral data (typical cubes: 100-600 MB)
- Default gRPC limit is 4 MB (too small)
- Both client and server must have matching limits

---

### TLS/SSL Connection (Production)

```python
import grpc

# Load TLS certificates
credentials = grpc.ssl_channel_credentials(
    root_certificates=open("ca.crt", "rb").read(),
)

# Create secure channel
channel = grpc.secure_channel(
    "production-server:50051",
    credentials,
    options=[
        ("grpc.max_send_message_length", 600 * 1024 * 1024),
        ("grpc.max_receive_message_length", 600 * 1024 * 1024),
    ],
)

stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
```

**With Client Certificate (mTLS):**
```python
credentials = grpc.ssl_channel_credentials(
    root_certificates=open("ca.crt", "rb").read(),
    private_key=open("client.key", "rb").read(),
    certificate_chain=open("client.crt", "rb").read(),
)
channel = grpc.secure_channel("production-server:50051", credentials, options=options)
```

**Notes:**

- Always use TLS in production
- mTLS provides client authentication
- Keep certificates secure (never commit to git)

---

### Retry Logic with Exponential Backoff

```python
import time
import grpc

def train_with_retry(stub, session_id, trainer_type, max_retries=3):
    """Train with automatic retry on transient failures."""
    for attempt in range(max_retries):
        try:
            for progress in stub.Train(
                cuvis_ai_pb2.TrainRequest(
                    session_id=session_id,
                    trainer_type=trainer_type,
                )
            ):
                yield progress
            break  # Success, exit retry loop
        except grpc.RpcError as e:
            if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s (error: {e.code()})")
                    time.sleep(wait_time)
                else:
                    print(f"Max retries reached, giving up")
                    raise
            else:
                # Non-retryable error (e.g., INVALID_ARGUMENT), fail immediately
                raise

# Usage
for progress in train_with_retry(stub, session_id, cuvis_ai_pb2.TRAINER_TYPE_GRADIENT):
    print(format_progress(progress))
```

**Retryable vs Non-Retryable Errors:**

- **Retry:** `UNAVAILABLE`, `DEADLINE_EXCEEDED`, `RESOURCE_EXHAUSTED` (transient)
- **Don't Retry:** `INVALID_ARGUMENT`, `NOT_FOUND`, `FAILED_PRECONDITION` (permanent)

---

### Connection Health Check

```python
def check_server_health(stub):
    """Check if server is responsive."""
    try:
        response = stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(
            session_id=response.session_id
        ))
        return True
    except grpc.RpcError as e:
        print(f"Server health check failed: {e.code()} - {e.details()}")
        return False

# Check before starting work
if not check_server_health(stub):
    print("Server is not available, exiting")
    exit(1)

print("Server is healthy, proceeding...")
```

---

## Session Management Patterns

### Context Manager Pattern

```python
from contextlib import contextmanager

@contextmanager
def grpc_session(stub, search_paths=None):
    """Context manager for safe session lifecycle."""
    from cuvis_ai.utils.grpc_workflow import create_session_with_search_paths

    session_id = create_session_with_search_paths(stub, search_paths)
    try:
        yield session_id
    finally:
        stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        print(f"Session {session_id} closed")

# Usage
with grpc_session(stub) as session_id:
    # Training or inference
    for progress in stub.Train(...):
        print(format_progress(progress))
    # Session automatically closed on exit
```

**Benefits:**

- Guaranteed cleanup (even on exceptions)
- Pythonic resource management
- Reduced boilerplate

---

### Manual Session Management with try/finally

```python
session_id = None
try:
    # Create session
    session_id = stub.CreateSession(...).session_id

    # Register search paths
    stub.SetSessionSearchPaths(...)

    # Training or inference
    for progress in stub.Train(...):
        print(progress)

    # Save results
    stub.SavePipeline(...)

finally:
    # Always close session
    if session_id:
        stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
```

---

### Concurrent Sessions for A/B Testing

```python
import concurrent.futures

def run_experiment(stub, trainrun_name, overrides):
    """Run single training experiment."""
    from cuvis_ai.utils.grpc_workflow import (
        create_session_with_search_paths,
        resolve_trainrun_config,
        apply_trainrun_config,
    )

    # Create isolated session
    session_id = create_session_with_search_paths(stub)

    try:
        # Resolve and apply config
        resolved, config_dict = resolve_trainrun_config(
            stub, session_id, trainrun_name, overrides
        )
        apply_trainrun_config(stub, session_id, resolved.config_bytes)

        # Train
        results = []
        for progress in stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            )
        ):
            results.append(progress)

        return config_dict, results

    finally:
        stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

# Run multiple experiments concurrently
experiments = [
    ("deep_svdd", ["training.optimizer.lr=0.001"]),
    ("deep_svdd", ["training.optimizer.lr=0.0005"]),
    ("deep_svdd", ["training.optimizer.lr=0.0001"]),
]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(run_experiment, stub, name, overrides)
        for name, overrides in experiments
    ]

    for future in concurrent.futures.as_completed(futures):
        config, results = future.result()
        final_metrics = results[-1].metrics
        print(f"Experiment {config['name']} (lr={config['training']['optimizer']['lr']})")
        print(f"  Final IoU: {final_metrics.get('val_iou', 0):.4f}")
```

---

## See Also

- [Client Workflows & Error Handling](client-workflows.md) - Configuration, training, inference, and error handling patterns
- [gRPC API Reference: Sessions](api-session.md) - Session RPC method documentation
- [gRPC API Reference: Config](api-config.md) - Configuration RPC method documentation
- [gRPC API Reference: Pipeline](api-pipeline.md) - Pipeline RPC method documentation
- [gRPC Overview](overview.md) - Architecture and concepts
- [Sequence Diagrams](sequence-diagrams.md) - Visual workflows
- [gRPC Tutorial](../tutorials/grpc-workflow.md) - Hands-on tutorial
- [Deployment Guide](../deployment/grpc_deployment.md) - Production patterns
