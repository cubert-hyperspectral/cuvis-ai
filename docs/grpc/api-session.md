!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Session Management API

Sessions provide isolated execution contexts for each client. Each session has independent pipeline state, training configuration, and resources.

---

## Connection & Setup

### Creating a gRPC Stub

```python
from cuvis_ai_core.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc
import grpc

# Configure message size limits for hyperspectral data
options = [
    ("grpc.max_send_message_length", 300 * 1024 * 1024),  # 300 MB
    ("grpc.max_receive_message_length", 300 * 1024 * 1024),
]

# Create channel and stub
channel = grpc.insecure_channel("localhost:50051", options=options)
stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
```

**For production**, use secure channels with TLS:
```python
credentials = grpc.ssl_channel_credentials()
channel = grpc.secure_channel("production-server:50051", credentials, options=options)
```

### Helper Utilities

The `cuvis_ai/utils/grpc_workflow.py` module provides convenience functions that simplify common operations:

```python
from cuvis_ai.utils.grpc_workflow import (
    build_stub,                         # Create configured stub
    config_search_paths,                # Build Hydra search paths
    create_session_with_search_paths,   # Session + search paths
    resolve_trainrun_config,            # Resolve config with Hydra
    apply_trainrun_config,              # Apply resolved config
    format_progress,                    # Pretty-print training progress
)

# Quick connection
stub = build_stub("localhost:50051", max_msg_size=600*1024*1024)

# Quick session setup
session_id = create_session_with_search_paths(stub)
```

---

## CreateSession

**Purpose:** Initialize a new isolated session context.

**Request:**
```protobuf
message CreateSessionRequest {}
```

**Response:**
```protobuf
message CreateSessionResponse {
  string session_id = 1;  // Unique session identifier (UUID)
}
```

**Python Example:**
```python
response = stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
session_id = response.session_id
print(f"Session created: {session_id}")
```

**Notes:**

- Sessions are isolated: separate pipeline, weights, data configs
- Sessions automatically expire after **1 hour** of inactivity
- Each session consumes GPU/CPU memory until closed
- Session IDs are UUIDs (e.g., `"7f3e4d2c-1a9b-4c8d-9e7f-2b5a6c8d9e0f"`)

**See Also:**

- [SetSessionSearchPaths](#setsessionsearchpaths) - Configure Hydra search paths
- [CloseSession](#closesession) - Release resources

---

## SetSessionSearchPaths

**Purpose:** Register Hydra configuration search paths for the session.

**Request:**
```protobuf
message SetSessionSearchPathsRequest {
  string session_id = 1;
  repeated string search_paths = 2;  // Absolute paths
  bool append = 3;                   // false = replace, true = append
}
```

**Response:**
```protobuf
message SetSessionSearchPathsResponse {
  bool success = 1;
}
```

**Python Example:**
```python
from pathlib import Path

# Build search paths (typical pattern)
config_root = Path(__file__).parent.parent / "configs"
search_paths = [
    str(config_root),
    str(config_root / "trainrun"),
    str(config_root / "pipeline"),
    str(config_root / "data"),
    str(config_root / "training"),
]

# Register search paths
stub.SetSessionSearchPaths(
    cuvis_ai_pb2.SetSessionSearchPathsRequest(
        session_id=session_id,
        search_paths=search_paths,
        append=False,  # Replace any existing paths
    )
)
```

**Notes:**

- **Must be called before** `ResolveConfig` for Hydra composition to work
- Paths must be **absolute paths** (not relative)
- Use `append=False` (default) to replace existing paths
- Use `append=True` to add paths to existing list
- Common pattern: call immediately after `CreateSession`

**Helper Function:**
```python
from cuvis_ai.utils.grpc_workflow import config_search_paths, create_session_with_search_paths

# Get standard search paths
paths = config_search_paths(extra_paths=["/custom/configs"])

# Create session with search paths in one call
session_id = create_session_with_search_paths(stub, search_paths=paths)
```

**See Also:**

- [ResolveConfig](api-config.md#resolveconfig) - Resolve configs using these paths
- [Hydra Basics](../config/hydra-basics.md)

---

## CloseSession

**Purpose:** Release all resources associated with a session.

**Request:**
```protobuf
message CloseSessionRequest {
  string session_id = 1;
}
```

**Response:**
```protobuf
message CloseSessionResponse {
  bool success = 1;
}
```

**Python Example:**
```python
# Always close sessions when done
stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
print("Session closed, resources freed")
```

**Best Practice - Use try/finally:**
```python
session_id = None
try:
    session_id = stub.CreateSession(...).session_id
    # ... training or inference ...
finally:
    if session_id:
        stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
```

**Notes:**

- **Always close sessions** to free GPU/CPU memory immediately
- Unclosed sessions expire after 1 hour but hold resources until then
- Closing releases: pipeline, weights, data loaders, CUDA memory
- Safe to call even if session doesn't exist (returns success=false)
- No error if session already expired

**See Also:**

- [Client Patterns: Connections & Sessions](client-connections.md)
