!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# gRPC API Reference

Complete reference documentation for the CuvisAIService with 46 RPC methods.

---

## Overview

The CUVIS.AI gRPC service exposes all functionality through a single `CuvisAIService` endpoint with 46 RPC methods organized into 6 functional categories:

| Category | Methods | Purpose |
|----------|---------|---------|
| [Session Management](#session-management) | 3 | Create, configure, and close isolated execution contexts |
| [Configuration Management](#configuration-management) | 4 | Resolve, validate, and apply Hydra configurations |
| [Pipeline Management](#pipeline-management) | 5 | Load, save, and manage pipeline state |
| [Training Operations](#training-operations) | 3 | Execute statistical and gradient-based training |
| [Inference Operations](#inference-operations) | 1 | Run predictions on trained pipelines |
| [Introspection & Discovery](#introspection-discovery) | 6 | Query capabilities, inspect pipelines, visualize graphs |

**Protocol Buffers:** All methods use Protocol Buffers (protobuf) for serialization.

**Service Definition:** `CuvisAIService` in `cuvis_ai_core.proto`

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

The `examples/grpc/workflow_utils.py` module provides convenience functions that simplify common operations:

```python
from examples.grpc.workflow_utils import (
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

## Session Management

Sessions provide isolated execution contexts for each client. Each session has independent pipeline state, training configuration, and resources.

### CreateSession

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

### SetSessionSearchPaths

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
from examples.grpc.workflow_utils import config_search_paths, create_session_with_search_paths

# Get standard search paths
paths = config_search_paths(extra_paths=["/custom/configs"])

# Create session with search paths in one call
session_id = create_session_with_search_paths(stub, search_paths=paths)
```

**See Also:**
- [ResolveConfig](#resolveconfig) - Resolve configs using these paths
- [Hydra Composition Guide](../config/hydra-composition.md)

---

### CloseSession

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
- [Client Patterns: Session Management](client-patterns.md#session-management-patterns)

---

## Configuration Management

The configuration service integrates with Hydra for powerful config composition, validation, and dynamic overrides.

### ResolveConfig

**Purpose:** Resolve configuration using Hydra composition with optional overrides.

**Request:**
```protobuf
message ResolveConfigRequest {
  string session_id = 1;
  string config_type = 2;        // "trainrun", "pipeline", "training", "data"
  string path = 3;                // Relative path in search paths
  repeated string overrides = 4;  // Hydra override syntax
}
```

**Response:**
```protobuf
message ResolveConfigResponse {
  bytes config_bytes = 1;    // JSON-serialized config
  string resolved_path = 2;  // Full path that was resolved
}
```

**Python Example:**
```python
import json

# Resolve trainrun config with overrides
response = stub.ResolveConfig(
    cuvis_ai_pb2.ResolveConfigRequest(
        session_id=session_id,
        config_type="trainrun",
        path="trainrun/deep_svdd",  # Or just "deep_svdd"
        overrides=[
            "training.trainer.max_epochs=50",
            "training.optimizer.lr=0.0005",
            "data.batch_size=8",
        ],
    )
)

# Parse returned JSON
config_dict = json.loads(response.config_bytes.decode("utf-8"))
print(f"Resolved config: {config_dict['name']}")
print(f"Pipeline: {config_dict['pipeline']['name']}")
```

**Config Types:**
- `"trainrun"` - Complete training run composition (pipeline + data + training)
- `"pipeline"` - Pipeline-only configuration
- `"training"` - Training parameters (optimizer, scheduler, trainer)
- `"data"` - Data loading configuration

**Override Patterns:**
```python
overrides = [
    # Training parameters
    "training.trainer.max_epochs=100",
    "training.trainer.accelerator=gpu",
    "training.optimizer.lr=0.001",
    "training.optimizer.weight_decay=0.01",
    "training.scheduler.mode=min",
    "training.scheduler.patience=10",

    # Data parameters
    "data.batch_size=16",
    "data.train_ids=[0,1,2]",
    "data.val_ids=[3,4]",
    "data.cu3s_file_path=/data/Lentils_000.cu3s",

    # Pipeline node parameters
    "pipeline.nodes.channel_selector.params.tau_start=8.0",
    "pipeline.nodes.rx_detector.params.eps=1e-6",
    "pipeline.nodes.normalizer.params.use_running_stats=true",
]
```

**Notes:**
- Requires `SetSessionSearchPaths` to be called first
- Returns JSON bytes (decode with `.decode("utf-8")` and parse with `json.loads()`)
- Hydra resolves config group composition, interpolations, and overrides
- Override syntax follows Hydra conventions (dot notation for nested fields)

**Helper Function:**
```python
from examples.grpc.workflow_utils import resolve_trainrun_config

# Resolve trainrun config (returns response + parsed dict)
resolved, config_dict = resolve_trainrun_config(
    stub,
    session_id,
    "deep_svdd",
    overrides=["training.trainer.max_epochs=10"],
)
```

**See Also:**
- [SetTrainRunConfig](#settrainrunconfig) - Apply resolved config
- [ValidateConfig](#validateconfig) - Pre-validate configs
- [Hydra Composition Guide](../config/hydra-composition.md)
- [TrainRun Schema](../config/trainrun-schema.md)

---

### SetTrainRunConfig

**Purpose:** Apply resolved trainrun configuration to session (builds pipeline, sets data/training configs).

**Request:**
```protobuf
message SetTrainRunConfigRequest {
  string session_id = 1;
  TrainRunConfig config = 2;
}

message TrainRunConfig {
  bytes config_bytes = 1;  // JSON from ResolveConfig
}
```

**Response:**
```protobuf
message SetTrainRunConfigResponse {
  bool success = 1;
}
```

**Python Example:**
```python
# First resolve config
resolved = stub.ResolveConfig(
    cuvis_ai_pb2.ResolveConfigRequest(
        session_id=session_id,
        config_type="trainrun",
        path="trainrun/rx_statistical",
    )
)

# Apply to session
stub.SetTrainRunConfig(
    cuvis_ai_pb2.SetTrainRunConfigRequest(
        session_id=session_id,
        config=cuvis_ai_pb2.TrainRunConfig(config_bytes=resolved.config_bytes),
    )
)
print("TrainRun config applied, pipeline built")
```

**What This Does:**
1. Parses trainrun configuration JSON
2. Builds pipeline from pipeline config
3. Initializes data loader from data config
4. Sets training parameters (optimizer, scheduler, trainer)
5. Prepares session for training or inference

**Notes:**
- Must be called after `ResolveConfig`
- Replaces any existing pipeline/config in session
- After this call, session is ready for `Train()` or `Inference()`
- Validates config structure (raises error if malformed)

**Helper Function:**
```python
from examples.grpc.workflow_utils import apply_trainrun_config

apply_trainrun_config(stub, session_id, resolved.config_bytes)
```

**See Also:**
- [ResolveConfig](#resolveconfig) - Resolve config first
- [Train](#train) - Execute training after config applied

---

### ValidateConfig

**Purpose:** Pre-validate configuration before applying (catch errors early).

**Request:**
```protobuf
message ValidateConfigRequest {
  string config_type = 1;  // "training", "pipeline", "data", etc.
  bytes config_bytes = 2;   // JSON configuration
}
```

**Response:**
```protobuf
message ValidateConfigResponse {
  bool valid = 1;
  repeated string errors = 2;     // Fatal errors
  repeated string warnings = 3;   // Non-fatal warnings
}
```

**Python Example:**
```python
import json

# Validate training config before use
training_config = {
    "trainer": {"max_epochs": 10, "accelerator": "gpu"},
    "optimizer": {"name": "adam", "lr": 0.001},
}

validation = stub.ValidateConfig(
    cuvis_ai_pb2.ValidateConfigRequest(
        config_type="training",
        config_bytes=json.dumps(training_config).encode("utf-8"),
    )
)

if not validation.valid:
    print("Configuration validation failed:")
    for error in validation.errors:
        print(f"  ERROR: {error}")
    raise ValueError("Invalid configuration")

for warning in validation.warnings:
    print(f"  WARNING: {warning}")
```

**Config Types:**
- `"training"` - Training parameters (optimizer, scheduler, trainer)
- `"pipeline"` - Pipeline structure and node configs
- `"data"` - Data loading configuration
- `"trainrun"` - Complete trainrun composition

**Common Validation Errors:**
- Missing required fields (e.g., `optimizer.lr`)
- Invalid values (e.g., negative `max_epochs`)
- Type mismatches (e.g., string for numeric field)
- Unknown optimizer/scheduler names
- Incompatible node connections in pipeline

**Notes:**
- Validation is **optional** but highly recommended
- Catches configuration errors before starting training
- Warnings are informational (config still valid)
- Errors mean config is invalid and will fail if applied

**See Also:**
- [SetTrainRunConfig](#settrainrunconfig) - Apply config after validation
- [TrainRun Schema](../config/trainrun-schema.md)

---

### GetTrainingCapabilities

**Purpose:** Discover supported optimizers, schedulers, callbacks, and their parameter schemas.

**Request:**
```protobuf
message GetTrainingCapabilitiesRequest {}
```

**Response:**
```protobuf
message GetTrainingCapabilitiesResponse {
  repeated string optimizer_names = 1;     // e.g., ["adam", "sgd", "adamw"]
  repeated string scheduler_names = 2;     // e.g., ["step_lr", "reduce_on_plateau"]
  repeated string callback_names = 3;      // e.g., ["early_stopping", "model_checkpoint"]
  map<string, ParameterSchema> optimizer_schemas = 4;
  map<string, ParameterSchema> scheduler_schemas = 5;
}
```

**Python Example:**
```python
capabilities = stub.GetTrainingCapabilities(
    cuvis_ai_pb2.GetTrainingCapabilitiesRequest()
)

print("Available optimizers:", capabilities.optimizer_names)
print("Available schedulers:", capabilities.scheduler_names)
print("Available callbacks:", capabilities.callback_names)

# Get parameter schema for Adam optimizer
adam_schema = capabilities.optimizer_schemas["adam"]
print(f"Adam parameters: {adam_schema}")
```

**Use Cases:**
- Dynamic UI generation (list available options)
- Config validation (check if optimizer exists)
- Documentation generation
- Discovery for programmatic workflows

**See Also:**
- [ValidateConfig](#validateconfig) - Validate configs using these capabilities

---

## Pipeline Management

Manage pipeline loading, saving, and restoration.

### LoadPipeline

**Purpose:** Build pipeline from YAML configuration.

**Request:**
```protobuf
message LoadPipelineRequest {
  string session_id = 1;
  PipelineConfig pipeline = 2;
}

message PipelineConfig {
  bytes config_bytes = 1;  // YAML or JSON serialized
}
```

**Response:**
```protobuf
message LoadPipelineResponse {
  bool success = 1;
}
```

**Python Example:**
```python
from pathlib import Path
import yaml
import json

# Load pipeline config from YAML file
pipeline_yaml = yaml.safe_load(Path("pipeline.yaml").read_text())
pipeline_json = json.dumps(pipeline_yaml).encode("utf-8")

stub.LoadPipeline(
    cuvis_ai_pb2.LoadPipelineRequest(
        session_id=session_id,
        pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_json),
    )
)
print("Pipeline loaded from config")
```

**Notes:**
- Pipeline config can be YAML or JSON (server parses both)
- Builds complete pipeline graph from node definitions and connections
- Does NOT load weights (use `LoadPipelineWeights` separately)
- After loading, pipeline is ready for training or inference
- See [Pipeline Schema](../config/pipeline-schema.md) for config format

**See Also:**
- [LoadPipelineWeights](#loadpipelineweights) - Load trained weights
- [SavePipeline](#savepipeline) - Save pipeline config + weights
- [Pipeline Schema](../config/pipeline-schema.md)

---

### LoadPipelineWeights

**Purpose:** Load trained weights into pipeline from checkpoint file.

**Request:**
```protobuf
message LoadPipelineWeightsRequest {
  string session_id = 1;
  string weights_path = 2;  // Path to .pt file
  bool strict = 3;          // Require exact match (default: true)
}
```

**Response:**
```protobuf
message LoadPipelineWeightsResponse {
  bool success = 1;
  repeated string missing_keys = 2;    // Keys in config but not in weights
  repeated string unexpected_keys = 3; // Keys in weights but not in config
}
```

**Python Example:**
```python
# Load pipeline first, then weights
stub.LoadPipeline(...)  # See LoadPipeline example

response = stub.LoadPipelineWeights(
    cuvis_ai_pb2.LoadPipelineWeightsRequest(
        session_id=session_id,
        weights_path="outputs/deep_svdd.pt",
        strict=True,  # Fail if weights don't match exactly
    )
)

if response.missing_keys:
    print(f"Warning: Missing keys: {response.missing_keys}")
if response.unexpected_keys:
    print(f"Warning: Unexpected keys: {response.unexpected_keys}")
```

**Strict vs Non-Strict Loading:**
- `strict=True` (default): Fails if weights don't match pipeline exactly
- `strict=False`: Loads matching weights, ignores mismatches (useful for transfer learning)

**Notes:**
- Must call `LoadPipeline` first to build pipeline structure
- Weights file is PyTorch `.pt` checkpoint
- Use `strict=False` for transfer learning or partial weight loading
- Check `missing_keys` and `unexpected_keys` for debugging

**See Also:**
- [LoadPipeline](#loadpipeline) - Load pipeline config first
- [SavePipeline](#savepipeline) - Save weights

---

### SavePipeline

**Purpose:** Save pipeline configuration and weights to files.

**Request:**
```protobuf
message SavePipelineRequest {
  string session_id = 1;
  string pipeline_path = 2;        // Path for YAML config
  PipelineMetadata metadata = 3;   // Optional metadata
}

message PipelineMetadata {
  string name = 1;
  string description = 2;
  repeated string tags = 3;
  string author = 4;
}
```

**Response:**
```protobuf
message SavePipelineResponse {
  string pipeline_path = 1;  // Saved YAML config path
  string weights_path = 2;   // Saved .pt weights path
}
```

**Python Example:**
```python
# Save pipeline after training
response = stub.SavePipeline(
    cuvis_ai_pb2.SavePipelineRequest(
        session_id=session_id,
        pipeline_path="outputs/my_pipeline.yaml",
        metadata=cuvis_ai_pb2.PipelineMetadata(
            name="Deep SVDD Anomaly Detector",
            description="Trained on Lentils dataset with 50 epochs",
            tags=["anomaly_detection", "deep_svdd", "production"],
            author="your_name",
        ),
    )
)

print(f"Pipeline saved to: {response.pipeline_path}")
print(f"Weights saved to: {response.weights_path}")
```

**What Gets Saved:**
1. **YAML config** - Complete pipeline structure and node parameters
2. **Weights file (.pt)** - PyTorch checkpoint with trained weights
3. **Metadata** - Optional name, description, tags, author

**Notes:**
- Automatically creates weights path by replacing `.yaml` with `.pt`
- Metadata is embedded in YAML config file
- Use this for inference-only deployment (no training state)
- For full reproducibility, use `SaveTrainRun` instead

**See Also:**
- [LoadPipeline](#loadpipeline) + [LoadPipelineWeights](#loadpipelineweights) - Restore pipeline
- [SaveTrainRun](#savetrainrun) - Save complete trainrun (includes data/training config)

---

### SaveTrainRun

**Purpose:** Save complete trainrun configuration (pipeline + data + training configs).

**Request:**
```protobuf
message SaveTrainRunRequest {
  string session_id = 1;
  string trainrun_path = 2;  // Path for trainrun YAML
  bool save_weights = 3;     // Include weights (default: true)
}
```

**Response:**
```protobuf
message SaveTrainRunResponse {
  string trainrun_path = 1;  // Saved trainrun config
  string weights_path = 2;   // Saved weights (if save_weights=true)
}
```

**Python Example:**
```python
# Save complete trainrun after training
response = stub.SaveTrainRun(
    cuvis_ai_pb2.SaveTrainRunRequest(
        session_id=session_id,
        trainrun_path="outputs/deep_svdd_run.yaml",
        save_weights=True,
    )
)

print(f"TrainRun saved to: {response.trainrun_path}")
print(f"Weights saved to: {response.weights_path}")
```

**What Gets Saved:**
- **Pipeline config** - Complete pipeline structure
- **Data config** - Data loading configuration (paths, train/val/test splits, batch size)
- **Training config** - Optimizer, scheduler, trainer parameters
- **Weights** (optional) - Trained model weights

**TrainRun vs Pipeline:**
| Feature | SavePipeline | SaveTrainRun |
|---------|--------------|--------------|
| Pipeline config | ✅ | ✅ |
| Weights | ✅ | ✅ |
| Data config | ❌ | ✅ |
| Training config | ❌ | ✅ |
| **Use for** | Inference deployment | Reproducibility, resume training |

**Notes:**
- Use `SaveTrainRun` for full reproducibility (can resume training later)
- Use `SavePipeline` for inference-only deployment (smaller, no training overhead)
- Trainrun can be restored with `RestoreTrainRun`

**See Also:**
- [RestoreTrainRun](#restoretrainrun) - Restore complete trainrun
- [SavePipeline](#savepipeline) - Save pipeline only

---

### RestoreTrainRun

**Purpose:** Restore complete training run (pipeline + data + training + weights).

**Request:**
```protobuf
message RestoreTrainRunRequest {
  string trainrun_path = 1;  // Path to trainrun YAML
  string weights_path = 2;   // Optional custom weights path
  bool strict = 3;           // Strict weight loading (default: true)
}
```

**Response:**
```protobuf
message RestoreTrainRunResponse {
  string session_id = 1;  // NEW session created automatically
  bool success = 2;
}
```

**Python Example:**
```python
# Restore trainrun (creates new session automatically)
response = stub.RestoreTrainRun(
    cuvis_ai_pb2.RestoreTrainRunRequest(
        trainrun_path="outputs/deep_svdd_run.yaml",
        weights_path="outputs/deep_svdd_run.pt",  # Optional override
        strict=True,
    )
)

session_id = response.session_id
print(f"TrainRun restored in session: {session_id}")

# Now ready for inference or continued training
inference_response = stub.Inference(
    cuvis_ai_pb2.InferenceRequest(session_id=session_id, inputs=...)
)
```

**Key Feature: Automatic Session Creation**
- `RestoreTrainRun` **creates a new session automatically**
- You don't need to call `CreateSession` first
- Returns the new `session_id` in response
- Session has pipeline + weights + configs fully loaded

**Notes:**
- Most convenient way to restore trained models
- If `weights_path` not specified, looks for `.pt` file next to `.yaml`
- Use `strict=False` for partial weight loading
- Remember to `CloseSession` when done

**See Also:**
- [SaveTrainRun](#savetrainrun) - Save trainrun for restoration
- [Restore Pipeline Guide](../how-to/restore-pipeline-trainrun.md)

---

## Training Operations

Execute statistical and gradient-based training with streaming progress updates.

### Train

**Purpose:** Execute training with streaming progress updates (statistical or gradient).

**Request:**
```protobuf
message TrainRequest {
  string session_id = 1;
  TrainerType trainer_type = 2;  // STATISTICAL or GRADIENT
}

enum TrainerType {
  TRAINER_TYPE_STATISTICAL = 0;
  TRAINER_TYPE_GRADIENT = 1;
}
```

**Response (Server-Side Streaming):**
```protobuf
message TrainResponse {
  ExecutionContext context = 1;      // Epoch, step, stage info
  TrainStatus status = 2;            // RUNNING, COMPLETED, FAILED
  map<string, float> losses = 3;     // Loss values
  map<string, float> metrics = 4;    // Metric values
  string message = 5;                // Progress message
}
```

**Python Example - Statistical Training:**
```python
# Phase 1: Statistical initialization (short, no backprop)
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
):
    stage = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
    status = cuvis_ai_pb2.TrainStatus.Name(progress.status)
    print(f"[{stage}] {status}: {progress.message}")
```

**Python Example - Gradient Training:**
```python
# Phase 2: Gradient-based training (full backprop)
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
    )
):
    stage = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
    status = cuvis_ai_pb2.TrainStatus.Name(progress.status)

    if progress.losses:
        print(f"[{stage}] Epoch {progress.context.epoch} | losses={dict(progress.losses)}")
    if progress.metrics:
        print(f"[{stage}] Epoch {progress.context.epoch} | metrics={dict(progress.metrics)}")
```

**Two-Phase Training Pattern:**
```python
# 1. Statistical initialization (RX, normalization stats, etc.)
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
):
    print(f"[statistical] {format_progress(progress)}")

# 2. Gradient-based fine-tuning (deep learning)
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
    )
):
    print(f"[gradient] {format_progress(progress)}")
```

**Execution Stages:**
- `EXECUTION_STAGE_TRAIN` - Training loop
- `EXECUTION_STAGE_VAL` - Validation loop
- `EXECUTION_STAGE_TEST` - Test loop

**Train Status:**
- `TRAIN_STATUS_RUNNING` - Training in progress
- `TRAIN_STATUS_COMPLETED` - Training finished successfully
- `TRAIN_STATUS_FAILED` - Training failed with error

**Helper Function:**
```python
from examples.grpc.workflow_utils import format_progress

for progress in stub.Train(...):
    print(format_progress(progress))
# Output: "[TRAIN] RUNNING | losses={'total': 0.42} | metrics={'iou': 0.85}"
```

**Notes:**
- Training is **server-side streaming** (client receives updates as they occur)
- No polling required (updates pushed in real-time)
- Process stream with for-loop (blocks until training completes)
- Statistical training is typically fast (1-2 passes over data)
- Gradient training duration depends on `max_epochs` in training config

**See Also:**
- [GetTrainStatus](#gettrainstatus) - Query training status
- [Sequence Diagrams](sequence-diagrams.md#2-two-phase-training-workflow)

---

### GetTrainStatus

**Purpose:** Query current training status for a session.

**Request:**
```protobuf
message GetTrainStatusRequest {
  string session_id = 1;
}
```

**Response:**
```protobuf
message GetTrainStatusResponse {
  TrainStatus status = 1;  // Current status
  string message = 2;      // Status message
}
```

**Python Example:**
```python
response = stub.GetTrainStatus(
    cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id)
)

status = cuvis_ai_pb2.TrainStatus.Name(response.status)
print(f"Training status: {status}")
if response.message:
    print(f"Message: {response.message}")
```

**Use Cases:**
- Checking if training is complete before inference
- Monitoring training from separate process
- Debugging training issues

**Notes:**
- Returns last known status (may be stale if training just started)
- For real-time updates, use `Train` streaming instead
- Status persists in session until next training call

---

## Inference Operations

Run predictions on trained pipelines.

### Inference

**Purpose:** Run predictions on trained pipeline with optional output filtering.

**Request:**
```protobuf
message InferenceRequest {
  string session_id = 1;
  InputBatch inputs = 2;
  repeated string output_specs = 3;  // Optional: filter outputs
}
```

**Response:**
```protobuf
message InferenceResponse {
  map<string, Tensor> outputs = 1;  // Output tensors by name
  map<string, float> metrics = 2;   // Optional metrics
}
```

**Python Example - Basic Inference:**
```python
from cuvis_ai_core.grpc import helpers
import numpy as np

# Prepare input data
cube = np.random.rand(1, 32, 32, 61).astype(np.float32)
wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.float32)

# Run inference
response = stub.Inference(
    cuvis_ai_pb2.InferenceRequest(
        session_id=session_id,
        inputs=cuvis_ai_pb2.InputBatch(
            cube=helpers.numpy_to_proto(cube),
            wavelengths=helpers.numpy_to_proto(wavelengths),
        ),
    )
)

# Process outputs
for name, tensor_proto in response.outputs.items():
    output_array = helpers.proto_to_numpy(tensor_proto)
    print(f"{name}: shape={output_array.shape}, dtype={output_array.dtype}")
```

**Python Example - Output Filtering:**
```python
# Request only specific outputs (reduces payload size)
response = stub.Inference(
    cuvis_ai_pb2.InferenceRequest(
        session_id=session_id,
        inputs=cuvis_ai_pb2.InputBatch(
            cube=helpers.numpy_to_proto(cube),
            wavelengths=helpers.numpy_to_proto(wavelengths),
        ),
        output_specs=[
            "selector.selected",   # Only selected channels
            "detector.scores",     # Anomaly scores
            "decider.decisions",   # Final decisions
        ],
    )
)

selected = helpers.proto_to_numpy(response.outputs["selector.selected"])
scores = helpers.proto_to_numpy(response.outputs["detector.scores"])
decisions = helpers.proto_to_numpy(response.outputs["decider.decisions"])
```

**Python Example - Complex Input Types:**
```python
# Inference with bounding boxes and points (e.g., SAM integration)
response = stub.Inference(
    cuvis_ai_pb2.InferenceRequest(
        session_id=session_id,
        inputs=cuvis_ai_pb2.InputBatch(
            cube=helpers.numpy_to_proto(cube),
            wavelengths=helpers.numpy_to_proto(wavelengths),
            bboxes=cuvis_ai_pb2.BoundingBoxes(
                boxes=[
                    cuvis_ai_pb2.BoundingBox(
                        element_id=0,
                        x_min=10, y_min=10,
                        x_max=20, y_max=20,
                    )
                ]
            ),
            points=cuvis_ai_pb2.Points(
                points=[
                    cuvis_ai_pb2.Point(
                        element_id=0,
                        x=15.5, y=15.5,
                        type=cuvis_ai_pb2.POINT_TYPE_POSITIVE,
                    )
                ]
            ),
            text_prompt="Find anomalies in lentils",
        ),
    )
)
```

**Output Filtering Benefits:**
- Reduces network payload (important for large tensors)
- Faster response time (server skips unused computations)
- Lower memory usage on client
- Use when you only need subset of pipeline outputs

**Notes:**
- Pipeline must be loaded and trained (or weights loaded) first
- `InputBatch` supports: cube, wavelengths, mask, bboxes, points, text_prompt
- Output filtering is optional (omit `output_specs` to get all outputs)
- Use `helpers.numpy_to_proto()` and `helpers.proto_to_numpy()` for conversions

**See Also:**
- [InputBatch Data Type](#inputbatch)
- [GetPipelineInputs](#getpipelineinputs) - Query required inputs
- [GetPipelineOutputs](#getpipelineoutputs) - Query available outputs

---

## Introspection & Discovery

Query pipeline capabilities, inspect structure, and visualize graphs.

### GetPipelineInputs

**Purpose:** Get input tensor specifications for current pipeline.

**Request:**
```protobuf
message GetPipelineInputsRequest {
  string session_id = 1;
}
```

**Response:**
```protobuf
message GetPipelineInputsResponse {
  repeated string input_names = 1;             // Input port names
  map<string, TensorSpec> input_specs = 2;    // Specs by name
}
```

**Python Example:**
```python
response = stub.GetPipelineInputs(
    cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
)

print("Pipeline inputs:")
for name in response.input_names:
    spec = response.input_specs[name]
    shape = list(spec.shape)
    dtype = cuvis_ai_pb2.DType.Name(spec.dtype)
    required = "required" if spec.required else "optional"
    print(f"  {name}: shape={shape}, dtype={dtype}, {required}")
```

**Example Output:**
```
Pipeline inputs:
  cube: shape=[1, -1, -1, 61], dtype=D_TYPE_FLOAT32, required
  wavelengths: shape=[1, 61], dtype=D_TYPE_FLOAT32, required
  mask: shape=[1, -1, -1], dtype=D_TYPE_UINT8, optional
```

**Notes:**
- Shape dimensions of `-1` indicate dynamic sizes
- `required=true` means input must be provided for inference
- `required=false` means input is optional
- Pipeline must be loaded first

**See Also:**
- [GetPipelineOutputs](#getpipelineoutputs)
- [TensorSpec Data Type](#tensorspec)

---

### GetPipelineOutputs

**Purpose:** Get output tensor specifications for current pipeline.

**Request:**
```protobuf
message GetPipelineOutputsRequest {
  string session_id = 1;
}
```

**Response:**
```protobuf
message GetPipelineOutputsResponse {
  repeated string output_names = 1;            // Output port names
  map<string, TensorSpec> output_specs = 2;   // Specs by name
}
```

**Python Example:**
```python
response = stub.GetPipelineOutputs(
    cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
)

print("Pipeline outputs:")
for name in response.output_names:
    spec = response.output_specs[name]
    shape = list(spec.shape)
    dtype = cuvis_ai_pb2.DType.Name(spec.dtype)
    print(f"  {name}: shape={shape}, dtype={dtype}")
```

**Example Output:**
```
Pipeline outputs:
  selector.selected: shape=[1, -1, -1, 4], dtype=D_TYPE_FLOAT32
  detector.scores: shape=[1, -1, -1], dtype=D_TYPE_FLOAT32
  decider.decisions: shape=[1, -1, -1], dtype=D_TYPE_UINT8
```

**Use Cases:**
- Discovering available outputs before inference
- Validating pipeline structure
- Generating documentation
- Dynamic UI generation

---

### GetPipelineVisualization

**Purpose:** Render pipeline graph as image or text format.

**Request:**
```protobuf
message GetPipelineVisualizationRequest {
  string session_id = 1;
  string format = 2;  // "png", "svg", "dot", "mermaid"
}
```

**Response:**
```protobuf
message GetPipelineVisualizationResponse {
  bytes image_data = 1;  // Binary image data or text
  string format = 2;     // Confirmed format
}
```

**Python Example - PNG:**
```python
from pathlib import Path

response = stub.GetPipelineVisualization(
    cuvis_ai_pb2.GetPipelineVisualizationRequest(
        session_id=session_id,
        format="png",
    )
)

Path("pipeline_graph.png").write_bytes(response.image_data)
print("Pipeline visualization saved to pipeline_graph.png")
```

**Python Example - Mermaid:**
```python
response = stub.GetPipelineVisualization(
    cuvis_ai_pb2.GetPipelineVisualizationRequest(
        session_id=session_id,
        format="mermaid",
    )
)

mermaid_text = response.image_data.decode("utf-8")
print("Pipeline in Mermaid format:")
print(mermaid_text)
```

**Supported Formats:**
- `"png"` - PNG image (binary)
- `"svg"` - SVG image (text/XML)
- `"dot"` - Graphviz DOT format (text)
- `"mermaid"` - Mermaid diagram (text)

**Use Cases:**
- Documentation generation
- Debugging pipeline structure
- Presenting architecture to stakeholders
- Automated diagram generation in CI/CD

---

### ListAvailablePipelines

**Purpose:** Discover registered pipelines available for loading.

**Request:**
```protobuf
message ListAvailablePipelinesRequest {
  string filter_tag = 1;  // Optional tag filter
}
```

**Response:**
```protobuf
message ListAvailablePipelinesResponse {
  repeated PipelineInfo pipelines = 1;
}

message PipelineInfo {
  string name = 1;
  PipelineMetadata metadata = 2;
  repeated string tags = 3;
}
```

**Python Example:**
```python
response = stub.ListAvailablePipelines(
    cuvis_ai_pb2.ListAvailablePipelinesRequest(
        filter_tag="anomaly_detection",  # Optional filter
    )
)

print("Available pipelines:")
for pipeline in response.pipelines:
    print(f"  - {pipeline.name}")
    print(f"    Description: {pipeline.metadata.description}")
    print(f"    Tags: {', '.join(pipeline.tags)}")
```

**Use Cases:**
- Pipeline discovery for users
- Dynamic pipeline selection in applications
- Catalog generation

---

### GetPipelineInfo

**Purpose:** Get detailed metadata for a specific pipeline.

**Request:**
```protobuf
message GetPipelineInfoRequest {
  string pipeline_name = 1;
}
```

**Response:**
```protobuf
message GetPipelineInfoResponse {
  PipelineInfo info = 1;
  map<string, TensorSpec> required_inputs = 2;
  map<string, TensorSpec> outputs = 3;
}
```

**Python Example:**
```python
response = stub.GetPipelineInfo(
    cuvis_ai_pb2.GetPipelineInfoRequest(
        pipeline_name="deep_svdd_anomaly"
    )
)

print(f"Pipeline: {response.info.name}")
print(f"Description: {response.info.metadata.description}")
print(f"Required inputs: {list(response.required_inputs.keys())}")
print(f"Outputs: {list(response.outputs.keys())}")
```

---

## Error Handling

All gRPC RPCs use standard gRPC status codes for error handling.

### Status Codes

```python
import grpc

try:
    response = stub.Inference(request)
except grpc.RpcError as e:
    code = e.code()
    details = e.details()

    if code == grpc.StatusCode.INVALID_ARGUMENT:
        print(f"Invalid request: {details}")
    elif code == grpc.StatusCode.NOT_FOUND:
        print(f"Resource not found: {details}")
    elif code == grpc.StatusCode.FAILED_PRECONDITION:
        print(f"Operation not allowed: {details}")
    elif code == grpc.StatusCode.RESOURCE_EXHAUSTED:
        print(f"Resource exhausted: {details}")
    elif code == grpc.StatusCode.INTERNAL:
        print(f"Internal server error: {details}")
    else:
        print(f"gRPC error: {code} - {details}")
        raise
```

### Common Error Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| `INVALID_ARGUMENT` | Malformed request | Missing required fields, invalid types |
| `NOT_FOUND` | Resource not found | Session ID doesn't exist, expired session |
| `FAILED_PRECONDITION` | Operation not allowed | Training before config set, inference before weights loaded |
| `RESOURCE_EXHAUSTED` | Resource limit exceeded | Message size exceeded, GPU out of memory |
| `INTERNAL` | Server error | Unexpected exception, configuration bug |
| `UNAVAILABLE` | Service unavailable | Server down, network issue |
| `DEADLINE_EXCEEDED` | Operation timeout | Long training, slow network |

### Error Examples

**Session Not Found:**
```
grpc.StatusCode.NOT_FOUND
"Session '7f3e4d2c-1a9b-4c8d-9e7f-2b5a6c8d9e0f' not found or expired"
```

**Message Size Exceeded:**
```
grpc.StatusCode.RESOURCE_EXHAUSTED
"Received message larger than max (100000000 vs. 4194304)"
Solution: Increase client/server message size limits
```

**Missing Config:**
```
grpc.StatusCode.FAILED_PRECONDITION
"Cannot train: trainrun config not set. Call SetTrainRunConfig first."
```

**CUDA Out of Memory:**
```
grpc.StatusCode.RESOURCE_EXHAUSTED
"CUDA out of memory. Tried to allocate 2.00 GiB"
Solution: Reduce batch size, close unused sessions
```

---

## Data Types Reference

### InputBatch

Complete input specification for inference.

```protobuf
message InputBatch {
  Tensor cube = 1;              // Required: [B, H, W, C] hyperspectral cube
  Tensor wavelengths = 2;       // Optional: [B, C] wavelengths in nm
  Tensor mask = 3;              // Optional: [B, H, W] binary mask
  BoundingBoxes bboxes = 4;     // Optional: Bounding boxes
  Points points = 5;            // Optional: Point prompts
  string text_prompt = 6;       // Optional: Text description
  map<string, Tensor> extra_inputs = 7;  // Optional: Additional inputs
}
```

**Example:**
```python
inputs = cuvis_ai_pb2.InputBatch(
    cube=helpers.numpy_to_proto(cube),                    # Required
    wavelengths=helpers.numpy_to_proto(wavelengths),      # Optional
    mask=helpers.numpy_to_proto(mask),                    # Optional
    bboxes=cuvis_ai_pb2.BoundingBoxes(boxes=[...]),       # Optional
    points=cuvis_ai_pb2.Points(points=[...]),             # Optional
    text_prompt="Find anomalies",                         # Optional
)
```

---

### Tensor

Protocol Buffer tensor representation.

```protobuf
message Tensor {
  repeated int64 shape = 1;  // Tensor dimensions
  DType dtype = 2;           // Data type
  bytes raw_data = 3;        // Raw binary data
}

enum DType {
  D_TYPE_FLOAT32 = 0;
  D_TYPE_FLOAT64 = 1;
  D_TYPE_INT32 = 2;
  D_TYPE_INT64 = 3;
  D_TYPE_UINT8 = 4;
  D_TYPE_UINT16 = 5;
  // ...
}
```

**Conversion Helpers:**
```python
from cuvis_ai_core.grpc import helpers

# NumPy array to Tensor
array = np.random.rand(1, 32, 32, 61).astype(np.float32)
tensor_proto = helpers.numpy_to_proto(array)

# Tensor to NumPy array
array_recovered = helpers.proto_to_numpy(tensor_proto)

# PyTorch tensor to Tensor
tensor = torch.randn(1, 32, 32, 61)
tensor_proto = helpers.tensor_to_proto(tensor)
```

---

### TensorSpec

Tensor specification (metadata without data).

```protobuf
message TensorSpec {
  string name = 1;           // Tensor name
  repeated int64 shape = 2;  // Shape (-1 for dynamic dimensions)
  DType dtype = 3;           // Data type
  bool required = 4;         // Required for inference
}
```

**Example:**
```
name: "cube"
shape: [1, -1, -1, 61]  // Batch=1, Height/Width dynamic, Channels=61
dtype: D_TYPE_FLOAT32
required: true
```

---

### BoundingBoxes

Bounding box collection for object detection or SAM integration.

```protobuf
message BoundingBoxes {
  repeated BoundingBox boxes = 1;
}

message BoundingBox {
  int32 element_id = 1;  // Batch element index
  float x_min = 2;
  float y_min = 3;
  float x_max = 4;
  float y_max = 5;
}
```

**Example:**
```python
bboxes = cuvis_ai_pb2.BoundingBoxes(
    boxes=[
        cuvis_ai_pb2.BoundingBox(
            element_id=0,
            x_min=10.0, y_min=10.0,
            x_max=20.0, y_max=20.0,
        ),
        cuvis_ai_pb2.BoundingBox(
            element_id=0,
            x_min=30.0, y_min=30.0,
            x_max=40.0, y_max=40.0,
        ),
    ]
)
```

---

### Points

Point prompts for interactive segmentation (SAM-style).

```protobuf
message Points {
  repeated Point points = 1;
}

message Point {
  int32 element_id = 1;   // Batch element index
  float x = 2;
  float y = 3;
  PointType type = 4;     // POSITIVE or NEGATIVE
}

enum PointType {
  POINT_TYPE_POSITIVE = 0;
  POINT_TYPE_NEGATIVE = 1;
}
```

**Example:**
```python
points = cuvis_ai_pb2.Points(
    points=[
        cuvis_ai_pb2.Point(
            element_id=0,
            x=15.5, y=15.5,
            type=cuvis_ai_pb2.POINT_TYPE_POSITIVE,
        ),
        cuvis_ai_pb2.Point(
            element_id=0,
            x=25.5, y=25.5,
            type=cuvis_ai_pb2.POINT_TYPE_NEGATIVE,
        ),
    ]
)
```

---

## See Also

### Related Documentation
- [gRPC Overview](overview.md) - Introduction and architecture
- [Client Patterns](client-patterns.md) - Common usage patterns and best practices
- [Sequence Diagrams](sequence-diagrams.md) - Visual workflows

### Tutorials & Guides
- [gRPC Workflow Tutorial](../tutorials/grpc-workflow.md) - Hands-on tutorial
- [Remote gRPC Access](../how-to/remote-grpc.md) - Detailed how-to guide

### Configuration
- [TrainRun Schema](../config/trainrun-schema.md) - Trainrun configuration reference
- [Pipeline Schema](../config/pipeline-schema.md) - Pipeline YAML schema
- [Hydra Composition](../config/hydra-composition.md) - Config composition patterns

### Deployment
- [Deployment Guide](../deployment/grpc_deployment.md) - Docker, Kubernetes, production
