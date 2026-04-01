!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Training & Inference API

Training, inference, and profiling operations for the `CuvisAIService`.

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
from cuvis_ai.utils.grpc_workflow import format_progress

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

- [InputBatch Data Type](api-types-errors.md#inputbatch)
- [GetPipelineInputs](api-types-errors.md#getpipelineinputs) - Query required inputs
- [GetPipelineOutputs](api-types-errors.md#getpipelineoutputs) - Query available outputs

---

## Profiling

Enable runtime profiling and retrieve per-node timing statistics. For a comprehensive guide on profiling workflows, see [Profiling & Performance](../how-to/profiling.md).

### SetProfiling

**Purpose:** Enable, disable, or reconfigure per-node runtime profiling on a session's pipeline.

**Request:**
```protobuf
message SetProfilingRequest {
  string session_id = 1;
  bool enabled = 2;
  optional bool synchronize_cuda = 3;
  optional bool reset = 4;
  optional int32 skip_first_n = 5;
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | Target session ID |
| `enabled` | `bool` | Activate or deactivate profiling |
| `synchronize_cuda` | `bool` (optional) | If `true`, call `torch.cuda.synchronize` before/after each node for accurate GPU timing. Default: `false` |
| `reset` | `bool` (optional) | If `true`, discard all previously accumulated statistics. Default: `false` |
| `skip_first_n` | `int32` (optional) | Number of initial samples per node to discard (warm-up). Default: `0`. Must be >= 0 |

**Response:**
```protobuf
message SetProfilingResponse {
  bool profiling_enabled = 1;
}
```

**Full-replace semantics:** Every call fully specifies the configuration. Omitted optional fields fall through to defaults — calling `SetProfiling(enabled=True)` without `synchronize_cuda` sets it to `false`, not the previous value.

**Example:**
```python
# Enable profiling with CUDA sync and warm-up skip
stub.SetProfiling(cuvis_ai_pb2.SetProfilingRequest(
    session_id=session_id,
    enabled=True,
    synchronize_cuda=True,
    skip_first_n=3,
))

# Run inference...

# Reset stats for a fresh measurement
stub.SetProfiling(cuvis_ai_pb2.SetProfilingRequest(
    session_id=session_id,
    enabled=True,
    reset=True,
))
```

**Error Codes:**

| Code | Condition |
|------|-----------|
| `NOT_FOUND` | Invalid `session_id` |
| `FAILED_PRECONDITION` | No pipeline loaded in session |
| `INVALID_ARGUMENT` | Negative `skip_first_n` |

### GetProfilingSummary

**Purpose:** Retrieve accumulated per-node profiling statistics.

**Request:**
```protobuf
message GetProfilingSummaryRequest {
  string session_id = 1;
  optional ExecutionStage stage = 2;
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | Target session ID |
| `stage` | `ExecutionStage` (optional) | Filter by execution stage. Omit to retrieve all stages |

**Response:**
```protobuf
message GetProfilingSummaryResponse {
  repeated NodeProfilingStats node_stats = 1;
}

message NodeProfilingStats {
  string node_name = 1;
  ExecutionStage stage = 2;
  int64 count = 3;
  double mean_ms = 4;
  double median_ms = 5;
  double std_ms = 6;
  double min_ms = 7;
  double max_ms = 8;
  double total_ms = 9;
  double last_ms = 10;
}
```

**Example:**
```python
# Get all profiling stats
response = stub.GetProfilingSummary(
    cuvis_ai_pb2.GetProfilingSummaryRequest(session_id=session_id)
)
for stat in response.node_stats:
    print(f"{stat.node_name} ({stat.stage}): "
          f"mean={stat.mean_ms:.2f}ms, count={stat.count}")

# Get only inference stage stats
response = stub.GetProfilingSummary(
    cuvis_ai_pb2.GetProfilingSummaryRequest(
        session_id=session_id,
        stage=cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE,
    )
)
```

**Error Codes:**

| Code | Condition |
|------|-----------|
| `NOT_FOUND` | Invalid `session_id` |
| `FAILED_PRECONDITION` | No pipeline loaded in session |

---

## See Also

- [gRPC API Reference](api-session.md) - Session, configuration, and pipeline management
- [Types & Errors API](api-types-errors.md) - Introspection, error handling, and data types
- [Client Workflows & Error Handling](client-workflows.md) - Training, inference, and error handling patterns
- [Sequence Diagrams](sequence-diagrams.md) - Visual workflows
