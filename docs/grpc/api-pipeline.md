!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Pipeline Management API

Manage pipeline loading, saving, and restoration.

---

## LoadPipeline

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
**See Also:**

- [LoadPipelineWeights](#loadpipelineweights) - Load trained weights
- [SavePipeline](#savepipeline) - Save pipeline config + weights

---

## LoadPipelineWeights

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

**See Also:**

- [LoadPipeline](#loadpipeline) - Load pipeline config first
- [SavePipeline](#savepipeline) - Save weights

---

## SavePipeline

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

## SaveTrainRun

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

**TrainRun vs Pipeline:**
| Feature | SavePipeline | SaveTrainRun |
|---------|--------------|--------------|
| Pipeline config | :white_check_mark: | :white_check_mark: |
| Weights | :white_check_mark: | :white_check_mark: |
| Data config | :x: | :white_check_mark: |
| Training config | :x: | :white_check_mark: |
| **Use for** | Inference deployment | Reproducibility, resume training |

---

## RestoreTrainRun

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
