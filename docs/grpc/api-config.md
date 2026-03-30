!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Configuration Management API

The configuration service integrates with Hydra for powerful config composition, validation, and dynamic overrides.

---

## ResolveConfig

**Purpose:** Resolve configuration using Hydra composition with optional overrides.

**Request:**
```protobuf
message ResolveConfigRequest {
  string session_id = 1;
  string config_type = 2;        // "trainrun", "pipeline", "training", "data"
  string path = 3;                // Relative path in search paths or an absolute server path
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
    "pipeline.nodes.channel_selector.hparams.tau_start=8.0",
    "pipeline.nodes.rx_detector.hparams.eps=1e-6",
    "pipeline.nodes.normalizer.hparams.use_running_stats=true",
]
```

**Notes:**

- Requires `SetSessionSearchPaths` to be called first
- Returns JSON bytes (decode with `.decode("utf-8")` and parse with `json.loads()`)
- Hydra resolves config group composition, interpolations, and overrides
- Override syntax follows Hydra conventions (dot notation for nested fields)

**Helper Function:**
```python
from cuvis_ai.utils.grpc_workflow import resolve_trainrun_config

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
- [Hydra Basics](../config/hydra-basics.md)
- [TrainRun Schema](../config/trainrun-schema.md)

---

## SetTrainRunConfig

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
from cuvis_ai.utils.grpc_workflow import apply_trainrun_config

apply_trainrun_config(stub, session_id, resolved.config_bytes)
```

**See Also:**

- [ResolveConfig](#resolveconfig) - Resolve config first
- [Train](api-training-inference.md#train) - Execute training after config applied

---

## ValidateConfig

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

**Common Validation Errors:**

- Missing required fields (e.g., `optimizer.lr`)
- Invalid values (e.g., negative `max_epochs`)
- Type mismatches (e.g., string for numeric field)
- Unknown optimizer/scheduler names
- Incompatible node connections in pipeline

**See Also:**

- [SetTrainRunConfig](#settrainrunconfig) - Apply config after validation
- [TrainRun Schema](../config/trainrun-schema.md)

---

## GetTrainingCapabilities

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
