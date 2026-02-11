!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Remote gRPC Access

Access CUVIS.AI pipelines remotely using gRPC for distributed training, inference, and deployment.

!!! info "Comprehensive gRPC Documentation"
    For complete gRPC documentation, see:

    - [gRPC Overview](../grpc/overview.md) - Introduction and architecture
    - [gRPC API Reference](../grpc/api-reference.md) - Complete documentation of all 46 RPC methods
    - [Client Patterns](../grpc/client-patterns.md) - Best practices and common patterns

    This guide focuses on detailed examples and deployment scenarios.

## Overview

CUVIS.AI provides comprehensive gRPC infrastructure for remote pipeline execution:

- **Server Deployment**: Production-ready server with Docker/Kubernetes support
- **Client SDK**: Simple API for training, inference, and configuration
- **Session Management**: Isolated execution contexts with automatic cleanup
- **Streaming Updates**: Real-time training progress without polling
- **Configuration Composition**: Hydra integration with dynamic overrides
- **Error Handling**: Standard gRPC error codes with retry logic

---

## Quick Start

### Server Setup

**Start server locally:**
```bash
uv run python -m cuvis_ai.grpc.production_server
```

**Start server with Docker:**
```bash
docker-compose up
```

**Default connection:** `localhost:50051`

### Basic Client Usage

```python
from cuvis_ai_core.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc
import grpc

# Connect to server
channel = grpc.insecure_channel("localhost:50051")
stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

# Create session
response = stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
session_id = response.session_id

# Run inference (example)
inference_response = stub.Inference(
    cuvis_ai_pb2.InferenceRequest(
        session_id=session_id,
        inputs=cuvis_ai_pb2.InputBatch(cube=...),
    )
)

# Clean up
stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
```

---

## gRPC Server Setup

### Local Development

**Launch server:**
```bash
uv run python -m cuvis_ai.grpc.production_server
```

**Default configuration:**
- Host: `0.0.0.0` (all network interfaces)
- Port: `50051`
- Max workers: `10` (thread pool)
- Max message size: `300 MB`

**Environment variables:**
```bash
export GRPC_PORT=50051
export GRPC_MAX_WORKERS=10
export LOG_LEVEL=INFO
export DATA_DIR=/path/to/data
```

### Docker Deployment

**Dockerfile** (production-ready):
```dockerfile
FROM cubertgmbh/cuvis_python:3.4.1-ubuntu24.04

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application
COPY cuvis_ai/ cuvis_ai/
COPY configs/ configs/

EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import grpc; ..."

CMD ["python", "-m", "cuvis_ai.grpc.production_server"]
```

**Build and run:**
```bash
docker build -t cuvis-ai-server .
docker run -p 50051:50051 --gpus all cuvis-ai-server
```

### Docker Compose

**docker-compose.yml:**
```yaml
version: "3.8"
services:
  cuvis-ai-server:
    build: .
    ports:
      - "50051:50051"
    environment:
      - GRPC_PORT=50051
      - GRPC_MAX_WORKERS=10
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Start services:**
```bash
docker-compose up
```

### Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cuvis-ai-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cuvis-ai-server
  template:
    metadata:
      labels:
        app: cuvis-ai-server
    spec:
      containers:
      - name: cuvis-ai-server
        image: cuvis-ai-server:latest
        ports:
        - containerPort: 50051
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
          requests:
            memory: 4Gi
---
apiVersion: v1
kind: Service
metadata:
  name: cuvis-ai-service
spec:
  selector:
    app: cuvis-ai-server
  ports:
  - protocol: TCP
    port: 50051
    targetPort: 50051
  type: LoadBalancer
```

**Deploy:**
```bash
kubectl apply -f deployment.yaml
kubectl get service cuvis-ai-service
```

### Production Configuration

**TLS/SSL (secure channel):**
```python
import grpc

# Server-side
credentials = grpc.ssl_server_credentials([
    (open("server.key", "rb").read(),
     open("server.crt", "rb").read())
])
server.add_secure_port("[::]:50051", credentials)

# Client-side
channel = grpc.secure_channel(
    "production-server:50051",
    grpc.ssl_channel_credentials(),
)
```

**Message size limits:**
```python
import grpc
from concurrent import futures

# Server
server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=10),
    options=[
        ("grpc.max_send_message_length", 1024 * 1024 * 1024),  # 1 GB
        ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
    ],
)

# Client
options = [
    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
]
channel = grpc.insecure_channel("localhost:50051", options=options)
```

---

## gRPC Client Usage

### Helper Utilities

**Recommended approach:** Use helper functions from `workflow_utils.py`:

**File:** `examples/grpc/workflow_utils.py`

```python
from cuvis_ai_core.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc
import grpc
from pathlib import Path

def build_stub(
    server_address: str = "localhost:50051",
    max_msg_size: int = 300 * 1024 * 1024
) -> cuvis_ai_pb2_grpc.CuvisAIServiceStub:
    """Create a gRPC stub with configured message limits."""
    options = [
        ("grpc.max_send_message_length", max_msg_size),
        ("grpc.max_receive_message_length", max_msg_size),
    ]
    channel = grpc.insecure_channel(server_address, options=options)
    return cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

def config_search_paths(
    extra_paths: list[str] | None = None
) -> list[str]:
    """Return absolute search paths for Hydra composition."""
    config_root = Path(__file__).parent.parent / "configs"
    seeds = [
        config_root,
        config_root / "trainrun",
        config_root / "pipeline",
        config_root / "data",
        config_root / "training",
    ]
    paths = [str(p.resolve()) for p in seeds]
    if extra_paths:
        paths.extend(extra_paths)
    return paths

def create_session_with_search_paths(
    stub: cuvis_ai_pb2_grpc.CuvisAIServiceStub,
    search_paths: list[str] | None = None
) -> str:
    """Create session and register config search paths."""
    response = stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
    session_id = response.session_id

    paths = search_paths or config_search_paths()
    stub.SetSessionSearchPaths(
        cuvis_ai_pb2.SetSessionSearchPathsRequest(
            session_id=session_id,
            search_paths=paths,
            append=False,
        )
    )
    return session_id

def resolve_trainrun_config(
    stub: cuvis_ai_pb2_grpc.CuvisAIServiceStub,
    session_id: str,
    name: str,
    overrides: list[str] | None = None,
):
    """Resolve trainrun config via Hydra composition."""
    import json
    response = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="trainrun",
            path=f"trainrun/{name}",
            overrides=overrides or [],
        )
    )
    config_dict = json.loads(response.config_bytes.decode("utf-8"))
    return response, config_dict

def apply_trainrun_config(
    stub: cuvis_ai_pb2_grpc.CuvisAIServiceStub,
    session_id: str,
    config_bytes: bytes,
):
    """Apply resolved trainrun config to session."""
    stub.SetTrainRunConfig(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=cuvis_ai_pb2.TrainRunConfig(config_bytes=config_bytes),
        )
    )
```

### Session Management

**Create session:**
```python
from workflow_utils import build_stub, create_session_with_search_paths

stub = build_stub("localhost:50051")
session_id = create_session_with_search_paths(stub)
```

**Close session:**
```python
stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
```

**Session lifecycle:**
- Sessions are isolated (separate pipelines, configs, weights)
- Sessions expire after 1 hour of inactivity
- Always close sessions when done to free resources

### Configuration Resolution

**Resolve config with overrides:**
```python
from workflow_utils import resolve_trainrun_config, apply_trainrun_config

# Resolve trainrun config
resolved, config_dict = resolve_trainrun_config(
    stub,
    session_id,
    "rx_statistical",  # trainrun name
    overrides=[
        "data.batch_size=4",
        "training.trainer.max_epochs=10",
        "training.optimizer.lr=0.001",
    ],
)

# Apply resolved config
apply_trainrun_config(stub, session_id, resolved.config_bytes)
```

**Available override patterns:**
```python
overrides = [
    # Training config
    "training.trainer.max_epochs=100",
    "training.optimizer.lr=0.001",
    "training.optimizer.weight_decay=0.01",
    "training.scheduler.patience=10",

    # Data config
    "data.batch_size=16",
    "data.train_ids=[0,1,2]",
    "data.val_ids=[3,4]",

    # Pipeline node params
    "pipeline.nodes.channel_selector.params.tau_start=8.0",
    "pipeline.nodes.rx_detector.params.eps=1e-6",
]
```

### Training Operations

#### Statistical Training

```python
# Run statistical training (streaming progress)
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
):
    stage = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
    status = cuvis_ai_pb2.TrainStatus.Name(progress.status)
    epoch = progress.context.epoch
    batch = progress.context.batch_idx

    print(f"[{stage}] {status} | epoch={epoch} batch={batch}")

    if progress.metrics:
        metrics = dict(progress.metrics)
        print(f"  Metrics: {metrics}")
```

#### Gradient Training

```python
# Run gradient training
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
    )
):
    stage = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
    status = cuvis_ai_pb2.TrainStatus.Name(progress.status)

    losses = dict(progress.losses) if progress.losses else {}
    metrics = dict(progress.metrics) if progress.metrics else {}

    print(f"[{stage}] {status} | losses={losses} | metrics={metrics}")
```

#### Two-Phase Training

```python
# Phase 1: Statistical initialization
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
):
    print(f"[Statistical Init] {format_progress(progress)}")

# Phase 2: Gradient training
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
    )
):
    print(f"[Gradient Training] {format_progress(progress)}")
```

### Inference Operations

**Basic inference:**
```python
from cuvis_ai_core.grpc import helpers
import numpy as np

# Prepare inputs
cube = np.random.rand(1, 32, 32, 61).astype(np.float32)
wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)

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

# Convert outputs
results = {
    name: helpers.proto_to_numpy(tensor_proto)
    for name, tensor_proto in response.outputs.items()
}

print(f"Outputs: {list(results.keys())}")
print(f"Decisions shape: {results['decider.decisions'].shape}")
```

**Inference with output filtering:**
```python
# Request specific outputs only
response = stub.Inference(
    cuvis_ai_pb2.InferenceRequest(
        session_id=session_id,
        inputs=cuvis_ai_pb2.InputBatch(cube=..., wavelengths=...),
        output_specs=[
            "selector.selected",
            "detector.scores",
            "decider.decisions",
        ],
    )
)
```

**Batch inference:**
```python
from torch.utils.data import DataLoader
from cuvis_ai.datamodule.cu3s_dataset import SingleCu3sDataset

dataset = SingleCu3sDataset(
    cu3s_file_path="data/Lentils/Lentils_000.cu3s",
    processing_mode="Reflectance",
)
dataloader = DataLoader(dataset, batch_size=1)

for batch in dataloader:
    inference_response = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.tensor_to_proto(batch["cube"]),
                wavelengths=helpers.tensor_to_proto(batch["wavelengths"]),
            ),
        )
    )

    # Process results
    decisions = helpers.proto_to_numpy(
        inference_response.outputs["decider.decisions"]
    )
```

### Pipeline Management

**Load pipeline from config:**
```python
import json

# Resolve pipeline config
pipeline_config = stub.ResolveConfig(
    cuvis_ai_pb2.ResolveConfigRequest(
        session_id=session_id,
        config_type="pipeline",
        path="pipeline/rx_statistical",
        overrides=[],
    )
)

# Load pipeline
stub.LoadPipeline(
    cuvis_ai_pb2.LoadPipelineRequest(
        session_id=session_id,
        pipeline=cuvis_ai_pb2.PipelineConfig(
            config_bytes=pipeline_config.config_bytes
        ),
    )
)
```

**Load trained weights:**
```python
stub.LoadPipelineWeights(
    cuvis_ai_pb2.LoadPipelineWeightsRequest(
        session_id=session_id,
        weights_path="outputs/my_experiment/weights.pt",
        strict=True,
    )
)
```

**Save pipeline and weights:**
```python
save_response = stub.SavePipeline(
    cuvis_ai_pb2.SavePipelineRequest(
        session_id=session_id,
        pipeline_path="outputs/my_pipeline.yaml",
        metadata=cuvis_ai_pb2.PipelineMetadata(
            name="My Pipeline",
            description="Trained anomaly detection pipeline",
        ),
    )
)

print(f"Pipeline saved: {save_response.pipeline_path}")
print(f"Weights saved: {save_response.weights_path}")
```

**Get pipeline specifications:**
```python
# Get input specs
inputs_response = stub.GetPipelineInputs(
    cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
)

for name, spec in inputs_response.inputs.items():
    dtype = cuvis_ai_pb2.DType.Name(spec.dtype)
    print(f"Input: {name} | dtype={dtype} | shape={spec.shape}")

# Get output specs
outputs_response = stub.GetPipelineOutputs(
    cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
)

for name, spec in outputs_response.outputs.items():
    dtype = cuvis_ai_pb2.DType.Name(spec.dtype)
    print(f"Output: {name} | dtype={dtype} | shape={spec.shape}")
```

### Restore TrainRun

**Restore from trainrun file:**
```python
restore_response = stub.RestoreTrainRun(
    cuvis_ai_pb2.RestoreTrainRunRequest(
        trainrun_path="outputs/my_experiment/trainrun.yaml",
        weights_path="outputs/my_experiment/weights.pt",  # optional
        strict=True,
    )
)

print(f"Restored session: {restore_response.session_id}")
```

**Continue training from checkpoint:**
```python
# Restore trainrun
restore_response = stub.RestoreTrainRun(
    cuvis_ai_pb2.RestoreTrainRunRequest(
        trainrun_path="outputs/my_experiment/trainrun.yaml",
        weights_path="outputs/my_experiment/checkpoints/epoch=10.ckpt",
    )
)

# Continue training
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=restore_response.session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
    )
):
    print(format_progress(progress))
```

---

## Complete Examples

### Example 1: Statistical Training

**File:** `examples/grpc/statistical_training_client.py`

```python
from workflow_utils import (
    build_stub, create_session_with_search_paths,
    resolve_trainrun_config, apply_trainrun_config, format_progress
)
from cuvis_ai_core.grpc import cuvis_ai_pb2

# Connect to server
stub = build_stub("localhost:50051")

# Create session
session_id = create_session_with_search_paths(stub)

# Resolve trainrun config
resolved, config_dict = resolve_trainrun_config(
    stub,
    session_id,
    "rx_statistical",
    overrides=[
        "data.batch_size=4",
        "training.trainer.max_epochs=1",
        "training.seed=42",
    ],
)

# Apply config
apply_trainrun_config(stub, session_id, resolved.config_bytes)

# Run statistical training
print("Running statistical training...")
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
):
    print(format_progress(progress))

# Clean up
stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
print("Done!")
```

### Example 2: Gradient Training

**File:** `examples/grpc/gradient_training_client.py`

```python
from workflow_utils import (
    build_stub, create_session_with_search_paths,
    resolve_trainrun_config, apply_trainrun_config, format_progress
)
from cuvis_ai_core.grpc import cuvis_ai_pb2

stub = build_stub("localhost:50051")
session_id = create_session_with_search_paths(stub)

# Resolve gradient trainrun
resolved, config_dict = resolve_trainrun_config(
    stub,
    session_id,
    "deep_svdd",
    overrides=[
        "training.trainer.max_epochs=5",
        "training.optimizer.lr=0.0005",
        "training.optimizer.weight_decay=0.005",
    ],
)
apply_trainrun_config(stub, session_id, resolved.config_bytes)

# Statistical initialization
print("Statistical initialization...")
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
):
    print(format_progress(progress))

# Gradient training
print("Gradient training...")
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
    )
):
    print(format_progress(progress))

stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
```

### Example 3: Complete Workflow

**File:** `examples/grpc/complete_workflow_client.py`

```python
from workflow_utils import (
    build_stub, create_session_with_search_paths,
    resolve_trainrun_config, apply_trainrun_config, format_progress
)
from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
import numpy as np

stub = build_stub("localhost:50051")
session_id = create_session_with_search_paths(stub)

# 1. Resolve and apply config
resolved, config_dict = resolve_trainrun_config(
    stub, session_id, "channel_selector",
    overrides=["training.trainer.max_epochs=3"],
)
apply_trainrun_config(stub, session_id, resolved.config_bytes)

# 2. Train pipeline
print("Training pipeline...")
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
    )
):
    print(format_progress(progress))

# 3. Save pipeline
save_response = stub.SavePipeline(
    cuvis_ai_pb2.SavePipelineRequest(
        session_id=session_id,
        pipeline_path="outputs/my_pipeline.yaml",
    )
)
print(f"Saved: {save_response.pipeline_path}, {save_response.weights_path}")

# 4. Run inference
cube = np.random.rand(1, 32, 32, 61).astype(np.float32)
wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)

inference = stub.Inference(
    cuvis_ai_pb2.InferenceRequest(
        session_id=session_id,
        inputs=cuvis_ai_pb2.InputBatch(
            cube=helpers.numpy_to_proto(cube),
            wavelengths=helpers.numpy_to_proto(wavelengths),
        ),
    )
)

results = {
    name: helpers.proto_to_numpy(tensor_proto)
    for name, tensor_proto in inference.outputs.items()
}
print(f"Inference outputs: {list(results.keys())}")

# 5. Clean up
stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
```

### Example 4: Inference with Pretrained Model

**File:** `examples/grpc/run_inference.py`

```python
from workflow_utils import build_stub, create_session_with_search_paths
from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from cuvis_ai.datamodule.cu3s_dataset import SingleCu3sDataset
from torch.utils.data import DataLoader
from pathlib import Path

def run_inference(
    pipeline_path: Path,
    weights_path: Path,
    cu3s_file_path: Path,
    server_address: str = "localhost:50051",
):
    stub = build_stub(server_address, max_msg_size=600 * 1024 * 1024)
    session_id = create_session_with_search_paths(stub)

    # Resolve pipeline config
    pipeline_config = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path=str(pipeline_path),
            overrides=[],
        )
    )

    # Load pipeline and weights
    stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=pipeline_config.config_bytes
            ),
        )
    )

    stub.LoadPipelineWeights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id,
            weights_path=str(weights_path),
            strict=True,
        )
    )

    # Load data
    dataset = SingleCu3sDataset(
        cu3s_file_path=str(cu3s_file_path),
        processing_mode="Reflectance",
    )
    dataloader = DataLoader(dataset, batch_size=1)

    # Run inference
    print(f"Running inference on {len(dataset)} samples...")
    for batch in dataloader:
        inference_response = stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(batch["cube"]),
                    wavelengths=helpers.tensor_to_proto(batch["wavelengths"]),
                ),
            )
        )

        # Process outputs
        decisions = helpers.proto_to_numpy(
            inference_response.outputs["decider.decisions"]
        )
        print(f"Sample decisions: {decisions.shape}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

if __name__ == "__main__":
    run_inference(
        pipeline_path=Path("outputs/my_experiment/pipeline.yaml"),
        weights_path=Path("outputs/my_experiment/weights.pt"),
        cu3s_file_path=Path("data/Lentils/Lentils_000.cu3s"),
    )
```

### Example 5: Restore TrainRun

**File:** `examples/grpc/restore_trainrun_grpc.py`

```python
from workflow_utils import build_stub, create_session_with_search_paths, format_progress
from cuvis_ai_core.grpc import cuvis_ai_pb2
from pathlib import Path
from typing import Literal

def restore_trainrun_grpc(
    trainrun_path: Path,
    mode: Literal["info", "train", "validate", "test"] = "info",
    weights_path: Path | None = None,
    server_address: str = "localhost:50051",
):
    stub = build_stub(server_address)
    session_id = create_session_with_search_paths(stub)

    # Restore trainrun
    restore_response = stub.RestoreTrainRun(
        cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=str(trainrun_path),
            weights_path=str(weights_path) if weights_path else None,
            strict=True,
        )
    )

    if mode == "info":
        # Display pipeline info
        inputs = stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
        )
        outputs = stub.GetPipelineOutputs(
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
        )

        print("Pipeline Inputs:")
        for name, spec in inputs.inputs.items():
            print(f"  {name}: {cuvis_ai_pb2.DType.Name(spec.dtype)} {spec.shape}")

        print("Pipeline Outputs:")
        for name, spec in outputs.outputs.items():
            print(f"  {name}: {cuvis_ai_pb2.DType.Name(spec.dtype)} {spec.shape}")

    elif mode == "train":
        # Continue training
        print("Continuing training...")
        for progress in stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            )
        ):
            print(format_progress(progress))

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

if __name__ == "__main__":
    restore_trainrun_grpc(
        trainrun_path=Path("outputs/my_experiment/trainrun.yaml"),
        mode="train",
        weights_path=Path("outputs/my_experiment/checkpoints/epoch=10.ckpt"),
    )
```

---

## Error Handling & Retries

### gRPC Error Codes

```python
import grpc

try:
    response = stub.CreateSession(request)
except grpc.RpcError as exc:
    code = exc.code()
    details = exc.details()

    if code == grpc.StatusCode.INVALID_ARGUMENT:
        print(f"Invalid request: {details}")
    elif code == grpc.StatusCode.NOT_FOUND:
        print(f"Resource not found: {details}")
    elif code == grpc.StatusCode.FAILED_PRECONDITION:
        print(f"Operation not allowed: {details}")
    elif code == grpc.StatusCode.UNAVAILABLE:
        print(f"Server unavailable: {details}")
    elif code == grpc.StatusCode.INTERNAL:
        print(f"Internal error: {details}")
    else:
        print(f"gRPC error [{code}]: {details}")
```

**Common error scenarios:**
- `INVALID_ARGUMENT`: Malformed inputs (e.g., missing cube, invalid shape)
- `NOT_FOUND`: Unknown session ID or checkpoint path
- `FAILED_PRECONDITION`: Operation not allowed in current state (e.g., inference before loading pipeline)
- `UNAVAILABLE`: Server not running or network issues
- `RESOURCE_EXHAUSTED`: Message size exceeded or server overloaded
- `INTERNAL`: Unexpected server error (check server logs)

### Retry Logic with Exponential Backoff

```python
import time
import grpc

def train_with_retry(
    stub,
    session_id: str,
    trainer_type,
    max_retries: int = 3,
):
    """Train with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            for progress in stub.Train(
                cuvis_ai_pb2.TrainRequest(
                    session_id=session_id,
                    trainer_type=trainer_type,
                )
            ):
                yield progress
            break  # Success

        except grpc.RpcError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached. Error: {e.details()}")
                raise
```

### Health Checks

```python
def check_server_health(server_address: str = "localhost:50051") -> bool:
    """Check if server is healthy."""
    try:
        stub = build_stub(server_address)
        response = stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=response.session_id)
        )
        return True
    except grpc.RpcError:
        return False

# Usage
if not check_server_health():
    print("Server is not available!")
else:
    print("Server is healthy")
```

---

## Best Practices

### 1. Session Management

**Always close sessions:**
```python
session_id = None
try:
    session_id = create_session_with_search_paths(stub)
    # ... operations ...
finally:
    if session_id:
        stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
        )
```

**Use context managers:**
```python
from contextlib import contextmanager

@contextmanager
def grpc_session(stub):
    session_id = create_session_with_search_paths(stub)
    try:
        yield session_id
    finally:
        stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
        )

# Usage
with grpc_session(stub) as session_id:
    # ... operations ...
    pass
# Session automatically closed
```

### 2. Message Size Configuration

**Client and server must agree:**
```python
# Client
stub = build_stub("localhost:50051", max_msg_size=1024 * 1024 * 1024)  # 1 GB

# Server must also have 1 GB limits
```

**Guidelines:**
- Default 300 MB: Suitable for most hyperspectral data
- 600 MB: Large cubes or high-resolution data
- 1 GB: Very large datasets or batch inference

### 3. Error Handling

**Catch specific errors:**
```python
try:
    response = stub.Inference(request)
except grpc.RpcError as exc:
    if exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
        print("Message too large. Reduce batch size or increase limits.")
    elif exc.code() == grpc.StatusCode.UNAVAILABLE:
        print("Server unavailable. Check connection and retry.")
    else:
        raise
```

### 4. Configuration Overrides

**Use structured overrides:**
```python
# Good: Clear intent
overrides = [
    "training.trainer.max_epochs=100",
    "training.optimizer.lr=0.001",
    "data.batch_size=16",
]

# Avoid: Hardcoded strings
overrides = ["max_epochs=100"]  # Ambiguous
```

### 5. Streaming Progress

**Handle all progress states:**
```python
for progress in stub.Train(...):
    status = cuvis_ai_pb2.TrainStatus.Name(progress.status)

    if status == "TRAIN_STATUS_RUNNING":
        # Update progress bar
        pass
    elif status == "TRAIN_STATUS_COMPLETED":
        # Training finished
        break
    elif status == "TRAIN_STATUS_FAILED":
        # Handle failure
        print(f"Training failed: {progress.error_message}")
        break
```

### 6. Production Deployment

**Use TLS in production:**
```python
# Never use insecure channels in production
# BAD: grpc.insecure_channel("production-server:50051")

# GOOD: Use TLS
channel = grpc.secure_channel(
    "production-server:50051",
    grpc.ssl_channel_credentials(),
)
```

**Implement health checks:**
```yaml
# docker-compose.yml
healthcheck:
  test: ["CMD", "python", "-c", "import grpc; ..."]
  interval: 30s
  timeout: 10s
  retries: 3
```

**Monitor resource usage:**
- Track active sessions
- Monitor GPU memory
- Log training progress
- Set session timeout

---

## Troubleshooting

### Server Not Running

**Problem:** `grpc.StatusCode.UNAVAILABLE`

**Solution:**
```bash
# Check if server is running
ps aux | grep production_server

# Check port availability
netstat -an | grep 50051

# Test connection
telnet localhost 50051

# Restart server
docker-compose restart
```

### Message Size Exceeded

**Problem:** `grpc.StatusCode.RESOURCE_EXHAUSTED`

**Solution:**
```python
# Increase client limits
stub = build_stub("localhost:50051", max_msg_size=1024 * 1024 * 1024)

# OR reduce batch size
overrides=["data.batch_size=1"]
```

### Session Expired

**Problem:** `grpc.StatusCode.NOT_FOUND` (session not found)

**Solution:**
- Sessions expire after 1 hour of inactivity
- Create a new session
- Implement periodic activity to keep session alive

```python
import threading
import time

def keep_alive(stub, session_id, interval=300):
    """Send periodic heartbeat to keep session alive."""
    while True:
        time.sleep(interval)
        try:
            stub.GetPipelineInputs(
                cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
            )
        except grpc.RpcError:
            break  # Session closed or expired
```

### CUDA Out of Memory

**Problem:** Training fails with CUDA OOM error

**Solution:**
```python
# Reduce batch size
overrides=["data.batch_size=1"]

# Close unused sessions
stub.CloseSession(...)

# Monitor GPU memory
# nvidia-smi
```

### Connection Timeout

**Problem:** Long-running operations timeout

**Solution:**
```python
# Increase timeout for long operations
channel = grpc.insecure_channel(
    "localhost:50051",
    options=[
        ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 10000),
    ],
)
```

---

## See Also

- **API Reference**:
  - [gRPC API Reference](../grpc/api-reference.md) - Complete RPC method documentation
  - [Protocol Definitions](../grpc/api-reference.md#protocol-definitions) - Message format specifications
- **Deployment**:
  - [gRPC Deployment Guide](../deployment/grpc_deployment.md) - Production deployment patterns
  - [Docker Configuration](../deployment/grpc_deployment.md#docker) - Container setup
  - [Kubernetes Setup](../deployment/grpc_deployment.md#kubernetes) - Orchestration
- **Tutorials**:
  - [gRPC Workflow Tutorial](../tutorials/grpc-workflow.md) - Comprehensive end-to-end guide
- **Examples**:
  - `examples/grpc/workflow_utils.py` - Helper utilities
  - `examples/grpc/statistical_training_client.py` - Statistical training
  - `examples/grpc/gradient_training_client.py` - Gradient training
  - `examples/grpc/complete_workflow_client.py` - End-to-end workflow
  - `examples/grpc/run_inference.py` - Inference with pretrained models
  - `examples/grpc/restore_trainrun_grpc.py` - TrainRun restoration
