!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Client Workflows & Error Handling

Configuration, training, inference, and error handling patterns for CUVIS.AI gRPC clients.

---

## Configuration Patterns

### Standard Config Resolution

```python
from cuvis_ai.utils.grpc_workflow import (
    resolve_trainrun_config,
    apply_trainrun_config,
)

# Resolve trainrun config with overrides
resolved, config_dict = resolve_trainrun_config(
    stub,
    session_id,
    "deep_svdd",
    overrides=[
        "training.trainer.max_epochs=50",
        "training.optimizer.lr=0.0005",
        "data.batch_size=8",
    ],
)

# Apply to session
apply_trainrun_config(stub, session_id, resolved.config_bytes)

# Inspect resolved config
print(f"Pipeline: {config_dict['pipeline']['name']}")
print(f"Optimizer: {config_dict['training']['optimizer']['name']}")
print(f"Learning rate: {config_dict['training']['optimizer']['lr']}")
```

---

### Config Validation Before Training

```python
import json

def validate_and_train(stub, session_id, trainrun_name, overrides=None):
    """Resolve, validate, and train with error handling."""
    from cuvis_ai.utils.grpc_workflow import resolve_trainrun_config

    # Resolve config
    resolved, config_dict = resolve_trainrun_config(
        stub, session_id, trainrun_name, overrides
    )

    # Validate training config
    training_json = json.dumps(config_dict["training"]).encode("utf-8")
    validation = stub.ValidateConfig(
        cuvis_ai_pb2.ValidateConfigRequest(
            config_type="training",
            config_bytes=training_json,
        )
    )

    if not validation.valid:
        print("Training configuration validation failed:")
        for error in validation.errors:
            print(f"  ERROR: {error}")
        raise ValueError("Invalid training configuration")

    # Show warnings
    for warning in validation.warnings:
        print(f"  WARNING: {warning}")

    # Apply config and train
    stub.SetTrainRunConfig(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=cuvis_ai_pb2.TrainRunConfig(config_bytes=resolved.config_bytes),
        )
    )

    # Train with validated config
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )
    ):
        yield progress

# Usage
for progress in validate_and_train(stub, session_id, "deep_svdd"):
    print(format_progress(progress))
```

---

## Training Patterns

### Two-Phase Training Pattern

```python
from cuvis_ai.utils.grpc_workflow import format_progress

def two_phase_training(stub, session_id):
    """Standard two-phase training: statistical then gradient."""

    # Phase 1: Statistical initialization (fast, no backprop)
    print("=== Phase 1: Statistical Training ===")
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        )
    ):
        print(f"[Statistical] {format_progress(progress)}")

    # Phase 2: Gradient-based fine-tuning (full training)
    print("\n=== Phase 2: Gradient Training ===")
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )
    ):
        print(f"[Gradient] {format_progress(progress)}")

    print("\n=== Training Complete ===")

# Usage
two_phase_training(stub, session_id)
```

---

### Progress Monitoring with Early Stopping

```python
def train_with_early_stopping(stub, session_id, trainer_type, patience=5):
    """Train with custom early stopping logic."""
    best_metric = float('-inf')
    patience_counter = 0

    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=trainer_type,
        )
    ):
        # Print all progress
        print(format_progress(progress))

        # Check validation metrics for early stopping
        if progress.context.stage == cuvis_ai_pb2.EXECUTION_STAGE_VAL:
            current_metric = progress.metrics.get('val_iou', 0.0)

            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                print(f"  → New best metric: {best_metric:.4f}")
            else:
                patience_counter += 1
                print(f"  → No improvement (patience: {patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n=== Early stopping triggered (patience={patience}) ===")
                break

    print(f"\nBest validation metric: {best_metric:.4f}")

# Usage
train_with_early_stopping(stub, session_id, cuvis_ai_pb2.TRAINER_TYPE_GRADIENT)
```

---

### Checkpoint Saving Strategy

```python
def train_with_checkpoints(stub, session_id, trainer_type, checkpoint_every_n_epochs=10):
    """Save checkpoints periodically during training."""
    from pathlib import Path

    output_dir = Path("outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_counter = 0

    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=trainer_type,
        )
    ):
        print(format_progress(progress))

        # Save checkpoint at end of each N epochs
        if progress.context.stage == cuvis_ai_pb2.EXECUTION_STAGE_TRAIN:
            current_epoch = progress.context.epoch

            if current_epoch > epoch_counter:
                epoch_counter = current_epoch

                if epoch_counter % checkpoint_every_n_epochs == 0:
                    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch_counter}.yaml"

                    print(f"\n  → Saving checkpoint at epoch {epoch_counter}...")
                    response = stub.SavePipeline(
                        cuvis_ai_pb2.SavePipelineRequest(
                            session_id=session_id,
                            pipeline_path=str(checkpoint_path),
                        )
                    )
                    print(f"  → Saved: {response.pipeline_path}\n")

# Usage
train_with_checkpoints(stub, session_id, cuvis_ai_pb2.TRAINER_TYPE_GRADIENT)
```

---

## Inference Patterns

### Single Inference Pattern

```python
from cuvis_ai_core.grpc import helpers
import numpy as np

def run_inference(stub, session_id, cube, wavelengths):
    """Run inference on single hyperspectral cube."""

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
    outputs = {
        name: helpers.proto_to_numpy(tensor_proto)
        for name, tensor_proto in response.outputs.items()
    }

    return outputs

# Usage
cube = np.random.rand(1, 32, 32, 61).astype(np.float32)
wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.float32)

outputs = run_inference(stub, session_id, cube, wavelengths)
print(f"Outputs: {list(outputs.keys())}")
```

---

### Batch Inference Pattern

```python
from torch.utils.data import DataLoader
from cuvis_ai_core.data.datasets import SingleCu3sDataModule

def batch_inference(stub, session_id, cu3s_path, batch_size=4):
    """Efficient batch inference on CU3S dataset."""
    from cuvis_ai_core.grpc import helpers

    # Create data loader
    datamodule = SingleCu3sDataModule(
        cu3s_file_path=cu3s_path,
        batch_size=batch_size,
        processing_mode="Reflectance",
    )
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    # Run inference on each batch
    all_results = []
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

        response = stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(batch["cube"]),
                    wavelengths=helpers.tensor_to_proto(batch["wavelengths"]),
                ),
            )
        )

        # Convert and store outputs
        batch_outputs = {
            name: helpers.proto_to_numpy(tensor_proto)
            for name, tensor_proto in response.outputs.items()
        }
        all_results.append(batch_outputs)

    return all_results

# Usage
results = batch_inference(stub, session_id, "data/Lentils_000.cu3s", batch_size=4)
print(f"Processed {len(results)} batches")
```

---

### Output Filtering Pattern

```python
def filtered_inference(stub, session_id, cube, wavelengths, output_specs):
    """Inference with output filtering (reduces payload)."""
    from cuvis_ai_core.grpc import helpers

    response = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
            ),
            output_specs=output_specs,  # Only request specific outputs
        )
    )

    outputs = {
        name: helpers.proto_to_numpy(tensor_proto)
        for name, tensor_proto in response.outputs.items()
    }

    return outputs

# Usage - only request final decisions
outputs = filtered_inference(
    stub,
    session_id,
    cube,
    wavelengths,
    output_specs=["decider.decisions", "detector.scores"],
)
```

---

## Error Handling

### Comprehensive Error Handling

```python
import grpc

def safe_inference(stub, session_id, cube, wavelengths):
    """Inference with comprehensive error handling."""
    from cuvis_ai_core.grpc import helpers

    try:
        response = stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.numpy_to_proto(cube),
                    wavelengths=helpers.numpy_to_proto(wavelengths),
                ),
            )
        )

        return {
            name: helpers.proto_to_numpy(tensor_proto)
            for name, tensor_proto in response.outputs.items()
        }

    except grpc.RpcError as e:
        code = e.code()
        details = e.details()

        if code == grpc.StatusCode.INVALID_ARGUMENT:
            print(f"Invalid inference request: {details}")
        elif code == grpc.StatusCode.NOT_FOUND:
            print(f"Session not found: {details}")
        elif code == grpc.StatusCode.FAILED_PRECONDITION:
            print(f"Cannot run inference: {details}")
        elif code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            print(f"Resource exhausted: {details}")
        else:
            print(f"gRPC error: {code} - {details}")

        raise
```

---

## Best Practices Summary

1. **Always close sessions** - Use try/finally or context managers
2. **Configure message size** - Set 600 MB limits for hyperspectral data
3. **Validate configs early** - Use `ValidateConfig` before training
4. **Filter outputs** - Specify `output_specs` to reduce payload
5. **Handle errors gracefully** - Check gRPC status codes
6. **Use helper functions** - Leverage `grpc_workflow.py`
7. **Monitor training progress** - Process streaming updates
8. **Save checkpoints** - Periodic `SavePipeline` during training
9. **Reuse channels** - Don't create new channels for each request
10. **Enable TLS in production** - Use `ssl_channel_credentials`

---

## Troubleshooting

### Connection Refused

**Error:** `grpc._channel._InactiveRpcError: failed to connect to all addresses`

**Solutions:**

1. Verify server is running: `ps aux | grep production_server`
2. Check port: `netstat -an | grep 50051`
3. Test connectivity: `telnet localhost 50051`
4. Check firewall (if remote): `sudo ufw allow 50051`

---

### Message Size Exceeded

**Error:** `grpc._channel._MultiThreadedRendezvous: Received message larger than max`

**Solution:**
```python
stub = build_stub("localhost:50051", max_msg_size=600*1024*1024)
```

---

### Session Not Found

**Error:** `Session ID not found`

**Solutions:**

1. Session expired (1-hour timeout) - create new session
2. Server restarted - sessions are not persisted
3. Verify session ID is correct

---

### CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**

1. Reduce batch size in data config
2. Close unused sessions
3. Restart server to clear GPU memory

---

## Plugin Management

CUVIS.AI supports a plugin system for extending functionality. Plugins can be loaded dynamically at runtime to add custom nodes, data sources, or processing capabilities.

For comprehensive plugin system documentation, see:

- [Plugin System Overview](../plugin-system/overview.md) - Architecture and core concepts
- [Plugin Usage Guide](../plugin-system/usage.md) - Selective manifests and loading examples
- [Plugin Development Guide](../plugin-system/development.md) - Creating custom plugins

**gRPC Integration:**
Plugins loaded on the server side are automatically available to all gRPC clients. Use the discovery RPCs to query available capabilities after plugins are loaded.

---

## See Also

- [Client Connections & Sessions](client-connections.md) - Connection management and session lifecycle patterns
- [gRPC API Reference: Sessions](api-session.md) - Session RPC method documentation
- [gRPC API Reference: Config](api-config.md) - Configuration RPC method documentation
- [gRPC API Reference: Training & Inference](api-training-inference.md) - Training and inference RPC documentation
- [gRPC API Reference: Pipeline](api-pipeline.md) - Pipeline RPC method documentation
- [gRPC Overview](overview.md) - Architecture and concepts
- [Sequence Diagrams](sequence-diagrams.md) - Visual workflows
- [gRPC Tutorial](../tutorials/grpc-workflow.md) - Hands-on tutorial
- [Deployment Guide](../deployment/grpc_deployment.md) - Production patterns
