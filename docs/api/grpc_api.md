# gRPC API Reference

The cuvis.ai gRPC surface exposes session management, training, inference, and pipeline introspection over a single `CuvisAIService` endpoint.

## Connect

```python
import grpc
from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
```

## Session Management

### CreateSession
Start a new pipeline session.

- Request: `pipeline_type` (`"channel_selector"`, `"statistical"`, `"gradient"`), optional `pipeline_config` JSON string, `data_config` (`DataConfig`)
- Response: `session_id` (string)

```python
create_resp = stub.CreateSession(
    cuvis_ai_pb2.CreateSessionRequest(
        pipeline_type="statistical",
        pipeline_config='{"n_select": 4}',
        data_config=cuvis_ai_pb2.DataConfig(
            cu3s_file_path="/data/Lentils_000.cu3s",
            annotation_json_path="/data/Lentils_000.json",
            train_ids=[0, 1, 2],
            val_ids=[3, 4],
            test_ids=[5, 6],
            batch_size=2,
            processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
        ),
    )
)
session_id = create_resp.session_id
```

### CloseSession
Release all resources for a session.

- Request: `session_id`
- Response: `success` (bool)

```python
stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
```

## Pipeline Introspection

### GetPipelineInputs / GetPipelineOutputs
Return the available input/output tensor specs.

- Response: `input_names` / `output_names` plus `input_specs` / `output_specs` (map of `TensorSpec`)

```python
inputs = stub.GetPipelineInputs(
    cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
)
outputs = stub.GetPipelineOutputs(
    cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
)
print(inputs.input_names, outputs.output_names)
```

### GetPipelineVisualization
Render the current pipeline graph.

- Request: `session_id`, `format` (`"png"`, `"svg"`, `"dot"`, `"mermaid"`)
- Response: `image_data` (bytes), `format`

## Training

### Train (Statistical)
Performs statistical initialization for the pipeline with a short progress stream.

- Request: `session_id`, `trainer_type=TRAINER_TYPE_STATISTICAL`
- Response: stream of `TrainResponse` messages

```python
for update in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
):
    print(update.status, update.message)
```

### Train (Gradient)
Runs gradient-based training with streaming progress, losses, and metrics.

- Request: `session_id`, `trainer_type=TRAINER_TYPE_GRADIENT`, `config` (`TrainingConfig`)
- Response: stream of `TrainResponse`

```python
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig

cfg = TrainingConfig(
    trainer=TrainerConfig(max_epochs=5, accelerator="cpu"),
    optimizer=OptimizerConfig(name="adam", lr=1e-3),
)

for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        config=cuvis_ai_pb2.TrainingConfig(config_json=cfg.to_json().encode()),
    )
):
    print(progress.context.epoch, progress.losses, progress.metrics)
```

### GetTrainStatus
Return the last known training status (`TRAIN_STATUS_*`).

### GetTrainingCapabilities
Discover supported optimizers, schedulers, callbacks, and parameter schemas.

### ValidateConfig (training)
Validate a training config via the generic `ValidateConfig` RPC.

- Request: `config_type="training"`, `config_bytes` (JSON payload)
- Response: `valid` (bool), `errors`, `warnings`

```python
validation = stub.ValidateConfig(
    cuvis_ai_pb2.ValidateConfigRequest(
        config_type="training",
        config_bytes=cfg.to_json().encode(),
    )
)
if not validation.valid:
    raise ValueError(validation.errors)
```

## Inference

Run inference on an input batch and optionally restrict outputs.

- Request: `session_id`, `inputs` (`InputBatch`), optional `output_specs` (list of output names or port names)
- Response: `outputs` (map<string, `Tensor`>), `metrics` (map<string, float>)

```python
cube = np.random.rand(1, 32, 32, 61).astype(np.float32)
resp = stub.Inference(
    cuvis_ai_pb2.InferenceRequest(
        session_id=session_id,
        inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
        output_specs=["selector.selected"],
    )
)
selected = helpers.proto_to_numpy(resp.outputs["selector.selected"])
```

## Checkpoints

### SaveCheckpoint
- Request: `session_id`, `checkpoint_path`
- Response: `success`

### LoadCheckpoint
- Request: `session_id`, `checkpoint_path`
- Response: `success`

## Data Types

### DataConfig
- `cu3s_file_path`: path to `.cu3s`
- `annotation_json_path`: COCO-style annotations (optional)
- `train_ids`, `val_ids`, `test_ids`: sample indices
- `batch_size`: int
- `processing_mode`: `PROCESSING_MODE_RAW` or `PROCESSING_MODE_REFLECTANCE`

### InputBatch
- `cube` (`Tensor`, required): hyperspectral cube `[B, H, W, C]`
- `wavelengths`, `mask` (`Tensor`, optional)
- `bboxes` (`BoundingBoxes`), `points` (`Points`), `text_prompt` (string)
- `extra_inputs`: map<string, `Tensor`>

### Tensor
- `shape`: repeated `int64`
- `dtype`: `D_TYPE_*`
- `raw_data`: bytes

### TensorSpec
- `name`: string
- `shape`: repeated `int64` (`-1` for dynamic dims)
- `dtype`: `D_TYPE_*`
- `required`: bool

### TrainingConfig (JSON payload)
Serialized with `TrainingConfig.to_json()` / `.from_json()`:
- `trainer`: epochs, accelerator, callbacks, logging cadence
- `optimizer`: name, `lr`, optional scheduler and params

## Error Handling

RPCs use gRPC status codes:
- `INVALID_ARGUMENT`: malformed inputs (e.g., missing `cube`)
- `NOT_FOUND`: unknown session or checkpoint
- `FAILED_PRECONDITION`: operation not allowed in current state
- `INTERNAL`: unexpected server error

```python
try:
    stub.CreateSession(request)
except grpc.RpcError as exc:
    print(exc.code(), exc.details())
```

## Best Practices
- Always call `CloseSession` after training/inference to release resources.
- Use `ValidateConfig` with `config_type="training"` before gradient training.
- Provide `output_specs` to limit payload sizes.
- Save checkpoints periodically and before shutdown.
- Reuse gRPC channels; set `grpc.max_send_message_length` / `grpc.max_receive_message_length` when sending large cubes.
