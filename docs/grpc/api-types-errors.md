!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Types & Errors API

Introspection, discovery, error handling, and data type definitions for the `CuvisAIService`.

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
  string pipeline_path = 1;  // Relative path from server pipeline root (includes .yaml)
  string resolved_path = 2;  // Concrete absolute server path for follow-on calls like ResolveConfig
  PipelineMetadata metadata = 3;
  string weights_path = 6;
  string yaml_content = 7;
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
    print(f"  - {pipeline.pipeline_path}")
    print(f"    Description: {pipeline.metadata.description}")
    print(f"    Tags: {', '.join(pipeline.metadata.tags)}")
    print(f"    Has weights: {bool(pipeline.weights_path)}")
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
  string pipeline_path = 1;  // Relative path with extension (e.g. "anomaly/deep_svdd/deep_svdd.yaml")
}
```

`pipeline_path` must be a relative path (not absolute), use `/` separators, and include the `.yaml` extension. `PipelineInfo.resolved_path` is the concrete server-side file path returned by discovery and can be passed into `ResolveConfigRequest.path` when you want config bytes for `LoadPipeline`.

**Response:**
```protobuf
message GetPipelineInfoResponse {
  PipelineInfo pipeline_info = 1;
}
```

**Python Example:**
```python
response = stub.GetPipelineInfo(
    cuvis_ai_pb2.GetPipelineInfoRequest(
        pipeline_path="anomaly/deep_svdd/deep_svdd.yaml"
    )
)

print(f"Pipeline: {response.pipeline_info.pipeline_path}")
print(f"Resolved path: {response.pipeline_info.resolved_path}")
print(f"Description: {response.pipeline_info.metadata.description}")
print(f"Tags: {', '.join(response.pipeline_info.metadata.tags)}")
print(f"Has weights: {bool(response.pipeline_info.weights_path)}")
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

- [gRPC API Reference](api-reference.md) - Session, configuration, and pipeline management
- [Training & Inference API](api-training-inference.md) - Training, inference, and profiling operations
- [Client Patterns](client-patterns.md) - Common usage patterns and best practices
- [Sequence Diagrams](sequence-diagrams.md) - Visual workflows
- [gRPC Workflow Tutorial](../tutorials/grpc-workflow.md) - Hands-on tutorial
- [TrainRun Schema](../config/trainrun-schema.md) - Trainrun configuration reference
- [Pipeline Schema](../config/pipeline-schema.md) - Pipeline YAML schema
- [Hydra Composition](../config/hydra-composition.md) - Config composition patterns
- [Deployment Guide](../deployment/grpc_deployment.md) - Docker, Kubernetes, production
