!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Ports API Reference

Complete API reference for the Typed I/O port system in Cuvis.AI.

## Overview

The port system provides typed input/output interfaces for all nodes, enabling type-safe connections, runtime validation, and flexible pipeline construction. Each node defines its input and output ports using `PortSpec` objects.

## Core Components

### PortSpec

The `PortSpec` class defines the specification for a port, including its type, shape constraints, and metadata.

**Attributes:**

- `dtype`: Expected element data type (e.g. `torch.float32`)
- `shape`: Expected tensor shape with dimension constraints (`-1` for dynamic dims)
- `description`: Human-readable description
- `optional`: Whether the port may be left unconnected
- `variadic`: Whether the port accepts a fan-in of multiple connections

The port *name* is the key under which a spec is registered in a node's
`INPUT_SPECS` / `OUTPUT_SPECS` dict, not a field on `PortSpec` itself. Input/output
direction likewise comes from which dict the spec lives in, not from the spec.

**Example:**
```python
import torch
from cuvis_ai_schemas.pipeline import PortSpec

# Define a spec for hyperspectral data (port name is the dict key on the node)
data_port = PortSpec(
    dtype=torch.float32,
    shape=(-1, -1, -1, -1),  # (batch, height, width, channels)
    description="Raw hyperspectral cube input"
)

# Define a spec for normalized data
normalized_port = PortSpec(
    dtype=torch.float32,
    shape=(-1, -1, -1, -1),
    description="Normalized hyperspectral cube"
)
```

### InputPort / OutputPort

Port instances that are attached to nodes and used for connections.

**Creating Ports:**
```python
from cuvis_ai_schemas.pipeline import InputPort, OutputPort

# Create port instances (node, name, spec)
input_port = InputPort(node=normalizer, name="data", spec=data_port)
output_port = OutputPort(node=normalizer, name="normalized", spec=normalized_port)
```

## Port Compatibility Rules

Ports can be connected if they satisfy compatibility rules:

### Shape Compatibility

- Fixed dimensions must match exactly
- Variable dimensions (`-1`) can match any size
- Batch dimensions are typically variable

### Type Compatibility

- Input ports can only connect to output ports
- Ports must have compatible data types
- Stage constraints must be satisfied

### Connection Validation
```python
# Check if specs are compatible (PortSpec.is_compatible_with returns (bool, message))
is_compatible, message = output_port.spec.is_compatible_with(
    input_port.spec, source_node=normalizer, target_node=normalizer
)
if is_compatible:
    pipeline.connect(output_port, input_port)
else:
    print(f"Ports are incompatible: {message}")
```

## Node Port Declarations

Nodes declare their ports using `INPUT_SPECS` and `OUTPUT_SPECS` class attributes.

### Example Node Implementation

```python
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.pipeline import PortSpec

class MinMaxNormalizer(Node):
    """Min-max normalization node."""
    
    # Input port specifications (dict keyed by port name)
    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Raw hyperspectral cube"
        )
    }
    
    # Output port specifications (dict keyed by port name)
    OUTPUT_SPECS = {
        "normalized": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Normalized cube [0, 1]"
        )
    }
    
    def __init__(self, eps=1e-6, use_running_stats=True):
        super().__init__()
        self.eps = eps
        self.use_running_stats = use_running_stats
        
    def forward(self, **inputs):
        data = inputs["data"]
        # Normalization logic here
        normalized = (data - self.running_min) / (self.running_max - self.running_min + self.eps)
        return {"normalized": normalized}
```

## Port-Based Connections

### Basic Connection

```python
# Connect two nodes using their ports
pipeline.connect(normalizer.normalized, selector.data)
```

### Multiple Connections

```python
# Fan-in multiple outputs to a single input (e.g., monitoring artifacts)
pipeline.connect(
    (viz_mask.artifacts, tensorboard_node.artifacts),
    (viz_rgb.artifacts, tensorboard_node.artifacts),
)
```

### Stage-Aware Connections

Stage routing is controlled per node via `execution_stages`, not on `connect`.
Connections themselves are stage-agnostic.

```python
# Connections carry no stage argument
pipeline.connect(normalizer.normalized, selector.data)
pipeline.connect(selector.selected, pca.features)
```

### Loss Nodes Without an Aggregator

`LossAggregator` has been removed—the trainer now collects individual loss nodes directly.
Register every loss/regularizer node with the `GradientTrainer` (or any custom trainer) and
feed their inputs through standard port connections, as shown in
[examples/channel_selector.py](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook/blob/main/examples/channel_selector.py).

```python
pipeline.connect(
    (logit_head.logits, bce_loss.predictions),
    (data_node.mask, bce_loss.targets),
    (selector.weights, entropy_loss.weights),
    (selector.weights, diversity_loss.weights),
)

grad_trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[bce_loss, entropy_loss, diversity_loss],
    metric_nodes=[metrics_anomaly],
    trainer_config=training_cfg.trainer,
    optimizer_config=training_cfg.optimizer,
)
```

## Batch Distribution

The port system enables explicit batch distribution to specific input ports.

### Single Input

```python
# Feed an input port by its bare port name
outputs = pipeline.forward(batch={"data": input_data})
```

### Multiple Inputs

```python
# Feed several input ports by name
outputs = pipeline.forward(batch={
    "data1": data1,
    "data2": data2,
    "features": features,
})
```

### Batch Key Format

Batch keys are **bare port names**. A value is distributed to every node whose
`INPUT_SPECS` declare a port of that name (an entry-point port left unconnected
from any predecessor). Keys are not node-qualified.

## Dimension Resolution

The port system automatically resolves variable dimensions during execution.

### Dynamic Shape Resolution

```python
# Port with variable dimensions
port_spec = PortSpec(
    dtype=torch.float32,
    shape=(-1, -1, -1, -1)  # All dimensions variable
)

# During execution, dimensions are resolved from input data
# Input shape: (32, 64, 64, 100) → Output shape: (32, 64, 64, n_components)
```

### Constraint Validation

```python
# Port with fixed channel dimension
port_spec = PortSpec(
    dtype=torch.float32,
    shape=(-1, -1, -1, 100)  # Fixed channel dimension
)

# Connection will fail if channel dimension doesn't match
```

## Common Port Patterns

### Normalization Nodes

**Input Ports:**

- `data`: Raw hyperspectral cube

**Output Ports:**

- `normalized`: Normalized data

### Feature Extraction

**Input Ports:**

- `features`: Input features for transformation

**Output Ports:**

- `projected`: Transformed features
- `explained_variance`: Statistical metrics

### Anomaly Detection

**Input Ports:**

- `data`: Features for anomaly scoring

**Output Ports:**

- `scores`: Anomaly detection scores
- `logits`: Logit-transformed scores

### Loss Nodes

**Input Ports:**

- Variadic inputs for loss computation

**Output Ports:**

- `loss`: Computed loss value

## Error Handling

### Port Not Found

```python
try:
    pipeline.connect(normalizer.nonexistent, selector.data)
except AttributeError as e:
    print(f"Port error: {e}")
    # Error: 'MinMaxNormalizer' object has no attribute 'nonexistent'
```

### Incompatible Ports

```python
from cuvis_ai_schemas.pipeline.exceptions import PortCompatibilityError

try:
    pipeline.connect(normalizer.normalized, pca.features)
except PortCompatibilityError as e:
    print(f"Compatibility error: {e}")
    # Error: Port shapes are incompatible: (-1, -1, -1, -1) vs (-1, -1, -1, 3)
```

### Missing Batch Distribution

```python
try:
    outputs = pipeline.forward(batch=input_data)
except KeyError as e:
    print(f"Batch error: {e}")
    # Error: Unable to find input port for batch key
```

## Advanced Usage

### Custom Port Specifications

```python
# Create custom port with specific constraints
custom_port = PortSpec(
    dtype=torch.float32,
    shape=(-1, 512),  # Fixed embedding dimension
    description="Feature embeddings"
)
```

### Port Inspection

```python
# Inspect port properties (shape/description live on the spec)
port = normalizer.normalized
print(f"Port name: {port.name}")
print(f"Expected shape: {port.spec.shape}")
print(f"Description: {port.spec.description}")
```

### Connection Graph

```python
# There is no public connection-listing method; iterate the pipeline's nodes
for node in pipeline.nodes:
    print(node.name)
```

## Best Practices

1. **Use Descriptive Port Names**: Choose names that clearly indicate the port's purpose
2. **Define Shape Constraints**: Use fixed dimensions when possible for early error detection
3. **Document Ports**: Provide clear descriptions for each port
4. **Test Port Compatibility**: Validate connections during development
5. **Use Stage Filtering**: Leverage stage-aware execution for performance

## API Reference

::: cuvis_ai_schemas.pipeline

## See Also

- **[Node Catalog](../../catalogs/nodes/index.md)**: Node implementations with port specifications
- **[Pipeline API](pipeline.md)**: Pipeline and connection management
- **[Core Concepts](../../concepts/index.md)**: Understand the architecture
- **[Quickstart](../../get-started/quickstart.md)**: Practical port usage examples
