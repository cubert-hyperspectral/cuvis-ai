!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Ports API Reference

Complete API reference for the Typed I/O port system in CUVIS.AI.

## Overview

The port system provides typed input/output interfaces for all nodes, enabling type-safe connections, runtime validation, and flexible pipeline construction. Each node defines its input and output ports using `PortSpec` objects.

## Core Components

### PortSpec

The `PortSpec` class defines the specification for a port, including its type, shape constraints, and metadata.

**Attributes:**
- `name`: Port identifier
- `port_type`: "input" or "output"
- `shape`: Expected tensor shape with dimension constraints
- `dtype`: Expected data type (optional)
- `description`: Human-readable description
- `stage`: Execution stage ("train", "eval", "both")

**Example:**
```python
from cuvis_ai_schemas.pipeline import PortSpec

# Define an input port for hyperspectral data
data_port = PortSpec(
    name="data",
    port_type="input",
    shape=(-1, -1, -1, -1),  # (batch, height, width, channels)
    description="Raw hyperspectral cube input"
)

# Define an output port for normalized data
normalized_port = PortSpec(
    name="normalized", 
    port_type="output",
    shape=(-1, -1, -1, -1),
    description="Normalized hyperspectral cube"
)
```

### InputPort / OutputPort

Port instances that are attached to nodes and used for connections.

**Creating Ports:**
```python
from cuvis_ai_schemas.pipeline import InputPort, OutputPort

# Create port instances
input_port = InputPort(spec=data_port, node=normalizer)
output_port = OutputPort(spec=normalized_port, node=normalizer)
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
# Check if ports are compatible
if input_port.is_compatible_with(output_port):
    pipeline.connect(output_port, input_port)
else:
    print("Ports are incompatible")
```

## Node Port Declarations

Nodes declare their ports using `INPUT_SPECS` and `OUTPUT_SPECS` class attributes.

### Example Node Implementation

```python
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.pipeline import PortSpec

class MinMaxNormalizer(Node):
    """Min-max normalization node."""
    
    # Input port specifications
    INPUT_SPECS = [
        PortSpec(
            name="data",
            port_type="input",
            shape=(-1, -1, -1, -1),
            description="Raw hyperspectral cube"
        )
    ]
    
    # Output port specifications  
    OUTPUT_SPECS = [
        PortSpec(
            name="normalized",
            port_type="output", 
            shape=(-1, -1, -1, -1),
            description="Normalized cube [0, 1]"
        )
    ]
    
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

```python
# Connect nodes for specific execution stages
pipeline.connect(normalizer.normalized, selector.data, stage="train")
pipeline.connect(selector.selected, pca.features, stage="both")
```

### Loss Nodes Without an Aggregator

`LossAggregator` has been removed—the trainer now collects individual loss nodes directly.
Register every loss/regularizer node with the `GradientTrainer` (or any custom trainer) and
feed their inputs through standard port connections, as shown in
`examples//03_channel_selector.py`.

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
# Distribute batch to a specific input port
outputs = pipeline.forward(batch={f"{normalizer.id}.data": input_data})
```

### Multiple Inputs

```python
# Distribute different data to different input ports
outputs = pipeline.forward(batch={
    f"{node1.id}.data1": data1,
    f"{node2.id}.data2": data2,
    f"{node3.id}.features": features
})
```

### Batch Key Format

Batch keys follow the pattern: `{node_id}.{port_name}`

## Dimension Resolution

The port system automatically resolves variable dimensions during execution.

### Dynamic Shape Resolution

```python
# Port with variable dimensions
port_spec = PortSpec(
    name="features",
    port_type="input", 
    shape=(-1, -1, -1, -1)  # All dimensions variable
)

# During execution, dimensions are resolved from input data
# Input shape: (32, 64, 64, 100) → Output shape: (32, 64, 64, n_components)
```

### Constraint Validation

```python
# Port with fixed channel dimension
port_spec = PortSpec(
    name="features",
    port_type="input",
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
try:
    pipeline.connect(normalizer.normalized, pca.features)
except ValueError as e:
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
    name="embedding",
    port_type="output",
    shape=(-1, 512),  # Fixed embedding dimension
    dtype=torch.float32,
    description="Feature embeddings"
)
```

### Port Inspection

```python
# Inspect port properties
port = normalizer.normalized
print(f"Port name: {port.name}")
print(f"Port type: {port.port_type}")
print(f"Expected shape: {port.shape}")
print(f"Description: {port.description}")
```

### Connection Graph

```python
# Get all connections in the pipeline
connections = pipeline.get_connections()
for source, target in connections:
    print(f"{source.node.name}.{source.name} → {target.node.name}.{target.name}")
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

- **[Nodes API](nodes.md)**: Node implementations with port specifications
- **[Pipeline API](pipeline.md)**: Pipeline and connection management
- **[Core Concepts](../concepts/overview.md)**: Understand the architecture
- **[Quickstart](../user-guide/quickstart.md)**: Practical port usage examples
