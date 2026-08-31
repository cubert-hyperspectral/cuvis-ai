Status: Needs Review

This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

______________________________________________________________________

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

The port *name* is the key under which a spec is registered in a node's `INPUT_SPECS` / `OUTPUT_SPECS` dict, not a field on `PortSpec` itself. Input/output direction likewise comes from which dict the spec lives in, not from the spec.

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

Stage routing is controlled per node via `execution_stages`, not on `connect`. Connections themselves are stage-agnostic.

```python
# Connections carry no stage argument
pipeline.connect(normalizer.normalized, selector.data)
pipeline.connect(selector.selected, pca.features)
```

### Loss Nodes Without an Aggregator

`LossAggregator` has been removed—the trainer now collects individual loss nodes directly. Register every loss/regularizer node with the `GradientTrainer` (or any custom trainer) and feed their inputs through standard port connections, as shown in [examples/channel_selector.py](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook/blob/main/examples/channel_selector.py).

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
    training_config=training_cfg,
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

Batch keys are **bare port names**. A value is distributed to every node whose `INPUT_SPECS` declare a port of that name (an entry-point port left unconnected from any predecessor). Keys are not node-qualified.

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
1. **Define Shape Constraints**: Use fixed dimensions when possible for early error detection
1. **Document Ports**: Provide clear descriptions for each port
1. **Test Port Compatibility**: Validate connections during development
1. **Use Stage Filtering**: Leverage stage-aware execution for performance

## API Reference

## pipeline

Pipeline structure schemas.

### ConnectionConfig

Bases: `BaseSchemaModel`

Connection between two nodes using compact string format.

Attributes:

| Name     | Type  | Description                                   |
| -------- | ----- | --------------------------------------------- |
| `source` | `str` | Source endpoint in format "node.outputs.port" |
| `target` | `str` | Target endpoint in format "node.inputs.port"  |

#### from_node

```python
from_node
```

Source node name.

#### from_port

```python
from_port
```

Source port name.

#### to_node

```python
to_node
```

Target node name.

#### to_port

```python
to_port
```

Target port name.

### NodeConfig

Bases: `BaseSchemaModel`

Node configuration within a pipeline.

Attributes:

| Name         | Type             | Description                                            |
| ------------ | ---------------- | ------------------------------------------------------ |
| `name`       | `str`            | Node identifier / base name                            |
| `class_name` | `str`            | Fully-qualified class name (e.g., 'my_package.MyNode') |
| `hparams`    | `dict[str, Any]` | Node hyperparameters                                   |

### PipelineConfig

Bases: `BaseSchemaModel`

Pipeline structure configuration.

Attributes:

| Name          | Type                     | Description      |
| ------------- | ------------------------ | ---------------- |
| `plugins`     | \`list[str]              | None\`           |
| `nodes`       | `list[NodeConfig]`       | Node definitions |
| `connections` | `list[ConnectionConfig]` | Node connections |
| `metadata`    | \`PipelineMetadata       | None\`           |

#### save_to_file

```python
save_to_file(path)
```

Save pipeline configuration to YAML file.

Parameters:

| Name   | Type  | Description | Default          |
| ------ | ----- | ----------- | ---------------- |
| `path` | \`str | Path\`      | Output file path |

Source code in `.venv/lib/python3.11/site-packages/cuvis_ai_schemas/pipeline/config.py`

```python
def save_to_file(self, path: str | Path) -> None:
    """Save pipeline configuration to YAML file.

    Parameters
    ----------
    path : str | Path
        Output file path
    """
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(self.to_dict(), f, sort_keys=False)
```

#### load_from_file

```python
load_from_file(path)
```

Load pipeline configuration from YAML file.

Parameters:

| Name   | Type  | Description | Default         |
| ------ | ----- | ----------- | --------------- |
| `path` | \`str | Path\`      | Input file path |

Returns:

| Type             | Description          |
| ---------------- | -------------------- |
| `PipelineConfig` | Loaded configuration |

Source code in `.venv/lib/python3.11/site-packages/cuvis_ai_schemas/pipeline/config.py`

```python
@classmethod
def load_from_file(cls, path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file.

    Parameters
    ----------
    path : str | Path
        Input file path

    Returns
    -------
    PipelineConfig
        Loaded configuration
    """
    from pathlib import Path

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return cls.from_dict(data)
```

### PipelineMetadata

Bases: `BaseSchemaModel`

Pipeline metadata for documentation and discovery.

Attributes:

| Name               | Type        | Description                                                                                                                                                                   |
| ------------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`             | `str`       | Pipeline name                                                                                                                                                                 |
| `description`      | `str`       | Human-readable description                                                                                                                                                    |
| `created`          | `str`       | Creation timestamp (ISO format)                                                                                                                                               |
| `tags`             | `list[str]` | Tags for categorization and search                                                                                                                                            |
| `author`           | `str`       | Author name or email                                                                                                                                                          |
| `cuvis_ai_version` | `str`       | cuvis-ai-schemas version that created the pipeline (auto-stamped from the installed package; an explicitly recorded value, e.g. from an older snapshot, is preserved on load) |

#### to_proto

```python
to_proto()
```

Convert to proto message.

Uses field-by-field mapping (not config_bytes) because the proto message has typed fields that gRPC services access directly.

Source code in `.venv/lib/python3.11/site-packages/cuvis_ai_schemas/pipeline/config.py`

```python
def to_proto(self) -> cuvis_ai_pb2.PipelineMetadata:
    """Convert to proto message.

    Uses field-by-field mapping (not config_bytes) because the proto
    message has typed fields that gRPC services access directly.
    """
    from cuvis_ai_schemas.base import _get_pb2

    pb2 = _get_pb2()
    return pb2.PipelineMetadata(
        name=self.name,
        description=self.description,
        created=self.created,
        tags=list(self.tags),
        author=self.author,
        cuvis_ai_version=self.cuvis_ai_version,
    )
```

### PortCompatibilityError

Bases: `Exception`

Raised when attempting to connect incompatible ports.

### DimensionResolver

Utility class for resolving symbolic dimensions in port shapes.

#### resolve

```python
resolve(shape, node)
```

Resolve symbolic dimensions to concrete values.

Parameters:

| Name    | Type         | Description  | Default                                                                      |
| ------- | ------------ | ------------ | ---------------------------------------------------------------------------- |
| `shape` | \`tuple\[int | str, ...\]\` | Shape specification with flexible (-1), fixed (int), or symbolic (str) dims. |
| `node`  | \`Any        | None\`       | Node instance to resolve symbolic dimensions from.                           |

Returns:

| Type              | Description                                  |
| ----------------- | -------------------------------------------- |
| `tuple[int, ...]` | Resolved shape with concrete integer values. |

Raises:

| Type             | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `AttributeError` | If symbolic dimension references non-existent node attribute. |

Source code in `.venv/lib/python3.11/site-packages/cuvis_ai_schemas/pipeline/ports.py`

```python
@staticmethod
def resolve(
    shape: tuple[int | str, ...],
    node: Any | None,
) -> tuple[int, ...]:
    """Resolve symbolic dimensions to concrete values.

    Parameters
    ----------
    shape : tuple[int | str, ...]
        Shape specification with flexible (-1), fixed (int), or symbolic (str) dims.
    node : Any | None
        Node instance to resolve symbolic dimensions from.

    Returns
    -------
    tuple[int, ...]
        Resolved shape with concrete integer values.

    Raises
    ------
    AttributeError
        If symbolic dimension references non-existent node attribute.
    """
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, int):
            # Flexible (-1) or fixed (int) dimension
            resolved.append(dim)
            continue

        if isinstance(dim, str):
            # Symbolic dimension - resolve from node
            if node is None:
                raise ValueError(
                    f"Cannot resolve symbolic dimension '{dim}' without node instance"
                )
            if not hasattr(node, dim):
                node_label = getattr(node, "id", None) or node
                raise AttributeError(
                    f"Node {node_label} has no attribute '{dim}' for dimension resolution"
                )

            value = getattr(node, dim)
            if not isinstance(value, int):
                raise TypeError(f"Dimension '{dim}' resolved to {type(value)}, expected int")
            resolved.append(value)
            continue

        raise TypeError(f"Invalid dimension type: {type(dim)}")

    return tuple(resolved)
```

### InputPort

```python
InputPort(node, name, spec)
```

Proxy object representing a node's input port.

Initialize an input port proxy.

Parameters:

| Name   | Type       | Description                                                 | Default    |
| ------ | ---------- | ----------------------------------------------------------- | ---------- |
| `node` | `Any`      | The node instance that owns this port.                      | *required* |
| `name` | `str`      | The name of the port on the node.                           | *required* |
| `spec` | `PortSpec` | The port specification defining type and shape constraints. | *required* |

Source code in `.venv/lib/python3.11/site-packages/cuvis_ai_schemas/pipeline/ports.py`

```python
def __init__(self, node: Any, name: str, spec: PortSpec) -> None:
    """Initialize an input port proxy.

    Parameters
    ----------
    node : Any
        The node instance that owns this port.
    name : str
        The name of the port on the node.
    spec : PortSpec
        The port specification defining type and shape constraints.
    """
    self.node = node
    self.name = name
    self.spec = spec
```

### OutputPort

```python
OutputPort(node, name, spec)
```

Proxy object representing a node's output port.

Initialize an output port proxy.

Parameters:

| Name   | Type       | Description                                                 | Default    |
| ------ | ---------- | ----------------------------------------------------------- | ---------- |
| `node` | `Any`      | The node instance that owns this port.                      | *required* |
| `name` | `str`      | The name of the port on the node.                           | *required* |
| `spec` | `PortSpec` | The port specification defining type and shape constraints. | *required* |

Source code in `.venv/lib/python3.11/site-packages/cuvis_ai_schemas/pipeline/ports.py`

```python
def __init__(self, node: Any, name: str, spec: PortSpec) -> None:
    """Initialize an output port proxy.

    Parameters
    ----------
    node : Any
        The node instance that owns this port.
    name : str
        The name of the port on the node.
    spec : PortSpec
        The port specification defining type and shape constraints.
    """
    self.node = node
    self.name = name
    self.spec = spec
```

### PortSpec

```python
PortSpec(
    dtype,
    shape,
    description="",
    optional=False,
    variadic=False,
)
```

Specification for a node input or output port.

`variadic` marks an **input** port that accepts fan-in from multiple upstream connections (each conforming to this one spec); the node then receives a list of values for that port. It is meaningless on outputs.

#### resolve_shape

```python
resolve_shape(node)
```

Resolve symbolic dimensions in shape using node attributes.

Parameters:

| Name   | Type  | Description                                        | Default    |
| ------ | ----- | -------------------------------------------------- | ---------- |
| `node` | `Any` | Node instance to resolve symbolic dimensions from. | *required* |

Returns:

| Type              | Description                                                                      |
| ----------------- | -------------------------------------------------------------------------------- |
| `tuple[int, ...]` | Resolved shape with all symbolic dimensions replaced by concrete integer values. |

See Also

DimensionResolver.resolve : Underlying resolution logic.

Source code in `.venv/lib/python3.11/site-packages/cuvis_ai_schemas/pipeline/ports.py`

```python
def resolve_shape(self, node: Any) -> tuple[int, ...]:
    """Resolve symbolic dimensions in shape using node attributes.

    Parameters
    ----------
    node : Any
        Node instance to resolve symbolic dimensions from.

    Returns
    -------
    tuple[int, ...]
        Resolved shape with all symbolic dimensions replaced by concrete integer values.

    See Also
    --------
    DimensionResolver.resolve : Underlying resolution logic.
    """
    return DimensionResolver.resolve(self.shape, node)
```

#### is_compatible_with

```python
is_compatible_with(other, source_node, target_node)
```

Check if this port can connect to another port.

Parameters:

| Name          | Type       | Description                                                       | Default                              |
| ------------- | ---------- | ----------------------------------------------------------------- | ------------------------------------ |
| `other`       | `PortSpec` | Target port spec (a single spec; variadic is a flag, not a list). | *required*                           |
| `source_node` | \`Any      | None\`                                                            | Source node for dimension resolution |
| `target_node` | \`Any      | None\`                                                            | Target node for dimension resolution |

Returns:

| Type               | Description                    |
| ------------------ | ------------------------------ |
| `tuple[bool, str]` | (is_compatible, error_message) |

Source code in `.venv/lib/python3.11/site-packages/cuvis_ai_schemas/pipeline/ports.py`

```python
def is_compatible_with(
    self,
    other: PortSpec,
    source_node: Any | None,
    target_node: Any | None,
) -> tuple[bool, str]:
    """Check if this port can connect to another port.

    Parameters
    ----------
    other : PortSpec
        Target port spec (a single spec; variadic is a flag, not a list).
    source_node : Any | None
        Source node for dimension resolution
    target_node : Any | None
        Target node for dimension resolution

    Returns
    -------
    tuple[bool, str]
        (is_compatible, error_message)
    """
    torch = _require_torch()

    def _format_dtype(value: Any) -> str:
        """Format a dtype value for display in error messages.

        Parameters
        ----------
        value : Any
            A dtype value (torch.dtype, type, or other).

        Returns
        -------
        str
            Human-readable string representation of the dtype.
        """
        if isinstance(value, torch.dtype):
            return str(value)
        return getattr(value, "__name__", str(value))

    def _is_tensor_related(dtype: Any) -> bool:
        """Check if dtype is torch.Tensor or a specific torch.dtype.

        Parameters
        ----------
        dtype : Any
            The dtype to check.

        Returns
        -------
        bool
            True if dtype is torch.Tensor or a torch.dtype instance.
        """
        return dtype is torch.Tensor or isinstance(dtype, torch.dtype)

    # Check dtype compatibility with smart tensor handling
    source_is_tensor = _is_tensor_related(self.dtype)
    target_is_tensor = _is_tensor_related(other.dtype)

    if source_is_tensor and target_is_tensor:
        # Both tensor-related types
        # Allow if either is generic torch.Tensor OR both are same dtype
        if not (
            self.dtype is torch.Tensor
            or other.dtype is torch.Tensor
            or self.dtype == other.dtype
        ):
            return False, (
                f"Dtype mismatch: source has {_format_dtype(self.dtype)}, "
                f"target expects {_format_dtype(other.dtype)}"
            )
    elif self.dtype != other.dtype:
        # Non-tensor types must match exactly
        return False, (
            f"Dtype mismatch: source has {_format_dtype(self.dtype)}, "
            f"target expects {_format_dtype(other.dtype)}"
        )

    # Resolve shapes
    try:
        source_shape = self.resolve_shape(source_node) if source_node else self.shape
        target_shape = other.resolve_shape(target_node) if target_node else other.shape
    except (AttributeError, ValueError, TypeError) as exc:
        return False, f"Shape resolution failed: {exc}"

    # Check rank compatibility
    if len(source_shape) != len(target_shape):
        return False, (
            f"Shape rank mismatch: source has {len(source_shape)} dimensions, "
            f"target expects {len(target_shape)}"
        )

    # Check dimension-by-dimension compatibility
    for idx, (src_dim, tgt_dim) in enumerate(zip(source_shape, target_shape, strict=True)):
        # -1 means flexible, always compatible
        if src_dim == -1 or tgt_dim == -1:
            continue

        # Both fixed - must match exactly
        if src_dim != tgt_dim:
            return False, (
                f"Dimension {idx} mismatch: source has size {src_dim}, target expects {tgt_dim}"
            )

    return True, ""
```

### NodeProfilingStats

```python
NodeProfilingStats(
    node_name,
    stage,
    count,
    mean_ms,
    median_ms,
    std_ms,
    min_ms,
    max_ms,
    total_ms,
    last_ms,
)
```

Immutable snapshot of accumulated runtime statistics for a single node.

All timing values are in milliseconds.

Attributes:

| Name        | Type    | Description                                                                                       |
| ----------- | ------- | ------------------------------------------------------------------------------------------------- |
| `node_name` | `str`   | Unique node identifier within the pipeline (e.g. "DoubleNode" or "DoubleNode-2" for counter > 0). |
| `stage`     | `str`   | Canonical lowercase execution stage value from ExecutionStage.value (e.g. "inference", "train").  |
| `count`     | `int`   | Number of accumulated samples (after warm-up skip).                                               |
| `mean_ms`   | `float` | Online mean of recorded durations.                                                                |
| `median_ms` | `float` | Approximate median (P² estimator after warm-up buffer).                                           |
| `std_ms`    | `float` | Population standard deviation of recorded durations.                                              |
| `min_ms`    | `float` | Minimum recorded duration.                                                                        |
| `max_ms`    | `float` | Maximum recorded duration.                                                                        |
| `total_ms`  | `float` | Sum of all recorded durations.                                                                    |
| `last_ms`   | `float` | Most recently recorded duration.                                                                  |

## See Also

- **[Node Catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.6/catalogs/nodes/index.md)**: Node implementations with port specifications
- **[Pipeline API](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.6/reference/python-api/pipeline/index.md)**: Pipeline and connection management
- **[Core Concepts](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.6/concepts/index.md)**: Understand the architecture
- **[Quickstart](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.6/get-started/quickstart/index.md)**: Practical port usage examples
