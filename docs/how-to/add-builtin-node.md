!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# How-To: Add Built-in Nodes to CUVIS.AI

## Overview
Learn how to create custom nodes and integrate them into the CUVIS.AI framework. This guide covers node architecture, implementation patterns, testing, and documentation requirements.

## Prerequisites
- cuvis-ai development environment set up
- Understanding of [Node System Deep Dive](../concepts/node-system-deep-dive.md)
- Familiarity with PyTorch `nn.Module`
- Python type hints knowledge

## Node Architecture Overview

All CUVIS.AI nodes inherit from the `Node` base class, which provides:

- **Port system**: Typed input/output connections
- **Serialization**: Automatic config saving/loading
- **Lifecycle hooks**: Statistical initialization, freezing/unfreezing
- **PyTorch integration**: Full `nn.Module` compatibility

```python
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
import torch
import torch.nn as nn

class MyNode(Node):
    INPUT_SPECS = {...}   # Define input ports
    OUTPUT_SPECS = {...}  # Define output ports

    def __init__(self, ...):
        # Initialization logic
        pass

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        # Processing logic
        pass
```

## Step 1: Define Port Specifications

Ports define the data contracts for your node.

### Basic Port Definition

```python
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
import torch

class ThresholdFilter(Node):
    """Filters values below a threshold."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),  # BHWC format
            description="Input tensor to filter"
        ),
        "threshold": PortSpec(
            dtype=torch.float32,
            shape=(),
            description="Threshold value (optional at runtime)",
            optional=True
        )
    }

    OUTPUT_SPECS = {
        "filtered": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Filtered output"
        ),
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Binary mask of filtered values"
        )
    }
```

### Port Specification Guidelines

**Shape Dimensions:**
- `-1`: Dynamic dimension (batch size, spatial dimensions)
- `()`: Scalar value
- `(C,)`: Fixed-size vector (C channels)
- `(-1, -1, -1, C)`: BHWC format with fixed channels

**Data Types:**
- `torch.float32`: Standard floating point
- `torch.int32`: Integer labels/indices
- `torch.bool`: Binary masks
- `np.int32`: NumPy arrays (e.g., wavelengths)

**Optional Ports:**
- `optional=True`: Connection not required
- Useful for conditional inputs or auxiliary outputs

## Step 2: Implement Initialization

The `__init__` method sets up node parameters and registers buffers.

### Simple Stateless Node

```python
class ThresholdFilter(Node):
    """Filters values below a threshold."""

    INPUT_SPECS = {...}
    OUTPUT_SPECS = {...}

    def __init__(
        self,
        default_threshold: float = 0.5,
        invert: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize threshold filter.

        Parameters
        ----------
        default_threshold : float, optional
            Default threshold value. Default is 0.5.
        invert : bool, optional
            If True, keep values below threshold. Default is False.
        """
        self.default_threshold = default_threshold
        self.invert = invert

        # IMPORTANT: Pass parameters to super().__init__
        # This enables automatic serialization
        super().__init__(
            default_threshold=default_threshold,
            invert=invert,
            **kwargs
        )
```

### Stateful Node with Buffers

```python
class AdaptiveThresholdFilter(Node):
    """Filters using learned adaptive threshold."""

    INPUT_SPECS = {...}
    OUTPUT_SPECS = {...}

    def __init__(
        self,
        num_channels: int,
        init_threshold: float = 0.5,
        **kwargs
    ) -> None:
        """Initialize with per-channel thresholds."""
        self.num_channels = num_channels
        self.init_threshold = init_threshold

        super().__init__(
            num_channels=num_channels,
            init_threshold=init_threshold,
            **kwargs
        )

        # Register buffers (frozen, non-trainable)
        self.register_buffer(
            "thresholds",
            torch.full((num_channels,), init_threshold)
        )

        # Track initialization state
        self._statistically_initialized = False
```

### Trainable Node with Parameters

```python
class LearnableFilter(Node):
    """Filters using learnable neural network."""

    INPUT_SPECS = {...}
    OUTPUT_SPECS = {...}

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        **kwargs
    ) -> None:
        """Initialize learnable filter network."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )

        # Define neural network layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
```

## Step 3: Implement Forward Method

The `forward` method defines node processing logic.

### Basic Forward Implementation

```python
def forward(
    self,
    data: torch.Tensor,
    threshold: torch.Tensor | None = None,
    **_
) -> dict[str, torch.Tensor]:
    """
    Filter data based on threshold.

    Parameters
    ----------
    data : torch.Tensor
        Input data in BHWC format
    threshold : torch.Tensor, optional
        Override default threshold

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "filtered" and "mask" keys
    """
    # Use provided threshold or default
    thresh = threshold if threshold is not None else self.default_threshold

    # Create mask
    if self.invert:
        mask = data < thresh
    else:
        mask = data >= thresh

    # Apply filter
    filtered = data * mask.float()

    return {
        "filtered": filtered,
        "mask": mask
    }
```

### Forward with Error Handling

```python
def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
    """Process data with adaptive thresholds."""
    # Validation
    if not self._statistically_initialized:
        raise RuntimeError(
            f"{self.__class__.__name__} requires statistical_initialization() "
            "before processing. Call node.statistical_initialization(data_stream) first."
        )

    B, H, W, C = data.shape
    if C != self.num_channels:
        raise ValueError(
            f"Expected {self.num_channels} channels, got {C}. "
            f"Initialize node with correct num_channels parameter."
        )

    # Process with per-channel thresholds
    # Shape broadcasting: (B,H,W,C) >= (C,) → (B,H,W,C)
    mask = data >= self.thresholds.view(1, 1, 1, -1)
    filtered = data * mask.float()

    return {"filtered": filtered, "mask": mask}
```

## Step 4: Add Statistical Initialization (Optional)

For nodes that learn parameters from data.

### Basic Statistical Initialization

```python
def statistical_initialization(self, input_stream) -> None:
    """
    Learn adaptive thresholds from initialization data.

    Parameters
    ----------
    input_stream : iterable
        Iterator yielding batches with "data" key
    """
    # Accumulate statistics
    channel_sums = torch.zeros(self.num_channels)
    channel_counts = torch.zeros(self.num_channels)

    for batch in input_stream:
        data = batch["data"]  # Shape: (B, H, W, C)
        B, H, W, C = data.shape

        # Accumulate per-channel statistics
        channel_sums += data.sum(dim=(0, 1, 2))
        channel_counts += B * H * W

    # Compute mean as threshold
    channel_means = channel_sums / channel_counts

    # Update buffers
    self.thresholds.copy_(channel_means)
    self._statistically_initialized = True
```

### Advanced: Welford's Online Algorithm

For numerically stable mean/variance computation ([reference](https://www.johndcook.com/blog/standard_deviation/)):

```python
def statistical_initialization(self, input_stream) -> None:
    """Initialize using Welford's algorithm for numerical stability."""
    count = 0
    mean = torch.zeros(self.num_channels)
    M2 = torch.zeros(self.num_channels)  # Sum of squared differences

    for batch in input_stream:
        data = batch["data"]
        B, H, W, C = data.shape

        # Flatten spatial dimensions
        flat_data = data.reshape(-1, C)  # (B*H*W, C)

        for value in flat_data:
            count += 1
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            M2 += delta * delta2

    # Compute variance and std
    variance = M2 / count
    std = torch.sqrt(variance + 1e-8)

    # Set threshold as mean + 2*std
    self.thresholds.copy_(mean + 2 * std)
    self._statistically_initialized = True
```

## Step 5: Add Unfreeze Method (Optional)

For nodes that transition from frozen statistics to trainable parameters.

### Two-Phase Training Pattern

```python
def unfreeze(self) -> None:
    """
    Convert frozen buffers to trainable parameters.

    Call after statistical_initialization() to enable gradient training.
    """
    if not self._statistically_initialized:
        raise RuntimeError(
            "Must call statistical_initialization() before unfreeze()"
        )

    # Convert buffer to parameter
    if isinstance(self.thresholds, torch.Tensor) and not isinstance(self.thresholds, nn.Parameter):
        self.thresholds = nn.Parameter(self.thresholds.clone())

    # Call parent unfreeze (handles other components)
    super().unfreeze()
```

### Usage Pattern

**For testing/development** (direct method call):

```python
# Phase 1: Statistical initialization
node = AdaptiveThresholdFilter(num_channels=61)
node.statistical_initialization(initialization_data)  # Direct call for testing

# At this point, thresholds are frozen buffers
print(node.thresholds.requires_grad)  # False

# Phase 2: Enable gradient training
node.unfreeze()
print(node.thresholds.requires_grad)  # True

# Now can train with backpropagation
optimizer = torch.optim.Adam(node.parameters(), lr=0.001)
```

**In production pipelines** (recommended):

```python
from cuvis_ai_core.training import StatisticalTrainer

# Add node to pipeline
node = AdaptiveThresholdFilter(num_channels=61)
pipeline.add_node(node)

# Phase 1: Statistical initialization
trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()  # Automatically initializes all nodes that need it

# Phase 2: Enable gradient training
node.unfreeze()
grad_trainer = GradientTrainer(pipeline=pipeline, datamodule=datamodule, ...)
grad_trainer.fit()
```

## Step 6: Add Documentation

Follow NumPy docstring style for consistency with SciPy/scikit-learn.

### Complete Docstring Example

```python
class AdaptiveThresholdFilter(Node):
    """
    Adaptive threshold filter with per-channel learned thresholds.

    This node learns optimal threshold values from training data and applies
    them independently to each spectral channel. Useful for hyperspectral
    anomaly detection where different wavelengths have different baseline
    intensities.

    Parameters
    ----------
    num_channels : int
        Number of spectral channels in input data
    init_threshold : float, optional
        Initial threshold value for all channels. Default is 0.5.

    Attributes
    ----------
    thresholds : torch.Tensor
        Per-channel threshold values (num_channels,)

    Raises
    ------
    RuntimeError
        If forward() called before statistical_initialization()
    ValueError
        If input channel count doesn't match num_channels

    See Also
    --------
    ThresholdFilter : Fixed threshold filtering
    MinMaxNormalizer : Normalization before thresholding

    Notes
    -----
    The statistical initialization computes per-channel means and sets
    thresholds to mean + 2*std to capture ~95% of normal variation.

    Memory complexity: O(C) where C is num_channels
    Time complexity: O(N*C) for initialization, O(B*H*W*C) for forward pass

    Examples
    --------
    Statistical initialization:

    >>> node = AdaptiveThresholdFilter(num_channels=61)
    >>> node.statistical_initialization(initialization_data)
    >>> result = node.forward(data=test_tensor)
    >>> result["filtered"].shape
    torch.Size([1, 256, 256, 61])

    Gradient-based training:

    >>> node = AdaptiveThresholdFilter(num_channels=61)
    >>> node.statistical_initialization(initialization_data)
    >>> node.unfreeze()  # Enable gradient descent
    >>>
    >>> optimizer = torch.optim.Adam(node.parameters(), lr=0.001)
    >>> for epoch in range(50):
    ...     result = node.forward(data=train_data)
    ...     loss = criterion(result["filtered"], targets)
    ...     loss.backward()
    ...     optimizer.step()
    """

    INPUT_SPECS = {...}
    OUTPUT_SPECS = {...}

    # Implementation...
```

## Step 7: Register Node

Make your node discoverable by the framework.

### Decorator Registration

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

@NodeRegistry.register
class AdaptiveThresholdFilter(Node):
    """Your node implementation."""
    pass
```

### Manual Registration

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai.node.filters import AdaptiveThresholdFilter

# Register after class definition
NodeRegistry.register(AdaptiveThresholdFilter)

# Verify registration
node_class = NodeRegistry.get("AdaptiveThresholdFilter")
assert node_class is AdaptiveThresholdFilter
```

### List Available Nodes

```python
# List all built-in nodes
all_nodes = NodeRegistry.list_builtin_nodes()
print(f"Available nodes: {len(all_nodes)}")
for name in sorted(all_nodes.keys()):
    print(f"  - {name}")
```

## Step 8: Write Tests

Comprehensive testing ensures node reliability.

### Basic Test Structure

```python
import pytest
import torch
from cuvis_ai.node.filters import AdaptiveThresholdFilter

class TestAdaptiveThresholdFilter:
    """Test suite for AdaptiveThresholdFilter node."""

    def test_creation(self):
        """Test node can be created with valid parameters."""
        node = AdaptiveThresholdFilter(num_channels=61, init_threshold=0.5)
        assert node.num_channels == 61
        assert node.init_threshold == 0.5
        assert not node._statistically_initialized

    def test_forward_requires_initialization(self):
        """Test forward raises error before initialization."""
        node = AdaptiveThresholdFilter(num_channels=61)
        data = torch.randn(1, 10, 10, 61)

        with pytest.raises(RuntimeError, match="requires statistical_initialization"):
            node.forward(data=data)

    def test_statistical_initialization(self):
        """Test statistical initialization from data."""
        node = AdaptiveThresholdFilter(num_channels=3)

        # Create initialization data
        data_stream = [
            {"data": torch.tensor([[[[1.0, 2.0, 3.0]]]])},
            {"data": torch.tensor([[[[2.0, 3.0, 4.0]]]])},
            {"data": torch.tensor([[[[3.0, 4.0, 5.0]]]])},
        ]

        node.statistical_initialization(data_stream)

        assert node._statistically_initialized
        assert node.thresholds.shape == (3,)
        # Thresholds should be close to means [2.0, 3.0, 4.0]
        assert torch.allclose(node.thresholds, torch.tensor([2.0, 3.0, 4.0]), atol=0.5)

    def test_forward_after_initialization(self):
        """Test forward pass after initialization."""
        node = AdaptiveThresholdFilter(num_channels=3)

        # Initialize
        data_stream = [{"data": torch.tensor([[[[1.0, 2.0, 3.0]]]]))}]
        node.statistical_initialization(data_stream)

        # Forward pass
        test_data = torch.tensor([[[[0.5, 2.0, 4.0]]]])
        result = node.forward(data=test_data)

        assert "filtered" in result
        assert "mask" in result
        assert result["filtered"].shape == (1, 1, 1, 3)
        assert result["mask"].dtype == torch.bool

    def test_unfreeze_conversion(self):
        """Test buffer to parameter conversion."""
        node = AdaptiveThresholdFilter(num_channels=3)

        # Initialize (creates buffer)
        data_stream = [{"data": torch.ones(1, 1, 1, 3)}]
        node.statistical_initialization(data_stream)
        assert not node.thresholds.requires_grad

        # Unfreeze (converts to parameter)
        node.unfreeze()
        assert node.thresholds.requires_grad
        assert isinstance(node.thresholds, torch.nn.Parameter)

    def test_channel_mismatch_error(self):
        """Test error on channel count mismatch."""
        node = AdaptiveThresholdFilter(num_channels=61)
        data_stream = [{"data": torch.ones(1, 10, 10, 61)}]
        node.statistical_initialization(data_stream)

        # Wrong number of channels
        wrong_data = torch.ones(1, 10, 10, 30)
        with pytest.raises(ValueError, match="Expected 61 channels, got 30"):
            node.forward(data=wrong_data)

    def test_serialization(self):
        """Test node can be serialized and deserialized."""
        node = AdaptiveThresholdFilter(num_channels=61, init_threshold=0.7)

        # Serialize to dict
        config = node.serialize()
        assert config["class"] == "AdaptiveThresholdFilter"
        assert config["params"]["num_channels"] == 61
        assert config["params"]["init_threshold"] == 0.7

        # Deserialize
        from cuvis_ai_core.node import Node
        restored = Node.from_config(config)
        assert isinstance(restored, AdaptiveThresholdFilter)
        assert restored.num_channels == 61
```

### Test Coverage Checklist

- [ ] Node creation with various parameter combinations
- [ ] Forward pass with valid inputs
- [ ] Forward pass with optional inputs
- [ ] Error handling (wrong shapes, uninitialized, invalid values)
- [ ] Statistical initialization (if applicable)
- [ ] Unfreeze behavior (if applicable)
- [ ] Serialization and deserialization
- [ ] Integration with pipeline

## Node Implementation Patterns

### Pattern 1: Stateless Transformation

For simple deterministic transforms:

```python
class SquareTransform(Node):
    """Squares all input values."""

    INPUT_SPECS = {"data": PortSpec(torch.float32, (-1, -1, -1, -1))}
    OUTPUT_SPECS = {"squared": PortSpec(torch.float32, (-1, -1, -1, -1))}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._statistically_initialized = True  # No initialization needed

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        return {"squared": data ** 2}
```

### Pattern 2: Statistical Node (Frozen)

For nodes with learned statistics:

```python
class ZScoreNormalizer(Node):
    """Normalizes using learned mean and std."""

    def __init__(self, num_channels: int, eps: float = 1e-8, **kwargs):
        self.num_channels = num_channels
        self.eps = eps
        super().__init__(num_channels=num_channels, eps=eps, **kwargs)

        self.register_buffer("mean", torch.zeros(num_channels))
        self.register_buffer("std", torch.ones(num_channels))
        self._statistically_initialized = False

    def statistical_initialization(self, input_stream) -> None:
        # Compute mean and std from data
        ...
        self._statistically_initialized = True

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        if not self._statistically_initialized:
            raise RuntimeError("Call statistical_initialization() first")
        normalized = (data - self.mean) / (self.std + self.eps)
        return {"normalized": normalized}
```

### Pattern 3: Two-Phase Trainable

For nodes with statistical init + gradient training:

```python
class AdaptiveScaler(Node):
    """Learns adaptive scaling factors."""

    def __init__(self, num_channels: int, **kwargs):
        self.num_channels = num_channels
        super().__init__(num_channels=num_channels, **kwargs)

        self.register_buffer("scale_factors", torch.ones(num_channels))
        self._statistically_initialized = False

    def statistical_initialization(self, input_stream) -> None:
        # Initialize scale_factors from data statistics
        ...
        self._statistically_initialized = True

    def unfreeze(self) -> None:
        # Convert buffer to parameter
        self.scale_factors = nn.Parameter(self.scale_factors.clone())
        super().unfreeze()

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        scaled = data * self.scale_factors.view(1, 1, 1, -1)
        return {"scaled": scaled}
```

### Pattern 4: Deep Learning Node

For neural network-based nodes:

```python
class CNNFeatureExtractor(Node):
    """Extracts features using CNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: list[int] = [64, 32],
        **kwargs
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dims=hidden_dims,
            **kwargs
        )

        # Build CNN layers
        layers = []
        prev_dim = in_channels
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Conv2d(prev_dim, out_channels, kernel_size=1))
        self.encoder = nn.Sequential(*layers)

        # No statistical initialization needed
        self._statistically_initialized = True

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        # BHWC → BCHW
        data_chw = data.permute(0, 3, 1, 2)
        features = self.encoder(data_chw)
        # BCHW → BHWC
        features_hwc = features.permute(0, 2, 3, 1)
        return {"features": features_hwc}
```

## Integration with Pipeline

### Add Node to YAML Pipeline

```yaml
nodes:
  - name: adaptive_filter
    class: cuvis_ai.node.filters.AdaptiveThresholdFilter
    params:
      num_channels: 61
      init_threshold: 0.5

connections:
  - from: normalizer.outputs.normalized
    to: adaptive_filter.inputs.data
  - from: adaptive_filter.outputs.filtered
    to: detector.inputs.data
```

### Add Node in Python

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai.node.filters import AdaptiveThresholdFilter

pipeline = CuvisPipeline("my_pipeline")

# Add node
filter_node = pipeline.add_node(
    "adaptive_filter",
    AdaptiveThresholdFilter,
    num_channels=61,
    init_threshold=0.5
)

# Connect to pipeline
pipeline.connect("normalizer.outputs.normalized", "adaptive_filter.inputs.data")
pipeline.connect("adaptive_filter.outputs.filtered", "detector.inputs.data")
```

## Contributing to CUVIS.AI

### Pre-contribution Checklist

- [ ] Node implements focused, single-responsibility functionality
- [ ] Complete docstrings (NumPy style) with examples
- [ ] Full type hints on all methods
- [ ] Comprehensive test suite (>90% coverage)
- [ ] Follows existing code style (ruff formatting)
- [ ] Added to appropriate module (data, normalization, selectors, etc.)
- [ ] Registered with NodeRegistry
- [ ] Documentation added to [Node Catalog](../node-catalog/index.md)

### File Organization

```
cuvis_ai/
└── node/
    ├── __init__.py           # Export your node here
    ├── data.py               # Data loading nodes
    ├── normalization.py      # Normalization nodes
    ├── filters.py            # ← Add filtering nodes here
    └── ...
```

### Export Node

In `cuvis_ai/node/__init__.py`:

```python
from cuvis_ai.node.filters import (
    ThresholdFilter,
    AdaptiveThresholdFilter,
)

__all__ = [
    ...,
    "ThresholdFilter",
    "AdaptiveThresholdFilter",
]
```

### Add to Documentation

Create entry in [Node Catalog](../node-catalog/index.md):

```markdown
## Filtering Nodes

### AdaptiveThresholdFilter

**Module:** `cuvis_ai.node.filters`

Adaptive threshold filter with per-channel learned thresholds.

**When to use:**
- Need channel-specific filtering
- Want data-driven threshold selection
- Require gradient-based threshold optimization

**Parameters:**
- `num_channels` (int): Number of spectral channels
- `init_threshold` (float): Initial threshold value

**Ports:**
- Input: `data` (BHWC float32 tensor)
- Output: `filtered` (BHWC float32 tensor), `mask` (BHWC bool tensor)

**Example:**
\`\`\`python
from cuvis_ai_core.training import StatisticalTrainer

node = AdaptiveThresholdFilter(num_channels=61)
pipeline.add_node(node)

# Initialize using trainer
trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()

result = node.forward(data=test_data)
\`\`\`

**See also:** ThresholdFilter, MinMaxNormalizer
```

## Troubleshooting

### Issue: Port Connection Error
```
ValueError: Port 'output' not found on node 'my_node'
```
**Solution:** Check port names in INPUT_SPECS/OUTPUT_SPECS match connection strings.

### Issue: Serialization Fails
```
TypeError: __init__() got unexpected keyword argument
```
**Solution:** Ensure all `__init__` parameters are passed to `super().__init__()`:
```python
def __init__(self, param1, param2, **kwargs):
    self.param1 = param1
    self.param2 = param2
    # MUST pass all params to super
    super().__init__(param1=param1, param2=param2, **kwargs)
```

### Issue: Gradient Not Flowing
```
RuntimeError: element 0 of tensors does not require grad
```
**Solution:** Call `unfreeze()` to convert buffers to parameters:
```python
node.statistical_initialization(data)
node.unfreeze()  # Enable gradients
optimizer = torch.optim.Adam(node.parameters())
```

### Issue: Shape Mismatch
```
RuntimeError: shape '[1, 256, 256, 61]' is invalid for input of size 3932160
```
**Solution:** Check tensor format (BHWC vs BCHW). CUVIS.AI uses BHWC:
```python
# Wrong: BCHW format
data = torch.randn(1, 61, 256, 256)

# Correct: BHWC format
data = torch.randn(1, 256, 256, 61)
```

## Best Practices

### 1. Keep Nodes Focused

```python
# Good: Single responsibility
class ChannelMeanCalculator(Node):
    """Computes mean across channels."""
    pass

# Avoid: Multiple responsibilities
class ChannelMeanAndVarianceAndSkewnessCalculator(Node):
    """Computes many statistics."""  # Too complex, split into multiple nodes
    pass
```

### 2. Validate Inputs Thoroughly

```python
def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
    # Check initialization
    if not self._statistically_initialized:
        raise RuntimeError(f"{self.__class__.__name__} requires initialization")

    # Check shape
    if data.ndim != 4:
        raise ValueError(f"Expected 4D tensor (BHWC), got {data.ndim}D")

    B, H, W, C = data.shape
    if C != self.num_channels:
        raise ValueError(
            f"Expected {self.num_channels} channels, got {C}. "
            f"Reinitialize node with num_channels={C}"
        )

    # Check value range
    if data.min() < 0 or data.max() > 1:
        raise ValueError("Data must be in [0, 1] range. Apply normalization first.")

    # Process data
    ...
```

### 3. Use Type Hints Consistently

```python
from typing import Any
import torch

def forward(
    self,
    data: torch.Tensor,
    mask: torch.Tensor | None = None,
    **_: Any
) -> dict[str, torch.Tensor]:
    """Type hints improve IDE support and catch errors early."""
    pass
```

### 4. Document Edge Cases

```python
def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
    """
    Process input data.

    Parameters
    ----------
    data : torch.Tensor
        Input in BHWC format

    Returns
    -------
    dict[str, torch.Tensor]
        Processed output

    Notes
    -----
    - Returns zeros for all-zero input (graceful degradation)
    - Handles NaN values by replacing with zero
    - Preserves gradient flow even when mask is all-False
    """
    # Handle edge cases explicitly
    if torch.isnan(data).any():
        data = torch.nan_to_num(data, nan=0.0)

    ...
```

### 5. Test Edge Cases

```python
def test_edge_cases():
    """Test boundary conditions."""
    node = MyNode(num_channels=3)
    node.statistical_initialization([{"data": torch.ones(1, 1, 1, 3)}])

    # Test zero input
    result = node.forward(data=torch.zeros(1, 10, 10, 3))
    assert not torch.isnan(result["output"]).any()

    # Test near-zero values
    result = node.forward(data=torch.full((1, 10, 10, 3), 1e-10))
    assert torch.isfinite(result["output"]).all()

    # Test large values
    result = node.forward(data=torch.full((1, 10, 10, 3), 1e6))
    assert torch.isfinite(result["output"]).all()
```

## See Also
- [Node System Deep Dive](../concepts/node-system-deep-dive.md)
- [Node Catalog](../node-catalog/index.md)
- [Build Pipelines in Python](build-pipeline-python.md)
- [Build Pipelines in YAML](build-pipeline-yaml.md)
- [Contributing Guide](../../CONTRIBUTING.md)
