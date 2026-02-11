!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Preprocessing Nodes

## Overview

Preprocessing nodes prepare hyperspectral data for downstream processing by normalizing values, filtering spectral bands, and standardizing formats. These transformations are critical for numerical stability, gradient flow, and feature extraction in both statistical and deep learning pipelines.

**When to use Preprocessing Nodes:**

- **Normalization**: Scale data to standard ranges ([0,1], z-scores) for gradient stability
- **Band filtering**: Reduce spectral dimensionality by selecting wavelength ranges
- **Standardization**: Remove mean/scale by variance for statistical methods
- **Transformation**: Apply nonlinear transformations (sigmoid) for bounded outputs

**Key concepts:**

- **Per-sample vs global normalization**: Some nodes compute statistics per batch (ZScoreNormalizer), others use running statistics from initialization (MinMaxNormalizer with `use_running_stats=True`)
- **Differentiability**: All nodes preserve gradients for end-to-end training
- **Statistical initialization**: Some nodes (MinMaxNormalizer) can be initialized with dataset statistics for consistent normalization

## Nodes in This Category

### BandpassByWavelength

**Description:** Filters hyperspectral cubes to a specified wavelength range

**Perfect for:**

- Removing noisy spectral bands at edges (e.g., <450nm, >900nm)
- Focusing on specific absorption features (e.g., 650-750nm for chlorophyll)
- Reducing dimensionality before band selection
- Standardizing spectral ranges across different sensors

**Training Paradigm:** None (fixed wavelength filtering)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| data | float32 | (B, H, W, C) | Input hyperspectral cube | No |
| wavelengths | int32 | (C,) | Wavelength values (nm) for each channel | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| filtered | float32 | (B, H, W, C_filtered) | Cube with selected channels only |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| min_wavelength_nm | float | Required | Minimum wavelength (inclusive) to keep |
| max_wavelength_nm | float \| None | None | Maximum wavelength (inclusive). If None, keeps all wavelengths >= min |

#### Example Usage (Python)

```python
from cuvis_ai.node.preprocessors import BandpassByWavelength

# Filter to visible-NIR range (500-800nm)
bandpass = BandpassByWavelength(
    min_wavelength_nm=500.0,
    max_wavelength_nm=800.0,
)

# Use in pipeline (wavelengths from data node)
pipeline.connect(
    (data_node.cube, bandpass.data),
    (data_node.wavelengths, bandpass.wavelengths),
    (bandpass.filtered, normalizer.data),
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  bandpass:
    type: BandpassByWavelength
    config:
      min_wavelength_nm: 500.0
      max_wavelength_nm: 800.0

connections:
  - [data.cube, bandpass.data]
  - [data.wavelengths, bandpass.wavelengths]
  - [bandpass.filtered, normalizer.data]
```

#### Common Use Cases

**1. Remove atmospheric absorption bands**

```python
# Filter out water absorption at 1400nm and 1900nm
bandpass = BandpassByWavelength(
    min_wavelength_nm=430.0,
    max_wavelength_nm=1300.0,  # Stop before 1400nm water absorption
)
```

**2. Focus on specific spectral features**

```python
# Extract red-edge region for vegetation analysis
red_edge = BandpassByWavelength(
    min_wavelength_nm=680.0,
    max_wavelength_nm=750.0,
)
```

**3. Progressive filtering**

```python
# Chain multiple bandpasses for complex filtering
# First: Remove edges
stage1 = BandpassByWavelength(min_wavelength_nm=450.0, max_wavelength_nm=900.0)
# Second: Extract specific region from filtered data
stage2 = BandpassByWavelength(min_wavelength_nm=600.0, max_wavelength_nm=700.0)
```

#### See Also

- [Tutorial 3: Deep SVDD Gradient](../tutorials/deep-svdd-gradient.md#step-2-preprocessing-chain) - Uses bandpass filtering
- [Concept: Port System](../concepts/port-system-deep-dive.md) - Port connections
- API Reference: ::: cuvis_ai.node.preprocessors.BandpassByWavelength

---

### MinMaxNormalizer

**Description:** Scales data to [0, 1] range using min-max normalization

**Perfect for:**

- Normalizing hyperspectral cubes for visualization
- Preparing data for sigmoid-based activations
- Ensuring bounded inputs for certain loss functions
- Providing consistent scale for statistical methods

**Training Paradigm:** Statistical initialization (optional, if `use_running_stats=True`)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| data | float32 | (B, H, W, C) | Input tensor to normalize | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| normalized | float32 | (B, H, W, C) | Min-max normalized tensor in [0, 1] |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| eps | float | 1e-6 | Small constant to prevent division by zero |
| use_running_stats | bool | True | If True, use global min/max from statistical initialization. If False, compute per-sample min/max |

#### Normalization Modes

**Mode 1: Global normalization** (`use_running_stats=True`, default)

- Requires statistical initialization to compute global min/max
- Normalizes all samples consistently: `(x - global_min) / (global_max - global_min)`
- Best for consistent scaling across train/val/test

**Mode 2: Per-sample normalization** (`use_running_stats=False`)

- No initialization required
- Computes min/max per batch: `(x - batch_min) / (batch_max - batch_min)`
- Best for visualization or when each sample has different dynamic range

#### Example Usage (Python)

**Global normalization with statistical initialization**

```python
from cuvis_ai.node.normalization import MinMaxNormalizer

# Create normalizer (requires initialization)
from cuvis_ai_core.training import StatisticalTrainer

normalizer = MinMaxNormalizer(
    eps=1e-6,
    use_running_stats=True,  # Use global statistics
)
pipeline.add_node(normalizer)

# Statistical initialization (run once)
trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()  # Automatically initializes normalizer

# Now use in pipeline
outputs = normalizer.forward(data=cube)  # Normalized to [0, 1] using global min/max
```

**Per-sample normalization (no initialization)**

```python
# Create normalizer without statistical init
normalizer = MinMaxNormalizer(
    eps=1e-6,
    use_running_stats=False,  # Per-sample normalization
)

# Use directly (no initialization needed)
outputs = normalizer.forward(data=cube)  # Each sample normalized to its own [0, 1]
```

#### Example Configuration (YAML)

```yaml
nodes:
  normalizer:
    type: MinMaxNormalizer
    config:
      eps: 1e-6
      use_running_stats: true  # Requires statistical initialization
```

#### Statistical Initialization

```python
# Create initialization stream
from cuvis_ai_core.pipeline.stream import DataLoaderInputStream

from cuvis_ai_core.training import StatisticalTrainer

# Initialize normalizer
trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()  # Automatically initializes normalizer

# Running statistics are now stored in normalizer.running_min and normalizer.running_max
```

#### Common Issues

**Issue: Normalized values outside [0, 1]**

```python
# Problem: Test data has values outside training range
normalizer.running_min = 0.0  # From training data
normalizer.running_max = 1.0
test_data = torch.tensor([[[[-0.5, 1.5]]] ])  # Values outside [0, 1]
normalized = normalizer.forward(data=test_data)["normalized"]
# Output: [[[[-0.5, 1.5]]]] (no clamping)

# Solution: Clip values if needed
normalized_clipped = torch.clamp(normalized, 0.0, 1.0)
```

#### See Also

- [Tutorial 1: RX Statistical](../tutorials/rx-statistical.md#step-2-preprocessing) - Uses MinMaxNormalizer
- [Tutorial 2: Channel Selector](../tutorials/channel-selector.md#normalization) - Statistical initialization example
- [Concept: Two-Phase Training](../concepts/two-phase-training.md#phase-1-statistical-initialization) - Statistical init workflow
- API Reference: ::: cuvis_ai.node.normalization.MinMaxNormalizer

---

### ZScoreNormalizer

**Description:** Applies z-score (standardization) normalization: (x - mean) / std

**Perfect for:**

- Preparing data for deep learning (zero mean, unit variance)
- Statistical anomaly detection (RX, Mahalanobis distance)
- Removing per-sample brightness variations
- Gradient-based optimization (improved convergence)

**Training Paradigm:** None (per-sample normalization)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| data | float32 | (B, H, W, C) | Input tensor | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| normalized | float32 | (B, H, W, C) | Z-score normalized tensor |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dims | list[int] | [1, 2] | Dimensions to compute statistics over (1=H, 2=W in BHWC format) |
| eps | float | 1e-6 | Stability constant for division |
| keepdim | bool | True | Whether to keep reduced dimensions for broadcasting |

#### Normalization Behavior

**Default (dims=[1, 2])**: Standardize over spatial dimensions (H, W)

- Computes mean and std across height and width
- Keeps batch and channel dimensions separate
- Result: Each channel per sample has zero mean and unit variance

**Alternative (dims=[1, 2, 3])**: Standardize over spatial + channel dimensions

- Computes mean and std across H, W, and C
- Result: Each sample has zero mean and unit variance globally

#### Example Usage (Python)

```python
from cuvis_ai.node.normalization import ZScoreNormalizer

# Default: Normalize over spatial dimensions (H, W)
zscore = ZScoreNormalizer(
    dims=[1, 2],  # Height and width
    eps=1e-6,
)

# Use in pipeline
outputs = zscore.forward(data=cube)  # Shape: [B, H, W, C]
# Each (B, C) slice has mean≈0, std≈1
```

**Global standardization**

```python
# Normalize over all spatial and spectral dimensions
zscore_global = ZScoreNormalizer(
    dims=[1, 2, 3],  # H, W, C
    eps=1e-6,
)

outputs = zscore_global.forward(data=cube)
# Each batch element has global mean≈0, std≈1
```

#### Example Configuration (YAML)

```yaml
nodes:
  zscore:
    type: ZScoreNormalizer
    config:
      dims: [1, 2]  # Normalize over spatial dimensions
      eps: 1e-6
      keepdim: true
```

#### Comparison with Other Normalizers

| Normalizer | Output Range | Statistics | Use Case |
|------------|--------------|------------|----------|
| MinMaxNormalizer | [0, 1] | Min/max | Bounded outputs, visualization |
| **ZScoreNormalizer** | **(-∞, ∞)** | **Mean/std** | **Deep learning, gradient optimization** |
| SigmoidNormalizer | [0, 1] | Median/std | Robust to outliers |
| PerPixelUnitNorm | [-1, 1] (approx) | Per-pixel L2 | Unit-norm features |

#### ZScoreNormalizerGlobal

A specialized variant of `ZScoreNormalizer` for Deep SVDD training. This normalizer computes statistics globally across all dimensions for consistent encoding initialization. See [Deep SVDD Tutorial](../tutorials/deep-svdd-gradient.md) for usage examples.

#### See Also

- [Tutorial 2: Channel Selector](../tutorials/channel-selector.md#preprocessing) - Uses ZScoreNormalizer
- [Concept: Two-Phase Training](../concepts/two-phase-training.md) - When to use z-score
- API Reference: ::: cuvis_ai.node.normalization.ZScoreNormalizer

---

### SigmoidNormalizer

**Description:** Median-centered sigmoid squashing to [0, 1] range

**Perfect for:**

- Robust normalization when data has outliers
- Converting unbounded scores to probabilities
- Visualization of anomaly scores
- Preprocessing for loss functions expecting [0, 1] inputs

**Training Paradigm:** None (per-sample normalization)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| data | float32 | (B, H, W, C) | Input tensor | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| normalized | float32 | (B, H, W, C) | Sigmoid-normalized tensor in [0, 1] |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| std_floor | float | 1e-6 | Minimum standard deviation (prevents division by zero) |

#### Normalization Formula

```
output = sigmoid((x - median) / std)
```

Where:

- `median`: Per-sample median across spatial dimensions
- `std`: Per-sample standard deviation (clamped to std_floor)
- `sigmoid(z) = 1 / (1 + exp(-z))`

**Properties:**

- Median maps to 0.5
- Values > median → (0.5, 1.0)
- Values < median → (0.0, 0.5)
- Robust to extreme outliers (sigmoid saturation)

#### Example Usage (Python)

```python
from cuvis_ai.node.normalization import SigmoidNormalizer

# Create sigmoid normalizer
sigmoid_norm = SigmoidNormalizer(std_floor=1e-6)

# Use in pipeline
outputs = sigmoid_norm.forward(data=scores)  # Scores → [0, 1]
```

**Comparison with MinMaxNormalizer**

```python
# MinMaxNormalizer: Sensitive to outliers
data = torch.tensor([[[[1.0, 2.0, 3.0, 100.0]]]])  # Outlier: 100.0
minmax = MinMaxNormalizer(use_running_stats=False)
minmax_out = minmax.forward(data=data)["normalized"]
# Output: [0.0, 0.01, 0.02, 1.0] - outlier dominates

# SigmoidNormalizer: Robust to outliers
sigmoid = SigmoidNormalizer()
sigmoid_out = sigmoid.forward(data=data)["normalized"]
# Output: [0.27, 0.5, 0.73, ~1.0] - better distribution
```

#### Example Configuration (YAML)

```yaml
nodes:
  sigmoid_norm:
    type: SigmoidNormalizer
    config:
      std_floor: 1e-6
```

#### When to Use

**Use SigmoidNormalizer when:**

- Data contains outliers that skew min/max normalization
- You need smooth, bounded outputs for visualization
- Inputs are unbounded (e.g., RX scores, z-scores)

**Use MinMaxNormalizer when:**

- Data is clean without outliers
- You need exact [0, 1] mapping of min/max values
- Consistent scaling across train/test is critical

#### See Also

- [SigmoidTransform](#sigmoidtransform) - Simple sigmoid without centering
- API Reference: ::: cuvis_ai.node.normalization.SigmoidNormalizer

---

### PerPixelUnitNorm

**Description:** Per-pixel mean-centering and L2 normalization across spectral channels

**Perfect for:**

- Creating unit-norm feature vectors for cosine similarity
- Removing per-pixel brightness variations
- Preprocessing for Deep SVDD (hypersphere embedding)
- Spectral signature normalization

**Training Paradigm:** None (per-sample normalization)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| data | float32 | (B, H, W, C) | Input cube | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| normalized | float32 | (B, H, W, C) | L2-normalized cube (unit norm per pixel) |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| eps | float | 1e-8 | Stability constant for L2 norm (prevents division by zero) |

#### Normalization Formula

For each pixel (h, w) in the spatial grid:

```
1. centered = x[h, w, :] - mean(x[h, w, :])
2. norm = ||centered||_2
3. output[h, w, :] = centered / max(norm, eps)
```

**Result:** Each pixel's spectral vector has zero mean and unit L2 norm.

#### Example Usage (Python)

```python
from cuvis_ai.node.normalization import PerPixelUnitNorm

# Create per-pixel normalizer
unit_norm = PerPixelUnitNorm(eps=1e-8)

# Use in pipeline
outputs = unit_norm.forward(data=cube)  # Shape: [B, H, W, C]

# Verify unit norm
import torch
norms = outputs["normalized"].norm(dim=-1)  # Should be ≈1.0 for all pixels
```

**Integration with Deep SVDD**

```python
# Deep SVDD preprocessing chain
from cuvis_ai.node.preprocessors import BandpassByWavelength
from cuvis_ai.node.normalization import PerPixelUnitNorm

# Step 1: Filter to relevant bands
bandpass = BandpassByWavelength(min_wavelength_nm=500.0, max_wavelength_nm=800.0)

# Step 2: Unit-norm per pixel
unit_norm = PerPixelUnitNorm(eps=1e-8)

# Connect in pipeline
pipeline.connect(
    (data.cube, bandpass.data),
    (bandpass.filtered, unit_norm.data),
    (unit_norm.normalized, encoder.data),  # To Deep SVDD encoder
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  unit_norm:
    type: PerPixelUnitNorm
    config:
      eps: 1e-8

connections:
  - [bandpass.filtered, unit_norm.data]
  - [unit_norm.normalized, encoder.data]
```

#### Why Unit Normalization?

**Benefits:**

1. **Scale invariance**: Removes absolute intensity, focuses on spectral shape
2. **Cosine similarity**: Unit vectors enable efficient similarity computation
3. **Hypersphere embedding**: Natural for Deep SVDD's hypersphere boundary
4. **Gradient flow**: Bounded gradients improve training stability

**When to avoid:**

- Absolute intensity is important (e.g., detection of faint anomalies)
- Data is already zero-mean, unit-variance (z-score normalized)

#### See Also

- [Tutorial 3: Deep SVDD Gradient](../tutorials/deep-svdd-gradient.md#per-pixel-unit-normalization) - Complete usage
- [Deep SVDD Projection](deep-learning.md#deepsvddprojection) - Downstream encoder
- API Reference: ::: cuvis_ai.node.normalization.PerPixelUnitNorm

---

### SigmoidTransform

**Description:** Applies sigmoid function to convert logits to probabilities [0, 1]

**Perfect for:**

- Converting raw model outputs (logits) to probabilities
- Visualization of anomaly scores
- Preparing unbounded scores for bounded loss functions
- Routing logits to both loss (raw) and visualization (sigmoid) ports

**Training Paradigm:** None (simple transformation)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| data | float32 | (B, H, W, C) | Input tensor (logits or scores) | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| transformed | float32 | (B, H, W, C) | Sigmoid-transformed tensor in [0, 1] |

#### Parameters

None (applies `torch.sigmoid` directly)

#### Example Usage (Python)

```python
from cuvis_ai.node.normalization import SigmoidTransform

# Create sigmoid transform
sigmoid = SigmoidTransform()

# Use for visualization routing
pipeline.connect(
    (rx.scores, loss_node.predictions),    # Raw logits to loss
    (rx.scores, sigmoid.data),             # Logits to sigmoid
    (sigmoid.transformed, viz.scores),     # Probabilities to viz
)
```

**Typical pattern: Dual routing**

```yaml
nodes:
  rx_detector:
    type: RXGlobal
    # Outputs: scores (logits)

  logit_head:
    type: ScoreToLogit
    # Outputs: logits

  sigmoid:
    type: SigmoidTransform

  loss:
    type: AnomalyBCEWithLogits  # Expects raw logits

  viz:
    type: ScoreHeatmapVisualizer  # Expects [0, 1] probabilities

connections:
  - [logit_head.logits, loss.predictions]      # Raw to loss
  - [logit_head.logits, sigmoid.data]          # Raw to sigmoid
  - [sigmoid.transformed, viz.scores]          # Probs to viz
```

#### Comparison with SigmoidNormalizer

| Node | Transformation | Use Case |
|------|----------------|----------|
| **SigmoidTransform** | `sigmoid(x)` | Simple logit → probability conversion |
| **SigmoidNormalizer** | `sigmoid((x - median) / std)` | Robust normalization with centering |

**When to use which:**

- **SigmoidTransform**: Logits already well-scaled (e.g., from ScoreToLogit)
- **SigmoidNormalizer**: Unbounded scores need robust normalization

#### Example Configuration (YAML)

```yaml
nodes:
  sigmoid:
    type: SigmoidTransform  # No config parameters

connections:
  - [model.logits, sigmoid.data]
  - [sigmoid.transformed, visualizer.scores]
```

#### See Also

- [Tutorial 1: RX Statistical](../tutorials/rx-statistical.md#visualization-with-sigmoid) - Routing example
- [ScoreToLogit](utility.md#scoretologit) - Produces logits for sigmoid
- API Reference: ::: cuvis_ai.node.normalization.SigmoidTransform

---

### IdentityNormalizer

**Description:** No-op normalizer that passes data through unchanged

**Perfect for:**

- Placeholder in modular pipeline configurations
- A/B testing different normalization strategies
- Debugging pipelines (bypass normalization)
- Config-driven architecture selection

**Training Paradigm:** None (pass-through)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| data | float32 | (B, H, W, C) | Input tensor | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| normalized | float32 | (B, H, W, C) | Unchanged output (identity) |

#### Parameters

None

#### Example Usage (Python)

```python
from cuvis_ai.node.normalization import IdentityNormalizer

# Create identity normalizer (pass-through)
identity = IdentityNormalizer()

# Use as placeholder
outputs = identity.forward(data=cube)  # outputs["normalized"] == cube
```

**Config-driven selection**

```python
# Select normalizer based on config
normalizer_type = config.get("normalizer", "identity")

if normalizer_type == "minmax":
    normalizer = MinMaxNormalizer()
elif normalizer_type == "zscore":
    normalizer = ZScoreNormalizer()
else:
    normalizer = IdentityNormalizer()  # No normalization
```

#### Example Configuration (YAML)

```yaml
# Baseline: No normalization
nodes:
  normalizer:
    type: IdentityNormalizer  # No-op

# Experiment: Try z-score
nodes:
  normalizer:
    type: ZScoreNormalizer
    config:
      dims: [1, 2]
```

#### When to Use

**Valid use cases:**

1. **Ablation studies**: Test impact of normalization
2. **Pre-normalized data**: Input already in correct range
3. **Modular configs**: Swap normalizers without code changes

**Anti-pattern (avoid):**

```python
# Bad: Unnecessary identity node
pipeline.connect(
    (data.cube, identity.data),
    (identity.normalized, next_node.input),
)

# Better: Direct connection
pipeline.connect(
    (data.cube, next_node.input),
)
```

Only use IdentityNormalizer when you need a normalizer placeholder in a modular architecture.

#### See Also

- [Concept: Pipeline Architecture](../concepts/pipeline-lifecycle.md) - Modular design patterns
- API Reference: ::: cuvis_ai.node.normalization.IdentityNormalizer

---

## Choosing the Right Normalizer

### Decision Tree

```mermaid
graph TD
    A[Need Normalization?] -->|Yes| B{Data Range?}
    A -->|No| I[IdentityNormalizer]

    B -->|Bounded to [0,1]| C{Outliers?}
    B -->|Unbounded| D{Use Case?}

    C -->|No| E[MinMaxNormalizer]
    C -->|Yes| F[SigmoidNormalizer]

    D -->|Deep Learning| G[ZScoreNormalizer]
    D -->|Unit Features| H[PerPixelUnitNorm]
    D -->|Logits → Probs| J[SigmoidTransform]
```

### Normalization Strategy by Task

| Task | Recommended Normalizer | Reason |
|------|----------------------|--------|
| Statistical detection (RX) | MinMaxNormalizer | Bounded [0,1] for visualization |
| Deep learning training | ZScoreNormalizer | Zero mean, unit variance for gradients |
| Deep SVDD | PerPixelUnitNorm | Unit-norm for hypersphere embedding |
| Anomaly visualization | SigmoidTransform | Convert logits to probabilities |
| Robust to outliers | SigmoidNormalizer | Median-based, sigmoid saturation |
| Pre-normalized data | IdentityNormalizer | No transformation needed |

### Chaining Preprocessors

**Common pattern 1: Band filtering + normalization**

```yaml
nodes:
  bandpass:
    type: BandpassByWavelength
    config:
      min_wavelength_nm: 500.0
      max_wavelength_nm: 800.0

  normalizer:
    type: MinMaxNormalizer
    config:
      use_running_stats: true

connections:
  - [data.cube, bandpass.data]
  - [bandpass.filtered, normalizer.data]
```

**Common pattern 2: Deep SVDD preprocessing chain**

```yaml
nodes:
  bandpass:
    type: BandpassByWavelength
    config:
      min_wavelength_nm: 500.0
      max_wavelength_nm: 800.0

  unit_norm:
    type: PerPixelUnitNorm
    config:
      eps: 1e-8

  encoder:
    type: ZScoreNormalizerGlobal  # Statistical encoder
    # ...

connections:
  - [data.cube, bandpass.data]
  - [bandpass.filtered, unit_norm.data]
  - [unit_norm.normalized, encoder.data]
```

---

## Performance Considerations

**Computational Cost (per batch):**

| Node | Complexity | Memory Overhead |
|------|------------|-----------------|
| BandpassByWavelength | O(C) indexing | None (views) |
| MinMaxNormalizer | O(B×H×W×C) | 2 scalars (running stats) |
| ZScoreNormalizer | O(B×H×W×C) | None (per-sample) |
| SigmoidNormalizer | O(B×H×W×C) | None (per-sample) |
| PerPixelUnitNorm | O(B×H×W×C) | None (per-sample) |
| SigmoidTransform | O(B×H×W×C) | None (elementwise) |
| IdentityNormalizer | O(1) | None (pass-through) |

**Optimization Tips:**

1. **Fuse operations**: Combine bandpass + normalization in a single custom node if performance-critical
2. **Use running stats**: Enable `use_running_stats=True` for consistent normalization across batches
3. **Avoid redundant normalization**: Don't normalize twice (e.g., MinMax → ZScore)
4. **GPU-friendly**: All nodes are GPU-accelerated via PyTorch operations

---

## Creating Custom Preprocessors

```python
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
import torch

class CustomPreprocessor(Node):
    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input cube [B, H, W, C]"
        ),
    }

    OUTPUT_SPECS = {
        "processed": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Processed cube"
        ),
    }

    def __init__(self, custom_param: float = 1.0, **kwargs):
        self.custom_param = custom_param
        super().__init__(custom_param=custom_param, **kwargs)

    def forward(self, data: torch.Tensor, **kwargs):
        # Your custom preprocessing logic
        processed = data * self.custom_param
        return {"processed": processed}
```

**Learn more:**

- [Plugin System Development](../plugin-system/development.md)
- [Node System Deep Dive](../concepts/node-system-deep-dive.md)

---

**Next Steps:**

- Explore [Selector Nodes](selectors.md) for learnable band selection
- Learn about [Statistical Nodes](statistical.md) for anomaly detection
- Review [Tutorial 1: RX Statistical](../tutorials/rx-statistical.md) for complete preprocessing examples
