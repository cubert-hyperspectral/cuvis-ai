!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Advanced Plugin Development

Critical requirements, advanced node patterns, and configuration support for plugin nodes.

## Critical Requirements for Node Development

### 1. Import Context and Datatypes from cuvis-ai-schemas

All nodes **must** import the `Context` class and any required datatypes from `cuvis_ai_schemas`:

```python
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context, InputStream, Metric
```

**Why this matters:** The `Context` parameter provides execution metadata that nodes can use to:

- Determine the current execution stage (training vs. inference)
- Access the current epoch and batch index for logging
- Implement stage-specific behavior (e.g., dropout during training only)

### 2. Forward Method Must Accept Context Parameter

All node `forward()` methods **must** include a `context: Context` parameter:

```python
def forward(self, input_data: torch.Tensor, context: Context, **kwargs: Any) -> dict[str, Any]:
    if context.stage == ExecutionStage.TRAIN:
        # Training-specific logic
        pass
    elif context.stage == ExecutionStage.INFERENCE:
        # Inference-specific logic
        pass
```

**Context can be optional** for nodes that don't need execution metadata:
```python
def forward(self, input_data: torch.Tensor, context: Context | None = None, **kwargs: Any) -> dict[str, Any]:
    ...
```

**Execution Stages and Node Filtering:**

By default, nodes execute in **all stages** (`ExecutionStage.ALWAYS`). Override this by passing `execution_stages` to `super().__init__()`:

```python
class LossNode(Node):
    """Loss computation node - only runs during training."""

    def __init__(self, loss_weight: float = 1.0, **kwargs):
        self.loss_weight = loss_weight
        super().__init__(
            execution_stages={ExecutionStage.TRAIN},
            loss_weight=loss_weight,
            **kwargs,
        )

class VisualizationNode(Node):
    """Visualization node - runs during training, validation, and testing."""

    def __init__(self, max_samples: int = 4, **kwargs):
        self.max_samples = max_samples
        super().__init__(
            execution_stages={ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST},
            max_samples=max_samples,
            **kwargs,
        )
```

**Available execution stages:**

- `ExecutionStage.ALWAYS` - Default, runs in all stages
- `ExecutionStage.TRAIN` - Training only
- `ExecutionStage.VAL` - Validation only
- `ExecutionStage.TEST` - Testing only
- `ExecutionStage.INFERENCE` - Inference only

### 3. Pass All Hyperparameters to super().__init__()

All hyperparameters that need to be **serialized** must be passed as keyword arguments to `super().__init__()`:

```python
def __init__(self, threshold: float = 0.95, method: str = "simple", hidden_dim: int = 128, **kwargs):
    self.threshold = threshold
    self.method = method
    self.hidden_dim = hidden_dim

    # CRITICAL: enables serialization/deserialization for pipeline saving
    super().__init__(threshold=threshold, method=method, hidden_dim=hidden_dim, **kwargs)
```

### 4. Use Port Specifications from cuvis-ai-schemas

```python
from cuvis_ai_schemas.pipeline import PortSpec

class MyNode(Node):
    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),  # [B, H, W, C]
            description="Input hyperspectral cube",
        ),
    }

    OUTPUT_SPECS = {
        "result": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),  # [B, H, W, 1]
            description="Output result",
        ),
    }
```

### 5. Always Use uv for Package Management

```bash
uv sync                  # Install dependencies (creates uv.lock)
uv sync --extra dev      # With dev dependencies
uv run pytest tests/ -v  # Run scripts
```

---

## Advanced Node Patterns

### Statistical Nodes

Nodes that require initialization with data:

```python
class StatisticalCustomNode(Node):
    """Node requiring statistical initialization."""

    def __init__(self, n_components: int = 10):
        super().__init__()
        self.n_components = n_components
        self.is_initialized = False
        self.statistics = None

    def initialize(self, initialization_data: np.ndarray):
        """Initialize with data (Phase 1 of two-phase training)."""
        pixels = initialization_data.reshape(-1, initialization_data.shape[-1])
        mean = pixels.mean(axis=0)
        centered = pixels - mean

        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.statistics = {
            "mean": mean,
            "components": eigenvectors[:, idx],
            "eigenvalues": eigenvalues[idx]
        }
        self.is_initialized = True

    def forward(self, data: np.ndarray) -> dict:
        if not self.is_initialized:
            raise RuntimeError("Node not initialized. Call initialize() first.")

        H, W, C = data.shape
        pixels = data.reshape(-1, C)
        centered = pixels - self.statistics["mean"]
        transformed = centered @ self.statistics["components"]

        return {"transformed": transformed.reshape(H, W, -1)}
```

See [Two-Phase Training](../concepts/two-phase-training.md) for details on statistical initialization.

### Deep Learning Nodes

Nodes using PyTorch:

```python
import torch.nn as nn

class DeepLearningDetector(Node):
    """Deep learning-based anomaly detector."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32, device: str = "auto"):
        super().__init__()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.model = nn.Sequential(self.encoder, self.decoder)
        self.model.to(self.device)

    def forward(self, data: np.ndarray) -> dict:
        H, W, C = data.shape
        pixels = data.reshape(-1, C)
        x = torch.from_numpy(pixels).float().to(self.device)

        with torch.no_grad():
            features = self.encoder(x)
            reconstructed = self.model(x)

        error = torch.mean((x - reconstructed) ** 2, dim=1)

        return {
            "predictions": error.cpu().numpy().reshape(H, W),
            "features": features.cpu().numpy().reshape(H, W, -1)
        }
```

### Multi-Output Nodes

```python
class FeatureExtractor(Node):
    """Extract multiple feature types from input."""

    OUTPUT_SPECS = {
        "spectral_features": PortSpec(dtype=np.ndarray),
        "spatial_features": PortSpec(dtype=np.ndarray),
        "texture_features": PortSpec(dtype=np.ndarray),
    }

    def forward(self, data: np.ndarray) -> dict:
        return {
            "spectral_features": self._extract_spectral(data),
            "spatial_features": self._extract_spatial(data),
            "texture_features": self._extract_texture(data),
        }
```

---

## Configuration Support

### Add Hydra Configuration

Create `cuvis_ai_plugin/configs/default.yaml`:

```yaml
custom_anomaly_detector:
  threshold: 0.95
  method: "simple"
  window_size: 5

training:
  batch_size: 32
  epochs: 50
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
```

### Use Configuration in Nodes

```python
from omegaconf import DictConfig

class ConfigurableNode(Node):
    """Node accepting Hydra configuration."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.threshold = cfg.threshold
        self.method = cfg.method
        self.window_size = cfg.window_size
```

## See Also

- [Plugin Quick Start](dev-quickstart.md) — create your first plugin
- [Testing & Publishing](dev-testing.md) — testing best practices
- [Node System Deep Dive](../concepts/node-system-deep-dive.md) — node architecture details
- [Two-Phase Training](../concepts/two-phase-training.md) — statistical initialization patterns
