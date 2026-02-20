!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin Development Guide

## Introduction

This guide walks you through creating a cuvis-ai plugin from scratch, implementing custom nodes, testing your plugin, and publishing it for others to use. By the end of this guide, you'll have a fully functional plugin that extends cuvis-ai with your own domain-specific algorithms.

## Prerequisites

- **Python 3.9+** installed
- **cuvis-ai** and **cuvis-ai-core** installed
- **Git** for version control
- **uv** for package management (recommended over pip)
- Basic understanding of the [Node System](../concepts/node-system-deep-dive.md)
- Familiarity with [Port System](../concepts/port-system-deep-dive.md)

**Note:** This guide uses `uv` for all package management and script execution. Use `uv run python ...` instead of `python ...` for running scripts.

## Quick Start

### Step 1: Create Plugin Structure

Create your plugin directory structure:

```bash
# Create plugin directory
mkdir my-cuvis-plugin
cd my-cuvis-plugin

# Create package structure
mkdir -p cuvis_ai_plugin/nodes
mkdir -p cuvis_ai_plugin/configs
mkdir -p tests

# Create necessary files
touch cuvis_ai_plugin/__init__.py
touch cuvis_ai_plugin/nodes/__init__.py
touch cuvis_ai_plugin/nodes/custom_node.py
touch cuvis_ai_plugin/configs/default.yaml
touch tests/test_custom_node.py
touch pyproject.toml
touch README.md
touch .gitignore
```

**Final Structure:**
```
my-cuvis-plugin/
├── pyproject.toml          # Project configuration (REQUIRED)
├── cuvis_ai_plugin/
│   ├── __init__.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   └── custom_node.py
│   └── configs/
│       └── default.yaml
├── tests/
│   └── test_custom_node.py
├── README.md
└── .gitignore
```

### Step 2: Configure pyproject.toml

Create a PEP 621 compliant `pyproject.toml`:

```toml
[project]
name = "cuvis-ai-my-plugin"
version = "0.1.0"
description = "Custom anomaly detection plugin for cuvis-ai"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}

dependencies = [
    "cuvis-ai-core>=0.1.0",
    "numpy>=1.20.0",
    "torch>=2.0.0",  # If using deep learning
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.ruff]
line-length = 100
target-version = "py39"
```

### Step 3: Implement Your First Node

Create `cuvis_ai_plugin/nodes/custom_node.py`:

```python
"""Custom anomaly detection node."""

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
from cuvis_ai_schemas.execution import Context


class CustomAnomalyDetector(Node):
    """
    Custom anomaly detection node using statistical methods.

    This node implements a simple threshold-based anomaly detection
    algorithm on hyperspectral data.

    Parameters
    ----------
    threshold : float
        Detection threshold for anomaly scores. Default is 0.95.
    method : str
        Detection method ('simple' or 'advanced'). Default is 'simple'.
    window_size : int
        Sliding window size for contextual analysis. Default is 5.

    Examples
    --------
    >>> detector = CustomAnomalyDetector(threshold=0.95)
    >>> outputs = detector(data=hyperspectral_cube, context=context)
    >>> anomaly_map = outputs["detections"]
    """

    # Define input ports (use dict format)
    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        )
    }

    # Define output ports (use dict format)
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Anomaly scores for each pixel [B, H, W, 1]",
        ),
        "detections": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly detections [B, H, W, 1]",
        ),
    }

    def __init__(
        self,
        threshold: float = 0.95,
        method: str = "simple",
        window_size: int = 5,
        **kwargs,
    ):
        """Initialize the custom anomaly detector.

        IMPORTANT: All hyperparameters that need to be serialized must be passed
        to super().__init__() as keyword arguments for proper serialization.
        """
        # Store hyperparameters as instance attributes
        self.threshold = threshold
        self.method = method
        self.window_size = window_size

        # Validate parameters
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        if method not in ["simple", "advanced"]:
            raise ValueError(f"method must be 'simple' or 'advanced', got {method}")

        # Pass all hyperparameters to super().__init__() for serialization
        super().__init__(
            threshold=threshold,
            method=method,
            window_size=window_size,
            **kwargs,
        )

    def forward(self, data: torch.Tensor, context: Context, **kwargs: Any) -> dict[str, Any]:
        """
        Process input data and detect anomalies.

        IMPORTANT: The forward method must accept a 'context' parameter containing
        execution metadata (stage, epoch, batch_idx, global_step).

        Parameters
        ----------
        data : torch.Tensor
            Input hyperspectral cube with shape [B, H, W, C].
        context : Context
            Execution context with stage, epoch, batch_idx, global_step.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'scores' and 'detections' keys.
        """
        # Validate input shape
        if data.ndim != 4:
            raise ValueError(f"Expected 4D input [B, H, W, C], got shape {data.shape}")

        # Compute anomaly scores
        scores = self._compute_scores(data)

        # Apply threshold (keep as float tensor, not binary)
        detections = (scores > self.threshold).float()

        return {
            "scores": scores,
            "detections": detections,
        }

    def _compute_scores(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for each pixel.

        Parameters
        ----------
        data : torch.Tensor
            Input data [B, H, W, C].

        Returns
        -------
        torch.Tensor
            Anomaly scores [B, H, W, 1].
        """
        B, H, W, C = data.shape

        if self.method == "simple":
            # Simple statistical method: Mahalanobis distance per batch
            # Flatten spatial dimensions
            pixels = data.reshape(B, H * W, C)  # [B, N, C] where N = H*W

            # Compute per-batch statistics
            mean = pixels.mean(dim=1, keepdim=True)  # [B, 1, C]
            centered = pixels - mean  # [B, N, C]

            # Compute covariance matrix [B, C, C]
            cov = torch.bmm(centered.transpose(1, 2), centered) / (H * W - 1)

            # Add regularization for numerical stability
            cov = cov + torch.eye(C, device=data.device) * 1e-6

            # Compute Mahalanobis distance
            cov_inv = torch.linalg.inv(cov)  # [B, C, C]
            mahal_sq = torch.bmm(
                torch.bmm(centered, cov_inv),  # [B, N, C]
                centered.unsqueeze(-1)  # [B, N, C, 1]
            ).squeeze(-1)  # [B, N]

            scores = torch.sqrt(mahal_sq.clamp(min=0))  # [B, N]
            scores = scores.reshape(B, H, W, 1)  # [B, H, W, 1]

            # Normalize to [0, 1] per batch
            scores_min = scores.amin(dim=(1, 2), keepdim=True)
            scores_max = scores.amax(dim=(1, 2), keepdim=True)
            scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)
            return scores

        elif self.method == "advanced":
            # Advanced method: Contextual anomaly detection
            return self._contextual_detection(data)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _contextual_detection(self, data: torch.Tensor) -> torch.Tensor:
        """
        Context-aware anomaly detection using sliding window.

        Parameters
        ----------
        data : torch.Tensor
            Input data [B, H, W, C].

        Returns
        -------
        torch.Tensor
            Anomaly scores [B, H, W, 1].
        """
        B, H, W, C = data.shape
        scores = torch.zeros(B, H, W, 1, device=data.device)
        half_window = self.window_size // 2

        # Process each batch and spatial location
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    # Extract local window
                    i_start = max(0, i - half_window)
                    i_end = min(H, i + half_window + 1)
                    j_start = max(0, j - half_window)
                    j_end = min(W, j + half_window + 1)

                    window = data[b, i_start:i_end, j_start:j_end, :]  # [h, w, C]
                    window_pixels = window.reshape(-1, C)  # [h*w, C]

                    # Compute local statistics
                    local_mean = window_pixels.mean(dim=0)  # [C]
                    local_std = window_pixels.std(dim=0) + 1e-8  # [C]

                    # Compute pixel's deviation from local context
                    pixel = data[b, i, j, :]  # [C]
                    deviation = torch.abs(pixel - local_mean) / local_std  # [C]
                    scores[b, i, j, 0] = deviation.mean()

        # Normalize per batch
        scores_min = scores.amin(dim=(1, 2), keepdim=True)
        scores_max = scores.amax(dim=(1, 2), keepdim=True)
        scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)
        return scores
```

### Step 4: Export Your Node

Update `cuvis_ai_plugin/__init__.py`:

```python
"""My Custom CUVIS-AI Plugin."""

from .nodes.custom_node import CustomAnomalyDetector

__version__ = "0.1.0"
__all__ = ["CustomAnomalyDetector"]
```

Update `cuvis_ai_plugin/nodes/__init__.py`:

```python
"""Node implementations."""

from .custom_node import CustomAnomalyDetector

__all__ = ["CustomAnomalyDetector"]
```

### Step 5: Create Tests

Create `tests/test_custom_node.py`:

```python
"""Tests for custom anomaly detector node."""

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_plugin.nodes import CustomAnomalyDetector


def test_custom_node_initialization():
    """Test node can be initialized."""
    node = CustomAnomalyDetector(threshold=0.95, method="simple")
    assert node.threshold == 0.95
    assert node.method == "simple"


def test_custom_node_invalid_threshold():
    """Test invalid threshold raises error."""
    with pytest.raises(ValueError, match="threshold must be in"):
        CustomAnomalyDetector(threshold=1.5)


def test_custom_node_simple_method():
    """Test simple detection method."""
    node = CustomAnomalyDetector(threshold=0.9, method="simple")

    # Create synthetic data [B, H, W, C] format
    data = torch.randn(2, 10, 10, 50, dtype=torch.float32)

    # Create context
    context = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)

    # Process
    outputs = node(data=data, context=context)

    # Check outputs
    assert "scores" in outputs
    assert "detections" in outputs
    assert outputs["scores"].shape == (2, 10, 10, 1)
    assert outputs["detections"].shape == (2, 10, 10, 1)
    assert outputs["detections"].dtype == torch.float32


def test_custom_node_advanced_method():
    """Test advanced detection method."""
    node = CustomAnomalyDetector(threshold=0.9, method="advanced", window_size=3)

    data = torch.randn(2, 10, 10, 50, dtype=torch.float32)
    context = Context(stage=ExecutionStage.INFERENCE)
    outputs = node(data=data, context=context)

    assert outputs["scores"].shape == (2, 10, 10, 1)
    assert outputs["detections"].shape == (2, 10, 10, 1)


def test_custom_node_threshold_behavior():
    """Test threshold parameter works correctly."""
    data = torch.randn(2, 10, 10, 50, dtype=torch.float32)
    context = Context(stage=ExecutionStage.INFERENCE)

    # High threshold = fewer detections
    node_high = CustomAnomalyDetector(threshold=0.99)
    outputs_high = node_high(data=data, context=context)

    # Low threshold = more detections
    node_low = CustomAnomalyDetector(threshold=0.5)
    outputs_low = node_low(data=data, context=context)

    assert outputs_low["detections"].sum() >= outputs_high["detections"].sum()


def test_custom_node_invalid_input_shape():
    """Test node rejects invalid input shapes."""
    node = CustomAnomalyDetector()
    context = Context(stage=ExecutionStage.INFERENCE)

    # 3D input (should be 4D [B, H, W, C])
    with pytest.raises(ValueError, match="Expected 4D input"):
        node(data=torch.randn(10, 10, 50), context=context)
```

### Step 6: Test Locally

```bash
# Install plugin with all dependencies using uv (recommended)
uv sync

# Install with dev dependencies
uv sync --extra dev
# Or install all optional dependencies
uv sync --all-extras

# Alternative: Install in editable mode using uv pip
# uv pip install -e ".[dev]"

# Run tests with uv
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=cuvis_ai_plugin --cov-report=html
```

### Step 7: Create Plugin Manifest

Create `examples/plugins.yaml` to test your plugin:

```yaml
plugins:
  my_plugin:
    path: "."  # Current directory
    provides:
      - cuvis_ai_plugin.nodes.custom_node.CustomAnomalyDetector
```

### Step 8: Test in Pipeline

Create `examples/test_plugin.py`:

```python
"""Test plugin in a pipeline."""

import torch
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.pipeline.pipeline import Pipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context


def main():
    # Create registry
    registry = NodeRegistry()

    # Load plugin from manifest
    registry.load_plugins("examples/plugins.yaml")

    # Verify node is available
    CustomAnomalyDetector = registry.get("CustomAnomalyDetector", instance=registry)
    print(f"✓ Node loaded: {CustomAnomalyDetector}")

    # Create pipeline
    pipeline_dict = {
        "nodes": [
            {
                "class_name": "CustomAnomalyDetector",
                "name": "detector",
                "params": {
                    "threshold": 0.95,
                    "method": "simple"
                }
            }
        ],
        "edges": []
    }

    pipeline = Pipeline.from_dict(pipeline_dict, node_registry=registry)

    # Test with synthetic data in [B, H, W, C] format
    test_data = torch.randn(4, 50, 50, 100, dtype=torch.float32)

    # Create execution context
    context = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)

    # Execute pipeline
    outputs = pipeline(data=test_data, context=context)

    print(f"✓ Pipeline executed successfully")
    print(f"  Anomaly scores shape: {outputs['detector']['scores'].shape}")
    print(f"  Detections: {outputs['detector']['detections'].sum().item()} pixels")


if __name__ == "__main__":
    main()
```

Run the test:

```bash
uv run python examples/test_plugin.py
```

## Critical Requirements for Node Development

### 1. Import Context and Datatypes from cuvis-ai-schemas

All nodes **must** import the `Context` class and any required datatypes from `cuvis_ai_schemas`:

```python
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context, InputStream, Metric

# For commonly used datatypes:
# - Context: Execution context with stage, epoch, batch_idx, global_step
# - ExecutionStage: Enum for execution stages (TRAIN, VAL, TEST, INFERENCE)
# - InputStream: Type hint for input data streams
# - Metric: Dataclass for metric logging
# - Artifact: Dataclass for artifact logging (images, etc.)
# - ArtifactType: Enum for artifact types (IMAGE, etc.)
```

**Why this matters:** The `Context` parameter provides execution metadata that nodes can use to:
- Determine the current execution stage (training vs. inference)
- Access the current epoch and batch index for logging
- Implement stage-specific behavior (e.g., dropout during training only)

### 2. Forward Method Must Accept Context Parameter

All node `forward()` methods **must** include a `context: Context` parameter:

```python
def forward(self, input_data: torch.Tensor, context: Context, **kwargs: Any) -> dict[str, Any]:
    """Process input data.

    Parameters
    ----------
    input_data : torch.Tensor
        Input data tensor.
    context : Context
        Execution context with stage, epoch, batch_idx, global_step.

    Returns
    -------
    dict[str, Any]
        Output dictionary.
    """
    # Use context to access execution metadata
    if context.stage == ExecutionStage.TRAIN:
        # Training-specific logic
        pass
    elif context.stage == ExecutionStage.INFERENCE:
        # Inference-specific logic
        pass

    # Your processing logic here
    ...
```

**Context can be optional** for nodes that don't need execution metadata:
```python
def forward(self, input_data: torch.Tensor, context: Context | None = None, **kwargs: Any) -> dict[str, Any]:
    # context is optional
    ...
```

**Execution Stages and Node Filtering:**

By default, nodes execute in **all stages** (`ExecutionStage.ALWAYS`). You can override this to make nodes execute only during specific stages by passing `execution_stages` directly to `super().__init__()`:

```python
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import ExecutionStage

class LossNode(Node):
    """Loss computation node - only runs during training."""

    def __init__(self, loss_weight: float = 1.0, **kwargs):
        self.loss_weight = loss_weight

        # Pass execution_stages directly to super().__init__()
        super().__init__(
            execution_stages={ExecutionStage.TRAIN},  # Only execute during training
            loss_weight=loss_weight,
            **kwargs,
        )

class VisualizationNode(Node):
    """Visualization node - runs during training, validation, and testing."""

    def __init__(self, max_samples: int = 4, **kwargs):
        self.max_samples = max_samples

        # Override to run in multiple stages (but not inference)
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

**Use cases for stage filtering:**
- Loss nodes: Only needed during training
- Metric nodes: Only needed during validation/testing
- Dropout/augmentation: Only active during training
- Expensive visualizations: Skip during training, show during inference

### 3. Pass All Hyperparameters to super().__init__()

All hyperparameters that need to be **serialized** (saved/loaded) must be passed as keyword arguments to `super().__init__()`:

```python
def __init__(
    self,
    threshold: float = 0.95,
    method: str = "simple",
    hidden_dim: int = 128,
    **kwargs,
):
    """Initialize node with hyperparameters."""
    # Store as instance attributes
    self.threshold = threshold
    self.method = method
    self.hidden_dim = hidden_dim

    # CRITICAL: Pass ALL hyperparameters to super().__init__()
    # This enables serialization/deserialization for pipeline saving
    super().__init__(
        threshold=threshold,
        method=method,
        hidden_dim=hidden_dim,
        **kwargs,
    )
```

**Why this matters:** cuvis-ai uses these parameters to:
- Serialize node configuration when saving pipelines
- Reconstruct nodes when loading pipelines from YAML
- Track hyperparameters for experiment logging

**Examples from cuvis-ai nodes:**

```python
# From BandpassByWavelength node:
def __init__(
    self,
    min_wavelength_nm: float,
    max_wavelength_nm: float | None = None,
    **kwargs,
) -> None:
    self.min_wavelength_nm = float(min_wavelength_nm)
    self.max_wavelength_nm = float(max_wavelength_nm) if max_wavelength_nm is not None else None

    super().__init__(
        min_wavelength_nm=self.min_wavelength_nm,
        max_wavelength_nm=self.max_wavelength_nm,
        **kwargs,
    )

# From RXGlobal node:
def __init__(self, eps: float = 1e-6, **kwargs) -> None:
    self.eps = eps
    super().__init__(eps=eps, **kwargs)
```

### 4. Use Port Specifications from cuvis-ai-schemas

Import `PortSpec` from `cuvis_ai_schemas.pipeline`:

```python
from cuvis_ai_schemas.pipeline import PortSpec

class MyNode(Node):
    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),  # [B, H, W, C]
            description="Input hyperspectral cube",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),  # [C]
            description="Wavelength array in nanometers",
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

Use `uv` instead of `pip` for all package management and script execution:

```bash
# Install dependencies (recommended - creates uv.lock)
uv sync
uv sync --extra dev  # With dev dependencies

# Alternative: Use uv pip for editable installs
uv pip install -e ".[dev]"

# Run scripts
uv run python examples/test_plugin.py
uv run pytest tests/ -v

# NOT: python examples/test_plugin.py
# NOT: pip install -e .
```

**Why uv?** `uv` provides:
- Faster dependency resolution (10-100x faster than pip)
- Automatic lock file generation (`uv.lock`) for reproducibility
- Consistent virtual environment handling
- Improved caching and performance

**uv sync vs uv pip install:**
- `uv sync`: Recommended for development, creates lock file, syncs environment to exact versions
- `uv pip install -e .`: Alternative for compatibility, works like traditional pip

---

## Advanced Node Development

### Creating Statistical Nodes

Nodes that require initialization:

```python
class StatisticalCustomNode(Node):
    """Node requiring statistical initialization."""

    INPUT_SPECS = [
        PortSpec(name="data", dtype=np.ndarray, required=True)
    ]

    OUTPUT_SPECS = [
        PortSpec(name="transformed", dtype=np.ndarray)
    ]

    def __init__(self, n_components: int = 10):
        super().__init__()
        self.n_components = n_components
        self.is_initialized = False
        self.statistics = None

    def initialize(self, initialization_data: np.ndarray):
        """
        Initialize with initialization data.

        Parameters
        ----------
        initialization_data : np.ndarray
            Initialization dataset for computing statistics.
        """
        # Compute PCA components
        pixels = initialization_data.reshape(-1, initialization_data.shape[-1])
        mean = pixels.mean(axis=0)
        centered = pixels - mean

        # Compute covariance
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Store top components
        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.statistics = {
            "mean": mean,
            "components": eigenvectors[:, idx],
            "eigenvalues": eigenvalues[idx]
        }
        self.is_initialized = True

    def forward(self, data: np.ndarray) -> dict:
        """Transform data using learned statistics."""
        if not self.is_initialized:
            raise RuntimeError("Node not initialized. Call initialize() first.")

        H, W, C = data.shape
        pixels = data.reshape(-1, C)

        # Project onto PCA components
        centered = pixels - self.statistics["mean"]
        transformed = centered @ self.statistics["components"]

        return {"transformed": transformed.reshape(H, W, -1)}
```

**Two-Phase Training Integration:**

See [Two-Phase Training](../concepts/two-phase-training.md) for details on statistical initialization patterns.

### Creating Deep Learning Nodes

Nodes using PyTorch or other deep learning frameworks:

```python
import torch
import torch.nn as nn


class DeepLearningDetector(Node):
    """Deep learning-based anomaly detector."""

    INPUT_SPECS = [
        PortSpec(name="data", dtype=np.ndarray, required=True)
    ]

    OUTPUT_SPECS = [
        PortSpec(name="predictions", dtype=np.ndarray),
        PortSpec(name="features", dtype=np.ndarray)
    ]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        device: str = "auto"
    ):
        super().__init__()

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Define model architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.model = nn.Sequential(self.encoder, self.decoder)
        self.model.to(self.device)

    def forward(self, data: np.ndarray) -> dict:
        """
        Forward pass through autoencoder.

        Parameters
        ----------
        data : np.ndarray
            Input data (H, W, C).

        Returns
        -------
        dict
            Predictions and latent features.
        """
        H, W, C = data.shape
        pixels = data.reshape(-1, C)

        # Convert to tensor
        x = torch.from_numpy(pixels).float().to(self.device)

        # Forward pass
        with torch.no_grad():
            features = self.encoder(x)
            reconstructed = self.model(x)

        # Compute reconstruction error (anomaly score)
        error = torch.mean((x - reconstructed) ** 2, dim=1)

        # Convert back to numpy
        predictions = error.cpu().numpy().reshape(H, W)
        features_np = features.cpu().numpy().reshape(H, W, -1)

        return {
            "predictions": predictions,
            "features": features_np
        }

    def train_step(self, data: np.ndarray) -> float:
        """
        Training step for gradient-based learning.

        Parameters
        ----------
        data : np.ndarray
            Training batch.

        Returns
        -------
        float
            Training loss.
        """
        # Enable training mode
        self.model.train()

        # Prepare data
        pixels = data.reshape(-1, data.shape[-1])
        x = torch.from_numpy(pixels).float().to(self.device)

        # Forward pass
        reconstructed = self.model(x)
        loss = nn.functional.mse_loss(reconstructed, x)

        return loss.item()
```

### Multi-Output Nodes

Nodes with multiple output ports:

```python
class FeatureExtractor(Node):
    """Extract multiple feature types from input."""

    INPUT_SPECS = [
        PortSpec(name="data", dtype=np.ndarray, required=True)
    ]

    OUTPUT_SPECS = [
        PortSpec(name="spectral_features", dtype=np.ndarray),
        PortSpec(name="spatial_features", dtype=np.ndarray),
        PortSpec(name="texture_features", dtype=np.ndarray)
    ]

    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, data: np.ndarray) -> dict:
        """Extract multiple feature types."""
        spectral = self._extract_spectral(data)
        spatial = self._extract_spatial(data)
        texture = self._extract_texture(data)

        return {
            "spectral_features": spectral,
            "spatial_features": spatial,
            "texture_features": texture
        }

    def _extract_spectral(self, data: np.ndarray) -> np.ndarray:
        """Extract spectral features via PCA."""
        # Implementation
        pass

    def _extract_spatial(self, data: np.ndarray) -> np.ndarray:
        """Extract spatial features via convolution."""
        # Implementation
        pass

    def _extract_texture(self, data: np.ndarray) -> np.ndarray:
        """Extract texture features."""
        # Implementation
        pass
```

## Configuration Support

### Add Hydra Configuration

Create `cuvis_ai_plugin/configs/default.yaml`:

```yaml
# Default configuration for CustomAnomalyDetector
custom_anomaly_detector:
  threshold: 0.95
  method: "simple"
  window_size: 5

# Configuration for DeepLearningDetector
deep_learning_detector:
  input_dim: 100
  hidden_dim: 128
  latent_dim: 32
  device: "auto"
  learning_rate: 0.001

# Training configuration
training:
  batch_size: 32
  epochs: 50
  optimizer:
    _target_: torch.optim.Adam
    lr: ${deep_learning_detector.learning_rate}
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

## Testing Best Practices

### Unit Tests

Test individual node functionality:

```python
def test_node_ports():
    """Test node has correct input/output ports."""
    node = CustomAnomalyDetector()

    assert len(node.INPUT_SPECS) == 1
    assert node.INPUT_SPECS[0].name == "data"

    assert len(node.OUTPUT_SPECS) == 2
    assert node.OUTPUT_SPECS[0].name == "scores"
    assert node.OUTPUT_SPECS[1].name == "detections"


def test_node_deterministic():
    """Test node produces deterministic results."""
    node = CustomAnomalyDetector(threshold=0.9, method="simple")

    np.random.seed(42)
    data = np.random.randn(10, 10, 50).astype(np.float32)

    output1 = node(data=data)
    output2 = node(data=data)

    np.testing.assert_array_equal(output1["scores"], output2["scores"])
    np.testing.assert_array_equal(output1["detections"], output2["detections"])
```

### Integration Tests

Test node in pipeline context:

```python
def test_node_in_pipeline():
    """Test node works in pipeline."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry
    from cuvis_ai_core.pipeline.pipeline import Pipeline

    registry = NodeRegistry()
    registry.load_plugins("examples/plugins.yaml")

    pipeline_dict = {
        "nodes": [
            {
                "class_name": "CustomAnomalyDetector",
                "name": "detector",
                "params": {"threshold": 0.95}
            }
        ],
        "edges": []
    }

    pipeline = Pipeline.from_dict(pipeline_dict, node_registry=registry)

    data = np.random.randn(20, 20, 50).astype(np.float32)
    outputs = pipeline(data=data)

    assert "detector" in outputs
    assert "scores" in outputs["detector"]
```

### Performance Tests

Test node performance:

```python
import time


def test_node_performance():
    """Test node processes data within time budget."""
    node = CustomAnomalyDetector()
    data = np.random.randn(100, 100, 224).astype(np.float32)

    start = time.time()
    outputs = node(data=data)
    elapsed = time.time() - start

    # Should process 100x100x224 in < 1 second
    assert elapsed < 1.0
```

## Packaging and Distribution

### Create README.md

```markdown
# CUVIS-AI Custom Anomaly Detection Plugin

Custom anomaly detection algorithms for cuvis-ai framework.

## Installation

### From Git

```bash
# Create plugins.yaml
cat > plugins.yaml << EOF
plugins:
  my_plugin:
    repo: "https://github.com/your-org/cuvis-ai-my-plugin.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_plugin.nodes.custom_node.CustomAnomalyDetector
EOF

# Load in Python
from cuvis_ai_core.utils.node_registry import NodeRegistry
registry = NodeRegistry()
registry.load_plugins("plugins.yaml")
```

### From Local Path

```bash
# For development (recommended)
uv sync

# Alternative: editable install
uv pip install -e .
```

## Usage

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.load_plugin(
    name="my_plugin",
    config={
        "path": "path/to/plugin",
        "provides": ["cuvis_ai_plugin.nodes.custom_node.CustomAnomalyDetector"]
    }
)

CustomDetector = registry.get("CustomAnomalyDetector", instance=registry)
detector = CustomDetector(threshold=0.95)
```

## Nodes

### CustomAnomalyDetector

Statistical anomaly detection using Mahalanobis distance or contextual methods.

**Parameters:**
- `threshold` (float): Detection threshold (0-1)
- `method` (str): 'simple' or 'advanced'
- `window_size` (int): Contextual window size

**Inputs:**
- `data` (np.ndarray): Hyperspectral cube (H, W, C)

**Outputs:**
- `scores` (np.ndarray): Anomaly scores (H, W)
- `detections` (np.ndarray): Binary detections (H, W)

## Development

```bash
git clone https://github.com/your-org/cuvis-ai-my-plugin.git
cd cuvis-ai-my-plugin
uv sync --extra dev
uv run pytest tests/
```

## License

MIT License
```

### Create .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# cuvis cache
.cuvis/
.cuvis_plugins/
```

### Versioning Strategy

Follow semantic versioning (semver):

**Version Format:** `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking API changes
- **MINOR:** New features, backwards compatible
- **PATCH:** Bug fixes

**Git Tags:**
```bash
# Create annotated tag
git tag -a v0.1.0 -m "Initial release"

# Push tags
git push origin v0.1.0
```

**Update Version:**

1. Update `pyproject.toml`:
   ```toml
   version = "0.2.0"
   ```

2. Update `__init__.py`:
   ```python
   __version__ = "0.2.0"
   ```

3. Create CHANGELOG.md entry:
   ```markdown
   ## [0.2.0] - 2026-02-15
   ### Added
   - New advanced detection method
   - GPU acceleration support

   ### Changed
   - Improved threshold handling

   ### Fixed
   - Bug in score normalization
   ```

4. Commit and tag:
   ```bash
   git add pyproject.toml cuvis_ai_plugin/__init__.py CHANGELOG.md
   git commit -m "Bump version to 0.2.0"
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin main v0.2.0
   ```

## Publishing Options

### Option 1: Git Repository (Recommended)

Host on GitHub/GitLab:

```bash
# Create repository on GitHub
gh repo create your-org/cuvis-ai-my-plugin --public

# Push code
git remote add origin https://github.com/your-org/cuvis-ai-my-plugin.git
git push -u origin main
git push origin v0.1.0
```

**Users install via:**
```yaml
plugins:
  my_plugin:
    repo: "https://github.com/your-org/cuvis-ai-my-plugin.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_plugin.nodes.custom_node.CustomAnomalyDetector
```

### Option 2: PyPI (Public/Private)

Build and upload to PyPI:

```bash
# Install build tools with uv
uv pip install build twine

# Build package
uv run python -m build

# Upload to PyPI
uv run python -m twine upload dist/*

# Or upload to TestPyPI first
uv run python -m twine upload --repository testpypi dist/*
```

**Users install via:**
```bash
# Using uv (recommended)
uv add cuvis-ai-my-plugin

# Or using uv pip
uv pip install cuvis-ai-my-plugin
```

Then use directly (automatic discovery):
```python
from cuvis_ai_core.utils.node_registry import NodeRegistry
CustomDetector = NodeRegistry.get("CustomAnomalyDetector")
```

### Option 3: Private Package Repository

For enterprise/private plugins:

```bash
# Upload to private PyPI server
twine upload --repository-url https://your-pypi.company.com/ dist/*

# Or use artifact repository (e.g., Artifactory, Nexus)
```

## Best Practices

### 1. Node Design

- **Single Responsibility:** Each node should do one thing well
- **Input Validation:** Validate all inputs in `forward()` method
- **Error Handling:** Raise clear, actionable exceptions
- **Port Documentation:** Document all ports in `PortSpec` descriptions
- **Docstrings:** Follow NumPy-style docstrings (see [Docstring Guide](../development/docstrings.md))

### 2. Testing

- **Coverage:** Aim for >90% test coverage
- **Edge Cases:** Test boundary conditions and error cases
- **Integration:** Test nodes in pipeline context
- **Performance:** Benchmark critical operations
- **Fixtures:** Use pytest fixtures for reusable test data

### 3. Documentation

- **README:** Clear installation and usage instructions
- **Examples:** Provide working code examples
- **Changelog:** Maintain version history
- **API Docs:** Document all public APIs
- **Tutorials:** Create guides for complex workflows

### 4. Performance

- **Profiling:** Profile code to identify bottlenecks
- **Vectorization:** Use NumPy vectorized operations
- **GPU Support:** Leverage GPUs for compute-heavy operations
- **Memory:** Minimize memory allocations
- **Caching:** Cache expensive computations when appropriate

### 5. Dependencies

- **Minimal:** Only include necessary dependencies
- **Pinning:** Pin versions for reproducibility
- **Optional:** Use optional dependencies for extras
- **Security:** Audit dependencies for vulnerabilities

## Troubleshooting

### Plugin Not Loading

**Issue:** Plugin fails to load with import errors.

**Solution:**
1. Verify `pyproject.toml` exists
2. Check all `__init__.py` files present
3. Ensure dependencies installed: `uv sync` (or `uv pip install -e .`)
4. Test import manually: `uv run python -c "from cuvis_ai_plugin.nodes import CustomAnomalyDetector"`

### Node Not Found

**Issue:** Node not found after loading plugin.

**Solution:**
1. Check `provides` list in manifest includes full class path
2. Verify class name spelling matches exactly
3. Ensure class inherits from `Node`
4. Check `__all__` exports in `__init__.py`

### Test Failures

**Issue:** Tests fail unexpectedly.

**Solution:**
1. Run with verbose output: `uv run pytest tests/ -v`
2. Check test data shapes match node expectations
3. Verify NumPy random seeds for reproducibility
4. Isolate failing test: `uv run pytest tests/test_custom_node.py::test_name -v`

### Performance Issues

**Issue:** Node is too slow.

**Solution:**
1. Profile code: `uv run python -m cProfile -o profile.stats your_script.py`
2. Analyze: `uv run python -m pstats profile.stats`
3. Vectorize loops using NumPy
4. Use `numba` JIT compilation for hot loops
5. Consider GPU acceleration with PyTorch/CuPy

## Example: Complete Plugin

See the [cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip) plugin for a complete real-world example.

## See Also

- **[Plugin System Overview](overview.md)** - Plugin architecture and concepts
- **[Plugin Usage Guide](usage.md)** - Using plugins in workflows
- **[Node System Deep Dive](../concepts/node-system-deep-dive.md)** - Node architecture details
- **[Port System Deep Dive](../concepts/port-system-deep-dive.md)** - Port specifications and connections
- **[Two-Phase Training](../concepts/two-phase-training.md)** - Statistical initialization patterns
- **[Docstring Standards](../development/docstrings.md)** - Documentation guidelines
