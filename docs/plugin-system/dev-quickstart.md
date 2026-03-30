!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin Development: Quick Start

This guide walks you through creating a cuvis-ai plugin from scratch.

## Prerequisites

- **Python 3.9+** installed
- **cuvis-ai** and **cuvis-ai-core** installed
- **Git** for version control
- **uv** for package management (recommended over pip)
- Basic understanding of the [Node System](../concepts/node-system-deep-dive.md)
- Familiarity with [Port System](../concepts/port-system-deep-dive.md)

**Note:** This guide uses `uv` for all package management. Use `uv run python ...` instead of `python ...`.

## Step 1: Create Plugin Structure

```bash
mkdir my-cuvis-plugin
cd my-cuvis-plugin

mkdir -p cuvis_ai_plugin/nodes
mkdir -p cuvis_ai_plugin/configs
mkdir -p tests

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

## Step 2: Configure pyproject.toml

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

## Step 3: Implement Your First Node

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

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        )
    }

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
        self.threshold = threshold
        self.method = method
        self.window_size = window_size

        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        if method not in ["simple", "advanced"]:
            raise ValueError(f"method must be 'simple' or 'advanced', got {method}")

        # CRITICAL: Pass ALL hyperparameters for serialization
        super().__init__(
            threshold=threshold,
            method=method,
            window_size=window_size,
            **kwargs,
        )

    def forward(self, data: torch.Tensor, context: Context, **kwargs: Any) -> dict[str, Any]:
        """Process input data and detect anomalies."""
        if data.ndim != 4:
            raise ValueError(f"Expected 4D input [B, H, W, C], got shape {data.shape}")

        scores = self._compute_scores(data)
        detections = (scores > self.threshold).float()

        return {
            "scores": scores,
            "detections": detections,
        }

    def _compute_scores(self, data: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores using Mahalanobis distance."""
        B, H, W, C = data.shape
        pixels = data.reshape(B, H * W, C)

        mean = pixels.mean(dim=1, keepdim=True)
        centered = pixels - mean

        cov = torch.bmm(centered.transpose(1, 2), centered) / (H * W - 1)
        cov = cov + torch.eye(C, device=data.device) * 1e-6

        cov_inv = torch.linalg.inv(cov)
        mahal_sq = torch.bmm(
            torch.bmm(centered, cov_inv),
            centered.unsqueeze(-1)
        ).squeeze(-1)

        scores = torch.sqrt(mahal_sq.clamp(min=0))
        scores = scores.reshape(B, H, W, 1)

        scores_min = scores.amin(dim=(1, 2), keepdim=True)
        scores_max = scores.amax(dim=(1, 2), keepdim=True)
        scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)
        return scores
```

## Step 4: Export Your Node

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

## Step 5: Create Tests

Create `tests/test_custom_node.py`:

```python
"""Tests for custom anomaly detector node."""

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_plugin.nodes import CustomAnomalyDetector


def test_custom_node_initialization():
    node = CustomAnomalyDetector(threshold=0.95, method="simple")
    assert node.threshold == 0.95
    assert node.method == "simple"


def test_custom_node_invalid_threshold():
    with pytest.raises(ValueError, match="threshold must be in"):
        CustomAnomalyDetector(threshold=1.5)


def test_custom_node_simple_method():
    node = CustomAnomalyDetector(threshold=0.9, method="simple")
    data = torch.randn(2, 10, 10, 50, dtype=torch.float32)
    context = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)

    outputs = node(data=data, context=context)

    assert "scores" in outputs
    assert "detections" in outputs
    assert outputs["scores"].shape == (2, 10, 10, 1)
    assert outputs["detections"].shape == (2, 10, 10, 1)


def test_custom_node_threshold_behavior():
    data = torch.randn(2, 10, 10, 50, dtype=torch.float32)
    context = Context(stage=ExecutionStage.INFERENCE)

    node_high = CustomAnomalyDetector(threshold=0.99)
    outputs_high = node_high(data=data, context=context)

    node_low = CustomAnomalyDetector(threshold=0.5)
    outputs_low = node_low(data=data, context=context)

    assert outputs_low["detections"].sum() >= outputs_high["detections"].sum()
```

## Step 6: Test Locally

```bash
uv sync --extra dev
uv run pytest tests/ -v
uv run pytest tests/ --cov=cuvis_ai_plugin --cov-report=html
```

## Step 7: Create Plugin Manifest

Create `examples/plugins.yaml`:

```yaml
plugins:
  my_plugin:
    path: "."  # Current directory
    provides:
      - cuvis_ai_plugin.nodes.custom_node.CustomAnomalyDetector
```

## Step 8: Test in Pipeline

Create `examples/test_plugin.py`:

```python
"""Test plugin in a pipeline."""

import torch
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.pipeline.pipeline import Pipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context


def main():
    registry = NodeRegistry()
    registry.load_plugins("examples/plugins.yaml")

    CustomAnomalyDetector = registry.get("CustomAnomalyDetector", instance=registry)
    print(f"Node loaded: {CustomAnomalyDetector}")

    pipeline_dict = {
        "nodes": [
            {
                "class_name": "CustomAnomalyDetector",
                "name": "detector",
                "hparams": {"threshold": 0.95, "method": "simple"}
            }
        ],
        "connections": []
    }

    pipeline = Pipeline.from_dict(pipeline_dict, node_registry=registry)

    test_data = torch.randn(4, 50, 50, 100, dtype=torch.float32)
    context = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    outputs = pipeline(data=test_data, context=context)

    print(f"Pipeline executed successfully")
    print(f"  Anomaly scores shape: {outputs['detector']['scores'].shape}")
    print(f"  Detections: {outputs['detector']['detections'].sum().item()} pixels")


if __name__ == "__main__":
    main()
```

```bash
uv run python examples/test_plugin.py
```

## Next Steps

- [Advanced Node Development](dev-advanced.md) — statistical nodes, deep learning nodes, multi-output nodes
- [Testing & Publishing](dev-testing.md) — testing best practices, publishing options
- [Plugin Packaging](packaging.md) — README templates, versioning, distribution
