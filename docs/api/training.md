# Training

Complete API reference for the PyTorch Lightning-based training infrastructure.

## Overview

The `cuvis_ai.training` module provides a comprehensive training system built on PyTorch Lightning, featuring:

- **Two-Phase Training**: Statistical initialization followed by gradient optimization
- **Leaf Nodes**: Attach losses, metrics, and visualizations to any graph node
- **Monitoring**: Multiple backends (DummyMonitor, WandB, TensorBoard) with graceful degradation
- **Configuration**: Full Hydra integration for reproducible experiments
- **Serialization**: Complete experiment state preservation

### Training Workflow

```mermaid
graph LR
    A[Configure] --> B[Statistical Init]
    B --> C[Prepare for Training]
    C --> D[Gradient Training]
    D --> E[Monitoring & Metrics]
    E --> F[Checkpointing]
```

## Quick Example

```python
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.training.losses import OrthogonalityLoss
from cuvis_ai.training.metrics import ExplainedVarianceMetric
from cuvis_ai.training.monitors import DummyMonitor
from cuvis_ai.training.config import TrainingConfig, TrainerConfig

# Build graph
graph = Graph("trainable_pca")
pca = TrainablePCA(n_components=3, trainable=True)
graph.add_node(pca)

# Add loss and metric leaves
loss = OrthogonalityLoss(weight=1.0)
metric = ExplainedVarianceMetric()
graph.add_leaf_node(loss, parent=pca)
graph.add_leaf_node(metric, parent=pca)

# Register monitoring
monitor = DummyMonitor(output_dir="./outputs")
graph.register_monitor(monitor)

# Configure training
config = TrainingConfig(
    seed=42,
    trainer=TrainerConfig(
        max_epochs=10,
        accelerator="auto"
    )
)

# Train - Phase 1 (statistical) + Phase 2 (gradient)
trainer = graph.train(datamodule=datamodule, training_config=config)
```

## Configuration

Training configuration using Hydra-compatible dataclasses.

### TrainingConfig

Top-level configuration bundle wrapping all training settings.

**Key Components:**
- `trainer`: Lightning Trainer configuration
- `optimizer`: Optimizer and scheduler settings
- `seed`: Random seed for reproducibility
- `monitor_plugins`: List of monitoring backends

### TrainerConfig

PyTorch Lightning Trainer settings.

**Key Parameters:**
- `max_epochs`: Number of training epochs (0 = statistical only)
- `accelerator`: Device type ("auto", "gpu", "cpu", "mps")
- `devices`: Number of devices to use
- `precision`: Training precision ("32", "16-mixed", "bf16-mixed")

### OptimizerConfig

Optimizer and learning rate scheduler configuration.

**Key Parameters:**
- `name`: Optimizer type ("adam", "adamw", "sgd")
- `lr`: Learning rate
- `weight_decay`: L2 regularization
- `scheduler`: LR scheduler ("reduce_on_plateau", "cosine", etc.)

::: cuvis_ai.training.config

## Data Module

Base class for PyTorch Lightning data modules with graph-specific contracts.

### GraphDataModule

Abstract base class enforcing dictionary-only batch contract.

**Required Batch Keys:**
- `"cube"` or `"x"`: Input hyperspectral cube
- `"mask"` or `"labels"` (optional): Ground truth annotations

**Example:**
```python
from cuvis_ai.training.datamodule import GraphDataModule

class MyDataModule(GraphDataModule):
    def setup(self, stage=None):
        self.train_dataset = MyDataset(...)
        self.val_dataset = MyDataset(...)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=4)
```

::: cuvis_ai.training.datamodule

## Lightning Module

Internal PyTorch Lightning module handling training logic.

### CuvisLightningModule

Orchestrates training, validation, and optimization.

**Key Methods:**
- `training_step()`: Forward pass + loss computation + backprop
- `validation_step()`: Forward pass + metrics + visualizations
- `configure_optimizers()`: Setup optimizer and scheduler

**Note:** This module is internal and typically not instantiated directly. Use `graph.train()` instead.

::: cuvis_ai.training.lightning_module

## Leaf Nodes

Base classes for attaching losses, metrics, and visualizations to graph nodes.

### LeafNode

Base class for all leaf nodes with parent validation.

**Key Features:**
- `compatible_parent_types`: Tuple of allowed parent node classes
- `required_parent_attributes`: Tuple of required parent attributes
- `validate_parent()`: Ensures parent compatibility

### LossNode

Compute losses for backpropagation.

**Protocol:**
- `compute_loss(parent_output, batch) -> torch.Tensor`

### MetricNode

Compute evaluation metrics (no gradients).

**Protocol:**
- `compute_metric(parent_output, batch) -> Dict[str, float]`

### VisualizationNode

Generate visual artifacts for monitoring.

**Protocol:**
- `visualize(parent_output, batch, step, stage) -> Dict[str, Any]`

### MonitoringNode

Backend for logging metrics and artifacts.

**Protocol:**
- `log_metrics(metrics, step, stage)`
- `log_artifact(artifact, name, step, stage)`

::: cuvis_ai.training.leaf_nodes

## Loss Functions

Collection of loss nodes for training graph nodes.

### OrthogonalityLoss

Maintain PCA component orthonormality.

**Formula:** `L = weight * ||W @ W^T - I||²_F`

**Example:**
```python
loss = OrthogonalityLoss(weight=1.0)
graph.add_leaf_node(loss, parent=pca)
```

### AnomalyBCEWithLogits

Binary cross-entropy for anomaly detection.

**Features:**
- Numerically stable (uses logits)
- Class imbalance handling via `pos_weight`

**Example:**
```python
loss = AnomalyBCEWithLogits(weight=1.0, pos_weight=10.0)
graph.add_leaf_node(loss, parent=logit_head)
```

### MSEReconstructionLoss

Mean squared error for reconstruction tasks.

**Example:**
```python
loss = MSEReconstructionLoss(reduction="mean")
graph.add_leaf_node(loss, parent=decoder)
```

### SelectorEntropyRegularizer

Encourage exploration in channel selection.

**Formula:** `L = -weight * entropy` where `entropy = -Σ(p log p)`

**Example:**
```python
loss = SelectorEntropyRegularizer(weight=0.01)
graph.add_leaf_node(loss, parent=selector)
```

### SelectorDiversityRegularizer

Prevent concentration on few channels.

**Formula:** `L = weight * (-variance(weights))`

**Example:**
```python
loss = SelectorDiversityRegularizer(weight=0.01)
graph.add_leaf_node(loss, parent=selector)
```

::: cuvis_ai.training.losses

## Metrics

Collection of metric nodes for evaluation.

### ExplainedVarianceMetric

Track PCA explained variance per component.

**Output:** `{comp_0: 0.85, comp_1: 0.10, cumulative: 0.95}`

### ComponentOrthogonalityMetric

Monitor PCA orthogonality during training.

**Output:** `{orthogonality_error: 0.001}`

### AnomalyDetectionMetrics

Comprehensive anomaly detection evaluation.

**Output:** `{precision: 0.9, recall: 0.85, f1: 0.87, iou: 0.77}`

### ScoreStatisticsMetric

Distribution statistics for anomaly scores.

**Output:** `{mean: 5.2, std: 2.1, q50: 4.8, q95: 9.5}`

::: cuvis_ai.training.metrics

## Visualizations

Collection of visualization nodes for monitoring.

### PCAVisualization

2D/3D scatter plots of PCA projections.

**Features:**
- Automatic 2D/3D mode
- Color by labels or scores
- Explained variance in axis labels

**Example:**
```python
viz = PCAVisualization(log_frequency=5, subsample=1000)
graph.add_leaf_node(viz, parent=pca)
```

### AnomalyHeatmap

Spatial heatmap of anomaly scores.

**Features:**
- 3-panel layout: heatmap, overlay, binary mask
- Configurable colormap and threshold

**Example:**
```python
viz = AnomalyHeatmap(log_frequency=10, colormap="hot")
graph.add_leaf_node(viz, parent=rx)
```

### ScoreHistogram

Distribution histogram with statistics.

**Features:**
- Histogram with KDE overlay
- Summary statistics
- Anomaly threshold line

**Example:**
```python
viz = ScoreHistogram(log_frequency=10, n_bins=50)
graph.add_leaf_node(viz, parent=rx)
```

### Selector Visualizations

Specialized visualizations for channel selection monitoring:

- `SelectorTemperaturePlot`: Temperature annealing curve
- `SelectorChannelMaskPlot`: Channel selection probabilities
- `SelectorStabilityPlot`: Selection stability (Jaccard similarity)

::: cuvis_ai.training.visualizations

## Monitoring Backends

Multiple monitoring backends with graceful degradation.

### DummyMonitor

Filesystem-based monitoring (no external dependencies).

**Features:**
- JSONL metric logging
- PKL + PNG artifact persistence
- Always available

**Example:**
```python
monitor = DummyMonitor(output_dir="./outputs/artifacts")
graph.register_monitor(monitor)
```

### WandBMonitor

Weights & Biases experiment tracking.

**Features:**
- Web UI dashboard
- Hyperparameter sweeps
- Team collaboration
- Online/offline/disabled modes
- Graceful degradation (works without wandb installed)

**Example:**
```python
monitor = WandBMonitor(
    project="my-project",
    entity="my-team",
    mode="online"
)
graph.register_monitor(monitor)
```

### TensorBoardMonitor

Local TensorBoard logging.

**Features:**
- Local web UI
- Scalar/image logging
- Histogram tracking
- Graceful degradation (works without tensorboard installed)

**Example:**
```python
monitor = TensorBoardMonitor(
    log_dir="./outputs/tensorboard",
    flush_secs=30
)
graph.register_monitor(monitor)
```

::: cuvis_ai.training.monitors

## See Also

- **[Graph API](pipeline.md)**: Build and train graphs
- **[Node API](nodes.md)**: Available node types
- **[Configuration Guide](../user-guide/configuration.md)**: Hydra configuration
- **[Tutorials](../tutorials/phase3_gradient_training.md)**: Training tutorials
