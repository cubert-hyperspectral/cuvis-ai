# Phase 2: Visualization & Monitoring

This tutorial covers Phase 2 of the CUVIS.AI training pipeline: adding visualization leaves and monitoring infrastructure. You'll learn how to generate insightful visualizations and track experiments with flexible monitoring backends.

## Overview

Phase 2 extends the statistical pipeline with:

- **Visualization Leaves**: Generate PCA plots, anomaly heatmaps, score histograms
- **Monitoring Infrastructure**: Track artifacts with DummyMonitor, WandB, TensorBoard
- **Leaf Node Protocol**: Validate parent compatibility, schedule logging frequency
- **Artifact Persistence**: Save figures as pickle + PNG pairs for analysis

This phase is valuable because it:

1. Provides visual feedback on model behavior
2. Enables experiment tracking and comparison
3. Helps identify issues early (e.g., collapsed PCA, poor anomaly detection)
4. Supports offline artifact generation for debugging

## What You'll Learn

- Types of visualization leaves (PCA, heatmap, histogram)
- Monitoring backends (DummyMonitor, WandB, TensorBoard)
- Leaf node parent validation
- Controlling logging frequency
- Interpreting visualization outputs
- Configuring monitoring with Hydra

## Prerequisites

- Completed [Phase 1 Tutorial](phase1_statistical.md)
- Understanding of statistical initialization
- Basic familiarity with matplotlib (helpful but not required)

## Concepts

### Leaf Nodes

Leaf nodes are special nodes that don't produce outputs for downstream nodes. Instead, they:

- **Generate artifacts** (visualizations, metrics, logs)
- **Validate parent compatibility** (e.g., PCAVisualization requires PCA parent)
- **Schedule execution** (log every N steps)
- **Emit to monitors** (WandB, TensorBoard, filesystem)

Leaf node types:

- `VisualizationNode`: Generates matplotlib figures
- `MetricNode`: Computes evaluation metrics (Phase 3)
- `LossNode`: Computes loss values (Phase 3)
- `MonitoringNode`: Custom monitoring logic

### Monitoring Architecture

CUVIS.AI uses a plugin-based monitoring system:

```
Graph → Lightning Module → Monitoring Plugins
                           ├─ DummyMonitor (filesystem)
                           ├─ WandBMonitor (Weights & Biases)
                           └─ TensorBoardMonitor (TensorBoard)
```

All monitors implement a common protocol:

- `setup()`: Initialize connection/session
- `log_metric(name, value, step)`: Log scalar metrics
- `log_artifact(name, artifact, step)`: Log figures/files
- `teardown()`: Cleanup resources

## Tutorial: Adding Visualizations to RX Pipeline

### Step 1: Imports

```python
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.training.config import TrainingConfig, TrainerConfig

# Visualization leaves
from cuvis_ai.training.visualizations import (
    AnomalyHeatmap,
    ScoreHistogram,
)

# Monitoring
from cuvis_ai.training.monitors import DummyMonitor
```

### Step 2: Build Graph with Visualizations

```python
# Create graph
graph = Graph("rx_with_visualizations")

# Add processing nodes (from Phase 1)
normalizer = MinMaxNormalizer(eps=1e-6)
rx = RXGlobal(eps=1e-6, trainable_stats=False)
graph.add_node(normalizer)
graph.add_node(rx, parent=normalizer)

# Add visualization leaves
heatmap = AnomalyHeatmap(
    log_frequency=1,              # Log every validation batch
    colormap="hot",               # Matplotlib colormap
    vmin=0.0,                     # Min value for colormap
    vmax=None,                    # Max value (auto if None)
)
graph.add_leaf_node(heatmap, parent=rx)

histogram = ScoreHistogram(
    log_frequency=1,              # Log every validation batch
    bins=50,                      # Number of histogram bins
    compute_threshold=True,       # Show mean + 2σ threshold
)
graph.add_leaf_node(histogram, parent=rx)
```

**Key Points:**

- `graph.add_leaf_node()` validates parent compatibility
- `log_frequency=1` means visualize every batch (can be slower)
- `log_frequency=5` means visualize every 5th batch (faster)
- Leaf nodes don't affect forward pass (no outputs to downstream nodes)

### Step 3: Register Monitoring Backend

```python
# Create DummyMonitor (saves to filesystem)
monitor = DummyMonitor(
    output_dir="./outputs/artifacts",  # Where to save artifacts
    save_thumbnails=True,              # Also save PNG previews
)

# Register monitor with graph
graph.register_monitor(monitor)
```

**Alternative Monitors:**

```python
# WandB Monitor (requires wandb package)
from cuvis_ai.training.monitors import WandBMonitor
wandb_monitor = WandBMonitor(
    project="cuvis_ai_experiments",
    entity="my_team",
    name="rx_baseline",
    tags=["phase2", "rx", "lentils"],
)
graph.register_monitor(wandb_monitor)

# TensorBoard Monitor
from cuvis_ai.training.monitors import TensorBoardMonitor
tb_monitor = TensorBoardMonitor(
    log_dir="./outputs/tensorboard",
    flush_secs=30,
)
graph.register_monitor(tb_monitor)
```

### Step 4: Configure Training

```python
# Training configuration (same as Phase 1)
config = TrainingConfig(
    seed=42,
    trainer=TrainerConfig(
        max_epochs=0,            # Still just statistical initialization
        accelerator="auto",
        devices=1,
    )
)
```

### Step 5: Run Training

```python
# Train with visualizations
trainer = graph.train(
    datamodule=LentilsAnomaly(
        data_dir="./data/Lentils",
        batch_size=4,
        num_workers=0,
    ),
    training_config=config
)
```

**Expected Output:**

```
[INFO] Initializing statistical nodes...
[INFO] Statistical initialization complete
[INFO] Running validation to generate visualizations...
[INFO] Validation batch 0: Generating visualizations...
[INFO]   - train/viz/rx_AnomalyHeatmap_0.pkl saved
[INFO]   - train/viz/rx_AnomalyHeatmap_0.png saved
[INFO]   - train/viz/rx_ScoreHistogram_0.pkl saved
[INFO]   - train/viz/rx_ScoreHistogram_0.png saved
[INFO] Validation complete
```

### Step 6: Examine Outputs

Check the generated artifacts:

```bash
ls outputs/artifacts/step_000000/
# train/
# └── viz/
#     ├── rx_AnomalyHeatmap_0.pkl
#     ├── rx_AnomalyHeatmap_0.png    # PNG preview
#     ├── rx_ScoreHistogram_0.pkl
#     └── rx_ScoreHistogram_0.png    # PNG preview
```

Load and display a visualization:

```python
import pickle
import matplotlib.pyplot as plt

# Load pickle artifact
with open("outputs/artifacts/step_000000/train/viz/rx_AnomalyHeatmap_0.pkl", "rb") as f:
    artifact = pickle.load(f)

# Display figure
fig = artifact["figure"]
plt.figure(fig.number)
plt.show()

# Or just open the PNG
from PIL import Image
img = Image.open("outputs/artifacts/step_000000/train/viz/rx_AnomalyHeatmap_0.png")
img.show()
```

## Visualization Types

### AnomalyHeatmap

Generates a 3-panel figure:

1. **Raw Heatmap**: Color-coded anomaly scores
2. **Overlay**: Scores overlaid on original cube channel
3. **Binary Mask**: Thresholded anomaly regions

**Configuration:**

```python
heatmap = AnomalyHeatmap(
    log_frequency=5,              # Log every 5th batch
    colormap="hot",               # Colormap: "hot", "viridis", "jet"
    vmin=0.0,                     # Min score for colormap
    vmax=15.0,                    # Max score for colormap
    threshold_multiplier=2.0,     # Threshold = mean + 2*std
)
```

**Output Interpretation:**

- **Panel 1 (Heatmap)**: Bright regions = high anomaly scores
- **Panel 2 (Overlay)**: Shows spatial context with original data
- **Panel 3 (Binary Mask)**: White = detected anomalies

**Common Issues:**

- All white/black: Adjust `vmin`/`vmax` or `threshold_multiplier`
- No anomalies detected: Check RX initialization, try different threshold

### ScoreHistogram

Shows distribution of anomaly scores with statistics:

- Histogram of all scores
- Mean, std, min, max statistics
- Threshold line (mean + 2σ by default)

**Configuration:**

```python
histogram = ScoreHistogram(
    log_frequency=1,              # Log frequency
    bins=50,                      # Number of histogram bins
    compute_threshold=True,       # Show threshold line
    threshold_multiplier=2.0,     # Threshold = mean + N*std
)
```

**Output Interpretation:**

- **Shape**: Should be roughly normal (bell curve) for good RX detector
- **Long tail**: High scores = potential anomalies
- **Bimodal**: Two distinct populations (normal + anomalous)
- **Threshold line**: Separates normal from anomalous

**Common Patterns:**

- **Tight distribution (low std)**: Homogeneous dataset, few anomalies
- **Wide distribution (high std)**: Heterogeneous dataset, many unusual pixels
- **Skewed right**: Many high scores, aggressive detector

### PCAVisualization

Shows 2D or 3D projections of data in principal component space:

```python
from cuvis_ai.training.visualizations import PCAVisualization
from cuvis_ai.node.pca import TrainablePCA

# Add PCA node
pca = TrainablePCA(n_components=3, trainable=False)
graph.add_node(pca, parent=normalizer)

# Add PCA visualization
pca_viz = PCAVisualization(
    log_frequency=1,
    n_dims=2,                     # 2D or 3D visualization
    subsample=1000,               # Max points to plot (for performance)
    show_variance=True,           # Show explained variance
)
graph.add_leaf_node(pca_viz, parent=pca)
```

**Output Interpretation:**

- **Clustering**: Similar pixels cluster together in PC space
- **Outliers**: Points far from main cluster may be anomalies
- **Explained variance**: Shows information retention (e.g., PC1: 85%, PC2: 10%, PC3: 3%)

## Monitoring Backends

### DummyMonitor (Filesystem)

Saves artifacts to local filesystem. Good for:

- Development and debugging
- Offline analysis
- No external dependencies

**Usage:**

```python
monitor = DummyMonitor(
    output_dir="./outputs/artifacts",
    save_thumbnails=True,          # Save PNG alongside PKL
    metrics_format="jsonl",        # Metrics format: "jsonl" or "json"
)
graph.register_monitor(monitor)
```

**Output Structure:**

```
outputs/artifacts/
├── step_000000/              # Validation step 0
│   ├── train/
│   │   └── viz/
│   │       ├── rx_AnomalyHeatmap_0.pkl
│   │       └── rx_AnomalyHeatmap_0.png
│   └── metrics.jsonl         # Metrics log
└── step_000100/              # Validation step 100
    └── ...
```

**Metrics JSONL:**

```json
{"step": 0, "stage": "train", "name": "loss/total", "value": 0.45}
{"step": 0, "stage": "train", "name": "metric/accuracy", "value": 0.92}
```

### WandBMonitor (Weights & Biases)

Integrates with [Weights & Biases](https://wandb.ai) for experiment tracking. Good for:

- Team collaboration
- Experiment comparison
- Rich visualizations
- Hyperparameter tracking

**Setup:**

```bash
# Install wandb
pip install wandb

# Login (one-time)
wandb login
```

**Usage:**

```python
from cuvis_ai.training.monitors import WandBMonitor

wandb_monitor = WandBMonitor(
    project="cuvis_ai_experiments",    # Project name
    entity="my_team",                  # Team/username
    name="phase2_rx_baseline",         # Run name
    tags=["phase2", "rx", "lentils"],  # Searchable tags
    notes="Adding visualizations",     # Run description
    config=None,                       # Auto-captured from TrainingConfig
    mode="online",                     # "online", "offline", or "disabled"
)
graph.register_monitor(wandb_monitor)
```

**Features:**

- Automatic config tracking
- Interactive visualizations
- Metric plots over time
- Hyperparameter comparison
- Model artifact storage

**Offline Mode:**

```python
wandb_monitor = WandBMonitor(
    project="cuvis_ai",
    mode="offline",  # Logs locally, sync later with `wandb sync`
)
```

### TensorBoardMonitor

Integrates with TensorBoard for local experiment tracking. Good for:

- No cloud dependency
- Fast local visualization
- Standard PyTorch workflow

**Usage:**

```python
from cuvis_ai.training.monitors import TensorBoardMonitor

tb_monitor = TensorBoardMonitor(
    log_dir="./outputs/tensorboard",   # TensorBoard log directory
    flush_secs=30,                     # Flush to disk every N seconds
)
graph.register_monitor(tb_monitor)
```

**Launch TensorBoard:**

```bash
tensorboard --logdir=./outputs/tensorboard
# Open browser to http://localhost:6006
```

**Features:**

- Scalar plots (loss, metrics)
- Image visualization
- Histogram plots
- Graph visualization
- Profiling tools

### Multiple Monitors

You can register multiple monitors simultaneously:

```python
# Register all three
graph.register_monitor(DummyMonitor(output_dir="./outputs/artifacts"))
graph.register_monitor(WandBMonitor(project="cuvis_ai"))
graph.register_monitor(TensorBoardMonitor(log_dir="./outputs/tensorboard"))

# Artifacts logged to all three backends
```

## Hydra Configuration

For reproducible experiments, use Hydra configuration.

### Configuration File

Create `phase2_config.yaml`:

```yaml
defaults:
  - general
  - _self_

graph:
  name: rx_with_visualizations

nodes:
  normalizer:
    _target_: cuvis_ai.normalization.normalization.MinMaxNormalizer
    eps: 1.0e-6
  
  rx:
    _target_: cuvis_ai.anomaly.rx_detector.RXGlobal
    eps: 1.0e-6
    trainable_stats: false

# Visualization leaves
visualization_leaves:
  anomaly_heatmap:
    _target_: cuvis_ai.training.visualizations.AnomalyHeatmap
    log_frequency: 1
    colormap: hot
    vmin: 0.0
    vmax: null
  
  score_histogram:
    _target_: cuvis_ai.training.visualizations.ScoreHistogram
    log_frequency: 1
    bins: 50
    compute_threshold: true

# Monitoring
monitoring:
  dummy:
    _target_: cuvis_ai.training.monitors.DummyMonitor
    output_dir: ./outputs/artifacts
    save_thumbnails: true
  
  wandb:
    enabled: false  # Set to true to enable
    _target_: cuvis_ai.training.monitors.WandBMonitor
    project: cuvis_ai_experiments
    entity: null
    name: phase2_rx_baseline
    tags: [phase2, rx, lentils]
    mode: online

datamodule:
  _target_: cuvis_ai.data.lentils_anomaly.LentilsAnomaly
  data_dir: ${oc.env:DATA_ROOT,./data/Lentils}
  batch_size: 4
  num_workers: 0

training:
  seed: 42
  trainer:
    max_epochs: 0
    accelerator: auto
    devices: 1
```

### Training Script

Create `train_phase2.py`:

```python
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.training.config import TrainingConfig

@hydra.main(config_path=".", config_name="phase2_config", version_base=None)
def main(cfg: DictConfig):
    # Build graph
    graph = Graph(cfg.graph.name)
    
    # Add processing nodes
    normalizer = instantiate(cfg.nodes.normalizer)
    rx = instantiate(cfg.nodes.rx)
    graph.add_node(normalizer)
    graph.add_node(rx, parent=normalizer)
    
    # Add visualization leaves
    heatmap = instantiate(cfg.visualization_leaves.anomaly_heatmap)
    histogram = instantiate(cfg.visualization_leaves.score_histogram)
    graph.add_leaf_node(heatmap, parent=rx)
    graph.add_leaf_node(histogram, parent=rx)
    
    # Register monitors
    dummy_monitor = instantiate(cfg.monitoring.dummy)
    graph.register_monitor(dummy_monitor)
    
    if cfg.monitoring.wandb.get('enabled', False):
        wandb_monitor = instantiate(cfg.monitoring.wandb)
        graph.register_monitor(wandb_monitor)
    
    # Instantiate datamodule
    datamodule = instantiate(cfg.datamodule)
    
    # Parse training config
    training_cfg = TrainingConfig.from_dict_config(cfg.training)
    
    # Train
    trainer = graph.train(
        datamodule=datamodule,
        training_config=training_cfg
    )
    
    print("✓ Training complete!")
    print(f"✓ Artifacts saved to {cfg.monitoring.dummy.output_dir}")

if __name__ == "__main__":
    main()
```

### Run with CLI Overrides

```bash
# Use defaults
python train_phase2.py

# Enable WandB
python train_phase2.py monitoring.wandb.enabled=true

# Change visualization frequency
python train_phase2.py \
    visualization_leaves.anomaly_heatmap.log_frequency=5 \
    visualization_leaves.score_histogram.log_frequency=5

# Different colormap
python train_phase2.py \
    visualization_leaves.anomaly_heatmap.colormap=viridis
```

## Advanced Patterns

### Conditional Visualization

Visualize only anomalous batches:

```python
class ConditionalHeatmap(AnomalyHeatmap):
    def should_log(self, step, stage):
        # Only log if anomaly rate > threshold
        if not super().should_log(step, stage):
            return False
        # Custom logic here
        return self.last_anomaly_rate > 0.01
```

### Custom Visualization

Create your own visualization leaf:

```python
from cuvis_ai.training.leaf_nodes import VisualizationNode
import matplotlib.pyplot as plt

class CustomVisualization(VisualizationNode):
    def __init__(self, log_frequency=1):
        super().__init__()
        self.log_frequency = log_frequency
    
    def visualize(self, parent_output, batch, batch_idx, stage):
        """Generate custom visualization"""
        x, y, m = parent_output
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        # ... custom plotting logic ...
        
        # Return artifact dict
        return {
            "figure": fig,
            "metadata": {"batch_idx": batch_idx, "stage": stage}
        }
```

### Subsampling for Performance

For large datasets, subsample visualizations:

```python
heatmap = AnomalyHeatmap(
    log_frequency=10,              # Log every 10th batch
    max_samples_per_batch=2,       # Only first 2 samples in batch
)
```

## Troubleshooting

### Out of Memory (Visualizations)

**Problem:** Training runs out of memory when generating visualizations.

**Solution:**

1. Increase `log_frequency`: `log_frequency=10`
2. Reduce batch size during validation
3. Close figures after saving: Monitors do this automatically
4. Disable visualizations for large-scale experiments

### Blank/Empty Figures

**Problem:** Visualizations are empty or all white/black.

**Solution:**

1. Check parent node outputs are correct
2. Verify data range: `print(parent_output.min(), parent_output.max())`
3. Adjust `vmin`/`vmax` in AnomalyHeatmap
4. Check for NaN values: `torch.isnan(parent_output).any()`

### WandB Connection Issues

**Problem:** WandB fails to connect or hangs.

**Solution:**

1. Check internet connection
2. Verify `wandb login` was successful
3. Use offline mode: `mode="offline"`
4. Check WandB status: https://status.wandb.ai

### Slow Visualization Generation

**Problem:** Validation is very slow due to visualizations.

**Solution:**

1. Increase `log_frequency`: Only visualize every N batches
2. Reduce number of visualization leaves
3. Use simpler visualizations (histogram vs heatmap)
4. Disable during training, enable for final validation

## Performance Tips

### Logging Frequency

Balance detail vs performance:

```python
# Development: Visualize everything
log_frequency=1  # Every batch

# Production: Periodic visualization
log_frequency=10  # Every 10th batch

# Final evaluation: Single visualization
log_frequency=999999  # Effectively once
```

### Monitor Selection

Choose monitors based on needs:

- **Development**: DummyMonitor only (fast, no dependencies)
- **Team experiments**: WandB (collaboration, comparison)
- **Local tracking**: TensorBoard (no cloud, fast)
- **Production**: Multiple monitors for redundancy

### Artifact Storage

Manage disk usage:

```python
# Save only PNG (no pickle)
monitor = DummyMonitor(
    output_dir="./outputs/artifacts",
    save_thumbnails=True,
    save_pickle=False,  # Skip pickle artifacts
)

# Periodic cleanup
import shutil
shutil.rmtree("./outputs/artifacts/old_experiments")
```

## Complete Example

See the full working example in `examples_torch/phase2_visualization_training.py`.

Run it with:

```bash
python examples_torch/phase2_visualization_training.py
```

## Next Steps

Now that you understand visualization and monitoring:

- **[Phase 3: Gradient Training](phase3_gradient_training.md)**: Enable gradient-based training with loss and metric leaves
- **[Phase 4: Channel Selection](phase4_channel_selection.md)**: Add soft channel selector
- **[Configuration Guide](../user-guide/configuration.md)**: Master Hydra configuration
- **[API Reference](../api/training.md)**: Explore all visualization and monitoring options

## Key Takeaways

✓ **Leaf nodes generate artifacts** - Visualizations, metrics, logs  
✓ **Parent validation** - Leaf nodes verify compatibility with parent  
✓ **Logging frequency** - Control performance vs detail tradeoff  
✓ **Multiple monitors** - Use DummyMonitor, WandB, TensorBoard simultaneously  
✓ **Artifact persistence** - Save pickle + PNG for offline analysis  
✓ **Configurable via Hydra** - Reproducible experiment configuration  
✓ **No impact on forward pass** - Leaf nodes don't affect inference
