!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Monitoring & Visualization

Monitor training progress and visualize results using TensorBoard integration, metrics tracking, and custom visualization nodes.

## Overview

CUVIS.AI provides comprehensive monitoring and visualization capabilities:

- **TensorBoard Integration**: Centralized logging with [TensorBoardMonitorNode](../node-catalog/visualization.md#tensorboardmonitornode)
- **Metrics Tracking**: Performance evaluation with specialized metric nodes
- **Visual Monitoring**: Real-time visualization of predictions, scores, and anomalies
- **Execution Stage Control**: Automatic filtering to minimize overhead

---

## Quick Start

### Basic Monitoring Setup

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer

# Create pipeline
pipeline = CuvisPipeline("monitored_pipeline")

# ... create processing nodes ...

# Create monitoring nodes
metrics = AnomalyDetectionMetrics()
viz_mask = AnomalyMask(channel=30, up_to=5)
score_viz = ScoreHeatmapVisualizer(up_to=5)

monitor = TensorBoardMonitorNode(
    output_dir="./runs",
    run_name="experiment_01"
)

# Connect monitoring
pipeline.connect(
    # Processing flow
    (detector.scores, decider.scores),
    (decider.decisions, metrics.decisions),
    (data.mask, metrics.targets),

    # Visualization flow
    (decider.decisions, viz_mask.decisions),
    (data.mask, viz_mask.mask),
    (data.cube, viz_mask.cube),
    (detector.scores, score_viz.scores),

    # Monitor connections
    (metrics.metrics, monitor.metrics),
    (viz_mask.artifacts, monitor.artifacts),
    (score_viz.artifacts, monitor.artifacts),
)

# Pass monitor to trainer
from cuvis_ai.trainers.gradient_trainer import GradientTrainer

trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[loss],
    metric_nodes=[metrics],
    monitors=[monitor],  # ← Enable monitoring
)

trainer.fit()
```

**View results:**
```bash
tensorboard --logdir=./runs
# Open: http://localhost:6006
```

---

## TensorBoard Integration

### TensorBoardMonitorNode

Centralized logging sink that writes metrics and images to TensorBoard.

**Key characteristics:**
- **Type**: Sink node (no outputs)
- **Execution**: All stages (TRAIN, VAL, TEST, INFERENCE)
- **Auto-increment**: Creates `run_01`, `run_02`, etc. if `run_name` is None

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `"./runs"` | Base directory for TensorBoard logs |
| `run_name` | str \| None | `None` | Specific run name (auto-increments if None) |
| `comment` | str | `""` | Additional comment appended to run name |
| `flush_secs` | int | `120` | Flush events to disk interval (seconds) |

**Input ports:**

| Port | Type | Required | Description |
|------|------|----------|-------------|
| `artifacts` | `list` | No | List of `Artifact` objects (images) |
| `metrics` | `list` | No | List of `Metric` objects (scalars) |

### Example: Custom Run Organization

```python
# Organized directory structure
monitor = TensorBoardMonitorNode(
    output_dir="./outputs/my_experiment/tensorboard",
    run_name=f"lr_{learning_rate}_bs_{batch_size}",
    comment="with_augmentation",
)

# Results in: ./outputs/my_experiment/tensorboard/lr_0.001_bs_2_with_augmentation/
```

### Direct Logging API

```python
# Log custom scalars programmatically
monitor.log("train/custom_metric", 0.75, step=100)
monitor.log("val/threshold", 0.5, step=100)
```

**Used internally by GradientTrainer** for training/validation loss logging.

---

## Metrics Tracking

### Available Metric Nodes

#### AnomalyDetectionMetrics

Comprehensive anomaly detection performance metrics.

**Metrics computed:**
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **IoU (Jaccard Index)**: TP / (TP + FP + FN)
- **Average Precision (AP)**: Area under precision-recall curve (if `logits` provided)

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `decisions` | `bool` | `(B,H,W,1)` | Binary predictions |
| `targets` | `bool` | `(B,H,W,1)` | Ground truth masks |
| `logits` | `float32` | `(B,H,W,1)` | Optional anomaly scores for AP |

**Execution**: VAL, TEST stages only

**Example:**

```python
from cuvis_ai.node.metrics import AnomalyDetectionMetrics

metrics = AnomalyDetectionMetrics()

# Connect
pipeline.connect(
    (decider.decisions, metrics.decisions),
    (data_node.outputs.mask, metrics.targets),
    (detector.scores, metrics.logits),  # Optional for AP
    (metrics.metrics, monitor.metrics),
)
```

**TensorBoard tags:**
```
val/precision
val/recall
val/f1_score
val/iou
val/average_precision
```

#### ScoreStatisticsMetric

Track distribution properties of anomaly scores.

**Metrics computed:**
- Mean, standard deviation
- Min, max, median
- Quantiles: q25, q75, q95, q99

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `scores` | `float32` | `(B,H,W,1)` | Anomaly scores |

**Example:**

```python
from cuvis_ai.node.metrics import ScoreStatisticsMetric

score_stats = ScoreStatisticsMetric()

pipeline.connect(
    (detector.scores, score_stats.scores),
    (score_stats.metrics, monitor.metrics),
)
```

**TensorBoard tags:**
```
scores/mean
scores/std
scores/min
scores/max
scores/median
scores/q25
scores/q75
scores/q95
scores/q99
```

#### ExplainedVarianceMetric

Monitor PCA variance breakdown.

**Metrics computed:**
- Per-component variance
- Total explained variance
- Cumulative variance ratios

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `explained_variance` | `float32` | `(num_components,)` | Variance per PC |

**Example:**

```python
from cuvis_ai.node.metrics import ExplainedVarianceMetric

variance_metric = ExplainedVarianceMetric()

pipeline.connect(
    (pca_node.explained_variance, variance_metric.explained_variance),
    (variance_metric.metrics, monitor.metrics),
)
```

**TensorBoard tags:**
```
explained_variance_pc1
explained_variance_pc2
...
total_explained_variance
cumulative_variance_pc1
cumulative_variance_pc2
```

#### SelectorEntropyMetric & SelectorDiversityMetric

Monitor channel selection diversity.

**Metrics computed:**
- **Entropy**: Shannon entropy of selection weights
- **Weight Variance**: Variance of selection weights
- **Gini Coefficient**: Inequality measure (0 = uniform, 1 = concentrated)

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `weights` | `float32` | `(C,)` | Channel selection weights |

**Example:**

```python
from cuvis_ai.node.metrics import SelectorEntropyMetric, SelectorDiversityMetric

entropy = SelectorEntropyMetric()
diversity = SelectorDiversityMetric()

pipeline.connect(
    (selector.weights, entropy.weights),
    (selector.weights, diversity.weights),
    (entropy.metrics, monitor.metrics),
    (diversity.metrics, monitor.metrics),
)
```

**TensorBoard tags:**
```
selector/entropy
weight_variance
gini_coefficient
```

#### ComponentOrthogonalityMetric

Monitor PCA component orthonormality.

**Metrics computed:**
- **Orthogonality Error**: Frobenius norm of `(G - I)`
- **Avg Off-Diagonal**: Mean absolute off-diagonal value
- **Diagonal Mean/Std**: Statistics of diagonal entries

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `components` | `float32` | `(num_components, features)` | PCA components |

**Example:**

```python
from cuvis_ai.node.metrics import ComponentOrthogonalityMetric

orthogonality = ComponentOrthogonalityMetric()

pipeline.connect(
    (pca_node.components, orthogonality.components),
    (orthogonality.metrics, monitor.metrics),
)
```

**TensorBoard tags:**
```
orthogonality_error
avg_off_diagonal
diagonal_mean
diagonal_std
```

---

## Visualization Nodes

### AnomalyMask

Side-by-side ground truth and prediction comparison with overlay.

**Visualization output:**
1. **Ground Truth Mask** (if available)
2. **Predicted Overlay** on selected cube channel:
   - Green: True Positives (TP)
   - Red: False Positives (FP)
   - Yellow: False Negatives (FN)
3. **Predicted Mask** with metrics (Precision, Recall, F1, IoU, AP)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `channel` | int | `30` | Cube channel index for background |
| `up_to` | int | `5` | Max images per batch to visualize |

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `decisions` | `bool` | `(B,H,W,1)` | Binary predictions |
| `mask` | `bool` | `(B,H,W,1)` | Ground truth (optional) |
| `cube` | `float32` | `(B,H,W,C)` | Hyperspectral cube |
| `scores` | `float32` | `(B,H,W,1)` | Scores for AP (optional) |

**Execution**: VAL, TEST, INFERENCE stages

**Example:**

```python
from cuvis_ai.node.visualizations import AnomalyMask

viz_mask = AnomalyMask(channel=30, up_to=5)

pipeline.connect(
    (decider.decisions, viz_mask.decisions),
    (data_node.outputs.mask, viz_mask.mask),
    (data_node.outputs.cube, viz_mask.cube),
    (detector.scores, viz_mask.scores),
    (viz_mask.artifacts, monitor.artifacts),
)
```

**TensorBoard images:**
```
val/anomaly_mask_img00
val/anomaly_mask_img01
...
```

### RGBAnomalyMask

Like [AnomalyMask](#anomalymask) but for RGB images.

**Use cases:**
- Band selector output visualization
- DRCNN mixer evaluation
- AdaCLIP RGB-like workflows

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `decisions` | `bool` | `(B,H,W,1)` | Binary predictions |
| `mask` | `bool` | `(B,H,W,1)` | Ground truth (optional) |
| `rgb_image` | `float32` | `(B,H,W,3)` | RGB background image |
| `scores` | `float32` | `(B,H,W,1)` | Scores for AP (optional) |

**Example:**

```python
from cuvis_ai.node.visualizations import RGBAnomalyMask

rgb_viz = RGBAnomalyMask(up_to=5)

pipeline.connect(
    (decider.decisions, rgb_viz.decisions),
    (data_node.outputs.mask, rgb_viz.mask),
    (mixer.output, rgb_viz.rgb_image),  # RGB from mixer
    (rgb_viz.artifacts, monitor.artifacts),
)
```

### ScoreHeatmapVisualizer

Heatmap visualization of anomaly scores.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize_scores` | bool | `True` | Normalize scores to [0,1] |
| `cmap` | str | `"inferno"` | Matplotlib colormap |
| `up_to` | int | `5` | Max heatmaps per batch |

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `scores` | `float32` | `(B,H,W,1)` | Anomaly scores |

**Example:**

```python
from cuvis_ai.node.visualizations import ScoreHeatmapVisualizer

score_viz = ScoreHeatmapVisualizer(
    normalize_scores=True,
    cmap="inferno",
    up_to=5,
)

pipeline.connect(
    (detector.scores, score_viz.scores),
    (score_viz.artifacts, monitor.artifacts),
)
```

**TensorBoard images:**
```
val/score_heatmap_img00
val/score_heatmap_img01
...
```

### PCAVisualization

Visualize first 2 PCA components with spatial encoding.

**Visualization output:**
1. **Scatter Plot**: PC1 vs PC2, colored by spatial position (HSV encoding)
2. **Spatial Reference**: HSV color coding guide
3. **Image Representation**: PC1 in red channel, PC2 in green
4. **Statistics Box**: Range, shape, point count

**HSV Encoding:**
- **Hue**: x-coordinate (left → right)
- **Saturation**: y-coordinate (top → bottom)
- **Value**: constant (brightness)

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `projections` | `float32` | `(B,H,W,num_components)` | PCA projections |

**Example:**

```python
from cuvis_ai.node.visualizations import PCAVisualization

pca_viz = PCAVisualization(up_to=3)

pipeline.connect(
    (pca_node.projections, pca_viz.projections),
    (pca_viz.artifacts, monitor.artifacts),
)
```

**TensorBoard images:**
```
val/pca_projection_img00
val/pca_projection_img01
```

### CubeRGBVisualizer

False-color RGB from selected hyperspectral channels.

**Visualization output:**
1. **False-color RGB**: Top 3 weighted channels mapped to R, G, B
2. **Channel Weights Bar Chart**: All channels with top 3 highlighted

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `cube` | `float32` | `(B,H,W,C)` | Hyperspectral cube |
| `weights` | `float32` | `(C,)` | Channel selection weights |
| `wavelengths` | `int32` | `(C,)` | Wavelengths per channel |

**Example:**

```python
from cuvis_ai.node.visualizations import CubeRGBVisualizer

rgb_viz = CubeRGBVisualizer(up_to=5)

pipeline.connect(
    (data_node.outputs.cube, rgb_viz.cube),
    (selector.weights, rgb_viz.weights),
    (data_node.outputs.wavelengths, rgb_viz.wavelengths),
    (rgb_viz.artifacts, monitor.artifacts),
)
```

**TensorBoard images:**
```
val/viz_rgb_sample_0
val/viz_rgb_sample_1
```

### DRCNNTensorBoardViz

Specialized visualizations for DRCNN pipelines.

**Visualization output:**
1. **HSI Input**: False-color RGB from selected channels
2. **Mixer Output**: AdaClip RGB input
3. **Ground Truth Mask**
4. **AdaClip Scores**: Anomaly heatmap

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hsi_channels` | list[int] | `[0, 20, 40]` | Channels for false-color RGB |
| `max_samples` | int | `4` | Max samples per batch |
| `log_every_n_batches` | int | `1` | Logging frequency |

**Input ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `hsi_cube` | `float32` | `(B,H,W,C)` | Input hyperspectral cube |
| `mixer_output` | `float32` | `(B,H,W,3)` | Mixed RGB output |
| `ground_truth_mask` | `bool` | `(B,H,W,1)` | Ground truth mask |
| `adaclip_scores` | `float32` | `(B,H,W,1)` | AdaClip anomaly scores |

**Execution**: TRAIN, VAL, TEST stages

**Example:**

```python
from cuvis_ai.node.drcnn_tensorboard_viz import DRCNNTensorBoardViz

drcnn_viz = DRCNNTensorBoardViz(
    hsi_channels=[0, 20, 40],
    max_samples=4,
    log_every_n_batches=1,
)

pipeline.connect(
    (data_node.outputs.cube, drcnn_viz.hsi_cube),
    (mixer.output, drcnn_viz.mixer_output),
    (data_node.outputs.mask, drcnn_viz.ground_truth_mask),
    (adaclip.scores, drcnn_viz.adaclip_scores),
    (drcnn_viz.artifacts, monitor.artifacts),
)
```

**TensorBoard images:**
```
train/hsi_input_sample_0
train/mixer_output_adaclip_input_sample_0
train/ground_truth_mask_sample_0
train/adaclip_scores_heatmap_sample_0
```

---

## Complete Examples

### Example 1: RX Statistical Training

Simple monitoring for statistical initialization.

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.node.conversion import ScoreToLogit
from cuvis_ai.anomaly.binary_decider import BinaryDecider
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.node.monitor import TensorBoardMonitorNode

# Create pipeline
pipeline = CuvisPipeline("RX_Statistical")

# Processing nodes
data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
rx = RXGlobal(num_channels=61, eps=1.0e-6)
logit_head = ScoreToLogit(init_scale=1.0, init_bias=0.0)
decider = BinaryDecider(threshold=0.5)

# Monitoring nodes
metrics = AnomalyDetectionMetrics()
viz_mask = AnomalyMask(channel=30, up_to=5)
score_viz = ScoreHeatmapVisualizer(up_to=5)
monitor = TensorBoardMonitorNode(
    output_dir="./outputs/rx_statistical/tensorboard",
    run_name="rx_baseline",
)

# Connect processing
pipeline.connect(
    (data_node.outputs.cube, normalizer.data),
    (normalizer.normalized, rx.data),
    (rx.scores, logit_head.scores),
    (logit_head.logits, decider.logits),
)

# Connect monitoring
pipeline.connect(
    # Metrics
    (decider.decisions, metrics.decisions),
    (data_node.outputs.mask, metrics.targets),
    (rx.scores, metrics.logits),

    # Visualizations
    (decider.decisions, viz_mask.decisions),
    (data_node.outputs.mask, viz_mask.mask),
    (data_node.outputs.cube, viz_mask.cube),
    (rx.scores, viz_mask.scores),
    (rx.scores, score_viz.scores),

    # Monitor sink
    (metrics.metrics, monitor.metrics),
    (viz_mask.artifacts, monitor.artifacts),
    (score_viz.artifacts, monitor.artifacts),
)

# Validate
pipeline.validate()

# Statistical training (no gradients)
from cuvis_ai.trainers.statistical_trainer import StatisticalTrainer
from cuvis_ai.datamodule.cu3s_datamodule import Cu3sDataModule

datamodule = Cu3sDataModule(
    cu3s_file_path="data/Lentils/Lentils_000.cu3s",
    train_ids=[0, 2, 3],
    val_ids=[1, 5],
    test_ids=[1, 5],
    batch_size=1,
)

trainer = StatisticalTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    metric_nodes=[metrics],
    monitors=[monitor],
)

trainer.fit()
trainer.test()
```

**View results:**
```bash
tensorboard --logdir=./outputs/rx_statistical/tensorboard
```

### Example 2: DRCNN + AdaClip Gradient Training

Comprehensive monitoring with multiple visualizations and losses.

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.anomaly.learnable_channel_mixer import LearnableChannelMixer
from cuvis_ai.anomaly.adaclip_anomaly_detector import AdaClipAnomalyDetector
from cuvis_ai.anomaly.binary_decider import BinaryDecider
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.node.drcnn_tensorboard_viz import DRCNNTensorBoardViz
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.anomaly.iou_loss import IoULoss

# Create pipeline
pipeline = CuvisPipeline("DRCNN_AdaClip_Gradient")

# Processing nodes
data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
channel_mixer = LearnableChannelMixer(num_channels=61)
adaclip = AdaClipAnomalyDetector()
decider = BinaryDecider(threshold=0.5)
iou_loss = IoULoss(weight=1.0, normalize_method="minmax")

# Monitoring nodes
metrics = AnomalyDetectionMetrics()
viz_mask = AnomalyMask(channel=30, up_to=5)
score_viz = ScoreHeatmapVisualizer(up_to=5)
drcnn_viz = DRCNNTensorBoardViz(
    hsi_channels=[0, 20, 40],
    max_samples=4,
    log_every_n_batches=1,
)
monitor = TensorBoardMonitorNode(
    output_dir="./outputs/drcnn_adaclip/tensorboard",
    run_name="drcnn_baseline",
)

# Connect processing
pipeline.connect(
    (data_node.outputs.cube, normalizer.data),
    (normalizer.normalized, channel_mixer.data),
    (channel_mixer.output, adaclip.image),
    (adaclip.scores, decider.scores),

    # Loss
    (decider.decisions, iou_loss.predictions),
    (data_node.outputs.mask, iou_loss.targets),
)

# Connect monitoring
pipeline.connect(
    # Metrics
    (decider.decisions, metrics.decisions),
    (data_node.outputs.mask, metrics.targets),
    (adaclip.scores, metrics.logits),

    # Visualizations
    (decider.decisions, viz_mask.decisions),
    (data_node.outputs.mask, viz_mask.mask),
    (normalizer.normalized, viz_mask.cube),
    (adaclip.scores, viz_mask.scores),
    (adaclip.scores, score_viz.scores),

    # DRCNN-specific visualization
    (data_node.outputs.cube, drcnn_viz.hsi_cube),
    (channel_mixer.output, drcnn_viz.mixer_output),
    (data_node.outputs.mask, drcnn_viz.ground_truth_mask),
    (adaclip.scores, drcnn_viz.adaclip_scores),

    # Monitor sink
    (metrics.metrics, monitor.metrics),
    (viz_mask.artifacts, monitor.artifacts),
    (score_viz.artifacts, monitor.artifacts),
    (drcnn_viz.artifacts, monitor.artifacts),
)

# Validate
pipeline.validate()

# Gradient training
from cuvis_ai.trainers.gradient_trainer import GradientTrainer
from cuvis_ai.datamodule.cu3s_datamodule import Cu3sDataModule

datamodule = Cu3sDataModule(
    cu3s_file_path="data/Lentils/Lentils_000.cu3s",
    train_ids=[0],
    val_ids=[3, 4],
    test_ids=[1, 5],
    batch_size=1,
)

trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[iou_loss],
    metric_nodes=[metrics],
    trainer_config={
        "max_epochs": 20,
        "log_every_n_steps": 10,
        "val_check_interval": 1.0,
    },
    optimizer_config={
        "name": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
    },
    monitors=[monitor],
)

trainer.fit()
trainer.test()
```

**View results:**
```bash
tensorboard --logdir=./outputs/drcnn_adaclip/tensorboard
```

### Example 3: Multi-Loss Training with Concrete Band Selector

Monitor multiple loss components and selector diversity.

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.anomaly.concrete_band_selector import ConcreteBandSelector
from cuvis_ai.anomaly.adaclip_anomaly_detector import AdaClipAnomalyDetector
from cuvis_ai.anomaly.binary_decider import BinaryDecider
from cuvis_ai.node.metrics import AnomalyDetectionMetrics, SelectorEntropyMetric
from cuvis_ai.node.visualizations import AnomalyMask, CubeRGBVisualizer
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.anomaly.iou_loss import IoULoss
from cuvis_ai.anomaly.distinctness_loss import DistinctnessLoss

# Create pipeline
pipeline = CuvisPipeline("Concrete_AdaClip")

# Processing nodes
data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
selector = ConcreteBandSelector(
    num_channels=61,
    num_bands_to_select=3,
    temperature=0.5,
)
adaclip = AdaClipAnomalyDetector()
decider = BinaryDecider(threshold=0.5)

# Loss nodes
iou_loss = IoULoss(weight=1.0, normalize_method="minmax")
distinctness_loss = DistinctnessLoss(weight=0.1)

# Monitoring nodes
metrics = AnomalyDetectionMetrics()
selector_entropy = SelectorEntropyMetric()
viz_mask = AnomalyMask(channel=30, up_to=5)
rgb_viz = CubeRGBVisualizer(up_to=5)
monitor = TensorBoardMonitorNode(
    output_dir="./outputs/concrete_adaclip/tensorboard",
    run_name="concrete_baseline",
)

# Connect processing
pipeline.connect(
    (data_node.outputs.cube, normalizer.data),
    (normalizer.normalized, selector.data),
    (selector.selected_bands, adaclip.image),
    (adaclip.scores, decider.scores),

    # Losses
    (decider.decisions, iou_loss.predictions),
    (data_node.outputs.mask, iou_loss.targets),
    (selector.weights, distinctness_loss.weights),
)

# Connect monitoring
pipeline.connect(
    # Metrics
    (decider.decisions, metrics.decisions),
    (data_node.outputs.mask, metrics.targets),
    (adaclip.scores, metrics.logits),
    (selector.weights, selector_entropy.weights),

    # Visualizations
    (decider.decisions, viz_mask.decisions),
    (data_node.outputs.mask, viz_mask.mask),
    (normalizer.normalized, viz_mask.cube),
    (normalizer.normalized, rgb_viz.cube),
    (selector.weights, rgb_viz.weights),
    (data_node.outputs.wavelengths, rgb_viz.wavelengths),

    # Monitor sink
    (metrics.metrics, monitor.metrics),
    (selector_entropy.metrics, monitor.metrics),
    (viz_mask.artifacts, monitor.artifacts),
    (rgb_viz.artifacts, monitor.artifacts),
)

# Validate
pipeline.validate()

# Gradient training with multiple losses
from cuvis_ai.trainers.gradient_trainer import GradientTrainer
from cuvis_ai.datamodule.cu3s_datamodule import Cu3sDataModule

datamodule = Cu3sDataModule(
    cu3s_file_path="data/Lentils/Lentils_000.cu3s",
    train_ids=[0],
    val_ids=[3, 4],
    test_ids=[1, 5],
    batch_size=1,
)

trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[iou_loss, distinctness_loss],  # Multiple losses
    metric_nodes=[metrics],
    trainer_config={
        "max_epochs": 20,
        "log_every_n_steps": 10,
    },
    optimizer_config={
        "name": "adamw",
        "lr": 0.001,
    },
    monitors=[monitor],
)

trainer.fit()
```

**Monitor selector entropy and loss components** in TensorBoard to ensure balanced learning.

---

## Configuration Patterns

### TrainRun YAML Configuration

```yaml
# @package _global_

name: drcnn_adaclip
output_dir: ./outputs/${name}

defaults:
  - /pipeline@pipeline: drcnn_adaclip
  - /data@data: lentils
  - /training@training: default
  - _self_

training:
  seed: 42
  trainer:
    max_epochs: 20
    accelerator: auto
    devices: 1
    log_every_n_steps: 10        # Log metrics every 10 steps
    val_check_interval: 1.0      # Validate every epoch
    enable_checkpointing: true

  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.01

  scheduler:
    name: reduce_on_plateau
    monitor: metrics_anomaly/iou  # Monitor IoU for scheduler
    mode: max
    factor: 0.5
    patience: 5

loss_nodes:
  - iou_loss

metric_nodes:
  - metrics_anomaly

unfreeze_nodes:
  - channel_mixer
```

**TensorBoard logging:**
- Logs are written to `./outputs/drcnn_adaclip/tensorboard/`
- `log_every_n_steps: 10` logs metrics every 10 training steps
- `val_check_interval: 1.0` runs validation (and visualizations) every epoch

### Output Directory Structure

```
outputs/drcnn_adaclip/
├── tensorboard/
│   ├── drcnn_baseline/
│   │   └── events.out.tfevents.TIMESTAMP.hostname
│   ├── drcnn_baseline_run_02/
│   └── drcnn_baseline_run_03/
├── checkpoints/
│   ├── epoch=00.ckpt
│   ├── epoch=01.ckpt
│   └── last.ckpt
├── pipeline/
│   ├── DRCNN_AdaClip_Gradient.png
│   └── DRCNN_AdaClip_Gradient.md
└── trained_models/
    ├── DRCNN_AdaClip_Gradient.yaml
    ├── DRCNN_AdaClip_Gradient.pt
    └── drcnn_adaclip_trainrun.yaml
```

---

## Best Practices

### 1. Visualization Performance

**Limit visualizations to reduce overhead:**

```python
# Good: Limit images per batch
viz_mask = AnomalyMask(channel=30, up_to=5)  # Max 5 images
score_viz = ScoreHeatmapVisualizer(up_to=5)

# Good: Log less frequently for large datasets
drcnn_viz = DRCNNTensorBoardViz(log_every_n_batches=5)  # Every 5th batch
```

**Avoid:**
```python
# Bad: Visualize all images (slow, large disk usage)
viz_mask = AnomalyMask(up_to=1000)
```

**Execution stages:**
- Visualization nodes run during **VAL, TEST, INFERENCE** by default
- This avoids training slowdown
- TensorBoardMonitorNode runs during **all stages** to accept any inputs

### 2. TensorBoard Organization

**Use descriptive run names:**

```python
# Good: Clear experiment identification
monitor = TensorBoardMonitorNode(
    output_dir="./outputs/my_experiment/tensorboard",
    run_name=f"lr_{lr}_bs_{batch_size}_aug",
)
```

**Group related experiments:**

```bash
# Directory structure for hyperparameter sweeps
outputs/
└── channel_selector/
    └── tensorboard/
        ├── lr_0.001_bs_1/
        ├── lr_0.0001_bs_1/
        ├── lr_0.001_bs_2/
        └── lr_0.0001_bs_2/

# View all runs together
tensorboard --logdir=outputs/channel_selector/tensorboard
```

**Monitor disk usage:**
- TensorBoard event files can grow large (100s of MB)
- Set `flush_secs` appropriately (default: 120 seconds)
- Clean old runs periodically

### 3. Monitoring Strategy

**Core metrics always:**

```python
# Always include basic performance metrics
metrics = AnomalyDetectionMetrics()
pipeline.connect(
    (decider.decisions, metrics.decisions),
    (data_node.outputs.mask, metrics.targets),
    (metrics.metrics, monitor.metrics),
)
```

**Add visualizations for validation:**

```python
# Visualizations for VAL/TEST only (no training overhead)
viz_mask = AnomalyMask(channel=30, up_to=5)
score_viz = ScoreHeatmapVisualizer(up_to=5)
```

**Specialized metrics for debugging:**

```python
# Add selector/PCA metrics when debugging specific components
selector_entropy = SelectorEntropyMetric()
orthogonality = ComponentOrthogonalityMetric()
score_stats = ScoreStatisticsMetric()
```

### 4. Metric Selection for Scheduler

**Monitor appropriate metrics:**

```yaml
# Good: Monitor IoU for anomaly detection
scheduler:
  name: reduce_on_plateau
  monitor: metrics_anomaly/iou
  mode: max

# Good: Monitor loss for early stopping
callbacks:
  early_stopping:
    monitor: val/loss
    mode: min
```

**Verify metric names:**
- Check TensorBoard to see exact metric names
- Metric tags: `val/precision`, `val/iou`, `scores/mean`, etc.
- Loss tags: `train/loss`, `val/loss`

### 5. Checkpoint Strategy

**Enable checkpointing with metric monitoring:**

```yaml
training:
  trainer:
    enable_checkpointing: true
    callbacks:
      model_checkpoint:
        dirpath: outputs/${name}/checkpoints
        monitor: metrics_anomaly/iou
        mode: max
        save_top_k: 3
        save_last: true
```

**Benefits:**
- Save best models based on validation metrics
- Resume training from checkpoints
- Compare multiple checkpoints in TensorBoard

### 6. Logging Intervals

**Balance between detail and performance:**

```yaml
training:
  trainer:
    log_every_n_steps: 10      # Log metrics every 10 steps
    val_check_interval: 1.0    # Validate every epoch (1.0 = 100%)
```

**For large datasets:**
```yaml
training:
  trainer:
    log_every_n_steps: 50      # Less frequent logging
    val_check_interval: 0.5    # Validate every half-epoch
```

**For small datasets:**
```yaml
training:
  trainer:
    log_every_n_steps: 1       # Log every step
    val_check_interval: 1.0    # Validate every epoch
```

---

## Troubleshooting

### TensorBoard Not Showing Images

**Problem**: Metrics appear but images are missing.

**Solution**: Verify artifact connections:

```python
# Ensure artifacts are connected to monitor
pipeline.connect(
    (viz_mask.artifacts, monitor.artifacts),
    (score_viz.artifacts, monitor.artifacts),
)

# Check execution stages
print(viz_mask.execution_stages)  # Should include VAL/TEST
```

### Metrics Not Logging

**Problem**: No metrics in TensorBoard.

**Solution**: Check metric node connections:

```python
# Connect metrics to monitor
pipeline.connect(
    (metrics.metrics, monitor.metrics),
)

# Verify metric nodes are passed to trainer
trainer = GradientTrainer(
    metric_nodes=[metrics],  # ← Must specify
    monitors=[monitor],
)
```

### Visualizations Slow Training

**Problem**: Training is slow with visualizations enabled.

**Solution**: Limit visualization frequency:

```python
# Reduce images per batch
viz_mask = AnomalyMask(up_to=3)  # Instead of up_to=10

# Log less frequently
drcnn_viz = DRCNNTensorBoardViz(log_every_n_batches=10)  # Every 10th batch
```

**Verify execution stages:**
```python
# Visualizations should NOT run during training
assert ExecutionStage.TRAIN not in viz_mask.execution_stages
```

### Scheduler Not Responding

**Problem**: Learning rate scheduler doesn't reduce loss.

**Solution**: Verify monitored metric name:

```yaml
# Check exact metric name in TensorBoard
scheduler:
  name: reduce_on_plateau
  monitor: metrics_anomaly/iou  # Must match TensorBoard tag exactly
  mode: max
```

**Debug:**
```python
# Print available metrics
for metric in metrics.metrics:
    print(f"{metric.name}: {metric.value}")
```

### Disk Space Issues

**Problem**: TensorBoard logs consuming too much disk space.

**Solution**: Clean old runs and adjust logging:

```bash
# Remove old runs
rm -rf outputs/*/tensorboard/old_run_*

# Or compress old runs
tar -czf old_runs.tar.gz outputs/*/tensorboard/run_01/
rm -rf outputs/*/tensorboard/run_01/
```

**Reduce logging frequency:**
```python
monitor = TensorBoardMonitorNode(
    flush_secs=300,  # Flush less frequently (5 minutes)
)

viz = DRCNNTensorBoardViz(
    log_every_n_batches=10,  # Log less often
    max_samples=2,  # Fewer images
)
```

---

## See Also

- **Node Catalog**:
  - [Visualization Nodes](../node-catalog/visualization.md) - TensorBoardMonitorNode and visual monitoring
  - [Metrics Nodes](../node-catalog/loss-metrics.md#metrics-nodes) - AnomalyDetectionMetrics, ScoreStatistics, etc.
  - [Visualization Nodes](../node-catalog/visualization.md) - All visualization node details
- **Guides**:
  - [Build Pipelines in Python](build-pipeline-python.md) - Pipeline construction basics
  - [Configuration Guide](../user-guide/configuration.md) - TrainRun configuration
- **Examples**:
  - `examples/adaclip/drcnn_adaclip_gradient_training.py` - Full DRCNN monitoring example
  - `examples/adaclip/concrete_adaclip_gradient_training.py` - Multi-loss monitoring example
