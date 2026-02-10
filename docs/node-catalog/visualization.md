!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Visualization Nodes

## Overview

Visualization nodes create image artifacts for monitoring training progress and model behavior. These are **sink nodes** that:

- Execute during validation, test, and/or inference stages
- Generate matplotlib figures as numpy arrays (Artifact objects)
- Feed into [TensorBoardMonitorNode](#tensorboardmonitornode) for logging
- No gradient computation (visualization only)

**Typical workflow:**
```
Model → Decisions/Scores → Visualization Node → TensorBoardMonitorNode
```

---

## Nodes in This Category

### AnomalyMask

**Description:** Visualize anomaly detection with ground truth comparison and overlay

**Perfect for:**
- Evaluating anomaly detection quality
- Comparing predictions vs ground truth
- Color-coded error analysis (TP/FP/FN)

**Visualization Components:**
1. **Ground Truth Mask** (if available)
2. **Overlay on Cube Image:**
   - Green: True Positives (correct detections)
   - Red: False Positives (false alarms)
   - Yellow: False Negatives (missed anomalies)
3. **Predicted Mask** with metrics

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| decisions | bool | (B,H,W,1) | Binary anomaly decisions | No |
| mask | bool | (B,H,W,1) | Ground truth mask | Yes |
| cube | float32 | (B,H,W,C) | Original cube for background | No |
| scores | float32 | (B,H,W,1) | Optional scores for AP | Yes |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| artifacts | list | () | List of Artifact objects |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| channel | int | required | Cube channel index for visualization |
| up_to | int | None | Max images to visualize (None = all) |

#### Example Usage (Python)

```python
from cuvis_ai.node.visualizations import AnomalyMask

# Create visualizer
viz = AnomalyMask(channel=30, up_to=5)

# Use in pipeline
pipeline.add_nodes(
    viz=viz,
    tb_monitor=TensorBoardMonitorNode(output_dir="./runs")
)
pipeline.connect(
    (decider.decisions, viz.decisions),
    (data.mask, viz.mask),
    (data.cube, viz.cube),
    (logit_head.logits, viz.scores),  # Optional for AP
    (viz.artifacts, tb_monitor.artifacts)
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  viz_mask:
    type: AnomalyMask
    config:
      channel: 30  # Wavelength channel for background
      up_to: 5

  tensorboard:
    type: TensorBoardMonitorNode
    config:
      output_dir: "./runs"

connections:
  - [decider.decisions, viz_mask.decisions]
  - [data.mask, viz_mask.mask]
  - [data.cube, viz_mask.cube]
  - [viz_mask.artifacts, tensorboard.artifacts]
```

#### See Also

- [Tutorial 1: RX Statistical](../tutorials/rx-statistical.md#visualization)
- [TensorBoardMonitorNode](#tensorboardmonitornode)
- API Reference: ::: cuvis_ai.node.visualizations.AnomalyMask

---

### ScoreHeatmapVisualizer

**Description:** Creates heatmaps of anomaly scores with colormap

**Perfect for:**
- Visualizing spatial score distributions
- Monitoring score evolution during training
- Debugging score ranges

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| scores | float32 | (B,H,W,1) | Anomaly scores | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| artifacts | list | () | List of heatmap artifacts |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| normalize_scores | bool | True | Normalize to [0,1] per image |
| cmap | str | "inferno" | Matplotlib colormap |
| up_to | int | 5 | Max heatmaps to generate |

#### Example Usage (Python)

```python
from cuvis_ai.node.visualizations import ScoreHeatmapVisualizer

# Create heatmap visualizer
heatmap_viz = ScoreHeatmapVisualizer(
    normalize_scores=True,
    cmap="inferno",
    up_to=10
)

# Use in pipeline
pipeline.add_nodes(heatmap_viz=heatmap_viz)
pipeline.connect(
    (scorer.scores, heatmap_viz.scores),
    (heatmap_viz.artifacts, tb_monitor.artifacts)
)
```

#### See Also

- [Tutorial 3: Deep SVDD Gradient](../tutorials/deep-svdd-gradient.md#score-visualization)
- API Reference: ::: cuvis_ai.node.visualizations.ScoreHeatmapVisualizer

---

### CubeRGBVisualizer

**Description:** Creates false-color RGB images from hyperspectral cube using channel weights

**Perfect for:**
- Visualizing channel selection results
- Displaying selected wavelengths for RGB channels
- Monitoring selector weight evolution

**Visualization Components:**
1. **False-Color RGB:** Top 3 weighted channels as R, G, B
2. **Weight Plot:** Bar chart showing all channel weights with top 3 highlighted

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| cube | float32 | (B,H,W,C) | Hyperspectral cube | No |
| weights | float32 | (C,) | Channel selection weights | No |
| wavelengths | int32 | (C,) | Wavelengths for each channel | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| artifacts | list | () | List of false-color RGB artifacts |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| up_to | int | 5 | Max images to visualize |

#### Example Usage (Python)

```python
from cuvis_ai.node.visualizations import CubeRGBVisualizer

# Create false-color visualizer
rgb_viz = CubeRGBVisualizer(up_to=5)

# Use in pipeline
pipeline.add_nodes(rgb_viz=rgb_viz)
pipeline.connect(
    (data.cube, rgb_viz.cube),
    (selector.weights, rgb_viz.weights),
    (data.wavelengths, rgb_viz.wavelengths),
    (rgb_viz.artifacts, tb_monitor.artifacts)
)
```

#### See Also

- [Tutorial 2: Channel Selector](../tutorials/channel-selector.md#monitoring)
- [SoftChannelSelector](selectors.md#softchannelselector)
- API Reference: ::: cuvis_ai.node.visualizations.CubeRGBVisualizer

---

### PCAVisualization

**Description:** Visualizes PCA-projected data with scatter plots and spatial color coding

**Perfect for:**
- Monitoring PCA projection quality
- Visualizing first 2 principal components
- Understanding spatial patterns in PC space

**Visualization Components:**
1. **Scatter Plot:** H×W points in 2D PC space (PC1 vs PC2), colored by spatial position
2. **Spatial Reference:** HSV color coding (Hue from x-coord, Saturation from y-coord)
3. **Image Representation:** PC1 in red, PC2 in green channels

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| data | float32 | (B,H,W,K) | PCA-projected data (uses first 2 PCs) | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| artifacts | list | () | List of PCA visualization artifacts |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| up_to | int | None | Max images to visualize (None = all) |

#### Example Usage (Python)

```python
from cuvis_ai.node.visualizations import PCAVisualization

# Create PCA visualizer
pca_viz = PCAVisualization(up_to=10)

# Use in pipeline
pipeline.add_nodes(pca_viz=pca_viz)
pipeline.connect(
    (pca.projected, pca_viz.data),
    (pca_viz.artifacts, tb_monitor.artifacts)
)
```

#### See Also

- [Tutorial 4: AdaCLIP Workflow](../tutorials/adaclip-workflow.md#variant-1-pca-baseline)
- [TrainablePCA](deep-learning.md#trainablepca)
- API Reference: ::: cuvis_ai.node.visualizations.PCAVisualization

---

### RGBAnomalyMask

**Description:** Like AnomalyMask but for RGB images (from band selectors)

**Perfect for:**
- AdaCLIP workflows with RGB-like inputs
- Band selector output visualization
- DRCNN mixer evaluation

**Key Difference:**
- Expects `rgb_image` input (3 channels) instead of `cube` (many channels)
- No channel parameter (uses full RGB for background)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| decisions | bool | (B,H,W,1) | Binary anomaly decisions | No |
| mask | bool | (B,H,W,1) | Ground truth mask | Yes |
| rgb_image | float32 | (B,H,W,3) | RGB image for background | No |
| scores | float32 | (B,H,W,1) | Optional scores for AP | Yes |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| artifacts | list | () | List of RGB anomaly mask artifacts |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| up_to | int | None | Max images to visualize (None = all) |

#### Example Usage (Python)

```python
from cuvis_ai.node.visualizations import RGBAnomalyMask

# Create RGB visualizer
rgb_viz = RGBAnomalyMask(up_to=5)

# Use in pipeline
pipeline.add_nodes(rgb_viz=rgb_viz)
pipeline.connect(
    (decider.decisions, rgb_viz.decisions),
    (data.mask, rgb_viz.mask),
    (mixer.rgb, rgb_viz.rgb_image),  # From DRCNN mixer
    (rgb_viz.artifacts, tb_monitor.artifacts)
)
```

#### See Also

- [Tutorial 4: AdaCLIP Workflow](../tutorials/adaclip-workflow.md#variant-2-drcnn-mixer)
- [LearnableChannelMixer](deep-learning.md#learnablechannelmixer)
- API Reference: ::: cuvis_ai.node.visualizations.RGBAnomalyMask

---

### TensorBoardMonitorNode

**Description:** Sink node logging artifacts and metrics to TensorBoard

**Perfect for:**
- Centralized monitoring of all visualizations
- Real-time training progress tracking
- Comparing runs and experiments

**Key Characteristics:**
- Executes during **all stages** (ALWAYS)
- Accepts multiple artifact/metric inputs (variadic ports)
- Auto-increments run directories (`run_01`, `run_02`, ...)
- Sink node (no outputs)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| artifacts | list | () | List of Artifact objects | Yes |
| metrics | list | () | List of Metric objects | Yes |

**Output Ports:** None (sink node)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| output_dir | str | "./runs" | Directory for TensorBoard logs |
| run_name | str | None | Run name (auto-increment if None) |
| comment | str | "" | Comment appended to log dir |
| flush_secs | int | 120 | Flush interval for disk writes |

#### Example Usage (Python)

```python
from cuvis_ai.node.monitor import TensorBoardMonitorNode

# Create TensorBoard monitor
tb_monitor = TensorBoardMonitorNode(
    output_dir="./runs",
    run_name="rx_experiment_1",  # Or None for auto-increment
    flush_secs=60
)

# Use in pipeline (sink node)
pipeline.add_nodes(tb_monitor=tb_monitor)
pipeline.connect(
    (viz_mask.artifacts, tb_monitor.artifacts),
    (heatmap_viz.artifacts, tb_monitor.artifacts),
    (metrics_node.metrics, tb_monitor.metrics)
)

# View logs
# Run in terminal: tensorboard --logdir=./runs
```

#### Example Configuration (YAML)

```yaml
nodes:
  tensorboard:
    type: TensorBoardMonitorNode
    config:
      output_dir: "./runs"
      run_name: null  # Auto-increment run_01, run_02, ...
      flush_secs: 120

connections:
  - [viz_mask.artifacts, tensorboard.artifacts]
  - [score_viz.artifacts, tensorboard.artifacts]
  - [metrics.metrics, tensorboard.metrics]
```

#### Viewing Logs

```bash
# Start TensorBoard server
tensorboard --logdir=./runs

# Open in browser: http://localhost:6006
```

#### Log Directory Structure

```
runs/
├── run_01/
│   └── events.out.tfevents...
├── run_02/
│   └── events.out.tfevents...
└── my_experiment_v1/
    └── events.out.tfevents...
```

#### See Also

- All tutorials (TensorBoard monitoring used throughout)
- [AnomalyDetectionMetrics](loss-metrics.md#anomalydetectionmetrics)
- API Reference: ::: cuvis_ai.node.monitor.TensorBoardMonitorNode

---

## Visualization Workflow Example

Complete monitoring setup:

```yaml
nodes:
  # Data & Model
  data:
    type: LentilsAnomalyDataNode

  rx_detector:
    type: RXGlobal
    config:
      num_channels: 61

  decider:
    type: BinaryDecider
    config:
      threshold: 0.5

  # Visualization Nodes
  score_heatmap:
    type: ScoreHeatmapVisualizer
    config:
      cmap: "inferno"
      up_to: 5

  anomaly_mask_viz:
    type: AnomalyMask
    config:
      channel: 30
      up_to: 5

  # Metrics
  metrics:
    type: AnomalyDetectionMetrics

  # TensorBoard Monitor (sink)
  tensorboard:
    type: TensorBoardMonitorNode
    config:
      output_dir: "./runs"

connections:
  # Model flow
  - [data.cube, rx_detector.data]
  - [rx_detector.scores, decider.scores]

  # Visualizations
  - [rx_detector.scores, score_heatmap.scores]
  - [decider.decisions, anomaly_mask_viz.decisions]
  - [data.mask, anomaly_mask_viz.mask]
  - [data.cube, anomaly_mask_viz.cube]

  # Metrics
  - [decider.decisions, metrics.decisions]
  - [data.mask, metrics.targets]

  # Logging to TensorBoard
  - [score_heatmap.artifacts, tensorboard.artifacts]
  - [anomaly_mask_viz.artifacts, tensorboard.artifacts]
  - [metrics.metrics, tensorboard.metrics]
```

---

## Best Practices

### Visualization Performance

- **Limit visualization count** with `up_to` parameter to avoid slowdown
  ```python
  viz = AnomalyMask(channel=30, up_to=5)  # Only first 5 images
  ```

- **Execute during val/test only** (default) to avoid training overhead
  ```python
  # Automatically set: execution_stages={ExecutionStage.VAL, ExecutionStage.TEST}
  ```

### TensorBoard Organization

- **Use descriptive run names** for easy comparison
  ```python
  tb = TensorBoardMonitorNode(
      output_dir="./runs",
      run_name="rx_ep50_lr0.001"  # Clear experiment name
  )
  ```

- **Group related experiments** in subdirectories
  ```bash
  runs/
  ├── baseline/
  │   ├── run_01/
  │   └── run_02/
  └── with_selector/
      ├── run_01/
      └── run_02/
  ```

### Artifact Management

- **Close figures** to free memory (done automatically by nodes)
- **Flush frequently** for real-time monitoring: `flush_secs=30`
- **Monitor disk usage** - TensorBoard logs can grow large

---

## Additional Resources

- **Tutorial:** [RX Statistical Detection](../tutorials/rx-statistical.md#visualization-with-tensorboard)
- **Tutorial:** [Channel Selector](../tutorials/channel-selector.md#monitoring-channel-weights)
- **Tutorial:** [Deep SVDD Gradient](../tutorials/deep-svdd-gradient.md#score-visualization)
- **Concepts:** [Execution Stages](../concepts/execution-stages.md)
- **API Reference:** [cuvis_ai.node.visualizations](../../api/node/#visualizations)
- **API Reference:** [cuvis_ai.node.monitor](../../api/node/#monitor)
