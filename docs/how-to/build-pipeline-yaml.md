!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# How-To: Build Pipelines in YAML

## Overview
Learn how to define pipelines using YAML configuration files. YAML pipelines enable version control, reproducibility, and easy experimentation through Hydra composition.

## Prerequisites
- cuvis-ai installed
- Basic understanding of [Pipeline Lifecycle](../concepts/pipeline-lifecycle.md)
- Familiarity with [YAML syntax](https://yaml.org/)
- Optional: Understanding of [Hydra composition](../config/hydra-composition.md)

## Pipeline YAML Structure

A pipeline YAML file has three main sections:

```yaml
metadata:
  name: My_Pipeline
  description: Pipeline description
  tags: [tag1, tag2]
  author: your_name

nodes:
  - name: node1
    class: cuvis_ai.module.NodeClass
    params:
      param1: value1
      param2: value2

connections:
  - from: node1.outputs.output_port
    to: node2.inputs.input_port
```

## Basic Pipeline Example

Here's a simple RX anomaly detection pipeline:

```yaml
metadata:
  name: RX_Statistical
  description: RX anomaly detector with statistical training
  tags:
    - statistical
    - rx
  author: cuvis.ai

nodes:
  - name: LentilsAnomalyDataNode
    class: cuvis_ai.node.data.LentilsAnomalyDataNode
    params:
      normal_class_ids: [0, 1]

  - name: MinMaxNormalizer
    class: cuvis_ai.node.normalization.MinMaxNormalizer
    params:
      eps: 1.0e-06
      use_running_stats: true

  - name: RXGlobal
    class: cuvis_ai.anomaly.rx_detector.RXGlobal
    params:
      num_channels: 61
      eps: 1.0e-06

  - name: BinaryDecider
    class: cuvis_ai.deciders.binary_decider.BinaryDecider
    params:
      threshold: 0.5

  - name: metrics
    class: cuvis_ai.node.metrics.AnomalyDetectionMetrics
    params: {}

connections:
  - from: LentilsAnomalyDataNode.outputs.cube
    to: MinMaxNormalizer.inputs.data

  - from: MinMaxNormalizer.outputs.normalized
    to: RXGlobal.inputs.data

  - from: RXGlobal.outputs.scores
    to: BinaryDecider.inputs.logits

  - from: BinaryDecider.outputs.decisions
    to: metrics.inputs.decisions

  - from: LentilsAnomalyDataNode.outputs.mask
    to: metrics.inputs.targets
```

## Multi-Branch Pipeline Example

Complex pipelines with multiple branches (channel selector with losses and metrics):

```yaml
metadata:
  name: Channel_Selector
  description: Channel selection with gradient training
  tags:
    - gradient
    - channel_selector
  author: cuvis.ai

nodes:
  # Data loading
  - name: data_node
    class: cuvis_ai.node.data.LentilsAnomalyDataNode
    params:
      normal_class_ids: [0, 1]

  # Preprocessing
  - name: normalizer
    class: cuvis_ai.node.normalization.MinMaxNormalizer
    params:
      eps: 1.0e-06
      use_running_stats: true

  # Channel selection
  - name: selector
    class: cuvis_ai.node.channel_selector.SoftChannelSelector
    params:
      n_select: 3
      input_channels: 61
      init_method: variance
      temperature_init: 5.0

  # Anomaly detection
  - name: rx
    class: cuvis_ai.anomaly.rx_detector.RXGlobal
    params:
      num_channels: 61
      eps: 1.0e-06

  # Losses
  - name: bce_loss
    class: cuvis_ai.node.losses.AnomalyBCEWithLogits
    params:
      weight: 10.0
      pos_weight: null

  - name: entropy_loss
    class: cuvis_ai.node.losses.SelectorEntropyRegularizer
    params:
      weight: 0.1

  # Metrics
  - name: metrics
    class: cuvis_ai.node.metrics.AnomalyDetectionMetrics
    params: {}

  # Monitoring
  - name: tensorboard
    class: cuvis_ai.node.monitor.TensorBoardMonitorNode
    params:
      output_dir: logs/
      run_name: channel_selector

connections:
  # Data → Preprocessing
  - from: data_node.outputs.cube
    to: normalizer.inputs.data

  # Preprocessing → Selection → Detection
  - from: normalizer.outputs.normalized
    to: selector.inputs.data
  - from: selector.outputs.selected
    to: rx.inputs.data

  # Selection weights → Regularization
  - from: selector.outputs.weights
    to: entropy_loss.inputs.weights

  # RX → Loss
  - from: rx.outputs.scores
    to: bce_loss.inputs.predictions
  - from: data_node.outputs.mask
    to: bce_loss.inputs.targets

  # RX → Metrics
  - from: rx.outputs.scores
    to: metrics.inputs.decisions
  - from: data_node.outputs.mask
    to: metrics.inputs.targets

  # Metrics → Monitoring
  - from: metrics.outputs.metrics
    to: tensorboard.inputs.metrics
```

## Loading YAML Pipelines in Python

### Basic Loading

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

# Load pipeline from YAML
pipeline = CuvisPipeline.load_pipeline("configs/pipeline/rx_statistical.yaml")

# Validate
pipeline.validate()

# Use with trainer
from cuvis_ai_core.training import StatisticalTrainer
from cuvis_ai_core.data.datasets import SingleCu3sDataModule

datamodule = SingleCu3sDataModule(
    cu3s_file_path="data/train.cu3s",
    batch_size=1
)
datamodule.setup(stage="fit")

trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()
```

### Loading with Overrides

Override specific parameters without modifying the YAML file:

```python
# Override node parameters
pipeline = CuvisPipeline.load_pipeline(
    config_path="configs/pipeline/rx_statistical.yaml",
    config_overrides={
        "nodes.2.params.threshold": 0.8,  # Override BinaryDecider threshold
        "nodes.3.params.eps": 1e-8,       # Override RXGlobal eps
    }
)

# Or using dot notation (requires Hydra)
pipeline = CuvisPipeline.load_pipeline(
    config_path="configs/pipeline/rx_statistical.yaml",
    config_overrides=[
        "nodes.2.params.threshold=0.8",
        "metadata.name=RX_Custom"
    ]
)
```

### Loading with Custom Weights

```python
# Load pipeline with specific weights file
pipeline = CuvisPipeline.load_pipeline(
    config_path="configs/pipeline/rx_statistical.yaml",
    weights_path="outputs/trained_models/rx_custom.pt",
    device="cuda",
    strict_weight_loading=True
)
```

## Hydra Composition for TrainRuns

For training experiments, use Hydra composition to combine pipeline, data, and training configs:

### TrainRun YAML Structure

```yaml
# @package _global_

name: my_trainrun

defaults:
  - /pipeline@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

# Override data configuration
data:
  train_ids: [0]
  val_ids: [3, 4]
  test_ids: [1, 5]
  batch_size: 1

# Override training configuration
training:
  seed: 42
  trainer:
    max_epochs: 50
    accelerator: auto
    devices: 1
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.01

# Pipeline-specific overrides
output_dir: outputs/my_experiment
unfreeze_nodes: []
metric_nodes: [metrics_anomaly]
loss_nodes: []
```

### Using TrainRun Configs

```python
import hydra
from omegaconf import DictConfig
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

@hydra.main(config_path="../configs/", config_name="trainrun/my_trainrun", version_base=None)
def main(cfg: DictConfig) -> None:
    # Access composed configuration
    print(f"Pipeline: {cfg.pipeline.metadata.name}")
    print(f"Data: {cfg.data}")
    print(f"Training: {cfg.training}")

    # Build pipeline from composed config
    # ... training code ...

if __name__ == "__main__":
    main()
```

### Command-line Overrides with Hydra

```bash
# Override parameters from command line
uv run python my_script.py \
    training.trainer.max_epochs=100 \
    training.optimizer.lr=0.0001 \
    data.batch_size=4 \
    output_dir=outputs/custom_experiment
```

## YAML Best Practices

### 1. Consistent Naming Conventions

```yaml
# Good: descriptive, consistent names
nodes:
  - name: data_loader
  - name: preprocessor
  - name: anomaly_detector
  - name: metrics_node

# Avoid: generic or inconsistent names
nodes:
  - name: node1
  - name: n2
  - name: Detector_Node
```

### 2. Organize Connections by Flow

```yaml
connections:
  # Data loading → Preprocessing
  - from: data_loader.outputs.cube
    to: preprocessor.inputs.data

  # Preprocessing → Detection
  - from: preprocessor.outputs.normalized
    to: detector.inputs.data

  # Detection → Metrics
  - from: detector.outputs.scores
    to: metrics.inputs.predictions
  - from: data_loader.outputs.labels
    to: metrics.inputs.targets

  # Metrics → Monitoring
  - from: metrics.outputs.results
    to: monitor.inputs.metrics
```

### 3. Use Comments for Clarity

```yaml
nodes:
  # ===== Data Loading =====
  - name: data_loader
    class: cuvis_ai.node.data.DataLoaderNode
    params:
      path: data/

  # ===== Preprocessing =====
  - name: normalizer
    class: cuvis_ai.node.normalization.MinMaxNormalizer
    params:
      eps: 1.0e-06  # Small epsilon for numerical stability
      use_running_stats: true  # Track running statistics during training
```

### 4. Version Control Metadata

```yaml
metadata:
  name: My_Pipeline
  description: Detailed description of pipeline purpose and capabilities
  created: 2026-02-04
  tags:
    - anomaly-detection
    - hyperspectral
    - production-ready
  author: your_name
  version: 1.2.0
  cuvis_ai_version: 0.1.5
```

### 5. Use Config Groups for Reusability

```
configs/
├── pipeline/
│   ├── rx_statistical.yaml
│   ├── channel_selector.yaml
│   └── deep_svdd.yaml
├── data/
│   ├── lentils.yaml
│   ├── concrete.yaml
│   └── custom.yaml
└── training/
    ├── default.yaml
    ├── fast.yaml
    └── production.yaml
```

## Converting Python to YAML

If you have a working Python pipeline, save it to YAML:

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import PipelineMetadata

# Build pipeline in Python (as shown in build-pipeline-python.md)
pipeline = CuvisPipeline("my_pipeline")
# ... add nodes and connections ...

# Save to YAML
pipeline.save_to_file(
    "configs/pipeline/my_pipeline.yaml",
    metadata=PipelineMetadata(
        name="my_pipeline",
        description="Converted from Python",
        tags=["custom"],
        author="your_name"
    )
)
```

This creates:
- `my_pipeline.yaml` - Pipeline configuration
- `my_pipeline.pt` - Trained weights (if nodes have parameters)

## Troubleshooting

### Issue: Invalid YAML Syntax
```
yaml.scanner.ScannerError: while scanning a simple key
```
**Solution:** Check indentation (use spaces, not tabs) and ensure colons have spaces:
```yaml
# Wrong
nodes:
  -name:node1  # Missing space after colon

# Correct
nodes:
  - name: node1
```

### Issue: Class Not Found
```
ModuleNotFoundError: No module named 'cuvis_ai.node.custom'
```
**Solution:** Verify the class path matches the actual module structure:
```python
# Check available nodes
from cuvis_ai.node import data
print(dir(data))  # List available classes
```

### Issue: Connection Error
```
ConnectionError: Port 'output' not found on node 'detector'
```
**Solution:** Check node's actual port names using the [Node Catalog](../node-catalog/index.md) or:
```python
from cuvis_ai.anomaly.rx_detector import RXGlobal
print(RXGlobal.INPUT_SPECS.keys())   # → ['data']
print(RXGlobal.OUTPUT_SPECS.keys())  # → ['scores']
```

### Issue: Parameter Type Mismatch
```
TypeError: expected int but got str for parameter 'num_channels'
```
**Solution:** Ensure parameter types match node requirements:
```yaml
# Wrong
params:
  num_channels: "61"  # String

# Correct
params:
  num_channels: 61  # Integer
```

## See Also
- [Build Pipelines in Python](build-pipeline-python.md)
- [Pipeline Schema Reference](../config/pipeline-schema.md)
- [Hydra Composition](../config/hydra-composition.md)
- [Config Groups](../config/config-groups.md)
- [Node Catalog](../node-catalog/index.md)
