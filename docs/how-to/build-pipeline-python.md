!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# How-To: Build Pipelines in Python

## Overview
Learn how to build pipelines programmatically in Python using the cuvis-ai framework. This guide demonstrates the recommended pattern used in all cuvis-ai examples.

## Prerequisites
- cuvis-ai installed
- Basic understanding of [Pipeline Lifecycle](../concepts/pipeline-lifecycle.md)
- Familiarity with [Nodes](../concepts/node-system-deep-dive.md)

## Recommended Approach: Direct Port Connections

This is the pattern used in all cuvis-ai examples. Nodes are instantiated directly and connected using tuples of port references.

### Basic Pipeline Construction

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.node.normalization import MinMaxNormalizer

# Create pipeline
pipeline = CuvisPipeline("my_pipeline")

# Instantiate nodes directly
data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
rx = RXGlobal(num_channels=61, eps=1.0e-6)

# Connect using tuples of port references
pipeline.connect(
    (data_node.outputs.cube, normalizer.data),
    (normalizer.normalized, rx.data),
)

# Validate and run
pipeline.validate()
```

### Multi-Branch Pipeline

Group connections by purpose using comments for better readability:

```python
from cuvis_ai.node.conversion import ScoreToLogit
from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode

pipeline = CuvisPipeline("multi_branch")

# Instantiate all nodes
data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
rx = RXGlobal(num_channels=61, eps=1.0e-6)
logit_head = ScoreToLogit(init_scale=1.0, init_bias=0.0)
decider = BinaryDecider(threshold=0.5)
metrics = AnomalyDetectionMetrics(name="metrics")
tensorboard = TensorBoardMonitorNode(output_dir="logs/", run_name="experiment")

# Connect all branches in one call
pipeline.connect(
    # Processing flow
    (data_node.outputs.cube, normalizer.data),
    (normalizer.normalized, rx.data),
    (rx.scores, logit_head.scores),
    (logit_head.logits, decider.logits),
    # Metrics flow
    (decider.decisions, metrics.decisions),
    (data_node.outputs.mask, metrics.targets),
    (metrics.metrics, tensorboard.metrics),
)
```

## Advanced Patterns

### Parallel Processing Branches

A common pattern from Deep SVDD example showing multiple processing branches:

```python
from cuvis_ai.anomaly.deep_svdd import (
    DeepSVDDProjection,
    DeepSVDDCenterTracker,
    DeepSVDDScores,
    ZScoreNormalizerGlobal
)
from cuvis_ai.node.preprocessors import BandpassByWavelength
from cuvis_ai.node.normalization import PerPixelUnitNorm

pipeline = CuvisPipeline("parallel_processing")

# Data and preprocessing nodes
data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
bandpass_node = BandpassByWavelength(min_wavelength_nm=450, max_wavelength_nm=900)
unit_norm_node = PerPixelUnitNorm(eps=1e-8)

# Processing branches
encoder = ZScoreNormalizerGlobal(num_channels=50, hidden=128)
projection = DeepSVDDProjection(in_channels=128, rep_dim=64, hidden=256)
center_tracker = DeepSVDDCenterTracker(rep_dim=64, alpha=0.1)
score_node = DeepSVDDScores()

# Monitoring
metrics_node = AnomalyDetectionMetrics(name="metrics")
tensorboard = TensorBoardMonitorNode(output_dir="logs/", run_name="parallel")

# Connect preprocessing chain
pipeline.connect(
    (data_node.outputs.cube, bandpass_node.data),
    (data_node.outputs.wavelengths, bandpass_node.wavelengths),
    (bandpass_node.filtered, unit_norm_node.data),
    (unit_norm_node.normalized, encoder.data),
)

# Connect parallel branches from encoder
pipeline.connect(
    (encoder.normalized, projection.data),
    (projection.embeddings, center_tracker.embeddings),
    (projection.embeddings, score_node.embeddings),
    (center_tracker.center, score_node.center),
)

# Connect metrics and monitoring
pipeline.connect(
    (score_node.scores, metrics_node.logits),
    (data_node.outputs.mask, metrics_node.targets),
    (metrics_node.metrics, tensorboard.metrics),
)
```

### Pipeline Factories

Create reusable factory functions for common pipeline patterns:

```python
def create_rx_pipeline(
    normal_class_ids: list[int],
    num_channels: int = 61,
    output_dir: str = "outputs/"
) -> CuvisPipeline:
    """Factory for RX statistical anomaly detection pipelines."""
    pipeline = CuvisPipeline("RX_Statistical")

    # Instantiate nodes
    data_node = LentilsAnomalyDataNode(normal_class_ids=normal_class_ids)
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    rx = RXGlobal(num_channels=num_channels, eps=1.0e-6)
    logit_head = ScoreToLogit(init_scale=1.0, init_bias=0.0)
    decider = BinaryDecider(threshold=0.5)
    metrics = AnomalyDetectionMetrics(name="metrics")
    tensorboard = TensorBoardMonitorNode(output_dir=output_dir, run_name="rx")

    # Connect
    pipeline.connect(
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, rx.data),
        (rx.scores, logit_head.scores),
        (logit_head.logits, decider.logits),
        (decider.decisions, metrics.decisions),
        (data_node.outputs.mask, metrics.targets),
        (metrics.metrics, tensorboard.metrics),
    )

    return pipeline

# Use factory
pipeline1 = create_rx_pipeline(normal_class_ids=[0, 1], output_dir="exp1/")
pipeline2 = create_rx_pipeline(normal_class_ids=[1, 2], output_dir="exp2/")
```

## Saving and Loading

### Save Pipeline

```python
from cuvis_ai_core.training.config import PipelineMetadata

# Save without metadata (simplest form)
pipeline.save_to_file("pipeline.yaml")
# Creates:
#   - pipeline.yaml (configuration)
#   - pipeline.pt (weights)

# Save with optional metadata for better organization
pipeline.save_to_file(
    "pipeline.yaml",
    metadata=PipelineMetadata(
        name="my_pipeline",
        description="RX anomaly detection pipeline",
        tags=["statistical", "rx"],
        author="your_name"
    )
)
```

### Load and Evaluate Pipeline

```python
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.training import StatisticalTrainer

# Load pipeline from configuration (automatically finds .pt weights)
loaded_pipeline = CuvisPipeline.load_pipeline("pipeline.yaml")

# Load with custom weights path and device
loaded_pipeline = CuvisPipeline.load_pipeline(
    config_path="pipeline.yaml",
    weights_path="custom_weights.pt",
    device="cuda",
    strict_weight_loading=True  # Fail if weights don't match exactly
)

# Load with config overrides
loaded_pipeline = CuvisPipeline.load_pipeline(
    config_path="pipeline.yaml",
    config_overrides={"nodes.0.params.threshold": 0.8}
)

# To evaluate the loaded pipeline, use a trainer with datamodule
datamodule = SingleCu3sDataModule(
    cu3s_file_path="data/test.cu3s",
    batch_size=1,
    processing_mode="Reflectance"
)
datamodule.setup(stage="test")

# For statistical pipelines
trainer = StatisticalTrainer(pipeline=loaded_pipeline, datamodule=datamodule)
test_results = trainer.test()

# For gradient-trained pipelines
from cuvis_ai_core.training import GradientTrainer
trainer = GradientTrainer(
    pipeline=loaded_pipeline,
    datamodule=datamodule,
    loss_nodes=[],  # Empty for inference-only
    metric_nodes=[metrics_node]
)
test_results = trainer.test()
```

## Best Practices

1. **Use direct port connections** - More readable and type-safe than string-based connections
2. **Group related connections with comments** - Organize connection tuples by purpose (processing flow, metrics flow, visualization flow)
3. **Store nodes in descriptive variables** - Use `data_node`, `normalizer`, `rx` instead of generic names
4. **Validate early** - Call `pipeline.validate()` before training to catch connection errors
5. **Leverage port attributes** - Use `node.port_name` for direct port access (e.g., `data_node.outputs.cube`)
6. **Connect in logical order** - Group connections by data flow (processing → metrics → visualization)
7. **Use factory functions** - Create reusable pipeline patterns for common workflows

## Troubleshooting

### Issue: Connection Error
```python
ConnectionError: Port 'output' not found on node 'loader'
```
**Solution:** Check available ports using class attributes:
```python
# Check input port specs
print(DataLoaderNode.INPUT_SPECS.keys())

# Check output port specs
print(DataLoaderNode.OUTPUT_SPECS.keys())

# Or check on an instance
data_node = DataLoaderNode()
print(dir(data_node.inputs))   # List available input ports
print(dir(data_node.outputs))  # List available output ports
```

### Issue: Type Mismatch
```python
TypeError: Port expects np.ndarray, got torch.Tensor
```
**Solution:** Check port specifications and add conversion if needed:
```python
# Check port dtype requirements
print(RXNode.INPUT_SPECS["data"].dtype)  # Expected dtype

# Add conversion node if types don't match
```

## See Also
- [Build Pipelines in YAML](build-pipeline-yaml.md)
- [Pipeline Lifecycle](../concepts/pipeline-lifecycle.md)
- [Node System](../concepts/node-system-deep-dive.md)
- [RX Statistical Tutorial](../tutorials/rx-statistical.md)
- [Deep SVDD Tutorial](../tutorials/deep-svdd-gradient.md)
