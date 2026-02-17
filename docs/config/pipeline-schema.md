!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Pipeline Configuration Schema

Complete reference for pipeline YAML structure, fields, validation rules, and examples.

## Overview

Pipeline configurations define the computational graph for hyperspectral image processing:

- **metadata**: Pipeline identification and documentation
- **nodes**: Processing components with parameters
- **connections**: Data flow between nodes

**File location:** `configs/pipeline/`

**Usage:** Referenced in trainrun configs via Hydra composition

---

## Quick Reference

### Minimal Pipeline

```yaml
metadata:
  name: My_Pipeline
  description: Pipeline description
  author: cuvis.ai

nodes:
  - name: data_loader
    class: cuvis_ai.node.data.LentilsAnomalyDataNode
    params:
      normal_class_ids: [0, 1]

  - name: detector
    class: cuvis_ai.anomaly.rx_detector.RXGlobal
    params:
      num_channels: 61
      eps: 1.0e-06

connections:
  - from: data_loader.outputs.cube
    to: detector.inputs.data
```

### Complete Pipeline Structure

```yaml
metadata:
  name: string                 # Required
  description: string           # Required
  created: string              # Optional
  tags: [...]                  # Optional
  author: string               # Required
  version: string              # Optional
  cuvis_ai_version: string     # Optional

nodes:
  - name: string               # Required, unique
    class: string              # Required, importable Python path
    params: {}                 # Optional, node-specific parameters

connections:
  - from: node.outputs.port    # Required
    to: node.inputs.port       # Required
```

---

## Metadata Section

### Required Fields

**name** (string)
- Pipeline identifier
- Used in logs and saved models
- Should be descriptive and unique

```yaml
metadata:
  name: RX_Statistical
```

**description** (string)
- Brief description of pipeline purpose
- Helps identify pipelines in multi-experiment setups

```yaml
metadata:
  description: RX anomaly detector with statistical initialization
```

**author** (string)
- Creator or organization name
- Used for tracking and attribution

```yaml
metadata:
  author: cuvis.ai
```

### Optional Fields

**created** (string)
- Creation timestamp
- Format: ISO 8601 or any consistent format

```yaml
metadata:
  created: '2026-02-04'
```

**tags** (list of strings)
- Classification tags for filtering and search
- Common tags: `statistical`, `gradient`, `anomaly`, `classification`

```yaml
metadata:
  tags:
    - statistical
    - rx
    - anomaly_detection
```

**version** (string)
- Pipeline version for tracking changes
- Use semantic versioning: `major.minor.patch`

```yaml
metadata:
  version: '1.2.0'
```

**cuvis_ai_version** (string)
- Framework version used to create pipeline
- Helps with compatibility tracking

```yaml
metadata:
  cuvis_ai_version: '0.1.5'
```

### Complete Metadata Example

```yaml
metadata:
  name: DRCNN_AdaClip_Gradient
  description: DRCNN channel mixer with AdaClip anomaly detection
  created: '2026-02-04T10:30:00'
  tags:
    - gradient
    - drcnn
    - adaclip
    - anomaly
  author: cuvis.ai
  version: '2.1.0'
  cuvis_ai_version: '0.1.5.post26'
```

---

## Nodes Section

### Node Structure

Each node has three components:

```yaml
nodes:
  - name: <unique_identifier>
    class: <fully_qualified_class_path>
    params: <dict_of_parameters>
```

### name (string, required)

- Unique identifier within the pipeline
- Used in connections to reference the node
- Case-sensitive
- Must be valid Python identifier (no spaces, special chars)

**Good names:**
```yaml
name: data_loader
name: MinMaxNormalizer
name: rx_detector
name: metrics_anomaly
```

**Avoid:**
```yaml
name: node1              # Not descriptive
name: my node            # Contains space
name: detector-rx        # Contains dash (not Python identifier)
```

### class (string, required)

- Fully qualified Python class path
- Must be importable from Python path
- Format: `module.submodule.ClassName`

**Common node classes:**

**Data nodes:**
```yaml
class: cuvis_ai.node.data.LentilsAnomalyDataNode
class: cuvis_ai.node.data.Cu3sDataNode
```

**Preprocessing nodes:**
```yaml
class: cuvis_ai.node.normalization.MinMaxNormalizer
class: cuvis_ai.node.normalization.StandardScaler
class: cuvis_ai.node.preprocessing.PCANode
```

**Detection nodes:**
```yaml
class: cuvis_ai.anomaly.rx_detector.RXGlobal
class: cuvis_ai.anomaly.lad_detector.LADDetector
class: cuvis_ai.node.conversion.ScoreToLogit
```

**Decision nodes:**
```yaml
class: cuvis_ai.deciders.binary_decider.BinaryDecider
```

**Loss nodes:**
```yaml
class: cuvis_ai.anomaly.iou_loss.IoULoss
class: cuvis_ai.anomaly.bce_loss.AnomalyBCEWithLogits
```

**Metric nodes:**
```yaml
class: cuvis_ai.node.metrics.AnomalyDetectionMetrics
```

**Monitoring nodes:**
```yaml
class: cuvis_ai.node.monitor.TensorBoardMonitorNode
class: cuvis_ai.node.visualizations.AnomalyMask
class: cuvis_ai.node.visualizations.ScoreHeatmapVisualizer
```

### params (dict, optional)

Node-specific configuration parameters. Each node class defines its own parameters.

**Example: Data node parameters**
```yaml
- name: LentilsAnomalyDataNode
  class: cuvis_ai.node.data.LentilsAnomalyDataNode
  params:
    normal_class_ids: [0, 1]
```

**Example: Normalization parameters**
```yaml
- name: MinMaxNormalizer
  class: cuvis_ai.node.normalization.MinMaxNormalizer
  params:
    eps: 1.0e-06
    use_running_stats: true
```

**Example: Detector parameters**
```yaml
- name: RXGlobal
  class: cuvis_ai.anomaly.rx_detector.RXGlobal
  params:
    num_channels: 61
    eps: 1.0e-06
    cache_inverse: true
```

**Example: Complex node with many parameters**
```yaml
- name: concrete_selector
  class: cuvis_ai.node.concrete_selector.ConcreteBandSelector
  params:
    input_channels: 61
    output_channels: 3
    tau_start: 10.0
    tau_end: 0.1
    max_epochs: 20
    use_hard_inference: true
    eps: 1.0e-06
```

**Example: Node with no parameters**
```yaml
- name: metrics_anomaly
  class: cuvis_ai.node.metrics.AnomalyDetectionMetrics
  params: {}
```

Or simply omit `params`:
```yaml
- name: metrics_anomaly
  class: cuvis_ai.node.metrics.AnomalyDetectionMetrics
```

### Complete Node Examples

**Statistical RX Pipeline Nodes:**
```yaml
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

  - name: ScoreToLogit
    class: cuvis_ai.node.conversion.ScoreToLogit
    params:
      init_scale: 1.0
      init_bias: 0.0

  - name: BinaryDecider
    class: cuvis_ai.deciders.binary_decider.BinaryDecider
    params:
      threshold: 0.5

  - name: metrics_anomaly
    class: cuvis_ai.node.metrics.AnomalyDetectionMetrics
```

---

## Connections Section

### Connection Structure

Connections define data flow between nodes using port references:

```yaml
connections:
  - from: <source_node>.<port_type>.<port_name>
    to: <target_node>.<port_type>.<port_name>
```

**Port types:**
- `outputs` - Source ports (produce data)
- `inputs` - Target ports (consume data)

### Connection Syntax

**Pattern:**
```
<node_name>.<port_type>.<port_name>
```

**Components:**
1. **node_name**: Must match a node name defined in `nodes` section
2. **port_type**: Either `outputs` (source) or `inputs` (target)
3. **port_name**: Port identifier defined by the node class

**Valid connection:**
```yaml
- from: data_loader.outputs.cube
  to: normalizer.inputs.data
```

**Invalid connections:**
```yaml
# Wrong: Using inputs as source
- from: normalizer.inputs.data
  to: detector.inputs.data

# Wrong: Using outputs as target
- from: data_loader.outputs.cube
  to: normalizer.outputs.normalized

# Wrong: Mismatched port names
- from: data_loader.outputs.cube
  to: normalizer.inputs.wrong_port_name
```

### Common Port Names

**Data node outputs:**
```yaml
LentilsAnomalyDataNode.outputs.cube         # Hyperspectral cube
LentilsAnomalyDataNode.outputs.mask         # Ground truth mask
LentilsAnomalyDataNode.outputs.wavelengths  # Wavelength array
```

**Normalizer ports:**
```yaml
MinMaxNormalizer.inputs.data        # Input data
MinMaxNormalizer.outputs.normalized # Normalized output
```

**Detector ports:**
```yaml
RXGlobal.inputs.data          # Input hyperspectral data
RXGlobal.outputs.scores       # Anomaly scores
RXGlobal.outputs.mean         # Computed mean (statistical nodes)
RXGlobal.outputs.covariance   # Computed covariance (statistical nodes)
```

**Decision ports:**
```yaml
BinaryDecider.inputs.logits       # Input logits or scores
BinaryDecider.outputs.decisions   # Binary decisions
```

**Metric ports:**
```yaml
AnomalyDetectionMetrics.inputs.decisions  # Binary predictions
AnomalyDetectionMetrics.inputs.targets    # Ground truth
AnomalyDetectionMetrics.inputs.logits     # Optional scores
AnomalyDetectionMetrics.outputs.metrics   # Computed metrics
```

**Monitor ports:**
```yaml
TensorBoardMonitorNode.inputs.metrics    # Metric objects
TensorBoardMonitorNode.inputs.artifacts  # Visualization artifacts
```

### Connection Patterns

**Linear flow:**
```yaml
connections:
  - from: data_loader.outputs.cube
    to: normalizer.inputs.data
  - from: normalizer.outputs.normalized
    to: detector.inputs.data
  - from: detector.outputs.scores
    to: decider.inputs.scores
```

**Multi-branch flow:**
```yaml
connections:
  # Main flow
  - from: data_loader.outputs.cube
    to: normalizer.inputs.data
  - from: normalizer.outputs.normalized
    to: detector.inputs.data

  # Branch 1: Metrics
  - from: decider.outputs.decisions
    to: metrics.inputs.decisions
  - from: data_loader.outputs.mask
    to: metrics.inputs.targets

  # Branch 2: Visualization
  - from: decider.outputs.decisions
    to: viz.inputs.decisions
  - from: data_loader.outputs.cube
    to: viz.inputs.cube
```

**Fan-out pattern (one source → multiple targets):**
```yaml
connections:
  # RX scores go to multiple destinations
  - from: detector.outputs.scores
    to: decider.inputs.scores
  - from: detector.outputs.scores
    to: score_viz.inputs.scores
  - from: detector.outputs.scores
    to: metrics.inputs.logits
```

**Convergence pattern (multiple sources → one target):**
```yaml
connections:
  # Monitoring receives from multiple sources
  - from: metrics.outputs.metrics
    to: monitor.inputs.metrics
  - from: viz_mask.outputs.artifacts
    to: monitor.inputs.artifacts
  - from: score_viz.outputs.artifacts
    to: monitor.inputs.artifacts
```

### Complete Connection Examples

**RX Statistical Pipeline:**
```yaml
connections:
  - from: LentilsAnomalyDataNode.outputs.cube
    to: MinMaxNormalizer.inputs.data
  - from: LentilsAnomalyDataNode.outputs.mask
    to: metrics_anomaly.inputs.targets
  - from: LentilsAnomalyDataNode.outputs.mask
    to: mask.inputs.mask
  - from: LentilsAnomalyDataNode.outputs.cube
    to: mask.inputs.cube
  - from: MinMaxNormalizer.outputs.normalized
    to: RXGlobal.inputs.data
  - from: RXGlobal.outputs.scores
    to: ScoreToLogit.inputs.scores
  - from: ScoreToLogit.outputs.logits
    to: BinaryDecider.inputs.logits
  - from: BinaryDecider.outputs.decisions
    to: metrics_anomaly.inputs.decisions
  - from: BinaryDecider.outputs.decisions
    to: mask.inputs.decisions
  - from: metrics_anomaly.outputs.metrics
    to: TensorBoardMonitorNode.inputs.metrics
  - from: mask.outputs.artifacts
    to: TensorBoardMonitorNode.inputs.artifacts
```

**DRCNN + AdaClip Pipeline (multi-branch):**
```yaml
connections:
  # Data loading
  - from: LentilsAnomalyDataNode.outputs.cube
    to: MinMaxNormalizer.inputs.data

  # Main processing
  - from: MinMaxNormalizer.outputs.normalized
    to: channel_mixer.inputs.data
  - from: channel_mixer.outputs.output
    to: adaclip.inputs.image

  # Loss computation
  - from: adaclip.outputs.scores
    to: iou_loss.inputs.predictions
  - from: LentilsAnomalyDataNode.outputs.mask
    to: iou_loss.inputs.targets

  # Decisions
  - from: adaclip.outputs.scores
    to: decider.inputs.logits

  # Metrics
  - from: decider.outputs.decisions
    to: metrics_anomaly.inputs.decisions
  - from: LentilsAnomalyDataNode.outputs.mask
    to: metrics_anomaly.inputs.targets
  - from: adaclip.outputs.scores
    to: metrics_anomaly.inputs.logits

  # Monitoring
  - from: metrics_anomaly.outputs.metrics
    to: TensorBoardMonitorNode.inputs.metrics
```

---

## Complete Pipeline Examples

### Example 1: Statistical RX Pipeline

```yaml
metadata:
  name: RX_Statistical
  description: RX anomaly detector with statistical initialization
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

  - name: ScoreToLogit
    class: cuvis_ai.node.conversion.ScoreToLogit
    params:
      init_scale: 1.0
      init_bias: 0.0

  - name: BinaryDecider
    class: cuvis_ai.deciders.binary_decider.BinaryDecider
    params:
      threshold: 0.5

  - name: metrics_anomaly
    class: cuvis_ai.node.metrics.AnomalyDetectionMetrics

  - name: TensorBoardMonitorNode
    class: cuvis_ai.node.monitor.TensorBoardMonitorNode
    params:
      output_dir: outputs/rx_statistical/tensorboard
      run_name: RX_Statistical

connections:
  - from: LentilsAnomalyDataNode.outputs.cube
    to: MinMaxNormalizer.inputs.data
  - from: MinMaxNormalizer.outputs.normalized
    to: RXGlobal.inputs.data
  - from: RXGlobal.outputs.scores
    to: ScoreToLogit.inputs.scores
  - from: ScoreToLogit.outputs.logits
    to: BinaryDecider.inputs.logits
  - from: BinaryDecider.outputs.decisions
    to: metrics_anomaly.inputs.decisions
  - from: LentilsAnomalyDataNode.outputs.mask
    to: metrics_anomaly.inputs.targets
  - from: metrics_anomaly.outputs.metrics
    to: TensorBoardMonitorNode.inputs.metrics

```

### Example 2: Channel Selector Gradient Pipeline

```yaml
metadata:
  name: Channel_Selector
  description: Learnable channel selection with RX detection
  tags:
    - gradient
    - channel_selection
    - rx
  author: cuvis.ai
  version: '1.0.0'

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

  - name: selector
    class: cuvis_ai.node.channel_selection.ChannelSelector
    params:
      num_channels: 61
      tau_start: 8.0
      tau_end: 0.05

  - name: rx_global
    class: cuvis_ai.anomaly.rx_detector.RXGlobal
    params:
      num_channels: 61
      eps: 1.0e-06

  - name: logit_head
    class: cuvis_ai.node.conversion.ScoreToLogit
    params:
      init_scale: 1.0
      init_bias: 0.0

  - name: decider
    class: cuvis_ai.deciders.binary_decider.BinaryDecider
    params:
      threshold: 0.5

  - name: bce_loss
    class: cuvis_ai.anomaly.bce_loss.AnomalyBCEWithLogits
    params:
      weight: 1.0

  - name: entropy_loss
    class: cuvis_ai.anomaly.entropy_loss.SelectorEntropyLoss
    params:
      weight: 0.001

  - name: metrics_anomaly
    class: cuvis_ai.node.metrics.AnomalyDetectionMetrics

connections:
  # Data flow
  - from: LentilsAnomalyDataNode.outputs.cube
    to: MinMaxNormalizer.inputs.data
  - from: MinMaxNormalizer.outputs.normalized
    to: selector.inputs.data
  - from: selector.outputs.selected
    to: rx_global.inputs.data
  - from: rx_global.outputs.scores
    to: logit_head.inputs.scores
  - from: logit_head.outputs.logits
    to: decider.inputs.logits

  # Loss computation
  - from: logit_head.outputs.logits
    to: bce_loss.inputs.predictions
  - from: LentilsAnomalyDataNode.outputs.mask
    to: bce_loss.inputs.targets
  - from: selector.outputs.weights
    to: entropy_loss.inputs.weights

  # Metrics
  - from: decider.outputs.decisions
    to: metrics_anomaly.inputs.decisions
  - from: LentilsAnomalyDataNode.outputs.mask
    to: metrics_anomaly.inputs.targets

```

### Example 3: Deep SVDD Pipeline

```yaml
metadata:
  name: Deep_SVDD
  description: Deep Support Vector Data Description for anomaly detection
  tags:
    - gradient
    - deep_learning
    - svdd
  author: cuvis.ai

nodes:
  - name: LentilsAnomalyDataNode
    class: cuvis_ai.node.data.LentilsAnomalyDataNode
    params:
      normal_class_ids: [0, 1]

  - name: normalizer
    class: cuvis_ai.node.normalization.MinMaxNormalizer
    params:
      eps: 1.0e-06
      use_running_stats: true

  - name: projection
    class: cuvis_ai.node.deep_svdd.ProjectionNetwork
    params:
      input_dim: 61
      hidden_dims: [128, 64, 32]
      output_dim: 16

  - name: deepsvdd_loss
    class: cuvis_ai.anomaly.deep_svdd_loss.DeepSVDDLoss
    params:
      radius: 0.0
      nu: 0.1

  - name: metrics_anomaly
    class: cuvis_ai.node.metrics.AnomalyDetectionMetrics

connections:
  - from: LentilsAnomalyDataNode.outputs.cube
    to: normalizer.inputs.data
  - from: normalizer.outputs.normalized
    to: projection.inputs.data
  - from: projection.outputs.embeddings
    to: deepsvdd_loss.inputs.embeddings
  - from: deepsvdd_loss.outputs.scores
    to: metrics_anomaly.inputs.logits
  - from: LentilsAnomalyDataNode.outputs.mask
    to: metrics_anomaly.inputs.targets

```

---

## Validation Rules

### 1. Node Name Uniqueness

All node names must be unique within a pipeline.

**Valid:**
```yaml
nodes:
  - name: normalizer_1
    class: cuvis_ai.node.normalization.MinMaxNormalizer
  - name: normalizer_2
    class: cuvis_ai.node.normalization.MinMaxNormalizer
```

**Invalid:**
```yaml
nodes:
  - name: normalizer
    class: cuvis_ai.node.normalization.MinMaxNormalizer
  - name: normalizer  # ✗ Duplicate name
    class: cuvis_ai.node.normalization.StandardScaler
```

### 2. Class Importability

All node classes must be importable from Python path.

**Valid:**
```yaml
class: cuvis_ai.anomaly.rx_detector.RXGlobal  # ✓ Exists
```

**Invalid:**
```yaml
class: cuvis_ai.anomaly.NonexistentNode  # ✗ Import error
class: RXGlobal                          # ✗ Not fully qualified
```

### 3. Connection Validity

**Source must be output port:**
```yaml
# Valid
- from: detector.outputs.scores
  to: decider.inputs.scores

# Invalid
- from: detector.inputs.data  # ✗ Can't use input as source
  to: decider.inputs.scores
```

**Target must be input port:**
```yaml
# Valid
- from: detector.outputs.scores
  to: decider.inputs.scores

# Invalid
- from: detector.outputs.scores
  to: decider.outputs.decisions  # ✗ Can't use output as target
```

**Referenced nodes must exist:**
```yaml
# Valid
nodes:
  - name: detector
    ...
connections:
  - from: detector.outputs.scores
    to: decider.inputs.scores

# Invalid
connections:
  - from: nonexistent_node.outputs.data  # ✗ Node not defined
    to: decider.inputs.scores
```

### 4. Parameter Types

Node parameters must match expected types.

**Valid:**
```yaml
params:
  num_channels: 61         # int
  eps: 1.0e-06             # float
  use_running_stats: true  # bool
  normal_class_ids: [0, 1] # list
```

**Invalid:**
```yaml
params:
  num_channels: "61"       # ✗ String instead of int
  eps: true                # ✗ Bool instead of float
  normal_class_ids: 0      # ✗ Int instead of list
```

---

## Best Practices

### 1. Descriptive Node Names

**Good:**
```yaml
name: data_loader
name: MinMaxNormalizer
name: rx_detector
name: metrics_anomaly
```

**Avoid:**
```yaml
name: node1
name: n1
name: temp
```

### 2. Consistent Naming Convention

**Pick a style and stick to it:**
```yaml
# snake_case (recommended)
name: data_loader
name: rx_detector
name: score_visualizer

# PascalCase (alternative)
name: DataLoader
name: RXDetector
name: ScoreVisualizer
```

### 3. Organized Connections

**Group connections by purpose:**
```yaml
connections:
  # Main processing flow
  - from: data_loader.outputs.cube
    to: normalizer.inputs.data
  - from: normalizer.outputs.normalized
    to: detector.inputs.data

  # Loss computation
  - from: detector.outputs.scores
    to: loss.inputs.predictions

  # Metrics and monitoring
  - from: metrics.outputs.metrics
    to: monitor.inputs.metrics
```

### 4. Comments for Complex Pipelines

```yaml
nodes:
  # Data loading and preprocessing
  - name: data_loader
    ...
  - name: normalizer
    ...

  # Feature extraction
  - name: channel_selector
    ...
  - name: pca
    ...

  # Anomaly detection
  - name: rx_detector
    ...
  - name: threshold_decider
    ...
```

### 5. Parameter Documentation

```yaml
- name: concrete_selector
  class: cuvis_ai.node.concrete_selector.ConcreteBandSelector
  params:
    input_channels: 61
    output_channels: 3
    tau_start: 10.0        # Initial temperature for Gumbel-Softmax
    tau_end: 0.1           # Final temperature after annealing
    max_epochs: 20         # Epochs for temperature schedule
    use_hard_inference: true  # Use hard selection during inference
```

---

## See Also

- **Configuration Guides**:
  - [Config Groups](config-groups.md) - Organizing pipeline configs
  - [TrainRun Schema](trainrun-schema.md) - Complete experiment configuration
  - [Hydra Composition](hydra-composition.md) - Composition patterns
- **How-To Guides**:
  - [Build Pipelines in YAML](../how-to/build-pipeline-yaml.md) - Creating pipeline configs
  - [Build Pipelines in Python](../how-to/build-pipeline-python.md) - Programmatic construction
  - [Add Builtin Node](../how-to/add-builtin-node.md) - Creating custom nodes
- **Node Catalog**:
  - [Data Nodes](../node-catalog/data-nodes.md) - Available data loading nodes
  - [Processing Nodes](../node-catalog/preprocessing.md) - Normalization, PCA, etc.
  - [Statistical Nodes](../node-catalog/statistical.md) - Anomaly detector implementations
  - [Loss & Metrics Nodes](../node-catalog/loss-metrics.md) - Loss and metric computations
