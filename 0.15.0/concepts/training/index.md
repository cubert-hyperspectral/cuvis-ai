Status: Needs Review

This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

______________________________________________________________________

# Two-Phase Training

*Many pipelines use two phases: statistical initialization from initialization data, then optional gradient training via backpropagation.*

Two-phase training combines statistical methods and deep learning. **Phase 1** computes statistics fast and deterministically. **Phase 2** refines parameters via gradients for task-specific optimization.

**Phase 1 Benefits:**

- ⚡ Fast (seconds to minutes)
- 💾 Memory efficient
- 🎯 Strong initialization for Phase 2
- 🔒 Deterministic and reproducible

**Phase 2 Benefits:**

- 📈 Learns task-specific features
- 🎨 Flexible optimization objectives
- 🔬 Fine-grained parameter refinement

______________________________________________________________________

## Phase 1: Statistical Initialization

*Compute statistics from initialization data without gradients using [Welford's algorithm](https://www.johndcook.com/blog/standard_deviation/).*

### Statistical Training Lifecycle

This flowchart shows the complete Phase 1 training process, from trainer creation through validation:

```
flowchart TD
    A[Create StatisticalTrainer] --> B["stat_trainer = StatisticalTrainer(pipeline, datamodule)"]
    B --> C["stat_trainer.fit() called"]
    C --> D{Find nodes with<br/>requires_initial_fit?}
    D -->|Yes| E[Sort nodes by<br/>port connections]
    D -->|No| M[Skip Phase 1]
    E --> F[For each statistical node]
    F --> G[Get transformed data<br/>via port routing]
    G --> H["Call node.statistical_initialization(input_stream)"]
    H --> I[Accumulate statistics<br/>using Welford's algorithm]
    I --> J[Store as frozen buffers<br/>register_buffer()]
    J --> K["Call node.prepare_for_gradient_train()"]
    K --> L{More nodes?}
    L -->|Yes| F
    L -->|No| M[Statistical initialization complete]
    M --> N["Optional: stat_trainer.validate()"]
    N --> O[Validation metrics]
```

**Key Steps:**

1. **Trainer Creation:** Instantiate `StatisticalTrainer` with pipeline and datamodule
1. **fit() Execution:** Calls the statistical initialization process
1. **Node Discovery:** Finds all nodes with `requires_initial_fit=True`
1. **Topological Ordering:** Sorts nodes by port connections for proper data flow
1. **Per-Node Initialization:**
1. Routes data through preceding nodes via port connections
1. Calls `statistical_initialization()` with data stream
1. Accumulates statistics using Welford's online algorithm
1. Stores statistics as PyTorch buffers (frozen, non-trainable)
1. Prepares node for potential gradient training phase
1. **Validation (Optional):** Test initialized pipeline on validation set

### Loading Initialization Data

```python
from cuvis_ai_dataloader.data import Cu3sDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

cu3s_path = "data/initialization/samples.cu3s"

datamodule = Cu3sDataModule(
    cu3s_file_path=cu3s_path,
    annotation_json_path="data/initialization/annotations.json",
    splits=DataSplitConfig(
        train=[Selector(kind=SelectorKind.FILE_INDICES, source=cu3s_path, ids=[0, 1, 2])],
        val=[Selector(kind=SelectorKind.FILE_INDICES, source=cu3s_path, ids=[3])],
        test=[Selector(kind=SelectorKind.FILE_INDICES, source=cu3s_path, ids=[4])],
    ),
    batch_size=4
)

datamodule.setup(stage="fit")
```

### Computing Statistics

```python
from cuvis_ai_core.training import StatisticalTrainer

stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
stat_trainer.fit()  # Process all batches

# Nodes now initialized
rx_detector = {node.name: node for node in pipeline.nodes()}["rx_detector"]
assert rx_detector._statistically_initialized
```

### Statistical Nodes

#### RXGlobal (Anomaly Detection)

```python
from cuvis_ai.node.anomaly.rx_detector import RXGlobal

from cuvis_ai_core.training import StatisticalTrainer

rx_node = RXGlobal(num_channels=61, eps=1e-6)
# Nodes are added automatically when you wire them with pipeline.connect(...)

# Use StatisticalTrainer to initialize
trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()  # Automatically calls statistical_initialization on rx_node

print(rx_node.mu.shape)      # (61,) - Background mean
print(rx_node.sigma.shape)   # (61, 61) - Covariance matrix
```

**Implementation Pattern:**

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Compute statistics using Welford's algorithm."""
    n = 0
    mean = torch.zeros(self.num_channels, dtype=torch.float64)
    M2 = torch.zeros((self.num_channels, self.num_channels), dtype=torch.float64)

    for batch_data in input_stream:
        data = batch_data["data"]
        flattened = data.flatten(0, 2).double()

        for sample in flattened:
            n += 1
            delta = sample - mean
            mean += delta / n
            delta2 = sample - mean
            M2 += torch.outer(delta, delta2)

    covariance = M2 / (n - 1) if n > 1 else M2
    self.register_buffer("mu", mean.float())
    self.register_buffer("sigma", covariance.float())
    self._statistically_initialized = True
```

#### MinMaxNormalizer

```python
from cuvis_ai.node.normalization import MinMaxNormalizer

from cuvis_ai_core.training import StatisticalTrainer

normalizer = MinMaxNormalizer(eps=1e-6, use_running_stats=True)
# Nodes are added automatically when you wire them with pipeline.connect(...)

# Use StatisticalTrainer to initialize
trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()  # Automatically calls statistical_initialization on normalizer

print(normalizer.running_min)  # Global min values
print(normalizer.running_max)  # Global max values
```

#### SoftChannelSelector

```python
from cuvis_ai.node.channel_selector import SoftChannelSelector

from cuvis_ai_core.training import StatisticalTrainer

selector = SoftChannelSelector(
    n_select=10,
    input_channels=61,
    init_method="variance",
    temperature_init=5.0
)
# Nodes are added automatically when you wire them with pipeline.connect(...)

# Use StatisticalTrainer to initialize
trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()  # Automatically calls statistical_initialization on selector
```

#### TrainablePCA

```python
from cuvis_ai.node.dimensionality_reduction import TrainablePCA

from cuvis_ai_core.training import StatisticalTrainer

pca = TrainablePCA(n_components=10, input_dim=61)
# Nodes are added automatically when you wire them with pipeline.connect(...)

# Use StatisticalTrainer to initialize
trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()  # Automatically calls statistical_initialization on pca

print(pca.components.shape)      # (10, 61)
print(pca.explained_variance)    # Variance per component
```

### Validation After Phase 1

```python
# validate() runs the validation pass; the metric nodes log their results to
# TensorBoard. StatisticalTrainer.validate() does not return a metrics object,
# so inspect the run in TensorBoard (or read the metric nodes directly).
stat_trainer.validate()

# Expected: Reasonable performance with just statistics
# Example: IoU > 0.5, F1 > 0.6
```

______________________________________________________________________

## Phase 2: Gradient Training

*Refine parameters via gradient descent.*

### Gradient Training Lifecycle

This flowchart shows the complete Phase 2 training process, from unfreezing nodes through test evaluation:

```
flowchart TD
    A[After Phase 1 completion] --> B["Select nodes to unfreeze"]
    B --> C["pipeline.unfreeze_nodes_by_name(['selector', 'rx_detector'])"]
    C --> D["Buffers → Parameters<br/>(requires_grad=True)"]
    D --> E["Create GradientTrainer"]
    E --> F["grad_trainer = GradientTrainer(<br/>pipeline, datamodule,<br/>loss_nodes, metric_nodes,<br/>training_config)"]
    F --> G["grad_trainer.fit() called"]
    G --> H[Create CuvisLightningModule]
    H --> I[Register loss nodes via ports]
    I --> J[Register metric nodes via ports]
    J --> K[Create PyTorch Lightning Trainer]
    K --> L[Training Loop Start]
    L --> M[training_step:<br/>Port-based forward pass]
    M --> N[Aggregate loss from loss_nodes]
    N --> O[Backward pass + optimizer step]
    O --> P[validation_step:<br/>Forward pass + metrics]
    P --> Q{Execute callbacks?}
    Q -->|Early stopping| R[Check stopping criteria]
    Q -->|Checkpoint| S[Save best model]
    Q -->|LR scheduler| T[Adjust learning rate]
    R --> U{Continue training?}
    S --> U
    T --> U
    U -->|Yes| M
    U -->|No| V[Training Complete]
    V --> W["Optional: grad_trainer.test()"]
    W --> X[Test metrics with best checkpoint]
```

### Unfreezing Nodes

```python
# After Phase 1
stat_trainer.fit()

# Select nodes to train
unfreeze_node_names = ["rx_detector", "selector"]
pipeline.unfreeze_nodes_by_name(unfreeze_node_names)

trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
```

The base class `unfreeze()` promotes buffers listed in `TRAINABLE_BUFFERS` to `nn.Parameter`; `freeze()` reverts them. See [Node System Deep Dive: TRAINABLE_BUFFERS](https://cubert-hyperspectral.github.io/cuvis-ai/0.15.0/concepts/node/#trainable_buffers) for the full mechanism.

### Defining Loss Nodes

```python
from cuvis_ai.node.losses import AnomalyBCEWithLogits

bce_loss = AnomalyBCEWithLogits(name="bce", weight=10.0)

pipeline.connect(
    (logit_head.logits, bce_loss.predictions),
    (data_node.mask, bce_loss.targets)
)
```

### Gradient Training

```python
from cuvis_ai_core.training import GradientTrainer
from cuvis_ai_schemas.training import (
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
)

training_config = TrainingConfig(
    seed=42,
    optimizer=OptimizerConfig(
        name="adamw",
        lr=0.001,
        weight_decay=1e-4
    ),
    scheduler=SchedulerConfig(
        name="reduce_on_plateau",
        monitor="metrics_anomaly/iou",
        mode="max",
        factor=0.5,
        patience=5
    ),
    max_epochs=50,
    accelerator="auto"
)

grad_trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[bce_loss],
    metric_nodes=[metrics_node],
    training_config=training_config
)

grad_trainer.fit()
test_results = grad_trainer.test()
```

______________________________________________________________________

## Callbacks

*Hooks into training loop for monitoring, checkpointing, and early stopping.*

```python
training_config = TrainingConfig(
    max_epochs=50,
    accelerator="auto",
    callbacks=CallbacksConfig(
        checkpoint=ModelCheckpointConfig(
            monitor="metrics_anomaly/iou",
            mode="max",
            save_top_k=3
        ),
        early_stopping=[
            EarlyStoppingConfig(
                monitor="val/bce",
                mode="min",
                patience=10,
                min_delta=0.001
            )
        ],
        lr_monitor=LearningRateMonitorConfig(logging_interval="epoch")
    )
)
```

______________________________________________________________________

## Complete Two-Phase Example

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.channel_selector import SoftChannelSelector
from cuvis_ai.node.anomaly.rx_detector import RXGlobal
from cuvis_ai.node.conversion import ScoreToLogit
from cuvis_ai.node.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.losses import AnomalyBCEWithLogits
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai_core.training import StatisticalTrainer, GradientTrainer
from cuvis_ai_schemas.training import (
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_dataloader.data import Cu3sDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

# ============ SETUP ============

pipeline = CuvisPipeline("Channel_Selector")

data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
normalizer = MinMaxNormalizer(eps=1e-6, use_running_stats=True)
selector = SoftChannelSelector(n_select=10, input_channels=61, init_method="variance")
rx_node = RXGlobal(num_channels=10, eps=1e-6)
logit_head = ScoreToLogit(init_scale=1.0, init_bias=0.0)
decider = BinaryDecider(threshold=0.5)

bce_loss = AnomalyBCEWithLogits(name="bce", weight=10.0)
metrics_node = AnomalyDetectionMetrics(name="metrics_anomaly")

pipeline.connect(
    (data_node.cube, normalizer.data),
    (normalizer.normalized, selector.data),
    (selector.selected, rx_node.data),
    (rx_node.scores, logit_head.scores),
    (logit_head.logits, bce_loss.predictions),
    (data_node.mask, bce_loss.targets),
    (logit_head.logits, decider.logits),
    (decider.decisions, metrics_node.decisions),
    (data_node.mask, metrics_node.targets)
)

cu3s_path = "data/Lentils/Lentils_000.cu3s"

datamodule = Cu3sDataModule(
    cu3s_file_path=cu3s_path,
    annotation_json_path="data/Lentils/Lentils_000.json",
    splits=DataSplitConfig(
        train=[Selector(kind=SelectorKind.FILE_INDICES, source=cu3s_path, ids=[0, 2, 3])],
        val=[Selector(kind=SelectorKind.FILE_INDICES, source=cu3s_path, ids=[1])],
        test=[Selector(kind=SelectorKind.FILE_INDICES, source=cu3s_path, ids=[4, 5])],
    ),
    batch_size=2
)

# ============ PHASE 1: STATISTICAL INITIALIZATION ============

print("Phase 1: Statistical initialization...")
stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
stat_trainer.fit()

print("✓ Statistical initialization complete")
# Run validation; the metric nodes log results (e.g. IoU) to TensorBoard.
stat_trainer.validate()

# ============ PHASE 2: GRADIENT TRAINING ============

print("\nPhase 2: Gradient training...")
unfreeze_nodes = [selector.name, rx_node.name, logit_head.name]
pipeline.unfreeze_nodes_by_name(unfreeze_nodes)

trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
print(f"✓ Unfrozen {len(unfreeze_nodes)} nodes ({trainable_params:,} parameters)")

training_config = TrainingConfig(
    seed=42,
    optimizer=OptimizerConfig(name="adamw", lr=0.001),
    scheduler=SchedulerConfig(name="reduce_on_plateau", monitor="metrics_anomaly/iou", mode="max"),
    max_epochs=20,
    accelerator="auto"
)

grad_trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[bce_loss],
    metric_nodes=[metrics_node],
    training_config=training_config
)

grad_trainer.fit()
test_results = grad_trainer.test()
print(f"✓ Test IoU: {test_results[0]['metrics_anomaly/iou']:.3f}")

# ============ SAVE ============

pipeline.save_to_file(
    "outputs/channel_selector.yaml",
    metadata=PipelineMetadata(
        name="Channel_Selector_v1",
        description="RX + learnable channel selection",
        tags=["two-phase", "gradient-trained", "production"]
    )
)

print("✓ Saved to outputs/channel_selector.yaml")
```

______________________________________________________________________

## Saving & Loading

### Option 1: Save Full Pipeline

```python
pipeline.save_to_file(
    "outputs/trained_pipeline.yaml",
    metadata=PipelineMetadata(name="Full_Pipeline")
)
# Generates: outputs/trained_pipeline.yaml, outputs/trained_pipeline.pt
```

### Loading a Saved Pipeline

```python
pipeline = CuvisPipeline.load_pipeline(
    config_path="outputs/trained_pipeline.yaml",
    weights_path="outputs/trained_pipeline.pt",
    device="cuda"
)

outputs = pipeline.forward(batch=test_data)
```

______________________________________________________________________

## Best Practices

| Practice               | Detail                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| Separate init data     | Use clean, representative samples for Phase 1; diverse data (including anomalies) for Phase 2 |
| Validate init quality  | Run `stat_trainer.validate()` after Phase 1; high mean RX scores indicate poor initialization |
| Detect data drift      | Compare `node.mu` against new data statistics; reinitialize if drift ratio exceeds threshold  |
| Version initialization | Save metadata (version, date, source path, sample count) alongside `.pt` checkpoints          |

______________________________________________________________________

Troubleshooting

**"Node not initialized"** -- Use `StatisticalTrainer` for Phase 1 before Phase 2:

```python
stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
stat_trainer.fit()   # Phase 1
grad_trainer.fit()   # Phase 2
```

**Poor Detection Performance** -- Bad initialization data, small sample size, or contaminated data. Inspect calibration statistics and remove outliers:

```python
print(f"Calibration samples: {len(calib_data)}")
percentiles = torch.quantile(calib_data, torch.tensor([0.01, 0.99]))
clean_data = remove_outliers(calib_data)
```

**Gradient Training Degrades Performance** -- Lower the learning rate, freeze more nodes, and add early stopping:

```python
optimizer_config = OptimizerConfig(lr=0.0001)
unfreeze_nodes = ["selector"]  # fewer nodes
early_stopping = EarlyStoppingConfig(
    monitor="val/metrics_anomaly/iou", mode="max", patience=5
)
```

**Calibration Takes Too Long** -- Subsample data, increase batch size, or use GPU:

```python
indices = torch.randperm(len(calib_data))[:10000]
calib_data = calib_data[indices]
datamodule = Cu3sDataModule(..., batch_size=32)
pipeline = pipeline.to("cuda")
```
