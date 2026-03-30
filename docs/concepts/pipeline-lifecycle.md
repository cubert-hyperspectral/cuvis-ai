!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Pipeline Lifecycle

*Central orchestrator managing node connections, data flow, and execution through distinct lifecycle phases.*

A **Pipeline** in CUVIS.AI is a directed acyclic graph (DAG) of connected nodes that processes data through transformations. It manages connections, validates ports, executes nodes in topological order, handles initialization, and serializes trained models.

**Key capabilities:**

* Port-based connections with validation
* Topological execution ordering
* Statistical initialization and gradient training
* Serialization and restoration
* Stage-aware execution filtering
* Introspection and debugging tools

---

## Pipeline States

```mermaid
stateDiagram-v2
    [*] --> Construction: create pipeline
    Construction --> Validation: verify()
    Validation --> Initialization: statistical_initialization()
    Initialization --> Execution: forward() / train()
    Execution --> Serialization: save()
    Serialization --> [*]
    Serialization --> Restoration: load()
    Restoration --> Validation
    Execution --> Cleanup: release resources
    Cleanup --> [*]
```

**States:**

1. **Construction**: Add nodes and connections
2. **Validation**: Verify graph integrity and port compatibility
3. **Initialization**: Statistical init, GPU transfer
4. **Execution**: Training, validation, inference
5. **Serialization**: Save structure and weights
6. **Restoration**: Load saved pipelines
7. **Cleanup**: Release resources

---

## Construction Phase

### Creating a Pipeline

```python
from cuvis_ai.pipeline.pipeline import CuvisPipeline

pipeline = CuvisPipeline(
    name="my_anomaly_detector",
    strict_runtime_io_validation=True
)
```

### Adding Nodes

```python
# Explicit addition
pipeline.add_node(data_loader)
pipeline.add_node(normalizer)

# Or automatic during connection
pipeline.connect((data_loader.cube, normalizer.data))
```

### Connecting Nodes

```python
data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
normalizer = MinMaxNormalizer(eps=1e-6)
rx_node = RXGlobal(num_channels=61)

pipeline.connect(
    (data_node.cube, normalizer.data),
    (normalizer.normalized, rx_node.data)
)
```

### Pipeline Builder (YAML)

```python
from cuvis_ai.pipeline.pipeline_builder import PipelineBuilder

builder = PipelineBuilder()
pipeline = builder.build_from_config("configs/pipeline/my_pipeline.yaml")
```

**YAML Format:**
```yaml
metadata:
  name: RX_Anomaly_Detector

nodes:
  - name: data_loader
    class_name: cuvis_ai.node.data.LentilsAnomalyDataNode
    hparams:
      normal_class_ids: [0, 1]

  - name: rx_detector
    class_name: cuvis_ai.anomaly.rx_detector.RXGlobal
    hparams:
      num_channels: 61

connections:
  - source: data_loader.outputs.cube
    target: rx_detector.inputs.data
```

---

## Validation Phase

*Automatic during construction, manual via verify().*

```python
pipeline.verify()  # Check all constraints
```

**Validation Checks:**

```mermaid
flowchart TD
    A[Start] --> B{Required ports<br/>connected?}
    B -->|No| C[ValidationError]
    B -->|Yes| D{Type<br/>compatible?}
    D -->|No| E[PortCompatibilityError]
    D -->|Yes| F{Shape<br/>compatible?}
    F -->|No| E
    F -->|Yes| G{No cycles<br/>in graph?}
    G -->|Cycle| H[CycleError]
    G -->|No cycle| I{Config<br/>valid?}
    I -->|No| J[ConfigError]
    I -->|Yes| K[✓ Valid]
    style K fill:#d4edda
    style C fill:#f8d7da
    style E fill:#f8d7da
    style H fill:#f8d7da
```


---

## Initialization Phase

*Prepare nodes for execution.*

### Node Initialization Order

```python
# Nodes initialized in topological order
sorted_nodes = pipeline._get_topologically_sorted_nodes()

for node in sorted_nodes:
    if node.requires_initial_fit:
        node.statistical_initialization(initialization_stream)
```

### Two-Phase Initialization

```python
from cuvis_ai.trainer.statistical_trainer import StatisticalTrainer

# Phase 1: Statistical initialization
datamodule = SingleCu3sDataModule(...)
stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
stat_trainer.fit()  # Initialize statistical nodes

# Phase 2: Unfreeze for gradient training
pipeline.unfreeze_nodes_by_name(["rx_detector", "selector"])
```

See [Two-Phase Training](two-phase-training.md) for details.

### GPU Transfer

```python
pipeline = pipeline.to("cuda")
```

---

## Execution Phase

### Sequential Execution

```python
from cuvis_ai_core.pipeline.context import Context, ExecutionStage

context = Context(stage=ExecutionStage.INFERENCE)
outputs = pipeline.forward(
    batch={"cube": hyperspectral_data},
    context=context
)

anomaly_scores = outputs[("rx_detector", "scores")]
```

### Batch Processing

```python
results = []
for batch in dataloader:
    outputs = pipeline.forward(batch=batch, context=context)
    results.append(outputs[("rx_detector", "scores")])

all_scores = torch.cat(results, dim=0)
```

### Trainer-Managed Execution

```python
from cuvis_ai.trainer.gradient_trainer import GradientTrainer

trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[bce_loss],
    metric_nodes=[metrics_node],
    trainer_config=training_config
)

trainer.fit()
trainer.test()
```

### Execution Flow

**High-Level Sequence:**

```mermaid
sequenceDiagram
    User->>Pipeline: forward(batch, context)
    Pipeline->>Pipeline: Topological sort
    Pipeline->>Pipeline: Filter by stage
    loop For each node
        Pipeline->>Node: Gather inputs
        Pipeline->>Node: forward(**inputs)
        Node-->>Pipeline: Return outputs
        Pipeline->>Pipeline: Store in port_data
    end
    Pipeline-->>User: Return results
```

**Detailed Forward Pass Mechanics:**

This diagram shows the internal port-based routing and validation that occurs during `pipeline.forward()`:

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 20, 'rankSpacing': 30}} }%%
flowchart TD
    A["pipeline.forward(batch=...)"] --> B[Port-Based Batch Distribution]
    B --> C[Resolve Input Ports]
    C --> D[Topological Sort by Port Connections]
    D --> E[For each node in order]
    E --> F[Collect inputs from connected output ports]
    F --> G[Execute node.forward(**inputs)]
    G --> H[Store outputs in port dictionary]
    H --> I{More nodes?}
    I -->|Yes| E
    I -->|No| J[Return port output dictionary]

    B --> K[Port Validation]
    K --> L[Type Checking]
    K --> M[Shape Compatibility]
    K --> N[Stage Filtering]

    style A fill:#e1f5ff
    style J fill:#d4edda
    style K fill:#fff3cd
```

Outputs are returned as a dictionary keyed by `(node_name, port_name)`:

```python
outputs = pipeline.forward(batch={"cube": data})
scores = outputs[("rx_detector", "scores")]
```

### Stage-Aware Execution

```python
# Different nodes execute per stage
context = Context(stage=ExecutionStage.TRAIN)
pipeline.forward(batch=data, context=context)  # All nodes

context = Context(stage=ExecutionStage.VAL)
pipeline.forward(batch=data, context=context)  # Only VAL/ALWAYS nodes

context = Context(stage=ExecutionStage.INFERENCE)
pipeline.forward(batch=data, context=context)  # Only INFERENCE/ALWAYS nodes
```

### Partial Execution

```python
# Run only up to normalizer (debugging)
outputs = pipeline.forward(
    batch={"cube": data},
    context=context,
    upto_node=normalizer
)
```

---

## Serialization Phase

*Save pipeline structure and trained weights.*

### Saving Pipelines

```python
from cuvis_ai.pipeline.config import PipelineMetadata

pipeline.save_to_file(
    config_path="outputs/my_pipeline.yaml",
    metadata=PipelineMetadata(
        name="RX_Anomaly_Detector_v1",
        description="Trained on Lentils dataset",
        tags=["anomaly-detection", "production"]
    ),
    validate_nodes=True,
    include_optimizer=False,
    include_scheduler=False
)

# Generates: outputs/my_pipeline.yaml, outputs/my_pipeline.pt
```

Generates a YAML config (structure, node hparams, connections) and a `.pt` checkpoint (state_dict with trained weights/statistics, metadata). Training data, intermediate activations, and Python code are **not** saved.

---

## Restoration Phase

*Load saved pipelines and resume work.*

### Loading Pipelines

```python
pipeline = CuvisPipeline.load_from_file(
    config_path="outputs/my_pipeline.yaml",
    weights_path="outputs/my_pipeline.pt",  # Auto-detected if None
    device="cuda",
    strict_weight_loading=True
)

outputs = pipeline.forward(batch=test_data)
```

### With Plugins

```python
registry = NodeRegistry()
registry.load_plugins("plugins.yaml")

builder = PipelineBuilder(node_registry=registry)
pipeline = builder.build_from_config("outputs/my_pipeline.yaml")
pipeline.load_state_dict(torch.load("outputs/my_pipeline.pt")["state_dict"])
```

---

## Cleanup Phase

*Release resources (GPU memory, file handles, caches).*

```python
pipeline.cleanup()

# Or context manager (automatic)
with CuvisPipeline.load_from_file("pipeline.yaml") as pipeline:
    results = pipeline.forward(batch=data)
# Auto-cleanup on exit

# Manual resource release
pipeline = pipeline.to("cpu")
torch.cuda.empty_cache()
```

---

## Training Workflow

### Statistical Training Only

```python
from cuvis_ai.trainer.statistical_trainer import StatisticalTrainer

stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
stat_trainer.fit()
stat_trainer.validate()
stat_trainer.test()

pipeline.save_to_file("outputs/statistical_pipeline.yaml")
```

### Gradient Training (Two-Phase)

```python
from cuvis_ai.trainer.gradient_trainer import GradientTrainer

# Phase 1: Statistical init (if needed)
if any(node.requires_initial_fit for node in pipeline.nodes):
    stat_trainer = StatisticalTrainer(pipeline, datamodule)
    stat_trainer.fit()

# Phase 2: Unfreeze and train
pipeline.unfreeze_nodes_by_name(["selector", "rx_detector"])

grad_trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[bce_loss],
    metric_nodes=[metrics_node],
    trainer_config=training_config,
    optimizer_config=optimizer_config
)

grad_trainer.fit()
grad_trainer.test()

pipeline.save_to_file("outputs/gradient_trained_pipeline.yaml")
```

---

## Monitoring & Debugging

### Pipeline Introspection

```python
summary = pipeline.summary()
print(summary)

input_specs = pipeline.get_input_specs()
output_specs = pipeline.get_output_specs()
```

### Visualization

```python
pipeline.visualize(output_path="pipeline_graph.png", format="png")
dot_source = pipeline.to_dot()
```

### Execution Profiling

```python
pipeline.set_profiling(enabled=True, skip_first_n=3)

# Run through Predictor or GradientTrainer for best results
predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
predictor.predict(max_batches=100)

print(pipeline.format_profiling_summary(total_frames=100))
```

For details, see [Profiling & Performance](../how-to/profiling.md).

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
pipeline.set_log_level(logging.DEBUG)

outputs = pipeline.forward(batch=data)
```

---

## Best Practices

| Practice | How |
|----------|-----|
| Validate early | `pipeline.verify()` before training |
| Check init state | Inspect `node.requires_initial_fit` and `node._statistically_initialized` |
| Save checkpoints | `pipeline.save_to_file(f"checkpoints/epoch_{epoch+1}.yaml")` periodically |
| Use context managers | `with CuvisPipeline.load_from_file(...) as pipeline:` for auto-cleanup |
| Monitor performance | `pipeline.set_profiling(enabled=True)` -- see [Profiling](../how-to/profiling.md) |
| Version pipelines | Pass `PipelineMetadata(version="2.1.0", tags=[...])` to `save_to_file()` |

---

???+ warning "Common Pitfalls"

    **Forgetting Statistical Initialization** -- Call `stat_trainer.fit()` before gradient training.

    ```python
    stat_trainer.fit()       # Required first!
    grad_trainer.fit()
    ```

    **Type Mismatches** -- All connected nodes must use the same dtype.

    ```python
    normalizer = MinMaxNormalizer(dtype=torch.float32)
    rx = RXGlobal(dtype=torch.float64)  # Mismatch!
    ```

    **Cyclic Dependencies** -- Pipeline graph must be a DAG. Connecting `node_c` back to `node_a` raises `CycleError`.

    **Memory Leaks** -- Use context managers when loading pipelines in a loop:

    ```python
    for i in range(1000):
        with CuvisPipeline.load_from_file("pipeline.yaml") as pipeline:
            pipeline.forward(batch=data)
    ```
