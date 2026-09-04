Status: Needs Review

This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

______________________________________________________________________

# Pipeline Lifecycle

*Central orchestrator managing node connections, data flow, and execution through distinct lifecycle phases.*

A **Pipeline** in Cuvis.AI is a directed acyclic graph (DAG) of connected nodes that processes data through transformations. It manages connections, validates ports, executes nodes in topological order, handles initialization, and serializes trained models.

**Key capabilities:**

- Port-based connections with validation
- Topological execution ordering
- Statistical initialization and gradient training
- Serialization and restoration
- Stage-aware execution filtering
- Introspection and debugging tools

______________________________________________________________________

## Pipeline States

```
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
1. **Validation**: Verify graph integrity and port compatibility
1. **Initialization**: Statistical init, GPU transfer
1. **Execution**: Training, validation, inference
1. **Serialization**: Save structure and weights
1. **Restoration**: Load saved pipelines
1. **Cleanup**: Release resources

______________________________________________________________________

## Construction Phase

### Creating a Pipeline

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pipeline = CuvisPipeline(
    name="my_anomaly_detector",
    strict_runtime_io_validation=True
)
```

### Adding Nodes

```python
# Nodes are added automatically the first time you connect their ports
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
from cuvis_ai_core.pipeline.factory import PipelineBuilder

import yaml

with open("cuvis_ai/configs/pipeline/my_pipeline.yaml") as f:
    config = yaml.safe_load(f)

builder = PipelineBuilder()
pipeline = builder.build_from_config(config)
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
    class_name: cuvis_ai.node.anomaly.rx_detector.RXGlobal
    hparams:
      num_channels: 61

connections:
  - source: data_loader.outputs.cube
    target: rx_detector.inputs.data
```

______________________________________________________________________

## Validation Phase

*Automatic during construction, manual via verify().*

```python
pipeline.verify()  # Check all constraints
```

**Validation Checks:**

```
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
    classDef success fill:#d4edda,stroke:#28a745,color:#155724
    classDef error fill:#f8d7da,stroke:#dc3545,color:#721c24
    class K success
    class C,E,H error
```

______________________________________________________________________

## Initialization Phase

*Prepare nodes for execution.*

### Node Initialization Order

```python
# Nodes are initialized in topological order. StatisticalTrainer does this for
# you; the resolved order is the internal cached `pipeline._sorted_nodes` property.
sorted_nodes = pipeline._sorted_nodes

for node in sorted_nodes:
    if node.requires_initial_fit:
        node.statistical_initialization(initialization_stream)
```

### Two-Phase Initialization

```python
from cuvis_ai_core.training import StatisticalTrainer

# Phase 1: Statistical initialization
datamodule = Cu3sDataModule(...)
stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
stat_trainer.fit()  # Initialize statistical nodes

# Phase 2: Unfreeze for gradient training
pipeline.unfreeze_nodes_by_name(["rx_detector", "selector"])
```

See [Two-Phase Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.15.1/concepts/training/index.md) for details.

### GPU Transfer

```python
pipeline = pipeline.to("cuda")
```

______________________________________________________________________

## Execution Phase

### Sequential Execution

```python
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.enums import ExecutionStage

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
from cuvis_ai_core.training import GradientTrainer

trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[bce_loss],
    metric_nodes=[metrics_node],
    training_config=training_config
)

trainer.fit()
trainer.test()
```

### Execution Flow

**High-Level Sequence:**

```
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

```
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

______________________________________________________________________

## Serialization Phase

*Save pipeline structure and trained weights.*

### Saving Pipelines

```python
from cuvis_ai_schemas.pipeline import PipelineMetadata

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

______________________________________________________________________

## Restoration Phase

*Load saved pipelines and resume work.*

### Loading Pipelines

```python
pipeline = CuvisPipeline.load_pipeline(
    config_path="outputs/my_pipeline.yaml",
    weights_path="outputs/my_pipeline.pt",  # Optional; None loads structure only
    device="cuda",
    strict_weight_loading=True,
)

outputs = pipeline.forward(batch=test_data)
```

### With Plugins

When a pipeline declares plugins (a top-level `plugins: [<name>, ...]` list), restore it with the directory that holds those manifests:

```python
from cuvis_ai_core.utils import restore_pipeline

pipeline = restore_pipeline(
    pipeline_path="outputs/my_pipeline.yaml",
    plugins_dirs=["cuvis_ai/configs/plugins"],   # dir holding the bare-named manifests
)
```

For manual, dev-mode control you can still load a manifest into a registry instance directly:

```python
registry = NodeRegistry()
registry.register_plugin("cuvis_ai/configs/plugins/adaclip.yaml")   # CLI / dev-mode path
pipeline = CuvisPipeline.load_pipeline(
    "outputs/my_pipeline.yaml", node_registry=registry
)
```

______________________________________________________________________

## Cleanup Phase

*Release resources (GPU memory, file handles, caches).*

```python
pipeline.cleanup()

# Manual resource release
pipeline = pipeline.to("cpu")
torch.cuda.empty_cache()
```

______________________________________________________________________

## Training Workflow

### Statistical Training Only

```python
from cuvis_ai_core.training import StatisticalTrainer

stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
stat_trainer.fit()
stat_trainer.validate()
stat_trainer.test()

pipeline.save_to_file("outputs/statistical_pipeline.yaml")
```

### Gradient Training (Two-Phase)

```python
from cuvis_ai_core.training import GradientTrainer

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
    training_config=training_config
)

grad_trainer.fit()
grad_trainer.test()

pipeline.save_to_file("outputs/gradient_trained_pipeline.yaml")
```

______________________________________________________________________

## Monitoring & Debugging

### Pipeline Introspection

```python
# Walk the graph
print(pipeline.name)
for node in pipeline.nodes:
    print(node.name)

input_specs = pipeline.get_input_specs()
output_specs = pipeline.get_output_specs()
```

### Visualization

```python
pipeline.visualize(output_path="pipeline_graph.png", format="render")
```

### Execution Profiling

```python
pipeline.set_profiling(enabled=True, skip_first_n=3)

# Run through Predictor or GradientTrainer for best results
predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
predictor.predict(max_batches=100)

print(pipeline.format_profiling_summary(total_frames=100))
```

For details, see [Profiling & Performance](https://cubert-hyperspectral.github.io/cuvis-ai/0.15.1/workflows/profiling/index.md).

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

outputs = pipeline.forward(batch=data)
```

______________________________________________________________________

## Best Practices

| Practice            | How                                                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Validate early      | `pipeline.verify()` before training                                                                                                            |
| Check init state    | Inspect `node.requires_initial_fit` and `node._statistically_initialized`                                                                      |
| Save checkpoints    | `pipeline.save_to_file(f"checkpoints/epoch_{epoch+1}.yaml")` periodically                                                                      |
| Release resources   | call `pipeline.cleanup()` at terminal teardown (session close / pipeline replacement)                                                          |
| Monitor performance | `pipeline.set_profiling(enabled=True)` -- see [Profiling](https://cubert-hyperspectral.github.io/cuvis-ai/0.15.1/workflows/profiling/index.md) |
| Annotate pipelines  | Pass `PipelineMetadata(description=..., tags=["v2.1.0", ...], author=...)` to `save_to_file()`                                                 |

______________________________________________________________________

Common Pitfalls

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

**Memory Leaks** -- Load the pipeline once and reuse it across iterations; call `cleanup()` when done:

```python
pipeline = CuvisPipeline.load_pipeline("pipeline.yaml")
for i in range(1000):
    pipeline.forward(batch=data)
pipeline.cleanup()
```
