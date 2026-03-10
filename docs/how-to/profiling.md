# Profiling & Performance

Enable per-node runtime profiling to identify bottlenecks in your pipeline.

---

## Overview

CUVIS.AI includes **opt-in, manual profiling** that wraps each `node.forward()` call with high-resolution timers (`time.perf_counter_ns()`). Profiling is configured on the pipeline object and works transparently with both `Predictor` (inference) and `GradientTrainer` (training), since both call `pipeline.forward()` internally.

**Key characteristics:**

- **Zero overhead** when disabled (single boolean check per node)
- **Cumulative** — stats accumulate across all `pipeline.forward()` calls until explicit reset
- **Per-node, per-stage** — timings are keyed by `(execution_stage, node_name)`
- **Constant memory** — online Welford mean/std + P² approximate median, no sample history stored
- **Thread-safe** — safe for concurrent gRPC requests on the same session

---

## Quick Start

```python
from cuvis_ai_core.training.predictor import Predictor

# 1. Enable profiling on the pipeline
pipeline.set_profiling(enabled=True, skip_first_n=3)

# 2. Run your workload through Predictor or GradientTrainer
predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
predictor.predict(max_batches=350)

# 3. Print the formatted summary
print(pipeline.format_profiling_summary(total_frames=350))
```

Example output:

```
Profiling Summary (350 frames, skip_first_n=3)
Node                                     Stage        Count   Mean(ms)    Std(ms)    Min(ms)    Max(ms) Median(ms)   Total(s)
-----------------------------------------------------------------------------------------------------------------------------
sam3_tracker                             inference      347     895.08     145.43     487.49    1747.95     891.04    310.591
tracking_coco_json                       inference      347      11.60       2.47       7.63      20.38      11.18      4.025
overlay                                  inference      347      10.23       2.19       6.27      25.60       9.68      3.551
to_video                                 inference      347       7.76       2.14       5.82      43.55       7.67      2.694
video_frame                              inference      347       0.01       0.00       0.01       0.03       0.01      0.005
-----------------------------------------------------------------------------------------------------------------------------
TOTAL                                                                                                                320.867
Average per-frame pipeline time: 924.69 ms (1.1 FPS)
```

---

## Profiling During Inference (Predictor)

```python
from cuvis_ai_core.training.predictor import Predictor

# Enable profiling before creating the Predictor
pipeline.set_profiling(
    enabled=True,
    synchronize_cuda=(device == "cuda"),
    skip_first_n=3,
)

predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
predictor.predict(max_batches=350)

# Retrieve and display results
print(pipeline.format_profiling_summary(total_frames=350))
```

`Predictor` calls `pipeline.forward(context=Context(stage=INFERENCE))` for each batch, so all node timings accumulate under the `"inference"` stage.

!!! tip "Use Predictor, not raw pipeline.forward()"
    `Predictor` handles batch iteration, device transfer, node reset/close, and progress bars.
    Running profiling through `Predictor` gives you realistic end-to-end timing that includes
    proper warm-up and teardown behavior.

---

## Profiling During Training (GradientTrainer)

```python
from cuvis_ai_core.training.trainers import GradientTrainer
from cuvis_ai_schemas.enums import ExecutionStage

# Enable profiling before training
pipeline.set_profiling(enabled=True, skip_first_n=5)

trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[loss_node],
)
trainer.fit()

# View training stage timings
print(pipeline.format_profiling_summary(stage=ExecutionStage.TRAIN))

# View validation stage timings
print(pipeline.format_profiling_summary(stage=ExecutionStage.VAL))

# View all stages combined
print(pipeline.format_profiling_summary())
```

`GradientTrainer` calls `pipeline.forward()` with `TRAIN`, `VAL`, or `TEST` execution stages depending on the training phase. Stats are accumulated per `(stage, node_name)` pair, so you can filter by stage to compare training vs validation performance.

!!! note "`skip_first_n` applies per accumulator key"
    Each `(stage, node_name)` pair has its own skip counter. If you set `skip_first_n=5`,
    the first 5 training forward passes **and** the first 5 validation forward passes are
    each skipped independently.

---

## API Reference

### `pipeline.set_profiling()`

```python
pipeline.set_profiling(
    enabled: bool,
    *,
    synchronize_cuda: bool = False,
    reset: bool = False,
    skip_first_n: int = 0,
)
```

Configure profiling with **full-replace semantics** — every call fully specifies the configuration. Omitted keyword arguments receive their defaults.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | — | Activate or deactivate profiling |
| `synchronize_cuda` | `bool` | `False` | Call `torch.cuda.synchronize` before/after each `node.forward()` for accurate GPU wall-clock timing |
| `reset` | `bool` | `False` | Discard all previously accumulated statistics |
| `skip_first_n` | `int` | `0` | Number of initial samples per node to discard (warm-up skip). Must be >= 0 |

### `pipeline.get_profiling_summary()`

```python
pipeline.get_profiling_summary(
    stage: ExecutionStage | None = None,
) -> list[NodeProfilingStats]
```

Return accumulated profiling stats as a list of frozen `NodeProfilingStats` dataclasses. Pass `stage` to filter by execution stage, or `None` for all stages.

### `pipeline.format_profiling_summary()`

```python
pipeline.format_profiling_summary(
    stage: ExecutionStage | None = None,
    *,
    total_frames: int | None = None,
) -> str
```

Convenience method that calls `get_profiling_summary()` and formats the result as a text table ready for logging or printing.

### `pipeline.reset_profiling()`

Clear all accumulated profiling statistics.

### `pipeline.profiling_enabled`

Read-only property returning whether profiling is currently active.

### `NodeProfilingStats` dataclass

Each entry in the profiling summary contains:

| Field | Type | Description |
|-------|------|-------------|
| `node_name` | `str` | Unique node name within the pipeline |
| `stage` | `str` | Execution stage (e.g. `"inference"`, `"train"`) |
| `count` | `int` | Number of recorded samples (after skip) |
| `mean_ms` | `float` | Mean execution time in milliseconds |
| `median_ms` | `float` | Approximate median (P² estimator) |
| `std_ms` | `float` | Population standard deviation |
| `min_ms` | `float` | Minimum execution time |
| `max_ms` | `float` | Maximum execution time |
| `total_ms` | `float` | Total accumulated time |
| `last_ms` | `float` | Most recent sample |

---

## Understanding the Output

| Column | Meaning |
|--------|---------|
| **Node** | The `node.name` — unique within the pipeline, assigned by `CuvisPipeline` |
| **Stage** | Execution stage (`inference`, `train`, `val`, `test`) |
| **Count** | Number of `node.forward()` calls recorded (after `skip_first_n`) |
| **Mean(ms)** | Average execution time per call |
| **Std(ms)** | Population standard deviation across all calls |
| **Min/Max(ms)** | Fastest and slowest individual calls |
| **Median(ms)** | Approximate median via P² estimator (constant memory) |
| **Total(s)** | Cumulative wall-clock time for this node (in seconds) |

The **TOTAL** row sums all nodes' total times. The **FPS** line divides total pipeline time by the first node's count to estimate per-frame throughput.

---

## gRPC Profiling

Profiling can also be controlled remotely via gRPC. See the [gRPC API Reference](../grpc/api-reference.md#profiling) for details on:

- **`SetProfiling`** — enable, disable, or reconfigure profiling on a session
- **`GetProfilingSummary`** — retrieve per-node profiling statistics

```python
# Example: gRPC client enabling profiling
stub.SetProfiling(
    cuvis_ai_pb2.SetProfilingRequest(
        session_id=session_id,
        enabled=True,
        synchronize_cuda=True,
        skip_first_n=3,
    )
)

# Run inference...

# Retrieve profiling summary
response = stub.GetProfilingSummary(
    cuvis_ai_pb2.GetProfilingSummaryRequest(
        session_id=session_id,
        stage=cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE,
    )
)
for stat in response.node_stats:
    print(f"{stat.node_name}: {stat.mean_ms:.2f} ms ({stat.count} calls)")
```

---

## Tips

!!! tip "Use Predictor / GradientTrainer for best estimates"
    Always run profiling through the standard orchestrators rather than calling
    `pipeline.forward()` directly. They handle device transfer, batch iteration,
    and node lifecycle correctly, giving you realistic timing.

!!! tip "Warm-up skip for CUDA pipelines"
    Use `skip_first_n=3` or higher for CUDA pipelines. The first few forward passes
    include JIT compilation and CUDA kernel caching, which inflate timings significantly.

!!! tip "CUDA synchronization trade-off"
    `synchronize_cuda=True` gives accurate GPU wall-clock times by forcing
    `torch.cuda.synchronize()` before and after each node. This adds overhead and
    disables CUDA kernel pipelining — use it for profiling, not production.

!!! tip "Cumulative stats and reset"
    Stats accumulate across all forward calls (including multiple `predict()` runs)
    until you explicitly call `pipeline.reset_profiling()` or `set_profiling(reset=True)`.
    This is useful for aggregating across a full dataset.

!!! tip "Stage filtering"
    Use the `stage` parameter to compare performance across execution stages:
    ```python
    train_summary = pipeline.format_profiling_summary(stage=ExecutionStage.TRAIN)
    val_summary = pipeline.format_profiling_summary(stage=ExecutionStage.VAL)
    ```
