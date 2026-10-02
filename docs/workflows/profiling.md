# Profiling & Performance

Enable per-node runtime profiling to identify bottlenecks in your pipeline, and see how much of each
batch goes to loading the data rather than to the nodes.

---

## Overview

Cuvis.AI includes **opt-in, manual profiling** that wraps each `node.forward()` call with high-resolution timers (`time.perf_counter_ns()`). Profiling is configured on the pipeline object and works transparently with both `Predictor` (inference) and `GradientTrainer` (training), since both call `pipeline.forward()` internally. Since cuvis-ai-core 0.18.1, `restore-pipeline` and `Predictor` also time the batch loop around the nodes: the fetch from the DataModule, the copy to the device and the whole iteration (see [Profiling the data load](#profiling-the-data-load)).

**Key characteristics:**

- **Zero overhead** when disabled (single boolean check per node)
- **Cumulative** — stats accumulate across all `pipeline.forward()` calls until explicit reset
- **Per-node, per-stage** — timings are keyed by `(execution_stage, node_name)`
- **Data loading included** (cuvis-ai-core 0.18.1) — `restore-pipeline` and `Predictor` feed their batches through `CuvisPipeline.iter_profiled_batches()`, which records `data_load`, `to_device` and `batch_loop` per stage next to the node timings
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

Runs that go through `restore-pipeline` or `Predictor` (cuvis-ai-core 0.18.1) print a `Data loading (outside the nodes)` block after the node table. The block below comes from a separate, synthetic run (two normalizer nodes, a simulated 20 ms fetch, 50 batches of shape `[1, 64, 64, 61]` on the CPU), so its numbers say nothing about the table above:

```text
Data loading (outside the nodes)
Step                                     Stage        Count   Mean(ms)    Std(ms)    Min(ms)    Max(ms) Median(ms)   Total(s)
-----------------------------------------------------------------------------------------------------------------------------
data_load                                inference       46      20.96       0.23      20.67      21.93      20.88      0.964
to_device                                inference       46       0.02       0.00       0.02       0.03       0.02      0.001
batch_loop                               inference       46      21.90       0.41      21.44      23.43      21.82      1.007
-----------------------------------------------------------------------------------------------------------------------------
First batch data load (inference, excluded from the rows): 0.03 s
Time per batch (inference, batch_loop, host wall time): 21.90 ms (45.7 batches/s)
```

The node rows of that run counted 47 samples and the data rows 46: the first iteration of a pass is left out of the data rows on top of `skip_first_n`, because its fetch carries one-time setup (its time is the "First batch data load" line). On a Jetson AGX Thor reading 1000x1080x61 cu3s frames, the same block showed `data_load` at about 98 ms against a 102 ms batch loop (9.8 batches/s) and a 14.6 s first batch: the recording read, not the nodes, set the frame rate there.

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

`Predictor` calls `pipeline.forward(context=Context(stage=INFERENCE))` for each batch, so all node timings accumulate under the `"inference"` stage. `predict(max_batches=)` stops fetching after that many batches (the limit is applied before the fetch, so the loader never reads one batch past it).

!!! tip "Use Predictor, not raw pipeline.forward()"
    `Predictor` handles batch iteration, device transfer, node reset/close, and progress bars.
    Running profiling through `Predictor` gives you realistic end-to-end timing that includes
    proper warm-up and teardown behavior.

---

## Profiling the data load

`restore-pipeline` (with `--data-module`) and `Predictor` drive their batch loop through `CuvisPipeline.iter_profiled_batches()` (cuvis-ai-core 0.18.1). While profiling is enabled, every iteration after the first records three samples under the run's stage:

| Step | What it measures |
|------|------------------|
| `data_load` | The fetch from the DataModule: file read, processing, collate |
| `to_device` | The `move` call that copies the batch to the pipeline's device (absent when no `move` is given) |
| `batch_loop` | From the start of the fetch until the consumer asks for the next batch: the whole iteration, nodes, orchestration and output handling included |

The first iteration of every pass is kept out of the rows because its fetch carries one-time setup (opening the recording, warming caches); its fetch time appears on the "First batch data load" line instead. `skip_first_n` applies to these rows as it does to the nodes. The per-batch line reports the `batch_loop` mean in batches per second, labelled **host wall time** unless the run synchronized CUDA.

From the command line, every data run prints the block at the end; add `--profile-sync` to synchronize CUDA around every profiled step, so the node, device-copy and per-batch times are GPU-complete instead of host wall time (slower, and it disables kernel pipelining):

```bash
restore-pipeline --pipeline-path my_pipeline.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --data-arg cu3s_file_path=recording.cu3s --profile-sync
```

In your own loop, wrap the raw DataLoader (never a progress bar: a bar's update would count as `batch_loop` time) and close the generator in a `finally`, because a loop sample only closes when the next batch is requested:

```python
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

pipeline.set_profiling(enabled=True, skip_first_n=3)


def to_device(batch: dict) -> dict:
    """Move the batch's tensors to the pipeline's device; other values pass through."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


batches = pipeline.iter_profiled_batches(dataloader, stage=ExecutionStage.INFERENCE, move=to_device)
global_step = 0
try:
    with torch.no_grad():
        for batch in batches:
            context = Context(
                stage=ExecutionStage.INFERENCE, batch_idx=global_step, global_step=global_step
            )
            pipeline.forward(batch=batch, context=context)
            global_step += 1
finally:
    batches.close()  # the caller owns the generator; an open loop sample is discarded

print(pipeline.format_profiling_summary(stage=ExecutionStage.INFERENCE, total_frames=global_step))
```

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

Return accumulated profiling stats as a list of frozen `NodeProfilingStats` dataclasses. Pass `stage` to filter by execution stage, or `None` for all stages. The list holds node rows only.

### `pipeline.get_data_profiling_summary()`

```python
pipeline.get_data_profiling_summary(
    stage: ExecutionStage | None = None,
) -> list[NodeProfilingStats]
```

Return the data-loading rows recorded by `iter_profiled_batches()` (cuvis-ai-core 0.18.1), one `NodeProfilingStats` per step and stage with `node_name` set to `data_load`, `to_device` or `batch_loop`. Empty when no batch loop was profiled yet.

### `pipeline.iter_profiled_batches()`

```python
pipeline.iter_profiled_batches(
    batches: Iterable[dict[str, Any]],
    *,
    stage: ExecutionStage,
    move: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]
```

Yield the batches (moved through `move` when given) and time the loop around them while profiling is enabled; see [Profiling the data load](#profiling-the-data-load). With `synchronize_cuda=True` the device is synchronized after the copy and before each loop sample closes, when the moved batch holds CUDA tensors.

### `pipeline.format_profiling_summary()`

```python
pipeline.format_profiling_summary(
    stage: ExecutionStage | None = None,
    *,
    total_frames: int | None = None,
) -> str
```

Convenience method that calls `get_profiling_summary()` and `get_data_profiling_summary()` and formats both as a text table ready for logging or printing (the data block only appears when it holds samples). The underlying `cuvis_ai_core.pipeline.profiling.format_profiling_table()` takes the rows as `stats=` and `data_stats=`, the excluded first fetches as `first_batch_ms=` and `cuda_synchronized=` for the per-batch line's label.

### `pipeline.reset_profiling()`

Clear all accumulated profiling statistics, the data-loading rows and the first-batch times included.

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

The `Data loading (outside the nodes)` block (cuvis-ai-core 0.18.1) uses the same columns with a **Step** instead of a node name:

| Line | Meaning |
|------|---------|
| **data_load** | Time to fetch one batch from the DataModule (read, processing, collate) |
| **to_device** | Time of the `move` call that copies the batch to the pipeline's device |
| **batch_loop** | Time from the start of a fetch until the next batch is requested: the whole iteration |
| **First batch data load** | The fetch of the first iteration of each pass, kept out of the rows because it carries one-time setup |
| **Time per batch** | The `batch_loop` mean in batches per second, labelled `host wall time` or `CUDA-synchronized` |

Nothing is summed across stages or between the two blocks; compare `data_load` with the nodes' TOTAL to see whether the loader or the nodes set the frame rate.

---

## gRPC Profiling

Profiling can also be controlled remotely via gRPC. See the [gRPC API Reference](../deployment/api/training-inference.md#profiling) for details on:

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
    Without it, queued GPU work can show up in the next batch's `data_load`;
    `restore-pipeline --profile-sync` is the command-line switch for the same thing.

!!! tip "Loader-bound or node-bound?"
    When `data_load` is close to `batch_loop` and the nodes' TOTAL is small, the DataModule
    sets the frame rate: look at the recording format, processing mode and `num_workers`
    before touching the nodes. A large "First batch data load" is setup cost (opening the
    recording, warming caches), not steady-state throughput.

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
