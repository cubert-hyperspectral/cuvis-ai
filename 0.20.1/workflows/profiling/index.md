# Profiling & Performance

Enable per-node runtime profiling to identify bottlenecks in your pipeline, and see how much of each batch goes to loading the data rather than to the nodes.

______________________________________________________________________

## Overview

Cuvis.AI includes **opt-in, manual profiling** that wraps each `node.forward()` call with high-resolution timers (`time.perf_counter_ns()`). Profiling is configured on the pipeline object and works transparently with both `Predictor` (inference) and `GradientTrainer` (training), since both call `pipeline.forward()` internally. Since cuvis-ai-core 0.18.1, `restore-pipeline` and `Predictor` also time the batch loop around the nodes: the fetch from the DataModule, the copy to the device and the whole iteration (see [Profiling the data load](#profiling-the-data-load)).

**Key characteristics:**

- **Zero overhead** when disabled (single boolean check per node)
- **Cumulative** — stats accumulate across all `pipeline.forward()` calls until explicit reset
- **Per-node, per-stage** — timings are keyed by `(execution_stage, node_name)`
- **Data loading included** (cuvis-ai-core 0.18.1) — `restore-pipeline` and `Predictor` feed their batches through `CuvisPipeline.iter_profiled_batches()`, which records `data_load`, `to_device` and `batch_loop` per stage next to the node timings
- **Compute cost included** (cuvis-ai-core 0.18.7): the first forward of each node after a reset also counts its floating-point operations, shown as `GFLOP` and `GFLOP/s` next to the times (see [Compute cost (GFLOP)](#compute-cost-gflop))
- **Constant memory** — online Welford mean/std + P² approximate median, no sample history stored
- **Thread-safe** — safe for concurrent gRPC requests on the same session

______________________________________________________________________

## Quick Start

```python
from cuvis_ai_core.training.predictor import Predictor

# 1. Enable profiling on the pipeline
pipeline.set_profiling(enabled=True, synchronize_cuda=True, skip_first_n=3)

# 2. Run your workload through Predictor or GradientTrainer
predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
predictor.predict(max_batches=10)

# 3. Print the formatted summary
print(pipeline.format_profiling_summary(total_frames=10))
```

Example output, from a lentils anomaly pipeline (a `SpectralAngleMapper` with 16 references on 1000 x 1080 x 61 frames of a cu3s recording, CUDA-synchronized, on an RTX 5070 Ti Laptop GPU with cuvis-ai-core 0.18.7):

```text
Profiling Summary (10 frames, skip_first_n=3)
Node                                     Stage        Count   Mean(ms)    Std(ms)    Min(ms)    Max(ms) Median(ms)   Total(s)      GFLOP    GFLOP/s
---------------------------------------------------------------------------------------------------------------------------------------------------
area_filter                              inference        7      17.50       0.71      16.81      19.08      17.22      0.123      0.000       0.00
validity                                 inference        7       8.54       1.77       6.65      11.13       8.35      0.060      0.000       0.00
spectral_angle                           inference        7       7.39       1.38       6.26      10.07       7.57      0.052      2.108     285.15
unit_normalize                           inference        7       4.70       0.68       4.02       5.79       4.48      0.033      0.000       0.00
calibrator                               inference        7       0.65       0.21       0.45       1.10       0.55      0.005      0.000       0.00
cube_source                              inference        7       0.16       0.03       0.13       0.22       0.14      0.001      0.000       0.00
references                               inference        7       0.12       0.02       0.10       0.15       0.10      0.001      0.000       0.00
---------------------------------------------------------------------------------------------------------------------------------------------------
TOTAL                                                                                                                   0.273
Average per-frame pipeline time: 39.07 ms (25.6 FPS), 2.108 GFLOP per forward (53.96 GFLOP/s)
```

The `GFLOP` and `GFLOP/s` columns are the node's compute cost (cuvis-ai-core 0.18.7, see [Compute cost (GFLOP)](#compute-cost-gflop)): the spectral angle node does 2.108 GFLOP of matrix multiplication per frame, the area filter and the validity check are elementwise and count 0, and the per-frame line sums the counts.

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

______________________________________________________________________

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

Use Predictor, not raw pipeline.forward()

`Predictor` handles batch iteration, device transfer, node reset/close, and progress bars. Running profiling through `Predictor` gives you realistic end-to-end timing that includes proper warm-up and teardown behavior.

______________________________________________________________________

## Profiling the data load

`restore-pipeline` (with `--data-module`) and `Predictor` drive their batch loop through `CuvisPipeline.iter_profiled_batches()` (cuvis-ai-core 0.18.1). While profiling is enabled, every iteration after the first records three samples under the run's stage:

| Step         | What it measures                                                                                                                               |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_load`  | The fetch from the DataModule: file read, processing, collate                                                                                  |
| `to_device`  | The `move` call that copies the batch to the pipeline's device (absent when no `move` is given)                                                |
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

______________________________________________________________________

## Compute cost (GFLOP)

Since cuvis-ai-core 0.18.7 the profiler also counts how much compute each node does, so a slow node can be told apart from a heavy one. While profiling is enabled, the first forward of every `(stage, node)` after a profiling reset runs under a counting mode that adds up the floating-point operations of its matrix multiplications, convolutions and attention blocks. The count is stored per node as `NodeProfilingStats.flops_per_call`, and `gflop_per_s` divides it by the node's mean time. The text table shows both as `GFLOP` and `GFLOP/s` columns whenever a node has a count, and the per-frame line adds the sum per forward when all rows belong to one stage:

```text
Profiling Summary (3 frames)
Node                                     Stage        Count   Mean(ms)    Std(ms)    Min(ms)    Max(ms) Median(ms)   Total(s)      GFLOP    GFLOP/s
---------------------------------------------------------------------------------------------------------------------------------------------------
dinomaly_detector                        inference        3     131.27     150.57      23.14     344.21      26.47      0.394    306.978    2338.46
cir_selector                             inference        3      64.07      86.92       2.34     187.00       2.87      0.192      0.000       0.00
MinMaxNormalizer                         inference        3      26.38      29.31       4.23      67.80       7.12      0.079      0.000       0.00
cube_source                              inference        3       0.51       0.41       0.21       1.10       0.23      0.002      0.000       0.00
---------------------------------------------------------------------------------------------------------------------------------------------------
TOTAL                                                                                                                   0.667
Average per-frame pipeline time: 222.24 ms (4.5 FPS), 306.978 GFLOP per forward (1381.31 GFLOP/s)
The first forward of each node also counted its FLOPs and runs slower; skip_first_n >= 1 leaves it out of the times.
```

The run above is a Dinomaly detector (DINOv2 ViT-B/14 encoder and an 8-layer decoder on a 448 x 448 false-colour image, bf16 autocast) on three 1000 x 1080 x 61 frames from a cu3s recording, CUDA-synchronized, on an RTX 5070 Ti Laptop GPU. The detector's 307 GFLOP per forward are its matrix multiplications and attention; the normalizer and the band selector are elementwise and count 0. With `skip_first_n=0` the mean of every node includes the first forward, which paid for the kernel warm-up and the count (344 ms against 26 ms afterwards), so the `GFLOP/s` column understates the steady state; `skip_first_n=3` on a longer run gives the rate of the warm frames.

What counts: `mm`, `addmm`, `bmm`, `matmul`, `linear`, `einsum`, every convolution and every scaled-dot-product attention op (any batch layout, grouped-query attention included), on the CPU as on the GPU and under `torch.no_grad` as under `torch.inference_mode`. Everything else counts 0: elementwise ops and reductions, normalisations, interpolation, `linalg` solves and inverses, numpy and OpenCV code, TensorRT and ONNX engines, and work a node hands to another thread. Compiled regions (`torch.compile`) run fused kernels the counter does not see, so a compiled node can count low; whether it does depends on how much of its counted first forward still runs op by op (the Dinomaly detector above, compiled with `mode="reduce-overhead"`, counted the same 307 GFLOP as in eager mode). A heavy node can therefore read 0: a diagonal Gaussian mixture scored per pixel with squares, divisions and sums takes 100 ms on the same frames and counts nothing, while a spectral angle mapper with 16 references counts its 2.108 GFLOP (2 x 1000 x 1080 pixels x 16 x 61 bands) exactly. A node that runs a profiled sub-pipeline during its counted forward includes the sub-pipeline's FLOPs in its own count. The data-loading rows carry no count.

The counted forward computes exactly what an uncounted one does: every op runs unchanged, the counter only reads shapes. It costs about 10 us per op plus one shape-only decomposition per distinct composite call (cached), so with `skip_first_n=0` the first sample of each node is a few milliseconds slower than the rest, and the table says so under the node block. With `skip_first_n >= 1` that forward stays out of the timing rows as before; the sample counts never change. A count keeps its first forward's value: a node whose work depends on the frame (a tracker with a varying object count) reports the cost of the frame it was counted on. When two forwards on one session run at the same time, the first to finish gets the count; a forward that raises leaves the count to the next one, and a reset during a counted forward discards its result.

GFLOP/s is a throughput only when the time under it is the node's own. For GPU nodes that needs `synchronize_cuda=True` (`restore-pipeline --profile-sync`); otherwise the node's time is the host's queueing time, the rate reads high, and the table prints `Node times are host wall time (no CUDA synchronize): GFLOP/s of GPU nodes reads high.` under the node block.

A formula that fails on an input leaves that node's count empty (an empty `GFLOP` cell, `flops_per_call` is `None`) and logs one warning naming the node and the op. `CUVIS_PROFILE_FLOPS=0` (also `false`, `no` or `off`) in the environment turns the counting off for every pipeline of the process while the timing stays on; the variable is read whenever profiling is switched on. A server that another program starts, such as the backend CuvisNEXT launches, inherits that program's environment, so set the variable before starting the program (on Windows as a user or system environment variable) when a plugin node misbehaves under the counting mode.

______________________________________________________________________

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

`skip_first_n` applies per accumulator key

Each `(stage, node_name)` pair has its own skip counter. If you set `skip_first_n=5`, the first 5 training forward passes **and** the first 5 validation forward passes are each skipped independently.

______________________________________________________________________

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

| Parameter          | Type   | Default | Description                                                                                         |
| ------------------ | ------ | ------- | --------------------------------------------------------------------------------------------------- |
| `enabled`          | `bool` | —       | Activate or deactivate profiling                                                                    |
| `synchronize_cuda` | `bool` | `False` | Call `torch.cuda.synchronize` before/after each `node.forward()` for accurate GPU wall-clock timing |
| `reset`            | `bool` | `False` | Discard all previously accumulated statistics                                                       |
| `skip_first_n`     | `int`  | `0`     | Number of initial samples per node to discard (warm-up skip). Must be >= 0                          |

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

| Field            | Type            | Description                                                                                                                                                                                       |
| ---------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `node_name`      | `str`           | Unique node name within the pipeline                                                                                                                                                              |
| `stage`          | `str`           | Execution stage (e.g. `"inference"`, `"train"`)                                                                                                                                                   |
| `count`          | `int`           | Number of recorded samples (after skip)                                                                                                                                                           |
| `mean_ms`        | `float`         | Mean execution time in milliseconds                                                                                                                                                               |
| `median_ms`      | `float`         | Approximate median (P² estimator)                                                                                                                                                                 |
| `std_ms`         | `float`         | Population standard deviation                                                                                                                                                                     |
| `min_ms`         | `float`         | Minimum execution time                                                                                                                                                                            |
| `max_ms`         | `float`         | Maximum execution time                                                                                                                                                                            |
| `total_ms`       | `float`         | Total accumulated time                                                                                                                                                                            |
| `last_ms`        | `float`         | Most recent sample                                                                                                                                                                                |
| `flops_per_call` | `int \| None`   | Floating-point operations of one forward (matrix multiplication, convolution, attention), counted on the first forward after a reset (cuvis-ai-core 0.18.7); `None` when the node was not counted |
| `gflop_per_s`    | `float \| None` | Property: `flops_per_call / (mean_ms * 1e6)`; `None` without a count or a positive mean                                                                                                           |

______________________________________________________________________

## Understanding the Output

| Column          | Meaning                                                                                                                                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Node**        | The `node.name` — unique within the pipeline, assigned by `CuvisPipeline`                                                                                                                                                          |
| **Stage**       | Execution stage (`inference`, `train`, `val`, `test`)                                                                                                                                                                              |
| **Count**       | Number of `node.forward()` calls recorded (after `skip_first_n`)                                                                                                                                                                   |
| **Mean(ms)**    | Average execution time per call                                                                                                                                                                                                    |
| **Std(ms)**     | Population standard deviation across all calls                                                                                                                                                                                     |
| **Min/Max(ms)** | Fastest and slowest individual calls                                                                                                                                                                                               |
| **Median(ms)**  | Approximate median via P² estimator (constant memory)                                                                                                                                                                              |
| **Total(s)**    | Cumulative wall-clock time for this node (in seconds)                                                                                                                                                                              |
| **GFLOP**       | Floating-point operations of one forward of the node, in billions, counted on its first forward after a reset (cuvis-ai-core 0.18.7); the column appears when any node has a count and the cell stays empty for a node without one |
| **GFLOP/s**     | `GFLOP` divided by `Mean(ms)`; a throughput only on a CUDA-synchronized run                                                                                                                                                        |

The **TOTAL** row sums all nodes' total times. The **FPS** line divides total pipeline time by the first node's count to estimate per-frame throughput; when all rows belong to one stage it also carries the sum of the counts per forward, the rate that follows from it, and `N node(s) not counted` when rows without a count exist. Two notes can follow the node block: one when the run was not CUDA-synchronized (the GFLOP/s of GPU nodes then reads high) and one when `skip_first_n` is 0 (the counted first forward is part of the times).

The `Data loading (outside the nodes)` block (cuvis-ai-core 0.18.1) uses the same columns with a **Step** instead of a node name:

| Line                      | Meaning                                                                                               |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| **data_load**             | Time to fetch one batch from the DataModule (read, processing, collate)                               |
| **to_device**             | Time of the `move` call that copies the batch to the pipeline's device                                |
| **batch_loop**            | Time from the start of a fetch until the next batch is requested: the whole iteration                 |
| **First batch data load** | The fetch of the first iteration of each pass, kept out of the rows because it carries one-time setup |
| **Time per batch**        | The `batch_loop` mean in batches per second, labelled `host wall time` or `CUDA-synchronized`         |

Nothing is summed across stages or between the two blocks; compare `data_load` with the nodes' TOTAL to see whether the loader or the nodes set the frame rate.

______________________________________________________________________

## gRPC Profiling

Profiling can also be controlled remotely via gRPC. See the [gRPC API Reference](https://docs.cuvis.ai/latest/deployment/api/training-inference/#profiling) for details on:

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
    line = f"{stat.node_name}: {stat.mean_ms:.2f} ms ({stat.count} calls)"
    if stat.HasField("flops_per_call"):  # presence means counted (cuvis-ai-core 0.18.7)
        line += f", {stat.flops_per_call / 1e9:.3f} GFLOP"
    print(line)
```

______________________________________________________________________

## Tips

Use Predictor / GradientTrainer for best estimates

Always run profiling through the standard orchestrators rather than calling `pipeline.forward()` directly. They handle device transfer, batch iteration, and node lifecycle correctly, giving you realistic timing.

Warm-up skip for CUDA pipelines

Use `skip_first_n=3` or higher for CUDA pipelines. The first few forward passes include JIT compilation and CUDA kernel caching, which inflate timings significantly.

CUDA synchronization trade-off

`synchronize_cuda=True` gives accurate GPU wall-clock times by forcing `torch.cuda.synchronize()` before and after each node. This adds overhead and disables CUDA kernel pipelining — use it for profiling, not production. Without it, queued GPU work can show up in the next batch's `data_load`; `restore-pipeline --profile-sync` is the command-line switch for the same thing.

Loader-bound or node-bound?

When `data_load` is close to `batch_loop` and the nodes' TOTAL is small, the DataModule sets the frame rate: look at the recording format, processing mode and `num_workers` before touching the nodes. A large "First batch data load" is setup cost (opening the recording, warming caches), not steady-state throughput.

Heavy or slow?

Read `GFLOP` and `Mean(ms)` together. A node with a large count and a time to match is compute-bound; a node with a small count and a long time waits for something else (a CPU-side loop, a device copy, a kernel launch per pixel). `GFLOP/s` on a synchronized run puts the first kind against the device's peak rate.

Cumulative stats and reset

Stats accumulate across all forward calls (including multiple `predict()` runs) until you explicitly call `pipeline.reset_profiling()` or `set_profiling(reset=True)`. This is useful for aggregating across a full dataset.

Stage filtering

Use the `stage` parameter to compare performance across execution stages:

```python
train_summary = pipeline.format_profiling_summary(stage=ExecutionStage.TRAIN)
val_summary = pipeline.format_profiling_summary(stage=ExecutionStage.VAL)
```
