# GPU Acceleration Guide

This guide explains when and how to effectively use GPU acceleration with cuvis.ai training pipelines.

## Understanding GPU Performance

GPU acceleration is **not always beneficial**. GPUs excel at parallel computation on large batches of data, but for small workloads, the overhead of transferring data between CPU and GPU can outweigh the computational benefits.

## When GPU Acceleration Helps

### ✓Use GPU When:

1. **Large datasets** (>500 samples)
2. **Large batch sizes** (≥16, preferably 32+)
3. **High-dimensional data** (100+ channels, 256×256+ spatial resolution)
4. **Complex models** (multiple trainable layers, many parameters)
5. **Many training epochs** (10+ epochs of gradient-based training)
6. **Gradient-intensive operations** (multiple loss terms, complex backpropagation)

### ❌ GPU May Not Help When:

1. **Small datasets** (<100 samples)
2. **Small batch sizes** (<8)
3. **Low-dimensional data** (<20 channels, <128×128 spatial)
4. **Simple models** (few trainable parameters)
5. **Statistical-only training** (Phase 1 initialization is CPU-bound)
6. **CPU-dominant operations** (data loading, augmentation, statistical computations)

## Performance Characteristics

### Statistical Initialization (Phase 1)

The statistical initialization phase runs on **CPU regardless of accelerator choice**. This includes:
- Computing running statistics (mean, variance, min, max)
- PCA initialization via eigenvalue decomposition
- Covariance matrix computation

**Impact:** For small datasets or training configs with many statistical operations, Phase 1 can dominate total training time, making GPU acceleration less impactful.

### Gradient Training (Phase 2)

GPU acceleration primarily benefits the gradient-based training phase:
- Forward passes through the graph
- Backpropagation and gradient computation
- Optimizer steps

**Impact:** GPU shows most benefit when gradient training is the dominant phase (many epochs, large batches).

## Real-World Example

From our stress tests with NVIDIA GeForce RTX 5070 Ti:

### Small Workload (200 samples, 128×128, 50 channels, batch=16)
```
CPU time:  64.91s
GPU time:  67.01s
Speedup:   0.97x (GPU slightly slower!)

Breakdown:
- Statistical init: ~25s (same on CPU and GPU)
- Gradient training: ~40s (CPU) vs ~42s (GPU)
```

**Analysis:** Statistical initialization dominates. Data transfer overhead makes GPU slightly slower.

### Large Workload (1000+ samples, 256×256, 100+ channels, batch=32)
```
CPU time:  ~300s
GPU time:  ~120s
Speedup:   2.5x (GPU much faster!)

Breakdown:
- Statistical init: ~30s (same on CPU and GPU)
- Gradient training: ~270s (CPU) vs ~90s (GPU)
```

**Analysis:** Gradient training dominates. GPU's parallel processing provides significant speedup.

## Optimization Tips

### 1. Increase Batch Size

GPU performance scales with batch size due to improved parallelism:

```python
# Poor GPU utilization
batch_size = 4  # Too small

# Better GPU utilization
batch_size = 16  # Good starting point

# Optimal GPU utilization (if memory allows)
batch_size = 32  # or higher
```

### 2. Use pin_memory for DataLoader

Enable `pin_memory` when using GPU to speed up data transfer:

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # Faster CPU->GPU transfer
    num_workers=4,    # Parallel data loading
)
```

### 3. Reduce Data Transfer

Minimize CPU-GPU data transfers by:
- Keeping data on GPU between operations
- Batching operations together
- Avoiding unnecessary .cpu() or .cuda() calls

### 4. Profile Your Code

Use PyTorch profiler to identify bottlenecks:

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    # Your training code
    trainer.fit(model, datamodule)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Hardware Considerations

### GPU Memory

Monitor GPU memory usage to avoid out-of-memory errors:

```python
import torch

# Check available GPU memory
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

### CPU Performance

Modern CPUs with AVX/AVX2/AVX-512 SIMD instructions can be very efficient for small workloads:
- High-end desktop CPUs (e.g., Intel i9, AMD Ryzen 9) may match or exceed entry-level GPUs for small models
- Server CPUs with many cores can handle batch processing efficiently

### GPU Performance Tiers

Expected speedup ranges by GPU tier (vs high-end CPU):

| GPU Tier | Example GPUs | Expected Speedup | Sweet Spot |
|----------|--------------|------------------|------------|
| Entry-level | GTX 1650, RTX 3050 | 1.2-1.8x | batch_size ≥ 16 |
| Mid-range | RTX 3060, RTX 4060 | 1.8-3x | batch_size ≥ 16 |
| High-end | RTX 4070 Ti, RTX 4080 | 2.5-4x | batch_size ≥ 32 |
| Professional | A100, H100 | 3-8x | batch_size ≥ 64 |

## Configuration Example

Example training config for optimal GPU utilization:

```python
from cuvis_ai.training.config import TrainingConfig, TrainerConfig

config = TrainingConfig(
    trainer=TrainerConfig(
        max_epochs=10,               # More epochs = more GPU benefit
        accelerator='gpu',           # Use GPU
        devices=1,                   # Single GPU
        precision='16-mixed',        # Mixed precision for speed
        enable_progress_bar=True,
    ),
    optimizer=OptimizerConfig(
        name='adam',
        lr=0.001,
    ),
)

# DataModule with GPU-optimized settings
datamodule = GraphDataModule(
    train_dataset=dataset,
    batch_size=32,          # Large batch for GPU
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster CPU->GPU transfer
)
```

## Measuring Performance

Always benchmark both CPU and GPU for your specific workload:

```python
import time
import torch

# CPU benchmark
start = time.time()
trainer_cpu = graph.train(
    datamodule=datamodule,
    training_config=config_cpu
)
cpu_time = time.time() - start

# GPU benchmark
torch.cuda.empty_cache()
start = time.time()
trainer_gpu = graph.train(
    datamodule=datamodule,
    training_config=config_gpu
)
torch.cuda.synchronize()
gpu_time = time.time() - start

print(f"Speedup: {cpu_time / gpu_time:.2f}x")
```

## Conclusion

GPU acceleration can provide significant speedups for large-scale training, but it's not a universal solution. Profile your specific workload, consider your hardware, and optimize your configuration for best results.

**Rule of thumb:** If your dataset has <200 samples or batch_size <16, start with CPU training. For larger workloads, GPU training will likely provide meaningful speedup.
