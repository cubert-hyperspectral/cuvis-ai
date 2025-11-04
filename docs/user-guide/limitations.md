# Known Limitations

This page documents known limitations, constraints, and considerations when using CUVIS.AI.

## Data Dimensions

### Maximum Recommended Sizes

| Dimension | Recommended Max | Hard Limit | Notes |
|-----------|----------------|------------|-------|
| Spatial (H×W) | 512×512 | System RAM | Larger sizes require chunked processing |
| Spectral Channels | 200 | ~500 | Performance degrades with >200 channels |
| Batch Size | 16 | GPU VRAM | Depends on cube dimensions and GPU memory |
| Dataset Size | 10,000 samples | Disk space | Larger datasets supported with streaming |

### Memory Requirements

Approximate GPU memory requirements for training:

| Cube Size | Batch Size | GPU Memory |
|-----------|-----------|------------|
| 64×64×50 | 16 | 4GB |
| 128×128×100 | 8 | 8GB |
| 256×256×100 | 4 | 16GB |
| 512×512×200 | 1 | 24GB+ |

## Performance Considerations

### Statistical Initialization

- RX detector covariance computation is O(C²) where C = channels
- For C > 200, initialization may take several minutes
- MinMaxNormalizer requires full dataset pass (not mini-batch friendly)

### Gradient Training

- PCA orthogonality loss is O(K²) where K = components
- Channel selector entropy computation is O(C) per batch
- Visualization generation adds 10-20% overhead when enabled

## Unsupported Features

### Data Types

- [ ] Non-hyperspectral data (RGB images, point clouds)
- [ ] Temporal/video hyperspectral sequences
- [ ] Multi-modal data (e.g., hyperspectral + LiDAR)
- [x] 3D hyperspectral cubes (H×W×C)
- [x] 4D batch tensors (B×C×H×W or B×H×W×C)

### Training Modes

- [ ] Semi-supervised learning
- [ ] Self-supervised pretraining
- [ ] Online/continual learning
- [x] Supervised learning with labels
- [x] Unsupervised anomaly detection

### Distributed Training

- [x] Single-GPU training
- [x] Multi-GPU DDP (tested up to 4 GPUs)
- [ ] Multi-node training (experimental, not fully tested)
- [ ] TPU training

## Edge Cases

### Empty Batches

**Issue**: Some operations fail with empty batches (e.g., batch_size=0).

**Workaround**: Ensure dataloader always returns non-empty batches.

### Missing Labels

**Issue**: Some loss/metric nodes require labels but datamodule doesn't provide them.

**Workaround**: Add dummy labels or skip those leaves.

### Extreme Values

**Issue**: Very large or small values can cause numerical instability.

**Workaround**: 
- Use normalization (MinMaxNormalizer, StandardNormalizer)
- Enable gradient clipping
- Use mixed precision training

## Platform-Specific Issues

### Windows

- [ ] `num_workers > 0` may cause issues with multiprocessing
- **Workaround**: Set `num_workers=0` or use `torch.utils.data.DataLoader` with `persistent_workers=True`

### macOS (Apple Silicon)

- CUDA not available, use MPS backend
- **Workaround**: Set `accelerator="mps"` in TrainerConfig

### Linux

- Best supported platform
- CUDA works out-of-the-box

## Monitoring Backends

### WandB

- Requires API key and internet connection
- Offline mode available but limited functionality
- Team/org features require paid plan

### TensorBoard

- Real-time updates may lag with high logging frequency
- Large runs can produce GBs of logs
- No built-in experiment comparison

### DummyMonitor

- Always available, no external dependencies
- Manual analysis required (JSONL + PKL files)
- No web UI

## Workarounds

### Large Datasets

For datasets that don't fit in memory:

```python
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    def __iter__(self):
        # Stream data from disk/network
        pass
```

### High-Dimensional Data

For >200 channels:

1. **Preprocessing**: Reduce channels before training
2. **Soft Selector**: Let the model select relevant channels
3. **PCA Pretraining**: Apply PCA offline, then train on reduced data

### Limited GPU Memory

```python
# Use smaller batch size
datamodule = LentilsAnomaly(batch_size=2)

# Enable gradient checkpointing (if supported)
config = TrainingConfig(
    trainer=TrainerConfig(
        precision="16-mixed",  # Mixed precision
        gradient_clip_val=1.0,  # Prevent exploding gradients
    )
)

# Reduce visualization frequency
viz = AnomalyHeatmap(log_frequency=10)  # Log every 10 steps
```

## Planned Improvements

Future releases will address:

- [x] Phase 5: Stress testing with large datasets
- ⏳ Phase 6: Streaming data support
- ⏳ Phase 7: Multi-node distributed training
- ⏳ Phase 8: TPU support

## Reporting Issues

If you encounter a limitation not listed here:

1. Check [GitHub Issues](https://github.com/cubert-hyperspectral/cuvis.ai/issues)
2. Search existing issues for similar problems
3. If not found, open a new issue with:
   - System information (OS, Python version, GPU)
   - Minimal reproducible example
   - Error traceback
   - Expected vs actual behavior

## Next Steps

- **[Quickstart](quickstart.md)**: Start with basic examples
- **[Configuration](configuration.md)**: Optimize for your use case
- **[Stress Testing](../reference/stress_testing.md)**: Performance benchmarks
