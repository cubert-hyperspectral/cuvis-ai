!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# AdaCLIP Workflow: Plugin-Based Anomaly Detection

Learn how to use plugin nodes with AdaCLIP for hyperspectral anomaly detection, comparing three dimensionality reduction approaches.

---

## Overview

This tutorial demonstrates **plugin-based anomaly detection** using the AdaCLIP plugin, which adapts CLIP (Contrastive Language-Image Pre-training) for hyperspectral imagery. You'll learn three approaches to reducing 61 hyperspectral channels to 3 RGB-compatible channels for AdaCLIP processing.

**What You'll Learn:**
- Loading external plugin nodes from Git repositories
- **PCA Baseline** - Statistical-only frozen reduction (fastest)
- **DRCNN Mixer** - Learnable continuous mixing with gradient optimization
- **Concrete Selector** - Learnable discrete band selection
- Performance comparison and visualization strategies
- TensorBoard monitoring of end-to-end processing

**Time to Complete:** 35-40 minutes (all 3 variants)

**Prerequisites:**
- Completion of [Channel Selector Tutorial](channel-selector.md) - Understanding two-phase training
- Familiarity with [Plugin System Overview](../plugin-system/overview.md)
- Python 3.10+, PyTorch 2.0+, CUDA-capable GPU (recommended)

---

## Background

### AdaCLIP for Hyperspectral Anomaly Detection

**AdaCLIP** adapts OpenAI's CLIP vision-language model for hyperspectral anomaly detection. CLIP was trained on millions of image-text pairs and excels at zero-shot visual understanding, making it powerful for detecting anomalies without extensive task-specific training.

**Key Challenge:** CLIP expects 3-channel RGB images, but hyperspectral data has 60+ channels. **Solution:** Learn an optimal mapping from hyperspectral → RGB that preserves anomaly-relevant information.

**Three Approaches Compared:**

| Approach | Type | Training | Speed | Flexibility | Best For |
|----------|------|----------|-------|-------------|----------|
| **PCA Baseline** | Linear projection | Statistical-only | ⚡ Fastest | ❌ Frozen | Quick baseline, comparison |
| **DRCNN Mixer** | Multi-layer convolution | Two-phase (stat + grad) | 🔥 Moderate | ✅ Continuous mixing | End-to-end optimization |
| **Concrete Selector** | Gumbel-Softmax sampling | Gradient-only | 🔥 Moderate | ✅ Discrete selection | Interpretable band choices |

### Plugin System Integration

Unlike built-in nodes, **plugin nodes** are loaded dynamically from external Git repositories. This allows:
- **Modularity** - Keep experimental/specialized nodes separate from core framework
- **Versioning** - Pin to specific plugin releases (tags) for reproducibility
- **Community Extensions** - Share custom nodes without modifying core codebase

See [Plugin System Usage](../plugin-system/usage.md) for installation details.

---

<a id="approach-1-pca-baseline"></a>
<a id="variant-1-pca-baseline"></a>

## Approach 1: PCA Baseline (Statistical-Only)

### When to Use PCA Baseline

- **Quick evaluation** - No gradient training required
- **Baseline comparison** - Measure improvement of learnable methods
- **CPU-friendly** - Statistical SVD decomposition is efficient
- **Interpretable** - Principal components explain variance

**Limitations:** Frozen after initialization, may not capture task-specific anomaly features.

### Step 1: Load AdaCLIP Plugin

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

# Load plugin from GitHub repository
registry = NodeRegistry()
registry.load_plugin(
    name="adaclip",
    config={
        "repo": "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git",
        "tag": "v0.1.0",  # Pin to specific version
        "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"],
    },
)

# Get the AdaCLIPDetector class
AdaCLIPDetector = NodeRegistry.get("cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector")
```

**For local development:**
```python
registry.load_plugin(
    name="adaclip",
    config={
        "path": "/path/to/local/cuvis-ai-adaclip",
        "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
    }
)
```

### Step 2: Build PCA Pipeline

```python
from cuvis_ai.node.dimensionality_reduction import TrainablePCA
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider

# Data and preprocessing
data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
normalizer = MinMaxNormalizer(eps=1e-6, use_running_stats=True, name="hsi_normalizer")

# PCA: 61 → 3 channels (frozen after initialization)
pca = TrainablePCA(
    n_components=3,          # RGB compatibility
    whiten=False,            # Don't whiten - normalize separately
    init_method="svd",       # SVD decomposition
    eps=1e-6,
    name="pca_baseline",
)

# Normalize PCA output to [0, 1] for AdaCLIP
pca_normalizer = MinMaxNormalizer(
    eps=1e-6,
    use_running_stats=False,  # Per-image normalization
    name="pca_output_normalizer",
)

# AdaCLIP detector (FROZEN)
adaclip = AdaCLIPDetector(
    weight_name="pretrained_all",
    backbone="ViT-L-14-336",
    prompt_text="normal: lentils, anomaly: stones",
    image_size=518,
    prompting_depth=4,
    prompting_length=5,
    gaussian_sigma=4.0,
    use_half_precision=True,
    enable_warmup=True,
    enable_gradients=False,  # No gradients needed
    name="adaclip",
)

# Decision threshold
decider = QuantileBinaryDecider(quantile=0.995, name="decider")
```

### Step 3: Connect Pipeline

```python
pipeline.connect(
    # Preprocessing: HSI → Normalizer → PCA → Normalize → AdaCLIP
    (data_node.outputs.cube, normalizer.data),
    (normalizer.normalized, pca.data),
    (pca.projected, pca_normalizer.data),
    (pca_normalizer.normalized, adaclip.rgb_image),

    # Inference: Scores → Decider → Metrics
    (adaclip.scores, decider.logits),
    (adaclip.scores, metrics_node.logits),
    (decider.decisions, metrics_node.decisions),
    (data_node.outputs.mask, metrics_node.targets),

    # Monitoring
    (metrics_node.metrics, tensorboard_node.metrics),
)
```

### Step 4: Statistical Training

```python
from cuvis_ai_core.training import StatisticalTrainer

# Initialize PCA and normalizer
stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
stat_trainer.fit()

# No gradient training - PCA is frozen baseline
logger.info("PCA baseline complete - no gradient training")

# Evaluate
stat_trainer.validate()
stat_trainer.test()
```

### Step 5: Analyze PCA Results

```python
# Check explained variance
with torch.no_grad():
    sample_batch = next(iter(train_loader))
    sample_cube = sample_batch["cube"].float()
    normalizer_output = normalizer.forward(data=sample_cube)
    pca_output = pca.forward(data=normalizer_output["normalized"])

    ev_ratio = pca_output.get("explained_variance_ratio", torch.zeros(3))
    logger.info(f"Explained variance ratio: {ev_ratio.tolist()}")
    logger.info(f"Total explained variance: {ev_ratio.sum().item():.4f}")
```

**Expected Output:**
```
Explained variance ratio: [0.4523, 0.2891, 0.1456]
Total explained variance: 0.8870
```

**Interpretation:** First 3 principal components capture ~89% of spectral variance, but may miss subtle anomaly-specific patterns.

---

<a id="variant-2-drcnn-mixer"></a>

## Approach 2: DRCNN Mixer (Learnable Continuous Mixing)

### When to Use DRCNN Mixer

- **End-to-end optimization** - Learn task-specific channel mixing
- **Continuous blending** - Smooth combinations of spectral bands
- **Gradient-based refinement** - Optimize directly for IoU or detection loss
- **Multi-layer reduction** - Gradual dimensionality decrease

**Based on:** Zeegers et al., "Task-Driven Learned Hyperspectral Data Reduction Using End-to-End Supervised Deep Learning," *J. Imaging* 6(12):132, 2020.

### Step 1: Build DRCNN Pipeline

```python
from cuvis_ai.node.channel_mixer import LearnableChannelMixer
from cuvis_ai.node.losses import IoULoss

# Data preprocessing
data_node = LentilsAnomalyDataNode(
    normal_class_ids=[0, 1],
    anomaly_class_ids=[3],  # Only Stone as anomaly for IoU loss
)
normalizer = MinMaxNormalizer(eps=1e-6, use_running_stats=True)

# DRCNN-style channel mixer: 61 → 3 channels
mixer = LearnableChannelMixer(
    input_channels=61,
    output_channels=3,              # RGB compatibility
    leaky_relu_negative_slope=0.1,
    use_bias=True,
    use_activation=True,
    normalize_output=True,          # Per-image min-max → [0, 1]
    init_method="pca",              # Initialize with PCA (requires statistical fit)
    eps=1e-6,
    reduction_scheme=[61, 16, 8, 3],  # Multi-layer gradual reduction
    name="channel_mixer",
)

# AdaCLIP detector (FROZEN, but gradients flow through)
adaclip = AdaCLIPDetector(
    weight_name="pretrained_all",
    backbone="ViT-L-14-336",
    prompt_text="",  # Empty for zero-shot
    image_size=518,
    prompting_depth=4,
    prompting_length=5,
    gaussian_sigma=4.0,
    use_half_precision=False,
    enable_warmup=False,
    enable_gradients=True,  # CRITICAL: Allow gradients to flow
    name="adaclip",
)

# IoU loss (differentiable, works on continuous scores)
iou_loss = IoULoss(
    weight=1.0,
    smooth=1e-6,
    normalize_method="minmax",  # Preserve dynamic range
    name="iou_loss",
)
```

### Step 2: Two-Phase Training

**Phase 1: Statistical Initialization**
```python
if mixer.requires_initial_fit:
    logger.info("Phase 1: Statistical initialization (PCA)...")
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()
else:
    logger.info("Phase 1: Skipping (using weight init)")
```

**Phase 2: Unfreeze Mixer**
```python
logger.info("Phase 2: Unfreezing channel mixer...")
pipeline.unfreeze_nodes_by_name([mixer.name])
logger.info("AdaClip remains frozen (enable_gradients=True allows gradient flow)")
```

**Phase 3: Gradient Training**
```python
from cuvis_ai_core.training import GradientTrainer
from cuvis_ai_core.training.config import ModelCheckpointConfig, SchedulerConfig

training_cfg = TrainingConfig.from_dict(cfg.training)
training_cfg.trainer.callbacks.checkpoint = ModelCheckpointConfig(
    dirpath=str(output_dir / "checkpoints"),
    monitor="metrics_anomaly/iou",
    mode="max",
    save_top_k=3,
    save_last=True,
)

training_cfg.scheduler = SchedulerConfig(
    name="reduce_on_plateau",
    monitor="metrics_anomaly/iou",
    mode="max",
    factor=0.5,
    patience=5,
)

grad_trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[iou_loss],
    metric_nodes=[metrics_node],
    trainer_config=training_cfg.trainer,
    optimizer_config=training_cfg.optimizer,
    monitors=[tensorboard_node],
)
grad_trainer.fit()
```

### Step 3: Analyze Mixer Weights

```python
# Before training
initial_weights = {}
for name, param in mixer.named_parameters():
    initial_weights[name] = param.data.clone()
    logger.info(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

# After training
for name, param in mixer.named_parameters():
    diff = (param.data - initial_weights[name]).abs()
    logger.info(f"{name} change: max_diff={diff.max().item():.6f}")
```

### Step 4: Visualize Processing Pipeline

```python
from cuvis_ai.node.pipeline_visualization import PipelineComparisonVisualizer

# TensorBoard visualization node
drcnn_tb_viz = PipelineComparisonVisualizer(
    hsi_channels=[0, 20, 40],  # False-color RGB visualization
    max_samples=4,
    log_every_n_batches=1,
    name="drcnn_tensorboard_viz",
)

# Connect visualization
pipeline.connect(
    (data_node.outputs.cube, drcnn_tb_viz.hsi_cube),
    (mixer.rgb, drcnn_tb_viz.mixer_output),
    (data_node.outputs.mask, drcnn_tb_viz.ground_truth_mask),
    (adaclip.scores, drcnn_tb_viz.adaclip_scores),
    (drcnn_tb_viz.artifacts, tensorboard_node.artifacts),
)
```

**TensorBoard will show:**
- HSI input (false-color RGB using channels 0, 20, 40)
- Mixer output (learned 3-channel representation)
- Ground truth masks
- AdaCLIP anomaly score heatmaps

---

<a id="variant-3-concrete-selector"></a>
<a id="concrete-band-selector"></a>

## Approach 3: Concrete Selector (Learnable Discrete Selection)

### When to Use Concrete Selector

- **Interpretable results** - Exact band indices selected
- **Discrete choices** - Pick 3 specific bands (not blended)
- **Temperature annealing** - Gumbel-Softmax for differentiable sampling
- **No statistical initialization** - Pure gradient-based learning

**Technical Note:** Uses Gumbel-Softmax (Concrete distribution) for differentiable discrete sampling during training, then hard argmax at inference.

### Step 1: Build Concrete Pipeline

```python
from cuvis_ai.node.channel_mixer import ConcreteChannelMixer
from cuvis_ai.node.losses import DistinctnessLoss
from cuvis_ai.deciders.two_stage_decider import TwoStageBinaryDecider

# Data preprocessing
data_node = LentilsAnomalyDataNode(
    normal_class_ids=[0, 1],
    anomaly_class_ids=[3],
)
normalizer = MinMaxNormalizer(eps=1e-6, use_running_stats=True)

# Concrete band selector: 61 → 3 channels
selector = ConcreteChannelMixer(
    input_channels=61,
    output_channels=3,
    tau_start=10.0,                # Initial temperature (soft)
    tau_end=0.1,                   # Final temperature (hard)
    max_epochs=50,                 # Epochs for annealing schedule
    use_hard_inference=True,       # Use argmax at inference
    eps=1e-6,
    name="concrete_selector",
)

# AdaCLIP detector (same as DRCNN)
adaclip = AdaCLIPDetector(
    weight_name="pretrained_all",
    backbone="ViT-L-14-336",
    prompt_text="",
    image_size=518,
    prompting_depth=4,
    prompting_length=5,
    gaussian_sigma=4.0,
    use_half_precision=False,
    enable_warmup=False,
    enable_gradients=True,
    name="adaclip",
)

# Dual loss: IoU + Distinctness
iou_loss = IoULoss(
    weight=1.0,
    smooth=1e-6,
    normalize_method="minmax",
    name="iou_loss",
)

# Distinctness loss prevents all channels selecting the same band
distinctness_loss = DistinctnessLoss(
    weight=0.1,
    name="distinctness_loss",
)

# Two-stage decider (more sophisticated than quantile)
decider = TwoStageBinaryDecider(
    image_threshold=0.20,
    top_k_fraction=0.001,
    quantile=0.995,
    name="decider",
)
```

### Step 2: Gradient Training (No Statistical Phase)

```python
# Phase 1: Skip statistical (weight init only)
logger.info("Phase 1: Skipping statistical - Concrete uses weight init")

# Phase 2: Unfreeze selector
logger.info("Phase 2: Unfreezing Concrete selector...")
pipeline.unfreeze_nodes_by_name([selector.name])

# Phase 3: Gradient training with dual loss
grad_trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=[iou_loss, distinctness_loss],  # Dual loss
    metric_nodes=[metrics_node],
    trainer_config=training_cfg.trainer,
    optimizer_config=training_cfg.optimizer,
    monitors=[tensorboard_node],
)
grad_trainer.fit()
```

### Step 3: Analyze Selected Bands

```python
# Before training
selector.eval()
with torch.no_grad():
    S_initial = selector.get_selection_weights(deterministic=True)
    bands_initial = selector.get_selected_bands()
    tau_initial = selector._get_tau(epoch=0)

logger.info(f"Initial selected bands (argmax): {bands_initial.tolist()}")
logger.info(f"Initial temperature (epoch 0): {tau_initial:.4f}")

# Print top-3 bands per channel
for c in range(selector.output_channels):
    top3 = torch.topk(S_initial[c], k=3)
    logger.info(f"Channel {c} top-3 bands: {top3.indices.tolist()} "
                f"(weights: {top3.values.tolist()})")
```

**After Training:**
```python
with torch.no_grad():
    S_final = selector.get_selection_weights(deterministic=True)
    bands_final = selector.get_selected_bands()

logger.info(f"Final selected bands (argmax): {bands_final.tolist()}")

# Check for band collapse
unique_bands = torch.unique(bands_final).numel()
if unique_bands < selector.output_channels:
    logger.warning(f"⚠️ Only {unique_bands} unique bands selected!")
else:
    logger.info(f"✅ All {selector.output_channels} channels selected different bands")
```

**Expected Output:**
```
Initial selected bands (argmax): [12, 45, 58]
Final selected bands (argmax): [8, 31, 54]
✅ All 3 channels selected different bands
```

### Step 4: Temperature Annealing Visualization

The temperature τ controls the "sharpness" of the Gumbel-Softmax distribution:
- **High τ (10.0):** Soft, continuous sampling (exploration)
- **Low τ (0.1):** Hard, discrete sampling (exploitation)

**Annealing Schedule:**
```python
def _get_tau(self, epoch: int) -> float:
    """Linear temperature annealing from tau_start to tau_end."""
    if epoch >= self.max_epochs:
        return self.tau_end
    progress = epoch / self.max_epochs
    return self.tau_start + (self.tau_end - self.tau_start) * progress
```

Monitor temperature in TensorBoard to verify annealing.

---

## Performance Comparison

### Quantitative Metrics

Run all three approaches and compare:

| Metric | PCA Baseline | DRCNN Mixer | Concrete Selector |
|--------|--------------|-------------|-------------------|
| **Val IoU** | 0.6823 | 0.7456 | 0.7389 |
| **Test IoU** | 0.6791 | 0.7512 | 0.7401 |
| **Precision** | 0.7234 | 0.8012 | 0.7956 |
| **Recall** | 0.8456 | 0.8723 | 0.8689 |
| **F1 Score** | 0.7801 | 0.8345 | 0.8301 |
| **Training Time** | 2 min (stat only) | 15 min (stat + grad) | 12 min (grad only) |
| **Inference Speed** | ⚡ 45 FPS | 🔥 42 FPS | 🔥 43 FPS |

*Example metrics - actual results depend on dataset and hyperparameters*

### Qualitative Comparison

**PCA Baseline:**
- ✅ Fastest to train
- ✅ Interpretable (variance-based)
- ❌ Task-agnostic (not optimized for anomalies)
- ❌ Linear projection only

**DRCNN Mixer:**
- ✅ Best quantitative performance
- ✅ End-to-end optimized
- ✅ Continuous channel blending
- ❌ Harder to interpret (weighted combinations)
- ❌ Requires statistical initialization

**Concrete Selector:**
- ✅ Interpretable selected bands
- ✅ No statistical phase needed
- ✅ Discrete, sparse selection
- ❌ Requires careful temperature tuning
- ❌ Risk of band collapse (mitigated by distinctness loss)

---

## Practical Workflow

### Step-by-Step Execution

**1. Start with PCA Baseline:**
```bash
# Quick baseline (2 minutes)
python examples/adaclip/pca_adaclip_baseline.py
```

**2. Try DRCNN Mixer:**
```bash
# Best performance (15 minutes)
python examples/adaclip/drcnn_adaclip_gradient_training.py
```

**3. Experiment with Concrete Selector:**
```bash
# Interpretable bands (12 minutes)
python examples/adaclip/concrete_adaclip_gradient_training.py
```

### Configuration Management

All three approaches use Hydra configs in `configs/trainrun/`:

```yaml
# configs/trainrun/pca_adaclip_baseline.yaml
name: pca_adaclip_baseline
output_dir: outputs/pca_adaclip

pipeline:
  normalizer:
    eps: 1.0e-6
    use_running_stats: true
  pca:
    eps: 1.0e-6
  adaclip:
    weight_name: pretrained_all
    backbone: ViT-L-14-336
    use_half_precision: true

data:
  data_dir: data/lentils
  batch_size: 4
  num_workers: 2
```

### TensorBoard Comparison

Launch TensorBoard to compare all three:
```bash
tensorboard --logdir=outputs/ --port=6006
```

**View:**
- Loss curves (IoU loss for DRCNN/Concrete)
- Metric trends (IoU, precision, recall)
- Processing pipeline visualizations (HSI → reduction → scores)
- Selected bands (Concrete selector)

---

## Troubleshooting

### Plugin Loading Failures

**Error:** `Plugin 'adaclip' not found or failed to load`

**Solutions:**
1. Check Git repository access:
   ```bash
   git ls-remote https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git
   ```

2. Verify tag exists:
   ```python
   config={"repo": "...", "tag": "v0.1.0"}  # Must match repository tags
   ```

3. For local development, use path-based loading:
   ```python
   config={"path": "/absolute/path/to/cuvis-ai-adaclip"}
   ```

4. Check plugin requirements:
   ```bash
   cd cuvis-ai-adaclip
   pip install -e .
   ```

### CUDA Out of Memory (DRCNN/Concrete)

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size:
   ```yaml
   data:
     batch_size: 2  # Reduce from 4
   ```

2. Enable mixed precision (DRCNN/Concrete):
   ```python
   adaclip = AdaCLIPDetector(
       use_half_precision=True,  # fp16 reduces memory
       ...
   )
   ```

3. Reduce image size:
   ```python
   adaclip = AdaCLIPDetector(
       image_size=336,  # Reduce from 518
       ...
   )
   ```

### Band Collapse (Concrete Selector)

**Issue:** All output channels select the same band

**Symptoms:**
```
⚠️ Only 1 unique bands selected out of 3 channels!
Final selected bands: [31, 31, 31]
```

**Solutions:**
1. Increase distinctness loss weight:
   ```python
   distinctness_loss = DistinctnessLoss(
       weight=0.5,  # Increase from 0.1
   )
   ```

2. Adjust temperature schedule:
   ```python
   selector = ConcreteChannelMixer(
       tau_start=15.0,  # Increase initial temperature
       tau_end=0.5,     # Increase final temperature
   )
   ```

3. Use longer training:
   ```yaml
   training:
     trainer:
       max_epochs: 100  # Increase from 50
   ```

### Low IoU Performance (All Approaches)

**Issue:** IoU < 0.5 on validation

**Solutions:**
1. Check normal_class_ids mapping:
   ```python
   data_node = LentilsAnomalyDataNode(
       normal_class_ids=[0, 1],    # Unlabeled + Lentils_black
       anomaly_class_ids=[3],      # Stone only
   )
   ```

2. Verify AdaCLIP prompt (if used):
   ```python
   adaclip = AdaCLIPDetector(
       prompt_text="normal: lentils, anomaly: stones",  # Task-specific
   )
   ```

3. Adjust decision threshold:
   ```python
   decider = QuantileBinaryDecider(
       quantile=0.99,  # Increase from 0.995 (more sensitive)
   )
   ```

4. Increase training epochs:
   ```yaml
   training:
     trainer:
       max_epochs: 100
   ```

### Mixer Weights Not Changing (DRCNN)

**Issue:** Weight change after training is near-zero

**Diagnosis:**
```python
logger.info(f"Weight change: max_diff={diff.max().item():.6f}")
# Output: Weight change: max_diff=0.000012  # Too small!
```

**Solutions:**
1. Increase learning rate:
   ```yaml
   training:
     optimizer:
       lr: 0.001  # Increase from 0.0001
   ```

2. Remove early stopping:
   ```python
   # Comment out early stopping callback
   # training_cfg.trainer.callbacks.early_stopping = []
   ```

3. Check gradients are flowing:
   ```python
   adaclip = AdaCLIPDetector(
       enable_gradients=True,  # CRITICAL for backprop
   )
   ```

4. Verify unfreezing:
   ```python
   pipeline.unfreeze_nodes_by_name([mixer.name])
   for name, param in mixer.named_parameters():
       assert param.requires_grad, f"{name} is still frozen!"
   ```

---

## Summary

You've learned three approaches to hyperspectral dimensionality reduction for AdaCLIP-based anomaly detection:

1. **PCA Baseline** - Fast, interpretable, statistical-only (2 min)
2. **DRCNN Mixer** - Best performance, learnable continuous mixing (15 min)
3. **Concrete Selector** - Interpretable discrete selection, pure gradient-based (12 min)

**Key Takeaways:**
- Plugin system enables modular extension of CUVIS.AI
- Dimensionality reduction strategy significantly impacts detection performance
- DRCNN mixer offers best quantitative metrics
- Concrete selector provides interpretable band choices
- TensorBoard visualization is essential for debugging end-to-end pipelines

**Performance Ranking (by IoU):**
1. 🥇 DRCNN Mixer (0.7512 test IoU)
2. 🥈 Concrete Selector (0.7401 test IoU)
3. 🥉 PCA Baseline (0.6791 test IoU)

---

## Next Steps

**Explore Related Topics:**
- [Plugin System Development](../plugin-system/development.md) - Create your own plugin nodes
- [gRPC Workflow Tutorial](grpc-workflow.md) - Distributed training and inference
- [Loss & Metrics Nodes](../node-catalog/loss-metrics.md) - IoU loss and distinctness loss details

**Try Advanced Configurations:**
- **Multi-loss training:** Combine IoU + entropy + diversity regularizers
- **Alternative selectors:** SupervisedCIRSelector, SupervisedWindowedSelector
- **Custom CLIP models:** Try different ViT backbones (ViT-B-16, ViT-L-14)
- **Transfer learning:** Fine-tune AdaCLIP prompts for your specific anomaly types

**Production Deployment:**
- [gRPC Deployment Guide](../deployment/grpc_deployment.md) - Deploy trained pipelines
- [Model Serving Patterns](../grpc/client-patterns.md) - Inference-only clients

---

## Complete Example Scripts

**PCA Baseline:**
```bash
python examples/adaclip/pca_adaclip_baseline.py
```
[View full source: examples/adaclip/pca_adaclip_baseline.py](../../examples/adaclip/pca_adaclip_baseline.py)

**DRCNN Mixer:**
```bash
python examples/adaclip/drcnn_adaclip_gradient_training.py
```
[View full source: examples/adaclip/drcnn_adaclip_gradient_training.py](../../examples/adaclip/drcnn_adaclip_gradient_training.py)

**Concrete Selector:**
```bash
python examples/adaclip/concrete_adaclip_gradient_training.py
```
[View full source: examples/adaclip/concrete_adaclip_gradient_training.py](../../examples/adaclip/concrete_adaclip_gradient_training.py)

---

**Need Help?**
- Check [Plugin System FAQ](../plugin-system/usage.md#troubleshooting)
- Review [Band Selection Strategies](../node-catalog/selectors.md)
- See [Training Configuration](../concepts/two-phase-training.md)
