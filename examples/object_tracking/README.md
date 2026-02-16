# Object Tracking — Channel Selector False RGB

Learnable channel mixing for hyperspectral object tracking. A `LearnableChannelMixer`
(1x1 conv) learns a linear combination of spectral bands into 3 RGB channels,
optimized with `ForegroundContrastLoss` to maximize foreground/background visual
separation. The false RGB output feeds SAM3 for video object tracking.

## Prerequisites

- CU3S recording: `Auto_002.cu3s`
- COCO annotations: `Auto_002.json`
- Update paths in `configs/data/tracking_cap_and_car.yaml`

## Step 1: Inspect — Discover Annotated Frames

Scan all frames, report which have mask annotations, and export a false RGB
video with mask overlays for visual inspection.

```bash
uv run python examples/object_tracking/channel_selector_false_rgb.py mode=inspect
```

**Outputs:**
- `outputs/channel_selector_false_rgb/inspect/false_rgb.mp4` — false RGB video for frame scrubbing
- `outputs/channel_selector_false_rgb/inspect/tensorboard/` — per-frame images with mask overlays

View per-frame artifacts in TensorBoard:

```bash
uv run tensorboard --logdir=outputs/channel_selector_false_rgb/inspect/tensorboard
```

The script prints annotated frame IDs. Copy them into `configs/data/tracking_cap_and_car.yaml`:

```yaml
train_ids: [0, 5, 10, 15, ...]
val_ids: [3, 8, ...]
test_ids: [4, 12, ...]
```

## Step 2: Train — Optimize Channel Mixer

Build and train the pipeline:

```
CU3SDataNode → MinMaxNormalizer → LearnableChannelMixer
     |                                  |          |
     mask ─────────────────────→ ForegroundContrastLoss
                                        |    DistinctnessLoss ← weights
                              ChannelSelectorFalseRGBViz → TensorBoardMonitorNode
```

```bash
uv run python examples/object_tracking/channel_selector_false_rgb.py mode=train
```

Training uses a two-phase approach:
1. **Statistical init** — PCA initialization for the mixer, running stats for the normalizer
2. **Gradient training** — AdamW (lr=0.005), 50 epochs with early stopping (patience=15)

**Outputs:**
- `outputs/channel_selector_false_rgb/trained_models/` — saved pipeline + trainrun config
- `outputs/channel_selector_false_rgb/tensorboard/` — loss curves + false RGB visualizations
- `outputs/channel_selector_false_rgb/pipeline/` — pipeline graph (PNG + Mermaid)

Monitor training:

```bash
uv run tensorboard --logdir=outputs/channel_selector_false_rgb/tensorboard
```

## Restore Trained Pipeline

Load the trained pipeline for inference on new data:

```bash
uv run restore-pipeline \
    --pipeline-path outputs/channel_selector_false_rgb/trained_models/Channel_Selector_FalseRGB.yaml \
    --cu3s-file-path <path-to-new-cu3s-file> \
    --processing-mode Raw
```

Export a pipeline graph visualization:

```bash
uv run restore-pipeline \
    --pipeline-path outputs/channel_selector_false_rgb/trained_models/Channel_Selector_FalseRGB.yaml \
    --pipeline-vis-ext png
```

## Restore Trainrun

Inspect the full training configuration or reproduce the training run:

```bash
# View trainrun info (data splits, losses, training config)
uv run restore-trainrun \
    --trainrun-path outputs/channel_selector_false_rgb/trained_models/channel_selector_false_rgb_trainrun.yaml \
    --mode info

# Re-run training
uv run restore-trainrun \
    --trainrun-path outputs/channel_selector_false_rgb/trained_models/channel_selector_false_rgb_trainrun.yaml \
    --mode train

# Run validation only
uv run restore-trainrun \
    --trainrun-path outputs/channel_selector_false_rgb/trained_models/channel_selector_false_rgb_trainrun.yaml \
    --mode validate

# Run test only
uv run restore-trainrun \
    --trainrun-path outputs/channel_selector_false_rgb/trained_models/channel_selector_false_rgb_trainrun.yaml \
    --mode test
```

Override config values when restoring:

```bash
uv run restore-trainrun \
    --trainrun-path outputs/channel_selector_false_rgb/trained_models/channel_selector_false_rgb_trainrun.yaml \
    --mode train \
    --override data.batch_size=4 \
    --override training.trainer.max_epochs=100
```

## Export False RGB Video (Standalone)

Export a false RGB video from any CU3S file using static wavelength-range averaging
(no training required):

```bash
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py \
    --cu3s-file-path <path-to-cu3s> \
    --output-video-path output.mp4 \
    --processing-mode Raw
```

## Config Files

| File | Purpose |
|------|---------|
| `configs/data/tracking_cap_and_car.yaml` | CU3S data paths and frame splits |
| `configs/trainrun/channel_selector_false_rgb.yaml` | Full training configuration (epochs, lr, losses, mixer) |
| `configs/training/default.yaml` | Base training defaults (inherited) |
