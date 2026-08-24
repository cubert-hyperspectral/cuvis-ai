# Quickstart Guide

Get up and running with Cuvis.AI in 5 minutes.

## Installation

First, ensure you have Python 3.10+ and [uv](https://docs.astral.sh/uv/) installed:

```bash
# Clone the repository
git clone https://github.com/cubert-hyperspectral/cuvis-ai.git
cd cuvis-ai

# Install dependencies
uv sync
```

See the [Installation Guide](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/get-started/installation/index.md) for detailed setup instructions.

## Download Sample Data

Download the Lentils dataset from Hugging Face:

```bash
# Automated download (default: lentils dataset)
uv run download-data

# Or explicitly specify dataset
uv run download-data --dataset lentils
```

This downloads ~1.0 GB of real hyperspectral data to `data/Lentils/`.

## Quick Demo: Run Pre-Trained Pipeline

Want to see Cuvis.AI in action first? Run inference with a pre-configured pipeline:

```bash
# View pipeline structure
uv run restore-pipeline --pipeline-path cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml

# Run inference on sample data
uv run restore-pipeline --pipeline-path cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --data-arg cu3s_file_path=data/Lentils/Demo_000.cu3s
```

This loads the pipeline configuration and runs anomaly detection on the sample hyperspectral cube.

## Train Your Own Pipeline

Train an RX anomaly detector from scratch using the script in the [cuvis-ai-cookbook](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook) repo:

```bash
# Clone the cookbook alongside this repo, then from cuvis-ai-cookbook/main:
uv run python examples/rx_statistical.py
```

Results are saved to `outputs/base_trainrun/`.

## What Just Happened?

1. **Loaded data** - The Lentils hyperspectral dataset
1. **Built pipeline** - RX statistical anomaly detector from `cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml`
1. **Trained model** - Statistical initialization on training data
1. **Saved results** - Pipeline, weights, and metrics to `outputs/`

## Use Your Trained Model

After training, restore and use your model for inference:

```bash
# Restore trained pipeline
uv run restore-pipeline --pipeline-path outputs/base_trainrun/trained_models/RX_Statistical.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --data-arg cu3s_file_path=data/Lentils/Lentils_000.cu3s
```

The pipeline will load your trained weights and run inference on new data.

## Next Steps

**Learn the fundamentals:**

- [Core Concepts Overview](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/concepts/index.md) - Understand the architecture
- [Configuration Basics](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/reference/configuration/index.md) - Master Hydra composition

**Follow comprehensive tutorials:**

- [RX Statistical Tutorial](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/tutorials/statistical/rx-anomaly/index.md) - Statistical anomaly detection
- [Channel Selector Tutorial](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/tutorials/statistical/channel-selector/index.md) - Learnable band selection
- [Deep SVDD Tutorial](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/tutorials/gradient/deep-svdd/index.md) - Deep learning approach

**Explore how-to guides:**

- [Build Pipelines in Python](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/workflows/build-pipeline-python/index.md)
- [Build Pipelines in YAML](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/workflows/build-pipeline-yaml/index.md)
- [Restore Trained Models](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/workflows/restore-pipeline/index.md)
