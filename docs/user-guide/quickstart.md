# Quickstart Guide

Get up and running with CUVIS.AI in 5 minutes.

## Installation

First, ensure you have Python 3.10+ and [uv](https://docs.astral.sh/uv/) installed:

```bash
# Clone the repository
git clone https://github.com/cubert-hyperspectral/cuvis-ai.git
cd cuvis-ai

# Install dependencies
uv sync
```

See the [Installation Guide](installation.md) for detailed setup instructions.

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

Want to see CUVIS.AI in action first? Run inference with a pre-configured pipeline:

```bash
# View pipeline structure
uv run restore-pipeline --pipeline-path configs/pipeline/anomaly/rx/rx_statistical.yaml

# Run inference on sample data
uv run restore-pipeline --pipeline-path configs/pipeline/anomaly/rx/rx_statistical.yaml --cu3s-file-path data/Lentils/Demo_000.cu3s
```

This loads the pipeline configuration and runs anomaly detection on the sample hyperspectral cube.

## Train Your Own Pipeline

Train an RX anomaly detector from scratch:

```bash
# Train RX detector
uv run python examples/rx_statistical.py
```

Results are saved to `outputs/base_trainrun/`.

## What Just Happened?

1. **Loaded data** - The Lentils hyperspectral dataset
2. **Built pipeline** - RX statistical anomaly detector from `configs/pipeline/anomaly/rx/rx_statistical.yaml`
3. **Trained model** - Statistical initialization on training data
4. **Saved results** - Pipeline, weights, and metrics to `outputs/`

## Use Your Trained Model

After training, restore and use your model for inference:

```bash
# Restore trained pipeline
uv run restore-pipeline --pipeline-path outputs/base_trainrun/trained_models/RX_Statistical.yaml --cu3s-file-path data/Lentils/Lentils_000.cu3s
```

The pipeline will load your trained weights and run inference on new data.

## Next Steps

**Learn the fundamentals:**

- [Core Concepts Overview](../concepts/overview.md) - Understand the architecture
- [Configuration Basics](configuration.md) - Master Hydra composition

**Follow comprehensive tutorials:**

- [RX Statistical Tutorial](../tutorials/rx-statistical.md) - Statistical anomaly detection
- [Channel Selector Tutorial](../tutorials/channel-selector.md) - Learnable band selection
- [Deep SVDD Tutorial](../tutorials/deep-svdd-gradient.md) - Deep learning approach

**Explore how-to guides:**

- [Build Pipelines in Python](../how-to/build-pipeline-python.md)
- [Build Pipelines in YAML](../how-to/build-pipeline-yaml.md)
- [Restore Trained Models](../how-to/restore-pipeline-trainrun.md)
