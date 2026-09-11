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

See the [Installation Guide](installation.md) for detailed setup instructions.

## Download Sample Data

Download the Lentils dataset from Hugging Face:

```bash
# Download the lentils dataset
uv run dataset download lentils
```

This downloads ~1.0 GB of real hyperspectral data to `data/Lentils/`.

## Quick Demo: Run a Packaged Pipeline

Want to see Cuvis.AI in action first? Run a packaged RX pipeline config; `restore-pipeline` statistically initialises it from the sample cube on the spot, no separate training step needed:

```bash
# View pipeline structure
uv run restore-pipeline --pipeline-path cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml

# Run inference on sample data
uv run restore-pipeline --pipeline-path cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --data-arg cu3s_file_path=data/Lentils/Demo_000.cu3s
```

This loads the pipeline configuration and runs anomaly detection on the sample hyperspectral cube.

## Train Your Own Pipeline

Fit the RX detector statistically using `StatisticalTrainer` (see
[Statistical Training](../workflows/statistical-training.md) for the full recipe):

```python
from cuvis_ai_core.training import StatisticalTrainer
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_dataloader.data import Cu3sDataModule

pipeline = CuvisPipeline.load_pipeline("cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml")
datamodule = Cu3sDataModule(cu3s_file_path="data/Lentils/Demo_000.cu3s")

trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()

pipeline.save_to_file("outputs/rx_statistical_fitted.yaml")
```

Results are saved to `outputs/rx_statistical_fitted.yaml`.

## What Just Happened?

1. **Loaded data** - The Lentils hyperspectral dataset
2. **Built pipeline** - RX statistical anomaly detector from `cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml`
3. **Trained model** - Statistical initialization on training data
4. **Saved results** - Pipeline, weights, and metrics to `outputs/`

## Use Your Trained Model

After training, restore and use your model for inference:

```bash
# Restore trained pipeline
uv run restore-pipeline --pipeline-path outputs/rx_statistical_fitted.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --data-arg cu3s_file_path=data/Lentils/Lentils_000.cu3s
```

The pipeline will load your trained weights and run inference on new data.

## Next Steps

**Learn the fundamentals:**

- [Core Concepts Overview](../concepts/index.md) - Understand the architecture
- [Configuration Basics](../reference/configuration/index.md) - Master Hydra composition

**Follow comprehensive tutorials:**

- [RX Statistical Tutorial](../tutorials/statistical/rx-anomaly.md) - Statistical anomaly detection
- [Channel Selector Tutorial](../tutorials/statistical/channel-selector.md) - Learnable band selection
- [Deep SVDD Tutorial](../tutorials/gradient/deep-svdd.md) - Deep learning approach

**Explore how-to guides:**

- [Build Pipelines in Python](../workflows/build-pipeline-python.md)
- [Build Pipelines in YAML](../workflows/build-pipeline-yaml.md)
- [Restore Trained Models](../workflows/restore-pipeline.md)
