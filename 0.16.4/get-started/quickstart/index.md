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

See the [Installation Guide](https://docs.cuvis.ai/latest/get-started/installation/index.md) for detailed setup instructions.

## Provision the Data Plugin

Reading `.cu3s` recordings goes through the `cuvis-ai-dataloader` plugin, which `uv sync` does not install. Provision it once for the pipeline you are about to run:

```bash
uv run provision --pipeline-path cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --apply
```

This resolves to `cuvis-ai-dataloader[cu3s,coco]` and installs it into the project environment. The `cu3s` extra wraps the system-wide C++ Cuvis SDK, which is a separate install; see the [Installation Guide](https://docs.cuvis.ai/latest/get-started/installation/index.md) (Cuvis SDK section). Re-run the command after any later `uv sync`: syncing removes plugins that are not listed in `pyproject.toml`.

## Download Sample Data

Download the Lentils dataset from Hugging Face:

```bash
# Download the lentils dataset
uv run dataset download lentils
```

This downloads ~1.0 GB of real hyperspectral data to `data/Lentils/`.

## Inspect the Packaged Pipeline

The packaged RX anomaly pipeline is a plain YAML graph: a Lentils data node, a min-max normalizer, the RX detector, a score-to-logit conversion and a binary decider, plus metric, mask and TensorBoard sinks. Print its nodes and port wiring without touching any data:

```bash
uv run restore-pipeline --pipeline-path cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml
```

The normalizer and the RX detector are statistical nodes: they need one pass over training frames before they can score anything, so the next step fits them.

## Train the Pipeline

The packaged trainrun `cuvis_ai/configs/trainrun/rx_statistical.yaml` pairs the pipeline with the Lentils data config (`cuvis_ai/configs/data/lentils.yaml`: train, val and test frames of `Lentils_000.cu3s`). It is statistical-only, so one pass over the train split fits the nodes, the test split is scored, and the fitted pipeline is written next to its weights:

```bash
uv run restore-trainrun --trainrun-path cuvis_ai/configs/trainrun/rx_statistical.yaml --mode train
```

The run takes about half a minute and writes `outputs/rx_statistical/trained_models/RX_Statistical_restored.yaml` plus the matching `.pt`. To do the same from Python, see the recipe in [Statistical Training](https://docs.cuvis.ai/latest/workflows/statistical-training/index.md).

## What Just Happened?

1. **Loaded data** - The Lentils recording, split into train, val and test frames by the data config
1. **Built pipeline** - RX statistical anomaly detector from `cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml`
1. **Fitted the statistical nodes** - The normalizer's bounds and the RX detector's background mean and covariance, from the train frames
1. **Scored the test split and saved the result** - Metrics and TensorBoard artifacts under `outputs/tensorboard/`, the fitted pipeline and weights under `outputs/rx_statistical/trained_models/`

## Use Your Trained Model

Restore the fitted pipeline and run inference over the recording. The pipeline now lives outside the packaged `configs/` tree, so pass the plugins directory that holds the `cu3s` data module's manifest:

```bash
# Restore trained pipeline
uv run restore-pipeline --pipeline-path outputs/rx_statistical/trained_models/RX_Statistical_restored.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --data-arg cu3s_file_path=data/Lentils/Lentils_000.cu3s
```

The pipeline loads your fitted weights, scores every frame and prints per-node timings.

## Next Steps

**Learn the fundamentals:**

- [Core Concepts Overview](https://docs.cuvis.ai/latest/concepts/index.md) - Understand the architecture
- [Configuration Basics](https://docs.cuvis.ai/latest/reference/configuration/index.md) - Master Hydra composition

**Follow comprehensive tutorials:**

- [RX Statistical Tutorial](https://docs.cuvis.ai/latest/tutorials/statistical/rx-anomaly/index.md) - Statistical anomaly detection
- [Channel Selector Tutorial](https://docs.cuvis.ai/latest/tutorials/statistical/channel-selector/index.md) - Learnable band selection
- [Deep SVDD Tutorial](https://docs.cuvis.ai/latest/tutorials/gradient/deep-svdd/index.md) - Deep learning approach

**Explore how-to guides:**

- [Build Pipelines in Python](https://docs.cuvis.ai/latest/workflows/build-pipeline-python/index.md)
- [Build Pipelines in YAML](https://docs.cuvis.ai/latest/workflows/build-pipeline-yaml/index.md)
- [Restore Trained Models](https://docs.cuvis.ai/latest/workflows/restore-pipeline/index.md)
