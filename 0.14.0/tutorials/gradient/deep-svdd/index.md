# Deep SVDD — One-Class Anomaly Detection

Deep Support Vector Data Description (Deep SVDD) learns a compact representation of "normal" data, then flags anything that lands far from the centre at inference time. It's the gradient-trained sibling of [RX anomaly detection](https://cubert-hyperspectral.github.io/cuvis-ai/0.14.0/tutorials/statistical/rx-anomaly/index.md) — slower to train, but able to capture non-Gaussian backgrounds.

This tutorial walks through training Deep SVDD end-to-end on a hyperspectral dataset and rendering anomaly heatmaps for each frame.

**Run the example:**

- [`examples/advanced/deep_svdd_gradient_training.py`](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook/blob/main/examples/advanced/deep_svdd_gradient_training.py) — cuvis-ai-cookbook

## What you'll learn

- How the [gradient training stage](https://cubert-hyperspectral.github.io/cuvis-ai/0.14.0/concepts/training/index.md) consumes a statistically-initialised pipeline.
- Configuring the representation dimension, soft boundary, and unfreeze schedule.
- Saving the trained pipeline and trainrun config so the run is reproducible via `restore-trainrun`.

## When to reach for Deep SVDD

- The background is non-Gaussian or multi-modal (RX would struggle).
- You have enough "normal" data to train on and no labelled anomalies.
- You're willing to spend training time to get a richer anomaly signal than statistical methods deliver.
