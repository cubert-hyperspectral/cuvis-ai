# RX Statistical Anomaly Detection

The Reed–Xiaoli (RX) detector is the classical baseline for unsupervised hyperspectral anomaly detection. It scores each pixel by its Mahalanobis distance from the background spectrum, so pixels that are spectrally unusual stand out.

This tutorial walks through building an end-to-end RX pipeline in cuvis.AI — loading cu3s data, fitting the detector with [statistical training](https://docs.cuvis.ai/latest/concepts/training/index.md), and rendering an anomaly heatmap.

**Run the example:**

- [`cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml) — packaged pipeline, run with `restore-pipeline` (statistical fit happens automatically on restore; see [Statistical Training](https://docs.cuvis.ai/latest/workflows/statistical-training/index.md) for the underlying recipe)

## What you'll learn

- How [statistical nodes](https://docs.cuvis.ai/latest/concepts/training/index.md) accumulate background statistics during initialization.
- Wiring a data loader → preprocessing → RX detector → visualization pipeline.
- Saving a fitted pipeline so it can be replayed with `restore-pipeline`.

## When to reach for RX

- Unsupervised baseline for hyperspectral anomaly detection.
- Background follows an approximately Gaussian distribution.
- You want a fast, interpretable signal before reaching for gradient-based methods.
