# Statistical Tutorials

Pipelines that learn from background statistics alone — no gradient steps, no optimizer. Statistical nodes accumulate moments (mean, covariance, histograms) during initialization and use them at inference time. Fast to train, interpretable, and strong baselines.

## In this section

- **[RX Anomaly Detection](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/tutorials/statistical/rx-anomaly/index.md)** — classical Mahalanobis-distance anomaly detector. The canonical statistical baseline for hyperspectral anomaly detection.
- **[Channel Selection](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/tutorials/statistical/channel-selector/index.md)** — two-phase training that learns which wavelengths matter. Statistical warm-up sets the initial weights; gradient refinement (later phase) sharpens them.
- **[Blood Perfusion](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/tutorials/statistical/blood-perfusion/index.md)** — normalised-difference (NDVI-style) two-band differential for tissue visualisation.

## See also

- [Concepts → Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/concepts/training/index.md) — the two-phase training model.
- [Gradient tutorials](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/tutorials/gradient/index.md) — the other half of the training story.
