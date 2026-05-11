# Statistical Tutorials

Pipelines that learn from background statistics alone — no gradient steps, no optimizer. Statistical nodes accumulate moments (mean, covariance, histograms) during initialization and use them at inference time. Fast to train, interpretable, and strong baselines.

## In this section

- **[RX Anomaly Detection](https://docs.cuvis.ai/0.7.2/tutorials/statistical/rx-anomaly/index.md)** — classical Mahalanobis-distance anomaly detector. The canonical statistical baseline for hyperspectral anomaly detection.
- **[Channel Selection](https://docs.cuvis.ai/0.7.2/tutorials/statistical/channel-selector/index.md)** — two-phase training that learns which wavelengths matter. Statistical warm-up sets the initial weights; gradient refinement (later phase) sharpens them.
- **[Blood Perfusion](https://docs.cuvis.ai/0.7.2/tutorials/statistical/blood-perfusion/index.md)** — normalised-difference (NDVI-style) two-band differential for tissue visualisation.

## See also

- [Concepts → Training](https://docs.cuvis.ai/0.7.2/concepts/training/index.md) — the two-phase training model.
- [Gradient tutorials](https://docs.cuvis.ai/0.7.2/tutorials/gradient/index.md) — the other half of the training story.
