# Statistical Tutorials

Pipelines that learn from background statistics alone — no gradient
steps, no optimizer. Statistical nodes accumulate moments
(mean, covariance, histograms) during initialization and use them at
inference time. Fast to train, interpretable, and strong baselines.

## In this section

- **[RX Anomaly Detection](rx-anomaly.md)** — classical Mahalanobis-distance anomaly detector. The canonical statistical baseline for hyperspectral anomaly detection.
- **[SIMCA Anomaly Detection](simca-anomaly.md)**: the class model chemometrics uses for authentication. A PCA of normal spectra, the score and orthogonal distance of every pixel to it, and a decision at a chosen significance level.
- **[Channel Selection](channel-selector.md)** — two-phase training that learns which wavelengths matter. Statistical warm-up sets the initial weights; gradient refinement (later phase) sharpens them.
- **[Blood Perfusion](blood-perfusion.md)** — normalised-difference (NDVI-style) two-band differential for tissue visualisation.

## See also

- [Concepts → Training](../../concepts/training.md) — the two-phase training model.
- [Gradient tutorials](../gradient/index.md) — the other half of the training story.
