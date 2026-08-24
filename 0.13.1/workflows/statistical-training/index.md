# Statistical Training

Fit a cuvis-ai pipeline using `StatisticalTrainer` — accumulate background moments (mean, covariance, histograms) during a single pass over the data, no gradient steps.

## Goal

Produce a saved, ready-to-run pipeline whose statistical nodes have been initialised from data. The resulting pipeline can be replayed with [`restore-pipeline`](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/workflows/restore-pipeline/index.md).

## Prerequisites

- A pipeline with at least one [statistical node](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/catalogs/nodes/#category=model) (RX, PCA, NormalizeFromStats, …).
- A datamodule that produces unlabelled training data: `Cu3sDataModule` with `cu3s_file_path=...` for one cube, or `data_dir=...` for a folder of cubes.
- The [Concepts → Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/concepts/training/index.md) page if you want the model behind the trainer.

## Recipe

```python
from cuvis_ai_core.training import StatisticalTrainer
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_dataloader.data import Cu3sDataModule

pipeline = CuvisPipeline.load_pipeline("cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml")
datamodule = Cu3sDataModule(cu3s_file_path="data/Lentils/Demo_000.cu3s")

trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()

pipeline.save_to_file("artifacts/rx_statistical_fitted.yaml")
```

## What happens under the hood

1. Trainer collects every node whose `execution_stages` includes `STATISTICAL`.
1. For each batch, it calls `statistical_initialization(batch)` on every collected node.
1. After the pass, each node finalises its accumulated stats (covariance inversion, normalisation, etc.).
1. The fitted pipeline is saved as a YAML with `TRAINABLE_BUFFERS` populated.

## Common variations

- **Inference only on the trained pipeline**: skip authoring a fresh YAML — run [`restore-pipeline --pipeline-path artifacts/rx_statistical_fitted.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --data-arg cu3s_file_path=…`](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/workflows/restore-pipeline/index.md).
- **Statistical phase as part of two-phase training**: pair with [`GradientTrainer`](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/workflows/gradient-training/index.md) — the statistical phase initialises weights for the gradient phase. See [Concepts → Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/concepts/training/index.md).
- **Multi-cube training**: point the same `Cu3sDataModule` at a directory of cubes with `data_dir=...` instead of `cu3s_file_path=...`.

## Related

- [Concepts → Execution stages](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/concepts/execution-stages/index.md) — which nodes run when.
- [Build Pipeline (YAML)](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/workflows/build-pipeline-yaml/index.md) — author the pipeline this trainer fits.
- [Gradient Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/workflows/gradient-training/index.md) — the next phase if your pipeline has trainable parameters.
