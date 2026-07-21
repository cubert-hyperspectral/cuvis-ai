# Statistical Training

Fit a cuvis-ai pipeline using `StatisticalTrainer` — accumulate
background moments (mean, covariance, histograms) during a single
pass over the data, no gradient steps.

## Goal

Produce a saved, ready-to-run pipeline whose statistical nodes have
been initialised from data. The resulting pipeline can be replayed
with [`restore-pipeline`](restore-pipeline.md).

## Prerequisites

- A pipeline with at least one [statistical node](../catalogs/nodes/index.md#category=model) (RX, PCA, NormalizeFromStats, …).
- A datamodule that produces unlabelled training data: `Cu3sDataModule` with `cu3s_file_path=...` for one cube, or `data_dir=...` for a folder of cubes.
- The [Concepts → Training](../concepts/training.md) page if you want the model behind the trainer.

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
2. For each batch, it calls `statistical_initialization(batch)` on every collected node.
3. After the pass, each node finalises its accumulated stats (covariance inversion, normalisation, etc.).
4. The fitted pipeline is saved as a YAML with `TRAINABLE_BUFFERS` populated.

## Common variations

- **Inference only on the trained pipeline**: skip authoring a fresh YAML — run [`restore-pipeline --pipeline-path artifacts/rx_statistical_fitted.yaml --cu3s-file-path …`](restore-pipeline.md).
- **Statistical phase as part of two-phase training**: pair with [`GradientTrainer`](gradient-training.md) — the statistical phase initialises weights for the gradient phase. See [Concepts → Training](../concepts/training.md).
- **Multi-cube training**: point the same `Cu3sDataModule` at a directory of cubes with `data_dir=...` instead of `cu3s_file_path=...`.

## Related

- [Concepts → Execution stages](../concepts/execution-stages.md) — which nodes run when.
- [Build Pipeline (YAML)](build-pipeline-yaml.md) — author the pipeline this trainer fits.
- [Gradient Training](gradient-training.md) — the next phase if your pipeline has trainable parameters.
