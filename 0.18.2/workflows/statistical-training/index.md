# Statistical Training

Fit a cuvis-ai pipeline using `StatisticalTrainer` — accumulate background moments (mean, covariance, histograms) during a single pass over the data, no gradient steps.

## Goal

Produce a saved, ready-to-run pipeline whose statistical nodes have been initialised from data. The resulting pipeline can be replayed with [`restore-pipeline`](https://docs.cuvis.ai/latest/workflows/restore-pipeline/index.md).

## Prerequisites

- A pipeline with at least one [statistical node](https://docs.cuvis.ai/latest/catalogs/nodes/#category=model) (RX, PCA, NormalizeFromStats, …).
- A datamodule with a declared train split: `Cu3sDataModule` with `cu3s_file_path=...` for one cube (a `file_indices` selector names the training frames), or `data_dir=...` for a folder of cubes. A cu3s module refuses to fit on an undeclared whole recording, so anomalous frames cannot slip into the background statistics unnoticed.
- The [Concepts → Training](https://docs.cuvis.ai/latest/concepts/training/index.md) page if you want the model behind the trainer.

## Recipe

```python
from cuvis_ai_core.training import StatisticalTrainer
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_dataloader.data import Cu3sDataModule
from cuvis_ai_schemas.training import DataSplitConfig, Selector

pipeline = CuvisPipeline.load_pipeline("cuvis_ai/configs/pipeline/anomaly/rx/rx_statistical.yaml")
splits = DataSplitConfig(
    train=[Selector(kind="file_indices", source="data/Lentils/Lentils_000.cu3s", ids=[0, 2, 3])]
)
datamodule = Cu3sDataModule(cu3s_file_path="data/Lentils/Lentils_000.cu3s", splits=splits)

trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()

pipeline.save_to_file("artifacts/rx_statistical_fitted.yaml")
```

`save_to_file` writes the YAML and the matching `.pt` weights next to it. The same fit from the command line is the packaged statistical-only trainrun, which carries these splits in `cuvis_ai/configs/data/lentils.yaml`:

```bash
uv run restore-trainrun --trainrun-path cuvis_ai/configs/trainrun/rx_statistical.yaml --mode train
```

## What happens under the hood

1. Trainer collects every node whose `execution_stages` includes `STATISTICAL`.
1. For each batch, it calls `statistical_initialization(batch)` on every collected node.
1. After the pass, each node finalises its accumulated stats (covariance inversion, normalisation, etc.).
1. The fitted pipeline is saved as a YAML with `TRAINABLE_BUFFERS` populated.

## Common variations

- **Inference only on the trained pipeline**: skip authoring a fresh YAML — run [`restore-pipeline --pipeline-path artifacts/rx_statistical_fitted.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --data-arg cu3s_file_path=…`](https://docs.cuvis.ai/latest/workflows/restore-pipeline/index.md).
- **Statistical phase as part of two-phase training**: pair with [`GradientTrainer`](https://docs.cuvis.ai/latest/workflows/gradient-training/index.md) — the statistical phase initialises weights for the gradient phase. See [Concepts → Training](https://docs.cuvis.ai/latest/concepts/training/index.md).
- **Multi-cube training**: point the same `Cu3sDataModule` at a directory of cubes with `data_dir=...` instead of `cu3s_file_path=...`.

## Related

- [Concepts → Execution stages](https://docs.cuvis.ai/latest/concepts/execution-stages/index.md) — which nodes run when.
- [Build Pipeline (YAML)](https://docs.cuvis.ai/latest/workflows/build-pipeline-yaml/index.md) — author the pipeline this trainer fits.
- [Gradient Training](https://docs.cuvis.ai/latest/workflows/gradient-training/index.md) — the next phase if your pipeline has trainable parameters.
