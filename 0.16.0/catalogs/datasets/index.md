# Datasets Catalog

Public datasets for cuvis-ai, hosted on [Hugging Face](https://huggingface.co/cubert-gmbh) under the `cubert-gmbh` organisation. Each page below mirrors the dataset's Hugging Face README; the HF page is the authoritative source.

Captured 5 with a Cubert [Ultris XMR](https://cubert-hyperspectral.com/de/ultris-xmr/) (61 bands, 430–910 nm); 1 with a Cubert Ultris X4 + SWIR rig (three visible and three short-wave-infrared bands).

The table is generated from the registry in cuvis-ai-core (`uv run dataset list`). `uv run dataset download <name> --data-dir data` fetches a dataset into `data/<folder>` and records a `.cuvis-dataset.json` marker beside it; `uv run dataset status --data-dir data` reports what is present. In CuvisNEXT the same list lives under Settings › Cuvis.AI › Datasets.

| Dataset                                                                                                                                               | Camera  | Size     | Files | Task                            | Licence    | Download                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | -------- | ----- | ------------------------------- | ---------- | ----------------------------------------------------- |
| [Demo: blood perfusion](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Demo_Blood_Perfusion/index.md)                   | XMR     | 11.2 GB  | 6     | Statistical                     | Apache-2.0 | `uv run dataset download Blood_Perfusion`             |
| [Demo: industrial FOD, lentils](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Demo_Industrial_FOD_Lentils/index.md)    | XMR     | 6.4 GB   | 21    | Anomaly detection, Segmentation | Apache-2.0 | `uv run dataset download Demo_Industrial_FOD_Lentils` |
| [Demo: object tracking](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Demo_Object_Tracking/index.md)                   | XMR     | 24.4 GB  | 18    | Tracking, Segmentation          | Apache-2.0 | `uv run dataset download Demo_Object_Tracking`        |
| [Industrial FOD, bedding (X4 SWIR)](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/X4_SWIR_Industrial_FOD_Bedding/index.md) | X4 SWIR | 178.2 GB | 683   | Anomaly detection, Segmentation | Apache-2.0 | `uv run dataset download Industrial_FOD_Bedding`      |
| [Industrial FOD, lentils](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Industrial_FOD_Lentils/index.md)               | XMR     | 57.0 GB  | 82    | Anomaly detection, Segmentation | Apache-2.0 | `uv run dataset download Industrial_FOD_Lentils`      |
| [Lentils (single session)](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Lentils/index.md)                             | XMR     | 921 MB   | 4     | Statistical                     | Apache-2.0 | `uv run dataset download Lentils`                     |

## Available datasets

- **[XMR_Demo_Blood_Perfusion](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Demo_Blood_Perfusion/index.md)**

  ______________________________________________________________________

  Tissue oxygenation visualised via two-band differential. Pairs with the [Blood Perfusion tutorial](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/tutorials/statistical/blood-perfusion/index.md). No training manifest: a statistical demo.

- **[XMR_Demo_Industrial_FOD_Lentils](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Demo_Industrial_FOD_Lentils/index.md)**

  ______________________________________________________________________

  Lentil-conveyor scene with stones, stems, aluminium shards, and flies. Pairs with the [AdaCLIP tutorial](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/tutorials/gradient/adaclip/index.md) and is the inference showcase of the trained Dinomaly pipelines on its companion model repo. No training manifest: the README declares no split.

- **[XMR_Demo_Object_Tracking](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Demo_Object_Tracking/index.md)**

  ______________________________________________________________________

  Crowded bus-station scene with passive and active (invisible-ink) tracking modes. No training manifest: the tracking nodes do not train.

- **[X4_SWIR_Industrial_FOD_Bedding](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/X4_SWIR_Industrial_FOD_Bedding/index.md)**

  ______________________________________________________________________

  252 VIS + SWIR frames of bedding: 193 normal training frames and a 59-frame validation split holding the 51 annotated foreign-object frames (`configs/data/industrial_fod_bedding.yaml`).

- **[XMR_Industrial_FOD_Lentils](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Industrial_FOD_Lentils/index.md)**

  ______________________________________________________________________

  Fifteen merged conveyor sessions over three days, 1,136 frames (696 annotated), seven foreign-object classes, with baked train / val / test selectors (`configs/data/industrial_fod_lentils.yaml`, `industrial_fod_lentils_normals.yaml`); the demo's trained Dinomaly pipelines were fitted on it.

- **[XMR_Lentils](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.0/catalogs/datasets/XMR_Lentils/index.md)**

  ______________________________________________________________________

  One lentil-conveyor session with COCO annotations, the recording the test suite and `configs/data/lentils.yaml` use.
