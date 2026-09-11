# AdaCLIP — Vision-Language Anomaly Detection

AdaCLIP couples a frozen CLIP backbone with a small adapter trained
against hyperspectral data, producing anomaly scores conditioned on a
natural-language prompt. It's the workhorse pipeline for the
`XMR_Demo_Industrial_FOD_Lentils` use case: "is
this a lentil, or something else?"

This tutorial walks through three AdaCLIP variants, each one a
different recipe for getting CLIP to work on hyperspectral data.

**Run the example:**

- [`cuvis_ai/configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml) — frozen-AdaCLIP baseline on fixed bands, run with `restore-pipeline` (the PCA-reduced variant from the retired cookbook script has no packaged equivalent yet).
- [`cuvis_ai/configs/pipeline/anomaly/adaclip/concrete_adaclip_gradient_two_stage.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/pipeline/anomaly/adaclip/concrete_adaclip_gradient_two_stage.yaml) — Concrete channel selector + AdaCLIP gradient training.
- [`cuvis_ai/configs/trainrun/drcnn_adaclip_trainrun.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/trainrun/drcnn_adaclip_trainrun.yaml) — DRCNN-based channel reducer + AdaCLIP, run with `restore-trainrun --mode train`.
- [`notebooks/use_cases/channel_selector_lentils.ipynb`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/notebooks/use_cases/channel_selector_lentils.ipynb) in cuvis-ai: the Concrete selector + frozen AdaCLIP recipe as a notebook on the full lentils dataset, ending with the learned bands and test-set metrics.
- [Dataset on HuggingFace](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils) — also surfaced in the [datasets catalog](../../catalogs/datasets/XMR_Demo_Industrial_FOD_Lentils.md)

## What you'll learn

- How [external plugin nodes](../../catalogs/nodes/index.md) integrate with cuvis-ai pipelines.
- Reducing 60+ band hyperspectral data to 3 channels CLIP can consume (PCA, Concrete, DRCNN).
- Using `restore-pipeline` to run AdaCLIP on new cu3s data after training.

## When to reach for AdaCLIP

- You want a strong anomaly detector that benefits from CLIP's pre-training but operates on hyperspectral data.
- You're comparing a frozen baseline against a trainable adapter and want both available in the same pipeline shape.
- You need text-prompted anomaly detection (the prompt conditions what counts as anomalous).
