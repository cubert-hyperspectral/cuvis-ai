# AdaCLIP — Vision-Language Anomaly Detection

AdaCLIP couples a frozen CLIP backbone with a small adapter trained against hyperspectral data, producing anomaly scores conditioned on a natural-language prompt. It's the workhorse pipeline for the `XMR_Demo_Industrial_FOD_Lentils` use case: "is this a lentil, or something else?"

This tutorial walks through three AdaCLIP variants, each one a different recipe for getting CLIP to work on hyperspectral data.

**Run the example:**

- [`examples/adaclip/pca_adaclip_baseline.py`](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook/blob/main/examples/adaclip/pca_adaclip_baseline.py) — PCA-reduced baseline (frozen AdaCLIP).
- [`examples/adaclip/concrete_adaclip_gradient_training.py`](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook/blob/main/examples/adaclip/concrete_adaclip_gradient_training.py) — Concrete channel selector + AdaCLIP gradient training.
- [`examples/adaclip/drcnn_adaclip_gradient_training.py`](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook/blob/main/examples/adaclip/drcnn_adaclip_gradient_training.py) — DRCNN-based channel reducer + AdaCLIP.
- [Dataset on HuggingFace](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils) — also surfaced in the [datasets catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.9.0/catalogs/datasets/XMR_Demo_Industrial_FOD_Lentils/index.md)

## What you'll learn

- How [external plugin nodes](https://cubert-hyperspectral.github.io/cuvis-ai/0.9.0/catalogs/nodes/index.md) integrate with cuvis-ai pipelines.
- Reducing 60+ band hyperspectral data to 3 channels CLIP can consume (PCA, Concrete, DRCNN).
- Using `restore-pipeline` to run AdaCLIP on new cu3s data after training.

## When to reach for AdaCLIP

- You want a strong anomaly detector that benefits from CLIP's pre-training but operates on hyperspectral data.
- You're comparing a frozen baseline against a trainable adapter and want both available in the same pipeline shape.
- You need text-prompted anomaly detection (the prompt conditions what counts as anomalous).
