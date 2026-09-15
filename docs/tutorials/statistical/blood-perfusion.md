# Blood Perfusion Visualization (NDVI-style)

Tissue oxygenation has a spectral signature: oxy- and deoxy-haemoglobin
absorb light differently across the visible and near-infrared range.
This tutorial builds a normalised-difference (NDVI-style) pipeline that
maps that contrast into an intuitive false-colour visualisation of
blood perfusion.

The pipeline runs on the `XMR_Demo_Blood_Perfusion` dataset and renders
a false-colour perfusion video from the normalised-difference index.

**Run the example:**

- [`notebooks/use_cases/blood_perfusion.ipynb`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/notebooks/use_cases/blood_perfusion.ipynb) — notebook walkthrough
- [`cuvis_ai/configs/pipeline/medical/blood_perfusion/ndvi.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/pipeline/medical/blood_perfusion/ndvi.yaml) — packaged pipeline, run with `restore-pipeline` as shown in [Your First Pipeline](../../get-started/first-pipeline.md)
- [Dataset on HuggingFace](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Blood_Perfusion) — also surfaced in the [datasets catalog](../../catalogs/datasets/XMR_Demo_Blood_Perfusion.md)

## What you'll learn

- Building a normalised-difference index across two hyperspectral bands.
- Producing false-RGB output suitable for clinical or demo settings.
- Saving the rendered frames as a video artifact.

## When to reach for this pattern

- Any biological signal with a known two-band differential (NDVI for vegetation, NDWI for water, blood perfusion for tissue).
- You want a visualisation pipeline that runs in real time from a streaming camera.
- You want a baseline before reaching for learned tissue classifiers.
