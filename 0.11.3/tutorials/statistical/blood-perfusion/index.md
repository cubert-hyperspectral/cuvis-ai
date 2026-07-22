# Blood Perfusion Visualization (NDVI-style)

Tissue oxygenation has a spectral signature: oxy- and deoxy-haemoglobin absorb light differently across the visible and near-infrared range. This tutorial builds a normalised-difference (NDVI-style) pipeline that maps that contrast into an intuitive false-colour visualisation of blood perfusion.

The pipeline runs on the `XMR_Demo_Blood_Perfusion` dataset and renders side-by-side frames showing tissue, perfusion overlay, and false-RGB.

**Run the example:**

- [`examples/blood_perfusion/nd_blood_perfusion.py`](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook/blob/main/examples/blood_perfusion/nd_blood_perfusion.py) — cuvis-ai-cookbook
- [Dataset on HuggingFace](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Blood_Perfusion) — also surfaced in the [datasets catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.3/catalogs/datasets/XMR_Demo_Blood_Perfusion/index.md)

## What you'll learn

- Building a normalised-difference index across two hyperspectral bands.
- Producing false-RGB output suitable for clinical or demo settings.
- Saving the rendered frames as a video artifact.

## When to reach for this pattern

- Any biological signal with a known two-band differential (NDVI for vegetation, NDWI for water, blood perfusion for tissue).
- You want a visualisation pipeline that runs in real time from a streaming camera.
- You want a baseline before reaching for learned tissue classifiers.
