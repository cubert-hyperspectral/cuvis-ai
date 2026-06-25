# Use-case notebooks

End-to-end Cuvis.AI walkthroughs on real hyperspectral sessions. Each notebook
builds a pipeline, runs it on a Hugging Face dataset, and renders a result
(video or anomaly overlays).

| Notebook | What it does | Extra plugin | Dataset (Hugging Face) | GPU |
|----------|--------------|--------------|------------------------|-----|
| [`blood_perfusion.ipynb`](./blood_perfusion.ipynb) | NDVI blood-perfusion video from a CU3S hand session | none (builtin nodes) | `XMR_Demo_Blood_Perfusion` (~7 GB) | recommended |
| [`object_tracking_active.ipynb`](./object_tracking_active.ipynb) | SPAM invisible-ink tracking (spectral-angle, no model) | none (builtin nodes) | `XMR_Demo_Object_Tracking` (~25 GB) | recommended |
| [`object_tracking_passive.ipynb`](./object_tracking_passive.ipynb) | SAM3 mask-propagation tracking | `cuvis-ai-sam3` | `XMR_Demo_Object_Tracking` | yes |
| [`lentils_dinomaly.ipynb`](./lentils_dinomaly.ipynb) | Dinomaly anomaly detection (RGB / CIR / custom selector) | `cuvis-ai-dinomaly` | fetched per method from the demo model repo | yes |
| [`node_catalog_lentils.ipynb`](./node_catalog_lentils.ipynb) | New node families end to end: SNV pretreatment, K-Means, NNLS unmixing, one-class novelty / foreign-material, morphology, wired as a `CuvisPipeline` | none (builtin nodes) | `XMR_Lentils` (~0.9 GB) | optional |

All four read `.cu3s` sessions, so all four need the **`cuvis-ai-dataloader`
plugin** (the cu3s reader + the cuvis SDK). It is **not** a dependency of
`cuvis-ai`: builtin/RGB pipelines pull no SDK, and the data layer is a plugin
you install when a run needs it.

## Environment setup

```bash
# 1. cuvis-ai itself (see the Installation Guide for CUDA / prerequisites:
#    https://docs.cuvis.ai/latest/get-started/installation/)
uv sync

# 2. The cu3s data module (the cuvis SDK lives here, behind the [cu3s] extra).
#    Required for every notebook in this folder.
uv pip install "cuvis-ai-dataloader[cu3s,coco]"

# 3. The model plugin a given notebook needs (skip for the builtin-only ones):
uv pip install cuvis-ai-dinomaly   # lentils_dinomaly
uv pip install cuvis-ai-sam3       # object_tracking_passive

# 4. Launch Jupyter from the cuvis-ai environment:
uv run jupyter lab
```

Open a notebook and run the cells top to bottom. The first dataset cell pulls
the session from Hugging Face on first run (large, see the table); reruns are
cached.

> **Working from local plugin checkouts (dev).** `uv sync --extra plugins`
> wires the sibling plugin repos (dinomaly, sam3, ...) as editable installs.
> The dataloader plugin is not in that extra; install it explicitly, e.g.
> `uv pip install -e ../../cuvis-ai-dataloader/cuvis-ai-dataloader[cu3s,coco]`.

## Google Colab

Each notebook opens with a Colab badge and a bootstrap cell that installs
`cuvis-ai` plus `cuvis-ai-dataloader[cu3s,coco]` automatically, so on Colab you
only need to pick a GPU runtime (Runtime > Change runtime type > GPU). The
plugin-backed notebooks (`lentils_dinomaly`, `object_tracking_passive`) also
install their model plugin in a later cell.

## Plugins, briefly

A notebook makes a plugin's node classes importable by registering its manifest
on a `NodeRegistry`:

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.register_plugin("configs/plugins/dinomaly.yaml")
```

`register_plugin` only registers already-installed packages; install the
plugin first (step 2/3 above). For a config-driven setup, `provision`
(shipped by `cuvis-ai-core`) resolves a pipeline's `plugins:` block plus a
`--data-module` into the exact install command.
