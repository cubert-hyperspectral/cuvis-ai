# Use-case notebooks

End-to-end Cuvis.AI walkthroughs on real hyperspectral sessions. Each notebook
builds a pipeline, runs it on a Hugging Face dataset, and renders a result
(video or anomaly overlays).

| Notebook | What it does | Extra plugin | Dataset (Hugging Face) | GPU |
|----------|--------------|--------------|------------------------|-----|
| [`blood_perfusion.ipynb`](./blood_perfusion.ipynb) | NDVI blood-perfusion video from a CU3S hand session | none (builtin nodes) | `XMR_Demo_Blood_Perfusion` (~7 GB) | recommended |
| [`object_tracking_active.ipynb`](./object_tracking_active.ipynb) | SPAM invisible-ink tracking (spectral-angle, no model) | none (builtin nodes) | `XMR_Demo_Object_Tracking` (~25 GB) | recommended |
| [`object_tracking_passive.ipynb`](./object_tracking_passive.ipynb) | SAM3 mask-propagation tracking | `cuvis-ai-sam3` | `XMR_Demo_Object_Tracking` | yes |
| [`object_selection_point_expansion.ipynb`](./object_selection_point_expansion.ipynb) | SAM3 point expansion: click points into a mask, then propagate it 100 frames | `cuvis-ai-sam3` | `XMR_Demo_Object_Tracking` | yes |
| [`lentils_dinomaly.ipynb`](./lentils_dinomaly.ipynb) | Dinomaly anomaly detection (RGB / CIR / custom selector) | `cuvis-ai-dinomaly` | fetched per method from the demo model repo | yes |
| [`channel_selector_lentils.ipynb`](./channel_selector_lentils.ipynb) | Learn the 3-band custom selector: a Gumbel-Softmax `ConcreteChannelMixer` trained through frozen AdaCLIP on the full lentils split (multi-hour GPU training; ~57 GB download plus ~155 GB of converted frames) | `cuvis-ai-adaclip` + `cuvis-ai-dinomaly` (AUROC metric node) | `XMR_Industrial_Foreign_Object_Detection_Lentils` (~57 GB) | yes |

All notebooks read `.cu3s` sessions, so all of them need the **`cuvis-ai-dataloader`
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

# 3. The model plugin(s) a given notebook needs (skip for the builtin-only ones). These
#    plugins are not on PyPI; install them from the release tags pinned in
#    cuvis_ai/configs/plugins/<name>.yaml:
uv pip install "cuvis-ai-dinomaly @ git+https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly.git@v0.6.3"  # lentils_dinomaly, channel_selector_lentils
uv pip install "cuvis-ai-sam3 @ git+https://github.com/cubert-hyperspectral/cuvis-ai-sam3.git@v0.3.3"          # object_tracking_passive, object_selection_point_expansion
uv pip install "cuvis-ai-adaclip @ git+https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git@v0.3.1"    # channel_selector_lentils

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
install their model plugin in a later cell. `channel_selector_lentils` is
workstation-sized (~57 GB download, ~155 GB of converted frames, multi-hour
training); on Colab its bootstrap switches to the dry-run knobs.

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
