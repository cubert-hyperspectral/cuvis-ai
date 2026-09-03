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

Six of the seven notebooks read `.cu3s` sessions, so they need the
**`cuvis-ai-dataloader` plugin** (the cu3s reader + the cuvis SDK binding);
`metal_scrap_classification` reads SWIR TIFFs through the same plugin's `[tiff]`
extra. It is **not** a dependency of `cuvis-ai`: builtin/RGB pipelines pull no SDK,
and the data layer is a plugin that each notebook's provisioning cell installs when
a run needs it.

## Environment setup

```bash
# 1. cuvis-ai itself plus JupyterLab (see the Installation Guide for CUDA / prerequisites:
#    https://docs.cuvis.ai/latest/get-started/installation/)
uv sync --extra dev

# 2. Launch Jupyter from the cuvis-ai environment:
uv run jupyter lab
```

Open a notebook and run the cells top to bottom. The cell right after the Colab
bootstrap **provisions the notebook's plugins**: it runs `uv run provision` against
the plugin manifests packaged with cuvis-ai (`cuvis_ai/configs/plugins/<name>.yaml`),
so the data module (`cuvis-ai-dataloader[cu3s,coco]`: the cu3s reader plus the cuvis
SDK binding) and the model plugins (`cuvis-ai-dinomaly`, `cuvis-ai-sam3`,
`cuvis-ai-adaclip`, ... none of them on PyPI) are installed from their pinned
release tags, and nothing already importable is installed twice. Restart the kernel
once after a fresh install. The same command works from a terminal at the repo root,
for example the two lines `channel_selector_lentils` runs:

```bash
uv run provision --pipeline-path cuvis_ai/configs/pipeline/anomaly/adaclip/concrete_adaclip_gradient_two_stage.yaml --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --apply
uv run provision --pipeline-path cuvis_ai/configs/pipeline/anomaly/dinomaly/dinomaly_cir_lentils.yaml --plugins-dir cuvis_ai/configs/plugins --apply
```

`provision` resolves a pipeline YAML's `plugins:` list plus `--data-module` into
install specs, so each notebook names the packaged pipeline(s) that use its plugins.
The first dataset cell pulls the session from Hugging Face on first run (large, see
the table); reruns are cached.

> **`uv sync` removes provisioned plugins.** A plain `uv sync` (the pre-push hook runs
> one) installs only what `pyproject.toml` lists and uninstalls everything else. Re-run
> the provisioning cell afterwards; `uv run` on its own removes nothing.
>
> **Working from local plugin checkouts (dev).** `uv sync --extra plugins`
> wires the sibling plugin repos (dinomaly, sam3, ...) as editable installs.
> The dataloader plugin is not in that extra; install it explicitly, e.g.
> `uv pip install -e ../../cuvis-ai-dataloader/cuvis-ai-dataloader[cu3s,coco]`.

## Google Colab

Each notebook opens with a Colab bootstrap cell that installs `cuvis-ai` (a no-op
locally), followed by the provisioning cell, which installs the notebook's plugins
into the Colab kernel from the same packaged manifests (`%pip` under the hood). Pick
a GPU runtime (Runtime > Change runtime type > GPU) and restart the kernel once after
the install. `channel_selector_lentils` is workstation-sized (~57 GB download,
~155 GB of converted frames, multi-hour training); on Colab its bootstrap switches
to the dry-run knobs.

## Plugins, briefly

A notebook makes a plugin's node classes importable by registering its manifest
on a `NodeRegistry`. Use the manifest packaged with cuvis-ai, not a `../../`
relative path (which does not exist on Colab):

```python
from importlib.resources import files

from cuvis_ai_core.utils.node_registry import NodeRegistry

CONFIGS = files("cuvis_ai") / "configs"
registry = NodeRegistry()
registry.register_plugin(str(CONFIGS / "plugins" / "dinomaly.yaml"))
```

`register_plugin` only registers already-installed packages; the provisioning cell
(or `uv run provision` from a terminal, see above) installs them first.
