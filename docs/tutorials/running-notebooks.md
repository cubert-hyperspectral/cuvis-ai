# Running the Notebooks Locally

The use-case tutorials ship as runnable notebooks in
[`notebooks/use_cases/`](https://github.com/cubert-hyperspectral/cuvis-ai/tree/main/notebooks/use_cases).
Each one runs on Google Colab (its first two cells bootstrap and provision it) or
locally. This
page is the repeatable recipe for provisioning a **local** environment to run any
of them.

cuvis-ai itself ships **no data module and no model plugins** — those live in
separate plugin packages. So beyond the base install, every notebook provisions
the one or two plugins it needs. The pattern is the same every time.

## The pattern

### 1. Base environment + JupyterLab

Clone the repo and sync the `dev` extra, which adds JupyterLab on top of the base
dependencies. (See [Installation](../get-started/installation.md) for `uv` itself
and the system dependencies below.)

```bash
git clone https://github.com/cubert-hyperspectral/cuvis-ai.git
cd cuvis-ai
uv sync --extra dev
```

!!! tip "Use `--extra dev`, not `--all-extras`"
    `--all-extras` also pulls the `plugins` extra (every model plugin at once),
    which is heavy and not always co-installable. Sync `dev` for the notebook
    runtime and provision per-notebook plugins in step 2.

### 2. Provision the notebook's plugins / data module

Every notebook carries the same provisioning cell right after its Colab bootstrap.
It names the packaged pipeline(s) that use the notebook's plugins plus the data
module, and runs `provision` (shipped by `cuvis-ai-core`) against the plugin
manifests packaged with cuvis-ai (`cuvis_ai/configs/plugins/<name>.yaml`). Git
plugins are pinned to their manifest tag, plugins that are already importable are
skipped, and after a fresh install the cell asks you to restart the kernel. Run it
and step 2 is done.

The same step from a terminal, for the Blood Perfusion notebook (builtin nodes
feeding a `cu3s` data module):

```bash
uv run provision \
  --pipeline-path cuvis_ai/configs/pipeline/medical/blood_perfusion/ndvi.yaml \
  --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --apply
```

resolves to `cuvis-ai-dataloader[cu3s,coco]` (the only plugin the pipeline's
builtin nodes plus `--data-module cu3s` require) and installs it. Drop `--apply`
to print the specs instead of installing; add `--include-satisfied` to list
plugins that are already present too. This is the same step
[`restore-pipeline`](../workflows/restore-pipeline.md) expects to have run first.

!!! warning "Re-provision after any `uv sync`"
    `uv sync` installs only what is in `pyproject.toml` and **removes** anything
    else, so it uninstalls out-of-tree plugins. Re-run the provisioning cell (step 2)
    after every `uv sync`; a plain `uv run` removes nothing.

!!! note "`.cu3s` I/O also needs the Cuvis SDK"
    The `cuvis-ai-dataloader[cu3s]` extra pulls the `cuvis` Python binding, which
    wraps the system-wide C++ Cuvis SDK. Install the SDK separately — see
    [Installation](../get-started/installation.md) (Cuvis SDK section). Notebooks
    that only use numpy / TIFF / video input do not need it.

### 3. Launch JupyterLab and run

```bash
uv run jupyter lab
```

`uv run` uses the project virtual environment directly, so there is **no
`ipykernel install` / kernel registration** to manage. Open
`notebooks/use_cases/<name>.ipynb` and run the cells top to bottom. For a fast
first pass, lower the frame-count knob (e.g. `N_FRAMES`) before the full sweep.

## What each notebook needs

The notebook's provisioning cell is the source of truth; this table is a quick
reference of what it installs.

| Notebook | Provision | System deps |
| --- | --- | --- |
| `blood_perfusion` | `cuvis-ai-dataloader[cu3s,coco]` | FFmpeg, Graphviz |
| `object_tracking_active` | `cuvis-ai-dataloader[cu3s,coco]` | FFmpeg, Graphviz |
| `object_tracking_passive` | `cuvis-ai-dataloader[cu3s,coco]`, `cuvis-ai-sam3` | FFmpeg, Graphviz |
| `object_selection_point_expansion` | `cuvis-ai-dataloader[cu3s,coco]`, `cuvis-ai-sam3` | FFmpeg |
| `lentils_dinomaly` | `cuvis-ai-dataloader[cu3s,coco]`, `cuvis-ai-dinomaly` | Graphviz |
| `metal_scrap_classification` | `cuvis-ai-inspecscrap[tiff]`, `cuvis-ai-dataloader[tiff]` | Graphviz |
| `channel_selector_lentils` | `cuvis-ai-dataloader[cu3s,coco]`, `cuvis-ai-adaclip`, `cuvis-ai-dinomaly` (AUROC metric node) | Graphviz (optional) |

`cuvis-ai-dinomaly`, `cuvis-ai-sam3`, `cuvis-ai-adaclip` and `cuvis-ai-inspecscrap` are not
published on PyPI. The provisioning cell installs them from the release tag pinned in
`cuvis_ai/configs/plugins/<name>.yaml`; do not hand-write the git spec, the pin is the
manifest's job.

FFmpeg is needed by notebooks that write an MP4 (`ToVideoNode`); Graphviz by
those that call `pipeline.visualize(format="render_graphviz", ...)`. Both are
covered in [Installation](../get-started/installation.md).

## Datasets

Each notebook downloads its demo dataset on first run via
`PublicDatasets.download_dataset(...)` (equivalent to
`uv run dataset download <name>`), cached under `data/`. These are large (Blood
Perfusion is ~7 GB), so the first run is slow, later runs reuse the cache. See
the [datasets catalog](../catalogs/datasets/index.md).

## Related

- [Installation](../get-started/installation.md) — `uv`, the Cuvis SDK, FFmpeg, Graphviz.
- [Restore Pipeline](../workflows/restore-pipeline.md) — the `provision` + `restore-pipeline` CLI flow.
- [Tutorials overview](index.md) — the notebooks themselves.
