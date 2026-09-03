# Running the Notebooks Locally

The use-case tutorials ship as runnable notebooks in
[`notebooks/use_cases/`](https://github.com/cubert-hyperspectral/cuvis-ai/tree/main/notebooks/use_cases).
Each one runs on Google Colab (the first cell bootstraps itself) or locally. This
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

Open the notebook and read its first cell: the Colab bootstrap lists exactly
what it installs (e.g. `cuvis-ai-dataloader[cu3s,coco]`). Mirror that into your
local env. Two equivalent ways:

=== "Direct install (simplest)"

    Install the plugin(s) the notebook imports, with the extras it uses. Blood
    Perfusion reads `.cu3s`, so it needs the data-layer plugin's `cu3s` extra:

    ```bash
    uv pip install "cuvis-ai-dataloader[cu3s,coco]"
    ```

=== "provision CLI (manifest-driven)"

    For a pipeline-YAML-backed run, the `provision` CLI resolves a pipeline's
    `plugins:` list plus `--data-module` into the right install specs (git
    plugins pinned to their manifest tag) and installs them. For example, the
    bundled Blood Perfusion NDVI pipeline (builtin nodes feeding a `cu3s` data
    module):

    ```bash
    uv run provision \
      --pipeline-path cuvis_ai/configs/pipeline/medical/blood_perfusion/ndvi.yaml \
      --plugins-dir cuvis_ai/configs/plugins --data-module cu3s --apply
    ```

    resolves to `cuvis-ai-dataloader[cu3s,coco]` (the only plugin the pipeline's
    builtin nodes plus `--data-module cu3s` require) and installs it. Drop
    `--apply` to print the specs instead of installing; add `--include-satisfied`
    to list plugins that are already present too. This is the same step
    [`restore-pipeline`](../workflows/restore-pipeline.md) expects to have run
    first.

!!! warning "Re-provision after any `uv sync`"
    `uv sync` installs only what is in `pyproject.toml` and **removes** anything
    else, so it uninstalls out-of-tree plugins. Re-run step 2 after every
    `uv sync`.

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

The notebook's first cell is the source of truth; this table is a quick
reference.

| Notebook | Provision | System deps |
| --- | --- | --- |
| `blood_perfusion` | `cuvis-ai-dataloader[cu3s,coco]` | FFmpeg, Graphviz |
| `object_tracking_active` | `cuvis-ai-dataloader[cu3s,coco]` | FFmpeg, Graphviz |
| `object_tracking_passive` | `cuvis-ai-dataloader[cu3s,coco]` | FFmpeg, Graphviz |
| `lentils_dinomaly` | `cuvis-ai-dinomaly` | Graphviz |
| `channel_selector_lentils` | `cuvis-ai-dataloader[cu3s,coco]`, `cuvis-ai-adaclip`, `cuvis-ai-dinomaly` (AUROC metric node) | Graphviz (optional) |

`cuvis-ai-dinomaly`, `cuvis-ai-sam3` and `cuvis-ai-adaclip` are not published on PyPI:
install them from the release tag pinned in `cuvis_ai/configs/plugins/<name>.yaml`, for
example `uv pip install "cuvis-ai-dinomaly @ git+https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly.git@v0.6.3"`,
or let `provision` resolve them from a pipeline YAML.

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
