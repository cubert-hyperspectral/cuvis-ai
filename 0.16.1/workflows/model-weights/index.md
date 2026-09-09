# Model Weights

Several plugins load pretrained weights on first use: SAM3 (`sam3`), EfficientTAM (`rtsam2`), the DINOv2 backbone (`dinomaly`), and the CLIP backbone plus the fine-tuned heads (`adaclip`). Every one of these files is served from a public Hugging Face repository under the [`cubert-gmbh`](https://huggingface.co/cubert-gmbh) organisation: byte-identical to the upstream release, pinned to a mirror commit, and sha256-verified on download. No Hugging Face account, token, or licence click-through is needed.

Each plugin declares its weights in its own `weights.py` (a tuple of `cuvis_ai_schemas.plugin.PluginWeightEntry` rows) and registers them with cuvis-ai-core's `ModelWeights` registry when it is imported. The plugin manifests in this repository carry the same rows in their `weights:` block, so an environment that holds cuvis-ai but not the plugins (the CuvisNEXT venv, the installer helper) reads the identical registry. Core owns one more group of rows itself: the Cubert-trained Dinomaly pipelines. There is one cache contract for all of them, and one tool, `download-model`, provisions them.

## What is registered

| Registry name                                                                                     | Plugin                      | Mirror repository                                                  | File                                                 | Size          | Licence                          |
| ------------------------------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------- | ------------- | -------------------------------- |
| `sam3`                                                                                            | sam3                        | `cubert-gmbh/sam3`                                                 | `sam3.pt` (+ `config.json`)                          | 3.45 GB       | SAM License                      |
| `efficienttam_s` (alias `efficienttam`, default), `efficienttam_ti`                               | rtsam2                      | `cubert-gmbh/efficient-track-anything`                             | `efficienttam_s.pt`, `efficienttam_ti.pt`            | 136 MB, 72 MB | Apache-2.0                       |
| `efficienttam_s_512x512`, `efficienttam_ti_512x512`                                               | rtsam2                      | `cubert-gmbh/efficient-track-anything`                             | 512 x 512 input variants                             | 136 MB, 72 MB | Apache-2.0                       |
| `dinov2_vitb14_reg4`                                                                              | dinomaly                    | `cubert-gmbh/dinov2`                                               | `dinov2_vitb14_reg4_pretrain.pth`                    | 346 MB        | Apache-2.0                       |
| `clip_vit_l_14_336`                                                                               | adaclip                     | `cubert-gmbh/clip`                                                 | `ViT-L-14-336px.pt`                                  | 934 MB        | unspecified upstream (code: MIT) |
| `adaclip_all` (alias `pretrained_all`, default), `adaclip_mvtec_colondb`, `adaclip_visa_clinicdb` | adaclip                     | `cubert-gmbh/adaclip`                                              | `pretrained_*.pth`                                   | 43 MB each    | unspecified upstream (code: MIT) |
| `dinomaly_bedding_all6`                                                                           | dinomaly (trained pipeline) | `cubert-gmbh/dinomaly-bedding-all6`                                | `dinomaly_bedding_all6.pt` (+ `.yaml`)               | 594 MB        | see the model card               |
| `dinomaly_lentils_cir`, `dinomaly_lentils_custom`, `dinomaly_lentils_rgb`                         | dinomaly (trained pipeline) | `cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils` | `dinomaly_*_full_pipeline/dinomaly_*.pt` (+ `.yaml`) | 592 MB each   | Apache-2.0                       |

Rows with `kind: weights` are what a plugin's nodes need out of the box; the rtsam2 rows are picked by the node's `model_type` hyperparameter and the AdaCLIP heads by `weight_name`, with the `default` row used when the pipeline sets neither. Rows with `kind: trained_pipeline` are complete pipelines (checkpoint plus its pipeline YAML) that the CuvisNEXT pipeline picker's "Load trained weights" path consumes; no shipped preset requires them. Every row also carries `used_for` labels from a fixed vocabulary (`Point expansion`, `Propagation`, `Text prompts`, `Segment everything`, `Anomaly detection`, `Zero-shot`, `Backbone`, `Trained pipeline`), a plain-language `summary`, and the node hyperparameters that bypass the cache (`explicit_path_hparams`: `checkpoint_path` for sam3 and adaclip, `model_dir` for rtsam2).

Each mirror repository carries the upstream `LICENSE` file verbatim (where the upstream states one) and a model card with the provenance: upstream source, upstream revision, and the sha256 of every file. The SAM License governs the SAM3 checkpoint; read it before shipping a product that embeds these weights.

```bash
uv run download-model list          # the live table
uv run download-model list --json   # the machine-readable registry
```

`list --json` is an object: `schema_version` (1), `used_for_labels` (the vocabulary in display order), `weights_hosts` (the hosts a download contacts, for firewall rules and preflight checks), and `models`, one entry per row with `name`, `display_name`, `summary`, `used_for`, `plugin`, `family`, `kind`, `repo_id`, `filename`, `revision`, `sha256`, `size_bytes`, `aux_files`, `total_bytes`, `license`, `license_file`, `aliases`, `selected_by`, `default`, `plugin_default`, `explicit_path_hparams`, `cache_dir_name`, `description`, `source` (`plugin`, `manifest` or `dict`) and `pin_mismatch`. The exact contract is the JSON Schema `download-model schema model_list` prints; `status`, `export`, `remove`, `progress_event`, `dataset_list` and `dataset_status` have schemas of their own. The committed `cuvis_ai/configs/plugins/weights.index.json` is this object as generated from the manifests in this repository (see [The index and the release step](#the-index-and-the-release-step)).

## Where the files go

Weights land in the Hugging Face hub cache: `HF_HUB_CACHE` if set, else `HF_HOME/hub`, else the huggingface_hub default (`~/.cache/huggingface/hub`). Every mirror gets its own folder, for example `models--cubert-gmbh--sam3/snapshots/<commit>/sam3.pt`.

A weight is **present** when `<cache>/<cache_dir_name>/snapshots/<pinned revision>/<filename>` exists with the registered size and every companion file (`aux_files`) sits beside it. That is the whole contract: `download-model status`, the plugins' loaders and CuvisNEXT all apply it, a hit downloads nothing and hashes nothing, and `refs/main` is written after a download for Hugging Face's own tooling but never read. Registry downloads send no token, so an ambient `HF_TOKEN` or a stored login never reaches a mirror.

Upgrading from cuvis-ai 0.14 or earlier

The folder names changed with the move to the mirrors (previously `models--facebook--sam3` and friends), so an existing install downloads its weights once more. The old folders are not reused; `download-model remove --dir models--facebook--sam3` deletes one.

## Provision ahead of time

In a notebook or a script that runs online, the plugin nodes download what they need on first use. Pre-fetching is still worth it for a large file (SAM3 is 3.45 GB) or for a machine that goes offline later:

```bash
uv run download-model download sam3 efficienttam_s
uv run download-model download dinov2_vitb14_reg4 clip_vit_l_14_336 adaclip_all
uv run download-model status                 # present / partial / damaged / absent per row
uv run download-model status --verify sam3   # also recompute and compare the sha256
```

`download` validates the sha256 pin, fetches the companion files (`config.json` for SAM3), and prints one resolved path per line on stdout so it composes with other tools; `--force` re-downloads a cached file and `--out <path>` additionally copies the file to a location of your choice, for example to hand it to a node's checkpoint-path hyperparameter. `--progress-json` streams one JSON event per line (`waiting`, `progress`, `verifying`, `done`, `error`) for a host application; `--json` and `--progress-json` are mutually exclusive. Every write under the cache takes `<cache>/.cuvis-cache.lock`, so two provisioners sharing a cache wait for each other instead of interleaving. Exit codes: 0 success, 1 a registry or download error (`error: …` on stderr), 2 a usage error.

`--plugins-dir DIR` (repeatable) names the plugin manifests to read the `weights:` blocks from; without it, cuvis-ai's packaged `configs/plugins` is used when cuvis-ai is installed, and an environment with a plugin imported needs no manifest at all.

## Provision a training room or an air-gapped site

Download once on a connected machine, then carry the folder:

```powershell
uv run download-model download sam3 efficienttam_s
uv run download-model export --to E:\cuvis-weights sam3 efficienttam_s
```

```bash
uv run download-model download sam3 efficienttam_s
uv run download-model export --to /media/usb/cuvis-weights sam3 efficienttam_s
```

The export directory has the cache layout itself (`models--<org>--<repo>/snapshots/<commit>/…` plus `cuvis-model-weights.json`, the registry snapshot with a sha256 per file), so it is a valid `HF_HUB_CACHE` on its own. On every other machine:

```bash
uv run download-model import E:\cuvis-weights
uv run download-model status --verify
```

`import` copies the snapshots into a staging folder on the cache volume, verifies every file's size and sha256 there, and only then moves each model folder into place; any mismatch leaves the cache exactly as it was. Models already present are skipped. In CuvisNEXT the same action is Settings › Cuvis.AI › Model weights › Import from folder.

## CuvisNEXT

CuvisNEXT provisions through these commands. The installer offers the weights of the ticked plugins as sub-components and downloads them during installation; Settings › Cuvis.AI › Model weights lists the registry with size, licence and state per row (download, re-download, remove, import from folder), and a first launch without an installer selection offers the download once. `download-model remove NAME` (or `--dir models--…` for a folder the registry no longer knows) is what the Remove action runs, so cache content is only ever deleted inside core's cache lock.

## The gRPC child runtime is offline

The orchestrated gRPC server runs every pipeline in a per-run child environment with `HF_HUB_OFFLINE=1` and no credentials (see [gRPC Deployment](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.1/deployment/grpc-deployment/index.md)). The child can only load weights that are already in the cache, and it resolves the same cache the provisioner writes to (`HF_HUB_CACHE` is exported to it explicitly). Provision the weights on the server host before the first run of a pipeline that needs them:

```bash
uv run download-model download sam3      # once per weight, on the host that runs the server
```

A run whose weight is missing fails while the pipeline loads, naming the command to run:

```text
ModelWeightsMissingError: 'sam3' is not in the model cache (...). Provision it with: uv run download-model download sam3
```

## Custom or private weights

The registry is the default, not a lock-in.

- `download-model download <name> --repo-id <org/repo> --filename <file> --revision <ref> --out <path>` fetches from another Hugging Face repository (a fork, a private mirror). `--token` defaults to `$HF_TOKEN` and is honoured only together with `--repo-id`; registry downloads never send one.
- A file fetched from a custom repository is not what the plugin's registry lookup finds, so point the node at it: the SAM3, RTSAM2 and AdaCLIP nodes accept a local checkpoint path or model directory hyperparameter (see the node's reference page in the [Nodes catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.1/catalogs/nodes/index.md)). A pipeline that sets one of these `explicit_path_hparams` needs no cache entry for that plugin.

## Troubleshooting

| Symptom                                                   | Cause                                                            | Fix                                                                                |
| --------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `'<repo>' not found or not public`                        | mis-pinned registry row, or a mirror was made private            | upgrade the plugin / cuvis-ai; the mirrors are public by policy                    |
| `Hugging Face is rate-limiting or unavailable (HTTP 429)` | transient                                                        | retry in a few minutes; a partial `.incomplete` blob resumes                       |
| `sha256 mismatch` after a download                        | truncated or tampered file                                       | `download-model download <name> --force`                                           |
| `status` reports `damaged`                                | file present with the wrong size or a missing companion file     | re-download                                                                        |
| `status` reports `partial`                                | an interrupted download left an `.incomplete` blob               | run `download` again; it resumes                                                   |
| `pin_mismatch: true` in `list --json`                     | the installed plugin declares another revision than the manifest | upgrade cuvis-ai (its manifest pin) or the plugin; the plugin's row wins meanwhile |
| a run in the child fails with `ModelWeightsMissingError`  | the child is offline and the cache lacks the row                 | provision on the host, or set the node's explicit path                             |
| the old `models--facebook--*` folder still takes space    | legacy cache, never reused                                       | `download-model remove --dir models--facebook--sam3`                               |

## For plugin authors

A plugin declares its weights in a side-effect-free module and registers them at import:

```python
# cuvis_ai_myplugin/weights.py: declarations only, no torch, no core
from cuvis_ai_schemas.plugin import AuxFile, PluginWeightEntry

PLUGIN_NAME = "myplugin"  # the manifest's `name`
WEIGHTS: tuple[PluginWeightEntry, ...] = (
    PluginWeightEntry(
        name="myweights",
        display_name="My weights",
        summary="What it is for, in plain words (60 characters or fewer)",
        used_for=["Anomaly detection"],
        repo_id="cubert-gmbh/myweights",
        filename="model.pt",
        revision="<40-hex mirror commit>",
        sha256="<64-hex>",
        size_bytes=123_456_789,
        aux_files=[AuxFile(path="config.json", size_bytes=1234, sha256="<64-hex>")],
        license="Apache-2.0",
        license_file="LICENSE",
        explicit_path_hparams=["checkpoint_path"],
        description="One sentence on which nodes need it and where it comes from.",
    ),
)
```

```python
# cuvis_ai_myplugin/__init__.py
from cuvis_ai_core.data.model_weights import ModelWeights

from cuvis_ai_myplugin.weights import PLUGIN_NAME, WEIGHTS

ModelWeights.register(PLUGIN_NAME, WEIGHTS)
```

- `register` is idempotent on full-entry equality and enforces one namespace over names and aliases across plugins; a collision raises `ModelRegistryConflict` at import.
- A row a pipeline selects through a node hyperparameter sets `selected_by` (the hyperparameter name), `aliases` (the values that pick it) and `default: true` on the row a pipeline gets when the hyperparameter is unset; a row every node of the plugin needs leaves `selected_by` unset. `explicit_path_hparams` names the hyperparameters that bypass the cache. Every name in these fields must be a constructor parameter of one of the plugin's nodes: `emit_metadata` validates that.
- Mirror the weights with `tools/mirror_weights.py` in cuvis-ai-core (`plan`, `upload`, `check`); `plan` and `upload` print the `PluginWeightEntry(...)` rows to paste.
- Load through the registry in the node: `ModelWeights.resolve(name)` returns the cached path and downloads when online (a miss with downloading disallowed raises `ModelWeightsMissingError`); `ModelWeights.materialize(name, dest_dir, filename=None)` places a hardlink (or a copy) at a fixed path for loaders that cannot read the hub cache layout. The offline child then finds what `download-model` provisioned. See the [Plugin Development Guide](https://cubert-hyperspectral.github.io/cuvis-ai/0.16.1/reference/plugin-development/guide/index.md).

### The index and the release step

The `weights:` block of a plugin manifest in `cuvis_ai/configs/plugins/` is a projection of the plugin's `WEIGHTS` tuple, written by cuvis-ai-core's `emit_metadata` from an environment that has the plugin installed, and `cuvis_ai/configs/plugins/weights.index.json` is the registry `download-model list --json` builds from those manifests (rows sorted by name, two-space indent). Whenever a plugin pin in a manifest changes, run before tagging cuvis-ai:

```bash
# from a checkout / venv of the plugin at the tag the manifest pins
uv run --with ruamel.yaml python -m scripts.emit_metadata --manifest <cuvis-ai>/cuvis_ai/configs/plugins/<plugin>.yaml
# from cuvis-ai
uv run python -m scripts.weights_index
uv run pytest tests/plugins/test_weights_declarations.py
```

`emit_metadata --check` reports a drifted field without writing; the `Weights Compatibility` workflow runs it for every weight-bearing manifest whenever a manifest changes and on manual dispatch, and the test suite fails when the committed index is stale. CuvisNEXT's installer generator reads the index with CMake's `string(JSON)`, so the file is a contract, not a cache.
