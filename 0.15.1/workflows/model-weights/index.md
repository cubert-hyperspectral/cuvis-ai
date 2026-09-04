# Model Weights

Several plugins load pretrained weights on first use: SAM3 (`sam3`), EfficientTAM (`rtsam2`), the DINOv2 backbone (`dinomaly`), and the CLIP backbone plus the fine-tuned heads (`adaclip`). From cuvis-ai-core 0.16.0 every one of these files is served from a public Hugging Face repository under the [`cubert-gmbh`](https://huggingface.co/cubert-gmbh) organisation: byte-identical to the upstream release, pinned to a mirror commit, and sha256-verified on download. No Hugging Face account, token, or licence click-through is needed.

The registry in `cuvis_ai_core.data.model_weights.ModelWeights` is the single source of truth for where a weight lives. Plugins resolve their checkpoints through it instead of hardcoding an upstream repo id, so the tool that provisions a weight and the runtime that loads it always look in the same cache folder.

## What is registered

| Registry name                                                   | Plugin   | Mirror repository                      | File                                      | Licence                          |
| --------------------------------------------------------------- | -------- | -------------------------------------- | ----------------------------------------- | -------------------------------- |
| `sam3`                                                          | sam3     | `cubert-gmbh/sam3`                     | `sam3.pt` (+ `config.json`)               | SAM License                      |
| `efficienttam_s`, `efficienttam_ti`                             | rtsam2   | `cubert-gmbh/efficient-track-anything` | `efficienttam_s.pt`, `efficienttam_ti.pt` | Apache-2.0                       |
| `efficienttam_s_512x512`, `efficienttam_ti_512x512`             | rtsam2   | `cubert-gmbh/efficient-track-anything` | 512x512 input variants                    | Apache-2.0                       |
| `dinov2_vitb14_reg4`                                            | dinomaly | `cubert-gmbh/dinov2`                   | `dinov2_vitb14_reg4_pretrain.pth`         | Apache-2.0                       |
| `clip_vit_l_14_336`                                             | adaclip  | `cubert-gmbh/clip`                     | `ViT-L-14-336px.pt`                       | unspecified upstream (code: MIT) |
| `adaclip_all`, `adaclip_mvtec_colondb`, `adaclip_visa_clinicdb` | adaclip  | `cubert-gmbh/adaclip`                  | `pretrained_*.pth`                        | unspecified upstream (code: MIT) |

Each mirror repository carries the upstream `LICENSE` file verbatim and a model card with the provenance: upstream source, upstream revision, and the sha256 of every file. The SAM License governs the SAM3 checkpoint; read it before shipping a product that embeds these weights.

`download-model list` prints the live table. `download-model list --json` emits one object per entry with `name`, `plugin`, `repo_id`, `filename`, `revision`, `sha256`, `aux_files`, `license`, `requires_token` and `cache_dir_token` (the `models--cubert-gmbh--<repo>` folder name), for tooling that needs to know what a pipeline will pull.

```bash
uv run download-model list
uv run download-model list --json
```

## Where the files go

Weights land in the Hugging Face hub cache: `HF_HUB_CACHE` if set, else `HF_HOME/hub`, else the huggingface_hub default (`~/.cache/huggingface/hub`). Every mirror gets its own folder, for example `models--cubert-gmbh--sam3/snapshots/<commit>/sam3.pt`.

Upgrading from cuvis-ai 0.14 or earlier

The folder names changed with the move to the mirrors (previously `models--facebook--sam3` and friends), so an existing install downloads its weights once more. Nothing else migrates.

## Provision ahead of time

In a notebook or a script that runs online, the plugin nodes download what they need on first use. Pre-fetching is still worth it for a large file (SAM3 is 3.4 GB) or for a machine that goes offline later:

```bash
uv run download-model download sam3
uv run download-model download efficienttam_s
uv run download-model download dinov2_vitb14_reg4
uv run download-model download clip_vit_l_14_336
uv run download-model download adaclip_all
```

Each command validates the sha256 pin, fetches the companion files (`config.json` for SAM3), and prints the resolved path on stdout so it composes with other tools. `--force` re-downloads a cached file. `--out <path>` additionally copies the file to a location of your choice, for example to hand it to a node's checkpoint-path hyperparameter.

## The gRPC child runtime is offline

The orchestrated gRPC server runs every pipeline in a per-run child environment with `HF_HUB_OFFLINE=1` and no credentials (see [gRPC Deployment](https://cubert-hyperspectral.github.io/cuvis-ai/0.15.1/deployment/grpc-deployment/index.md)). The child can only load weights that are already in the cache, and it resolves the same cache the provisioner writes to (`HF_HUB_CACHE` is exported to it explicitly). Provision the weights on the server host before the first run of a pipeline that needs them:

```bash
uv run download-model download sam3      # once per weight, on the host that runs the server
```

A run whose weight is missing fails while the pipeline loads, naming the command to run:

```text
ModelWeightsMissingError: 'sam3' is not in the model cache (...). Provision it with: uv run download-model download sam3
```

## Custom or private weights

The registry is the default, not a lock-in.

- `download-model download <name> --repo-id <org/repo> --filename <file> --revision <ref> --out <path>` fetches from another Hugging Face repository (a fork, a private mirror). `--token` defaults to `$HF_TOKEN` and is only needed for a private repository.
- A file fetched from a custom repository is not what the plugin's registry lookup finds, so point the node at it: the SAM3, RTSAM2 and AdaCLIP nodes accept a local checkpoint path or model directory hyperparameter (see the node's reference page in the [Nodes catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.15.1/catalogs/nodes/index.md)).

## For plugin authors

```python
from cuvis_ai_core.data.model_weights import ModelWeights

path = ModelWeights.resolve("sam3")                  # cached path; downloads when online
path = ModelWeights.resolve("sam3", download=False)  # pure lookup; raises when missing
seed = ModelWeights.materialize("dinov2_vitb14_reg4", vendored_dir)  # hardlink or copy
```

- `resolve(name, download=None)` returns the cached file. The default for `download` is "allowed unless `HF_HUB_OFFLINE` is set", so the same call works in a notebook and in the offline child. A miss with downloading disallowed raises `ModelWeightsMissingError`.
- `materialize(name, dest_dir, filename=None)` places a hardlink (or a copy, when linking is not possible) at a fixed path for loaders that cannot read the hub cache layout, such as anomalib's DINOv2 loader or the vendored CLIP loader. An existing destination is never overwritten.
- New weights belong in the core registry as a `cubert-gmbh` mirror (built with `tools/mirror_weights.py` in cuvis-ai-core) rather than behind an upstream URL inside the plugin; otherwise the offline child cannot find them. See the [Plugin Development Guide](https://cubert-hyperspectral.github.io/cuvis-ai/0.15.1/reference/plugin-development/guide/index.md).
