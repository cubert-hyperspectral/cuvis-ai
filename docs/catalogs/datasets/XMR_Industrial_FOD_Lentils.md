!!! info "Mirrored from HuggingFace"
    This page mirrors the README of [`cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils).

<p align="center">
  <img src="https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png" alt="Cubert Hyperspectral" width="560"/>
</p>

<p align="center">
  <a href="https://docs.cuvis.ai"><img src="https://img.shields.io/badge/Docs-docs.cuvis.ai-0aa?logo=readthedocs&logoColor=white" alt="Cuvis.AI docs"/></a>
  <a href="https://github.com/cubert-hyperspectral/cuvis-ai"><img src="https://img.shields.io/badge/GitHub-cuvis--ai-24292e?logo=github" alt="Cuvis.AI on GitHub"/></a>
  <a href="https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils"><img src="https://img.shields.io/badge/Demo-XMR%5FDemo%5FLentils-ffd21e?logo=huggingface&logoColor=000" alt="Companion demo"/></a>
</p>

# Hyperspectral Foreign-Object Detection in Lentils — Full Dataset

The larger counterpart to the small tutorial demo at
[`cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils`](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils).
Captured with a Cubert [**Ultris XMR**](https://cubert-hyperspectral.com/de/ultris-xmr/) camera — **61 bands per pixel, 430–910 nm, 1080 × 1000 pixels**. Three acquisition days, **15 merged `.cu3s` capture sessions**, **1,136 frames** total, **696 frames** with pixel-level COCO annotations across **7 foreign-object classes**.

Foreign-object detection in food sorting is an industrial-inspection problem — the rejected target could be a stone, a stem, a piece of packaging, a metal shard, or an insect. In this dataset the bulk product is bag-grade lentils (Emershofer Beluga and dark green marbled). Contaminants span seven classes (`stem_k`, `stone`, `alu_shard`, `blue_paper`, `white_paper`, `fly`, `rubber`). The same hyperspectral pipeline carries over to any product whose foreign objects differ spectrally from the bulk — even when they look near-identical in visible RGB.

## Summary

| | |
|---|---:|
| Total frames | **1,136** |
| Annotated frames | **696** (61.3 %) |
| Annotated foreign-object regions | **1,536** |
| Hyperspectral cubes (merged `.cu3s` files) | **15** |
| Spectral resolution | **61 bands · 430–910 nm · ≈8 nm spacing** |
| Spatial resolution | **1080 × 1000** |
| Processing mode | **Reflectance** (55 % gray reference + dark reference) |
| Splits | **train 808 · val 148 · test 180** (71.1 / 13.0 / 15.8 %) |
| Total size on disk | **~57 GB** |
| License | **Apache-2.0** |

### Per-day breakdown

| Day | Capture date | Subfolders | Frames | Annotated | Foreign-object regions |
|---|---|---:|---:|---:|---:|
| day2 | 2026-03-03 | 6 | 384 | 188 |   368 |
| day3 | 2026-03-10 | 6 | 492 | 328 |   648 |
| day4 | 2026-03-17 | 3 | 260 | 180 |   520 |
| **Total** | | **15** | **1,136** | **696** | **1,536** |

## Foreign-object classes

| id | name        | object count |
|---:|---          |---:|
| 0  | `Unlabeled`   | (background / normal lentils + belt) |
| 1  | `stem_k`      |  288 |
| 2  | `stone`       |  516 |
| 3  | `alu_shard`   |  112 |
| 4  | `blue_paper`  |   80 |
| 5  | `white_paper` |   60 |
| 6  | `fly`         |  420 |
| 7  | `rubber`      |   60 |

Class id 0 (`Unlabeled`) is the implicit background. Five subfolders are
**normal/background captures** (no foreign objects); their frames appear in
`universe.csv` (with no entries in their per-cu3s COCO `annotation`) and are
selected as the normal/background training set by `splits/dinomaly.json` for
SSL / unsupervised methods.

## Why hyperspectral

An RGB sensor collapses incoming light into three bands; the human eye does the
same. Hyperspectral video records **61 continuous bands per pixel, per frame** —
a material fingerprint that separates dyes, fabrics, coatings, pigments,
organic-vs-mineral matter, and surface chemistry.

Foreign objects that are colour-matched to the bulk product (small stones in
brown lentils, aluminium shards under warm lighting) are often near-isoluminant
in visible RGB. They typically reveal themselves in the near-infrared
(different surface scattering, different moisture content) or in narrow visible
bands the eye can't resolve.

The three views below show the same frame rendered through three 3-channel
projections of the 61-band cube (per-channel min-max, uint8). Bands chosen with
[`cuvis_ai.node.channel_selector`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/node/channel_selector.py)
classes `FixedWavelengthSelector` (defaults `650 / 550 / 450 nm`) and
`CIRSelector` (defaults NIR=860, R=670, G=560 nm).

## Example frames

All examples were rendered by downloading the `.cu3s` + `.json` from this
dataset on Hugging Face, applying `cuvis.ProcessingContext(sf).processing_mode = ProcessingMode.Reflectance`, picking the canonical band indices via
`cuvis_ai`'s `FixedWavelengthSelector` (RGB) and `CIRSelector` (CIR),
min-max-normalising each channel to `[0, 255]` and saving as PNG.

### Train · 1 foreign object (`stone`)

`data/day3/2026_03_10_10-58-55.cu3s` · `image_id=0` · `split=train` · 1 annotation (`stone`)

| RGB composite | RGB + annotation | CIR composite | CIR + annotation |
|:---:|:---:|:---:|:---:|
| ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_10_10-58-55_id0000_rgb_minmax_u8.png) | ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_10_10-58-55_id0000_rgb_annotated.png) | ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_10_10-58-55_id0000_cir_minmax_u8.png) | ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_10_10-58-55_id0000_cir_annotated.png) |

### Train · 3 foreign objects (`alu_shard` + `fly` + `stone`)

`data/day4/2026_03_17_11-41-54.cu3s` · `image_id=40` · `split=train` · 3 annotations

| RGB composite | RGB + annotations | CIR composite | CIR + annotations |
|:---:|:---:|:---:|:---:|
| ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_17_11-41-54_id0040_rgb_minmax_u8.png) | ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_17_11-41-54_id0040_rgb_annotated.png) | ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_17_11-41-54_id0040_cir_minmax_u8.png) | ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_17_11-41-54_id0040_cir_annotated.png) |

### Train · normal / background (no foreign objects)

`data/day2/2026_03_03_11-11-01.cu3s` · `image_id=0` · `split=train` · 0 annotations

| RGB composite | CIR composite |
|:---:|:---:|
| ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_03_11-11-01_id0000_rgb_minmax_u8.png) | ![](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_03_11-11-01_id0000_cir_minmax_u8.png) |

### Split-loader sanity check

Verified by downloading the cu3s via `huggingface_hub`, opening with
`cuvis.SessionFile`, and asserting `get_measurement(index).name`
matches the expected `camera_name` for each selected frame:

| split | cu3s | `index` | expected | got | ok |
|---|---|---:|---|---|---|
| train | `data/day2/2026_03_03_11-11-01.cu3s` |  0 | `Auto_000_4261` | `Auto_000_4261` | ✅ |
| val   | `data/day2/2026_03_03_11-31-31.cu3s` | 14 | `Auto_000_1339` | `Auto_000_1339` | ✅ |
| test  | `data/day3/2026_03_10_10-58-55.cu3s` | 12 | `Auto_000_1370` | `Auto_000_1370` | ✅ |

Polygon-bounds sanity: every annotation polygon vertex in the loaded frames
lies inside `(0..1080, 0..1000)`.

## Acquisition setup

- Camera: **Cubert Ultris XMR** hyperspectral, operated through Cuvis Next
- Illumination: 4 halogen lamps in 4 configurations (`l0`–`l3`) per scene arrangement
- Background: blue FDA-compliant conveyor-belt material (belt stationary during capture)
- Field of view: ≈12.5 × 12 cm at 46.6 cm working distance
- Exposure: 15 ms
- White reference: 55 % gray target; dark reference acquired by covering the lens
- Lentils: **Emershofer Beluga** and **Emershofer dark green marbled**

For each scene arrangement, **four captures under different lighting conditions**
form a grouped unit (the `group` column in `universe.csv`). All four images of a group are
always kept in the same train / val / test split to prevent lighting-only
information leakage.

The setup is a lab proof-of-concept with production-relevant design elements,
not a full production deployment study. See the
[**whitepaper PDF**](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/whitepaper/lentils_hsi_whitepaper.pdf) for the full
acquisition protocol, method comparison (RGB AdaCLIP / finetuned AdaCLIP / Dinomaly with custom selector), and limitations discussion.

## Repository layout

```
README.md
LICENSE                                 (Apache-2.0)
.gitattributes                          (LFS for *.cu3s)
universe.csv                            # unified manifest — 1 row per frame: source,index,annotation,group
splits/                                 # baked selector splits (core DataSplitConfig, file_indices)
  dinomaly.json                         # unsupervised (train-on-normals) selector
  adaclip.json                          # supervised baseline selector
annotations_canonical/                  # reference: per-day concatenated COCO (time-ordered global ids)
  day{2,3,4}_global_coco.json
assets/
  examples/                             # rendered example frames (see Example frames above)
whitepaper/
  lentils_hsi_whitepaper.pdf            # full whitepaper PDF
  lentils_hsi_whitepaper.md             # markdown source
data/
  day2/
    <subfolder>.cu3s                    # merged hyperspectral cube (capture session)
    <subfolder>.info                    # sensor sidecar (frame indexing)
    <subfolder>.json                    # per-cu3s COCO annotations (image_ids are local 0..N-1)
    <subfolder>_README.md               # data log for this capture session
    …                                   # 6 subfolders for day2
  day3/                                 # 6 subfolders for day3
  day4/                                 # 3 subfolders for day4
```

`<subfolder>` is the capture-session timestamp `YYYY_MM_DD_HH-MM-SS` (with `_1`/`_2`
suffix when the camera was restarted at the same wall-clock second).

### Per-`<subfolder>.json` COCO schema

Standard COCO with extra per-image fields for hyperspectral and traceability:

```jsonc
{
  "info": { "subfolder": "…", "day": "…", "frame_count": N, "annotation_count": M },
  "categories": [ { "id": 0..7, "name": "Unlabeled|stem_k|…|rubber" } ],
  "images": [
    {
      "id": <local_image_id>,           // 0..N-1, matches index inside the .cu3s
      "file_name": "<subfolder>.cu3s",
      "width": 1080, "height": 1000,
      "global_frame_id": <int>,         // 0..(day_total-1) — keys to the canonical day COCO
      "camera_frame_num": <int>,        // raw camera frame counter (matches `.info`)
      "camera_name": "Auto_000_<n>"
    }
  ],
  "annotations": [
    { "id": …, "image_id": <local_image_id>, "category_id": 1..7,
      "bbox": [x, y, w, h], "segmentation": [[…polygon…]],
      "iscrowd": 0, "area": 0.0, "mask": {"counts": [], "size": []}, "auxiliary": {} }
  ]
}
```

Annotations are **semantic masks**, not instance-level. Individual objects of
the same class in the same frame share a polygon contour, not separate instance
ids.

### `universe.csv` columns

The unified manifest: one row per frame, split-agnostic (split membership lives in `splits/*.json`).

| column | meaning |
|---|---|
| `source` | relative posix path to the cu3s session, e.g. `data/day2/2026_03_03_13-58-04_2.cu3s` |
| `index` | read position inside the merged `.cu3s` = COCO `image_id` (was `local_image_id`) |
| `annotation` | relative path to the per-cu3s COCO json (was `json_path`); the frame's masks are looked up in it by `index` |
| `group` | 4-frame lighting-quad group; all frames sharing a value stay in one split (was `group_id`) |

The former `splits.csv` columns (`day`, `subfolder`, `global_image_id`, `camera_frame_num`,
`camera_name`, `group_index`, `has_annotation`, `category_labels`) are recoverable from the per-cu3s
COCO json + the `annotations_canonical/` day COCOs; they are not needed to load or split the data.

## Splits

Splits ship as **selector files** under `splits/` (core `DataSplitConfig`, `file_indices`) that
resolve against `universe.csv` by `(source, index)`. Two views over the same universe are provided:

- **`splits/adaclip.json`** — supervised baseline: **train 808 · val 148 · test 180** (train = 500
  annotated positives + 308 normals).

- **`splits/dinomaly.json`** — unsupervised (train-on-normals): **train 308 · val 148 · test 180**,
  holding the 500 annotated positives out of training. `val` / `test` are identical to the adaclip
  view, so the two methods are directly comparable.

| adaclip split | frames | annotated | normal/background |
|---|---:|---:|---:|
| train | 808 | 500 | 308 |
| val   | 148 |  84 |  64 |
| test  | 180 | 112 |  68 |

The splits were generated with stratified group-aware splitting (lighting quads kept intact,
category balance preserved), then baked into the selector files. The same `splits/*.json` resolves
against the raw cu3s (via `cu3s_multi`) and against per-frame NPZ produced by `convert_universe`,
because both keep the `(source, index)` identities.

## How to load

### List the test set

```python
import json
from huggingface_hub import hf_hub_download

repo = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils"

# Split membership lives in the selector files under splits/, not in universe.csv.
splits_json = hf_hub_download(repo_id=repo, repo_type="dataset",
                              filename="splits/dinomaly.json")  # or splits/adaclip.json
sel = json.load(open(splits_json))
# each test selector is {source, ids}; ids are read positions == COCO image_ids
test = [(s["source"], i) for s in sel["test"] for i in s["ids"]]
print(len(test), "test frames")
```

### Stream one cu3s + annotations and render an RGB composite

```python
from huggingface_hub import hf_hub_download
import json, cuvis, numpy as np
from PIL import Image

repo = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils"
sub  = "data/day4/2026_03_17_11-11-50"

cu3s = hf_hub_download(repo_id=repo, repo_type="dataset", filename=f"{sub}.cu3s")
js   = hf_hub_download(repo_id=repo, repo_type="dataset", filename=f"{sub}.json")

cuvis.init()  # or cuvis.init("/path/to/cuvis/user/settings")
sf = cuvis.SessionFile(cu3s)
m  = sf.get_measurement(0)

# Cubes are stored in Preview mode; convert to Reflectance for analysis:
ctx = cuvis.ProcessingContext(sf)
ctx.processing_mode = cuvis.ProcessingMode.Reflectance
ctx.apply(m)

cube = m.cube.array         # shape (1000, 1080, 61), dtype uint16
wl   = list(m.cube.wavelength)  # 430..910 nm

# RGB composite (FixedWavelengthSelector defaults — 650 / 550 / 450 nm)
RGB = (650, 550, 450)
idx = [int(np.argmin(np.abs(np.asarray(wl) - t))) for t in RGB]
sel = cube[..., idx].astype(np.float32)
u8  = np.zeros_like(sel, dtype=np.uint8)
for c in range(3):
    lo, hi = np.percentile(sel[..., c], (0.5, 99.5))
    u8[..., c] = (np.clip((sel[..., c] - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
Image.fromarray(u8, "RGB").save("frame_rgb.png")

anns = json.load(open(js))
print("frames:", len(anns["images"]), "annotations:", len(anns["annotations"]))
```

### Mirror everything to a local directory

```bash
huggingface-cli download \
  cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils \
  --repo-type=dataset \
  --local-dir=./lentils_full
```

Or programmatically with `huggingface_hub.snapshot_download(...)` using
`allow_patterns=` to fetch only specific days / files.

## Citation

```bibtex
@techreport{raj2026lentilshsi,
  title  = {Spectral Foreign Object Detection in Lentils Using a Compact Hyperspectral Channel Selector},
  author = {Raj, Anish},
  institution = {Cubert GmbH},
  year   = {2026},
  note   = {Whitepaper, May 2026},
  url    = {https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/whitepaper/lentils_hsi_whitepaper.pdf}
}
```

## License

Released under the **Apache License 2.0** — see [`LICENSE`](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/LICENSE).
Matches the licensing of other Cubert public datasets on Hugging Face.

## Contact

Recorded and processed by the AI Team @ [Cubert](mailto:cuvis.ai@cubert-gmbh.de).
Reach out for collaboration, evaluation pilots, or to discuss running this
methodology on your own product line.

- Author: **Anish Raj** — <raj@cubert-gmbh.de>
- Team:   <cuvis.ai@cubert-gmbh.de>
