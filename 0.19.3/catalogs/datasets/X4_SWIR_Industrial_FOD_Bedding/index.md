Mirrored from HuggingFace

This page mirrors the README of [`cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding`](https://huggingface.co/datasets/cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding).

# Hyperspectral Foreign-Object Detection in Bedding — Full Dataset (VIS + SWIR)

A **6-band VIS+SWIR** hyperspectral dataset for industrial foreign-object detection on a **bedding substrate** (a tray of wood-shaving / sawdust animal bedding). Captured with a Cubert **Ultris X4 + SWIR** rig — **6 spectral bands** at **450 / 550 / 625 nm (VIS)** and **1050 / 1200 / 1450 nm (SWIR)**, **2400 × 4900** pixels per frame. **252 frames** (193 train · 59 val), **51 frames** carry pixel-level polygon annotations spanning **23 foreign-object classes**.

This is the bedding counterpart to the 61-band lentils dataset at [`cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils). Where the lentils set uses the visible-only 61-band Ultris XMR, this set pairs **three visible bands with three SWIR bands** — the SWIR channels make water, alcohol, transparent plastics and colour-matched polymers separable from the organic bedding even when they are near-invisible in RGB.

The dataset is structured for **anomaly detection**: the **train split is 100 % normal/background** (clean bedding, no foreign objects), and the **val split** holds the anomalous frames (foreign objects placed by hand) plus a few normal frames. This mirrors a real inspection deployment — you learn the appearance of clean product, then flag anything that deviates.

## Summary

|                                   |                                                     |
| --------------------------------- | --------------------------------------------------- |
| Total frames                      | **252**                                             |
| Train (all normal/background)     | **193**                                             |
| Val (51 annotated + 8 normal)     | **59**                                              |
| Annotated frames                  | **51** (all in val)                                 |
| Annotated foreign-object polygons | **261**                                             |
| Foreign-object classes            | **23**                                              |
| Spectral bands                    | **6 · VIS 450/550/625 nm + SWIR 1050/1200/1450 nm** |
| Native spatial resolution         | **2400 × 4900** (H × W)                             |
| Camera                            | **Cubert Ultris X4 + SWIR**                         |
| Processing mode                   | **Reflectance** (u16, ×10000 scale)                 |
| Total size on disk                | **~167 GB** (cu3s)                                  |
| License                           | **Apache-2.0**                                      |

## Foreign-object classes

Class id 0 (`background`) is the bedding substrate + tray. The 23 foreign-object classes and their **annotated polygon counts** (object-level, from the LabelMe source polygons) are:

| id  | name             | polygons | id  | name                  | polygons |
| --- | ---------------- | -------- | --- | --------------------- | -------- |
| 1   | `water`          | 114      | 13  | `PLA_blue_2mm`        | 6        |
| 2   | `alcohol`        | 6        | 14  | `PLA_blue_4mm`        | 6        |
| 3   | `POMC`           | 15       | 15  | `PLA_blue_8mm`        | 7        |
| 4   | `PET`            | 13       | 16  | `PLA_blue_16mm`       | 7        |
| 5   | `leaf`           | 10       | 17  | `PLA_white_1mm`       | 2        |
| 6   | `fake_leaf`      | 0 ⚠      | 18  | `PLA_white_2mm`       | 5        |
| 7   | `PLA_black_1mm`  | 2        | 19  | `PLA_white_4mm`       | 10       |
| 8   | `PLA_black_2mm`  | 5        | 20  | `PLA_white_8mm`       | 10       |
| 9   | `PLA_blacK_4mm`  | 10       | 21  | `PLA_white_16mm`      | 9        |
| 10  | `PLA_black_8mm`  | 10       | 22  | `transparent_plastic` | 2        |
| 11  | `PLA_black_16mm` | 8        | 23  | `water&alcohol-tray`  | 1        |
| 12  | `PLA_blue_1mm`   | 3        |     |                       |          |

The `PLA_<colour>_<size>` families are 3-D-printed PLA plastic fragments at five physical sizes (1, 2, 4, 8, 16 mm) in three colours (black, blue, white). The 2 mm and 1 mm fragments — only a handful of pixels each — are the hardest to localise and the most interesting test of spectral (vs spatial) discrimination. `water` and `alcohol` are liquid spills; `POMC`, `PET`, `transparent_plastic` are polymer pieces; `leaf` / `fake_leaf` are organic vs synthetic foliage.

> ⚠ See **Known data issues** below for `fake_leaf` (id 6), the `PLA_blacK_4mm` spelling, and the `frame_10` label gap.

## Why VIS + SWIR

An RGB sensor collapses light into three visible bands. This rig adds **three short-wave-infrared bands (1050 / 1200 / 1450 nm)**. The 1450 nm band sits on a water-absorption feature, so **water and alcohol spills go dark in SWIR** while looking like wet sawdust in RGB. Transparent and colour-matched plastics that blend into the bedding in the visible range scatter differently in SWIR. The six bands together give a compact material fingerprint that separates foreign objects which are near-isoluminant in RGB.

The example frames below render the same cube two ways: a **VIS-RGB composite** (625 / 550 / 450 nm → R/G/B) and a **SWIR pseudo-RGB** (1450 / 1200 / 1050 nm → R/G/B), both per-channel min-max to uint8.

## Example frames

All examples are rendered directly from the `.cu3s` in this repo (native 2400 × 4900), per-channel min-max normalised to uint8. Annotated overlays draw the LabelMe polygons (native space).

### Small PLA fragments (`ok_nok`, val) — the hard case

`data/val/20250311_104919_frame_33_ok_nok_rdx_rwx.cu3s`

| VIS-RGB | VIS-RGB + annotations | SWIR pseudo-RGB |
| ------- | --------------------- | --------------- |
|         |                       |                 |

### Water spill (`nok_ok`, val)

`data/val/20250311_101004_frame_9_nok_ok_rdx_rwx.cu3s` · 17 annotated regions

| VIS-RGB | VIS-RGB + annotations | SWIR pseudo-RGB |
| ------- | --------------------- | --------------- |
|         |                       |                 |

### Multi-object scene (`nok_nok`, val)

`data/val/20250310_153943_frame_121_nok_nok_rdx_rwx.cu3s`

| VIS-RGB | VIS-RGB + annotations | SWIR pseudo-RGB |
| ------- | --------------------- | --------------- |
|         |                       |                 |

### Normal / background (`ok_ok`, val) — no foreign objects

`data/val/20250310_083936_frame_29_ok_ok_rd4_rw8.cu3s`

| VIS-RGB | SWIR pseudo-RGB |
| ------- | --------------- |
|         |                 |

## Known data issues

This dataset is released **as-is from the lab capture**, with three documented quirks. None of them block training/eval, but you should know about them:

1. **`frame_10` label gap (the documented fault).** `data/val/20250311_101035_frame_10_nok_ok_rdx_rwx.cu3s` has a **`nok` (anomalous) filename but no annotation** — its LabelMe polygon set and mask are absent. The filename label is correct (the frame *does* contain a foreign object), but the pixel-level ground truth was never drawn. It is flagged in `splits.csv` with `label_fault=1` (the only such row). For **image-level** evaluation use the filename label (`filename_label=1`); for **pixel-level** evaluation this frame contributes no positives and should be excluded or treated as a known miss.
1. **`fake_leaf` (class id 6) has no polygon annotations.** The class exists in the integer class map and appears as a few thin pixel slivers in the *rasterised* mask PNGs (`annotations_raw/`), but there is **no corresponding LabelMe polygon**, so the COCO files contain **0 `fake_leaf` regions**. Treat `fake_leaf` as effectively unannotated at the object level.
1. **`PLA_blacK_4mm` spelling.** Class id 9 is spelled with a capital `K` (`PLA_blacK_4mm`) in the canonical class map, while the LabelMe source labels spell it `PLA_black_4mm`. The COCO build maps both case-insensitively to id 9. The id is authoritative; the string is kept verbatim for traceability.

## Coordinate spaces

There are **two** pixel spaces in play — be explicit about which you use:

- **Native (this repo's COCO + masks): 2400 × 4900.** This is what `cuvis` returns when you open a `.cu3s` and read `m.data["cube"].array`. All COCO polygons (`annotations_canonical/`, `data/<split>/<stem>.json`) and the raw mask PNGs (`annotations_raw/`) live in this space.
- **Training crop: 1800 × 4300.** The reference Dinomaly/EfficientAD pipelines crop the tray borders with `cube[300:-300, 300:-300]` before training. If you reproduce those pipelines, apply the same crop to both cube and annotations; otherwise work in native space directly.

## Repository layout

```text
README.md
LICENSE                                  (Apache-2.0)
.gitattributes                           (LFS for *.cu3s, *.png, *.mp4)
splits.csv                               # 1 row per frame (252 rows)
class_map.json                           # {class_name: id}, 0..23
annotations_canonical/
  train_global_coco.json                 # merged COCO (train; all background)
  val_global_coco.json                   # merged COCO (val; 261 polygons, native space)
annotations_raw/
  labels/                                # original LabelMe JSON (*_RGB.json) + rasterised
                                         #   mask PNGs (*_mask.png), verbatim, 2400×4900
  README.md                              # raw-label format notes
assets/
  examples/                              # rendered VIS-RGB / SWIR / annotated PNGs
data/
  train/
    <stem>.cu3s                          # one hyperspectral frame (Ultris X4 + SWIR)
    <stem>.json                          # per-cu3s COCO (image_id 0; background for train)
  val/
    <stem>.cu3s
    <stem>.json                          # per-cu3s COCO (native 2400×4900)
```

`<stem>` encodes capture metadata: `YYYYMMDD_HHMMSS_frame_<n>_<tray-state>_<object-state>_<recipe>` where the `_ok_ok_` / `_ok_nok_` / `_nok_ok_` / `_nok_nok_` token pair is the filename-level normal/anomalous label (see `splits.csv` → `filename_label`).

### Per-`<stem>.json` COCO schema

```text
{
  "info": { "file_name": "<stem>.cu3s", "split": "train|val", "space": "native 4900x2400" },
  "licenses": [ { "id": 0, "name": "Apache-2.0", "url": "…" } ],
  "categories": [ { "id": 0..23, "name": "background|water|…|water&alcohol-tray" } ],
  "images": [
    { "id": 0, "file_name": "<stem>.cu3s",
      "width": 4900, "height": 2400, "channels": 6,
      "wavelength": [450, 550, 625, 1050, 1200, 1450] }
  ],
  "annotations": [
    { "id": …, "image_id": 0, "category_id": 1..23,
      "bbox": [x, y, w, h], "segmentation": [[…polygon…]],
      "area": <shoelace px area>, "iscrowd": 0 }
  ]
}
```

Annotations are **semantic polygons** transcribed from the LabelMe source. Normal/background frames have an `images` entry and an empty `annotations` list. `annotations_canonical/val_global_coco.json` is the same content merged across the split with global `image_id` (0..58, matching `splits.csv` row order within val) and globally-unique annotation ids.

### `splits.csv` columns

| column           | meaning                                                                  |
| ---------------- | ------------------------------------------------------------------------ |
| `split`          | `train` / `val`                                                          |
| `stem`           | frame identifier (filename without `.cu3s`)                              |
| `cu3s_path`      | path inside this repo, e.g. `data/val/<stem>.cu3s`                       |
| `coco_json_path` | matching per-cu3s COCO path                                              |
| `image_id`       | always 0 (one frame per cu3s)                                            |
| `filename_label` | 0 if `_ok_ok_` (normal), else 1 (anomalous) — image-level label          |
| `has_annotation` | 1 if the frame has ≥1 foreign-object polygon, else 0                     |
| `category_ids`   | `;`-separated class ids present (empty for normal frames)                |
| `label_fault`    | 1 for the single `frame_10` filename-vs-mask gap (see Known data issues) |

## Splits

| split | frames | annotated | normal/background |
| ----- | ------ | --------- | ----------------- |
| train | 193    | 0         | 193               |
| val   | 59     | 51        | 8                 |

The all-normal train split is intentional: this is an **anomaly-detection** benchmark. Train an unsupervised / SSL model (e.g. [Dinomaly](https://github.com/cubert-hyperspectral/cuvis-ai) or EfficientAD) on the clean bedding, then evaluate foreign-object localisation on the annotated val frames.

## How to load

### List the annotated val frames

```python
import csv
from huggingface_hub import hf_hub_download

splits = hf_hub_download(
    repo_id="cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding",
    repo_type="dataset", filename="splits.csv")
with open(splits) as f:
    val_ann = [r for r in csv.DictReader(f)
               if r["split"] == "val" and r["has_annotation"] == "1"]
print(len(val_ann), "annotated val frames")
```

### Stream one cu3s + COCO and render a VIS-RGB composite

```python
from huggingface_hub import hf_hub_download
import json, cuvis, numpy as np
from PIL import Image

repo = "cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding"
stem = "20250311_104919_frame_33_ok_nok_rdx_rwx"

cu3s = hf_hub_download(repo_id=repo, repo_type="dataset", filename=f"data/val/{stem}.cu3s")
js   = hf_hub_download(repo_id=repo, repo_type="dataset", filename=f"data/val/{stem}.json")

cuvis.init()                       # or cuvis.init("/path/to/cuvis/settings")
sf = cuvis.SessionFile(cu3s)
cube = sf[0].data["cube"].array    # (2400, 4900, 6) uint16 reflectance×10000
wl   = list(sf[0].data["cube"].wavelength)   # [450, 550, 625, 1050, 1200, 1450]

# VIS-RGB composite: 625 / 550 / 450 nm -> R / G / B
idx = [wl.index(t) for t in (625, 550, 450)]
sel = cube[..., idx].astype(np.float32)
u8  = np.zeros_like(sel, np.uint8)
for c in range(3):
    lo, hi = np.percentile(sel[..., c], (0.5, 99.5))
    u8[..., c] = (np.clip((sel[..., c] - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
Image.fromarray(u8, "RGB").save("frame_rgb.png")

anns = json.load(open(js))
print("polygons:", len(anns["annotations"]))
```

### Mirror everything to a local directory

```bash
huggingface-cli download \
  cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding \
  --repo-type=dataset --local-dir=./bedding_full
```

Or fetch only the lightweight metadata (skip the 167 GB of cubes) with `huggingface_hub.snapshot_download(..., allow_patterns=["*.csv","*.json","*.md","assets/**","annotations_raw/**"])`.

## Acquisition setup

- Camera: **Cubert Ultris X4 + SWIR** (6 bands: VIS 450/550/625 nm, SWIR 1050/1200/1450 nm)
- Subject: animal-bedding substrate (wood shavings / sawdust) in a tray, with hand-placed foreign objects (PLA fragments, liquids, polymer pieces, foliage)
- Processing mode: **Reflectance** (cu3s carry the calibration; `cuvis` returns u16 reflectance ×10000)
- Native frame: **2400 × 4900 × 6**

Lab proof-of-concept, not a production deployment study.

## Citation

```bibtex
@misc{raj2026beddinghsi,
  title  = {Hyperspectral VIS+SWIR Foreign-Object Detection in Bedding Substrate},
  author = {Raj, Anish},
  institution = {Cubert GmbH},
  year   = {2026},
  howpublished = {Hugging Face Datasets},
  url    = {https://huggingface.co/datasets/cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding}
}
```

## License

Released under the **Apache License 2.0** — see [`LICENSE`](https://huggingface.co/datasets/cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding/resolve/main/LICENSE). Matches the licensing of other Cubert public datasets on Hugging Face.

## Contact

Recorded and processed by the AI Team @ [Cubert](mailto:cuvis.ai@cubert-gmbh.de).

- Author: **Anish Raj** — [raj@cubert-gmbh.de](mailto:raj@cubert-gmbh.de)
- Team: [cuvis.ai@cubert-gmbh.de](mailto:cuvis.ai@cubert-gmbh.de)
