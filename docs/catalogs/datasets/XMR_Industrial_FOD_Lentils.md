!!! info "Mirrored from HuggingFace"
    This page mirrors the README of [`cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils).

<p align="center">
  <img src="https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png" alt="Cubert Hyperspectral" width="560"/>
</p>

<p align="center">
  <a href="https://docs.cuvis.ai"><img src="https://img.shields.io/badge/Docs-docs.cuvis.ai-0aa?logo=readthedocs&logoColor=white" alt="Cuvis.AI docs"/></a>
  <a href="https://github.com/cubert-hyperspectral/cuvis-ai"><img src="https://img.shields.io/badge/GitHub-cuvis--ai-24292e?logo=github" alt="Cuvis.AI on GitHub"/></a>
  <a href="https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils"><img src="https://img.shields.io/badge/Dataset-HuggingFace-ffd21e?logo=huggingface&logoColor=000" alt="Companion dataset"/></a>
</p>

# Hyperspectral foreign-object detection in lentils: the full dataset

The larger counterpart of the tutorial demo at cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils.
Foreign-object detection in food sorting is an inspection problem: the rejected target can be a stone, a stem, a
piece of packaging, a metal shard or an insect. Here the bulk product is bag-grade lentils (Emershofer Beluga and
dark green marbled) and the contaminants span seven classes (stem_k, stone, alu_shard, blue_paper, white_paper,
fly, rubber). The same frames serve supervised detection (the labelled frames) and unsupervised anomaly detection
(the normal-only recordings), and the two baked splits evaluate both on identical held-out frames. The whitepaper
at `whitepaper/lentils_hsi_whitepaper.pdf` holds the acquisition protocol, the method comparison (RGB AdaCLIP,
fine-tuned AdaCLIP, Dinomaly with a custom channel selector) and the limitations. The setup is a laboratory proof
of concept with production-relevant design elements, not a production deployment study.


Captured with a Cubert Ultris XMR camera (61 bands per pixel, 430 to 910 nm, about 8 nm spacing, 1080 x 1000 pixels). 3 acquisition days, 15 `.cu3s` recordings, 1,136 frames, 696 of them with pixel-level COCO annotations across 7 classes.

## Summary

| | |
|---|---:|
| Total frames | **1,136** |
| Annotated frames | **696** (61.3 %) |
| Annotated regions | **1,536** |
| Hyperspectral cubes (`.cu3s` recordings) | **15** |
| Spectral resolution | **61 bands per pixel, 430 to 910 nm, about 8 nm spacing, 1080 x 1000 pixels** |
| Processing mode | **Reflectance** (white and dark reference recorded per session) |
| Splits | dinomaly: train 308, val 148, test 180; adaclip: train 808, val 148, test 180 |
| Total size on disk | **about 57.0 GB** |
| License | **Apache-2.0** |

### Per-day breakdown

| Day | Capture date | Recordings | Frames | Annotated | Regions |
|---|---|---:|---:|---:|---:|
| day2 | 2026-03-03 | 6 | 384 | 188 | 368 |
| day3 | 2026-03-10 | 6 | 492 | 328 | 648 |
| day4 | 2026-03-17 | 3 | 260 | 180 | 520 |
| **Total** | | **15** | **1,136** | **696** | **1,536** |

## Classes

| id | name | regions |
|---:|---|---:|
| 1 | `stem_k` | 288 |
| 2 | `stone` | 516 |
| 3 | `alu_shard` | 112 |
| 4 | `blue_paper` | 80 |
| 5 | `white_paper` | 60 |
| 6 | `fly` | 420 |
| 7 | `rubber` | 60 |

Class id 0 (`Unlabeled`) is the implicit background: every COCO file lists it in `categories` and no annotation carries it. A recording with no annotation file holds normal product only; its frames are in `universe.csv` and carry no regions.

## Why hyperspectral

An RGB sensor collapses incoming light into three bands; the human eye does the same. Hyperspectral video records
61 continuous bands per pixel and frame: a material fingerprint that separates dyes, fabrics, coatings, pigments,
organic from mineral matter and surface chemistry. Foreign objects that match the colour of the bulk product (small
stones in brown lentils, aluminium shards under warm lighting) are often near-isoluminant in visible RGB; they
reveal themselves in the near infrared (different surface scattering, different moisture) or in narrow visible
bands the eye cannot resolve. The example frames below show the same cube through three-channel projections built
with the Cuvis.AI channel selectors (`FixedWavelengthSelector` at 650, 550 and 450 nm; `CIRSelector` at NIR 860,
R 670 and G 560 nm), each channel min-max scaled to 8 bit.


## Example frames

| 2026_03_03_11-11-01_id0000_cir_minmax_u8 | 2026_03_03_11-11-01_id0000_rgb_minmax_u8 | 2026_03_10_10-58-55_id0000_cir_annotated |
|:---:|:---:|:---:|
| ![2026_03_03_11-11-01_id0000_cir_minmax_u8](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_03_11-11-01_id0000_cir_minmax_u8.png) | ![2026_03_03_11-11-01_id0000_rgb_minmax_u8](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_03_11-11-01_id0000_rgb_minmax_u8.png) | ![2026_03_10_10-58-55_id0000_cir_annotated](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_10_10-58-55_id0000_cir_annotated.png) |

| 2026_03_10_10-58-55_id0000_cir_minmax_u8 | 2026_03_10_10-58-55_id0000_rgb_annotated | 2026_03_10_10-58-55_id0000_rgb_minmax_u8 |
|:---:|:---:|:---:|
| ![2026_03_10_10-58-55_id0000_cir_minmax_u8](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_10_10-58-55_id0000_cir_minmax_u8.png) | ![2026_03_10_10-58-55_id0000_rgb_annotated](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_10_10-58-55_id0000_rgb_annotated.png) | ![2026_03_10_10-58-55_id0000_rgb_minmax_u8](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_10_10-58-55_id0000_rgb_minmax_u8.png) |

| 2026_03_17_11-41-54_id0040_cir_annotated | 2026_03_17_11-41-54_id0040_cir_minmax_u8 | 2026_03_17_11-41-54_id0040_rgb_annotated |
|:---:|:---:|:---:|
| ![2026_03_17_11-41-54_id0040_cir_annotated](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_17_11-41-54_id0040_cir_annotated.png) | ![2026_03_17_11-41-54_id0040_cir_minmax_u8](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_17_11-41-54_id0040_cir_minmax_u8.png) | ![2026_03_17_11-41-54_id0040_rgb_annotated](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_17_11-41-54_id0040_rgb_annotated.png) |

| 2026_03_17_11-41-54_id0040_rgb_minmax_u8 |
|:---:|
| ![2026_03_17_11-41-54_id0040_rgb_minmax_u8](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/examples/2026_03_17_11-41-54_id0040_rgb_minmax_u8.png) |

## Acquisition setup

- Camera: Cubert Ultris XMR (serial 254902), operated through CuvisNEXT
- Lens: 50 mm / f2.0
- Light source: four halogen lamps in four configurations per scene arrangement: l0 all four lights, l1 front and back, l2 front and two side lights, l3 front light only
- Working distance: 46.6 cm (field of view about 12.5 x 12 cm)
- Background: blue FDA-compliant conveyor-belt material, belt stationary during capture
- Measurement mode: snapshot, one capture per lighting configuration
- Integration time: 15 ms
- White reference: 55 % gray target
- Dark reference: recorded with the lens covered

The README of each day under `data/<day>/` lists its hardware, lighting, settings, scene and recordings.

## Repository layout

```
README.md
LICENSE                                 (Apache-2.0)
NOTICE.md                               third-party credits
manifest.json                           every file with size and sha256
fetch.py                                stdlib downloader that reads manifest.json (no token)
universe.csv                            one row per frame: source,index,annotation,group[,tags]
splits/                                 baked selector splits (core DataSplitConfig, file_indices)
  dinomaly.json                         unsupervised anomaly detection (Dinomaly): trains on the normal frames only (train 308, val 148, test 180)
  adaclip.json                          supervised baseline (AdaCLIP): the annotated positives are in train (train 808, val 148, test 180)
annotations_canonical/                  per-day concatenated COCO (time-ordered global ids)
  day2_global_coco.json                 384 images, 368 regions
  day3_global_coco.json                 492 images, 648 regions
  day4_global_coco.json                 260 images, 520 regions
assets/                                 example renderings
data/
  day2/
    README.md                             the day's hardware, lighting, settings and scene
    2026_03_03_11-11-01.cu3s              2 frames
    2026_03_03_11-11-01.info
    2026_03_03_11-11-01.json              COCO, image_id = read index
    2026_03_03_11-31-31.cu3s              17 frames
    2026_03_03_11-31-31.info
    2026_03_03_11-31-31.json              COCO, image_id = read index
    2026_03_03_11-38-39.cu3s              81 frames
    2026_03_03_11-38-39.info
    2026_03_03_11-38-39.json              COCO, image_id = read index
    2026_03_03_13-58-04_1.cu3s            96 frames
    2026_03_03_13-58-04_1.info
    2026_03_03_13-58-04_1.json            COCO, image_id = read index
    2026_03_03_13-58-04_2.cu3s            136 frames, 136 labelled
    2026_03_03_13-58-04_2.info
    2026_03_03_13-58-04_2.json            COCO, image_id = read index
    2026_03_03_15-25-02.cu3s              52 frames, 52 labelled
    2026_03_03_15-25-02.info
    2026_03_03_15-25-02.json              COCO, image_id = read index
  day3/
    README.md                             the day's hardware, lighting, settings and scene
    2026_03_10_10-17-20.cu3s              84 frames, 44 labelled
    2026_03_10_10-17-20.info
    2026_03_10_10-17-20.json              COCO, image_id = read index
    2026_03_10_10-58-55.cu3s              36 frames, 36 labelled
    2026_03_10_10-58-55.info
    2026_03_10_10-58-55.json              COCO, image_id = read index
    2026_03_10_11-30-45.cu3s              120 frames
    2026_03_10_11-30-45.info
    2026_03_10_11-30-45.json              COCO, image_id = read index
    2026_03_10_12-00-18.cu3s              40 frames, 40 labelled
    2026_03_10_12-00-18.info
    2026_03_10_12-00-18.json              COCO, image_id = read index
    2026_03_10_14-32-01.cu3s              92 frames, 88 labelled
    2026_03_10_14-32-01.info
    2026_03_10_14-32-01.json              COCO, image_id = read index
    2026_03_10_15-12-17.cu3s              120 frames, 120 labelled
    2026_03_10_15-12-17.info
    2026_03_10_15-12-17.json              COCO, image_id = read index
  day4/
    README.md                             the day's hardware, lighting, settings and scene
    2026_03_17_11-11-50.cu3s              80 frames, 80 labelled
    2026_03_17_11-11-50.info
    2026_03_17_11-11-50.json              COCO, image_id = read index
    2026_03_17_11-41-54.cu3s              80 frames, 40 labelled
    2026_03_17_11-41-54.info
    2026_03_17_11-41-54.json              COCO, image_id = read index
    2026_03_17_14-38-58.cu3s              100 frames, 60 labelled
    2026_03_17_14-38-58.info
    2026_03_17_14-38-58.json              COCO, image_id = read index
```

### Per-recording COCO json

Standard COCO, one file per recording, `image_id` = the read index inside the `.cu3s`:

```jsonc
{
  "info": { "recording": "<stem>", "day": "<day>", "frame_count": N, "annotation_count": M },
  "categories": [ { "id": 0..7, "name": "..." } ],
  "images": [ { "id": <read index>, "file_name": "<stem>.cu3s", "width": W, "height": H, "camera_name": "..." } ],
  "annotations": [ { "id": ..., "image_id": <read index>, "category_id": 1..7, "bbox": [x, y, w, h],
                     "segmentation": [[...]], "iscrowd": 0, "area": 0.0, "mask": {"counts": [], "size": [H, W]} } ]
}
```

Annotations are semantic masks, not instances: objects of the same class in one frame share a polygon contour and
carry no instance ids, and the `mask` field is empty (polygons only). Each image record carries three extra fields
for traceability: `global_frame_id` (the key into the day's concatenated COCO under `annotations_canonical/`),
`camera_frame_num` (the raw camera frame counter, as in the `.info` sidecar) and `camera_name` (`Auto_000_<n>`).
Five recordings hold normal product only; their json files list the frames and no regions, and their frames are
the training set of `splits/dinomaly.json`.

The integer `id` of an annotation in the day-level files under `annotations_canonical/` is its position in
time order and is reassigned whenever those files are rebuilt (they were in the October 2026 rebuild), so it is
not a stable identifier across revisions. A region is identified by `(image_id, category_id, bbox)`, which is
unchanged between revisions.

### `universe.csv` columns

| column | meaning |
|---|---|
| `source` | relative posix path of the recording, e.g. `data/day2/2026_03_03_11-11-01.cu3s` |
| `index` | read position inside the `.cu3s`, equal to the COCO `image_id` |
| `annotation` | relative path of the recording's COCO json, empty for recordings without labels |
| `group` | frames sharing a value stay in one split (the four captures of one scene arrangement under the four lighting configurations form one group (for example `day2_g000000`); all four frames stay in one split so lighting alone cannot leak between splits) |
| `tags` | free labels of the recording, `;`-separated (for example `clean`, `fo`); frames tagged with no tag (nothing is excluded) are kept out of every split |

## Splits

Splits ship as selector files under `splits/` (core `DataSplitConfig`, `file_indices`) that resolve against `universe.csv` by `(source, index)`.

| file | train | val | test | for |
|---|---:|---:|---:|---|
| `splits/dinomaly.json` | 308 | 148 | 180 | unsupervised anomaly detection (Dinomaly): trains on the normal frames only |
| `splits/adaclip.json` | 808 | 148 | 180 | supervised baseline (AdaCLIP): the annotated positives are in train |

`splits/adaclip.json` is the supervised view: train 808 (500 annotated positives and 308 normals), val 148, test

180. `splits/dinomaly.json` is the unsupervised view: train 308 (the normals only), the same val 148 and test 180,
with the 500 annotated positives held out of training. The two files share val and test, so a supervised and an
unsupervised method are evaluated on identical frames.

## How to load

List the test frames of a split:

```python
import json
from huggingface_hub import hf_hub_download

repo = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils"
path = hf_hub_download(repo_id=repo, repo_type="dataset", filename="splits/dinomaly.json")
sel = json.load(open(path))
test = [(s["source"], i) for s in sel["test"] for i in s["ids"]]
print(len(test), "test frames")
```

Read one recording and its labels:

```python
import json, cuvis
from huggingface_hub import hf_hub_download

repo = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils"
stem = "data/day2/2026_03_03_11-11-01"
cu3s = hf_hub_download(repo_id=repo, repo_type="dataset", filename=f"{stem}.cu3s")
labels = json.load(open(hf_hub_download(repo_id=repo, repo_type="dataset", filename=f"{stem}.json")))

cuvis.init()
session = cuvis.SessionFile(cu3s)
ctx = cuvis.ProcessingContext(session)
ctx.processing_mode = cuvis.ProcessingMode.Reflectance
m = ctx.apply(session.get_measurement(0))
cube = m.data["cube"].array            # (H, W, bands)
print(cube.shape, len(labels["images"]), "frames", len(labels["annotations"]), "regions")
```

Everything at once, with the registry:

```bash
uv run dataset download Industrial_FOD_Lentils
```

or without Cuvis.AI: `python fetch.py` from a checkout of this repo, or `hf download cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils --repo-type dataset --local-dir ./XMR_Industrial_Foreign_Object_Detection_Lentils`.

Train on it with the `cu3s_multi` data module and a split file:

```yaml
data:
  data_module: cu3s_multi
  splits:
    splits_path: splits/dinomaly.json
```

## Citation

```bibtex
@techreport{raj2026lentilshsi,
  title       = {Spectral Foreign Object Detection in Lentils Using a Compact Hyperspectral Channel Selector},
  author      = {Raj, Anish},
  institution = {Cubert GmbH},
  year        = {2026},
  note        = {Whitepaper, May 2026},
  url         = {https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils/resolve/main/whitepaper/lentils_hsi_whitepaper.pdf}
}
```

## License

Released under the Apache-2.0 license, see `LICENSE`. Third-party material is credited in `NOTICE.md`.

## Contact

Recorded and processed by the AI team at Cubert GmbH: <cuvis.ai@cubert-gmbh.de>. Reach out for evaluation pilots or to run this methodology on your own product line.
