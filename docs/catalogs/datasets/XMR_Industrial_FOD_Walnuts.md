!!! info "Mirrored from HuggingFace"
    This page mirrors the README of [`cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts`](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts).

<p align="center">
  <img src="https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png" alt="Cubert Hyperspectral" width="560"/>
</p>

<p align="center">
  <a href="https://docs.cuvis.ai"><img src="https://img.shields.io/badge/Docs-docs.cuvis.ai-0aa?logo=readthedocs&logoColor=white" alt="Cuvis.AI docs"/></a>
  <a href="https://github.com/cubert-hyperspectral/cuvis-ai"><img src="https://img.shields.io/badge/GitHub-cuvis--ai-24292e?logo=github" alt="Cuvis.AI on GitHub"/></a>
  <a href="https://huggingface.co/cubert-gmbh/Industrial_Foreign_Object_Detection_Walnuts"><img src="https://img.shields.io/badge/Model-HuggingFace-ffd21e?logo=huggingface&logoColor=000" alt="Companion model"/></a>
</p>

# Hyperspectral foreign-object detection in walnuts

Walnut kernels and shell fragments with foreign objects, recorded with a Cubert Ultris XMR hyperspectral camera between August and October 2026 for Cubert's foreign-object demonstration: first on a lab station with a blue mat, then on the demonstration stand with production lights, a turntable and a belt. The contaminants are stones, stems, rubber, aluminium shards and plastic imitation shells, placed by hand into the product; the shell fragments are labelled as a class of their own, so the same frames serve shell segmentation and unsupervised foreign-object detection. The trained pipelines fitted on these recordings are published in the companion model repository, cubert-gmbh/Industrial_Foreign_Object_Detection_Walnuts.

Captured with a Cubert Ultris XMR camera (61 bands per pixel, 430 to 910 nm, 1080 x 1000 pixels). 7 acquisition days, 90 `.cu3s` recordings, 2,283 frames, 820 of them with pixel-level COCO annotations across 7 classes.

## Summary

| | |
|---|---:|
| Total frames | **2,283** |
| Annotated frames | **820** (35.9 %) |
| Annotated regions | **7,244** |
| Hyperspectral cubes (`.cu3s` recordings) | **90** |
| Spectral resolution | **61 bands per pixel, 430 to 910 nm, 1080 x 1000 pixels** |
| Processing mode | **Reflectance** (white and dark reference recorded per session) |
| Splits | fo_v4: train 767, val 34, test 728; fo_v5a: train 322, val 34, test 259; fo_v5b: train 819, val 34, test 149; seg_v2: train 279, val 28, test 142 |
| Total size on disk | **about 125.0 GB** |
| License | **Apache-2.0** |

### Per-day breakdown

| Day | Capture date | Recordings | Frames | Annotated | Regions |
|---|---|---:|---:|---:|---:|
| 2026_08_18 | 2026-08-18 | 23 | 304 | 236 | 2,781 |
| 2026_09_01 | 2026-09-01 | 18 | 106 | 106 | 586 |
| 2026_09_15 | 2026-09-15 | 3 | 14 | 14 | 98 |
| 2026_09_22 | 2026-09-22 | 18 | 99 | 99 | 763 |
| 2026_09_30 | 2026-09-30 | 3 | 495 | 106 | 1,146 |
| 2026_10_01 | 2026-10-01 | 24 | 1,143 | 259 | 1,870 |
| 2026_10_02 | 2026-10-02 | 1 | 122 | 0 | 0 |
| **Total** | | **90** | **2,283** | **820** | **7,244** |

## Classes

| id | name | regions |
|---:|---|---:|
| 1 | `shell` | 5,381 |
| 2 | `stone` | 496 |
| 3 | `stem` | 133 |
| 4 | `rubber` | 103 |
| 5 | `alu_shard` | 124 |
| 6 | `fake_shell` | 689 |
| 7 | `kernel` | 318 |

Class id 0 is the implicit background. A recording with no annotation file holds normal product only; its frames are in `universe.csv` and carry no regions.

## Why hyperspectral

A shell fragment and a kernel share their colour; they separate in the near infrared (the discriminative bands of this product lie at 894 to 910 nm and 718 to 742 nm), where a linear per-pixel rule already tells them apart on about 90 % of the pixels. The plastic imitation shells are indistinguishable from real shells in RGB and obvious in a colour-infrared projection (a flat near-infrared reflectance turns them green). Stones, rubber and aluminium are spectrally distinct from every normal walnut material (one-class Mahalanobis AUROC 0.96 on the first session) while they hide in RGB. The example frames show the same cube through an RGB projection (650, 550, 450 nm) and a colour-infrared projection (870, 640, 550 nm).

## Example frames

| 2026_10_01_test_kernel_shell_fake_007_i020_cir | 2026_10_01_test_kernel_shell_fake_007_i020_rgb | 2026_10_01_test_kernel_shell_fake_stone_alu_stem_rubber_008_i024_cir |
|:---:|:---:|:---:|
| ![2026_10_01_test_kernel_shell_fake_007_i020_cir](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts/resolve/main/assets/examples/2026_10_01_test_kernel_shell_fake_007_i020_cir.png) | ![2026_10_01_test_kernel_shell_fake_007_i020_rgb](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts/resolve/main/assets/examples/2026_10_01_test_kernel_shell_fake_007_i020_rgb.png) | ![2026_10_01_test_kernel_shell_fake_stone_alu_stem_rubber_008_i024_cir](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts/resolve/main/assets/examples/2026_10_01_test_kernel_shell_fake_stone_alu_stem_rubber_008_i024_cir.png) |

| 2026_10_01_test_kernel_shell_fake_stone_alu_stem_rubber_008_i024_rgb | 2026_10_01_test_kernel_shell_stones_004_i010_cir | 2026_10_01_test_kernel_shell_stones_004_i010_rgb |
|:---:|:---:|:---:|
| ![2026_10_01_test_kernel_shell_fake_stone_alu_stem_rubber_008_i024_rgb](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts/resolve/main/assets/examples/2026_10_01_test_kernel_shell_fake_stone_alu_stem_rubber_008_i024_rgb.png) | ![2026_10_01_test_kernel_shell_stones_004_i010_cir](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts/resolve/main/assets/examples/2026_10_01_test_kernel_shell_stones_004_i010_cir.png) | ![2026_10_01_test_kernel_shell_stones_004_i010_rgb](https://huggingface.co/datasets/cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts/resolve/main/assets/examples/2026_10_01_test_kernel_shell_stones_004_i010_rgb.png) |

## Acquisition setup

- Camera: Cubert Ultris XMR (VNIR, 61 bands 430 to 910 nm, 1080 x 1000 pixels), operated through CuvisNEXT
- Lens: standard lens of the station
- Light source: 2026_08_18: the four lamps of the lab station (the 15 September live test showed four lights best, three acceptable, two or fewer failing); 2026_09_01: the four lamps of the lab station (the 15 September live test showed four lights best, three acceptable, two or fewer failing); 2026_09_15: the four lamps of the lab station (the 15 September live test showed four lights best, three acceptable, two or fewer failing); 2026_09_22: the four lamps of the lab station (the 15 September live test showed four lights best, three acceptable, two or fewer failing); 2026_09_30: two production LED lights, one on each side, each at 20 degrees; 2026_10_01: two production LED lights, one on each side, each at 20 degrees; 2026_10_02: two production LED lights, one on each side, each at 20 degrees
- Working distance: 2026_08_18: 50.5 cm (field of view about 19 x 17.5 cm, about 0.176 mm per pixel); 2026_09_01: 50.5 cm (field of view about 19 x 17.5 cm, about 0.176 mm per pixel); 2026_09_15: 50.5 cm (field of view about 19 x 17.5 cm, about 0.176 mm per pixel); 2026_09_22: 50.5 cm (field of view about 19 x 17.5 cm, about 0.176 mm per pixel); 2026_09_30: 33 cm above the product; 2026_10_01: 33 cm above the product; 2026_10_02: 33 cm above the product
- Background: 2026_08_18: blue mat; 2026_09_01: blue mat; 2026_09_15: blue mat; 2026_09_22: blue mat; 2026_09_30: the dark belt of the demonstration stand; 2026_10_01: the dark belt of the demonstration stand; 2026_10_02: the dark belt of the demonstration stand
- Measurement mode: 2026_08_18: snapshot recordings in Internal mode, several frames per scene; 2026_09_01: snapshot recordings in Internal mode, several frames per scene; 2026_09_15: snapshot recordings in Internal mode, several frames per scene; 2026_09_22: snapshot recordings in Internal mode, several frames per scene; 2026_09_30: continuous recording in Internal mode; 2026_10_01: continuous recording in Internal mode; 2026_10_02: continuous recording in Internal mode
- Integration time: 2026_08_18: 8.0 ms (auto exposure settled; constant through the day); 2026_09_01: 12.0 ms (auto exposure settled); 2026_09_15: 12.0 ms; 2026_09_22: 12.0 ms (every recording); 2026_09_30: 50 ms; 2026_10_01: 50 ms; 2026_10_02: 50 ms
- White reference: 2026_08_18: 55 % reflectance target, embedded in every recording; 2026_09_01: 55 % reflectance target, embedded in every recording; 2026_09_15: 55 % reflectance target, embedded in every recording; 2026_09_22: 55 % reflectance target, embedded in every recording; 2026_09_30: the stand's white target under the camera at the camera's integration time; 2026_10_01: the stand's white target under the camera at the camera's integration time; 2026_10_02: the stand's white target under the camera at the camera's integration time
- Dark reference: 2026_08_18: recorded, embedded in every recording; 2026_09_01: recorded, embedded in every recording; 2026_09_15: recorded, embedded in every recording; 2026_09_22: recorded, embedded in every recording; 2026_09_30: recorded with the lens covered; 2026_10_01: recorded with the lens covered; 2026_10_02: recorded with the lens covered

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
  fo_v4.json                            the foreign-object study's evaluation split: clean lab frames in train, the 34 normaliser frames in val, every session's test buckets (clean, foreign objects, imitation shells, the turntable, the 1 October stand) in test (train 767, val 34, test 728)
  fo_v5a.json                           bank arm a: clean frames of every session except the 1 October afternoon, which is the test (train 322, val 34, test 259)
  fo_v5b.json                           bank arm b: clean frames of every session except the 1 October morning, which is the test (train 819, val 34, test 149)
  seg_v2.json                           the shell segmentation split (RF-DETR v2): 18 August, 1 September and 22 September frames (train 279, val 28, test 142)
annotations_canonical/                  per-day concatenated COCO (time-ordered global ids)
  2026_08_18_global_coco.json           236 images, 2781 regions
  2026_09_01_global_coco.json           106 images, 586 regions
  2026_09_15_global_coco.json           14 images, 98 regions
  2026_09_22_global_coco.json           99 images, 763 regions
  2026_09_30_global_coco.json           106 images, 1146 regions
  2026_10_01_global_coco.json           259 images, 1870 regions
  2026_10_02_global_coco.json           0 images, 0 regions
assets/                                 example renderings
data/
  2026_08_18/
    README.md                             the day's hardware, lighting, settings and scene
    11-02-52/kernels_only_000.cu3s        1 frames
    11-02-52/kernels_only_000.info
    11-02-52/kernels_only_000+01.cu3s     20 frames
    11-02-52/kernels_only_000+01.info
    11-02-52/kernels_only_less_dense_000.cu3s1 frames
    11-02-52/kernels_only_less_dense_000.info
    11-02-52/kernels_only_less_dense_000+02.cu3s21 frames
    11-02-52/kernels_only_less_dense_000+02.info
    11-02-52/kernels_only_removed_1_small_skin_000.cu3s1 frames
    11-02-52/kernels_only_removed_1_small_skin_000.info
    11-02-52/kernels_only_removed_1_small_skin_000+01.cu3s21 frames
    11-02-52/kernels_only_removed_1_small_skin_000+01.info
    11-02-52/shells_only_000.cu3s         1 frames, 1 labelled
    11-02-52/shells_only_000.info
    11-02-52/shells_only_000.json         COCO, image_id = read index
    11-02-52/shells_only_000+02.cu3s      22 frames, 21 labelled
    11-02-52/shells_only_000+02.info
    11-02-52/shells_only_000+02.json      COCO, image_id = read index
    11-02-52/shells_only_less_dense_000.cu3s1 frames, 1 labelled
    11-02-52/shells_only_less_dense_000.info
    11-02-52/shells_only_less_dense_000.jsonCOCO, image_id = read index
    11-02-52/shells_only_less_dense_000+02.cu3s21 frames, 21 labelled
    11-02-52/shells_only_less_dense_000+02.info
    11-02-52/shells_only_less_dense_000+02.jsonCOCO, image_id = read index
    11-02-52/shells_with_kernel_inside_000.cu3s1 frames, 1 labelled
    11-02-52/shells_with_kernel_inside_000.info
    11-02-52/shells_with_kernel_inside_000.jsonCOCO, image_id = read index
    11-02-52/shells_with_kernel_inside_000+02.cu3s33 frames, 33 labelled
    11-02-52/shells_with_kernel_inside_000+02.info
    11-02-52/shells_with_kernel_inside_000+02.jsonCOCO, image_id = read index
    11-02-52/whole_shells.cu3s            25 frames, 23 labelled
    11-02-52/whole_shells.info
    11-02-52/whole_shells.json            COCO, image_id = read index
    13-24-28/shells_with_kernels_separate_a_000.cu3s41 frames, 41 labelled
    13-24-28/shells_with_kernels_separate_a_000.info
    13-24-28/shells_with_kernels_separate_a_000.jsonCOCO, image_id = read index
    13-24-28/shells_with_kernels_separate_a_000_0015.cu3s1 frames, 1 labelled
    13-24-28/shells_with_kernels_separate_a_000_0015.info
    13-24-28/shells_with_kernels_separate_a_000_0015.jsonCOCO, image_id = read index
    13-24-28/shells_with_kernels_separate_a_000_0026.cu3s1 frames, 1 labelled
    13-24-28/shells_with_kernels_separate_a_000_0026.info
    13-24-28/shells_with_kernels_separate_a_000_0026.jsonCOCO, image_id = read index
    13-24-28/shells_with_kernels_separate_a_000_0039.cu3s1 frames, 1 labelled
    13-24-28/shells_with_kernels_separate_a_000_0039.info
    13-24-28/shells_with_kernels_separate_a_000_0039.jsonCOCO, image_id = read index
    13-24-28/shells_with_kernels_separate_a_000_0665.cu3s1 frames, 1 labelled
    13-24-28/shells_with_kernels_separate_a_000_0665.info
    13-24-28/shells_with_kernels_separate_a_000_0665.jsonCOCO, image_id = read index
    13-24-28/shells_with_kernels_separate_b_000.cu3s1 frames, 1 labelled
    13-24-28/shells_with_kernels_separate_b_000.info
    13-24-28/shells_with_kernels_separate_b_000.jsonCOCO, image_id = read index
    13-24-28/shells_with_kernels_separate_b_000+01.cu3s21 frames, 21 labelled
    13-24-28/shells_with_kernels_separate_b_000+01.info
    13-24-28/shells_with_kernels_separate_b_000+01.jsonCOCO, image_id = read index
    13-24-28/shells_with_kernels_separate_with_fo_a_000.cu3s26 frames, 26 labelled
    13-24-28/shells_with_kernels_separate_with_fo_a_000.info
    13-24-28/shells_with_kernels_separate_with_fo_a_000.jsonCOCO, image_id = read index
    13-24-28/shells_with_kernels_separate_with_fo_b_000.cu3s1 frames, 1 labelled
    13-24-28/shells_with_kernels_separate_with_fo_b_000.info
    13-24-28/shells_with_kernels_separate_with_fo_b_000.jsonCOCO, image_id = read index
    13-24-28/shells_with_kernels_separate_with_fo_b_000+01.cu3s41 frames, 41 labelled
    13-24-28/shells_with_kernels_separate_with_fo_b_000+01.info
    13-24-28/shells_with_kernels_separate_with_fo_b_000+01.jsonCOCO, image_id = read index
  2026_09_01/
    README.md                             the day's hardware, lighting, settings and scene
    11-01-40/fake_shells_a_000.cu3s       1 frames, 1 labelled
    11-01-40/fake_shells_a_000.info
    11-01-40/fake_shells_a_000.json       COCO, image_id = read index
    11-01-40/fake_shells_a_000+01.cu3s    7 frames, 7 labelled
    11-01-40/fake_shells_a_000+01.info
    11-01-40/fake_shells_a_000+01.json    COCO, image_id = read index
    11-01-40/fake_shells_b_000.cu3s       8 frames, 8 labelled
    11-01-40/fake_shells_b_000.info
    11-01-40/fake_shells_b_000.json       COCO, image_id = read index
    11-01-40/fake_shells_b_000_0013.cu3s  1 frames, 1 labelled
    11-01-40/fake_shells_b_000_0013.info
    11-01-40/fake_shells_b_000_0013.json  COCO, image_id = read index
    11-01-40/fake_shells_b_000_0052.cu3s  1 frames, 1 labelled
    11-01-40/fake_shells_b_000_0052.info
    11-01-40/fake_shells_b_000_0052.json  COCO, image_id = read index
    11-01-40/fake_shells_real_shells_kernels_a_000.cu3s13 frames, 13 labelled
    11-01-40/fake_shells_real_shells_kernels_a_000.info
    11-01-40/fake_shells_real_shells_kernels_a_000.jsonCOCO, image_id = read index
    11-01-40/fake_shells_real_shells_kernels_a_000_0067.cu3s1 frames, 1 labelled
    11-01-40/fake_shells_real_shells_kernels_a_000_0067.info
    11-01-40/fake_shells_real_shells_kernels_a_000_0067.jsonCOCO, image_id = read index
    11-01-40/fake_shells_real_shells_kernels_b_000.cu3s13 frames, 13 labelled
    11-01-40/fake_shells_real_shells_kernels_b_000.info
    11-01-40/fake_shells_real_shells_kernels_b_000.jsonCOCO, image_id = read index
    11-01-40/fake_shells_real_shells_kernels_b_000_0108.cu3s1 frames, 1 labelled
    11-01-40/fake_shells_real_shells_kernels_b_000_0108.info
    11-01-40/fake_shells_real_shells_kernels_b_000_0108.jsonCOCO, image_id = read index
    11-01-40/fake_shells_real_shells_kernels_b_000_0276.cu3s1 frames, 1 labelled
    11-01-40/fake_shells_real_shells_kernels_b_000_0276.info
    11-01-40/fake_shells_real_shells_kernels_b_000_0276.jsonCOCO, image_id = read index
    11-01-40/fake_upside_down_with_real_shells_kernels_a_000.cu3s9 frames, 9 labelled
    11-01-40/fake_upside_down_with_real_shells_kernels_a_000.info
    11-01-40/fake_upside_down_with_real_shells_kernels_a_000.jsonCOCO, image_id = read index
    11-01-40/fake_upside_down_with_real_shells_kernels_a_000_0026.cu3s1 frames, 1 labelled
    11-01-40/fake_upside_down_with_real_shells_kernels_a_000_0026.info
    11-01-40/fake_upside_down_with_real_shells_kernels_a_000_0026.jsonCOCO, image_id = read index
    11-01-40/fake_upside_down_with_real_shells_kernels_b_000.cu3s13 frames, 13 labelled
    11-01-40/fake_upside_down_with_real_shells_kernels_b_000.info
    11-01-40/fake_upside_down_with_real_shells_kernels_b_000.jsonCOCO, image_id = read index
    11-01-40/fake_upside_down_with_real_shells_kernels_b_000_0122.cu3s1 frames, 1 labelled
    11-01-40/fake_upside_down_with_real_shells_kernels_b_000_0122.info
    11-01-40/fake_upside_down_with_real_shells_kernels_b_000_0122.jsonCOCO, image_id = read index
    11-01-40/real_shells_kernels_000.cu3s 1 frames, 1 labelled
    11-01-40/real_shells_kernels_000.info
    11-01-40/real_shells_kernels_000.json COCO, image_id = read index
    11-01-40/real_shells_kernels_000+01.cu3s7 frames, 7 labelled
    11-01-40/real_shells_kernels_000+01.info
    11-01-40/real_shells_kernels_000+01.jsonCOCO, image_id = read index
    12-16-58/real_world_live_000.cu3s     26 frames, 26 labelled
    12-16-58/real_world_live_000.info
    12-16-58/real_world_live_000.json     COCO, image_id = read index
    12-16-58/real_world_live_000_0031.cu3s1 frames, 1 labelled
    12-16-58/real_world_live_000_0031.info
    12-16-58/real_world_live_000_0031.jsonCOCO, image_id = read index
  2026_09_15/
    README.md                             the day's hardware, lighting, settings and scene
    13-50-47/real_fake_and_kernel_overlap_000.cu3s1 frames, 1 labelled
    13-50-47/real_fake_and_kernel_overlap_000.info
    13-50-47/real_fake_and_kernel_overlap_000.jsonCOCO, image_id = read index
    13-50-47/real_fake_and_kernel_overlap_000+01.cu3s11 frames, 11 labelled
    13-50-47/real_fake_and_kernel_overlap_000+01.info
    13-50-47/real_fake_and_kernel_overlap_000+01.jsonCOCO, image_id = read index
    13-50-47/real_fake_and_kernel_overlap_000+02.cu3s2 frames, 2 labelled
    13-50-47/real_fake_and_kernel_overlap_000+02.info
    13-50-47/real_fake_and_kernel_overlap_000+02.jsonCOCO, image_id = read index
  2026_09_22/
    README.md                             the day's hardware, lighting, settings and scene
    14-39-28/kernel_shell_c_000.cu3s      13 frames, 13 labelled
    14-39-28/kernel_shell_c_000.info
    14-39-28/kernel_shell_c_000.json      COCO, image_id = read index
    14-39-28/kernel_shell_d_000.cu3s      1 frames, 1 labelled
    14-39-28/kernel_shell_d_000.info
    14-39-28/kernel_shell_d_000.json      COCO, image_id = read index
    14-39-28/kernel_shell_d_000+01.cu3s   7 frames, 7 labelled
    14-39-28/kernel_shell_d_000+01.info
    14-39-28/kernel_shell_d_000+01.json   COCO, image_id = read index
    14-39-28/kernel_shell_fakes_c_000.cu3s1 frames, 1 labelled
    14-39-28/kernel_shell_fakes_c_000.info
    14-39-28/kernel_shell_fakes_c_000.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_fakes_c_000+05.cu3s14 frames, 14 labelled
    14-39-28/kernel_shell_fakes_c_000+05.info
    14-39-28/kernel_shell_fakes_c_000+05.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_fakes_d_000.cu3s1 frames, 1 labelled
    14-39-28/kernel_shell_fakes_d_000.info
    14-39-28/kernel_shell_fakes_d_000.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_fakes_d_000+06.cu3s7 frames, 7 labelled
    14-39-28/kernel_shell_fakes_d_000+06.info
    14-39-28/kernel_shell_fakes_d_000+06.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_fo_d_000.cu3s   1 frames, 1 labelled
    14-39-28/kernel_shell_fo_d_000.info
    14-39-28/kernel_shell_fo_d_000.json   COCO, image_id = read index
    14-39-28/kernel_shell_fo_d_000+04.cu3s6 frames, 6 labelled
    14-39-28/kernel_shell_fo_d_000+04.info
    14-39-28/kernel_shell_fo_d_000+04.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_fo_hand_c_000.cu3s1 frames, 1 labelled
    14-39-28/kernel_shell_fo_hand_c_000.info
    14-39-28/kernel_shell_fo_hand_c_000.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_fo_hand_c_000+03.cu3s5 frames, 5 labelled
    14-39-28/kernel_shell_fo_hand_c_000+03.info
    14-39-28/kernel_shell_fo_hand_c_000+03.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_hand_other_c_000.cu3s1 frames, 1 labelled
    14-39-28/kernel_shell_hand_other_c_000.info
    14-39-28/kernel_shell_hand_other_c_000.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_hand_other_c_000+02.cu3s10 frames, 10 labelled
    14-39-28/kernel_shell_hand_other_c_000+02.info
    14-39-28/kernel_shell_hand_other_c_000+02.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_hand_other_d_000.cu3s1 frames, 1 labelled
    14-39-28/kernel_shell_hand_other_d_000.info
    14-39-28/kernel_shell_hand_other_d_000.jsonCOCO, image_id = read index
    14-39-28/kernel_shell_hand_other_d_000+03.cu3s11 frames, 11 labelled
    14-39-28/kernel_shell_hand_other_d_000+03.info
    14-39-28/kernel_shell_hand_other_d_000+03.jsonCOCO, image_id = read index
    14-39-28/shell_only_c_000.cu3s        11 frames, 11 labelled
    14-39-28/shell_only_c_000.info
    14-39-28/shell_only_c_000.json        COCO, image_id = read index
    14-39-28/shell_only_d_000.cu3s        1 frames, 1 labelled
    14-39-28/shell_only_d_000.info
    14-39-28/shell_only_d_000.json        COCO, image_id = read index
    14-39-28/shell_only_d_000+01.cu3s     7 frames, 7 labelled
    14-39-28/shell_only_d_000+01.info
    14-39-28/shell_only_d_000+01.json     COCO, image_id = read index
  2026_09_30/
    README.md                             the day's hardware, lighting, settings and scene
    11-11-03/4fps_rotate_000.cu3s         116 frames
    11-11-03/4fps_rotate_000.info
    11-11-03/5fps_rotate_002.cu3s         106 frames, 106 labelled
    11-11-03/5fps_rotate_002.info
    11-11-03/5fps_rotate_002.json         COCO, image_id = read index
    11-11-03/8fps_rotate_001.cu3s         273 frames
    11-11-03/8fps_rotate_001.info
  2026_10_01/
    README.md                             the day's hardware, lighting, settings and scene
    10-10-58/clean_a_000.cu3s             25 frames
    10-10-58/clean_a_000.info
    10-55-30/clean_b_001.cu3s             20 frames
    10-55-30/clean_b_001.info
    10-57-47/clean_c_002.cu3s             20 frames
    10-57-47/clean_c_002.info
    10-59-25/fo_c_stones_003.cu3s         17 frames
    10-59-25/fo_c_stones_003.info
    11-01-02/fo_c_stone_004.cu3s          18 frames
    11-01-02/fo_c_stone_004.info
    11-01-25/fo_c_rubber_005.cu3s         19 frames
    11-01-25/fo_c_rubber_005.info
    11-02-05/fo_c_alu_006.cu3s            17 frames
    11-02-05/fo_c_alu_006.info
    11-03-31/fo_c_stem_007.cu3s           17 frames
    11-03-31/fo_c_stem_007.info
    11-04-04/fo_c_stem_rubber_alu_stone_008.cu3s22 frames
    11-04-04/fo_c_stem_rubber_alu_stone_008.info
    11-05-34/clean_c_009.cu3s             19 frames
    11-05-34/clean_c_009.info
    11-47-51/train_kernel_shell_000.cu3s  27 frames
    11-47-51/train_kernel_shell_000.info
    11-51-28/train_kernel_shell_2_001.cu3s29 frames
    11-51-28/train_kernel_shell_2_001.info
    11-52-37/val_kernel_shell_002.cu3s    28 frames, 28 labelled
    11-52-37/val_kernel_shell_002.info
    11-52-37/val_kernel_shell_002.json    COCO, image_id = read index
    11-54-18/test_kernel_shell_003.cu3s   31 frames, 31 labelled
    11-54-18/test_kernel_shell_003.info
    11-54-18/test_kernel_shell_003.json   COCO, image_id = read index
    11-55-43/test_kernel_shell_stones_004.cu3s49 frames, 49 labelled
    11-55-43/test_kernel_shell_stones_004.info
    11-55-43/test_kernel_shell_stones_004.jsonCOCO, image_id = read index
    12-00-30/test_kernel_shell_alu_005.cu3s34 frames, 34 labelled
    12-00-30/test_kernel_shell_alu_005.info
    12-00-30/test_kernel_shell_alu_005.jsonCOCO, image_id = read index
    12-01-04/test_kernel_shell_stem_006.cu3s26 frames, 26 labelled
    12-01-04/test_kernel_shell_stem_006.info
    12-01-04/test_kernel_shell_stem_006.jsonCOCO, image_id = read index
    12-01-50/test_kernel_shell_fake_007.cu3s44 frames, 44 labelled
    12-01-50/test_kernel_shell_fake_007.info
    12-01-50/test_kernel_shell_fake_007.jsonCOCO, image_id = read index
    12-02-35/test_kernel_shell_fake_stone_alu_stem_rubber_008.cu3s47 frames, 47 labelled
    12-02-35/test_kernel_shell_fake_stone_alu_stem_rubber_008.info
    12-02-35/test_kernel_shell_fake_stone_alu_stem_rubber_008.jsonCOCO, image_id = read index
    12-59-31/only_background_material_000.cu3s210 frames
    12-59-31/only_background_material_000.info
    13-42-55/train_shell_only_000.cu3s    107 frames
    13-42-55/train_shell_only_000.info
    14-40-27/train_kernel_only_001.cu3s   123 frames
    14-40-27/train_kernel_only_001.info
    14-44-41/train_shell_kernel_dense_002.cu3s113 frames
    14-44-41/train_shell_kernel_dense_002.info
    14-46-25/train_shell_kernel_dense_003.cu3s81 frames
    14-46-25/train_shell_kernel_dense_003.info
  2026_10_02/
    README.md                             the day's hardware, lighting, settings and scene
    14-27-46/fps_000.cu3s                 122 frames
    14-27-46/fps_000.info
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

Only targets are labelled: shell fragments (class 1), the foreign-object materials (stone 2, stem 3, rubber 4, aluminium shard 5) and the plastic imitation shells (fake_shell 6). Kernels, hands, other objects, the mat and the belt are unlabelled background, except in the 30 September turntable recording, where the kernels are labelled too (class 7, the only recording with it). Polygons were drawn in CuvisNEXT with SAM3 assistance and normalised: category ids by name, video tracks to per-frame annotations, and every geometry decoded to a binary mask stored as an uncompressed RLE (`mask.counts` as a list, `size` as `[H, W]`); the `segmentation` polygons are kept where the export had them and `track_id` links the frames of one object. One annotation of an undecidable object (`unknown_not_sure`) in the turntable recording was dropped. Recordings without a json are unlabelled.

### `universe.csv` columns

| column | meaning |
|---|---|
| `source` | relative posix path of the recording, e.g. `data/2026_08_18/11-02-52/shells_only_000.cu3s` |
| `index` | read position inside the `.cu3s`, equal to the COCO `image_id` |
| `annotation` | relative path of the recording's COCO json, empty for recordings without labels |
| `group` | frames sharing a value stay in one split (the recording stem under its day, with `+NN` continuation parts merged) |
| `tags` | free labels of the recording, `;`-separated (for example `clean`, `fo`); frames tagged `hand` are kept out of every split |

## Splits

Splits ship as selector files under `splits/` (core `DataSplitConfig`, `file_indices`) that resolve against `universe.csv` by `(source, index)`.

| file | train | val | test | for |
|---|---:|---:|---:|---|
| `splits/fo_v4.json` | 767 | 34 | 728 | the foreign-object study's evaluation split: clean lab frames in train, the 34 normaliser frames in val, every session's test buckets (clean, foreign objects, imitation shells, the turntable, the 1 October stand) in test |
| `splits/fo_v5a.json` | 322 | 34 | 259 | bank arm a: clean frames of every session except the 1 October afternoon, which is the test |
| `splits/fo_v5b.json` | 819 | 34 | 149 | bank arm b: clean frames of every session except the 1 October morning, which is the test |
| `splits/seg_v2.json` | 279 | 28 | 142 | the shell segmentation split (RF-DETR v2): 18 August, 1 September and 22 September frames. Caveat: drawn frame by frame inside the 18 August and 1 September recordings before the recording became the leakage unit, so its 28 validation frames come from recordings whose other frames are in train; the test frames come from other recordings, and the 22 September material is split by recording (c in train, d in test) (9 groups appear in more than one stage.) |

`fo_v4` is the evaluation split of the foreign-object study: train holds the clean frames of the lab sessions, val the 34 frames the score normalisers were fitted on, and test every session's held-out buckets (clean, foreign objects, imitation shells, the 30 September turntable, the 1 October stand). `fo_v5a` and `fo_v5b` are the bank arms of the refit study: a bank may use the clean frames of every session except one held-out 1 October session (the afternoon for a, the morning for b), which is the test; a recording is the unit, a recording with any foreign-object frame is never banked, the 30 September turntable is never banked. `seg_v2` is the shell segmentation split of the RF-DETR models (18 August, 1 September and 22 September); it predates the recording-as-unit rule, so in the 18 August and 1 September recordings its validation frames share recordings with train (the splits README says so). Frames tagged `hand` (the 22 September hand recordings and the 1 September live recording) stay out of every published split; the internal evaluation used them as extra test buckets.

## How to load

List the test frames of a split:

```python
import json
from huggingface_hub import hf_hub_download

repo = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts"
path = hf_hub_download(repo_id=repo, repo_type="dataset", filename="splits/fo_v4.json")
sel = json.load(open(path))
test = [(s["source"], i) for s in sel["test"] for i in s["ids"]]
print(len(test), "test frames")
```

Read one recording and its labels:

```python
import json, cuvis
from huggingface_hub import hf_hub_download

repo = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts"
stem = "data/2026_08_18/11-02-52/shells_only_000"
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
uv run dataset download Industrial_FOD_Walnuts
```

or without Cuvis.AI: `python fetch.py` from a checkout of this repo, or `hf download cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts --repo-type dataset --local-dir ./XMR_Industrial_Foreign_Object_Detection_Walnuts`.

Train on it with the `cu3s_multi` data module and a split file:

```yaml
data:
  data_module: cu3s_multi
  splits:
    splits_path: splits/fo_v4.json
```



## License

Released under the Apache-2.0 license, see `LICENSE`. Third-party material is credited in `NOTICE.md`.

## Contact

Recorded and processed by the AI team at Cubert GmbH: <cuvis.ai@cubert-gmbh.de>. Reach out for evaluation pilots or to run this methodology on your own product line.
