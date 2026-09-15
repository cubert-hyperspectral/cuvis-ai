Mirrored from HuggingFace

This page mirrors the README of [`cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils`](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils).

# Demo for Hyperspectral Foreign-Object Detection in Lentils

Video spectroscopy beyond the visible spectrum, applied to foreign-object detection on a sliding lentil conveyor. Captured with a Cubert [**Ultris XMR**](https://cubert-hyperspectral.com/de/ultris-xmr/) camera — **61 bands per pixel**, 430–910 nm, 1080 × 1000 pixels at 4 fps.

Foreign-object detection in food sorting is a general industrial-inspection problem — the rejected target could be a stone, a stem, a piece of packaging, a metal shard, or an insect. In this demo the bulk product is bag-grade lentils on a sliding belt; the contaminants are a small set of representative classes (stone, stem, aluminium shard, fly). The same hyperspectral pipeline and the same spectroscopic argument carry over to any other product whose foreign objects differ from the bulk in spectral reflectance — even when they look near-identical in visible RGB.

[](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/lentils_3method_input.mp4)

*Input view — the same lentil-belt sequence rendered as each method's 3-channel projection with no annotations. The belt and foreign objects are visible in the raw material signal; the bands each method selects determine which objects stand out.*

[](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils/resolve/main/assets/lentils_3method_teaser.mp4)

*Overlay teaser — the same frames with GT mask (green), detection (red), and overlap (yellow) added. Visible RGB misses the near-isoluminant stone and fly; CIR and the custom selector recover them via near-infrared contrast but miss out on aluminium shard.*

______________________________________________________________________

## The scene

Lentils slide past the camera on an industrial conveyor belt at constant speed. Foreign objects are sprinkled into the bulk to simulate real contamination events. The dataset captures a single sliding sequence with four contaminant classes:

- `1` **stone** — small mineral fragment, near-isoluminant with lentils in visible RGB.
- `2` **stem_k** — organic stem fragment.
- `3` **alu_shard** — small aluminium piece (packaging contamination).
- `4` **fly** — small insect.

Class `0` is normal/background. Detection is treated as a binary anomaly task: any non-zero label is a foreign object.

The CU3S session contains 69 frames recorded under constant industrial-belt lighting, with **white reference** and **dark current** measured before the run. Annotations are stored in a sibling COCO-style JSON.

## Why hyperspectral

An RGB sensor collapses the incoming light into three bands; the human eye does the same. Hyperspectral video records faithfully at **61 continuous bands per pixel, per frame** — a material fingerprint that separates dyes, fabrics, coatings, pigments, organic vs mineral matter, and surface chemistry.

Foreign objects that are colour-matched to the bulk product (small stones in brown lentils, aluminium shards under industrial lighting) are often near-isoluminant in visible RGB. They typically reveal themselves in the near-infrared (different surface scattering, different moisture content) or in narrow visible bands the eye can't resolve. A learned channel selector picks the three bands that maximise that contrast, and the rest of the pipeline (a standard pre-trained anomaly detector) just sees them as a clean 3-channel image.

The table below shows frame 27 rendered through each method's 3-channel projection (per-channel min-max, uint8). The band-choice argument is clearest here: objects that blend into the lentil background in visible RGB become distinct in the CIR and custom-selector views.

| RGB (650/550/450 nm) | CIR (NIR/R/G) | Custom selector (542/902/886 nm) |
| -------------------- | ------------- | -------------------------------- |
|                      |               |                                  |

## Processing pipeline

The full anomaly-detection pipeline is built in [**Cuvis.AI**](https://docs.cuvis.ai) — Cubert's open-source hyperspectral AI pipeline framework. [**Dinomaly**](https://github.com/guojiajeremy/Dinomaly) (an Anomalib-backed anomaly detector built on a frozen DINOv2 backbone) is plugged in as a third-party plugin node and receives a 3-channel projection of the hyperspectral cube — the only thing that changes between the three runs is which three channels Dinomaly sees.

1. **LentilsAnomalyDataNode** — reads the `.cu3s` cube, converts the cube to float32 and maps the multi-class GT mask to a binary anomaly mask.
1. **MinMaxNormalizer** — running per-cube min-max so the selector sees data in a consistent dynamic range.
1. **FixedWavelengthSelector / CIRSelector** — picks three target wavelengths (or a CIR triple) and emits a 3-channel image. This is the only node that differs between the three methods.
1. **DinomalyDetector** — pre-trained DINOv2 backbone + Dinomaly anomaly head; produces a per-pixel anomaly score map.
1. **TwoStageBinaryDecider** — first gates the frame on the mean of the top-k per-pixel scores (image-level threshold), then quantile-thresholds the score map per pixel.

The trained pipelines (YAML + weights) live on the companion model repo: [**cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils**](https://huggingface.co/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils).

## Demo videos

The following clips render the same 69-frame CU3S session through each of the three method variants. The *input* clip is the 3-channel projection that the detector sees; the *overlay* clip adds GT mask (green), prediction (red), and overlap (yellow) on top of the same projection. All clips are 4 fps to match the acquisition rate.

### Dinomaly RGB — visible-light baseline

[](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils/resolve/main/measurements/cu3s/2026_04_15_13_32_55/Auto_003+01_rgb_input.mp4) [](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils/resolve/main/measurements/cu3s/2026_04_15_13_32_55/Auto_003+01_rgb_overlay.mp4)

*Bands at 650/550/450 nm — what an ordinary colour camera would see. Foreign objects that share the lentils' visible reflectance (stones, aluminium under warm light) are easily missed.*

### Dinomaly CIR — colour-infrared

[](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils/resolve/main/measurements/cu3s/2026_04_15_13_32_55/Auto_003+01_cir_input.mp4) [](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils/resolve/main/measurements/cu3s/2026_04_15_13_32_55/Auto_003+01_cir_overlay.mp4)

*NIR=860 nm, R=670 nm, G=560 nm. Near-infrared exposes organic-vs-mineral differences invisible to RGB; performance jumps for stone and stem.*

### Dinomaly Custom Selector — learned bands

[](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils/resolve/main/measurements/cu3s/2026_04_15_13_32_55/Auto_003+01_concrete_input.mp4) [](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils/resolve/main/measurements/cu3s/2026_04_15_13_32_55/Auto_003+01_concrete_overlay.mp4)

*Three fixed bands at (542, 902, 886) nm chosen by a frozen-AdaCLIP selector experiment — two NIR + one green. Best per-pixel separability on this conveyor scene; cleanest detections of the three.*

## Reproduce locally

A complete walk-through notebook ships with cuvis-ai:

```bash
git clone https://github.com/cubert-hyperspectral/cuvis-ai
cd cuvis-ai && uv sync
uv run jupyter lab notebooks/use_cases/lentils_dinomaly.ipynb
```

The notebook fetches the pipeline YAMLs + weights + this CU3S session on first run via `huggingface_hub.hf_hub_download`, so no manual asset placement is required.

## Learn more

- **Cuvis.AI documentation** — <https://docs.cuvis.ai>
- **Cuvis.AI on GitHub** — <https://github.com/cubert-hyperspectral/cuvis-ai>
- **Tutorial notebook** — `notebooks/use_cases/lentils_dinomaly.ipynb`
- **Companion model card** — <https://huggingface.co/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils>

## Contact

Recorded and processed by the [AI Team @ Cubert](mailto:cuvis.ai@cubert-gmbh.de). Get in touch if you'd like to run a pilot on your own product line.
