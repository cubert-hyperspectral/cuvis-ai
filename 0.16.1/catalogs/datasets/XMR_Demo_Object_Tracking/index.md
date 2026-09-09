Mirrored from HuggingFace

This page mirrors the README of [`cubert-gmbh/XMR_Demo_Object_Tracking`](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Object_Tracking).

# Demo for Hyperspectral Object Tracking

Video spectroscopy beyond the visible spectrum, applied to object tracking in a crowded bus-station scene. Captured with a Cubert [**Ultris XMR**](https://cubert-hyperspectral.com/de/ultris-xmr/) camera — **61 bands per pixel**, 430–910 nm, 1080 × 1000 pixels at 15 Hz.

Object tracking is a general computer-vision problem — the "object" could be a vehicle, a container, a piece of equipment, or an animal. In this demo the target class is humans: a crowded public scene with people in near-identical clothing is the hardest case for a standard RGB tracker and the clearest showcase for what hyperspectral features add. The same pipeline and the same spectroscopic arguments carry over to any other object class whose materials differ beyond the visible range.

[](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Object_Tracking/resolve/main/assets/UC1_invisible_ink_tracking.mp4)

*Teaser — invisible-ink marking. One actor is sprayed with an ink that is invisible to the eye but bright in a specific spectral band; a Spectral Angle Mapper then tracks that signature with no model, no training, and no drift. The zoomed-in crop tracks the centre of the ink blob.*

______________________________________________________________________

## The scene

Actors wear visually identical outfits. T wears a single garment made of a material that looks distinct in **CIR** (colour-infrared), or is sprayed with invisible ink that is visible in a tuned-in spectral view. The dataset has two modes:

- **Passive** — the three actors walk in single file toward or away from the camera, occluding one another along the optical axis. A visible-only tracker tends to swap IDs in this setting; hyperspectral cues preserve the correct lock.
- **Active** — while on camera, one actor sprays T with an "invisible" ink. The ink is visible only in specific spectral bands, adding a trackable spectral signature to T mid-recording.

All captured as spectral radiance (no white reference), under outdoor daylight, camera on a static tripod roughly 100 m from the subject area.

## Why hyperspectral

An RGB sensor collapses the incoming light into three bands; the human eye does the same. Hyperspectral video records faithfully at **61 continuous bands per pixel, per frame** — a material fingerprint that separates dyes, fabrics, coatings, pigments, skin, and foliage. People who are indistinguishable in RGB often separate cleanly in CIR or in a learned spectral projection.

## Two ways to use it

|                 | **Passive**                      | **Active**                         |
| --------------- | -------------------------------- | ---------------------------------- |
| Target access   | none needed                      | ink must be applied                |
| Data / training | rides SOTA RGB training (SAM3)   | none                               |
| Latency         | tracker-bound                    | millisecond, deterministic         |
| Failure mode    | soft — drift, ambiguity          | hard — ink present or absent       |
| Typical use     | surveillance of unknown subjects | friendlies, assets, covert markers |

## Processing pipeline

The full tracking pipeline is built in [**Cuvis.AI**](https://docs.cuvis.ai) — Cubert's open-source hyperspectral AI pipeline framework. [**SAM3**](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) (Meta Platforms Inc.'s Segment Anything Model 3) is plugged in as a third-party plugin node and receives either standard RGB or a spectral false-RGB projection — the only thing that changes between the RGB-ceiling baseline and the hyperspectral run is which three channels SAM3 sees.

1. **CU3SDataNode** — reads the `.cu3s` hyperspectral cube (61 bands / pixel per frame).
1. **CIETristimulusFalseRGBSelector** — projects the spectrum to a 3-channel false-RGB frame.
1. **SAM3MaskPropagation** — runs SAM3 in mask-propagation mode over the clip, seeded once by a `MaskPrompt` node (`--prompt 17:1@65` = seed mask for object 17 at frame 65).
1. **TrackingOverlay + ToVideoNode** — renders the mask contours on the false-RGB frames and encodes an mp4.

## Demo videos

### Passive tracking

[](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Object_Tracking/resolve/main/measurements/cu3s/2026_04_15_16_28_10/Auto_000_gt_vs_pred.mp4)

*Ground truth vs prediction during the passive phase — the three actors walk in single file with mutual occlusions; the hyperspectral tracker preserves the correct ID lock on T.*

### Active tracking

[](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Object_Tracking/resolve/main/measurements/cu3s/2026_04_20_16_28_54/Auto_000_combined.mp4)

*Combined view after T sprays D1 with the spectral ink — the marker becomes a deterministic, training-free signature that the tracker locks onto.*

### All humans (baseline)

[](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Object_Tracking/resolve/main/measurements/cu3s/2026_04_15_16_28_10/Auto_000-all-humans.mp4)

*Untargeted run with every detected person tracked — the same pipeline configured for class-level tracking instead of single-target lock.*

## Learn more

- **Cuvis.AI documentation** — <https://docs.cuvis.ai>
- **Cuvis.AI on GitHub** — <https://github.com/cubert-hyperspectral/cuvis-ai>

## Contact

Recorded and processed by the [AI Team@Cubert](mailto:cuvis.ai@cubert-gmbh.de). Get in touch if you'd like to run a pilot on your own scene.
