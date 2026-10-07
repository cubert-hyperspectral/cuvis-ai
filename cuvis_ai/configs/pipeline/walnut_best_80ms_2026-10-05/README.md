# walnut_best_80ms_2026-10-05: walnut_best calibrated for the 5 Oct live setup (80 ms)

The six walnut_best pipelines with the stand thresholds of 5 Oct 2026: camera at 80 ms, white and dark taken that
morning at 80 ms, clean turn `Cubert/2026_10_05/10-55-26/Auto_clean_without_fake_80ms_005.cu3s` (93 frames: real
walnuts, shells and kernels only; corner covered). Rule: threshold = 1.10 x the highest clean frame score; mask
threshold = 1.10 (refit) / 1.0 (original) x the highest clean pixel. Only the gate thresholds differ from walnut_best.

| file | threshold | mask_threshold |
|---|---|---|
| walnut_best_original_shellaware | 1.4816 | 1.3926 |
| walnut_best_refit_shellaware | 1.5016 | 1.5652 |
| walnut_best_original (+ _cut) | 1.5531 | 1.5305 |
| walnut_best_refit (+ _cut) | 1.5727 | 1.6212 |

Checked offline on Thor (5 Oct):
- FO recordings at 80 ms: 10-56-42/Auto_fo_80ms_007 (117 frames) and 09-22-11/Auto_000 (100): all frames flagged by
  both shell-aware pipelines and the original; the refit 115/117.
- Clean turn above: no frame flagged (the calibration data itself).
- The shell-aware versions do not mark the real shells next to an FO; the cut versions are about 20 ms slower on Thor
  and lose stem marks: do not use them here.

Valid only for this setup: recalibrate after any change of integration time, light or references (50 ms needs its
own values: shell-aware refit 1.5101 / 1.6362, original 1.5721 / 1.5370; the plain versions miss FO frames at 50 ms).

## smooth3/ and the dark lentils (5 Oct afternoon)

`smooth3/` holds the six pipelines above with `smooth_k: 3` (the gate decides on the median of its last 3 frame
scores: a one-frame spike or dip does not flip it; about one frame later), same thresholds, plus:

- **`walnut_best_original_shellaware_smooth3_px135` (load this first)**: the shell-aware original with the pixel
  threshold of the composite smooth3 (1.35) and the calibrated gate threshold (1.4816). Same weights as the composite.

`walnut_combo_gated_fp16trt_trt16_composite_smooth3_cuvisnext_cube` (this folder): the composite smooth3 with the
original's 80 ms values (1.5531 / 1.5305). Not recommended: its pixel threshold loses most dark lentils.

Checked on Thor against `walnut_combo/composite/smooth3` (1.35 / 1.35). Dark objects found in the image (dark, lying
free on the belt), share of frames in which each is marked; FO 10-56-42/..._007 (117 frames) / 09-22-11/Auto_000 (100):

| pipeline | gate / pixel threshold | dark lentils | stems, dark pieces | real-shell marks | clean frames flagged |
|---|---|---|---|---|---|
| composite smooth3 (walnut_combo) | 1.35 / 1.35 | 75 % / 99 % | 100 % | 40 / 117 frames; 6071 px | 1 / 93 |
| composite smooth3, 80 ms values | 1.5531 / 1.5305 | 33 % / 70 % | 91-100 % | 2 / 117; 706 px | 0 / 93 |
| original shell-aware smooth3 | 1.4816 / 1.3926 | 63 % / 94 % | 98-100 % | 0 / 117; 402 px | 0 / 93 |
| **original shell-aware smooth3 px135** | 1.4816 / 1.35 | 74 % / 99 % | 100 % | 3 / 117; 670 px | 0 / 93 |
| refit shell-aware smooth3 | 1.5016 / 1.5652 | 29 % / 55 % | 83-100 % | 0 / 117; 69 px | 0 / 93 |

Real-shell marks: frames of 007 with >= 100 px on the SEG shells / px per frame on them in 09:22, where the fake
touches a real shell (the composite marks that whole real shell; px135 a strip along the contact).

- The dark lentils score 1.30-1.69 (median 1.46 in 007): every pixel threshold above ~1.40 loses them.
- A dark lentil ALONE on the plate (estimate: its scores added to the clean frames) opens the gate of the shell-aware
  original in about 41 % (007) / 72 % (09:22) of frames; the composite at 1.35 in 70 % / 96 %, but 1.35 is below the
  clean turn's own highest frame score (1.41). With any other FO on the plate the gate is open anyway.
- Not solved by the shell-aware step: marks on kernels (007 frames 43-48, 09:22 frame 55); the suppression covers the
  SEG shells only.
