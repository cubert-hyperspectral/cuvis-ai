# walnut_final_robust_v2: the robust FO mask without the flicker filter (4 Oct 2026, for testing)

Every pipeline here is a copy of the one with the same name in `walnut_final/`, with one node after the gate:
**FOBlobFilter** (MaskBlobFilter). A mark stays only if it has at least 100 pixels and at least 16 of them are not
empty belt (spectrum more than 6 degrees from the frame's own belt). Compared with `walnut_final_robust/`:
- **no flicker filter.** On every recording it hid more real FOs than false marks it removed;
- **100 px size test for both models.** The original's 250 px lost small FOs on the older sessions;
- **refit mask threshold 1.2178 instead of 1.2732:** mask margin 1.10 instead of 1.15, so more of each FO is marked.

The alarm (`gate.passed`, the frame score), the FO heatmap and the shells are unchanged.

| file | copy of | thresholds (gate / mask) |
|---|---|---|
| `walnut_final_original_robust_v2_cuvisnext_cube.yaml` + `.pt` | `walnut_final_original` | 1.35 / 1.35 |
| `walnut_final_refit_1oct_robust_v2_cuvisnext_cube.yaml` + `.pt` | `walnut_final_refit_1oct` | 1.1512 / **1.2178** |
| `walnut_final_original_shellaware_robust_v2_cuvisnext_cube.yaml` + `.pt` | `..._original_shellaware` | 1.35 / 1.35 |
| `walnut_final_refit_1oct_shellaware_robust_v2_cuvisnext_cube.yaml` + `.pt` | `..._refit_1oct_shellaware` | 1.1512 / **1.2178** |

**`*_robust_v2_cut_*` (four more files, the same thresholds): v2 plus the pixel-level cut, rebuilt 4 Oct late.**
Each mark is cut to the objects of the frame, so the mask no longer spills over the belt, but a clear detection always
stays. After FOBlobFilter:
1. **FOObjects** (SpectralObjectMask) finds the objects: the frame's own Otsu level of the belt angle (at least 2 deg),
   holes filled, plus 4 px around each object;
2. **FOCut** keeps only the mark pixels on those objects;
3. **FOCutSize** drops leftover pieces under 100 px;
4. **FOCutPeak** (MaskPeakGate) drops pieces whose highest score is below 0.8 × the highest score of the mark they
   came from: halo left on neighbouring objects goes, the FO stays;
5. **FOCutLost** (MaskBlobGate, `invert`) brings back, whole, a mark the cut would remove entirely;
6. **FOCore** keeps the gate's new `core` output: the pixels above 1.3 × the mask threshold (gate `core_ratio: 1.3`;
   it follows the mask threshold when you recalibrate);
7. **FOCutTrim** joins 4, 5 and 6. FOMask and the Composite read it.

**Why 5 and 6 (found 4 Oct, late):** the first cut (steps 1-4 only) removed loose stems on the 2 Oct production
recording. With the right white reference it removed 74 marks that v2 shows (refit) and 117 (original), most of them
on about 6 stems over several frames.
- The model finds these stems clearly (1.5-1.7 × the mask threshold).
- The object finder misses them: a dark stem differs from the belt mostly in brightness, which the spectral angle
  ignores (8-11 deg on a few cells, below the frame's Otsu level of about 12 deg).
- With 5 and 6: refit 74 → 8, original 117 → 3 marks removed, none of them a stem (belt specks, one walnut, one weak
  frame of a dark dot).
- Contact sheets: `walnut_fo_v3/data/improve_2026-10-04/gone_2oct/` (`*_b/`: the first cut and the fixed one).

**Measured offline on every session** (each pipeline calibrated as at the stand; an FO counts as found when at least
50 marked pixels lie on it, a quarter of it for FOs under 200 px):

| | your labels 1 Oct: FOs found (stems) | labels: false marks / frame | 30 Sep turntable: FOs found | 1 Oct morning stem file: frames marked | 2 Oct: marks only on the belt / frame (wrong / right reference) | older sessions: FOs found / false marks per frame |
|---|---|---|---|---|---|---|
| refit, final | 546/575 (64/73) | 0.13 | 170/318 | 8/17 | 3.23 / 0.83 | 191/536 / 0.48 |
| refit, robust v1 | 515 (56) | 0.02 | 118 | 3/17 | 0.04 / 0.05 | 188 / 0.40 |
| **refit, robust v2** | **565 (69)** | **0.15** | **240** | **8/17** | **0.09 / 0.05** | **232 / 0.47** |
| original, final | 556 (58) | 0.37 | 309 | 0/17 | 10.62 / 2.55 | 335 / 0.97 |
| original, robust v1 | 526 (50) | 0.04 | 284 | 0/17 | 0.11 / 0.16 | 322 / 0.69 |
| **original, robust v2** | **552 (57)** | **0.18** | **303** | **0/17** | **0.04 / 0.05** | **330 / 0.78** |
| refit, robust v2 + cut | 565 (69) | 0.16 | 240 | 8/17 | 0.10 / 0.07 | 232 / 0.52 |
| original, robust v2 + cut | 552 (57) | 0.19 | 303 | 0/17 | 0.05 / 0.14 | 329 / 0.87 |

What the cut changes:
- **your labels:** about 81 % (refit, 6,127 → 1,170 px per frame) and 84 % (original, 5,539 → 880) less marked area
  off FOs and shells;
- **2 Oct:** marked belt pixels per frame, wrong / right reference: refit 50.6k / 54.5k → 23.6k / 19.3k, original
  45.0k / 50.3k → 15.4k / 13.9k;
- **FOs found:** the same as v2, except one FO fewer for the original on the older sessions (329 vs 330 of 536).

**Speed** (RTX 4070 laptop, median of 50 runs, against `walnut_final/`):
- v2: refit +0.3 to +1.8 ms per frame, original −0.8 to +1.1 ms;
- v2 + cut: refit +1.8 to +4.9 ms, original +2.0 to +4.4 ms (the first cut: +0.9 to +4.4 ms). The object finder runs
  on every frame (about 1.1 ms); bringing marks back costs about 0.5 ms, the core 0.1 ms. Runs vary by about 0.5 ms.

Thor: not measured yet.

**Checked end to end:** the eight pipelines on 32 consecutive real frames each. All 256 frame checks pass
(`robust_e2e.py`; the rebuilt cut pipelines: 128 / 128):
- the refit's new mask threshold: on passing frames, its gate mask is the base gate's input map above 1.2178;
- every step of the cut against independent NumPy / OpenCV / SciPy references: the angle within 0.01 deg, the Otsu
  objects, the AND, the size test, the peak rule, the marks brought back whole, the gate's core (the base gate's input
  map above 1.3 × the mask threshold on passing frames) and the final union.

**Restart CuvisNEXT once** after updating the patchcore plugin: the cut pipelines need MaskPeakGate, the new
SpectralObjectMask options, MaskBlobGate `invert` and the gate's `core_ratio` / `core` output (patchcore 87d823b).

**Calibration:**
- **original:** as the source.
- **refit:** `calibrate_from_log.py --yaml <this file> --mask-margin 1.10` from a clean turn of THIS pipeline. The 1.2178
  here is the 2 Oct calibration rescaled: 1.2732 × 1.10 / 1.15.
- **More FOs:** `--mask-margin 1.05` finds 573 of 575 FOs on your labels (all 73 stems) and 288 on the turntable. It
  costs about twice the false marks on objects (0.32 per frame).

**Limits:**
- Without the flicker filter, one-frame false marks on kernels and shells can show. On your labels the refit shows 0.15
  per frame (0.02 with the flicker filter).
- False marks on kernels and shells are objects, so the blob filter keeps them. On the moving turntable that is about
  1.4 per frame.
- Hands are objects too and keep their marks.
- An FO with the belt's own spectrum cannot be shown.
- The alarm still counts belt specks; only the shown mask is filtered.

**Not chosen**, measured the same way (`walnut_fo_v3/runs/improve_4oct*.md`):
- **Flicker filter on whole marks, or a radius of 80 / 120 px:** it still hides most of the same FOs.
- **For the stems** (refit, 2 Oct right reference, marks removed that v2 shows):
  - Otsu level capped at 6 deg: 11 removed, but false marks on your labels 0.17 → 1.46 per frame;
  - objects = angle > 6 deg without small islands: 28 (18 with the marks brought back);
  - cells darker than half the belt as objects: 13, stems still among them;
  - core at 1.2 × (with the 6-deg objects): 7, but twice the marked area off objects of the chosen cut.

Built with:
- `make_robust.py --src walnut_final/<name>_cuvisnext_cube.yaml --dst walnut_final_robust_v2/<name>_robust_v2_cuvisnext_cube.yaml --min-area 100 --no-flicker`
- the refit versions add `--scale-mask-threshold 0.9565217391304348` (1.10 / 1.15);
- the `_cut` versions add `--cut` (floor 2 deg, margin 4 px, 100 px, peak 0.8, the marks brought back whole, core
  1.3). The first cut (4 Oct 20:10, cuvis-ai 7fff4232) was `--cut --cut-may-delete`.
