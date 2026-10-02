# walnut_final_robust: the four final pipelines with a robust FO mask (2 Oct 2026, for testing)

Every pipeline here is a copy of the one with the same name in `walnut_final/`. After the gate, the FO mask passes
two nodes before it is shown:
1. **FOBlobFilter** (MaskBlobFilter): a mark stays only if
   - it has at least N pixels (N = 250 for the original, 100 for the refit). A real FO's mark is the object plus its
     halo, usually thousands of pixels; specks on the belt are a few dozen;
   - and at least 16 of its pixels are **not empty belt**: their spectrum differs from the frame's own belt by more
     than 6 degrees. The belt's spectrum is the median spectrum of the frame (every 8th pixel), because the belt
     covers most of the frame (objects cover 11-28 % in every recording so far).
     - Class-agnostic: any object counts, known or unknown.
     - Brightness-invariant: shadows stay belt.
     - Keeps working with a wrong white reference: every pixel is compared with the same frame's belt.
   Marks are grouped on 4 x 4 pixel cells; nothing is computed on frames without a mark.
2. **FOFlickerFilter** (MaskPersistence): a pixel shows only if the frame before showed FO within 40 px (as in
   `walnut_final_flicker_filter/`).

**What changes:**
- Changed: the FO mask, both `FOMask.decisions` and the FO colour (255) in `Composite.mask`.
- Unchanged:
  - the alarm (`gate.passed`, the frame score);
  - the FO heatmap (`gate.scores`);
  - the shells;
  - the thresholds.

| file | copy of | size test | thresholds (gate / mask) |
|---|---|---|---|
| `walnut_final_original_robust_cuvisnext_cube.yaml` + `.pt` | `walnut_final_original` | 250 px | 1.35 / 1.35 |
| `walnut_final_refit_1oct_robust_cuvisnext_cube.yaml` + `.pt` | `walnut_final_refit_1oct` | 100 px | 1.1512 / 1.2732 |
| `walnut_final_original_shellaware_robust_cuvisnext_cube.yaml` + `.pt` | `..._original_shellaware` | 250 px | 1.35 / 1.35 |
| `walnut_final_refit_1oct_shellaware_robust_cuvisnext_cube.yaml` + `.pt` | `..._refit_1oct_shellaware` | 100 px | 1.1512 / 1.2732 |

**Restart CuvisNEXT once** before using them: two new nodes of the patchcore plugin (MaskBlobFilter,
MaskPersistence).

**Speed** (RTX 4070 laptop, median of 50 runs per frame): +1.5 to +2.7 ms on a frame with FO marks, ~0 on a clean
frame (FOBlobFilter ~1.0 ms, FOFlickerFilter ~0.4 ms). Thor: not measured yet.

**Checked end to end:** the four pipelines on 32 consecutive real frames, every node's output equal to an independent
reimplementation (`robust_e2e.py`, all checks pass).

**Measured offline** on your labelled 1-Oct afternoon frames (200 FO frames; each pipeline calibrated as at the
stand):

| | false blobs per FO frame | FO frames with a false blob | FO objects lost |
|---|---|---|---|
| original | 0.475 | 34 % | |
| original robust | **0.050** | **5 %** | 0 (the flicker filter hides an object's first frame: fakes 153 -> 148, stems 57 -> 50 of 182 / 73) |
| refit | 0.170 | 15 % | |
| refit robust | **0.030** | **3 %** | 0 (first frames only: fakes 106 -> 102, stems 60 -> 54) |

The refit recipe with the afternoon held out (the honest version): 0.025 -> 0.000, no FO lost.

**Your 14:27 recording** (FOs in every frame), marks on the empty belt per frame, measured with the first version of
this filter (separate nodes, the same rules on full-resolution blobs):

| | right white reference | wrong white reference (19.75 ms) |
|---|---|---|
| original | 2.55 | 10.62 |
| original robust | 0.12 | 0.07 |
| refit | 0.83 | 3.23 |
| refit robust | 0.02 | 0.03 |

"Empty belt" in this table is the same spectral test the filter uses, so before the flicker filter these counts
are 0 by construction. The small numbers left come from the flicker filter splitting object marks into fragments,
which the test then counts as belt. The evidence that no FO is lost comes from your labels instead.

**Limits:**
- An FO with the belt's own spectrum (a piece of the belt, blue plastic like it) is not an object to the filter, so
  it cannot be shown. No such FO was in the recordings.
- False marks on kernels and shells are objects too, so they stay. The shell-aware versions lower the shell part.
- A mark that touches an object stays whole, including the part of its halo that lies on the belt.
- The alarm still counts belt specks. Only the shown mask is filtered.
- A small FO whose mark stays below the size test is not shown. This is why the refit, with its stricter mask
  margin and smaller blobs, uses 100 px.
- The flicker filter delays every FO by one frame (1/8 s).
- If objects ever cover more than about half of a frame, the median is no longer the belt and the object test
  weakens.

**Next candidate (testing):** cutting the mark to the objects pixel by pixel (objects found per frame with an
automatic threshold and hole filling, 4 px margin) removes ~88 % of the marked area off FOs and shells on your labels,
with no FO lost and the same false-blob rate. Risk seen on the 14:27 recording with the wrong white reference: thin
dark FOs (the stem) lose most of their mark. Results in `walnut_fo_v3/runs/objpx_*` on asai1.

**Not chosen**, measured the same way, details in `THOR_DEPLOY_NOTES.md` §35:
- Gaussian smoothing of the FO map: results depend on the seed.
- 3 x 3 feature pooling and a per-channel white balance: both lose many FOs.
- Banks with more clean data: more false alarms held out.

**Calibration** is the same as for the source: the gate is before the filters. Copy the source's thresholds, or run
`calibrate_from_log.py --yaml <this file>` from a clean turn of THIS pipeline. For the refit versions, add
`--mask-margin 1.15`.

Built with `make_robust.py --src walnut_final/<name>_cuvisnext_cube.yaml --dst
walnut_final_robust/<name>_robust_cuvisnext_cube.yaml --min-area 250` (100 for the refit versions).
