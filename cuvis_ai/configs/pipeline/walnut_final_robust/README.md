# walnut_final_robust: the four final pipelines with a robust FO mask (2 Oct 2026, for testing)

Every pipeline here is a copy of the one with the same name in `walnut_final/`. After the gate, the FO mask passes
three filters before it is shown:
1. **FOSpeckFilter:** marks smaller than N pixels go. N is 250 for the original and 100 for the refit. A real FO's
   mark is the object plus its halo, usually thousands of pixels; specks on the belt are a few dozen.
2. **FOObjectGate:** a mark stays only if it touches at least 16 pixels that are **not empty belt**. FOObjects
   decides "not belt": a pixel whose spectrum differs from the frame's own belt by more than 6 degrees.
   - It is class-agnostic: any object counts, known or unknown.
   - It ignores brightness, so shadows stay belt.
   - It keeps working with a wrong white reference, because it compares every pixel with the same frame's belt.
3. **FOFlickerFilter:** a pixel shows only if the frame before showed FO within 40 px (as in
   `walnut_final_flicker_filter/`).

**What changes:**
- Changed: the FO mask, both `FOMask.decisions` and the FO colour (255) in `Composite.mask`.
- Unchanged:
  - the alarm (`gate.passed`, the frame score);
  - the FO heatmap (`gate.scores`);
  - the shells;
  - the thresholds.

| file | copy of | speck filter | thresholds (gate / mask) |
|---|---|---|---|
| `walnut_final_original_robust_cuvisnext_cube.yaml` + `.pt` | `walnut_final_original` | 250 px | 1.35 / 1.35 |
| `walnut_final_refit_1oct_robust_cuvisnext_cube.yaml` + `.pt` | `walnut_final_refit_1oct` | 100 px | 1.1512 / 1.2732 |
| `walnut_final_original_shellaware_robust_cuvisnext_cube.yaml` + `.pt` | `..._original_shellaware` | 250 px | 1.35 / 1.35 |
| `walnut_final_refit_1oct_shellaware_robust_cuvisnext_cube.yaml` + `.pt` | `..._refit_1oct_shellaware` | 100 px | 1.1512 / 1.2732 |

**Restart CuvisNEXT once** before using them: four new nodes of the patchcore plugin (MaskMinArea,
SpectralObjectMask, MaskBlobGate, MaskPersistence).

**Measured offline** on your labelled 1-Oct afternoon frames (200 FO frames; each pipeline calibrated as at the
stand):

| | false blobs per FO frame | FO frames with a false blob | FO objects lost (without / with the flicker filter) |
|---|---|---|---|
| original | 0.475 | 34 % | |
| original robust | **0.050** | **5 %** | 0 / first frames only (fakes 153 -> 148, stems 57 -> 50 of 182 / 73) |
| refit | 0.170 | 15 % | |
| refit robust | **0.030** | **3 %** | 0 / first frames only (fakes 106 -> 102, stems 60 -> 54) |

The refit recipe with the afternoon held out (two seeds, the honest version): 0.025 / 0.030 -> 0.000 / 0.005.

**Your 14:27 recording** (FOs in every frame), counting marks on the empty belt per frame:

| | right white reference | wrong white reference (19.75 ms) |
|---|---|---|
| original | 2.55 | 10.62 |
| original robust | 0.12 | 0.07 |
| refit | 0.83 | 3.23 |
| refit robust | 0.02 | 0.03 |

"Empty belt" in this table is the same spectral test the gate uses, so before the flicker filter these counts are
0 by construction. The small numbers left come from the flicker filter splitting object marks into fragments,
which the test then counts as belt. The evidence that the gate loses no FO comes from your labels instead: no FO
was lost on 200 frames.

**Limits:**
- An FO with the belt's own spectrum (a piece of the belt, blue plastic like it) is not an object to the gate, so it
  cannot be shown. No such FO was in the recordings.
- False marks on kernels and shells are objects too, so they stay. The shell-aware versions lower the shell part.
- The alarm still counts belt specks. Only the shown mask is filtered.
- A small FO whose mark stays below the speck size is not shown. This is why the refit, with its stricter mask
  margin and smaller blobs, uses 100 px.
- The flicker filter delays every FO by one frame (1/8 s).

**Not chosen**, measured the same way, details in `THOR_DEPLOY_NOTES.md` §35:
- Gaussian smoothing of the FO map: results depend on the seed.
- 3 x 3 feature pooling and a per-channel white balance: both lose many FOs.
- Banks with more clean data: more false alarms held out.

**Calibration** is the same as for the source: the gate is before the filters. Copy the source's thresholds, or run
`calibrate_from_log.py --yaml <this file>` from a clean turn of THIS pipeline. For the refit versions, add
`--mask-margin 1.15`.

Built with `make_robust.py --src walnut_final/<name>_cuvisnext_cube.yaml --dst
walnut_final_robust/<name>_robust_cuvisnext_cube.yaml --min-area 250` (100 for the refit versions).
