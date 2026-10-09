# walnut_best: the pipelines to load (5 Oct 2026)

Six pipelines: both FO models plain, shell-aware and with the pixel cut. Everything else in `configs/pipeline/` stays
where it was, for comparison only.

| file | model | mask | use |
|---|---|---|---|
| `walnut_best_refit_cuvisnext_cube.yaml` | refit (1 Oct banks) | v2 | **first choice at the production stand** |
| `walnut_best_original_cuvisnext_cube.yaml` | original | v2 | other setups, older lighting |
| `walnut_best_refit_cut_cuvisnext_cube.yaml` | refit | v2 + pixel cut | the same, mask tighter around the objects; about 20 ms slower on Thor |
| `walnut_best_original_cut_cuvisnext_cube.yaml` | original | v2 + pixel cut | the same, mask tighter around the objects; about 20 ms slower on Thor |
| `walnut_best_refit_shellaware_cuvisnext_cube.yaml` | refit | v2, FO map x 0.8 on the SEG shells | when real shells next to an FO light up (5 Oct live test) |
| `walnut_best_original_shellaware_cuvisnext_cube.yaml` | original | v2, FO map x 0.8 on the SEG shells | the same for the original model |

They are copies of `walnut_final_robust_v2/` (`make_best.py`; only the name and description differ, checked on real
frames). The weights (`.pt`) are hard links; the SEG weights and TensorRT engines are read from their usual folders.
Restart CuvisNEXT once after the plugin update of 4 Oct (patchcore 87d823b) before loading the `_cut` files.

These pipelines run TensorRT nodes and list `steervit_trt` and `rfdetr_seg_trt` next to `steervit` and `rfdetr_seg`,
so the composer installs TensorRT into their environment; that needs the stack on cuvis-ai-core 0.18.1 or later
(older schemas reject the manifests' `extras`). Details: `walnut_seg/README_SEG.md`, "The FO pipelines need the same".

**5 Oct live test on Thor (FOs, a fake shell touching a real shell, stem, dark dots):** the plain v2 marks the fake,
the stem and the dots, and also the real shell touching the fake (the FO score spills across touching objects) and
some kernel edges. The shell-aware v2 removes the marks on the real shells (SEG outlines them; the fake has no
outline) and keeps the fake, the stem and the dots. Kernels are not a SEG class, so they are not helped by it; the
clean-turn calibration is their fix. The cut removed the stem's mark with the original model again, so it stays an
option for the laptop only. An FO lying on a real shell needs 1.25 x the thresholds in the shell-aware versions:
calibrate them from a clean turn of their own (their gate sees the dampened map).

## What runs, step by step

1. **Detector and alarm (unchanged):** the FO model makes the anomaly heatmap. A frame alarms when its score is above
   `threshold`. On an alarm frame, the mask is every pixel above `mask_threshold`. The SEG shell mask and the
   Composite are as before.
2. **Blob filter (FOBlobFilter, new in v2):** each mark stays only if it has at least 100 pixels and at least 16 of
   them lie on something that is not empty belt (spectrum more than 6 deg from the frame's own belt). Specks and marks
   on the empty belt go, for example the edge marks of a wrong white reference.
3. **No flicker filter:** the robust version of 2 Oct hid every mark that the frame before did not have. That cost
   real FOs (an FO's first frame, the moving turntable), so v2 drops it.
4. **Refit mask threshold 1.10 x the highest clean pixel (was 1.15):** more of each FO is marked.
5. **Pixel cut (`_cut` files only):**
   - the objects of the frame are found (spectral angle above the frame's Otsu level, holes filled, plus 4 px);
   - each mark is cut to those objects, so its halo over the empty belt goes;
   - leftover pieces under 100 px, or far below the mark's peak score (halo on a neighbour), go;
   - two safety rules: a mark the cut would remove completely comes back whole, and every pixel above 1.3 x the mask
     threshold always stays. Without them the cut removed loose stems on the 2 Oct recording: a dark stem barely
     differs from the belt in spectral angle, but the model marks it clearly.

## How much it helped

Measured offline with the stand calibration of each session. An FO counts as found with at least 50 marked pixels
on it (a quarter of it under 200 px). False marks: marks touching no FO, per frame.

| session | refit: final → v2 → v2 + cut | original: final → v2 → v2 + cut |
|---|---|---|
| your 1 Oct labels: FOs found (of 575) | 546 → **565** → 565 | 556 → 552 → 552 |
| labels: false marks / frame | 0.13 → 0.15 → 0.16 | 0.37 → **0.18** → 0.19 |
| labels: marked px off FOs and shells / frame | 4,701 → 6,127 → **1,170** | 5,561 → 5,539 → **880** |
| 30 Sep moving turntable: FOs found (of 318) | 170 → **240** → 240 | 309 → 303 → 303 |
| 1 Oct morning stem file: frames marked (of 17) | 8 → 8 → 8 | 0 → 0 → 0 |
| 2 Oct production, marks only on the belt / frame (wrong / right ref) | 3.23 / 0.83 → **0.09 / 0.05** → 0.10 / 0.07 | 10.62 / 2.55 → **0.04 / 0.05** → 0.05 / 0.14 |
| older sessions 18 Aug - 22 Sep (the original test data): FOs found (of 536) | 191 → **232** → 232 | 335 → 330 → 329 |
| older sessions: false marks / frame | 0.48 → 0.47 → 0.52 | 0.97 → **0.78** → 0.87 |

- **Refit, older sessions:** +41 FOs (fake shells 95 → 119 of 333, rubber 12 → 19 of 44, stones 54 → 60 of 74).
- **Original, older sessions:** 5 FOs fewer (2 stems, 2 fake shells, 1 alu) for 20 % fewer false marks.
- **The cut** removes 81-84 % of the marked area off FOs and shells on your labels. On the older sessions it does the
  same where there are no hands: 18 Aug 4.0k → 1.1k px per frame (original), 22 Sep 10.2k → 1.7k. It costs a little
  in false-mark count, because a cut mark can leave two pieces.
- **Which model:** the refit is the better one at the production stand (more FOs, fewer false marks). It is weaker on
  the older setups (232 vs 330 of 536), so use the original there.

Speed against the final pipelines, per frame:
- **laptop** (RTX 4070): v2 about +1 ms; v2 + cut +2 to +5 ms;
- **Thor** (5 Oct, CuvisNEXT running its camera as in the live setup; the final pipelines take 120-140 ms): v2 +0 to
  +5 ms (within the noise); **v2 + cut +17 to +23 ms on frames with marks**, +0 to +3 ms on clean frames. The cut's
  extra steps run partly on Thor's CPU. **On Thor use the plain v2**; try the cut only if the frame rate holds up.

Checked on Thor (5 Oct): all four on 16 consecutive frames of Thor's own recordings (your 1 Oct labelled file, a
clean 1 Oct file, the 2 Oct production recording), every step against independent references: 64 / 64 pass; no mark
on the clean frames.

## Calibrate at the stand

Run the pipeline you will show on clean product only (16-20 frames), then, while it still runs:

```
python calibrate_from_log.py --precision fp16trt --mask-margin 1.10 --yaml cuvis-ai/cuvis_ai/configs/pipeline/walnut_best/walnut_best_refit_cuvisnext_cube.yaml --yaml cuvis-ai/cuvis_ai/configs/pipeline/walnut_best/walnut_best_refit_cut_cuvisnext_cube.yaml
```

Check the values, add `--write`, reload the pipeline. For the original model: the two `walnut_best_original*` yamls
and `--mask-margin 1.0`, from a log of an original pipeline. The two files of a model share the gate, so one log
calibrates both. Run it from the stack root; on the laptop use the stack's Python
(`cuvis-ai\.venv\Scripts\python.exe`), on Thor `python3`.

## Limits

- False marks on kernels and shells count as objects and stay; on the moving turntable about 1.4 per frame.
- Hands are objects too and keep their marks.
- An FO with the belt's own spectrum cannot be shown.
- The alarm still counts belt specks; only the shown mask is filtered.
- The 1.3 core of the cut was chosen on the 2 Oct recording: check the cut on the next production recording before
  you rely on it.
