# SEG v2 — B+C ensemble, camera-height / defocus test, C hand-FP review, packaging (2026-09-24)

Follow-up to `SEG_V2_RESULTS_2026-09-24.md` (user go 24 Sep: "yes go ahead with 1 and 2, then package").
Full tables: `SEG_V2_HEIGHT_ENSEMBLE_tables_2026-09-24.md`. C hand-FP images: `SEG_V2_C_hand_FP_2026-09-24\`.

## Decision
**Deploy the B+C ensemble (`walnut_seg_ens_rgb_cir_mean_v2`) as the primary live pipeline.** C alone is the best
single model on the as-captured test sets, but it sometimes calls an entire hand "shell". That happens in 1 of 39
hand frames as captured and in 11 of 39 at 2× (lower camera). The ensemble never does it (0 frames with a >20k px
false-positive region in any of the 8 conditions), matches C on fakes, and is the most robust model under every
height / blur change. [Certain for these test sets; single training seed]

## 1. Ensemble at the captured height (1.0×)
Harness `eval_v2_height.py` (asai1) — **reproduces `eval_full_v2.py` exactly** for all 5 single models (every
number in `compare_v2.md` matched), so singles and ensembles are on one footing. Ensembles = members at RF-DETR
threshold 0.05 + ScoreFusion(mean), shell = fused score ≥ 0.5 (= the deployed `ens_rgb_cir_mean` recipe).

| test set (frames) | deployed RGB | deployed CIR | deployed ens | B | C | **B+C ens** |
|---|---|---|---|---|---|---|
| 1-Sep ALL (106) — IoU / FP px | 0.944 / 842 | 0.900 / 19.9k | 0.956 / 661 | 0.945 / 808 | 0.958 / 3.4k | **0.968 / 511** |
| 1-Sep live hands (27) — IoU / FP px | 0.917 / 974 | 0.788 / 75.1k | 0.918 / 652 | 0.943 / 809 | 0.941 / 11.2k | **0.961 / 718** |
| 15-Sep fake-on-real (14) — IoU / fake→shell | 0.774 / 28 % | 0.790 / 7 % | 0.810 / 18 % | 0.886 / 2 % | **0.963 / 1 %** | 0.957 / 0 % |
| 22-Sep ALL (43) — IoU / FP px | 0.904 / 4.9k | 0.913 / 2.9k | 0.910 / 3.9k | 0.931 / 1.9k | **0.963 / 1.5k** | 0.957 / 1.2k |
| 22-Sep fakes — IoU / fake→shell | 0.732 / 43 % | 0.776 / 19 % | 0.749 / 37 % | 0.857 / 0 % | **0.935 / 1 %** | 0.926 / 0 % |
| 22-Sep hands + puck — IoU / FP px | 0.914 / 1.5k | 0.881 / 1.0k | 0.889 / 666 | 0.888 / 2.8k | **0.954 / 891** | 0.937 / 705 |
| all 163 — IoU / precision | 0.916 / 0.959 | 0.893 / 0.940 | 0.928 / 0.967 | 0.935 / 0.981 | 0.960 / 0.979 | **0.964 / 0.989** |

- B+C ≈ C on the 22-Sep / 15-Sep sets (−0.006 IoU), better on 1-Sep (+0.010), lowest FP and highest precision
  overall, fake→shell 0 % everywhere.
- The deployed ensemble is worse than C alone on the new fakes (37 % fake→shell): both of its members leak.

## 2. Camera height + defocus (synthetic)
No real height data exists, so each test cube was transformed before the pipeline (GT follows the same geometry):
**0.5× / 0.75×** = camera higher (cube shrunk, antialiased, reflect-padded back to full size — NOT the median
canvas used in training, so the test does not reward the training trick); **1.5× / 2×** = camera lower (4 corner
windows upsampled to full size, counts summed per frame); **blur σ 1/2/3 px** = defocus.

Shell IoU, all 163 test frames:

| model | 0.5× | 0.75× | 1.0× | 1.5× | 2× | blur σ1 | σ2 | σ3 | worst |
|---|---|---|---|---|---|---|---|---|---|
| deployed RGB | 0.684 | 0.872 | 0.916 | 0.875 | 0.789 | 0.913 | 0.883 | 0.773 | 0.684 |
| deployed CIR | 0.762 | 0.911 | 0.893 | 0.877 | 0.832 | 0.884 | 0.879 | 0.872 | 0.762 |
| deployed ens | 0.762 | 0.917 | 0.928 | 0.931 | 0.887 | 0.930 | 0.917 | 0.891 | 0.762 |
| A RGB, v2 data, old aug | 0.792 | 0.905 | 0.935 | 0.916 | 0.877 | 0.934 | 0.900 | 0.884 | 0.792 |
| B RGB, v2 data, aug v2 | 0.846 | 0.907 | 0.935 | 0.932 | 0.910 | 0.918 | 0.892 | 0.876 | 0.846 |
| C CIR, v2 data, aug v2 | 0.882 | 0.950 | 0.960 | 0.954 | 0.910 | 0.944 | 0.953 | 0.947 | 0.882 |
| **B+C ens** | **0.903** | 0.949 | **0.964** | **0.971** | **0.958** | **0.959** | **0.960** | **0.948** | **0.903** |

Frames with a large false-positive region (> 20k px, "a hand lit up") among the 39 hand-scene frames:

| model | 0.5× | 0.75× | 1.0× | 1.5× | 2× | σ1 | σ2 | σ3 |
|---|---|---|---|---|---|---|---|---|
| deployed CIR | 3 | 1 | 5 | 10 | 14 | 5 | 6 | 5 |
| A | 1 | 1 | 3 | 3 | 4 | 2 | 2 | 2 |
| B | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| **C** | 2 | 0 | 1 | 4 | **11** | 3 | 1 | 2 |
| **B+C ens** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

Fake→shell stays ≤ 1 % for B+C in every condition (C ≤ 2 %, deployed RGB up to 23 % at 2×).

Findings:
1. **The deployed RGB model breaks when the camera moves**: 0.684 at 0.5×, 0.789 at 2×, 0.773 at blur σ3. The
   v2 models hold much better; B+C's worst case is 0.903. [Certain, synthetic test]
2. **Aug v2 (zoom) helps the height case** — B vs A (same data): 0.5× 0.846 vs 0.792, 2× 0.910 vs 0.877.
   Blur is the exception on RGB: B is not more blur-robust than A (σ1 0.918 vs 0.934). [Likely, one seed each]
3. **C's weakness is hands, and it grows when hands get bigger** (lower camera) or blurred — see the table above.
   The RGB member vetoes those regions in the ensemble. [Certain for these frames]

Caveats: synthetic geometry (reflect padding mirrors content at the borders; zoom-in = interpolation, a real closer
camera would add detail, perspective and a focus change); single seed; still no real height data — capture a few
frames at the fair heights to confirm.

## 3. C's false positives on the 1-Sep live hand scene (user asked to see them)
Images: `SEG_V2_C_hand_FP_2026-09-24\live_1sep\overview.png` (all 27 frames, FP bars C vs B vs B+C + C overlays),
`sheet_01..03.png` (detail rows: RGB + GT outlines | C on CIR | zoom of C's largest FP blob | B | B+C),
`fp_table.csv`. FP is split into hand/object, mat, fake, and "outline" (≤ 3 px outside a GT shell = boundary
disagreement, not a false detection).
- **The 11.2k mean is one frame**: `real_world_live_000_f0010` = 278,245 FP px — C paints the whole hand (a large
  hand entering from the left, next to the black puck) as shell. B: 1,210 px; B+C: 1,962 px (a thin sliver at one
  finger).
- Next worst: `f0003` 3,008 px — a 2.4k px blob on a knuckle between two fingers (B+C keeps a smaller blob: 2,696
  total incl. outline).
- **All other 25 frames: 203–1,574 px, almost all outline slop** (same level as B: 135–1,461). Without f0010 C's mean
  is 951 px/frame vs B 809.
- 22-Sep hand + puck scene (`hand_other_22sep\`): C is clean (≤ 2.3k px, mostly outline; B has up to 14k there).
- In the cuvis.next library version (rfdetr 1.10.1) the same failure hits `f0025` instead of `f0010` (460k px, a
  large hand filling the right of the frame; B 241, B+C 117) — `live_1sep_cuvisnext_rfdetr1101\sheet_01.png`.
- **Verdict:** as a single model C is not acceptable for a live demo where hands come in — a whole hand turns into
  "shell" in some frames, more often with a lower camera. Inside the B+C ensemble it is acceptable.

## 4. Packaging (walnut_fo_stack, cuvis.next cube mode)
In `walnut_fo_stack\cuvis-ai\cuvis_ai\configs\pipeline\walnut_seg\` (built by `build_seg_v2.py`):
| pipeline | what | weights |
|---|---|---|
| **`walnut_seg_ens_rgb_cir_mean_v2_cuvisnext_cube`** | **B+C ensemble — primary** | `weights/rgb_v2_ema.pth` + `weights/cir_v2_ema.pth` |
| `walnut_seg_cir_v2_cuvisnext_cube` | C single (faster; hand risk above) | `weights/cir_v2_ema.pth` |
| `walnut_seg_rgb_v2_cuvisnext_cube` | B single (RGB fallback) | `weights/rgb_v2_ema.pth` |
Same node recipe as the deployed `*_full` pipelines + `ShellMask` (boolean mask, scores ≥ 0.5); each has its
`.pt` and `cuvis_picker_*.pt`. Weights = asai1 `out_{cir,perband}_v2aug2/checkpoint_best_ema.pth` (md5
`acfb07bb…` / `0b5b19ac…`, verified after copy). In cuvis.next set Pipeline = the `.yaml` AND Weights = the `.pt`.

### Parity — PASSED (like for like)
- Input: frame 0 of `real_world_live_000.cu3s` exported on asai1 exactly as the cu3s reader delivers it (float32
  reflectance, 0–22036; wavelengths = 430 + 8k nm) → fed to the packaged stack pipelines on the laptop and to the
  eval pipelines on asai1.
- **Mask agreement IoU 0.9998 (cir_v2) / 0.9999 (rgb_v2) / 0.9999 (ens_v2)** vs asai1 with the same rfdetr
  (1.10.1); pixel counts 71743 vs 71750 / 71234 vs 71227 / 70631 vs 70632. ShellMask == (scores ≥ 0.5) exactly.
  The 1–26 px residual = torch 2.11/cu128 (laptop) vs 2.13/cu130 (asai1) float noise; exact equality across machines
  is not achievable. [Certain]
- The old `.npy` parity cube (`deploy_wise08/real_world_live_000_f0000.npy`) is **float16, globally scaled to
  [0, 1]** — not what cuvis.next feeds (raw float32 reflectance). Its counts (72424 for rgb_full …) are valid drift
  references for this machine only, not an end-to-end check of the eval path. New drift references: cir_v2 71684,
  rgb_v2 71205, ens_v2 70671.

### Library-version skew found: training/eval env ≠ cuvis.next env
- asai1 training + all evaluations: **rfdetr 1.8.3**, torch 2.13. The stack / cuvis.next venv: **rfdetr 1.10.1**,
  torch 2.11. Plugin code (`rfdetr_segmenter.py`, `functional.py`) is byte-identical; the selector code is identical.
- Re-ran the 1.0× eval on asai1 with rfdetr 1.10.1 (private `--target` install, shared env untouched):

| model | all 163 IoU 1.8.3 → 1.10.1 | 22-Sep ALL | 1-Sep live FP px | fake→shell |
|---|---|---|---|---|
| deployed ens | 0.928 → 0.936 | 0.910 → 0.916 | 652 → 575 | 5 % → 5 % |
| B | 0.935 → 0.940 | 0.931 → 0.942 | 809 → 767 | 0 % → 0 % |
| C | 0.960 → 0.960 | 0.963 → 0.965 | **11,221 → 17,981** | 0 % → 0 % |
| **B+C ens** | **0.964 → 0.966** | 0.957 → 0.963 | 718 → 645 | 0 % → 0 % |

- Per frame the shell masks move by a median 0.06 % — **except C's whole-hand failure, which jumps frames**: with
  1.8.3 it hits `f0010` (278k px), with 1.10.1 `f0010` is fine (930) but `f0025` lights up (460k px; image in
  `SEG_V2_C_hand_FP_2026-09-24\live_1sep_cuvisnext_rfdetr1101\`). Both are frames with a large hand close to the
  camera. So C's hand behaviour sits on a knife edge — tiny numeric changes decide which frame fails. B+C has no
  frame whose FP changes by more than 5k px. [Certain for these frames]
- Conclusion: the version skew does not change any conclusion (B+C best, slightly better under 1.10.1), but future
  evals should run with the deploy rfdetr version (or pin the deploy env to 1.8.3) so eval numbers = live numbers.

## 5. Repo
`cuvis-ai-rfdetr` PR #12 (`feat/walnut-deploy-nodes`) updated 24 Sep: SamShellGate + the 4 fair transforms + a
ruff-format fix (the PR's lint check had been failing), a CHANGELOG entry for SamShellGate, the manifest test
now lists SamShellGate, and `transforms.py` omitted from coverage (imports cuvis-ai-augment, which CI cannot
install; its 52 tests pass where augment is installed). **CI 7/7 green** on 77edc27. Still draft, not merged.

## Next
1. **Live test in cuvis.next** with `walnut_seg_ens_rgb_cir_mean_v2` (A/B vs the deployed `ens_rgb_cir_mean`),
   ≥ 3 light settings, hands in the scene, and — if possible — two camera heights. Latency: the ensemble is 2
   RF-DETR passes (~2× a single).
2. Capture a few frames at the fair's real camera heights to replace the synthetic height test.
3. Optional: second seed of B and C.
4. Z:/Thor sync of the v2 pipelines — pending, only after the live test, without touching Nima's scoped pins /
   `.release-0171-spec`.
