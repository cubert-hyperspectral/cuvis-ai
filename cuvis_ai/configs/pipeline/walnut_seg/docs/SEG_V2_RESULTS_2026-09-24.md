# SEG v2 results (trained 23 Sep, evaluated overnight) — 2026-09-24

Full tables: `SEG_V2_compare_2026-09-24.md` (= asai1 `rfdetr_train/compare_v2.md`). Metric = shell IoU / FP px per
frame / fake→shell (% of fake pixels called shell), threshold 0.5, `eval_full_v2.py`, identical test sets for all.

## What was trained (asai1, same recipe as the deployed `*_full` models, 60 epochs, single seed)
| run | input | data | augmentation | time |
|---|---|---|---|---|
| **B** `out_perband_v2aug2` | RGB 640/550/470 | universe v2 (old FULL + 22-Sep c) | **aug v2** | 3.4 h |
| **C** `out_cir_v2aug2` | CIR 850/660/550 | universe v2 | **aug v2** | 2.8 h |
| **A** `out_perband_v2data` | RGB | universe v2 | old aug (data-only control) | 2.4 h |
Deployed baselines: `rgb_full` (= out_perband_full) and `cir_full` (= out_cir_full), md5-verified.

## Headline (ALL = mean over the set's frames)
| test set | deployed rgb | deployed cir | B RGB+aug2 | **C CIR+aug2** | A RGB+old aug |
|---|---|---|---|---|---|
| 22-Sep NEW test (43) — ALL IoU | 0.904 | 0.913 | 0.931 | **0.963** | 0.917 |
| 22-Sep fakes — IoU / fake→shell | 0.732 / 43 % | 0.776 / 19 % | 0.857 / 0 % | **0.935 / 1 %** | 0.830 / 7 % |
| 22-Sep hands+puck — IoU / FP px | 0.914 / 1.5k | 0.881 / 1.0k | 0.888 / 2.8k | **0.954 / 0.9k** | 0.848 / 10.3k |
| 22-Sep clean / FO scenes — IoU | 0.95 / 0.98 | 0.98 / 0.98 | 0.98 / 0.98 | 0.98 / 0.98 | 0.98 / 0.98 |
| 15-Sep fake-on-real (14) — IoU / fake→shell | 0.774 / 28 % | 0.790 / 7 % | 0.886 / 2 % | **0.963 / 1 %** | 0.890 / 7 % |
| 1-Sep robustness (106) — ALL IoU | 0.944 | 0.900 | 0.945 | **0.958** | 0.951 |
| 1-Sep live hand scene — IoU / FP px | 0.917 / 974 | 0.788 / 75k | **0.943 / 809** | 0.941 / 11.2k | 0.936 / 15.6k |
| 1-Sep normal — IoU | 0.979 | 0.940 | 0.958 | **0.979** | 0.960 |

## Findings
1. **C (CIR + new data + aug v2) is the best single model** on every test set: 22-Sep ALL 0.963 (+0.05 over the
   best deployed), fakes 0.935 with 1 % fake→shell (deployed 19–43 %), hands 0.954, 15-Sep fake-on-real 0.963
   (+0.17, an independent session), 1-Sep 0.958. [Certain for these sets; single seed]
2. **Its remaining weak spot: false positives in the old 1-Sep live hand scene** (11.2k px/frame, precision 0.959) —
   far better than deployed CIR (75k) but above the RGB models (B 809 px). [Certain]
3. **Aug v2 works (B vs A, same data):** hand/object FP 3.7× lower on 22-Sep (2.8k vs 10.3k) and 19× lower on the
   1-Sep live scene (809 vs 15.6k); fake→shell 0 % vs 7 %. [Likely — single seed each]
4. **New data drives the fake fix (A vs deployed rgb, same aug):** 22-Sep fakes 0.830 vs 0.732, fake→shell 7 % vs
   43 %; 15-Sep 0.890 vs 0.774 — but data alone (without aug v2) raised hand FP. [Likely]
5. **NIR matters for the new fakes (C vs B, same data + aug):** fakes 0.935 vs 0.857 (22-Sep), 0.963 vs 0.886
   (15-Sep). [Likely]
6. **Validation mAP does not predict robustness:** A had the highest val mAP (0.897) and the worst hand FP. [Certain]

## Caveats
- One seed per config (earlier seed spread was ±0.02–0.03, larger on FO); the C vs deployed gaps (0.05–0.17) are
  well above that, the B vs A gaps partly are not.
- 22-Sep test (d) is from the same session as the new training data (c) — different objects, same day/light; the
  15-Sep and 1-Sep sets are the independent checks (C wins those too).
- No real camera-height data → the zoom augmentation's effect on height changes is NOT measured yet.
- Not yet run: the RGB+CIR ensemble of B+C, the live test in CuvisNEXT, latency.

**Follow-up done 24 Sep (steps 1–3): `SEG_V2_HEIGHT_ENSEMBLE_2026-09-24.md` — B+C ensemble picked for live, packaged.**

## Next steps (proposed)
1. Evaluate the **B+C mean ensemble** (as deployed today with rgb_full+cir_full) — may combine C's fake rejection
   with B's low hand FP. Inference only (~15 min on asai1).
2. **Synthetic camera-height test**: rescale the test cubes 0.5×/0.75×/1.5×/2× (+ blur) and compare C/B vs deployed.
3. Package the winner(s) into `walnut_seg` (with ShellMask), parity check, **CuvisNEXT live test** (≥ 3 lights).
4. Optional: second seed of C to confirm before the fair.
