# walnut_refit_1oct: the FO refit tier (2 Oct 2026)

A second tier next to the deployed pipelines, for comparison at the stand. The original pipelines in `walnut_fo/`
and `walnut_combo/` are unchanged (thresholds 1.35 / 1.35).

**What it is.** Copies of three deployed fp16trt pipelines with only the FO banks replaced (`make_refit_tier.py` in
the stack root; every other weight entry byte-identical to the source):

| file (relative to this folder) | source | role |
|---|---|---|
| `walnut_combo/composite/walnut_combo_gated_fp16trt_trt16_composite_cuvisnext_cube.yaml` | `walnut_combo/composite/` | FO + shells in one view (the fair pipeline) |
| `walnut_combo/walnut_combo_gated_fp16trt_trt16_cuvisnext_cube.yaml` | `walnut_combo/` | FO + shells, separate outputs |
| `walnut_fo/walnut_fo_multiscale_gated_fp16trt_cuvisnext_cube.yaml` | `walnut_fo/` | FO only (heatmap) |

The `.pt` beside each yaml holds the weights (not in git, as for every deploy pipeline). In CuvisNEXT, set Pipeline
to the yaml and Weights to its `.pt`; the metadata name starts with `refit_1oct_`.

**The banks.** Experiment `D:\experiments\2026-09-23_walnut-fo-v3`, `refit_v4.py`, run `v4_ab_s0` on asai1 (banks.pt
md5 `5e92b0e10d6463c663c6bfe6b8f51e88`):
- the 200 split-v3 TRAIN frames plus the 567 clean frames of the 1-Oct stand session (production lights, 5 and 8 fps);
- seed 0, coreset 7200 from 80 000 (the deployed sizes, so the same speed);
- normalisers on the 34 VAL frames; SDK 3.6 frame cache.

**Thresholds.** gate 1.1512, mask 1.2732, set from the 1-Oct afternoon clean files (float32 on asai1):
- gate = 1.10 x the highest clean frame score;
- mask = **1.15** x the highest clean pixel. The mask margin is 1.15 instead of the deployed 1.0 because the refit
  needs it for the same false-blob rate (`TASKS.md`, 2 Oct 12:00).

**Calibrate every morning, separately from the original tier.** The values above are for 1 Oct, and the refit's
level depends more on the session.
1. Load this tier's composite.
2. Run it on clean product for about one turn.
3. Then:
   ```
   python3 calibrate_from_log.py --precision fp16trt --last 60 --mask-margin 1.15 \
     --yaml cuvis-ai/cuvis_ai/configs/pipeline/walnut_refit_1oct/walnut_combo/composite/walnut_combo_gated_fp16trt_trt16_composite_cuvisnext_cube.yaml \
     --yaml cuvis-ai/cuvis_ai/configs/pipeline/walnut_refit_1oct/walnut_combo/walnut_combo_gated_fp16trt_trt16_cuvisnext_cube.yaml \
     --yaml cuvis-ai/cuvis_ai/configs/pipeline/walnut_refit_1oct/walnut_fo/walnut_fo_multiscale_gated_fp16trt_cuvisnext_cube.yaml --write
   ```
   The original tier's calibration (`calibrate_from_log.py --precision fp16trt ...` without `--yaml`) never writes
   into this folder. A log from the original pipeline must not be used for this tier, or the other way round.

**Evidence** (`TASKS.md`, 2 Oct):
- Held-out sessions: aluminium / rubber / stem caught 53 / 39 / 0 % -> 100 / 100 / 97 % with no extra clean false alarm (morning).
- On your labels (afternoon), at the same false-blob rate, the weak objects are shown more often.
- Caveats: one session, one object per weak material, kernels not labelled.
