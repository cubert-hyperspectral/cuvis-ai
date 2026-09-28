# walnut combined pipelines: FO anomaly + SEG shells on one cube (cuvis.next cube mode)

One pipeline runs the foreign-object (FO) anomaly detector and the shell segmentation (SEG) ensemble on the same cube.
The FO output and the SEG outputs are separately selectable in cuvis.next's Displayed Output. Built by `build_combo.py`
from the deployed standalone pipelines, and checked by `check_combo.py` and `bench_combo.py`.

## Pipelines

Pipeline names: `walnut_combo_<family>_<tier>_cuvisnext_cube`, set with `Weights` = the sibling `.pt`.

| family | FO branch (frame-gated) | SEG branch (no frame gate) |
|---|---|---|
| `or` | `walnut_fo_multiscale_effad_or_gated`: multi-scale gate OR EfficientAD gate | `walnut_seg_ens_rgb_cir_mean_v2` |
| `gated` | `walnut_fo_multiscale_gated`: the multi-scale gate alone | `walnut_seg_ens_rgb_cir_mean_v2` |

| tier | FO variant (thresholds) | SEG tier | needs TensorRT |
|---|---|---|---|
| `tf32_exact` | `_tf32` (gate 1.337, EfficientAD 1.9587) | `_exact`: fp32 PyTorch + GPU input, bit-identical to the SEG default | no |
| `tf32_fast` | `_tf32` | `_fast`: fp16 + JIT trace + GPU input | no |
| `tf32_trt32` | `_tf32` | `_trt_fp32`: TensorRT fp32 engine (TF32 allowed) | yes |
| `tf32_trt16` | `_tf32` | `_trt_fp16`: TensorRT fp16 engine | yes |
| `fp16_trt16` | `_fp16` (gate 1.336, EfficientAD 1.9546) | `_trt_fp16` | yes |

- **FO is frame-gated**: its map is shown only on frames where a gate passes, and is zeros otherwise.
  - OR: the multi-scale gated map when the multi-scale gate opens, else the EfficientAD gated map.
  - gated: the multi-scale map when its gate opens.
- **SEG is not gated**: the mask is the fused shell score >= 0.5 on every frame, as in the standalone ensemble. Its
  selectors calibrate on the first 20 frames of a session, so show a normal scene first.
- **No FO float32 tier.** `import rfdetr` (the SEG plugin) sets PyTorch's float32 matmul precision to "high" (TF32)
  for the whole process. Next to the SEG models the FO branch therefore runs TF32 anyway. The tiers use the FO
  `_tf32` / `_fp16` variants with their own calibrated thresholds, both validated on all 287 VAL + TEST frames.
- `tf32_exact` and `tf32_fast` are the fallbacks without TensorRT.

## Outputs (pick explicitly in cuvis.next, not "Automatic")

- `display.scores` (or) / `gate.scores` (gated) · heatmap: the FO anomaly map, only on anomalous frames.
- `ShellMask.decisions` · mask: the SEG shell mask.
- `ShellHeatmap.scores` · heatmap: the SEG shell probability map.
- The dropdown also lists `steervit_t1.scores` / `steervit_t2.scores` (SteerViT's zero-shot prompt maps, not the
  detector output) and the selectors' `rgb_image` ports (images), as the standalone pipelines do.

## Graph

- One `CU3SDataNode`, `cu3s_data` (FO's name), feeds both branches. SEG's `DataSource` is dropped.
- SEG's `Fuse` is `ShellFuse`: FO already has a `fuse`, and the names differ only by case. It is a stateless
  ScoreFusion(mean) with the same hparams.
- Every other node, hparam and connection is the standalone pipelines'. `build_combo.py` checks this node for node;
  the only hparams it drops are the class defaults `save_to_file` writes out.
- Plugins: `cuvis_ai_builtin, patchcore, steervit, efficientad` (or only), `rfdetr_seg`. Never also `rfdetr`: both
  manifests point at the same package.

## Weights and files

- One `.pt` per family (FO PatchCore banks, normalisers, EfficientAD; the SEG selectors' fresh state). Every tier of a
  family shares it: the other tiers' `.pt` are hardlinks, made after comparing the state tensor by tensor.
- The RF-DETR models load from the absolute `checkpoint_path` (`walnut_seg/weights/*_v2_ema.pth`). On another machine,
  run `walnut_seg/restamp_thor.py --root <stack root>`. The TensorRT engines are in `walnut_seg/weights/*.pth.trt/`
  (laptop RTX 4070 and Thor), shared with the SEG pipelines.

## Requirements

- The deploy checkouts: cuvis-ai-patchcore / -steervit / -efficientad v0.2.0, cuvis-ai-rfdetr v0.5.1.
- **The first load composes a new cuvis.next child env** (a new plugin set; it needs internet). Do it before the
  fair, not at the venue.
- **`_trt` tiers:** `tensorrt` 10.15.1.29 in that env (`tensorrt-cu12` on the laptop, `tensorrt-cu13` on Thor).
  A composed env does not have it: `walnut_seg/check_trt_env.py --include-fo --fix` installs it (with the user's go),
  and it has to be re-checked whenever cuvis.next composes a new env.

## Thresholds (per-session recalibration)

- **Known issue (28 Sep, THOR_DEPLOY_NOTES §0 and §19):** at the shipped thresholds the FO OR branch misses 13 of the
  47 18-Aug FO frames, and its EfficientAD gate fires on 25 of the 34 clean 18-Aug kernel-in-shell frames. 1, 15 and
  22 Sep are clean. Recalibrate per session on clean frames that cover the day's arrangements.
- The combined yamls' FO gate thresholds are those of the FO variant of their precision (`tf32` 1.337 / 1.9587,
  `fp16` 1.336 / 1.9546). The combined FO branch is bit-identical to that variant, so one calibration serves both.
- `calibrate_live.py --precision tf32|fp16` runs the variant. `--write` sets it and every combined yaml of that
  precision: `gate` in both families, `gate_effad` in the `or` ones. The `.pt` files stay as they are.
- On the laptop, from the stack root, with the session's clean cu3s (copied from Thor if recorded there):

  ```
  D:\experiments\2026-09-23_walnut-fo-v3\envs\effad_overlay\Scripts\python.exe calibrate_live.py --pipeline or --precision <tf32|fp16> --clean <clean.cu3s> [--clean ...] [--anom <fo.cu3s>] [--write]
  ```

- On Thor, write the values the laptop printed. Nothing runs and no cu3s is read; the system `python3` has PyYAML:

  ```
  cd /home/dev/walnut_fo_stack && python3 calibrate_live.py --precision <tf32|fp16> --set gate=<value> --set gate_effad=<value> [--write]
  ```

- Without `--write` both only print what they would set, next to the current values.
- Why calibrate on the laptop:
  - No cuvis.next child env and no Thor env has `cuvis` + `cuvis_ai_dataloader`, which reading cu3s needs (checked
    28 Sep).
  - `effad_overlay` borrows the stack `.venv`: cuvis 3.5.3.1, core 0.17.3, the editable plugin clones.
  - Laptop vs Thor frame scores differ by ≤ 0.0011, well inside the 1.10 margin.
- Verified 28 Sep on the 22-Sep stand frames (16 clean, 24 FO / fake / hand), laptop:
  - the shipped thresholds reproduce exactly: float32 1.3373 / 1.9587, `tf32` 1.337 / 1.9587, `fp16` 1.336 / 1.9546;
  - both gates flag all 24 anomaly frames at every precision;
  - `--write` / `--set` on copies of the laptop and Thor yamls change exactly the targets' threshold lines (line
    endings and Thor's restamped paths kept).

## Later: one composite output (option b, not built)

- b1: `ScoreMapFusion(mode="first")` over [FO gated map, SEG heatmap]. It shows the FO map when FO alarms, else the
  shell heatmap. It needs no new node, but the two maps have different scales and meanings in one colour map.
- b2: a small composite node with fixed value bands, e.g. the shell mask at a constant level and the FO map rescaled
  into a band above it, so that one heatmap shows both at once.
- Not recommended: gating FO by the shell mask. Shell-gating was shown to swallow fake shells (an open-set
  liability), so the two detectors run side by side.

## Verification (28 Sep)

**Structure** (`build_combo.py` + a check script), all 10 yamls:
- every node's class and hparams equal its source node's;
- the connections are exactly FO ∪ SEG (with the two renames);
- the yaml round-trips through `PipelineBuilder().build_from_config`.

**Branch parity** (`check_combo.py`, on the laptop and on Thor). Frames: the SEG reference frame (1-Sep live f0) and
the two FO reference cubes (clean, fo). All 10 pipelines on both machines are **bit-identical, every array**:
- the FO branch vs the standalone FO variant in the same process state (45 arrays for or, 30 for gated);
- the SEG branch vs the standalone SEG tier (21 arrays), the TensorRT tiers included;
- the FO branch vs the standalone FO variant in its own cuvis.next env, where rfdetr is not imported. So
  `import rfdetr` changes nothing in the FO tiers.
- The gate scores equal the standalone variants' (or TF32: clean 0.9647 / 1.6560 pass 0/0, fo 1.6508 / 3.6483 pass
  1/1).

**Laptop vs Thor** (the same frames):
- every FO gate decision is identical;
- frame scores within 0.0004 (TF32) and 0.0011 (float16);
- displayed FO map within 0.0063;
- shell-mask IoU 0.9980–0.9999.

**Timing** (`bench_combo.py`: the combined pipeline and its two standalone pipelines loaded side by side, interleaved,
CUDA-synchronised, 10 reps × 3 frames). Median ms per frame, "combined (FO + SEG standalone)":

| tier | Thor or | Thor gated | laptop or | laptop gated |
|---|---|---|---|---|
| `tf32_exact` | 182.0 (89.8 + 94.0) | 146.0 (58.2 + 88.3) | 215.0 (109.9 + 92.9) | 173.8 (72.6 + 102.7) |
| `tf32_fast` | 149.0 (91.5 + 61.4) | 113.6 (58.8 + 54.7) | 173.6 (104.9 + 46.0) | 135.6 (71.2 + 65.0) |
| `tf32_trt32` | 141.9 (89.5 + 52.7) | 110.2 (58.1 + 52.3) | 140.3 (96.1 + 43.7) | 107.3 (63.8 + 43.5) |
| `tf32_trt16` | 118.0 (91.1 + 29.4) | 86.5 (59.4 + 28.9) | 115.1 (95.3 + 19.0) | 84.0 (63.9 + 20.7) |
| `fp16_trt16` | **94.8** (66.9 + 28.1) | **70.9** (43.2 + 27.9) | 79.5 (59.6 + 20.1) | 74.2 (51.8 + 23.8) |

- The combined pipeline costs the sum of its two branches: the branches run one after the other, with no measurable
  overhead on Thor.
- First frame: 2.2–2.9 s, and 8.1–8.4 s for the `_fast` tiers (the JIT traces).
- Peak GPU memory 3.2–4.2 GB, with all three pipelines in one process.
- The laptop times drift a little between runs (clocks); compare within one row.
- The FO branch is the bigger half, except in the `_exact` tiers.

**The cuvis.next path** (`grpc_check.py`: the production gRPC server composes the child env, then gRPC vs in-process on
the two FO reference cubes):
- `tf32_exact` and `tf32_fast`, both families: bit-identical on every port on both machines (or 18 ports, gated 13).
- These runs composed the envs cuvis.next now reuses:
  - laptop: or `e2f834cb1d8bfd1a` (169 s), gated `5e1e952c0f54b6d3` (118 s);
  - Thor: or `d0bcacae7e0a7c03` (22 s), gated `618a5a3756fa7f1a` (18 s).
- TensorRT 10.15.1.29 was installed into those four envs with `walnut_seg/check_trt_env.py --env <hash> --include-fo
  --fix` (28 Sep, the user's go). The script's own check passes: all four engines of each machine present. Re-check on
  every demo morning, since a newly composed env loses the install.
- The `_trt` tiers through gRPC, with TensorRT in the composed envs: all six bit-identical on every port, on Thor and on
  the laptop (the laptop run 28 Sep evening; no env composed or evicted).
- So all 10 pipelines pass the cuvis.next path bit-identically on both machines.
- SEG standalone in the new SEG envs: `check_seg_pipelines.py` gives laptop 25 / 25 and Thor 24 / 25 (only the known
  `pca_full`). `_trt_fp16` through gRPC is bit-identical on both.
