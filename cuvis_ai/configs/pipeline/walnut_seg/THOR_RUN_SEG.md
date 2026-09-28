# SEG on Thor — fair-demo run checklist

Companion to `../../../../../THOR_DEPLOY_NOTES.md` (FO chat's notes). The **shared** steps are identical for SEG:
env rebuild = §5, cuvis.next recipe = §9, HF model-cache = §9. This file adds only the SEG specifics.

**Copy staged at `Z:\anish\walnut_fo_stack`** (venvs / `.git` / caches excluded — rebuild the venv on Thor).

## State on Thor (updated 2026-09-24 by the SEG chat)
Thor `dev@<thor>`, stack at **`/home/dev/walnut_fo_stack`** (already there). On 24 Sep the SEG files were
refreshed directly from the laptop (scp/tar, not via the NAS): all 10 `walnut_seg` pipelines (with the two outputs
below), the v2 weights, `ref/`, the scripts, the `cuvis-ai-rfdetr` code (PR #12 head) and the FULL
`cuvis_ai_builtin.yaml`; then `restamp_thor.py --root /home/dev/walnut_fo_stack` (0 laptop paths left).
Pre-change Thor files: `/home/dev/walnut_fo_stack/_backup_thor_pre_2026-09-24/`.
- **Builtin manifest:** Thor still had the 21-Sep 3-node FO trim; the pipelines' `ShellMask` (BinaryDecider) is not
  in it, so it would fail with "not provided by any plugin" (the heatmap tap is `ScoreFusion` from `rfdetr_seg`). Replaced by
  the full list (152 classes, all import in cuvis.next's seg child env on Thor); the trimmed file is in the backup
  folder and `cuvis_ai_builtin.full.yaml.bak` is unchanged. FO pipelines are unaffected (full = superset).
- **Stack upgraded 25 Sep** to cuvis-ai 0.17.2 + core 0.17.4, which the new cuvis.next requires
  (`../../../../../STACK_UPGRADE_2026-09-25.md`).
  - cuvis.next's seg child env on Thor is now `~/.cuvis_runs/35c7cbe28fcb4f11/.venv` (Python 3.11, torch 2.14, core
    0.17.4, rfdetr 1.11.0, cuvis-ai-rfdetr editable from the stack).
  - All 19 seg pipelines are bit-identical to the previous env, `~/.cuvis_runs/0b55f16fc78a1178/.venv` (core 0.17.3,
    rfdetr 1.10.1), which the pipelines were forwarded and profiled in on 24/25 Sep.
- Parity vs the laptop on the reference frame: v2 masks agree at IoU 0.9989–0.9999 (table in `README_SEG.md`).

## Which SEG pipelines to test live (in order)
1. **`walnut_seg_ens_rgb_cir_mean_v2`** — RGB v2 + CIR v2 mean fusion. **The main demo model** (all-163 test IoU
   0.964, fake→shell 0 %, no whole-hand false positive in any height/blur test).
2. **`walnut_seg_rgb_v2`** — single RGB v2, the fast hand-safe fallback.
3. `walnut_seg_cir_v2` — best single on fakes, but **not solo with hands in view** (can paint a whole hand).
4. Older deployed set for A/B: `walnut_seg_ens_rgb_cir_mean` (old main), `walnut_seg_ens_sam_t13` / `_t11`
   (fake-on-real gate), `rgb_full`, `int_rgb_cir`, `cir_full`, `pca_full` (do not run `cir_full` solo).

**Speed tiers of 1–3 (24 Sep):** each v2 pipeline also exists as `<name>_exact` (fp32 + frame handed to RF-DETR on
the GPU — bit-identical to `<name>`, just faster) and `<name>_fast` (fp16 + JIT + GPU input — fastest; masks agree
with fp32 at IoU ≈ 0.995, ensemble accuracy unchanged on the 163 test frames; the first frame takes a few seconds
longer because the network is traced then). For the demo: `walnut_seg_ens_rgb_cir_mean_v2_fast` if the frame rate
matters, `…_exact` if the numbers must equal the evaluated model, `…_fp16` (fp16 without JIT, 25 Sep) if the start-up
must be quick: first frame 1.5 s instead of 7.6 s, ensemble 69 ms (14.5 fps) instead of 54–56 ms. Every pipeline combines the instance masks on the
GPU now (node default, bit-identical). What the options do: `README_SEG.md` → Speed.

**TensorRT tiers (25 Sep evening):** `<name>_trt_fp16` / `<name>_trt_fp32` run the RF-DETR network as a TensorRT
engine.
- Thor, all tiers side by side (ensemble / single, median ms): `_trt_fp16` **28.4 ms (35 fps) / 13 ms (75–78 fps)**,
  first frame 1–2 s; `_trt_fp32` 57.5 / 27.5 ms. That compares with `_fast` 52.6 / 24 ms and default 99.6 / 47–51 ms.
- Accuracy on the 163 GT frames is unchanged. Ensemble shell IoU: fp32 0.966, fp16 0.964, PyTorch fp32 0.964.
- Engines for Thor are built: `weights/{rgb,cir}_v2_ema.pth.trt/{fp32,fp16}_r504_NVIDIA-Thor-sm110_trt10.15.1.29.engine`.
- **Runnable in cuvis.next since 28 Sep:** `tensorrt-cu13==10.15.1.29` was installed by hand into the seg child env
  `~/.cuvis_runs/35c7cbe28fcb4f11` with `python3 check_trt_env.py --fix`. The install was additive (3 packages).
  - `check_seg_pipelines.py` run with that env's Python: 24 / 25 OK (only the known `pca_full`), the `_trt` pipelines
    at IoU 0.9981–0.9996. Log: `/home/dev/walnut_trt_node_2026-09-25/childenv_check_2026-09-28.log`.
  - It must stay TensorRT 10 (TensorRT 11 has no FP16) and match the version in the engine names.
- **Re-check on each demo morning and after any cuvis.next update:** `python3 check_trt_env.py --fix`. A new env (core
  update, plugin `pyproject.toml` change, eviction beyond 10 envs) comes without TensorRT. A combined FO + SEG pipeline
  has its own env: `--include-fo`. Overlay runner for measurements: `/home/dev/walnut_trt_node_thor.sh`.

**Outputs:** every pipeline ends in `ShellHeatmap.scores` (heatmap) and `ShellMask.decisions` (mask). cuvis.next's
Displayed Output lists them as `ShellHeatmap.scores · heatmap` and `ShellMask.decisions · mask` — **confirmed in
cuvis.next on the laptop, 25 Sep** (it classifies unconnected ports by name; the earlier `ShellHeatmap.normalized` was
not offered — fixed 24 Sep 13:39, see `README_SEG.md` → Outputs). Pick the one you want explicitly; do not rely on
"Automatic".

## SEG-specific notes (on top of §5 / §9)
- **RF-DETR backbone from HF:** RF-DETR-large loads its DINOv2 backbone from `models--timm--vit_base_patch14_dinov2.lvd142m` (+ `vit_small`) — **already in §9's HF pre-populate list**, so populating the model cache per §9 covers SEG. The fine-tuned weights come from the `.pth` checkpoint; the "not loading DINOv2 backbone" warning is benign. No HF repo beyond §9's list. (Confirm on the first online build.)
- **rfdetr install = same `--no-sources` rule as §5** (the plugin pins cu128 torch in `[tool.uv.sources]`; on Jetson install aarch64 torch first, then `uv pip install --no-sources -e ./cuvis-ai-rfdetr` + `rfdetr`). Needs `rfdetr>=1.8` (the seg tier).
- **Seg manifest is separate:** `cuvis_ai/configs/plugins/rfdetr_seg.yaml` (exposes RFDETRSegmenter / FixedPCAProjection / ScoreFusion / ScoreIntersection / SamShellGate). FO's `rfdetr.yaml` (ScoreFusion-only) is untouched — both coexist.
- **Absolute paths → re-stamp:** the seg yamls' `checkpoint_path` / `projection_path` and `rfdetr_seg.yaml`'s `path:` are absolute `D:/walnuts/walnut_fo_stack/…`. Run `restamp_thor.py` once on Thor — it rewrites them (and the FO manifests) to the Thor root in one pass.

## Ordered steps on Thor
1. Copy `Z:\anish\walnut_fo_stack` → `<thor-root>` (e.g. `/home/anish/walnut_fo_stack`).
2. **Rebuild the cuvis.next server venv** in the clone (§5/§9): install Jetson aarch64 torch/torchvision first, then the `uv pip install --no-sources -e ./cuvis-ai -e ./cuvis-ai-rfdetr … cuvis-ai-core==0.17.4 cuvis-ai-schemas[full]==0.12.0 cuvis==3.5.3.1` line from §5 (it already includes `cuvis-ai-rfdetr`; §5 still says core 0.17.3, but the cuvis.next build from 25 Sep requires ≥ 0.17.4 and the stack checkout is cuvis-ai v0.17.2 — without `.git` set `SETUPTOOLS_SCM_PRETEND_VERSION=0.17.2`).
3. **Re-stamp paths:** `python cuvis-ai/cuvis_ai/configs/pipeline/walnut_seg/restamp_thor.py --root <thor-root>` (rewrites `D:/walnuts/walnut_fo_stack` → `<thor-root>` across all `plugins/*.yaml` + `pipeline/**/*.yaml`, seg **and** FO). Add `--dry` first to preview.
4. **Pre-populate the HF cache** per §9 (copy the `models--…` dirs incl. `timm vit_base/small dinov2`; bridge `\hf`↔`\hub`).
5. **cuvis.next:** cuvis.ai Directory = `<thor-root>/cuvis-ai`; Settings Directory = `<thor-root>/cuvis-ai/cuvis_ai/configs`; **do NOT** "Take ownership". Per pipeline set BOTH `Pipeline = …/walnut_seg/<name>.yaml` AND `Weights = …/walnut_seg/<name>.pt`.
6. Load `walnut_seg_rgb_full` on a cu3s → confirm a shell mask renders. Then `ens_rgb_cir_mean`, then `ens_sam_t13`.
7. **Parity smoke (optional):** forward the RGB anchor on the reference cube; expect ~**72424** shell px (numeric parity across Jetson torch is expected but unverified — re-confirm here).

## Speed on Thor
Measured 24 Sep with cuvis-ai's profiler on real cu3s in the seg child env (MAXN): **rgb_v2 56–65 ms/frame
(15–18 fps), cir_v2 63–69 ms (14.5–16 fps), ensemble v2 169–186 ms (≈ 6 fps)**; cu3s read + reflectance 118 ms/frame.
Details + reproduction: `docs/SEG_V2_PROFILING_2026-09-24.md` (raw logs: Thor `/home/dev/walnut_profiling_2026-09-24/`). The ensemble
runs its two RF-DETR passes one after the other (cost = sum of both).

**Speed options (24 Sep pm, RFDETRSegmenter hparams — `README_SEG.md` → Speed):** every pipeline now combines the
instance masks on the GPU (default, bit-identical); `_exact` adds GPU input (bit-identical), `_fast` adds fp16 + JIT.
Thor, interleaved A/B with the plugin code (median ms, rgb / cir / ensemble): before 52.5 / 53.9 / 145.9 → default
(combined masks) 46.6 / 46.1 / 103.3 → `_exact` 42.5 / 42.0 / 91.5 → **`_fast` 24.2 / 24.4 / 56.0 ms (41 / 41 / 18
fps)**; first frame of `_fast` 3.7 s (single) / 7.6 s (ensemble) because of the JIT trace. Raw data in
`/home/dev/walnut_node_options_2026-09-24/`; bit-exactness of combined masks + GPU input on Thor: 37/37 frames
`torch.equal` (`verify.log`); `check_seg_pipelines.py` on 25 Sep: the 3 `_exact` pipelines bit-identical to their
defaults on Thor, the `_fast` ones at IoU 0.9994–0.9995 vs the fp32 reference (`check_pipelines_0925.log`).

**TensorRT (25 Sep evening; cuvis-ai-rfdetr 736605a deployed to the editable plugin checkout, backup
`_backup_thor_pre_2026-09-25_trt/`).** Interleaved A/B in the seg child env 35c7 + overlay
(`tensorrt-cu13==10.15.1.29`, `onnx`), median ms rgb / cir / ensemble:

| tier | rgb / cir / ens (ms) | first frame |
|---|---|---|
| before | 57.2 / 54.7 / 145.1 | |
| default | 50.6 / 47.3 / 99.6 | |
| `_exact` | 43.0 / 42.5 / 89.4 | |
| `_fp16` | 31.8 / 31.4 / 66.5 | |
| `_fast` | 24.3 / 23.8 / 52.6 | 3.9 / 3.7 / 7.6 s |
| `_trt_fp32` | 27.6 / 27.3 / 57.5 | 1.0 / 1.1 / 2.0 s |
| **`_trt_fp16`** | **13.3 / 12.9 / 28.4** | 1.0 / 1.0 / 1.9 s |

- Engine builds on Thor took fp32 25–30 s and fp16 76–79 s.
- `check_seg_pipelines.py`: 24 / 25 OK (only the known `pca_full`). `_exact` is still bit-identical; the `_trt` pipelines
  are at IoU 0.9981–0.9996 vs the laptop fp32 reference.
- 163-frame GT check through the node: accuracy equal or better for every pipeline (`validate_trt_node.py`).

## Getting the stack from Z: onto Thor (Nima's recipe, 22 Sep)
`Z:` **is** `\\DSNAS1\CompanyCache`, so the synced stack already lives on the NAS at
`\\dsnas1\CompanyCache\anish\walnut_fo_stack`. Thor pulls it **directly from the NAS** — no scp needed, and
faster than laptop→Thor scp (single hop vs double).
- Thor host: `dev@<thor>`. CuvisNEXT binary: `/opt/Cubert/CuvisNEXT/bin/CuvisNEXT`.
- **Mount the NAS on Thor and copy (preferred for the 5.3 GB stack):**
  ```bash
  sudo apt install cifs-utils
  sudo mkdir -p /mnt/companycache
  sudo mount -t cifs //dsnas1/CompanyCache /mnt/companycache -o username=<your-nas-user>,uid=$(id -u),gid=$(id -g),vers=3.0
  cp -r /mnt/companycache/anish/walnut_fo_stack ~/walnut_fo_stack      # <thor-root>
  # wafer test cu3s (optional): cp -r /mnt/companycache/anish/wafer_thickness_cuvisnext ~/
  ```
  (mount prompts for the NAS password — enter it on Thor.)
- **scp alternative** (simpler for a single file, e.g. one cu3s; slower for the whole stack): from the laptop
  `scp -r "Z:\anish\walnut_fo_stack" dev@<thor>:/home/dev/` (prompts for Thor's `dev` password).
- After copying: rebuild the venv (§2 above), `restamp_thor.py --root ~/walnut_fo_stack`, pre-populate the HF cache,
  then run `/opt/Cubert/CuvisNEXT/bin/CuvisNEXT` and point it at the clone (§5).

## cuvis.next operating rules (learned on the laptop, 22 Sep — apply on Thor too)
Verified live: after these, `walnut_seg_rgb_full` + `walnut_seg_ens_rgb_cir_mean` load, run, and display in cuvis.next.
1. **Manifest needs `package_name`.** Without it the child env registers but never installs the plugin →
   `No module named 'cuvis_ai_rfdetr'`. Fixed in `rfdetr_seg.yaml` (`package_name: cuvis-ai-rfdetr`); FO/wafer already have it.
2. **Always SET the Weights field** in the picker (Pipeline `.yaml` **and** Weights `.pt`). Blank Weights →
   "No weights file for pipeline (cuvis_picker_<name>.pt)" (it hunts `%TEMP%`, not the pipeline folder).
3. **One child env per cuvis.next SESSION**, composed for the family you load FIRST. Loading seg then FO (or wafer)
   in the same session → `No module named 'cuvis_ai_patchcore'` etc. **Restart cuvis.next between families**
   (seg / fo / wafer) — one family per app session.
   > **⚠️ TO SOLVE (raise with Nima / cuvis.next team):** this restart-between-families requirement is a blocker for
   > a live demo that shows seg + FO + wafer. cuvis.next needs to recompose/extend the per-pipeline child env when the
   > next pipeline needs different plugins (or support multi-plugin-family sessions), instead of failing with
   > `No module named 'cuvis_ai_<plugin>'` until a restart.
- **Display:** the shell mask is `Seg.scores` (magic-mask overlay on `rgb_full`) or `SegRGB/SegCIR/Fuse.scores`
  (selectable graph-output layers on the ensemble) — same layer selector as FO's heatmap. The RF-DETR
  "not loading DINOv2 backbone" / "unmapped class_id 0" warnings are benign.
