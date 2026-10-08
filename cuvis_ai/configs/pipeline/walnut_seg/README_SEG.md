# walnut SEG pipelines (consolidated into walnut_fo_stack)

Self-contained in-pipeline RF-DETR walnut SHELL segmentation, packaged into this stack alongside the FO-anomaly
pipelines so cuvis.next can pick **either** family from one place. Trained end-to-end in cuvis-ai (240-frame full
set, native split + kernels-only negatives); details/metrics in `D:\walnuts\walnut_deploy_v2\RESULTS.md`.

## Pipelines (this folder, cuvis.next cube mode)
| yaml | what | deploy pick |
|---|---|---|
| `walnut_seg_rgb_full` | RGB 640/550/470 single | fastest single, hand-robust (1-Sep ALL 0.944) |
| `walnut_seg_cir_full` | CIR 850/660/550 single | fake-specialist (don't run solo — huge hand-FP) |
| `walnut_seg_pca_full` | top-3 PCA single | — |
| `walnut_seg_ens_rgb_cir_mean` | RGB+CIR ScoreFusion(mean) | **best all-rounder** (ALL 0.956, lowest FP) |
| `walnut_seg_int_rgb_cir` | RGB∧CIR ScoreIntersection | max fake-rejection via agreement |
| `walnut_seg_ens_sam_t13` | ensemble + SAM gate T=13° | **fake-on-real fix**; ~0 erosion on old product |
| `walnut_seg_ens_sam_t11` | ensemble + SAM gate T=11° | stronger fake-rejection (fake→shell 6%) at small clean cost |

All reference `plugins: [cuvis_ai_builtin, rfdetr_seg]`. Weights in `weights/` (checkpoints load via absolute
`checkpoint_path`; the SAM reference is baked into the `_sam_` yamls/`.pt`).

### v2 models (added 2026-09-24) — retrained on old + 22-Sep data with the fair-robust augmentation
| yaml | what | pick |
|---|---|---|
| **`walnut_seg_ens_rgb_cir_mean_v2`** | RGB v2 + CIR v2 ScoreFusion(mean) | **live primary**: all-163 test IoU 0.964, precision 0.989, fake→shell 0 %, no whole-hand FP in any height/blur test |
| `walnut_seg_cir_v2` | CIR v2 single | best single (0.960, fakes 1 %) but can light up a whole hand (worse with a lower camera) — don't run solo with hands in view |
| `walnut_seg_rgb_v2` | RGB v2 single | hand-safe fallback (0.935) |
Weights `weights/rgb_v2_ema.pth` / `weights/cir_v2_ema.pth` (asai1 `out_{perband,cir}_v2aug2/checkpoint_best_ema.pth`,
md5 `0b5b19ac…` / `acfb07bb…`). Built + parity-checked by `build_seg_v2.py`: on the float32 cu3s frame
real_world_live_000 f0 the masks agree with asai1 (same rfdetr 1.10.1) at IoU 0.9998–0.9999; `.npy` drift
references: cir_v2 71684, rgb_v2 71205, ens_v2 70671. Evaluation + decision: `docs/SEG_V2_HEIGHT_ENSEMBLE_2026-09-24.md` (copy of `D:\walnuts\…`).
Note: these models were trained and evaluated with rfdetr 1.8.3 (asai1); this venv runs rfdetr 1.10.1 — the
metric effect is recorded in that doc. Since the 25-Sep stack upgrade (cuvis-ai 0.17.2 + core 0.17.4,
`../../../../../STACK_UPGRADE_2026-09-25.md`), cuvis.next's composed seg child envs run rfdetr **1.11.0**. On both
machines all 19 seg pipelines are bit-identical to the 1.10.1 envs on the reference frame.

### Speed variants `_exact` / `_fast` (added 2026-09-24) — same weights and outputs
Four tiers per v2 model (`rgb_v2`, `cir_v2`, `ens_rgb_cir_mean_v2`); all combine the instance masks on the GPU
(`fast_paste`, the node default, bit-identical to the old per-instance CPU paste):
| suffix | RFDETRSegmenter hparams | result vs default |
|---|---|---|
| (none) | fp32 | — |
| **`_exact`** | fp32 + `gpu_input: true` (frame handed to RF-DETR on the GPU) | **bit-identical** (163 + 37 frames, and heatmap + mask `torch.equal` on the reference frame) |
| **`_fp16`** | `precision: fp16`, `gpu_input: true` (no JIT trace) | fp16 rounding: reference-frame mask IoU vs fp32 0.9994–0.9996 (Thor 0.9986–0.9997); ensemble agreement with before 0.997 on 37 Thor frames; first frame ≈ as fp32 (no trace) |
| **`_fast`** | `precision: fp16`, `jit_trace: true`, `gpu_input: true` | fp16 rounding: reference-frame mask IoU vs fp32 0.9994–0.9996; all-163 ensemble shell IoU 0.963 = fp32 (fake→shell 0.1 %, mask agreement 0.995; `validate_speed_options.py`, option o11); first frame a few seconds longer (JIT trace) |
Built by `build_fast_pipelines.py [exact,fp16,fast]`. Pick `_exact` when the numbers must equal the evaluated fp32
model, `_fast` for the frame rate, `_fp16` when the start-up must be quick (no ~4 s trace per model) at a bit less speed. Speed: see "Speed" below.

### TensorRT variants `_trt_fp32` / `_trt_fp16` (added 2026-09-25) — same weights and outputs
RFDETRSegmenter `backend: tensorrt`: the RF-DETR network runs as a TensorRT engine with rfdetr's own pre- and
post-processing around it, plus `gpu_input` and fast paste (cuvis-ai-rfdetr v0.5.0; the stack checkout runs the same
code at `736605a`).
| suffix | engine | result vs default |
|---|---|---|
| **`_trt_fp32`** | `precision: fp32` = TensorRT default build, TF32 allowed (= what the deployed "fp32" really runs) | reference-frame mask IoU 0.9990–0.9998; 163-frame shell IoU equal or better (ens 0.966 Thor / 0.965 laptop vs 0.964 / 0.963), ens agreement 0.998 |
| **`_trt_fp16`** | `precision: fp16` (TensorRT FP16 builder flag) | reference-frame mask IoU 0.9985–0.9996; 163-frame ens shell IoU 0.964 = fp32, ens agreement 0.999 (Thor) / 0.995 (laptop) |
**Fastest tier:** `_trt_fp16` Thor ens 28.4 ms (35 fps), singles 13 ms (75–78 fps); first frame 1–2 s, no trace.
`_trt_fp32` on Thor runs at about `_fast` speed with fp32-level agreement. Full tables are in "Speed" below.
**Requirements:**
- The `tensorrt` package in the env. It must be TensorRT 10: `tensorrt-cu13==10.15.1.29` on Thor,
  `tensorrt-cu12==10.15.1.29` on the laptop. TensorRT 11 has no FP16 builder flag.
- This machine's engines, built once with
  `python build_fast_pipelines.py trt_fp32,trt_fp16` or `python -m cuvis_ai_rfdetr.trt_engine build-pipeline <yaml>`.
  They land in `weights/{rgb,cir}_v2_ema.pth.trt/<precision>_r504_<GPU>_trt<version>.engine`: fp32 ~135 MB, fp16
  69 MB; build 25–80 s on Thor, 45–170 s on the laptop.
- Built so far for the laptop (RTX 4070) and Thor.

**TensorRT in cuvis.next (28 Sep):** cuvis.next does not install the plugin's optional `tensorrt` extra for node
plugins (cuvis-ai-core#89). So `tensorrt` was installed by hand into the seg child envs with `check_trt_env.py --fix`:
- laptop `~/.cuvis_runs/39e330a00f73dab6`: `tensorrt-cu12==10.15.1.29`;
- Thor `~/.cuvis_runs/35c7cbe28fcb4f11`: `tensorrt-cu13==10.15.1.29`.

The install was additive (3 packages). `check_seg_pipelines.py` run with that env's own Python passes on both machines:
- laptop 25 / 25 (vs `ref/seg_ref_masks_laptop_cuda.npz`);
- Thor 24 / 25, only the known `pca_full`;
- all six `_trt` pipelines at IoU 0.9981–0.9998.

**A hand install survives env reuse, not a new env.** A new env comes with a cuvis.next / core update, a change to a
plugin's `pyproject.toml`, or eviction beyond 10 cached envs. After any of those, and on each demo morning, run
`python check_trt_env.py --fix`. If a `_trt` pipeline stops with "needs the TensorRT Python package", run the fix and
reload the pipeline. A combined FO + SEG pipeline gets its own env: use `--include-fo`. The measured numbers above come
from throwaway overlays of the same envs.

**The `rfdetr_seg_trt` manifest (8 Oct):** the six `_trt` pipelines also list `rfdetr_seg_trt`
(`cuvis_ai/configs/plugins/rfdetr_seg_trt.yaml`): the same source and package as `rfdetr_seg`, with
`extras: [tensorrt]`. The composer of cuvis-ai-core 0.18.1 or later installs cuvis-ai-rfdetr once, with that
extra, into the environments of these six pipelines only: `provision --pipeline-path <yaml>` shows
`cuvis-ai-rfdetr[tensorrt]` for each of them and plain `cuvis-ai-rfdetr` for the others. The hand install above is
then not needed for them. With cuvis-ai-rfdetr 0.5.2 the segmenter also builds a missing engine when the pipeline
loads (a few minutes, once per checkpoint, precision and GPU), into `weights/<checkpoint>.trt/<fingerprint>/`; the
engines built so far, directly in `weights/<checkpoint>.trt/`, keep loading without a build.
**Needs the stack on cuvis-ai-core 0.18.2.** Older schemas reject the manifest (`extras`: extra inputs are not
permitted; checked against the stack's core 0.17.4 / schemas 0.12.0), and all six `_trt` pipelines would then fail
to load. This change therefore lives on its own branch until the stack moves to core 0.18.2 and
cuvis-ai-rfdetr 0.5.2.

## Outputs — heatmap AND mask in cuvis.next's Displayed Output (2026-09-24)
Every pipeline here ends in two output nodes fed by the model scores (`<term>` = `Seg` / `Fuse` / `Inter` /
`Gate`):
- **`ShellHeatmap.scores`** = shell probability heatmap [B, H, W, 1] — `ScoreFusion(mode="max")` (cuvis-ai-rfdetr)
  with BOTH inputs on `<term>.scores`: max(x, x) = x, so it is an exact copy (bit-identical, < 0.2 ms).
- **`ShellMask.decisions`** = boolean shell mask = `scores >= 0.5` — builtin `BinaryDecider`. It applies a sigmoid
  before its threshold and the scores are already probabilities, so its `threshold` is `sigmoid(0.5) = 0.62246`.
  For a stricter/looser mask set it to `sigmoid(t)` for the probability cut `t` you want.

Why this node: cuvis.next's Displayed Output dropdown lists the pipeline's **unconnected** ports and classifies them
**by port name** (read from the QML/strings embedded in CuvisNEXT.exe): `decisions` / `*.mask` → "· mask",
`*scores` → "· heatmap", `*_thickness` / `*value_map` → "· value map", `*rgb_image` / `false_color` → "· image";
ports with other names are not offered. Once ShellMask (23 Sep) consumed `<term>.scores`, no `scores` port was left
unconnected. The first fix (24 Sep morning, builtin `IdentityNormalizer`) had an unconnected port named `normalized`,
which cuvis.next ignores; the user's test showed only `ShellMask.decisions` and the two `rgb_image` ports. The tap now
ends in a port named `scores`. **Confirmed in cuvis.next (laptop, 25 Sep):** the Displayed Output dropdown now
lists `ShellHeatmap.scores · heatmap` next to `ShellMask.decisions · mask` (checked on
`walnut_seg_ens_rgb_cir_mean_v2`; all 16 pipelines end in the same two output nodes). Pick the output explicitly
rather than "Automatic".

Built by `add_heatmap_output.py` from `_backup_pre_heatmap_2026-09-24/` for all 10 pipelines; verified on the
reference frame: heatmap == scores and mask unchanged, bit for bit; both ports in `get_output_specs()`. The
IdentityNormalizer versions are in `_backup_identity_heatmap_2026-09-24/` (ShellMask history: `add_shell_mask.py`,
`_backup_pre_mask_2026-09-23/`).

## Parity (validated on this stack)
Built + forwarded in `cuvis-ai\.venv` (core **0.17.2** — cuvis.next's version). Shell px on the old float16 `.npy`
reference cube are **bit-identical to the 0.14/0.15-era research env**: rgb_full **72424**, cir 63257, pca 62712,
ens_mean 71899, int 62394, sam_t13 69100, sam_t11 66232. See `STACK_SEG_PARITY.md`. That `.npy` is a globally
scaled float16 copy, not what cuvis.next feeds, so it is only a drift reference.

**Current reference (24 Sep): `ref/real_world_live_000_f0000_reflectance.npz`** = frame 0 of the 1-Sep live cu3s
exactly as the cu3s reader delivers it (raw-scale reflectance, stored uint16-lossless). `check_seg_pipelines.py`
forwards all 10 pipelines on it and compares masks with `ref/seg_ref_masks_laptop.npz`:
| pipeline | laptop px | Thor px (agreement IoU) |
|---|---|---|
| rgb_v2 / cir_v2 / ens_rgb_cir_mean_v2 | 71234 / 71743 / 70631 | 71229 (0.9999) / 71730 (0.9997) / 70700 (0.9989) |
| rgb_full / cir_full / ens_rgb_cir_mean | 72617 / 63270 / 72082 | 72627 (0.9993) / 63250 (0.9996) / 72060 (0.9989) |
| int_rgb_cir / ens_sam_t13 / ens_sam_t11 | 62399 / 69270 / 66376 | 62397 (0.9997) / 69249 (0.9989) / 66357 (0.9989) |
| pca_full | 62694 | 70767 (0.886) — see below |
Laptop CPU vs laptop CUDA: bit-identical except pca_full (3 px). Thor (torch 2.14 aarch64, Blackwell) differs in the
last float bits, which moves a few mask cells; the mean ensembles move slightly more (members run at threshold
0.05). **pca_full is numerically fragile on this frame**: same 4 detections on both machines (confidences within
0.01), but over a large area its mask logits sit right at RF-DETR's own mask binarization cut, so last-bit float
differences flip ~8k px of instance mask. It is not a recommended model. The v2 picks agree at IoU ≥ 0.9989.
Speed variants (Thor, 25 Sep, same check over all 16 pipelines): each `_exact` pipeline is bit-identical to its default
on Thor; the `_fast` ones agree with the laptop fp32 reference at IoU 0.9994–0.9995.

## Speed (24 Sep)
### What the RF-DETR node does per frame, and what each speed option changes
The pipeline keeps its data on the GPU from node to node (cube → band selector → 3-band image). The RF-DETR node wraps
rfdetr's `predict()`, which takes NumPy / PIL images and returns NumPy masks, so inside that one node the data made two
trips through the CPU:
1. the node copied the 3-band image GPU → CPU as uint8 NumPy;
2. `predict()` made an unused CPU copy of it ("source image" metadata), copied it CPU → GPU, resized to 504 × 504 and
   normalised;
3. the network ran in fp32, one PyTorch operation at a time from Python;
4. rfdetr upsampled every instance mask to full frame size and returned the masks to the CPU;
5. the node pasted the instances into a CPU score map one by one, then copied the map CPU → GPU.

| option (RFDETRSegmenter hparam) | step | what changes | output |
|---|---|---|---|
| `fast_paste` (default **on**) | 5 | all instance masks go to the GPU at once and are combined with one max | bit-identical |
| `gpu_input` (opt-in) | 1–2 | `predict()` gets the GPU image directly (same uint8 rounding; the unused source-image copy is skipped) | bit-identical |
| `precision: fp16` (opt-in) | 3 | the network runs in half precision on the tensor cores (rfdetr exports + casts it once at load) | rounding: pixels at the 0.5 cut can flip |
| `jit_trace` (opt-in) | 3 | the network is recorded once into a fixed graph and replayed without Python (below) | rounding |

**JIT trace:** PyTorch normally runs the network eagerly — every frame Python walks through hundreds of small layers
and launches each GPU operation separately, which alone costs several ms. `jit_trace` (rfdetr's
`model.inference(compile=True)` = `torch.jit.trace`) runs the network once on an example 1 × 3 × 504 × 504 input,
records every operation into a TorchScript graph and replays that graph in C++ from then on (no Python per layer;
some elementwise operations fused). "Just in time" = it happens at run time, on the first frame, not ahead of time. It
works because the input shape never changes (rfdetr always resizes to 504 × 504, batch 1).

**Why `gpu_input` is bit-identical:** rfdetr 1.10 widens its uint8 frame on the GPU and divides by a 0-dim *tensor*
255; the node does the same. Dividing by the Python number 255 on CUDA is computed as a multiplication by 1/255 and
differs from correctly rounded division for 126 of the 256 byte values (1 ULP) — why the morning runtime patch was
only ~0.999-identical. Still on the CPU after all options: rfdetr's post-processing returns the instance masks as NumPy
(step 4); removing that needs own GPU post-processing instead of `predict()` (≈ 5–8 ms per model, not done).

### Numbers
**Laptop RTX 4070, GPU otherwise idle** — interleaved A/B (`bench_node_options.py`: 37 real cu3s frames held in
memory, every variant runs back to back on the same frame in rotating order, 3 reps = 111 samples each; median ms of
`pipeline.forward`, cube already on the GPU):
| median ms (fps) | rgb_v2 | cir_v2 | ens_rgb_cir_mean_v2 | mask IoU vs before |
|---|---|---|---|---|
| before today (per-instance CPU paste) | 55.7 (18.0) | 61.6 (16.2) | 120.9 (8.3) | — |
| **default** = fp32 + combined masks | 55.3 (18.1) | 59.7 (16.8) | 113.7 (8.8) | 1.0 (bit-identical) |
| **`_exact`** = + GPU input | 50.8 (19.7) | 54.3 (18.4) | 106.5 (9.4) | 1.0 (bit-identical) |
| **`_fast`** = fp16 + JIT + GPU input | **32.6 (30.6)** | **37.0 (27.0)** | **73.4 (13.6)** | 0.995 / 0.988 / 0.997 |
| first frame (model build; `_fast` incl. trace) | 0.8 s / 5.2 s | 0.8 s / 4.4 s | 1.7–1.9 s / 9.1 s | |
On the laptop's fast x86 CPU the combined masks gain little (1–6 %); on Thor's ARM CPU the per-instance paste costs
far more. Frames arriving with pauses (live camera, or the cu3s profiler's 380 ms decode between frames) let the GPU
clock down, so expect higher ms there (the morning profiler: singles 79–89 ms, ensemble 161–207 ms).

**Inside cuvis.next (laptop, 25 Sep, the app's own `MAGIC PROFILING` log lines; ensemble v2 on the 11-frame 22-Sep
hand recording):** per frame after the first — default ≈ 104 ms (one warm frame), `_exact` ≈ 89 ms (medians SegRGB
48.8 + SegCIR 40.4), `_fast` ≈ 63 ms (33.6 + 29.3, ≈ 15 fps). First frame: `_exact` ≈ 1.4 s (model build), `_fast`
≈ 8.9 s (JIT trace of both members, ≈ 4.8 + 4.0 s; once per pipeline load, not per frame). User-confirmed 25 Sep:
both `_fast` and `_exact` run in cuvis.next and show heatmap + mask.

**Thor (Jetson AGX Thor, MAXN, cuvis.next's seg child env)** — same interleaved A/B with the plugin code (24 Sep
14:15; the `exact` row = the node flipped to `gpu_input`, the `_exact` configuration):
| median ms (fps) | rgb_v2 | cir_v2 | ens_rgb_cir_mean_v2 | mask IoU vs before |
|---|---|---|---|---|
| before today (per-instance CPU paste) | 52.5 (19.0) | 53.9 (18.5) | 145.9 (6.9) | — |
| **default** = fp32 + combined masks | 46.6 (21.4) | 46.1 (21.7) | 103.3 (9.7) | 1.0 (bit-identical) |
| **`_exact`** = + GPU input | 42.5 (23.6) | 42.0 (23.8) | 91.5 (10.9) | 1.0 (bit-identical) |
| **`_fp16`** = fp16 + GPU input, no JIT (25 Sep run) | 31.5 (31.7) | 31.8 (31.4) | 68.9 (14.5) | 0.985 / 0.981 / 0.997 |
| **`_fast`** = fp16 + JIT + GPU input | **24.2 (41.3)** | **24.4 (41.0)** | **56.0 (17.9)** | 0.975 / 0.991 / 0.993 |
| first frame (fp32 / `_fp16` / `_fast` incl. trace) | 0.7 / 0.9 / 3.7 s | 0.7–0.9 / 0.8 / 3.9 s | 1.4 / 1.5 / 7.6 s | |
| p90 ms before → `_fast` | 65.8 → 25.8 | 66.1 → 25.8 | 227.1 → 73.3 | |
Repeat run 25 Sep: every cell within ±2 ms (ensemble 146.1 → 104.8 → 92.1 → 55.1 ms).
**With the TensorRT tiers (25 Sep evening, all 7 variants side by side, seg child env 35c7 + overlay):**
| median ms (fps) | before | default | `_exact` | `_fp16` | `_fast` | **`_trt_fp32`** | **`_trt_fp16`** |
|---|---|---|---|---|---|---|---|
| Thor rgb_v2 | 57.2 | 50.6 | 43.0 | 31.8 | 24.3 (41) | 27.6 (36) | **13.3 (75.5)** |
| Thor cir_v2 | 54.7 | 47.3 | 42.5 | 31.4 | 23.8 (42) | 27.3 (37) | **12.9 (77.8)** |
| **Thor ens** | 145.1 | 99.6 | 89.4 | 66.5 | 52.6 (19.0) | 57.5 (17.4) | **28.4 (35.2)** |
| laptop rgb_v2 | 61.4 | 57.5 | 47.3 | 43.5 | 31.4 (32) | 23.4 (43) | **11.8 (85.0)** |
| laptop cir_v2 | 58.2 | 55.2 | 48.8 | 45.0 | 32.5 (31) | 24.3 (41) | **11.7 (85.8)** |
| **laptop ens** | 138.1 | 118.5 | 108.3 | 99.9 | 74.4 (13.4) | 49.2 (20.3) | **24.9 (40.1)** |
| first frame ens (`_fast` / `_trt_fp32` / `_trt_fp16`) | | | | | Thor 7.6 s / laptop 17.2 s | 2.0 / 3.0 s | 1.9 / 2.8 s |
Ensemble mask IoU vs before (pooled over the 37 bench frames): Thor `_fast` 0.993, `_trt_fp32` 0.999, `_trt_fp16` 0.998.
 On Thor the combined masks alone already give the ensemble 1.41× (its members run at threshold 0.05 → many
instances, and Thor's ARM CPU pastes slowly); `_fast` = 2.2× singles / 2.6× ensemble. The `_fast` singles' lower
agreement (rgb 0.975 pooled) comes from a few knife-edge hand frames flipping — the ensemble stays at 0.993. The cu3s
profiler on Thor (frames with the 118 ms decode in between) gives the same ranking: before 55.5 / 59.7 / 165.3 ms
(live cu3s) → default 51.8 / 55.6 / 126.4 → `_exact` 47.8 / 47.7 / 112.3 → `_fast` 28.9 / 28.8 / 78.6. Raw data:
`docs/SEG_V2_PROFILING_2026-09-24.md` (+ Thor `/home/dev/walnut_node_options_2026-09-24/`).

## Run in cuvis.next (same recipe as FO — THOR_DEPLOY_NOTES §9)
- Settings Directory = `<clone>\cuvis_ai\configs` (this folder is under it). The seg nodes come from the
  `rfdetr_seg` manifest (`configs/plugins/rfdetr_seg.yaml`, absolute local path → the `cuvis-ai-rfdetr` checkout);
  FO's `rfdetr.yaml` (ScoreFusion-only) is untouched.
- Per pipeline set BOTH `Pipeline = …\<name>.yaml` AND `Weights = …\<name>.pt` (blank Weights → "No weights file").
- cuvis.next composes a child env per pipeline and installs `cuvis-ai-rfdetr` editable + `rfdetr`; the stack's
  `cuvis-ai\.venv` also has them now (seg installed via `uv pip install --no-sources -e ..\cuvis-ai-rfdetr`).
- **HF model cache:** RF-DETR loads its finetuned weights from the checkpoint (the "not loading DINOv2 backbone"
  warning is benign), but first build may fetch the RF-DETR-large base config from HF. Pre-populate
  `<modelCacheDirectory>\hf` on Thor (mirror the FO SteerViT/dinov2 step) — verify which repo it pulls on first run.

## Another machine / Thor
Done for Thor on 24 Sep — see `THOR_RUN_SEG.md` (state, pipeline order, cuvis.next rules). For a fresh machine:
copy the stack without `.venv`, rebuild the env (`uv pip install --no-sources …`, THOR_DEPLOY_NOTES §5), run
`restamp_thor.py --root <new stack root>`, make sure the FULL `cuvis_ai_builtin.yaml` is in the catalog, then
`check_seg_pipelines.py --ref ref/seg_ref_masks_laptop.npz` (expect agreement IoU ≥ 0.998 for the v2 pipelines).

## Scripts in this folder
| script | what |
|---|---|
| `build_seg_v2.py` | builds the 3 v2 pipelines (yaml + .pt + picker .pt) and parity-checks them vs asai1 |
| `add_shell_mask.py` / `add_heatmap_output.py` | added the ShellMask (23 Sep) / ShellHeatmap (24 Sep) outputs |
| `build_fast_pipelines.py` | builds the `_exact` / `_fp16` / `_fast` / `_trt_fp32` / `_trt_fp16` variants of the 3 v2 pipelines (the TensorRT ones build this machine's engines first) and checks them against the default |
| `check_seg_pipelines.py` | forwards all pipelines on `ref/` and compares masks with a reference file (`--ref`, `--write`; `_fast` / `_fp16` / `_trt_*` are compared with their fp32 counterpart at IoU ≥ 0.99; `_trt_*` are skipped in envs without tensorrt) |
| `validate_trt_node.py` | 163-frame GT accuracy + node timing of the `_trt_fp32` / `_trt_fp16` pipelines vs the deployed fp32 ones |
| `validate_trt.py` | the same check with a hand-built engine next to the node (the pre-plugin TensorRT check) |
| `check_trt_env.py` | is `tensorrt` 10.15.1.29 in the cuvis.next child env the seg pipelines use, and are this machine's 4 engines there? `--fix` installs it (additive, 3 packages); `--include-fo` for the env of a combined FO + SEG pipeline, `--env <hash>` for one env. A hand install survives env reuse, not a new env (cuvis.next / core update, plugin `pyproject.toml` change, eviction beyond 10 envs) |
| `print_detections.py` | prints a pipeline's RF-DETR detections on the reference frame (cross-machine diagnosis) |
| `profile_seg_cu3s.py` | cuvis-ai profiler on real cu3s (synchronised, warm-up skipped; `--variant` flips node options) |
| `bench_node_options.py` | interleaved A/B timing of the node speed options on real cu3s frames held in memory |
| `net_share_probe.py` | splits one frame's time into network vs rest of the RF-DETR node vs other nodes (what TensorRT could speed up) |
| `trt_probe.py` | times the RF-DETR network on a real frame: fp32 / fp16 / fp16 JIT / Torch-TensorRT (throwaway overlay with torch-tensorrt), with mask parity |
| `trt_onnx_export.py` / `trt_parity.py` | rfdetr-style ONNX export for a TensorRT (`trtexec`) build + mask-level parity of the engine vs fp32 |
| `verify_node_speed_options.py` | bit-exactness of fast paste / GPU input vs the per-instance paste (163 cached frames or `--cu3s`) |
| `speed_options.py` / `validate_speed_options.py` | the 24-Sep option exploration (runtime patches) and its 163-frame GT check |
| `restamp_thor.py` | rewrites the absolute stack paths for another machine (idempotent) |

## Provenance
The seg nodes (`RFDETRSegmenter` incl. its speed hparams, `ScoreFusion` (also the `ShellHeatmap` tap),
`ScoreIntersection`, `SamShellGate`, `FixedPCAProjection`) come from `cuvis-ai-rfdetr`:
- PR #12 (merged) is released as **v0.5.0** (28 Sep; the TensorRT backend is included).
- **v0.5.1** only adds the optional `tensorrt` extra (PR #20). cuvis.next does not install extras of node plugins yet
  (cuvis-ai-core#89).
- The stack checkouts (laptop git checkout, Thor copy) stay at `736605a`: the same node code as v0.5.0 but an older
  `pyproject.toml`, so the child envs are not recomposed before the fair.

`BinaryDecider`, `FixedWavelengthSelector`, `CU3SDataNode` are cuvis-ai
builtins (the FULL `cuvis_ai_builtin.yaml` must be in the catalog — the old 3-node FO trim lacks `BinaryDecider`).
Pipeline yamls and weights are untracked deploy artifacts (they travel with the folder copy, like `walnut_fo/`).
