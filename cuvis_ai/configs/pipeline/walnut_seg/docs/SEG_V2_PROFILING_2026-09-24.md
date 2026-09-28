# SEG v2 pipelines — cuvis-ai profiling on real cu3s (laptop + Thor), 2026-09-24

User ask (24 Sep): run the packaged pipelines with cuvis-ai profiling on the venv we use for Thor, single and
ensemble, on cu3s files (not npy), laptop + Thor; document results, methods and reproduction.

## What was profiled
| pipeline (cuvis.next cube mode, `walnut_seg/`) | model(s) |
|---|---|
| `walnut_seg_rgb_v2_cuvisnext_cube` | B — RGB 640/550/470 RF-DETR-L seg (single) |
| `walnut_seg_cir_v2_cuvisnext_cube` | C — CIR 850/660/550 RF-DETR-L seg (single) |
| `walnut_seg_ens_rgb_cir_mean_v2_cuvisnext_cube` | B + C, ScoreFusion(mean) (ensemble; 2 RF-DETR passes) |

All three end in `ShellMask` (BinaryDecider, mask) + `ShellHeatmap` (heatmap). In the morning runs below the heatmap
tap was an `IdentityNormalizer` (port `normalized`), which cuvis.next does not offer; since 13:39 it is
`ScoreFusion(max)` of the scores with themselves (port `scores`, bit-identical copy) so the dropdown lists it (see
`walnut_seg/README_SEG.md` → Outputs). Either tap costs < 0.2 ms, so the timings are unaffected.

Data (real recordings, read through cuvis-ai's cu3s DataModule in Reflectance mode — the npy frame is used only
for the bit-exact parity checks, never for timing):
- `2026_09_01/12-16-58/real_world_live_000.cu3s` — 26 frames, live scene with hands (1.54 GB)
- `2026_09_22/14-39-28/kernel_shell_hand_other_d_000+03.cu3s` — 11 frames, hands + puck test scene (0.83 GB)

## Method (cuvis-ai-inference skill)
Two runs per machine, both with cuvis-ai's built-in node profiler (`CuvisPipeline.set_profiling`, per-node
count / mean / std / min / max / median):
1. **Path 1 — `restore-pipeline` CLI (the supported entry point)**: `--data-module cu3s --device cuda`. The CLI
   enables the profiler without CUDA synchronisation and without skipping warm-up, so its per-node split is
   approximate and its mean includes the first (warm-up) frame; use its **median** column.
2. **Path 3 — `walnut_seg/profile_seg_cu3s.py`** (CuvisPipeline + Cu3sDataModule + Predictor):
   `set_profiling(enabled=True, synchronize_cuda=True, skip_first_n=2)` — per-node GPU wall-clock and a steady
   state without warm-up. Also measures the cu3s read + reflectance cost per frame (DataModule iterated alone)
   and peak CUDA memory. **These are the headline numbers.**

Env: the stack env that cuvis.next and Thor use (cuvis-ai 0.17.1 editable, core 0.17.2, rfdetr 1.10.1,
cuvis-ai-rfdetr = PR #12 head, torch 2.11 cu128 on the laptop). The cu3s reader (`cuvis-ai-dataloader[cu3s,coco]`
+ `cuvis` 3.5.3.2 matching the laptop SDK 3.5.3) is layered in with `uv run --no-sync --with …` — not installed
into the stack venv (skill rule; `--no-sync` so uv does not re-sync and drop the editable plugin). The overlay
re-links identical versions of numpy / scipy / torch / core (checked), so timings are the stack env's.

## Results — laptop (RTX 4070 Laptop GPU, Windows 11, torch 2.11.0+cu128)
Path 3, steady state (2 warm-up frames skipped, CUDA-synchronised per node):

| pipeline | ms / frame (live 1-Sep) | ms / frame (hand 22-Sep) | fps | peak CUDA memory |
|---|---|---|---|---|
| rgb_v2 (single) | 79.2 | 84.2 | 11.9–12.6 | 577 MiB |
| cir_v2 (single) | 89.2 | 80.8 | 11.2–12.4 | 716 MiB |
| **ens_rgb_cir_mean_v2** | 206.9 (median-based ≈ 175) | 160.8 | 4.8–6.2 | 869 MiB |

Per node (live scene, mean ms): rgb_v2 = Seg 74.9 · DataSource 2.5 · Selector 1.6 · ShellMask 0.19 ·
ShellHeatmap 0.02. Ensemble = SegRGB 106.3 (median 87.7) · SegCIR 95.7 (median 81.9) · DataSource 2.1 · SelRGB 1.3 ·
SelCIR 1.2 · Fuse 0.18 · ShellMask 0.12 · ShellHeatmap 0.02. **The RF-DETR node is ≥ 95 % of the pipeline time;
the ensemble costs the sum of its two members** (the two branches run one after the other). The new heatmap output
costs nothing (0.02 ms, it is a passthrough).

Reading the recording: **cu3s decode + reflectance ≈ 325 ms / frame** (SDK 3.5.3) plus a one-off ~35 s to open a
cu3s and set up the processing context. That is a file-playback cost — in live mode the camera path in cuvis.next
does the acquisition/reflectance, not this DataModule — so it is reported separately and not added to the fps.

Path 1 CLI (reference; no sync, warm-up included; 1-Sep live, 26 frames): rgb_v2 Seg median 88.6 ms (mean 206 ms
incl. a 3.2 s first frame), cir_v2 Seg median 74.3 ms, ensemble SegRGB 64.8 + SegCIR 73.5 ms medians (mean 229 ms).
Consistent with Path 3 within the no-sync attribution error.

GPU state during the laptop run: P0, 1605 MHz SM clock at the before/after snapshots (max 3105 MHz), ~12 W at idle —
the laptop was NOT in a pinned performance/turbo mode, so these are conservative; earlier turbo-on measurements were
~1.5–2× faster for the same models.

## Results — Thor
Thor `<thor-host>` (Jetson AGX Thor, NVIDIA Thor GPU sm_110, JetPack R38.2, NV power mode **MAXN**), in
cuvis.next's own seg child env there (`~/.cuvis_runs/0b55f16fc78a1178/.venv`: Python 3.11, **torch 2.14.0+cu130**,
core 0.17.3, rfdetr 1.10.1, cuvis-ai-rfdetr editable from the stack) + the cu3s reader overlay (`cuvis` 3.6.0 to match
Thor's SDK 3.6.0; `--no-sources` so the plugins' cu128 torch pin is ignored and the Jetson torch is used). Same two
cu3s files (md5-verified copies). GPU otherwise idle (only the desktop session).

Path 3, steady state:

| pipeline | ms / frame (live 1-Sep) | ms / frame (hand 22-Sep) | fps | peak CUDA memory |
|---|---|---|---|---|
| rgb_v2 (single) | 56.5 | 65.4 | 15.3–17.7 | 592 MiB |
| cir_v2 (single) | 62.9 | 69.1 | 14.5–15.9 | 731 MiB |
| **ens_rgb_cir_mean_v2** | 169.2 (median-based ≈ 138) | 186.1 | 5.4–5.9 (≈ 7 on medians) | 1022 MiB |

Per node (live, mean ms): rgb_v2 = Seg 50.5 · Selector 3.2 · DataSource 2.6 · ShellMask 0.12 · ShellHeatmap 0.02.
Ensemble = SegCIR 83.5 (median 65.7) · SegRGB 76.4 (median 66.1) · SelRGB 3.3 · SelCIR 3.2 · DataSource 2.7 · Fuse 0.18
· ShellMask 0.09 · ShellHeatmap 0.02. The ensemble members are slower and noisier (std 30–37 ms) than the same model
run alone (Seg 50 ms, std 5) — back-to-back passes on the shared Thor GPU/memory; the median is the better estimate.

Reading the recording on Thor: **118–120 ms / frame** (SDK 3.6 vs 325 ms on the laptop's SDK 3.5.3).

Path 1 CLI on Thor (1-Sep live, no sync, warm-up included): Seg medians rgb_v2 53.2 ms / cir_v2 58.7 ms; ensemble
SegRGB 78.4 + SegCIR 63.7 ms (means 111 / 111 / 241 ms incl. a 1.35 s first frame).

## Laptop vs Thor (Path 3 steady state, 1-Sep live cu3s)
| pipeline | laptop RTX 4070 (not in turbo) | Thor MAXN | Thor speed-up |
|---|---|---|---|
| rgb_v2 | 79.2 ms (12.6 fps) | 56.5 ms (17.7 fps) | 1.4× |
| cir_v2 | 89.2 ms (11.2 fps) | 62.9 ms (15.9 fps) | 1.4× |
| ens_rgb_cir_mean_v2 | 206.9 ms (4.8 fps) | 169.2 ms (5.9 fps) | 1.2× |
| cu3s read + reflectance | 325 ms | 118 ms | 2.7× |

Takeaways:
- **One RF-DETR-L pass ≈ 50–60 ms on Thor, 75–85 ms on the laptop**; everything else in the pipeline is < 7 ms.
- **The ensemble costs two passes** (~6 fps on Thor). If the live demo needs more than ~6 fps, the next levers are
  fp16 inference (not yet validated for these checkpoints) or running the two members concurrently (on the 4070
  they did not overlap; untested on Thor). `rgb_v2` alone gives ~16–18 fps on Thor.
- Mask sanity on Thor: frame-0 ShellMask px 71196 (rgb_v2) / 71740 (cir_v2) / 70619 (ensemble) vs laptop 71211 / 71786
  / 70747 through the same cu3s path — consistent (last-bit float differences).
- Peak GPU memory ≤ 1.0 GiB for the ensemble — no memory concern on either machine. (Note added in the afternoon:
  these peaks are cumulative — the three pipelines ran in one process and the earlier ones' models were not freed —
  so they are upper bounds, e.g. rgb_v2 alone ≈ 0.5 GiB.)

## Reproduce on Thor
```bash
ssh dev@<thor>
bash /home/dev/walnut_run_thor_profile.sh      # the exact launcher used (copy: SEG_V2_PROFILING_2026-09-24/thor/run_thor.sh)
# core of it:
export CUVIS=/etc/cuvis
/snap/bin/uv run --no-project --no-sources --python ~/.cuvis_runs/0b55f16fc78a1178/.venv/bin/python   --with "cuvis-ai-dataloader[cu3s,coco] @ file:///home/dev/walnut_fo_stack/cuvis-ai-dataloader" --with "cuvis==3.6.0"   python /home/dev/walnut_fo_stack/cuvis-ai/cuvis_ai/configs/pipeline/walnut_seg/profile_seg_cu3s.py --out <dir>   --cu3s /home/dev/walnut_data/2026_09_01/12-16-58/real_world_live_000.cu3s   --cu3s "/home/dev/walnut_data/2026_09_22/14-39-28/kernel_shell_hand_other_d_000+03.cu3s"
```
The seg child env id (`0b55f16fc78a1178`) is the one cuvis.next composed for the seg family; if cuvis.next
recomposes it (new id), point `--python` at the new `~/.cuvis_runs/<id>/.venv/bin/python`.
Parity check on Thor: `…/.venv/bin/python check_seg_pipelines.py --ref ref/seg_ref_masks_laptop.npz --device cuda`.

## Reproduce
Laptop (from `D:\walnuts\walnut_fo_stack\cuvis-ai`):
```bash
WITH='cuvis-ai-dataloader[cu3s,coco] @ file:///D:/walnuts/walnut_fo_stack/cuvis-ai-dataloader'
# Path 1 (CLI), one pipeline:
uv run --no-sync --with "$WITH" restore-pipeline \
  --pipeline-path D:/walnuts/walnut_fo_stack/cuvis-ai/cuvis_ai/configs/pipeline/walnut_seg/walnut_seg_ens_rgb_cir_mean_v2_cuvisnext_cube.yaml \
  --device cuda --data-module cu3s --data-arg cu3s_file_path=D:/Cubert/2026_09_01/12-16-58/real_world_live_000.cu3s \
  --data-arg processing_mode=Reflectance
# Path 3 (synchronised steady state), all three pipelines, both cu3s:
uv run --no-sync --with "$WITH" python cuvis_ai/configs/pipeline/walnut_seg/profile_seg_cu3s.py --out <dir> \
  --cu3s D:/Cubert/2026_09_01/12-16-58/real_world_live_000.cu3s \
  --cu3s "D:/Cubert/2026_09_22/14-39-28/kernel_shell_hand_other_d_000+03.cu3s"
```
(Git Bash: `export MSYS_NO_PATHCONV=1` first.) Raw logs + JSON: `D:\walnuts\SEG_V2_PROFILING_2026-09-24\`.

## Making it faster on Thor — options tested (24 Sep, `walnut_seg/speed_options.py`)
Same 37 real cu3s frames with ground truth, cuvis.next's seg child env on Thor (MAXN), each option applied to the
loaded pipelines without changing the plugin code, CUDA-synchronised timing, first 2 frames skipped. Raw:
`SEG_V2_PROFILING_2026-09-24/thor_speed/` (run 1) and `thor_speed2/` (run 2).

**The current pipelines run in fp32** (rfdetr warns "Model is not optimized for inference"; no precision hparam is
set). Where the time goes today (fp32, rgb_v2, per frame): network 31 ms · rfdetr pre/post-processing inside
`predict` ~4 ms · the node's per-instance CPU mask paste ~13 ms · rest of the pipeline (cube → GPU, selector,
decider) ~7 ms. In the ensemble the members run at threshold 0.05 → ~10 instances each → ~35 ms of CPU pasting per
member.

| option | rgb_v2 ms (fps) | cir_v2 ms | ensemble ms (fps) | accuracy vs fp32 (37 frames) |
|---|---|---|---|---|
| fp32 today | 56.3 (17.8) | 54.7 | 151.0 (6.6) | — |
| fp16 (rfdetr `inference(dtype=fp16)`) | 43.4 (23.0) | 46.0 | 118.5 (8.4) | ensemble IoU 0.944 vs 0.946 |
| bf16 | 43.9 | 48.6 | 128.0 | ensemble 0.949 |
| fp16 + JIT trace (`inference(compile=True)`) | 42.0 | 42.5 | 108.0 | ensemble 0.950 |
| fp16 + GPU input (frame passed to rfdetr on the GPU, no source-image copy) | 40.1 | 40.8 | 110.5 | 0.947 |
| fp32 + GPU input + GPU paste (no precision change) | 47.8 (20.9) | 47.3 | 98.8 (10.1) | 0.944 |
| fp16 + GPU input + GPU paste | 36.9 (27.1) | 38.4 | 79.4 (12.6) | 0.947 |
| **fp16 + JIT + GPU input + GPU paste** | **27.8 (36.0)** | **28.8 (34.7)** | **63.0 (15.9)** | **ensemble 0.947 vs 0.946, agreement 0.998** |
| … + resolution 432 | 35.0 | 34.9 | 73.6 | ensemble agreement 0.968 (worse) |
| … + resolution 384 | 33.3 | 32.2 | 69.0 | ensemble IoU 0.934 (worse) |
| parallel ensemble members (threads + CUDA streams) | — | — | 77.4 vs 79.4 | no gain |
| mixed precision (`torch.autocast` fp16/bf16) | 46.9 | 46.4 | 100.3 | no speed gain over fp32+GPU paste |
| `torch.compile` / CUDA graphs via compile | failed | failed | failed | triton kernel compile error on Thor |

"GPU paste" = the node's per-instance CPU mask paste replaced by one max over all instance masks on the GPU —
mathematically identical output (the only change with zero accuracy risk). Repeat run of the best option: 29.0 /
29.2 / 67.1 ms.

Single-model accuracy on 37 frames is noisy (±0.02 shell IoU between options, up and down) because a few knife-edge
frames flip — the full 163-frame check is below. The ensemble stays within ±0.003 IoU of fp32 for every precision /
input option.

**Accuracy on ALL 163 test frames** (`walnut_seg/validate_speed_options.py`, laptop 4070, same code paths as
Thor; shell IoU mean over frames / FP px; eval-harness metrics):

| pipeline | option | all 163 | 1-Sep live hands | 22-Sep hands | 22-Sep fakes | 15-Sep fake-on-real | fake→shell |
|---|---|---|---|---|---|---|---|
| **B+C ensemble** | fp32 today | **0.963** / 676 | 0.948 / 635 | 0.941 / 684 | 0.927 | 0.939 | 0.1 % |
| | fp32 + GPU input + GPU paste | 0.963 / 676 | 0.949 | 0.940 | 0.927 | 0.939 | 0.1 % |
| | fp16 + GPU input + GPU paste | 0.965 / 685 | 0.951 | 0.940 | 0.927 | 0.958 | 0.1 % |
| | **fp16 + JIT + GPU input + GPU paste** | **0.963** / 683 | 0.939 / 626 | 0.941 / 667 | 0.927 | 0.958 | 0.1 % |
| | bf16 | 0.965 / 679 | 0.949 | 0.959 | **0.904** | 0.957 | 0.1 % |
| rgb_v2 | fp32 → fp16+JIT+GPU | 0.938 → 0.941 | 0.943 → 0.943 | 0.923 → 0.891 | 0.858 → 0.858 | 0.847 → 0.887 | 0.4 % → 0.4 % |
| cir_v2 | fp32 → fp16+JIT+GPU | 0.954 → 0.949 | 0.907 → 0.876 | 0.957 → 0.957 | 0.936 → 0.936 | 0.963 → 0.963 | 0.2 % → 0.2 % |

- **The ensemble is unchanged overall with the fast combo (0.963 = 0.963, fake→shell 0.1 %, FP ≈ same).** [Certain
  for these sets]
- The singles move ±0.03–0.04 on individual scenes, in both directions, with ANY numeric change — the same happens
  with GPU input alone in fp32 (rgb_v2 22-Sep hands 0.923 → 0.892) — i.e. the singles' knife-edge frames, not an fp16
  effect. Overall ±0.005. [Likely]
- bf16 is not better than fp16 and loses 0.023 on the 22-Sep fakes for the ensemble → use fp16.
- Mask agreement with fp32 (mean IoU over the 163 frames): ensemble 0.9993 (fp32 + GPU input + GPU paste) / 0.9967
  (+ fp16) / 0.9947 (+ fp16 + JIT); rgb_v2 0.9955 / 0.9946 / 0.9902; cir_v2 0.9996 / 0.9958 / 0.9937.

**Recommendation:** for Thor use the ensemble with **fp16 + JIT trace + GPU input + GPU paste** (≈ 15 fps instead of
6.6, accuracy unchanged on 163 frames). If more speed is needed, `rgb_v2` with the same tricks runs ≈ 35 fps. To use
them in cuvis.next the tricks have to become node hparams in `RFDETRSegmenter` (the benchmark patches them in at
runtime) — **done in the afternoon, see the next section** (pipelines `…_v2_fast` / `…_v2_exact`).


**Bigger options (not tried, more work):** TensorRT fp16 engine (TensorRT 10.13 is on Thor; the network is 12 of the
28 ms after the tricks, so ~6 ms more at best unless pre/post-processing moves too); a custom GPU post-processing that
skips rfdetr's per-instance full-resolution mask upsampling + numpy round trip (~5–8 ms per model); moving only the 3
selected bands to the GPU instead of the full 61-band cube (~3–5 ms); INT8 (accuracy risk, needs calibration).

## In the plugin (24 Sep afternoon): RFDETRSegmenter speed hparams + `_exact` / `_fast` pipelines
cuvis-ai-rfdetr PR #12 head `0b21fc3` adds four hparams (what each changes: `walnut_seg/README_SEG.md` → Speed):
`fast_paste` (default **True** — instance masks combined with one max on the input's device), `gpu_input` (frame handed
to rfdetr as a CUDA tensor), `precision` (`fp32` | `fp16` | `bf16`) and `jit_trace` (rfdetr `model.inference`).
Packaged per v2 model: default (fp32 + combined masks — the node default, no yaml change), **`_exact`** (+ GPU input) and
**`_fast`** (fp16 + JIT + GPU input); `walnut_seg/build_fast_pipelines.py`.

**Bit-exactness** (`walnut_seg/verify_node_speed_options.py`, `torch.equal` of every score map of all 4 segmenter nodes
(rgb_v2, cir_v2, ensemble RGB + CIR members) and of the fused ensemble mask, vs the per-instance CPU paste):
| machine | frames | combined masks | combined masks + GPU input |
|---|---|---|---|
| laptop RTX 4070 (torch 2.11, rfdetr 1.10.1) | 163 test frames (cached model inputs) | 163/163, max \|diff\| 0 | 163/163, max \|diff\| 0 |
| Thor (child env torch 2.14, rfdetr 1.10.1) | 37 cu3s frames (live 26 + 22-Sep hand 11) | 37/37, 0 | 37/37, 0 |
The `_exact` yamls reproduce the default pipelines' heatmap and mask `torch.equal` on the reference frame. **Correction
to the morning numbers:** the runtime patch's GPU input divided by the Python number 255, which CUDA evaluates as a
multiplication by 1/255 — 1 ULP different from correctly rounded division for 126 of the 256 byte values (checked). That
made "GPU input" look only ~0.999-identical (agreement 0.9993 / 0.9955 / 0.9996 above); the plugin divides by a 0-dim
CUDA tensor like rfdetr 1.10's own uint8 path and is bit-identical.

**Interleaved A/B — laptop** (`walnut_seg/bench_node_options.py`; RTX 4070 Laptop, GPU otherwise idle 16:34–16:40;
37 real cu3s frames (1-Sep live 26 + 22-Sep hand 11) decoded once and held in host memory; per frame the cube is moved
to the GPU once (untimed) and all four variants run `pipeline.forward` back to back in rotating order;
CUDA-synchronised; 3 reps = 111 samples per variant; 3 warm-up frames excluded; raw
`SEG_V2_PROFILING_2026-09-24/laptop_node_options/bench_node_options_laptop.json` + `bench.log`):
| median ms (fps) | rgb_v2 | cir_v2 | ens_rgb_cir_mean_v2 |
|---|---|---|---|
| before today (`fast_paste=0`) | 55.7 (18.0) | 61.6 (16.2) | 120.9 (8.3) |
| default: fp32 + combined masks | 55.3 (18.1) · 1.01× | 59.7 (16.8) · 1.03× | 113.7 (8.8) · 1.06× |
| `_exact`: + GPU input | 50.8 (19.7) · 1.10× | 54.3 (18.4) · 1.14× | 106.5 (9.4) · 1.14× |
| `_fast`: fp16 + JIT + GPU input | **32.6 (30.6) · 1.71×** | **37.0 (27.0) · 1.67×** | **73.4 (13.6) · 1.65×** |
| p90 ms before → `_fast` | 68.5 → 39.5 | 69.8 → 40.6 | 143.9 → 88.1 |
| mask IoU of `_fast` vs before (pooled) | 0.995 | 0.988 | 0.997 |
| first frame (build; `_fast` incl. JIT trace) | 0.8 s → 5.2 s | 0.8 s → 4.4 s | 1.7–1.9 s → 9.1 s |
- The combined masks gain little on the laptop (1–6 %): its x86 CPU pastes fast. On Thor's ARM CPU the per-instance
  paste costs far more (Thor A/B below: 1.13–1.17× singles, 1.41× ensemble). [Certain]
- Back-to-back frames keep the GPU at full clock; with pauses between frames (live camera, the profiler's 380 ms cu3s
  decode) expect higher ms (morning profiler: singles 79–89, ensemble 161–207 ms). The ratios are what carries over.
- **Invalid runs:** the laptop profiler passes started at 13:54 ran while another job shared the 8 GB GPU (VRAM full
  from 14:16) and while `profile_seg_cu3s.py` still kept earlier variants' models on the GPU (no `gc.collect` — fixed);
  archived in `laptop_node_options/_invalid_shared_gpu_1354/` with a README. Do not use them.

**Interleaved A/B — Thor** (same script, Jetson AGX Thor MAXN, cuvis.next's seg child env, GPU otherwise idle,
24 Sep 14:15–14:18; the frames stay in memory on Thor's unified memory; `exact` = the default pipeline with the node
flipped to `gpu_input`, i.e. the `_exact` configuration; raw `SEG_V2_PROFILING_2026-09-24/thor_node_options/`
`bench.log` + `bench_node_options_thor.json`):
| median ms (fps) | rgb_v2 | cir_v2 | ens_rgb_cir_mean_v2 |
|---|---|---|---|
| before today (`fast_paste=0`) | 52.5 (19.0) | 53.9 (18.5) | 145.9 (6.9) |
| default: fp32 + combined masks | 46.6 (21.4) · 1.13× | 46.1 (21.7) · 1.17× | 103.3 (9.7) · 1.41× |
| `_exact`: + GPU input | 42.5 (23.6) · 1.24× | 42.0 (23.8) · 1.29× | 91.5 (10.9) · 1.59× |
| `_fast`: fp16 + JIT + GPU input | **24.2 (41.3) · 2.17×** | **24.4 (41.0) · 2.21×** | **56.0 (17.9) · 2.61×** |
| p90 ms before → `_fast` | 65.8 → 25.8 | 66.1 → 25.8 | 227.1 → 73.3 |
| mask IoU of `_fast` vs before (pooled) | 0.975 | 0.991 | 0.993 |
| first frame (build; `_fast` incl. JIT trace) | 0.7 s → 3.7 s | 0.7–0.9 s → 3.9 s | 1.4 s → 7.6 s |
- On Thor the combined masks alone are worth 1.41× for the ensemble: its members run at threshold 0.05 (many
  instances) and Thor's ARM CPU pastes each one slowly. [Certain for these frames]
- `_fast` single rgb_v2 agrees with fp32 at only 0.975 pooled on Thor (laptop 0.995): a few knife-edge hand frames
  flip with fp16 on this GPU — the known single-model knife-edge behaviour (163-frame check); the ensemble stays at
  0.993. Keep singles for fallback, the ensemble for the demo. [Likely]
- Morning runtime-patch study vs plugin code: `_exact` 48 / 47 / 99 ms (patch) vs 42.5 / 42.0 / 91.5 (plugin),
  `_fast` 28 / 29 / 63 vs 24.2 / 24.4 / 56.0 — the plugin is a bit faster (no `include_source_image` copy in any variant,
  bounded paste chunks).

**cu3s profiler on Thor** (`profile_seg_cu3s.py --variant …`, per-node tables in
`thor_node_options/run1_options.log` / `run2_fast.log`; pipeline ms = sum of node means, live 1-Sep / hand 22-Sep):
| | rgb_v2 | cir_v2 | ensemble |
|---|---|---|---|
| before (old paste) | 55.5 / 75.5 | 59.7 / 65.8 | 165.3 / 194.7 |
| default (combined masks) | 51.8 / 50.1 | 55.6 / 49.9 | 126.4 / 110.4 |
| `_exact` (flip) | 47.8 / 48.2 | 47.7 / 46.4 | 112.3 / 102.3 |
| `_fast` | 28.9 / 29.4 | 28.8 / 28.1 | 78.6 / 74.1 |
Frame-0 ShellMask px identical for before / default / `_exact` (71196 / 71740 / 70619 on the live cu3s) — bit-identical
in the real reader path too.

**Thor pipeline check (25 Sep, `check_seg_pipelines.py --device cuda --ref ref/seg_ref_masks_laptop_cuda.npz`, all 16
pipelines):** the 3 `_exact` pipelines bit-identical to their defaults on Thor; `_fast` at IoU 0.9994–0.9995 vs the
laptop fp32 reference; the other pipelines as on 24 Sep (v2 0.9989–0.9999); only `pca_full` fails (0.886, known).

**Reproduce:**
```bash
# stack env + cu3s reader overlay (laptop); on Thor the child-env python + --no-sources overlay (THOR_RUN_SEG.md)
uv run --no-sync --with "cuvis-ai-dataloader[cu3s,coco] @ file:///D:/walnuts/walnut_fo_stack/cuvis-ai-dataloader" \
  python cuvis_ai/configs/pipeline/walnut_seg/bench_node_options.py --cu3s <live.cu3s> --cu3s <hand.cu3s> --reps 3 --out bench.json
python cuvis_ai/configs/pipeline/walnut_seg/verify_node_speed_options.py --cache <frame_cache.pt>   # or --cu3s ...
python cuvis_ai/configs/pipeline/walnut_seg/build_fast_pipelines.py exact,fast
```

**Laptop cu3s profiler passes (16:38–17:00, GPU idle; `profile_seg_cu3s.py`, per-node tables in
`laptop_node_options/profile_{default,exact,fast}.log`; pipeline ms = sum of node means, live 1-Sep / hand 22-Sep):**
| | rgb_v2 | cir_v2 | ensemble |
|---|---|---|---|
| before (old paste) | 80.4 / 61.9 | 80.0 / 77.0 | 135.9 / 133.0 |
| default (combined masks) | 76.6 / 66.5 | 68.6 / 65.9 | 122.4 / 116.6 |
| `_exact` | 73.2 / 64.1 | 70.5 / 61.6 | 120.4 / 115.3 |
| `_fast` | 37.9 / 33.2 | 42.6 / 32.8 | 80.1 / 80.2 |
Slower than the A/B in absolute terms (the GPU clocks down during the 380 ms cu3s decode between frames), same ranking;
`_fast` ≈ 1.6–2.1× over before. The RF-DETR node remains ≥ 95 % of the pipeline time.

**Inside cuvis.next (laptop, 25 Sep 07:40–07:44, from `%APPDATA%\Cubert\CuvisNext\CuvisNext_2026-09-25.log`, the app's
per-node `MAGIC PROFILING` summary; ensemble v2 variants on `kernel_shell_hand_other_d_000+03.cu3s`, 11 frames):**
| variant | SegRGB median / total (ms) | SegCIR median / total (ms) | per frame after the first | first frame (≈ total − 10 × median) |
|---|---|---|---|---|
| default | 63.3 (1 warm frame) | 41.0 (1 warm frame) | ≈ 104 ms | cold run 1768 + 965 ms (first pipeline of the session, incl. CUDA init) |
| `_exact` | 48.8 / 1170 | 40.4 / 1136 | ≈ 89 ms | ≈ 0.7 + 0.7 s |
| `_fast` | 33.6 / 5174 | 29.3 / 4335 | ≈ 63 ms (≈ 15 fps) | ≈ 4.8 + 4.0 s (JIT trace) |
Other nodes < 5 ms. Consistent with the A/B (laptop `_fast` 73.4 ms, first frame 9.1 s). The user confirmed both
variants run and display heatmap + mask in cuvis.next.

**Thor repeat run (25 Sep 09:47–09:50, MAXN, GPU idle, same A/B script, now with the packaged `_exact` / `_fast` yamls;
raw `SEG_V2_PROFILING_2026-09-24/thor_node_options_0925/`):**
| median ms (fps) | rgb_v2 | cir_v2 | ens_rgb_cir_mean_v2 |
|---|---|---|---|
| before (old paste) | 54.7 (18.3) | 52.7 (19.0) | 146.1 (6.8) |
| default: fp32 + combined masks | 46.6 (21.4) | 45.8 (21.8) | 104.8 (9.5) |
| `_exact` | 42.5 (23.5) | 42.7 (23.4) | 92.1 (10.9) |
| `_fast` | 24.6 (40.6) | 24.8 (40.3) | 55.1 (18.2) |
| first frame `_fast` | 3.8 s | 3.6 s | 7.4 s |
Within ±2 ms of the 24-Sep run for every cell; mask IoU of `_fast` vs before identical (0.975 / 0.991 / 0.993 —
deterministic). [Certain]

**JIT-trace caching (25 Sep, probe on Thor):** saving the traced `_fast` model to disk to skip the ~3 s trace per model
on later starts is **not possible with rfdetr 1.10.1**: `torch.jit.save` fails with `Could not export Python function
call '_DepthwiseConvWithoutCuDNN'` (a Python autograd function in rfdetr's segmentation head; tracing and running it
work, serialising does not). Start-delay alternatives: an fp16-without-JIT variant (no trace, ~1.5 s first frame,
ensemble ≈ 70 ms estimated from the morning study), an in-process model cache (helps only re-loads inside one cuvis.next
session), or a TensorRT engine (one-time build, fast load and run; larger project).

## fp16 without JIT (`_fp16`, 25 Sep) and where a frame's time goes
`walnut_seg_{rgb_v2,cir_v2,ens_rgb_cir_mean_v2}_fp16` = precision fp16 + GPU input + combined masks, **no JIT trace**
(`build_fast_pipelines.py fp16`; reference-frame mask IoU vs fp32 0.9994–0.9996, Thor 0.9986–0.9997 via
`check_seg_pipelines.py`). Thor A/B, 25 Sep 10:05 (all five variants side by side; raw `thor_node_options_0925_fp16/`):
| median ms (fps) | rgb_v2 | cir_v2 | ens_rgb_cir_mean_v2 | first frame rgb / cir / ens |
|---|---|---|---|---|
| before (old paste) | 52.8 (18.9) | 56.5 (17.7) | 142.3 (7.0) | 1.5 / 0.8 / 1.5 s |
| default | 46.6 (21.5) | 46.9 (21.3) | 102.2 (9.8) | 0.8 / 0.8 / 1.4 s |
| `_exact` | 42.3 (23.6) | 42.8 (23.3) | 92.9 (10.8) | 0.8 / 0.7 / 1.7 s |
| **`_fp16`** | **31.5 (31.7)** | **31.8 (31.4)** | **68.9 (14.5)** | **0.9 / 0.8 / 1.5 s** |
| `_fast` | 24.1 (41.5) | 24.2 (41.3) | 53.8 (18.6) | 3.6 / 3.7 / 7.6 s |
Mask IoU vs before: `_fp16` 0.985 / 0.981 / 0.997, `_fast` 0.975 / 0.991 / 0.993. **`_fp16` trades ~7 ms (single)
/ ~15 ms (ensemble) for a start-up without the JIT trace.**

**Time split of one frame on Thor** (`walnut_seg/net_share_probe.py`, rgb_v2, reference frame, median of 40; the
network call inside rfdetr's `predict` wrapped with a CUDA-synchronised timer):
| variant | pipeline | RF-DETR node | network alone | rest of the node (resize/normalise, post-processing, mask copies, paste) | other nodes |
|---|---|---|---|---|---|
| `_fast` | 25.0 ms | 21.4 | **16.5 (66 %)** | 4.9 | 3.6 |
| `_fp16` | 28.7 | 25.2 | 20.1 (70 %) | 5.1 | 3.5 |
| `_exact` | 44.6 | 40.9 | 34.1 (76 %) | 6.8 | 3.7 |

## TensorRT — measured (25 Sep; `walnut_seg/trt_probe.py`, `trt_onnx_export.py`, `trt_parity.py`)
Network only (the walnut RF-DETR-Seg-L rgb_v2 network prepared as rfdetr's `inference()` does, fed ONE real frame:
reference cu3s frame -> pipeline Selector -> rfdetr's predict preprocessing), median of 50 CUDA-synchronised calls;
parity = the outputs through rfdetr's own post-processing + the node's rasterisation, shell-mask IoU vs fp32.
Throwaway uv overlays only (the cuvis.next child envs and the stack venv untouched). Raw: `SEG_V2_PROFILING_2026-09-24/trt_probe/`.

| network, 1 frame | Thor (torch 2.14 cu130) | laptop RTX 4070 (torch 2.11 cu128) | mask IoU vs fp32 (dets) |
|---|---|---|---|
| PyTorch fp32 eager | 28.9 ms | 23.1 ms | 1.0 (5) |
| PyTorch fp16 eager (= `_fp16`) | 17.3 ms | 16.2 ms | 0.9997 (5) |
| PyTorch fp16 + jit.trace (= `_fast`) | 13.8–14.2 ms | 11.5–13.2 ms | 0.9995 (5) |
| **TensorRT fp16 engine** (ONNX → `trtexec`, system TRT 10.13.3) | **7.0 ms** (6.5 with CUDA graph) | — | **0.9994 (5)** |
| TensorRT fp32 engine, TF32 allowed (`trtexec` default) = like-for-like with the deployed fp32 | **17.5 ms** (17.0 with CUDA graph; build 70 s) | — | **0.9998 (5)** — 6 of 71,229 shell px differ |
| TensorRT strict fp32 (`--noTF32`) | 40.0 ms (38.9 CUDA graph; build 23 s) | — | 0.9999 (5) |
| PyTorch fp32 with TF32 switched OFF (strict) | 59.4 ms | — | reference for the two rows below it |
| PyTorch fp32 with TF32 on (= the deployed default, see below) | 29.5 ms | — | 0.9999 vs strict (5) |
| **Torch-TensorRT fp16** (2.14.0 + pip TRT 11.1.0.106; attention kept in PyTorch) | **6.1 ms** | 9–10 ms but **WRONG** (0 detections; 2.11.0 + TRT 10.15.1.29) | 0.9995 (5) on Thor |

- **"fp32" is TF32 in the deployed pipelines:** `import rfdetr` sets `torch.set_float32_matmul_precision("high")`
  process-wide (checked on Thor: `matmul.allow_tf32` False → True on import) and cuDNN convolutions use TF32 by default,
  so every fp32 RF-DETR run (pipelines, cuvis.next, the asai1 evals) has used TF32 tensor-core math. Strict fp32 would
  be 59 ms on Thor. The like-for-like TensorRT build is therefore the default one with TF32 allowed: **17.5 ms vs
  28.9 ms (1.65×), mask IoU 0.9998** — projected Thor pipeline ~31 ms single (vs `_exact` 42 ms), ~70 ms ensemble (vs
  93 ms; about `_fp16` speed, but fp32-level agreement 0.9998 instead of 0.997). [Likely — projection]
- **Torch-TensorRT's attention converter is broken for this model** in both releases tried: the attention's `value`
  operand is a frozen network parameter and the converter hands it to TensorRT unconverted (`add_attention[_v2]():
  incompatible function arguments`); `enable_experimental_decompositions` does not avoid it. Workaround: keep the
  attention ops in PyTorch (`torch_executed_ops` = the four `aten.*scaled_dot_product*` overloads; torch 2.14 exports
  `scaled_dot_product_attention`, torch 2.11 `_scaled_dot_product_efficient_attention`). On Thor that compiles
  (350 s) and is correct; on the laptop (Windows, 2.11) the compiled module outputs garbage, with explicit and weak
  typing alike. [Certain for these versions]
- **Build cost** (once per machine and model): `trtexec` fp16 173 s, fp32 70 s; Torch-TensorRT 350 s. Engine files:
  fp16 73 MB, fp32 135 MB (ONNX 131 MB). Engines are tied to the GPU + TensorRT version.
- **Python 3.11 hurdle is solvable with pip:** `tensorrt-cu13` 11.1.0.106 installs and runs in cuvis.next's Python 3.11
  child env on Thor (via the overlay); the system TensorRT (10.13, Python 3.12 bindings) is not needed for that path.
- **Projected pipeline effect on Thor** (network ~14 → ~6–7 ms per model; everything else unchanged): single
  24 → ~16–17 ms (~60 fps), ensemble 54 → ~38–40 ms (~25–26 fps), vs `_fast`. [Likely — a projection from
  network-only timings; post-processing kept on the GPU (no NumPy round trip) could save a further 1–3 ms per model.]
  Parity above is on ONE frame; the 163-frame check follows.

## TensorRT — 163-frame accuracy check (25 Sep, Thor; `walnut_seg/validate_trt.py`)
Thor child env 35c7… (torch 2.14.0+cu130, rfdetr 1.11.0) + throwaway overlay `tensorrt-cu13==10.15.1.29` + `onnx`
(TensorRT 11.1 has no FP16 builder flag any more, 10.15 does). Engines built with the TensorRT Python API from an
rfdetr-style ONNX export: **fp32** = TensorRT default (TF32 allowed, like-for-like with the deployed fp32), **fp16** =
FP16 builder flag. Builds: rgb fp32 26.5 s / fp16 76.3 s, cir fp32 22.5 s / fp16 73.4 s. Engine call, median over the
163 frames (CUDA-synchronised): **fp32 19.9 ms, fp16 5.8 ms** per model.
Reference = the deployed RFDETRSegmenter forward (fp32 PyTorch) on the same 163 cached frames (`frame_cache.pt`, md5
430d225c…); TensorRT path = the node's preprocessing → engine → rfdetr's post-processing → the node's rasterisation,
node thresholds; ensemble = mean of the member maps ≥ 0.5. Raw: `SEG_V2_PROFILING_2026-09-24/trt_probe/thor/validate_163/`.

Shell IoU (mean over frames) / mean FP px; agreement = mask IoU vs the PyTorch fp32 reference.

| pipeline | backend | all 163 | 1-Sep (106) | 15-Sep (14) | 22-Sep (43) | fake→shell | frames > 20k FP | agreement | frames < 0.9 agreement |
|---|---|---|---|---|---|---|---|---|---|
| rgb_v2 | PyTorch fp32 (deployed) | 0.934 / 1.2k | 0.949 / 754 | 0.847 / 2.4k | 0.934 / 1.7k | 0.4 % | 2 | 1 | 0 |
| rgb_v2 | TensorRT fp32 | 0.941 / 1.2k | 0.953 / 753 | 0.887 / 2.6k | 0.933 / 1.7k | 0.4 % | 2 | 0.9917 | 3 |
| rgb_v2 | TensorRT fp16 | 0.940 / 1.2k | 0.953 / 756 | 0.887 / 2.6k | 0.931 / 1.7k | 0.4 % | 2 | 0.9922 | 3 |
| cir_v2 | PyTorch fp32 (deployed) | 0.954 / 5.5k | 0.948 / 7.7k | 0.963 / 1.5k | 0.965 / 1.5k | 0.2 % | 3 | 1 | 0 |
| cir_v2 | TensorRT fp32 | 0.954 / 5.5k | 0.948 / 7.6k | 0.963 / 1.5k | 0.965 / 1.5k | 0.2 % | 3 | 0.9994 | 0 |
| cir_v2 | TensorRT fp16 | 0.951 / 5.5k | 0.942 / 7.6k | 0.963 / 1.5k | 0.965 / 1.5k | 0.2 % | 3 | 0.9954 | 1 |
| **ens** | PyTorch fp32 (deployed) | 0.964 / 682 | 0.970 / 466 | 0.958 / 878 | 0.954 / 1.2k | 0.1 % | 1 | 1 | 0 |
| **ens** | TensorRT fp32 | 0.965 / 683 | 0.970 / 467 | 0.957 / 885 | 0.959 / 1.2k | 0.1 % | 1 | 0.9978 | 1 |
| **ens** | TensorRT fp16 | 0.965 / 683 | 0.970 / 467 | 0.958 / 876 | 0.959 / 1.2k | 0.1 % | 1 | 0.9974 | 1 |

- **Accuracy-neutral on all 163 frames** [Certain for these frames]: IoU equal or better on every test set for both
  precisions, fake→shell and the FP-heavy frames unchanged. The only drop is cir_v2 fp16 on 1-Sep (0.948 → 0.942, one
  frame); the ensemble does not show it.
- **Where they differ it is one borderline shell crossing the confidence threshold**, and it is the SAME frames that
  flip under PyTorch's own numeric changes (laptop `validate_speed` run, 24 Sep): rgb `real_fake_and_kernel_overlap_000+01_f0010`
  agreement 0.290 under PyTorch fp16+JIT vs 0.291 TensorRT; rgb `kernel_shell_hand_other_d_000+03_f0004` 0.525 under
  a one-ulp input change vs 0.768 TensorRT fp32; cir `real_world_live_000_f0021` 0.465 under PyTorch fp16 vs 0.466
  TensorRT fp16; ens `real_world_live_000_f0022` 0.966 vs 0.963. So the disagreement is threshold sensitivity of those
  instances, not a TensorRT defect. Direction is mixed: TensorRT recovers a shell PyTorch fp32 misses on rgb f0010
  (FN 52k → 11k px) and live f0017 (37k → 1.5k) and on the ensemble for `kernel_shell_fakes_d_000+06_f0001`
  (14.6k → 1.3k); cir fp16 loses one on live f0021 (FN 2.6k → 28k).
- For comparison, ensemble agreement on the laptop run: PyTorch `_fp16` 0.9967, `_fast` 0.9947; TensorRT fp32 0.9978,
  fp16 0.9974.


## TensorRT backend in the plugin (25 Sep evening): `backend: tensorrt` + `_trt_fp32` / `_trt_fp16` pipelines
**cuvis-ai-rfdetr `feat/rfdetr-tensorrt-backend` @ 736605a (local commit, not pushed).** `RFDETRSegmenter` gains
`backend: "torch" | "tensorrt"` and `engine_dir` (default `<checkpoint>.trt/`). With `tensorrt`, `precision` picks the
engine: `fp32` = TensorRT's default build (TF32 allowed = like-for-like with the deployed fp32), `fp16` = FP16 builder
flag. rfdetr's own preprocessing and post-processing run around the engine, so everything after the network (paste,
class filter, outputs) is unchanged. New module `cuvis_ai_rfdetr.trt_engine`:
- ONNX via rfdetr's own `model.export()`, engine build with the TensorRT Python API;
- a runner on its own CUDA stream (on the default stream TensorRT syncs the host on every call; its warning is gone
  with the own stream, numbers identical);
- CLI `python -m cuvis_ai_rfdetr.trt_engine build | build-pipeline <yaml>`.

Engines are per machine: file names carry precision, resolution, GPU and TensorRT version, and a JSON build record with
the checkpoint MD5 is checked at load. Tests: 38 mocked + 2 real-TensorRT (`slow`); clean-clone CI parity 155 passed /
49 skipped, coverage 75.6 %, ruff clean.

**Pipelines** (`build_fast_pipelines.py trt_fp32,trt_fp16`, builds this machine's engines first):
`walnut_seg_{rgb_v2,cir_v2,ens_rgb_cir_mean_v2}_{trt_fp32,trt_fp16}_cuvisnext_cube` = `backend: tensorrt` + `precision`
+ `gpu_input` + fast paste. Engines: `weights/{rgb,cir}_v2_ema.pth.trt/`, fp32 134–136 MB and fp16 69 MB each.
Builds: laptop fp32 44–53 s, fp16 163–172 s; Thor fp32 25–30 s, fp16 76–79 s.

Mask IoU vs default on the reference frame (rgb / cir / ens):

| | laptop RTX 4070 (TRT 10.15.1.29 cu12) | Thor (TRT 10.15.1.29 cu13, child env 35c7) |
|---|---|---|
| `_trt_fp32` | 0.9998 / 0.9997 / 0.9996 | 0.9990 / 0.9996 / 0.9991 |
| `_trt_fp16` | 0.9989 / 0.9985 / 0.9988 | 0.9996 / 0.9994 / 0.9985 |

Full `check_seg_pipelines.py` on Thor (25 pipelines vs the laptop reference): 24 OK. `_exact` is still bit-identical,
every `_trt` pipeline is ≥ 0.998, and only the known `pca_full` fails (0.886, identical to 24 and 25 Sep morning).

**163-frame accuracy through the node** (`walnut_seg/validate_trt_node.py`: the `_trt` pipelines' Seg nodes vs the
default pipelines' Seg nodes on `frame_cache.pt`). Cells: shell IoU all 163 / fake→shell / frames > 20k FP /
agreement with PyTorch fp32.

| pipeline | PyTorch fp32 (deployed) | TRT fp32 | TRT fp16 |
|---|---|---|---|
| rgb_v2, Thor | 0.934 / 0.4 % / 2 / 1 | 0.939 / 0.4 % / 2 / 0.9946 | 0.939 / 0.4 % / 2 / 0.9915 |
| cir_v2, Thor | 0.954 / 0.2 % / 3 / 1 | **0.960** / 0.2 % / 2 / 0.9948 | 0.954 / 0.2 % / 3 / 0.9990 |
| **ens, Thor** | 0.964 / 0.1 % / 1 / 1 | 0.966 / 0.1 % / 1 / 0.9981 | 0.964 / 0.1 % / 1 / **0.9990** |
| rgb_v2, laptop | 0.938 / 0.4 % / 2 / 1 | 0.939 / 0.4 % / 2 / 0.9921 | 0.939 / 0.4 % / 2 / 0.9896 |
| cir_v2, laptop | 0.954 / 0.2 % / 3 / 1 | 0.951 / 0.2 % / 3 / 0.9963 | 0.949 / 0.2 % / 3 / 0.9936 |
| **ens, laptop** | 0.963 / 0.1 % / 1 / 1 | 0.965 / 0.1 % / 1 / 0.9976 | 0.964 / 0.1 % / 1 / 0.9954 |

- **Accuracy-neutral on both machines** [Certain for these 163 frames]. The differing frames are the known borderline
  shells: rgb `…overlap_000+01_f0010`, `…hand_other_d_000+03_f0004`, `shell_only_d_000+01_f0000`, live f0017; cir live
  f0021 / f0010; ens `…fakes_d_000+06_f0001`. Thor cir fp32 drops the large hand false positive on live f0010:
  live-hands FP 28.3k → 17.8k px.
- The Thor ensemble with TRT fp16 agrees with PyTorch fp32 at 0.9990. That is higher than PyTorch `_fp16` (0.9967) or
  `_fast` (0.9947) on the laptop run.

**Timing — node forward** (median over the 163 frames, CUDA-synchronised, whole RFDETRSegmenter incl. pre/post):
Thor PyTorch fp32 41.4–43.1 → TRT fp32 23.8–25.1 → **TRT fp16 9.4–10.6 ms**; laptop 51.4–53.7 → 22.1–22.9 →
**9.8–10.7 ms**. The first forward is about 1 s per model (model build + engine load, no trace).

**Timing — whole pipeline, interleaved A/B** (`bench_node_options.py`, 37 real cu3s frames × 3 reps, all variants side
by side, cube on the GPU → mask + heatmap; median ms, fps in brackets; raw data in
`SEG_V2_PROFILING_2026-09-24/trt_node/{thor,laptop_bench}/`):

| variant | Thor rgb_v2 | Thor cir_v2 | **Thor ens** | laptop rgb_v2 | laptop cir_v2 | **laptop ens** |
|---|---|---|---|---|---|---|
| before (per-instance paste) | 57.2 (17.5) | 54.7 (18.3) | 145.1 (6.9) | 61.4 (16.3) | 58.2 (17.2) | 138.1 (7.2) |
| default | 50.6 (19.8) | 47.3 (21.2) | 99.6 (10.0) | 57.5 (17.4) | 55.2 (18.1) | 118.5 (8.4) |
| `_exact` | 43.0 (23.3) | 42.5 (23.5) | 89.4 (11.2) | 47.3 (21.1) | 48.8 (20.5) | 108.3 (9.2) |
| `_fp16` | 31.8 (31.5) | 31.4 (31.8) | 66.5 (15.0) | 43.5 (23.0) | 45.0 (22.2) | 99.9 (10.0) |
| `_fast` | 24.3 (41.2) | 23.8 (42.0) | 52.6 (19.0) | 31.4 (31.8) | 32.5 (30.7) | 74.4 (13.4) |
| **`_trt_fp32`** | 27.6 (36.2) | 27.3 (36.7) | 57.5 (17.4) | 23.4 (42.8) | 24.3 (41.2) | 49.2 (20.3) |
| **`_trt_fp16`** | **13.3 (75.5)** | **12.9 (77.8)** | **28.4 (35.2)** | **11.8 (85.0)** | **11.7 (85.8)** | **24.9 (40.1)** |
| first frame `_fast` / `_trt_fp32` / `_trt_fp16` | 3.9 / 1.0 / 1.0 s | 3.7 / 1.1 / 1.0 s | 7.6 / 2.0 / 1.9 s | 9.7 / 2.3 / 1.6 s | 7.7 / 1.5 / 1.4 s | 17.2 / 3.0 / 2.8 s |
| ens mask IoU vs before, pooled (`_fast` / `_trt_fp32` / `_trt_fp16`) | | | 0.993 / 0.999 / 0.998 | | | 0.997 / 0.999 / 0.992 |

- **`_trt_fp16` is the fastest tier on both machines** [Certain]. Thor ensemble 28.4 ms (35 fps) = 1.85× `_fast` and
  3.5× default; singles 75–78 fps. The earlier projection (single ~16–17 ms, ens ~38–40 ms) was pessimistic.
- **On Thor `_trt_fp32` is not a speed tier** [Certain]: 27.6 ms single vs `_fast` 24.3, ens 57.5 vs 52.6. It is the
  "fp32 numbers, near-`_fast` speed" option (ens agreement 0.999 vs `_fast` 0.993). On the laptop it beats `_fast`
  (ens 49.2 vs 74.4 ms).
- **Start-up**: 1.0–2.0 s first frame on Thor vs 3.7–7.6 s for `_fast` (no trace). The engine build happens once per
  machine, before the first run.
- Bench-frame caveat: Thor cir_v2 `_trt_fp32` has pooled IoU 0.921 vs before on the 37 bench frames. That is the cir
  single's borderline whole-hand FP instances on the 12-16-58 live recording. The GT check shows this flip goes in the
  good direction (cir live-hands FP 28.3k → 17.8k px), and the ensemble is steady (0.999).
- **Blocker for in-app use** [Certain]: cuvis.next's composed child envs have no `tensorrt`, so the `_trt` pipelines
  fail on the first frame there with "needs the TensorRT Python package". Running them in-app needs `tensorrt`
  provisioned in the seg child env: `tensorrt-cu13==10.15.1.29` on Thor, `tensorrt-cu12==10.15.1.29` on the laptop.
  It must be TensorRT 10, not 11 (11 has no FP16 builder flag), and the same version the engines were built with,
  which is in their file names. All numbers here are from throwaway overlays; the stack venv and the child envs are
  untouched.
