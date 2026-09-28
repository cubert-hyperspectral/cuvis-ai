# SEG parity on the walnut_fo_stack (core 0.17.2)

**Question this answers:** the seg models + `cuvis_ai_rfdetr` deploy nodes were built/validated on the
cuvis-ai 0.14–0.15-era research env; the stack is core **0.17.2** (`cuvis-ai\.venv`, cuvis.next's server base) /
0.17.3 (top-level `.venv`). Does the core-0.17 bump change forward behavior? **No.**

## How
`repro/build_seg_stack.py` (in `D:\walnuts\walnut_deploy_v2\repro\`) built all 7 seg pipelines into
`walnut_seg/` and forwarded each on the fixed reference cube
`D:/walnuts/deploy_wise08/real_world_live_000_f0000.npy` (61-band, 430+8·k nm), counting shell pixels
(`scores ≥ 0.5`) at the terminal node. Run in `cuvis-ai\.venv` (Py 3.13, core 0.17.2) after installing the seg
deps (`rfdetr` + `cuvis-ai-rfdetr` editable). The `plugins: [cuvis_ai_builtin, rfdetr_seg]` list resolves the seg
nodes through the new `rfdetr_seg` manifest — i.e. the same catalog-resolution path cuvis.next uses.

## Result — shell px on the reference cube
| pipeline | stack (core 0.17.2) | research-env reference | match |
|---|---|---|---|
| `walnut_seg_rgb_full` | **72424** | 72424 | ✅ bit-identical |
| `walnut_seg_cir_full` | 63257 | 63257 | ✅ bit-identical |
| `walnut_seg_pca_full` | 62712 | 62712 | ✅ bit-identical |
| `walnut_seg_ens_rgb_cir_mean` | 71899 | 71899 | ✅ bit-identical |
| `walnut_seg_int_rgb_cir` | 62394 | 62394 | ✅ bit-identical |
| `walnut_seg_ens_sam_t13` | 69100 | — (new this session) | derived from ens (71899) − T=13° gate |
| `walnut_seg_ens_sam_t11` | 66232 | — (new this session) | derived from ens (71899) − T=11° gate |

Reference values recorded in `walnut_deploy_v2/REPRODUCE.md` §"smoke" (RGB 72424 / CIR 63257 / PCA 62712 /
ens 71899 / int 62394). The 5 pre-existing pipelines are **bit-identical** across the core version bump. The 2
SAM variants have no research-env reference (the `SamShellGate` node was authored this session); their gated px
match the SAM T-sweep validation (T=13 removes ~4% of ensemble shell px here, T=11 ~8%, consistent with the
15-Sep fake→shell reduction T13 ~10% / T11 ~6% and near-zero erosion on old product at T=13).

## Interpretation
The RF-DETR segmenter forward, the false-RGB/PCA selectors, `ScoreFusion(mean)`, `ScoreIntersection`, and
`SamShellGate` all reproduce exactly on core 0.17.2 — the version cuvis.next actually loads. This mirrors the FO
side's finding (`STACK_PARITY.md`: 72/72 anomaly maps bit-identical 0.15 ↔ 0.17.3). **Consolidation is safe:** the
seg pipelines behave on the stack exactly as they did in the research env. Both `.pt` and yaml-only load paths were
exercised by the build (`build_from_config` → `save_to_file` → `forward`).

## Not yet covered (user-driven, on Thor)
- Live cu3s through cuvis.next (this parity used a `.npy` cube — no SDK). cuvis.next lists + loads the pipelines
  and yields a shell mask: to be confirmed by the user on the live rig.
- aarch64 (Jetson) rebuild — numeric parity on Thor's torch build is expected but unverified; re-run this same
  script there after the path re-stamp as the Thor smoke.
