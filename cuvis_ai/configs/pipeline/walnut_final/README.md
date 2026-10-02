# walnut_final: the two fair pipelines side by side (2 Oct 2026, for testing)

| file | what | thresholds (gate / mask) |
|---|---|---|
| `walnut_final_original_cuvisnext_cube.yaml` + `.pt` | the deployed fair pipeline, an exact copy of `walnut_combo/composite/walnut_combo_gated_fp16trt_trt16_composite_cuvisnext_cube` | 1.35 / 1.35 |
| `walnut_final_refit_1oct_cuvisnext_cube.yaml` + `.pt` | the same pipeline with the FO banks refitted on the 1-Oct stand recordings, a copy of `walnut_refit_1oct/walnut_combo/composite/...` | 1.1512 / 1.2732 (mask margin 1.15) |

Both show FO + shells in one view (`Composite.mask`); `FOMask.decisions` and `gate.scores` stay selectable. In
CuvisNEXT set Pipeline to the yaml and Weights to the `.pt` with the same name.

**Calibrate each one separately, from a clean turn with THAT pipeline loaded** (stack root on Thor,
`/home/dev/walnut_fo_stack`). The two pipelines score on different scales.

Original:
```
python3 calibrate_from_log.py --precision fp16trt --last 60 \
  --yaml cuvis-ai/cuvis_ai/configs/pipeline/walnut_final/walnut_final_original_cuvisnext_cube.yaml
```

Refit:
```
python3 calibrate_from_log.py --precision fp16trt --last 60 --mask-margin 1.15 \
  --yaml cuvis-ai/cuvis_ai/configs/pipeline/walnut_final/walnut_final_refit_1oct_cuvisnext_cube.yaml
```

Add `--write` to write the values, then load the pipeline again.

These are copies: the default calibration (`calibrate_from_log.py --precision fp16trt` without `--yaml`) writes
`walnut_fo/` and `walnut_combo/`, not this folder. Calibrating this folder does not change those either.
Background: `TASKS.md` and `THOR_DEPLOY_NOTES.md` §30-31.
