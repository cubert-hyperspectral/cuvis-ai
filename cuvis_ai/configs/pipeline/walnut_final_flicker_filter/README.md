# walnut_final_flicker_filter: the four final pipelines with a flicker filter on the FO mask (2 Oct 2026, for testing)

Every pipeline here is a copy of the one with the same name in `walnut_final/`, with one node added:
`FOFlickerFilter` after `gate.decisions`. **An FO mark shows only where the frame before also had an FO mark
within 40 px.**
- A false blob that appears for one frame never shows.
- A real FO stays in view while the turntable moves it (up to about 30 px per frame), so it shows from its
  second frame on: 1/8 s later at 8 fps.

| file | copy of | thresholds (gate / mask) |
|---|---|---|
| `walnut_final_original_flickerfilter_cuvisnext_cube.yaml` + `.pt` | `walnut_final_original` | 1.35 / 1.35 |
| `walnut_final_refit_1oct_flickerfilter_cuvisnext_cube.yaml` + `.pt` | `walnut_final_refit_1oct` | 1.1512 / 1.2732 |
| `walnut_final_original_shellaware_flickerfilter_cuvisnext_cube.yaml` + `.pt` | `walnut_final_original_shellaware` | 1.35 / 1.35 |
| `walnut_final_refit_1oct_shellaware_flickerfilter_cuvisnext_cube.yaml` + `.pt` | `walnut_final_refit_1oct_shellaware` | 1.1512 / 1.2732 |

**What changes:**
- Changed: the FO mask, both `FOMask.decisions` and the FO colour (255) in `Composite.mask`.
- Unchanged:
  - the alarm (`gate.passed`, the frame score);
  - the FO heatmap (`gate.scores`), which still flickers;
  - the shells.

In CuvisNEXT set Pipeline to the yaml and Weights to the `.pt` with the same name. **Restart CuvisNEXT once** after
this update: the filter is a new node of the patchcore plugin (`MaskPersistence`), so a running session does not
know it yet.

**Measured offline** on your labelled 1-Oct afternoon frames (each base calibrated as at the stand):

| | false blobs per FO frame | FO frames with a false blob | fake shells | stones | alu | rubber | stems |
|---|---|---|---|---|---|---|---|
| original | 0.47 | 34 % | 153/182 | 192/192 | 81/81 | 45/47 | 57/73 |
| original + filter | 0.25 | 20 % | 150 | 188 | 79 | 44 | 50 |
| refit | 0.17 | 15 % | 106 | 192 | 81 | 47 | 60 |
| refit + filter | 0.07 | 8 % | 102 | 188 | 79 | 46 | 56 |

The FO columns count the frames on which each object is shown, out of the labelled frames.
- Most FO frames lost are an object's first frame in view.
- Stems lose more, 4-7 frames, because they flicker themselves.

**Limits:**
- A false blob that stays for two frames or more still shows. This includes a fixed false spot at the image
  edge, which would also show in the same place on every frame.
- If the turntable moves an object more than 40 px between two frames, that object no longer shows. That is
  about 320 px/s at 8 fps.
- After loading, the first frame shows no FO.

**Calibration** is the same as for the source pipeline, because the gate sits before the filter: same log lines,
same values. You can copy the source's thresholds, or calibrate from a run of THIS pipeline (stack root):
```
python3 calibrate_from_log.py --precision fp16trt --last 60 \
  --yaml cuvis-ai/cuvis_ai/configs/pipeline/walnut_final_flicker_filter/walnut_final_original_flickerfilter_cuvisnext_cube.yaml
```
For the refit pipelines, add `--mask-margin 1.15`. Add `--write` to write the values, then load the pipeline again.

Built with `make_flicker_filter.py --src walnut_final/<name>_cuvisnext_cube.yaml --dst
walnut_final_flicker_filter/<name>_flickerfilter_cuvisnext_cube.yaml` (radius 40 px). Background: `TASKS.md`,
`THOR_DEPLOY_NOTES.md` §33.
