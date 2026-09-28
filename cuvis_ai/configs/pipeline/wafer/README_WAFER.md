# Wafer thickness pipelines (in the stack, for Thor)

Per-pixel SiO₂-on-Si film **thickness map** from ULTRIS XMR cubes via thin-film interference peak-counting.
Added to this stack so cuvis.next can pick it alongside walnut seg + FO. Plugin: `cuvis-ai-wafer-thickness` 0.3.1
(stateless physics nodes — **no trained weights**), manifest `configs/plugins/wafer_thickness.yaml`.

## Pipelines (cube mode)
| yaml | regime | terminal |
|---|---|---|
| `wafer_thickness_pipeline_cuvisnext_cube` | **1000 nm** (orders k6/5/4) | `WaferThickness.outputs.thickness` (nm map) |
| `wafer_thickness_pipeline_500nm_cuvisnext_cube` | **500 nm** (orders k3/k2) | same |

Nodes: `CU3SDataNode → WaferSegmentation → WaferThickness → NumpyFeatureWriterNode`. Plugins:
`[wafer_thickness, cuvis_ai_builtin]`.

## Verified on this stack (core 0.17.2, cuvis-ai/.venv)
Build + save (.pt) + reload OK. Forward on the 1000 nm bench cube → **median 1013.6 nm / 427,564 px** — bit-exact
to the research benchmark. (torch backend, GPU.)

## ⚠️ Recipe-based — pick the regime that matches the wafer
The `orders` windows **assume** the thickness regime; they are not auto-detected:
- **Harmonic blind spot:** a 1000 nm wafer run under the 500 nm pipeline reads a *confident* ~505 nm (verified here) —
  wrong, and `uncertainty` won't catch a 2×/3× error. Use the pipeline matching the wafer's nominal thickness.
- Capture range ≈ ±5–6 % per recipe; floor ≈ 300–350 nm (needs ≥2 maxima in 430–910 nm; 100/200 nm N/A),
  ceiling ≈ 2–3 µm. Non-harmonic mismatch IS caught (uncertainty ≫ baseline).

## cuvis.next (same recipe as seg/FO — THOR_DEPLOY_NOTES §9)
- Settings Directory = `<clone>/cuvis_ai/configs`. Plugin `wafer_thickness` resolves via
  `configs/plugins/wafer_thickness.yaml` (absolute local path → the `cuvis-ai-wafer-thickness` checkout).
- Per pipeline set BOTH `Pipeline = …/wafer/<name>.yaml` AND `Weights = …/wafer/<name>.pt`.
- Terminal is a float nm map (with NaN/0 background). Open question from the Aug handover (Jira ALL-6078): whether
  cuvis.next's overlay renders a float nm map with NaNs — confirm on the live view.

## Thor
- Deps: `scipy` + `scikit-image` (Otsu / connected-components / median) + torch (default `backend=torch`).
- The wafer plugin **already scoped its cu128 torch pin to a `cuda` dependency group**, so on Jetson
  `uv sync --no-default-groups` uses the aarch64 torch (or install with `--no-sources` like the other plugins).
- `restamp_thor.py --root <thor-root>` rewrites `wafer_thickness.yaml`'s `path:` (it globs `plugins/*.yaml`).
- Wafer test data (cu3s) is staged separately at `Z:\anish\wafer_thickness_cuvisnext\data\` (+ `extra_samples\`)
  from the Aug handover — copy what you want to test to Thor.
