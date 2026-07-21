# Changelog

## [Unreleased]

## 0.11.1 - 2026-07-21

- Bumped plugin manifest pins: adaclip v0.1.5 -> v0.2.0, augment v0.3.2 -> v0.3.3, cuvis_ai_dataloader v0.2.0 -> v0.4.0, cuvis_ai_inspecscrap v0.2.1 -> v0.2.2, deepeiou v0.2.0 -> v0.2.1, dinomaly v0.2.0 -> v0.4.1, trackeval v0.1.3 -> v0.1.4, ultralytics v0.1.3 -> v0.1.4.

- **Compacted the docs node-catalog filter into a one-row toolbar.** The `/catalogs/nodes/` filter
  bar (previously a sticky block with three always-open chip rows) is now a single sticky row:
  search field, foldable Category/Tags/Source buttons with active-count badges, a prerendered item
  count, and a state-aware Clear. Active facet filters render as a removable chip strip under the
  bar; zero matches show an empty state with a recovery action; facet panels cap at `min(40vh, 16rem)`
  with internal scroll. Tag filtering now combines **OR within the facet** (AND across facets).
  Hardening while touching the code: a malformed URL hash (`#q=%`) no longer crashes the filter,
  stale hash values render as removable raw-value chips, `init()` is guarded against double-binding
  under instant navigation, and the toolbar is screen-reader friendly (labelled controls,
  `aria-pressed`/`aria-expanded`, debounced `aria-live` result announcements). Covered by a new
  generator contract test and a Playwright E2E suite (`tests/docs/test_node_catalog_e2e.py`,
  `slow`-marked; `playwright`/`pytest-playwright` added to the dev group).

- **`AnomalyDataNode` gained a `class_mask` input and output port** so downstream per-class metrics can read the multi-class ground truth as a pipeline port instead of reloading it from disk. The new optional `class_mask` input binds to the data module's separate `class_mask` batch key (per-pixel category id) and is re-emitted channel-last (`[B, H, W, 1]` int32); the output is no longer derived from the binary `mask`, which had collapsed every class to `{0, 1}`. The input port is a generic tensor so the module's `uint8` category ids bind without a strict-dtype rejection.

- **Fixed the built-in plugin manifest's local `path` so the gRPC child-env compose can install it.** `configs/plugins/cuvis_ai_builtin.yaml` set `path: "../.."`, which resolves to the `cuvis_ai` package directory (no `pyproject.toml`), so composing a child env from a source checkout could not resolve the `cuvis-ai` project. Corrected to `../../..` (the repo root); any gRPC-from-source pipeline listing `cuvis_ai_builtin` now composes.

- **Flattened the trainrun/training configs for the folded `TrainingConfig` (needs `cuvis-ai-core>=0.11.0` / `cuvis-ai-schemas>=0.8.0`).** The nested `trainer:` block is gone: its `pytorch_lightning.Trainer` fields now sit directly under `training:` in every `configs/trainrun/*.yaml` and in `configs/training/default.yaml`, and the dead `training.batch_size` / `training.num_workers` keys are dropped. Hydra overrides change from `training.trainer.<field>=…` to `training.<field>=…`. Added `tests/configs/test_trainrun_configs_valid.py`, which validates every shipped training block against the flat schema.

- **Unified the SAM3 RGB input port on `rgb_image`.** Renamed `rgb_frame` -> `rgb_image` in
  `configs/plugins/sam3.yaml` and every `configs/pipeline/sam3/*` connection target and view-preset
  description, matching the sam3 nodes' renamed port and the `rgb_image` name every other RGB
  producer/consumer in the library already uses.
- **Exposed the SAM3 tracker thresholds on every propagation preset.** The `configs/pipeline/sam3/*`
  propagation presets now list `score_threshold_detection`, `new_det_thresh`, `det_nms_thresh`,
  `overlap_suppress_thresh`, and `max_tracker_states` in the tracker node's `hparams`, so they render
  as editable knobs in the host pipeline picker. Mask- and bbox-seeded propagation default
  `new_det_thresh` to `0.95` (was `0.7`) so a seeded track is not swamped by newly detected objects;
  text propagation keeps the detection-driven `0.7`.
- **`ToVideoNode` now finalizes its video on `cleanup()`.** The sink flushes the ffmpeg trailer when
  the hosting pipeline is torn down (session close / pipeline replacement / run stop), so it produces a
  playable file in a gRPC/session context that has no explicit driver `close()` call. `close()` stays
  idempotent, so an explicit driver `close()` is still fine.
- **Added `PointPrompt`, an interactive point-prompt source node.** Emits a scheduled per-frame list
  of `{element_id, x, y, type}` dicts (`type` in `positive` / `negative` / `neutral`) on a configured
  `prompt_frame_id`, and an empty list on every other frame. Accepts `(x, y[, type])` tuples or
  dicts, validates the type, and defaults to `positive`. Registered in
  `configs/plugins/cuvis_ai_builtin.yaml`. Its output dict shape matches `SAM3PointExpansion`'s
  `points` input, so it drives point expansion in a scripted (non-interactive) pipeline.
- **Added the SAM3 single-frame point-expansion use case.** Two pipeline configs
  (`configs/pipeline/sam3/sam3_point_expansion.yaml`, cu3s-sourced, and `…_video.yaml`, video-frame
  sourced) plus the `notebooks/use_cases/object_selection_point_expansion.ipynb` walkthrough.
- **Pinned the sam3 plugin to `v0.2.1`.** `configs/plugins/sam3.yaml` uses a `repo:` + `tag: v0.2.1`
  pin (restored from the temporary local `path:` checkout), the first tagged sam3 release to ship
  `SAM3PointExpansion` and the `rgb_frame` -> `rgb_image` port rename.
- **Registered the rtsam2 plugin.** Added `configs/plugins/rtsam2.yaml` pinned to
  `cuvis-ai-rtsam2` `v0.2.0`, exposing the `RTSAM2BboxPropagation` and `RTSAM2MaskPropagation`
  streaming tracker nodes (SAM2.1 / EfficientTAM camera predictors; prompt once on the first
  frame, then track frame by frame).
- **Added the rtsam2 mask-propagation pipelines.** The
  `configs/pipeline/rtsam2/rtsam2_mask_propagation{,_view}.yaml` pair mirrors the sam3
  mask-propagation set: a cu3s source (or the host, in the `_view` preset) feeds
  `RTSAM2MaskPropagation`, seeded at runtime through the tracker's optional `mask` port (the
  builtin `MaskPrompt` node is the drop-in producer), with tracked masks written by
  `CocoTrackMaskWriter`.
- **Dropped the abstract `SAM3TrackerInference` entry from `configs/plugins/sam3.yaml`.** Base
  classes are not instantiable from pipelines, so the manifest now lists only concrete nodes.

## 0.11.0 - 2026-07-14

- **Docs: the node catalog now lists plugin capabilities.** The Catalogs â†’ Nodes generator reads
  the plugin manifests (`configs/plugins/*.yaml`) instead of the never-created
  `docs/data/plugin_sources.yaml` that left the published catalog at "0 from plugins". Plugin nodes
  render manifest-driven I/O port tables, data modules get their own rows and pill, and the filter
  gains a Source facet (built-in / plugin / data module). An unparseable manifest or an empty
  plugin collection now fails the docs build instead of silently shipping an empty list.
- **Added a Savitzky-Golay / pretreatment node family.** Seven composable `cube -> cube` spectral
  pretreatments under `cuvis_ai/node/pretreatments/`: `SavitzkyGolay` (frozen-kernel `conv1d`,
  validated against `scipy.signal.savgol_filter`), `ContinuumRemoval` (convex-hull), `SpectralDerivative`,
  `SNVCorrection`, `Logarithm`, and the fit-required `MeanCenter` / `UnitVarianceScaling` (streaming
  Welford). All chain into any existing cube consumer.
- **Added 11 vegetation-index selectors.** `EVI`, `EVI2`, `SAVI`, `MSAVI`, `NDWI`, `NBR`, `GNDVI`,
  `NDRE`, `CIRedEdge`, `MCARI`, and `PRI` selectors, reusing `NDVISelector`'s `index_image` /
  `rgb_image` colormap machinery.
- **Added spectral unmixing nodes.** `NNLSUnmixing` (stateless projected-gradient NNLS, validated
  against `scipy.optimize.nnls`) and the fit-required `NMFUnmixing` (blind, learns endmembers via
  sklearn NMF then solves per-pixel abundances in pure torch).
- **Added clustering nodes.** `KMeansClusterer` and `GaussianMixtureClusterer` fit with scikit-learn
  during statistical initialization, freeze the result as torch buffers, and run a pure-torch
  `forward` (nearest-centroid / closed-form Gaussian posterior).
- **Added a one-class SVM detector.** `OneClassSVMDetector` fits `sklearn.svm.OneClassSVM`, persists
  the support vectors and resolved gamma, and evaluates the RBF decision function in chunked pure
  torch (emits `scores` + `decisions`).
- **Added a shape-morphology descriptor.** `ShapeMorphology` derives per-object area / centroid /
  axes / eccentricity / orientation in torch (`scatter_reduce` + closed-form covariance), reusing a
  shared OpenCV connected-components helper factored out of `MaskRobustifier`.
- **Added saturated-pixel, multi-range slicer, and intensity-threshold nodes.**
  `SaturatedPixelDetector` (`scores` + `decisions`), `MultiRangeSlicer` (`torch.bucketize` into a
  `class_mask`), and `IntensityThresholdSegmenter` (`cube` -> binary `mask`).
- **Added object-level inspection nodes.** `BlobDetector` (brightness reduction + Otsu/quantile/fixed
  threshold + morphological clean + the shared OpenCV connected-components helper, with an area
  filter and `keep_largest` count-pinning, emitting a blob label map plus bboxes / centroids /
  count), `SignaturesToReferences` (per-object signatures `[1, N, C]` -> Spectral Angle Mapper
  references `[N, 1, 1, C]`), and `MajorityVoteByBlob` (collapse a noisy per-pixel label map to one
  majority label per blob). `SpectralSignatureExtractor` is now also exported from `cuvis_ai.node`.
- **Added image-assembly nodes.** `ImageConcatenator` (variadic fan-in of RGB frames into one
  side-by-side or stacked strip, padding to a common size with a background colour) and `PngWriter`
  (a sink that writes RGB frames to PNG via `torchvision.io.write_png`), so a result image can be
  composed and written entirely inside a pipeline.
- **Clustering nodes can fit on a foreground mask.** `KMeansClusterer` and `GaussianMixtureClusterer`
  take an optional `mask` input; when connected, the statistical fit uses only pixels where the mask
  is non-zero while inference still labels every pixel. Other statistical-fit nodes are unaffected.
- **Added `scipy` as a direct dependency floor** (`>=1.17.1`, tracking the lock) for the
  Savitzky-Golay coefficient build.
- **Added two patch-inference nodes.** `PatchSampler` extracts labelled center-pixel patches from a
  cube plus an integer target map (for training a classifier); `ClassMapAccumulator` (a sink)
  scatters per-patch predictions back into per-frame class maps. They are separate nodes, not a
  directly wired pair: `ClassMapAccumulator` consumes a `frame_id`/`y`/`x`/`height`/`width`
  provenance contract supplied by a patch-tiler data module (not emitted by `PatchSampler`), runs as
  `reset()` -> `forward()*` -> `close()`, and retains one map per frame until `reset()` (finished
  maps read from `class_maps`).
- **`TitleOverlay` gained a per-frame caption port.** An optional `caption` input (a `list[str]`,
  one entry per batch element) lets a DataModule title each frame independently; it falls back to
  the `text=` forward argument and then the constructor default. Captions render through the
  pure-torch `draw_text` bitmap font.
- **Added a metal-scrap classification cookbook notebook.**
  `notebooks/use_cases/metal_scrap_classification.ipynb` rebuilds the Gursch et al. 2026 SWIR
  steel-scrap classifier as a Cuvis.AI pipeline (per-band standardizer, 3D-CNN, weighted
  cross-entropy and segmentation metrics), derives the inverse-frequency class weights in the
  statistical-training phase, runs dense per-pixel inference, and adds a mask-cleanup stage. It runs
  on Colab: the dataset is provisioned from Zenodo (DOI 10.5281/zenodo.17076238) and training is
  gated on a CUDA GPU.

## 0.10.2 - 2026-07-13

- `AnomalyDetectionMetrics.average_precision` is now epoch-pooled through the trainer's native
  metric-object logging: the node exposes the live `BinaryAveragePrecision` via `pooled_metrics()`
  (replacing `compute_epoch_metrics()`), and the trainer logs it once per epoch with `on_epoch=True`,
  so the reported AP is the exact pooled value rather than the per-batch running value. Floors
  `cuvis-ai-core` to `>=0.10.1` (the release carrying the native `pooled_metrics()` logging).

## 0.10.1 - 2026-06-24

- **Refreshed plugin manifest pins to the latest releases.** Bumped the `configs/plugins/` tags to
  the published standards-adoption releases: adaclip `v0.1.5`, dinomaly `v0.2.0`, deepeiou `v0.2.0`,
  trackeval `v0.1.3`, ultralytics `v0.1.3`, and the `cuvis_ai_dataloader` data-module plugin
  `v0.2.0`. These are tag-only bumps; the releases carried no node or port changes, so each
  manifest's `capabilities:` block is unchanged.
- **Pinned sam3 to its first co-installable release.** `configs/plugins/sam3.yaml` swaps from the
  local dev `path:` to `repo:` + `tag: v0.1.7`; v0.1.7 relaxes `setuptools<83`, resolving the
  `setuptools>=81` conflict that had forced the local checkout.
- **Registered the augment plugin.** Added `configs/plugins/augment.yaml` (capabilities format,
  `tag: v0.3.1`) exposing `AugmentationCompose` for training-time cube/mask augmentation, plus a
  manifest-sync test.
- **Added a no-local-sources CI gate.** `.github/workflows/no-local-sources.yml` fails if
  `pyproject.toml` declares a local `[tool.uv.sources]` path entry, so a machine-specific editable
  path can never ship in a release.
- **Added a plugin-pin auto-bump workflow.** `.github/workflows/plugin_pin_bump.yml` +
  `scripts/bump_plugin_pins.py` open a PR whenever a pinned plugin publishes a newer release. The
  bump is tag-only, so it also compares the plugin's declared node set at the new tag against the
  manifest's `capabilities` and flags the PR (title + `needs-capabilities-review` label) when the
  release declares a node the manifest is missing (or the node set can't be verified), prompting a
  manual capabilities regen. The per-plugin
  manifest-sync tests now assert the pinned tag's *shape* (a well-formed `vX.Y.Z`) instead of a
  frozen value, so a routine refresh only touches YAML. Also fixed
  `scripts/fetch_plugin_pyprojects.py` to read the one-file manifest format (it had silently skipped
  every manifest in the registry-compat audit).

## 0.10.0 - 2026-06-24

- **`FixedWavelengthSelector` generalized to n-channel output.** `OUTPUT_SPECS["rgb_image"]` is
  relaxed to `(-1, -1, -1, -1)` and `target_wavelengths` accepts any tuple of length `>= 1`, so the
  node can emit any number of bands (e.g. a 6-channel VIS+SWIR stack) instead of exactly three. The
  3-channel default and its running/gamma normalization path are unchanged; `STATISTICAL` /
  `RUNNING` normalization stays 3-channel only and raises for `n != 3`, while `per_frame` works for
  any `n`. The contract stays tight on `ChannelSelectorBase` (the 11 fixed-3-channel selectors keep
  their `(-1, -1, -1, 3)` validation); only `FixedWavelengthSelector` overrides it.
- **Added `PercentileNormalizer` and `DisplayNormalizer`.** Per-channel percentile / min-max
  normalization for any channel count (`per_frame`, `running`, or `statistical` modes), plus a thin
  stateless sRGB gamma step, factored out of `ChannelSelectorBase` so a selector can be a pure
  band-picker and normalization composes downstream. `NormMode` moves to `normalization.py` and is
  re-exported from `channel_selector` for backward compatibility. Both nodes are registered in the
  built-in catalog.

## 0.9.0 - 2026-06-23

- **Trainrun configs reference their pipeline by path.** Following the schemas/core change to a `pipeline:` reference, the bundled trainrun configs no longer inline or Hydra-compose a pipeline: the twelve `@pipeline`-group configs drop that `defaults` entry for a top-level `pipeline: ../pipeline/<group>/<name>.yaml` reference, and the two resolved snapshot configs (`drcnn_adaclip_trainrun`, `adaclip_cir_false_color_optimal_threshold`) extract their inline pipeline to a `<name>_pipeline.yaml` sibling and reference it. The migration-equivalence test now also asserts every string `pipeline` reference resolves to a loadable `PipelineConfig`. Script-driven configs that use `pipeline:` as a parameter-override mapping are unchanged.
- **Plugin registration is import-only via `register_plugin(path)`.** Core dropped the
  in-process clone / install plugin loader and collapsed registration to a single file-path
  front door, so call sites in the use-case notebooks, `scripts/render_pipelines.py`, the plugin
  contract / runtime-smoke tests, and the docs move to `registry.register_plugin(<manifest.yaml>)`
  (in-memory manifests register via `register_plugins_installed`).
  Plugins must be provisioned into the environment first (see the new `provision` CLI in core); the
  runtime-smoke test `importorskip`s the plugin package so it skips cleanly when unprovisioned.
  Floors `cuvis-ai-core>=0.10.0` (renamed plugin API) and `cuvis-ai-schemas[full]>=0.7.0`.
- **Dropped the `cuvis` SDK (and `cuvis-il`, `ftfy`) from base dependencies.** The SDK now lives only
  in the `cuvis-ai-dataloader` plugin behind its `[cu3s]` extra. Builtin/RGB pipelines no longer pull `cuvis` / `cuvis-il`,
  closing the Windows `cuvis-il` no-`win_amd64`-wheel gap. The node library imports neither `cuvis`
  nor `ftfy` (grep-verified); `rle.py` consumers (`occlusion`, `json_file`, `prompts`) are unchanged.
- **Pinned the `cuvis_ai_dataloader` plugin manifest to the published `v0.1.0` tag.**
  `configs/plugins/cuvis_ai_dataloader.yaml` moves from a local dev `path:` to `repo:` + `tag: v0.1.0`,
  matching the sibling plugin manifests now that the data plugin is released.
- **Migrated to the module-agnostic `DataConfig`.** The test data fixture and the
  `configs/data/*.yaml` + `configs/trainrun/*.yaml` data blocks move from the flat cu3s shape to
  `{data_module, splits, params}`. cu3s loading imports moved from `cuvis_ai_core.data.datasets` to
  `cuvis_ai_dataloader.data` (the `SingleCu3sDataModule` / `SingleCu3sDataset` names survive as
  back-compat aliases, so call sites change only the import path). Docs/notebooks repointed.
- **`apply_trainrun_config` forwards the data-module name on `LoadPipeline`.** The gRPC example
  helper now sets `LoadPipelineRequest.data_module` from the trainrun's `data.data_module` (a bare
  name) instead of copying the whole `DataConfig`, so the server composes the child env with that
  module's plugin. Only a pipeline run needs a data module, the pipeline graph does not.
- **`load_manifest_bytes` sends one bare manifest with an absolute local path.** The gRPC example
  helper now loads a single bare plugin manifest (`name` + source + `capabilities`) for a `LoadPlugin`
  call and resolves a local plugin's relative `path` to absolute against the manifest file's
  directory, since the server runs elsewhere and `LoadPlugin` rejects a client-relative path. Git
  manifests (`repo` + `tag`) are sent unchanged.
- **Refreshed the `cuvis_ai_version` metadata stamps in the bundled configs.** The pipeline
  library under `configs/pipeline/**` and the two `configs/trainrun/*_pipeline.yaml` snapshots
  carried stale `0.1.x` dev-version stamps (down to a hardcoded `0.1.0`); bumped to `0.5.3` to
  match the current schemas line. Pairs with the schemas fix that auto-stamps freshly serialized
  pipelines with the installed version.
- **Dropped the `cuvis-ai-dataloader` dev/test dependency entirely.** cuvis-ai no longer depends on
  the data plugin (or, transitively, the cuvis SDK) at all. Tests mock the data layer
  (`SyntheticAnomalyDataModule` / `create_test_cube`); the real cu3s reader is covered by the
  plugin's own suite. Removed the unused real-cube fixtures (`data_config_factory`,
  `test_data_files_cached`) and the plugin's `[tool.uv.sources]` editable entry. The default test
  env is now fully SDK-free (no Windows `cuvis-il` wheel gap); notebooks that load `.cu3s` provision
  the plugin themselves.
- **Use-case notebooks updated for the plugin-based data layer.** The four `notebooks/use_cases/`
  notebooks now install `cuvis-ai-dataloader[cu3s,coco]` in their Colab bootstrap (cuvis-ai no longer
  bundles the cu3s reader) and select frames via `measurement_indices` instead of the removed
  `predict_ids` id-lists. Added `notebooks/use_cases/README.md` covering the env setup (install the
  data plus model plugins, then `uv run jupyter lab`).
- **Bumped the `cuvis-ai-core` floor to `>=0.8.0` and consume core + schemas from PyPI.** The floor
  matches the adopted `cuvis-ai-dataloader` / `Cu3sDataModule` APIs (0.8.0 dropped
  `SingleCu3sDataModule` / `load_plugins`) and pairs with `cuvis-ai-schemas[full]>=0.6.0`. The
  dev-only editable `[tool.uv.sources]` for core and the unpublished `plugins` extra are kept as a
  commented local-dev scaffold, so the committed lock resolves both libs from their published
  releases instead of sibling working copies (which a CI checkout cannot resolve).
- **Security:** upgraded dev/doc tooling to clear `pip-audit` advisories: `bleach 6.4.0`,
  `cryptography 49.0.0`, `jupyter-server 2.20.0`, `jupyterlab 4.6.0`, `msgpack 1.2.1`, and
  `tornado 6.5.7` (lock-only; these are dev/doc transitive dependencies, not runtime deps).

## 0.8.0 - 2026-06-11

- Bumped the `cuvis-ai-core` floor to `>=0.7.1` and `cuvis-ai-schemas[full]` to `>=0.5.2` for the reworked `NodeRegistry`: the plugin contract / runtime-smoke tests now read a loaded plugin's config from `registry.plugin_catalog[name]` (core dropped the redundant `plugin_configs` dict; `plugin_registry` became `loaded_plugin_nodes`, which `cuvis_ai_schemas.is_plugin` reads as of 0.5.1). The `>=0.7.1` floor also inherits core's transitive security patches, pulling `aiohttp` to `3.14.1` (CVE-2026-34993, CVE-2026-47265) and `idna` to `3.18` in the lock.
- **Security:** raised the dependency floors that lagged the lock so the floor audit and `pip-audit` pass: `cuvis>=3.5.1.0`, `ftfy>=6.3.1`, `setuptools>=81.0.0,<83`, a Windows `cuvis-il>=3.5.0` lower bound, and `pip>=26.1.2` (PYSEC-2026-196) for the release-tooling extra.
- Backfilled a bare-name **`plugins:` block** into every pipeline YAML (e.g. `- cuvis_ai_builtin`) so each pipeline declares its plugin set explicitly; `scripts/backfill_pipeline_plugins.py` seeds the field, delegating to core's `reorder_pipeline_with_plugins` helper.
- Carried the **inline node catalog** in each plugin manifest: `configs/plugins/<name>.yaml` `provides:` now lists `CatalogNodeEntry` items directly. Added the `turbovec` manifest, declared `package_name` overrides, and imported `PluginManifest` from `cuvis_ai_schemas.plugin`.
- Regenerated all plugin manifests to the single-`CatalogPortSpec`-per-port shape and removed the per-manifest `*.metadata.json` sidecars; updated the manifest-contract tests and the port/node docs for the `variadic` flag.
- Marked `TensorBoardMonitorNode` `artifacts` / `metrics` inputs as single `PortSpec`s with `variadic=True` (was the implicit `list[PortSpec]` fan-in form).
- Fixed the gRPC train flow and made `SoftChannelSelector`'s statistical initialization device-safe — it now computes on the same device as `channel_logits`, so it survives a pipeline moved to GPU.
- **Security/docs:** removed the `https://polyfill.io/v3/polyfill.min.js` include from `mkdocs.yml`. The `polyfill.io` domain is hijacked and was injecting a credential-phishing "Sign in" prompt on the docs site; MathJax 3 needs no polyfill on supported browsers. Added a `Redeploy docs (manual)` workflow (`workflow_dispatch` with a version input) so the gh-pages site can be republished without cutting a PyPI release.
- Fixed `AnomalyDetectionMetrics` average precision: switched to histogram-based `BinaryAveragePrecision(thresholds=N)` so state is bounded by construction, and reset only on `(stage, epoch)` boundaries so the metric accumulates across batches via `update()` within a validation epoch — the per-batch emitted value is a running AP that converges to true epoch-level AP, instead of a leak-prone cumulative or a noisy per-batch AP.
- Added dependency-compatibility CI: `.github/workflows/dep_compat.yml` (host-floor audit) and `registry_compat.yml` (plugin-vs-core audit); pinned dependency floors; documented plugin loading via `--plugins-dir`.
- Added pipeline render tooling: `scripts/render_pipelines.py` emits transparent PNG renders of every pipeline YAML. Stopped tracking the generated render artifacts.
- Made `scripts/` a PEP 420 namespace package (dropped its `__init__.py`) so it merges with `cuvis-ai-core`'s `scripts/`; unblocked the orchestrator smoke on Windows.
- Docs: aligned the plugin docs to the bare-name `plugins:` model and clarified that runnable examples live as notebooks in `cuvis-ai-cookbook` (datasets on Hugging Face).
- Docs: fixed the README documentation links broken by the docs IA restructure (`user-guide/installation` and `user-guide/quickstart` to `get-started/`, `node-catalog` to `catalogs/nodes`, `plugin-system` to `reference/plugin-development`, `api` to `reference/python-api`, `development/contributing` to `reference/contributing`).

## 0.7.3 - 2026-05-18

- Added GoatCounter page-view analytics to the docs site via Material's custom-analytics partial (`overrides/partials/integrations/analytics/custom.html`, `extra.analytics.provider: custom`). No cookies, no consent banner; tracked at `https://cuvis-ai.goatcounter.com`.

## 0.7.2 - 2026-05-11

- **CI:** run the gh-pages `deploy-docs` job inside `cubertgmbh/cuvis_pyil:3.5.0-ubuntu24.04` with `libgl1` / `libglib2.0-0` / `ffmpeg` installed, matching the working `doc-build` job in `ci.yml`. The 0.7.1 deploy failed because the auto-generated Nodes-catalog generator imports `cuvis_ai.node`, which transitively initializes the `cuvis` package and aborts on a vanilla runner. No `cuvis_ai` code changes — this release exists solely to re-trigger the release pipeline so gh-pages actually updates.

## 0.7.1 - 2026-05-11

- **Docs IA restructure** (ALL-5655). Nine top-level sections — Home, Get Started, Concepts, Tutorials, Catalogs, Workflows, Agentic Integration, Deployment, Reference — ordered as a learning path. Major moves: `user-guide/{installation,quickstart}` → `get-started/`; `how-to/*` → `workflows/`; `node-catalog/*` → `catalogs/nodes/*`; `config/*` → `reference/configuration/`; `api/*` → `reference/python-api/`; `plugin-system/*` → `reference/plugin-development/`; `development/*` → `reference/contributing/`; `grpc/*` + `use_cases/grpc-workflow.md` → `deployment/`. New: agentic-integration section, datasets catalog mirrored from HuggingFace, notebook tutorial gallery, `get-started/first-pipeline.md`, `workflows/{statistical,gradient}-training.md`. Removed `docs/use_cases/`, `user-guide/configuration.md` stub, duplicate `plugin-system/overview.md`. **All URLs change — no redirects.** `mkdocs build --strict` clean.
- **Auto-generated Nodes catalog.** `mkdocs-gen-files` + `scripts/generate_node_catalog.py` build a category-grouped index page from `NodeRegistry` with per-category SVG icons (`docs/images/node-categories/`) and a client-side filter (`docs/javascripts/node_catalog_filter.js`). Replaces nine hand-maintained `docs/catalogs/nodes/*.md` pages. Added `scripts/math_directive_hook.py` MkDocs hook (RST `.. math::` → MathJax; auto-hide TOC on catalog pages).
- **Lentils Dinomaly use-case notebook** (`notebooks/use_cases/lentils_dinomaly.ipynb`) — HF dataset integration and H.264 video export.
- **Helper-scripts package renamed** `tools/` → `scripts/`. Updated `[project.scripts]` (`create-stubs = "scripts.generate_node_port_stubs:main"`), MkDocs macros, codecov ignore, `.gitignore`, git hooks, copilot-instructions.
- **Removed `configs/plugins/registry.yaml`.** Use per-plugin manifests (`configs/plugins/<plugin>.yaml`).
- **Bundled ffmpeg via `imageio-ffmpeg`.** `ToVideoNode` resolves the binary from the wheel by default — no system install needed. Override with `CUVIS_AI_FFMPEG_BIN` for `h264_nvenc` / `vaapi` / `amf`. Blood-perfusion notebook MP4 export gains `+faststart`.
- **Site rebrand to Cubert CI.** `palette: custom` lets `docs/stylesheets/extra.css` drive both Material schemes; Rajdhani headings via Google Fonts `@import`, Roboto / Roboto Mono for body and code. Mermaid theme variables updated to match. Mermaid diagrams in `docs/concepts/*.md` switched from inline `style X fill:` to `classDef` so node colors stay legible in dark mode.
- **mkdocs plugin swap.** Dropped `mkdocs-literate-nav`; added `mkdocs-macros-plugin` (drives `scripts/docs_macros.py`) and `mkdocs-llmstxt` (emits `llms-full.txt`). `mkdocs-gen-files` re-added for the Nodes-catalog generator. API reference consolidated into the Nodes catalog. Install guide gains a Cuvis SDK section.
- **Renamed local blood-perfusion dataset folder** `data/XMR_Blood_Perfusion/` → `data/XMR_Demo_Blood_Perfusion/` following the HuggingFace rename to `cubert-gmbh/XMR_Demo_Blood_Perfusion`. Users with the old folder can rename it in place; otherwise `uv run dataset download blood_perfusion` re-fetches ~7 GB.
- **Fixed broken cross-doc links** surfaced by `mkdocs build --strict`.
- **Dependency floors:** `cuvis-ai-core>=0.5.3` (Blood_Perfusion registry repoint), `cuvis-ai-schemas[full]>=0.4.1`. Locked dev deps bumped to clear pip-audit CVEs.

## 0.7.0 - 2026-05-04

- Extracted the `examples/` tree (70 files) into a new sister repo, [`cuvis-ai-cookbook`](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook). Removed `docs/grpc/example-clients.md` (now redundant) and rerouted all in-doc `examples/...` links to cookbook GitHub URLs.
- Renamed `CIETristimulusFalseRGBSelector` → `CIETristimulusRGBSelector`. The CIE 1931 tristimulus integration produces a faithful RGB rendering, not a false-color rendering — the previous name was misleading. Updated the plugin registry (`configs/plugins/cuvis_ai_builtin.yaml`), the four SAM3 pipeline configs (`configs/pipeline/sam3/sam3_*.yaml`, including `name: false_rgb` → `name: true_rgb`, edge references, and `false-rgb` metadata tags/description), and the object-tracking notebooks under `notebooks/use_cases/`. No deprecation shim — direct rename.
- Added `_category` (`NodeCategory`) and `_tags` (`frozenset[NodeTag]`) ClassVars on every auto-registered `Node` subclass across `cuvis_ai/node`, `cuvis_ai/anomaly`, and `cuvis_ai/deciders` (105 classes, including private/base classes such as `_ScoreNormalizerBase` and `_BaseJsonWriterNode`). New `tests/test_node_categories.py` enforces per-class declarations (via `__dict__`), requires at least one modality or lifecycle tag per node, and rejects any single category covering >70% of the catalog.
- Added `assets/node_icons/*.svg` to package-data and bumped `cuvis-ai-schemas[full]>=0.4.0`.
- Added Windows FFmpeg DLL bootstrap (`cuvis_ai/__init__.py`): walks `PATH` at import time and registers every directory containing an `avcodec-*.dll` via `os.add_dll_directory`, so torchcodec can load `libtorchcodec_core*.dll` on Python 3.8+ where `PATH` is no longer consulted for DLL dependencies. No-op on non-Windows.
- Refactored `cuvis_ai.anomaly` and `cuvis_ai.deciders` modules into `cuvis_ai.node.anomaly` and `cuvis_ai.node.deciders`; the legacy locations now emit `DeprecationWarning` and re-export. `cuvis_ai.node` namespace exposes `BinaryDecider`, `DeepSVDDProjection`, `LADGlobal`, `QuantileBinaryDecider`, `RXGlobal`, `RXPerBatch`, `TwoStageBinaryDecider`, and `ZScoreNormalizerGlobal` directly. Plugin manifest and 15 pipeline YAMLs updated to the new `class_name` paths.
- Added `InsetComposer` (`cuvis_ai/node/compositing.py`): pastes a fixed-size inset frame into a corner of a larger base frame for picture-in-picture video output. Pairs with `ROIZoomNode` (the inset is expected at final pixel size). Configurable corner (`top-left` / `top-right` / `bottom-left` / `bottom-right`), `margin_px`, `border_px`, and `border_color`; per-frame `valid` port leaves the base untouched when the ROI is stale.
- Changed `ToVideoNode` to drop its unused `video_path` output port — it is a SINK and now matches the canonical contract used by `NumpyFeatureWriterNode` and the JSON writers (empty `OUTPUT_SPECS`, `forward()` returns `{}`).
- Bumped pinned plugin tags to latest published patches: adaclip `v0.1.2 → v0.1.3`, ultralytics `v0.1.0 → v0.1.1`, deepeiou `v0.1.0 → v0.1.1`, trackeval `v0.1.0 → v0.1.1`, sam3 `v0.1.3 → v0.1.5`. Picks up `_category` / `_tags` palette annotations and the `cuvis-ai-schemas>=0.4.0` floor across all upstream plugins.
- Removed `imantics` from runtime deps (no longer used).
- Added `notebook` to the `dev` extra so `uv sync --extra dev` is sufficient to run the use-case notebooks locally.
- Synchronized the plugin-manifest test-tag expectations to `v0.1.1` (`tests/test_plugin_manifest.py`) to match the latest published plugin tags.
- Removed three orphaned test files that imported helpers from the extracted `examples/object_tracking/` and `examples/export_cu3s_false_rgb_video.py` (now in `cuvis-ai-cookbook`); coverage now belongs in the cookbook repo.
- Renamed `docs/tutorials/` → `docs/usecases/` → `docs/use_cases/` (mkdocs nav heading and 9 doc pages); rewrote `docs/use_cases/blood-perfusion.md` to mirror the notebook section-for-section (NDVI flow + custom-node SpO2 example), dropping the unused PCA+HSV and band-limited PCA sections.
- Moved `notebooks/blood_perfusion/nd_blood_perfusion.ipynb` to `notebooks/use_cases/blood_perfusion.ipynb` and split the object-tracking walkthrough into two notebooks under `notebooks/use_cases/`: `object_tracking_passive.ipynb` (text-prompt + SAM3 mask propagation on RGB and CIR video, sharing a cached `rgb_video.mp4` + COCO `tracking_results.json`) and `object_tracking_active.ipynb` (invisible-spectral-ink active tracking via SPAM). Both follow the prepare → build → run → watch rhythm.
- Added an Open-in-Colab badge and a bootstrap install cell to every use-case notebook so they run end-to-end on `colab.research.google.com` without a pre-cloned checkout. Badges link the notebook on `main` so released revisions stay reproducible.
- Standardized "Cuvis.AI" casing across the docs (was a mix of "CUVIS.AI" / "cuvis.ai").
- Added `docs/javascripts/os-tab-sync.js` to keep OS-tabbed install snippets (Linux / macOS / Windows) in sync across the install guide.
- Documented the Graphviz system-binary requirement in `docs/installation.md` so `dot`-based pipeline visualization works out of the box on a fresh install (Python `graphviz` is just bindings).
- Refreshed the `README.md` status badges to flat-square style and added a link to the [`cuvis-ai-agentic-skills`](https://github.com/cubert-hyperspectral/cuvis-ai-agentic-skills) sister repo.
- Regenerated the docstring-coverage badge (`assets/interrogate_badge.svg`) at `96.0%`.
- Fixed `auto_register_package` registry-size floor regression after the anomaly/deciders move: pointed the auto-register walk at `cuvis_ai.node.{anomaly,deciders}` since the legacy shims re-export classes whose `__module__` now points at the new locations.
- Bumped `cuvis-ai-core` floor `>=0.3.4` → `>=0.5.0` and `cuvis-ai-schemas[full]` `>=0.3.0` → `>=0.4.0` to pick up the gRPC `list_available_nodes` metadata populator, the new `MissingNodeMetadataWarning` runtime check, and the `NodeCategory` / `NodeTag` / `NodeInfo.{category,tags,icon_svg}` schema additions. Widened `requires-python` from `>=3.11,<3.12` to `>=3.11,<3.14`. Removed `opencv-python-headless` from the docs extra (not required by the current mkdocs config).
- Updated the gRPC workflow helper docstring to describe the concrete session lifecycle (create / build / train / predict) instead of internal release vocabulary.

## 0.6.0 - 2026-04-27

- Removed `examples/hugging_face/` example scripts (`huggingface_api_demo.py`, `huggingface_local_demo.py`, `huggingface_gradient_training.py`, `test_huggingface_local_minimal.py`) and the in-tree `cuvis_ai/node/adaclip.py` (`AdaCLIPLocalNode`). The released AdaCLIP plugin (`cuvis_ai_adaclip` via `configs/plugins/adaclip.yaml`) is unaffected.
- Removed the `### AdaCLIP Nodes` autodoc section from `docs/api/nodes.md`; it pointed at the deleted in-tree module.
- Renamed `PipelineComparisonVisualizer` input port `adaclip_scores` → `anomaly_scores` (and the corresponding TensorBoard heatmap artifact `adaclip_scores_heatmap_sample_*` → `anomaly_scores_heatmap_sample_*`). The port is plugin-agnostic; updated `tests/node/test_pipeline_visualization.py`, `cuvis_ai/node/losses.py` docstring example, AdaCLIP pipeline/trainrun YAMLs, `examples/adaclip/*_training.py`, `docs/tutorials/adaclip-workflow.md`, and `docs/how-to/monitoring-and-viz.md`.
- Registered the previously-omitted built-in nodes `ROIZoomNode`, `MaskRobustifier`, `MaskToBBoxKalman`, `MaskedMeanSpectrum`, and `SpectrumPlotNode` in `configs/plugins/cuvis_ai_builtin.yaml` so they are discoverable when the gRPC server runs in a separate venv.
- Changed `ToVideoNode` encoder backend from OpenCV `cv2.VideoWriter` (FOURCC `mp4v`, uncontrollable bitrate, ~1.6 Mbps MPEG-4 Part 2 output) to a lazily-spawned `ffmpeg` subprocess that pipes raw `rgb24` frames over stdin. Produces H.264 (`libx264`) at a configurable target bitrate (default `12M`). Requires the `ffmpeg` binary on `PATH`.
- Added `video_codec` (default `"libx264"`) and `bitrate` (default `"12M"`) parameters to `ToVideoNode`. Hardcoded `-pix_fmt yuv420p` plus `-vf pad=ceil(iw/2)*2:ceil(ih/2)*2` to guarantee valid output dimensions for 4:2:0 chroma subsampling.
- Removed `ToVideoNode(codec=...)` (FourCC) parameter — renamed to `video_codec` (ffmpeg codec name) since the value namespace changed. Pipeline YAML configs do not set `codec=` explicitly, so no existing config files need updates.
- Added robust subprocess lifecycle handling to `ToVideoNode`: `close()` sends EOF, waits for mux completion, and raises `RuntimeError` with drained stderr on non-zero ffmpeg exit. Per-frame `stdin.write` catches `BrokenPipeError` and surfaces the encoder error rather than silently truncating the video.
- Relocated cu3s false-RGB video exporter from `examples/object_tracking/export_cu3s_false_rgb_video.py` to `examples/export_cu3s_false_rgb_video.py`; updated `tests/node/test_export_cu3s_false_rgb_video.py` and `tests/node/test_range_average_false_rgb_selector.py` imports accordingly.
- Added `ffmpeg` to CI apt-install steps (`ci.yml`, `plugin-runtime-smoke.yml`) so future integration tests can exercise the encoder end-to-end.
- Consolidated `NpyReader` and `NumpyFeatureWriterNode` into a single `cuvis_ai.node.numpy_file` module, mirroring the existing `json_file` pattern. Updated imports, plugin manifest, tests, docs, and examples.
- Registered `TrackingPointerOverlayNode`, `BBoxPrompt`, `MaskPrompt`, and `TextPrompt` in `configs/plugins/cuvis_ai_builtin.yaml` so they are discoverable via the plugin manifest.
- Added `ROIZoomNode` (`cuvis_ai/node/compositing.py`): crops a bbox region from an RGB frame and resizes to fixed dimensions for zoom-inset video streams.
- Added `MaskRobustifier` and `MaskToBBoxKalman` (`cuvis_ai/node/mask_ops.py`): morphological cleanup of binary masks and Kalman-smoothed bbox tracking from mask outputs.
- Added `SpectrumPlotNode` (`cuvis_ai/node/spectrum_plot.py`): renders per-frame matplotlib line plots (reference vs tracked spectrum) to RGB frames for secondary spectrum video export.
- Added `MaskedMeanSpectrum` (in `cuvis_ai/node/spectral_extractor.py`): computes per-frame mean spectrum of a hyperspectral cube over a binary mask; clarified batch semantics on `BBoxSpectralExtractor` / `SpectralSignatureExtractor` docstrings.
- Updated `examples/spectral_angle_mapper/spam_invisible_ink.py` to emit synchronized side videos (ROI zoom via `ROIZoomNode`, spectrum plot via `SpectrumPlotNode`) alongside the main overlay, with conditional profiling summary and refactored output-directory handling.
- Removed `--rgb-xml-path` argument from `spam_invisible_ink_every_where.py` and simplified downstream bootstrap dispatch.
- Added `--overlay-frame-id` flag to `examples/object_tracking/render_tracking_overlay.py`, which renders the frame index in the top-left corner of each output frame.
- Rewrote `.github/copilot-instructions.md` to clarify that this repo is the plugin node catalog (with `cuvis-ai-core` and `cuvis-ai-schemas` as sibling repos), and to document the Python 3.11 / uv / node-registration conventions.
- Updated `README.md` to use a locally-hosted banner (`docs/images/banner.png`) instead of an external CDN URL, and revised the project description to emphasize extensibility and video pipelines. Minor capitalization fixes in `CONTRIBUTING.md`.

## 0.5.0 - 2026-04-10

- Added `TextPrompt`, scheduled `--prompt <text@frame_id>` parsing, and updated local/gRPC SAM3 text-propagation examples to drive `SAM3TextPropagation` through a runtime `text_prompt` port instead of constructor hparams.
- Added SAM3 prompt-free segment-everything tooling: `SAM3SegmentEverything`, local CLI wiring, and CU3S/video pipeline YAMLs for per-frame automatic mask generation with overlay/video/JSON outputs.
- Added runtime SAM3 bbox propagation tooling: `BBoxPrompt`, local/gRPC bbox-propagation examples, and CU3S/video bbox-propagation pipeline YAMLs using scheduled `--prompt <object_id:detection_id@frame_id>` bbox updates from detection JSON.
- Added runtime SAM3 mask propagation tooling: `MaskPrompt`, local/gRPC mask-propagation examples, and CU3S/video mask-propagation pipeline YAMLs using scheduled `--prompt <object_id:detection_id@frame_id>` mask updates from detection JSON.
- Added SAM3 text-propagation pipeline configs and a new gRPC client (`examples/grpc/sam3/sam3_text_propagation_client.py`) supporting CU3S/video inputs plus plugin-manifest bootstrap.
- Added SAM3 tracking workflow updates across propagation scripts and examples, including batch processing for full-folder video runs, per-node profiling, threshold/name-suffix options, and frame-lookup support in `TrackingResultsReader`.
- Added `NDVISelector` for normalized-difference vegetation index band selection, `ScalarHSVColormapNode` for scalar-to-HSV colormap rendering, and `DetectionCocoJsonNode` for streaming COCO detection JSON output.
- Added per-frame `PCA` dimensionality reduction node alongside the existing trainable variant.
- Added Spectral Angle Mapper (SPAM) pipeline nodes and tooling for spectral-angle-based workflows.
- Added `BBoxSpectralExtractor`, sparkline visualization helpers, and richer `BBoxesOverlayNode` annotations (`draw_labels`, `frame_id`).
- Added occlusion and Poisson inpainting utilities with tests and object-tracking example integrations.
- Added ByteTrack and tracker workflow expansion: spectral-aware association, COCO JSON sinks, threshold/JSON sweep tooling, spectral re-ID validation, RT-DETR/YOLO integration points, and overlay/transcoding helpers for rendered tracking outputs.
- Added DeepEIOU plugin integration plus related preprocessing, NumPy writer, and tracking overlay renderer updates.
- Added TrackEval preparation/evaluation tooling updates for aligned HOTA benchmarking workflows, including prediction frame-id passthrough in evaluator pipelines when supported by the metric plugin.
- Added released tracking plugin manifests for ByteTrack, DeepEIOU, TrackEval, Ultralytics, RT-DETR, and a `cuvis_ai_builtin` manifest.
- Added blood perfusion tutorial (`docs/tutorials/blood-perfusion.md`) and four example scripts under `examples/blood_perfusion/` covering NDVI, PCA, and PCA-HSV visualizations.
- Added plugin node catalog documentation page listing all available plugin nodes.
- Added ~41 new test files covering PCA, NDVI, colormap, text prompt, manifest sync, spectral extractor, occlusion, video, tracking overlay, and more.
- Changed tracking JSON export so `CocoTrackMaskWriter` can consume optional `category_ids` and `category_semantics` inputs, preserving the old single-category behavior when they are absent and writing multi-category `categories` headers when they are present.
- Changed local SAM3 bbox propagation from the archived `--detection` single-seed flow to the same scheduled prompt contract used by mask propagation, including optional bbox prompt debug overlays.
- Changed local SAM3 mask propagation from archived PNG prompts to detection-JSON-driven label-map prompting, and clarified that gRPC mask propagation sends masks directly through `InputBatch.mask`.
- Renamed `CocoTrackMaskWriter(category_name=...)` to `CocoTrackMaskWriter(default_category_name=...)`, changed the default fallback label to `"object"`, and clarified that this constructor value is only the fallback label when `category_semantics` is absent.
- Refactored and consolidated video/tracking utilities (including `cuvis_ai/node/video.py`), moved SAM3 examples into a dedicated subdirectory, and adopted shorthand port syntax across updated examples.
- Refactored shared XML plugin helpers into `cuvis_ai/utils/xml_plugin_parser.py`.
- Refactored prompt specs, parsers, and frame-hw resolution to deduplicate shared logic across text/bbox/mask propagation modes.
- Reorganized AdaCLIP gRPC examples under `examples/grpc/adaclip/` and updated gRPC workflow/docs utilities around explicit config resolution and session search paths.
- Refined tracking output tooling with JSON IO/overlay updates, new CLI output-dir helpers, and expanded tracking regression tests.
- Updated SAM3 text-propagation pipeline YAMLs and example docs to match runtime text prompting plus category-aware tracking JSON output.
- Updated ByteTrack and tracking documentation, including multi-pipeline usage and FFmpeg/torchcodec setup guidance.
- Updated plugin/trainrun configs to match current SAM3 and channel-selector runtime paths.
- Updated SAM3 plugin to v0.1.3 and switched AdaCLIP plugin to released repository.
- Updated docs: removed 7 redundant pages and cleaned up stale references across the documentation site.
- Bumped cuvis-ai-schemas to >=0.3.0 and cuvis-ai-core to >=0.3.4.
- Consolidated `json_reader` and `json_writer` modules into a single `json_file` module; updated all node registrations, pipeline configs, imports, and documentation.
- Switched from `opencv-python` to `opencv-python-headless`.
- Fixed SAM3 batch-runner control flow and mask-overlay color handling.
- Fixed JSON reader robustness and pre-push regressions in manifest sync, CLI commands, and statistical-contract tests.
- Fixed video/tracking fallback and output handling: `VideoIterator` now falls back to OpenCV when torchcodec is unavailable, `output_video_path` naming is normalized, and ByteTrack JSON output path heuristics were hardened.

## 0.4.0 - 2026-02-27

- Added reusable `WelfordAccumulator` utility (`cuvis_ai.utils.welford`) for streaming mean/variance/covariance
- Added `resolve_reduce_dims()` as shared module-level utility in `binary_decider`
- Added `TRAINABLE_BUFFERS` class attribute — 5 nodes declare trainable buffers, base class handles buffer↔parameter conversion in freeze/unfreeze automatically
- Added `freeze()` for `LearnableChannelMixer` matching existing `unfreeze()` override
- Added `ConcreteChannelMixer` and `LearnableChannelMixer` exported from `cuvis_ai.node`
- Added all 6 visualization nodes exported from `cuvis_ai.node`: `AnomalyMask`, `RGBAnomalyMask`, `ScoreHeatmapVisualizer`, `CubeRGBVisualizer`, `PCAVisualization`, `PipelineComparisonVisualizer`
- Added insufficient-samples guard to `RXGlobal` and `ScoreToLogit` — raises early when training data has too few samples
- Added plugin runtime smoke CI workflow (`plugin-runtime-smoke.yml`) with slow plugin tests
- Added AdaCLIP standalone plugin manifest (`configs/plugins/adaclip.yaml`) and 6 example scripts
- Added plugin contract, manifest sync, and runtime smoke test files
- Added 8 new test files: `test_welford`, `test_freeze_unfreeze`, `test_channel_selector_coverage`, `test_concrete_channel_mixer`, `test_pipeline_visualization`, `test_binary_decider`, `test_data_node`, `test_rx_per_batch`
- Added pytest markers (`unit`/`integration`/`slow`) on all 30 test files; session-scoped fixtures for expensive operations; pytest config consolidated in `pytest.ini`
- Added SAM3 plugin integration scaffolding with plugin registry, pipeline configs, and example manifests
- Added CU3S video data support with restructured data nodes and CU3SDataNode
- Added RangeAverageFalseRGBSelector for wavelength-range-averaged false RGB
- Added CIETristimulusFalseRGBSelector using CIE 1931 2-degree observer color matching functions
- Added CameraEmulationFalseRGBSelector using Gaussian spectral response curves
- Added NormMode enum and unified percentile-based RGB normalization in ChannelSelectorBase (per_frame/running/statistical with warmup+accumulation)
- Added sRGB gamma, _compute_raw_rgb() hook, and statistical_initialization() to ChannelSelectorBase
- Added LearnableChannelMixer weights output port for loss/viz consumption
- Added ForegroundContrastLoss with OKLab color space and anchor_weight anti-gaming penalty
- Added OKLab perceptual color space utilities (rgb_to_oklab, linear_rgb_to_oklab, srgb_to_linear)
- Added ImageArtifactVizBase, ChannelSelectorFalseRGBViz, and ChannelWeightsViz visualization nodes
- Added MaskOverlayNode and create_mask_overlay shared PyTorch utility
- Added TrackingOverlayNode for per-object colored mask overlays with contour lines and ID labels
- Added multi-object overlay rendering utilities (render_multi_object_overlay)
- Added TrackingCocoJsonNode for streaming COCO instance-segmentation JSON with RLE masks and atomic writes
- Added ToVideoNode for streaming RGB frames to MP4 via OpenCV
- Added channel selector false RGB experiment with Hydra configs, inspect mode, and training pipeline
- Added sam3_hsi_tracker.py end-to-end SAM3 tracking example using CIE false RGB and core Predictor
- Added SAM3 pipeline configs: naive false RGB, learned projection, and spectral signature extraction
- Added mesu_index passthrough in CU3SDataNode and ChannelSelectorFalseRGBViz for frame tracking
- Added LR scheduling (reduce_on_plateau) wired to GradientTrainer
- Added unit tests for data, video, band selection, tracking COCO JSON, and tracking overlay nodes
- Changed RXGlobal, ScoreToLogit, LADGlobal to use `WelfordAccumulator` instead of inline Welford implementations
- Changed `_compute_band_correlation_matrix` to single-pass streaming with `WelfordAccumulator`
- Changed TrainablePCA and LearnableChannelMixer to use streaming covariance + `eigh` instead of concat + SVD
- Changed SoftChannelSelector variance init to use streaming `WelfordAccumulator`
- Changed ZScoreNormalizerGlobal to use streaming `WelfordAccumulator` instead of concat + subsample
- Changed supervised band selectors to use template method pattern
- Changed YAML configs and docs to use new schema field names (`hparams`, `class_name`)
- Changed LearnableChannelMixer output normalization from per-image min-max to BatchNorm2d + sigmoid
- Changed channel selector training config to max_epochs=200 with early stopping and LR scheduling
- Changed ForegroundContrastLoss to vectorized batch computation (no per-sample loop)
- Changed export_cu3s_false_rgb_video.py from argparse to Click CLI and DataLoader to core Predictor
- Changed plugin registry to use relative path for SAM3 and AdaCLIP repo tag v0.1.2
- **Breaking**: Reorganized channel selector and mixer nodes into separate files
- **Breaking**: Renamed 9 classes to reflect selector/mixer distinction
- **Breaking**: Deleted old files — no deprecation stubs or re-exports
- Removed redundant `.to(device)` calls — pipeline handles device placement
- Updated 13 pipeline + 17 trainrun YAML configs with new `class_name` paths
- Updated 11 example scripts with new import paths
- Updated 19 documentation files with new class names and import paths
- Fixed `pyproject.toml` uv source field (`develop` to `editable`)
- Fixed Werkzeug CVE-2026-27199 by bumping 3.1.5 → 3.1.6
- Fixed ToVideoNode parameter typo: output__video_path renamed to output_video_path
- Fixed setuptools<82 pin for tensorboard pkg_resources compatibility
- Fixed Windows uv script path errors by using python -m in hooks and tests
- Fixed CU3SDataNode cube input spec to use torch.Tensor dtype
- Expanded training data splits in tracking_cap_and_car.yaml
- Pinned cuvis-ai-schemas to git main branch
- Removed examples/adaclip/plugins.yaml (consolidated into central registry)
- Removed Phase 1 scaffold files (sam3_example.md, sam3_tracking_example.py)

## 0.3.0 - 2026-02-11

- Fixed README documentation links to use docs.cuvis.ai with correct version prefix
- Fixed Pillow CVE-2026-25990 by bumping 12.1.0 to 12.1.1
- Added comprehensive documentation site with tutorials, API reference, and node catalog
- Added MkDocs Material theme with dark mode and versioned deployment via mike
- Added AnomalyPixelStatisticsMetric node replacing duplicate SampleCustomMetrics
- Added deep_svdd_factory utility module with ChannelConfig dataclass
- Added central plugin registry at configs/plugins/registry.yaml
- Added statistical-only training config (default_statistical.yaml)
- Added CI/CD pipeline with test, lint, security, and typecheck jobs
- Added PyPI release workflow with TestPyPI verification and docs deployment
- Added Dependabot configuration for GitHub Actions and pip dependencies
- Added automated test data downloader script with CLI entry point
- Added documentation test suite for link checking and code example validation
- Added Git hooks for ruff format and module case checking
- Added Apache-2.0 LICENSE file
- Changed TrainablePCA to require num_channels parameter (breaking)
- Changed type imports to use cuvis-ai-schemas package
- Changed RXLogitHead to ScoreToLogit and moved to cuvis_ai.node.conversion
- Changed BaseDecider import to BinaryDecider in deciders module
- Changed trainrun config into separate statistical and gradient variants
- Changed pyproject.toml for PyPI compliance and updated tooling configs
- Changed dependencies to add cuvis-ai-schemas and loosen cuvis version pin
- Changed restore-pipeline/restore-trainrun entry points to use cuvis_ai_core
- Improved README and CONTRIBUTING.md with plugin contribution workflow
- Improved docstring coverage to 95%+ across all public APIs
- Fixed LAD detector reset() initializing buffers with wrong shapes
- Fixed LAD detector unfreeze() losing device when converting buffers
- Fixed TrainablePCA with proper num_channels parameter and buffer shapes
- Fixed node import paths for cuvis-ai-schemas migration
- Fixed config references for trainrun and ScoreToLogit rename
- Fixed documentation links, module references, and placeholder content
- Fixed MkDocs build warnings and docstring formatting
- Fixed package metadata for PyPI submission
- Removed restore_pipeline.md from repo root
- Removed old changelog.md replaced by CHANGELOG.md
- Removed run_tests.yml replaced by ci.yml
- Removed outdated docs pages replaced by expanded docs sections

## 0.2.3 - 2026-01-29

- Added plugin system with Git repository and local filesystem support
- Added Pydantic plugin configuration with strict validation
- Added plugin caching in ~/.cuvis_plugins/ with version verification
- Added session-scoped plugin isolation for gRPC services
- Added plugin management gRPC RPCs (LoadPlugins, ListLoadedPlugins, GetPluginInfo, ClearPluginCache)
- Added JSON transport pattern for plugin manifests
- Added test migration infrastructure with 426 tests moved to cuvis-ai-core
- Changed repository architecture to split into cuvis-ai-core and cuvis-ai
- Changed import pattern to use cuvis_ai_core for framework components
- Fixed DataLoader access violation with num_workers=0
- Fixed gRPC servers to use single-threaded mode for cuvis SDK compatibility

## 0.2.2 - 2026-01-15

- Added restoration utilities in cuvis_ai.utils.restore module
- Added CLI entry points for restore-pipeline and restore-trainrun
- Added restore_pipeline.md guide with CLI and Python API examples
- Changed restoration utilities to auto-detect statistical vs gradient workflows
- Changed Python API surface to standardized imports
- Removed duplicate example scripts replaced with library utilities

## 0.2.1 - 2026-01-08

- Added Pydantic v2 config models as single source of truth with validation
- Added server-side Hydra composition with session-scoped search paths
- Added config RPCs: ResolveConfig, ValidateConfig, GetParameterSchema, SetSessionSearchPaths
- Added explicit 4-step workflow for gRPC API
- Changed terminology from Experiment to TrainRun across configs and RPCs
- Changed gRPC service into modular components
- Changed config transport to use config_bytes with central registry
- Fixed RPC surface for new config resolution flow
- Fixed tests and examples (596 tests passing, 65% coverage)

## 0.2.0 - 2025-12-20

- Added YAML-driven pipeline configuration with OmegaConf interpolation
- Added hybrid NodeRegistry for built-ins and custom nodes
- Added end-to-end pipeline serialization with YAML structure and .pt weights
- Added version/schema compatibility guards on load
- Added gRPC canvas management and discovery RPCs
- Added pipeline path resolution helpers with CUVIS_CANVAS_DIR environment variable
- Changed gRPC API to use PipelineConfig via config_bytes
- Changed Train RPC to require DataConfig and TrainingConfig
- Changed SaveCanvas/LoadCanvas to replace SaveCheckpoint/LoadCheckpoint
- Changed node state management to standard state_dict()
- Removed custom serialize/load patterns

## 0.1.5 - 2025-12-01

- Added gRPC service stack with proto definitions
- Added Buf Schema Registry integration for cross-language codegen
- Added session management and PipelineBuilder
- Added file-based data access via DataConfig
- Added two-phase training (statistical init then gradient fine-tuning)
- Added pipeline introspection RPCs and streaming training progress
- Changed output selection to use output_specs
- Changed node naming to deterministic counter-based scheme

## 0.1.3 - 2025-11-06

- Added port-based typed I/O system with PortSpec, InputPort, OutputPort
- Added graph connection API with auto-validation
- Added multi-input/output support for nodes
- Added training integration with PyTorch Lightning
- Changed nodes to declare INPUT_SPECS/OUTPUT_SPECS with auto-created ports
- Changed executor for port-based routing and stage-aware execution
