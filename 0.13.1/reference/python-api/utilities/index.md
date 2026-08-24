Status: Needs Review

This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

______________________________________________________________________

# Utilities API

Current helper modules used by the tracking, visualization, restore, and gRPC workflows.

## Workflow Helpers

### grpc_workflow

Shared helpers for gRPC example clients (session create / build / train / predict).

#### config_search_paths

```python
config_search_paths(extra_paths=None)
```

Return absolute search paths covering all config groups.

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def config_search_paths(extra_paths: Iterable[str | Path] | None = None) -> list[str]:
    """Return absolute search paths covering all config groups."""
    seeds = [
        CONFIG_ROOT,
        CONFIG_ROOT / "trainrun",
        CONFIG_ROOT / "pipeline",
        CONFIG_ROOT / "data",
        CONFIG_ROOT / "training",
    ]

    seen: set[Path] = set()
    paths: list[str] = []

    for path in [*seeds, *(extra_paths or [])]:
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(str(resolved))

    return paths
```

#### build_stub

```python
build_stub(
    server_address="localhost:50051",
    max_msg_size=300 * 1024 * 1024,
)
```

Create a gRPC stub for the CuvisAI service.

Parameters:

| Name             | Type  | Description                                    | Default             |
| ---------------- | ----- | ---------------------------------------------- | ------------------- |
| `server_address` | `str` | Server address (default: localhost:50051)      | `'localhost:50051'` |
| `max_msg_size`   | `int` | Maximum message size in bytes (default: 300MB) | `300 * 1024 * 1024` |

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def build_stub(
    server_address: str = "localhost:50051", max_msg_size: int = 300 * 1024 * 1024
) -> cuvis_ai_pb2_grpc.CuvisAIServiceStub:
    """Create a gRPC stub for the CuvisAI service.

    Parameters
    ----------
    server_address : str
        Server address (default: localhost:50051)
    max_msg_size : int
        Maximum message size in bytes (default: 300MB)
    """
    options = [
        ("grpc.max_send_message_length", max_msg_size),
        ("grpc.max_receive_message_length", max_msg_size),
    ]
    channel = grpc.insecure_channel(server_address, options=options)
    return cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
```

#### create_session_with_search_paths

```python
create_session_with_search_paths(stub, search_paths=None)
```

Create a session and register search paths.

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def create_session_with_search_paths(
    stub: cuvis_ai_pb2_grpc.CuvisAIServiceStub, search_paths: list[str] | None = None
) -> str:
    """Create a session and register search paths."""
    session_id = stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id
    paths = search_paths or config_search_paths()
    stub.SetSessionSearchPaths(
        cuvis_ai_pb2.SetSessionSearchPathsRequest(
            session_id=session_id,
            search_paths=paths,
            append=False,
        )
    )
    return session_id
```

#### resolve_trainrun_config

```python
resolve_trainrun_config(
    stub, session_id, name, overrides=None
)
```

Resolve a trainrun config via the ConfigService.

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def resolve_trainrun_config(
    stub: cuvis_ai_pb2_grpc.CuvisAIServiceStub,
    session_id: str,
    name: str,
    overrides: list[str] | None = None,
) -> tuple[cuvis_ai_pb2.ResolveConfigResponse, dict]:
    """Resolve a trainrun config via the ConfigService."""
    config_path = name if name.startswith("trainrun/") else f"trainrun/{name}"
    response = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="trainrun",
            path=config_path,
            overrides=overrides or [],
        )
    )
    config_dict = json.loads(response.config_bytes.decode("utf-8"))
    return response, config_dict
```

#### apply_trainrun_config

```python
apply_trainrun_config(stub, session_id, config_bytes)
```

Build the trainrun's pipeline, then attach the rest of the trainrun config.

`SetTrainRunConfig` no longer builds a pipeline and rejects a trainrun config that still carries a `pipeline:` section. So this helper splits the resolved trainrun: it `LoadPipeline`s the embedded `pipeline` section first (the server composes the per-run env from that pipeline's `plugins:` block), then sends the remaining trainrun config to `SetTrainRunConfig`.

The embedded pipeline must declare a `plugins:` block (the standalone `cuvis_ai/configs/pipeline/` yamls already do); a trainrun whose resolved pipeline omits it is rejected at `LoadPipeline` with a `suggest-plugins-fix` hint.

A trainrun whose `pipeline` is a path *reference* (a string) cannot be split this way: the pipeline lives in a separate YAML resolved relative to the trainrun file on the server, which only `RestoreTrainRun` knows how to locate and build. Use `RestoreTrainRun` for referenced trainruns (e.g. `examples/grpc/core/restore_trainrun_grpc.py`); it is also the single-call equivalent of this helper for inline pipelines.

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def apply_trainrun_config(
    stub: cuvis_ai_pb2_grpc.CuvisAIServiceStub,
    session_id: str,
    config_bytes: bytes,
) -> cuvis_ai_pb2.SetTrainRunConfigResponse:
    """Build the trainrun's pipeline, then attach the rest of the trainrun config.

    ``SetTrainRunConfig`` no longer builds a pipeline and rejects a trainrun
    config that still carries a ``pipeline:`` section. So this helper splits the
    resolved trainrun: it ``LoadPipeline``s the embedded ``pipeline`` section
    first (the server composes the per-run env from that pipeline's ``plugins:``
    block), then sends the remaining trainrun config to ``SetTrainRunConfig``.

    The embedded pipeline must declare a ``plugins:`` block (the standalone
    ``cuvis_ai/configs/pipeline/`` yamls already do); a trainrun whose resolved pipeline
    omits it is rejected at ``LoadPipeline`` with a ``suggest-plugins-fix`` hint.

    A trainrun whose ``pipeline`` is a path *reference* (a string) cannot be
    split this way: the pipeline lives in a separate YAML resolved relative to
    the trainrun file on the server, which only ``RestoreTrainRun`` knows how to
    locate and build. Use ``RestoreTrainRun`` for referenced trainruns (e.g.
    ``examples/grpc/core/restore_trainrun_grpc.py``); it is also the single-call
    equivalent of this helper for inline pipelines.
    """
    config = json.loads(config_bytes.decode("utf-8"))
    pipeline_section = config.pop("pipeline", None)
    data_section = config.get("data")
    if isinstance(pipeline_section, str):
        raise ValueError(
            "This trainrun references its pipeline by path "
            f"({pipeline_section!r}); apply_trainrun_config can only inline an "
            "embedded pipeline. Use RestoreTrainRun, which resolves the reference "
            "relative to the trainrun file and builds the pipeline server-side."
        )
    if pipeline_section is not None:
        load_request = cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=normalize_pipeline_bytes(json.dumps(pipeline_section).encode("utf-8"))
            ),
        )
        # Forward the data-module name so the server composes the per-run child
        # env with that module's plugin (e.g. cu3s -> cuvis-ai-dataloader).
        # Without it the compose runs with no data module and Train fails with
        # "no plugin provides data module '<name>'".
        data_module = (data_section or {}).get("data_module")
        if data_module:
            load_request.data_module = data_module
        stub.LoadPipeline(load_request)
    return stub.SetTrainRunConfig(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=cuvis_ai_pb2.TrainRunConfig(config_bytes=json.dumps(config).encode("utf-8")),
        )
    )
```

#### format_progress

```python
format_progress(progress)
```

Pretty-print training progress messages.

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def format_progress(progress: cuvis_ai_pb2.TrainResponse) -> str:
    """Pretty-print training progress messages."""
    stage = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
    status = cuvis_ai_pb2.TrainStatus.Name(progress.status)

    parts = [f"[{stage}] {status}"]
    if progress.losses:
        parts.append(f"losses={dict(progress.losses)}")
    if progress.metrics:
        parts.append(f"metrics={dict(progress.metrics)}")
    if progress.message:
        parts.append(progress.message)

    return " | ".join(parts)
```

#### load_manifest_bytes

```python
load_manifest_bytes(path)
```

Load one bare plugin manifest and return the JSON bytes for a LoadPlugin call.

The file is a single bare manifest (`name` + source + `capabilities`). A local plugin's relative `path` is resolved to absolute against the manifest file's directory, since the server runs elsewhere and cannot resolve a client-relative path (`LoadPlugin` rejects a relative local path). Git manifests (`repo` + `tag`) are returned unchanged.

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def load_manifest_bytes(path: Path) -> bytes:
    """Load one bare plugin manifest and return the JSON bytes for a LoadPlugin call.

    The file is a single bare manifest (``name`` + source + ``capabilities``).
    A local plugin's relative ``path`` is resolved to absolute against the
    manifest file's directory, since the server runs elsewhere and cannot
    resolve a client-relative path (``LoadPlugin`` rejects a relative local
    path). Git manifests (``repo`` + ``tag``) are returned unchanged.
    """
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(manifest, dict) and "repo" not in manifest:
        plugin_path = manifest.get("path")
        if isinstance(plugin_path, str) and plugin_path:
            resolved = Path(plugin_path)
            if not resolved.is_absolute():
                manifest["path"] = str((path.parent / resolved).resolve())
    return json.dumps(manifest).encode("utf-8")
```

#### normalize_pipeline_bytes

```python
normalize_pipeline_bytes(config_bytes)
```

Unwrap Hydra group wrappers until a PipelineConfig payload with `nodes` is reached.

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def normalize_pipeline_bytes(config_bytes: bytes) -> bytes:
    """Unwrap Hydra group wrappers until a PipelineConfig payload with ``nodes`` is reached."""
    payload: Any = json.loads(config_bytes.decode("utf-8"))

    for _ in range(6):
        if isinstance(payload, dict) and "nodes" in payload:
            return json.dumps(payload).encode("utf-8")
        if isinstance(payload, dict) and len(payload) == 1:
            candidate = next(iter(payload.values()))
            if isinstance(candidate, dict):
                payload = candidate
                continue
        break

    raise ValueError(
        "Resolved pipeline config could not be normalized to a PipelineConfig payload."
    )
```

#### resolve_pipeline_ref

```python
resolve_pipeline_ref(ref, *, trainrun_dir=None)
```

Load a trainrun's path-referenced pipeline YAML into an inline dict.

The bundled `cuvis_ai/configs/trainrun/*.yaml` reference their pipeline by a path relative to the trainrun file. Resolve it against `trainrun_dir` (default the package `cuvis_ai/configs/trainrun`) and load the pipeline YAML directly. The bundled pipeline yamls are self-contained `PipelineConfig`s (nodes + `plugins:`), so a direct file load avoids the Hydra group-path wrapping a server-side `ResolveConfig(config_type="pipeline")` would add.

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def resolve_pipeline_ref(ref: str, *, trainrun_dir: Path | None = None) -> dict:
    """Load a trainrun's path-referenced pipeline YAML into an inline dict.

    The bundled ``cuvis_ai/configs/trainrun/*.yaml`` reference their pipeline by a path
    relative to the trainrun file. Resolve it against ``trainrun_dir`` (default
    the package ``cuvis_ai/configs/trainrun``) and load the pipeline YAML directly. The
    bundled pipeline yamls are self-contained ``PipelineConfig``s (nodes +
    ``plugins:``), so a direct file load avoids the Hydra group-path wrapping a
    server-side ``ResolveConfig(config_type="pipeline")`` would add.
    """
    base = Path(trainrun_dir) if trainrun_dir is not None else (CONFIG_ROOT / "trainrun")
    pipeline_path = (base / ref).resolve()
    return yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
```

#### inline_pipeline_ref

```python
inline_pipeline_ref(config_dict, *, trainrun_dir=None)
```

Return `config_dict` with a path-referenced `pipeline` inlined.

No-op when `pipeline` is already an inline mapping or absent. Lets a resolved trainrun be handed to `apply_trainrun_config` even though the bundled trainruns reference their pipeline by path.

Source code in `cuvis_ai/utils/grpc_workflow.py`

```python
def inline_pipeline_ref(config_dict: dict, *, trainrun_dir: Path | None = None) -> dict:
    """Return ``config_dict`` with a path-referenced ``pipeline`` inlined.

    No-op when ``pipeline`` is already an inline mapping or absent. Lets a
    resolved trainrun be handed to ``apply_trainrun_config`` even though the
    bundled trainruns reference their pipeline by path.
    """
    pipeline = config_dict.get("pipeline")
    if isinstance(pipeline, str):
        return {
            **config_dict,
            "pipeline": resolve_pipeline_ref(pipeline, trainrun_dir=trainrun_dir),
        }
    return config_dict
```

### cli_helpers

CLI and experiment bookkeeping helpers shared across tracking examples.

#### resolve_run_output_dir

```python
resolve_run_output_dir(
    *, output_root, source_path, out_basename
)
```

Resolve the per-run output directory from `--output-dir` and `--out-basename`.

Source code in `cuvis_ai/utils/cli_helpers.py`

```python
def resolve_run_output_dir(
    *,
    output_root: Path,
    source_path: Path,
    out_basename: str | None,
) -> Path:
    """Resolve the per-run output directory from ``--output-dir`` and ``--out-basename``."""
    resolved_basename = source_path.stem
    if out_basename is not None:
        candidate = out_basename.strip()
        if not candidate:
            raise click.BadParameter(
                "--out-basename must not be empty or whitespace only",
                param_hint="--out-basename",
            )
        if "/" in candidate or "\\" in candidate:
            raise click.BadParameter(
                "--out-basename must be a folder name, not a path",
                param_hint="--out-basename",
            )
        resolved_basename = candidate
    return output_root / resolved_basename
```

#### compute_real_fps_from_dataset

```python
compute_real_fps_from_dataset(dataset)
```

Estimate real wall-clock fps from first/last measurement `capture_time`.

Reads timestamps directly from the Cuvis session held by a cu3s predict dataset and returns `(n_frames - 1) / span_seconds`. This is more reliable than `session.fps` (a nominal value) for spectral cameras where per-frame exposure dominates the real capture cadence.

Returns `None` when timestamps are unavailable (missing session/indices, fewer than 2 selected frames, zero span, or any attribute-access failure).

Source code in `cuvis_ai/utils/cli_helpers.py`

```python
def compute_real_fps_from_dataset(dataset: object) -> float | None:
    """Estimate real wall-clock fps from first/last measurement ``capture_time``.

    Reads timestamps directly from the Cuvis session held by a cu3s predict dataset
    and returns ``(n_frames - 1) / span_seconds``. This is more reliable than
    ``session.fps`` (a nominal value) for spectral cameras where per-frame exposure
    dominates the real capture cadence.

    Returns ``None`` when timestamps are unavailable (missing session/indices,
    fewer than 2 selected frames, zero span, or any attribute-access failure).
    """
    session = getattr(dataset, "session", None)
    indices = getattr(dataset, "measurement_indices", None)
    if session is None or indices is None or len(indices) < 2:
        return None
    try:
        first = session.get_measurement(int(indices[0]))
        last = session.get_measurement(int(indices[-1]))
        span = (last.capture_time - first.capture_time).total_seconds()
    except Exception:
        return None
    if span <= 0:
        return None
    return (len(indices) - 1) / span
```

#### resolve_end_frame

```python
resolve_end_frame(*, start_frame, end_frame, max_frames)
```

Reconcile `--end-frame` and `--max-frames` into an effective end-frame index.

Source code in `cuvis_ai/utils/cli_helpers.py`

```python
def resolve_end_frame(
    *,
    start_frame: int,
    end_frame: int,
    max_frames: int | None,
) -> int:
    """Reconcile ``--end-frame`` and ``--max-frames`` into an effective end-frame index."""
    if max_frames is None:
        return end_frame
    if max_frames == -1:
        derived_end = -1
    elif max_frames <= 0:
        raise click.BadParameter("--max-frames must be -1 or positive", param_hint="--max-frames")
    else:
        derived_end = start_frame + max_frames
    if end_frame != -1 and derived_end != -1 and end_frame != derived_end:
        raise click.BadParameter(
            "--end-frame and --max-frames conflict; use one or set consistent values.",
            param_hint="--end-frame",
        )
    return derived_end
```

#### write_experiment_info

```python
write_experiment_info(output_dir, **params)
```

Write an `experiment_info.txt` alongside outputs for traceability.

Source code in `cuvis_ai/utils/cli_helpers.py`

```python
def write_experiment_info(output_dir: Path, **params: object) -> None:
    """Write an ``experiment_info.txt`` alongside outputs for traceability."""
    lines = [
        f"Experiment: {output_dir.name}",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Parameters:",
    ]
    for k, v in params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    (output_dir / "experiment_info.txt").write_text("\n".join(lines), encoding="utf-8")
```

#### append_tracking_metrics

```python
append_tracking_metrics(info_path, tracking_json_path)
```

Append diagnostic track-count metrics from a COCO tracking JSON.

Source code in `cuvis_ai/utils/cli_helpers.py`

```python
def append_tracking_metrics(info_path: Path, tracking_json_path: Path) -> None:
    """Append diagnostic track-count metrics from a COCO tracking JSON."""
    import collections

    try:
        data = json.loads(tracking_json_path.read_text(encoding="utf-8"))
    except Exception:
        return

    annots = data.get("annotations", [])
    frame_ids = [int(img["id"]) for img in data.get("images", [])]
    n_frames = len(frame_ids)
    frame_tracks: dict[int, set[int]] = collections.defaultdict(set)
    all_ids: set[int] = set()
    for a in annots:
        tid = a.get("track_id", -1)
        if tid == -1:
            continue
        frame_tracks[a["image_id"]].add(tid)
        all_ids.add(tid)

    counts = [len(frame_tracks.get(frame_id, set())) for frame_id in frame_ids]
    avg = sum(counts) / len(counts) if counts else 0.0
    mx = max(counts) if counts else 0
    zeros = sum(1 for c in counts if c == 0)

    lines = [
        "Results:",
        f"  frames: {n_frames}",
        f"  unique_track_ids: {len(all_ids)}",
        f"  avg_tracks_per_frame: {avg:.1f}",
        f"  max_tracks_per_frame: {mx}",
        f"  zero_track_frames: {zeros}",
        "",
    ]
    with info_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
```

## Visualization And Drawing Helpers

### vis_helpers

Visualization helper utilities for converting figures and tensors to arrays.

#### fig_to_array

```python
fig_to_array(fig, dpi=150)
```

Convert matplotlib figure to numpy array in RGB format.

This utility handles the conversion of a matplotlib figure to a numpy array by saving it to a BytesIO buffer, loading it with PIL, and converting to a numpy array. The figure is automatically closed after conversion.

Parameters:

| Name  | Type     | Description                                   | Default    |
| ----- | -------- | --------------------------------------------- | ---------- |
| `fig` | `Figure` | The matplotlib figure to convert              | *required* |
| `dpi` | `int`    | Resolution for the saved image (default: 150) | `150`      |

Returns:

| Type      | Description                                                   |
| --------- | ------------------------------------------------------------- |
| `ndarray` | RGB image as numpy array with shape (H, W, 3) and dtype uint8 |

Examples:

```pycon
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [1, 4, 9])
>>> img_array = fig_to_array(fig, dpi=150)
>>> img_array.shape
(height, width, 3)
```

Source code in `cuvis_ai/utils/vis_helpers.py`

```python
def fig_to_array(fig: matplotlib.figure.Figure, dpi: int = 150) -> np.ndarray:
    """Convert matplotlib figure to numpy array in RGB format.

    This utility handles the conversion of a matplotlib figure to a numpy array
    by saving it to a BytesIO buffer, loading it with PIL, and converting to
    a numpy array. The figure is automatically closed after conversion.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert
    dpi : int, optional
        Resolution for the saved image (default: 150)

    Returns
    -------
    np.ndarray
        RGB image as numpy array with shape (H, W, 3) and dtype uint8

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> img_array = fig_to_array(fig, dpi=150)
    >>> img_array.shape
    (height, width, 3)
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img.convert("RGB"))
    buf.close()

    # Close the figure to free memory
    import matplotlib.pyplot as plt

    plt.close(fig)

    return img_array
```

#### tensor_to_uint8

```python
tensor_to_uint8(tensor)
```

Convert float tensor [0, 1] to uint8 [0, 255].

Parameters:

| Name     | Type     | Description                        | Default    |
| -------- | -------- | ---------------------------------- | ---------- |
| `tensor` | `Tensor` | Input tensor with values in [0, 1] | *required* |

Returns:

| Type     | Description                                                           |
| -------- | --------------------------------------------------------------------- |
| `Tensor` | Tensor converted to uint8 in range [0, 255], stays on original device |

Source code in `cuvis_ai/utils/vis_helpers.py`

```python
def tensor_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """Convert float tensor [0, 1] to uint8 [0, 255].

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with values in [0, 1]

    Returns
    -------
    torch.Tensor
        Tensor converted to uint8 in range [0, 255], stays on original device
    """
    return (tensor.clamp(0, 1) * 255).to(torch.uint8)
```

#### tensor_to_numpy

```python
tensor_to_numpy(tensor)
```

Convert torch tensor to numpy array on CPU.

Parameters:

| Name     | Type     | Description                         | Default    |
| -------- | -------- | ----------------------------------- | ---------- |
| `tensor` | `Tensor` | Input tensor (can be on any device) | *required* |

Returns:

| Type      | Description                |
| --------- | -------------------------- |
| `ndarray` | Numpy array representation |

Source code in `cuvis_ai/utils/vis_helpers.py`

```python
def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array on CPU.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor (can be on any device)

    Returns
    -------
    np.ndarray
        Numpy array representation
    """
    return tensor.detach().cpu().numpy()
```

#### create_mask_overlay

```python
create_mask_overlay(
    rgb, mask, alpha=0.4, color=(1.0, 0.0, 0.0)
)
```

Alpha-blend a colored tint on foreground pixels.

Pure PyTorch, no gradients. Works for both single images `[H, W, 3]` and batched `[B, H, W, 3]` thanks to broadcasting.

Parameters:

| Name    | Type                         | Description                                                           | Default           |
| ------- | ---------------------------- | --------------------------------------------------------------------- | ----------------- |
| `rgb`   | `Tensor`                     | RGB image(s) in [0, 1]. Shape [H, W, 3] or [B, H, W, 3].              | *required*        |
| `mask`  | `Tensor`                     | Segmentation mask where > 0 is foreground. Shape [H, W] or [B, H, W]. | *required*        |
| `alpha` | `float`                      | Blend factor for the overlay colour (default: 0.4).                   | `0.4`             |
| `color` | `tuple[float, float, float]` | RGB overlay colour in [0, 1] (default: red (1, 0, 0)).                | `(1.0, 0.0, 0.0)` |

Returns:

| Type     | Description                                                     |
| -------- | --------------------------------------------------------------- |
| `Tensor` | Blended image, same shape and device as rgb, clamped to [0, 1]. |

Source code in `cuvis_ai/utils/vis_helpers.py`

```python
@torch.no_grad()
def create_mask_overlay(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.4,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
    """Alpha-blend a colored tint on foreground pixels.

    Pure PyTorch, no gradients.  Works for both single images ``[H, W, 3]``
    and batched ``[B, H, W, 3]`` thanks to broadcasting.

    Parameters
    ----------
    rgb : torch.Tensor
        RGB image(s) in ``[0, 1]``.  Shape ``[H, W, 3]`` or ``[B, H, W, 3]``.
    mask : torch.Tensor
        Segmentation mask where ``> 0`` is foreground.
        Shape ``[H, W]`` or ``[B, H, W]``.
    alpha : float, optional
        Blend factor for the overlay colour (default: 0.4).
    color : tuple[float, float, float], optional
        RGB overlay colour in ``[0, 1]`` (default: red ``(1, 0, 0)``).

    Returns
    -------
    torch.Tensor
        Blended image, same shape and device as *rgb*, clamped to ``[0, 1]``.
    """
    if mask.device != rgb.device:
        mask = mask.to(rgb.device)
    fg = (mask > 0).unsqueeze(-1).to(dtype=rgb.dtype)  # [..., 1] for channel broadcast
    tint = torch.tensor(color, dtype=rgb.dtype, device=rgb.device)
    return ((1.0 - alpha * fg) * rgb + alpha * fg * tint).clamp(0.0, 1.0)
```

#### object_color

```python
object_color(object_id)
```

Return a deterministic RGB colour for *object_id* (0-255 per channel).

Source code in `cuvis_ai/utils/vis_helpers.py`

```python
def object_color(object_id: int) -> tuple[int, int, int]:
    """Return a deterministic RGB colour for *object_id* (0-255 per channel)."""
    return OBJECT_PALETTE[object_id % len(OBJECT_PALETTE)]
```

#### render_multi_object_overlay

```python
render_multi_object_overlay(
    frame,
    masks,
    *,
    alpha=0.4,
    draw_contours=True,
    draw_ids=True,
    contour_thickness=2,
    font_scale=0.7,
    font_thickness=2,
)
```

Render coloured mask overlays with contours and ID labels onto a frame.

This is the shared rendering path used by both the SAM3 tracking script's built-in overlay output and the standalone `render_tracking_overlay.py`.

Parameters:

| Name                | Type                        | Description                                                                                                                | Default    |
| ------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `frame`             | `ndarray`                   | RGB image, shape (H, W, 3), dtype uint8.                                                                                   | *required* |
| `masks`             | `list[tuple[int, ndarray]]` | List of (object_id, binary_mask) pairs. Each binary_mask has shape (H, W) and dtype bool or uint8 (non-zero = foreground). | *required* |
| `alpha`             | `float`                     | Overlay opacity (default 0.4).                                                                                             | `0.4`      |
| `draw_contours`     | `bool`                      | Draw contour outlines on mask edges (default True).                                                                        | `True`     |
| `draw_ids`          | `bool`                      | Render object ID labels above each mask (default True).                                                                    | `True`     |
| `contour_thickness` | `int`                       | Pixel width of contour lines (default 2).                                                                                  | `2`        |
| `font_scale`        | `float`                     | Legacy text scale knob (default 0.7). Mapped to bitmap font scale.                                                         | `0.7`      |
| `font_thickness`    | `int`                       | Legacy text thickness knob (default 2). Mapped to bitmap font scale.                                                       | `2`        |

Returns:

| Type      | Description                                        |
| --------- | -------------------------------------------------- |
| `ndarray` | Copy of frame with overlays, same shape and dtype. |

Source code in `cuvis_ai/utils/vis_helpers.py`

```python
def render_multi_object_overlay(
    frame: np.ndarray,
    masks: list[tuple[int, np.ndarray]],
    *,
    alpha: float = 0.4,
    draw_contours: bool = True,
    draw_ids: bool = True,
    contour_thickness: int = 2,
    font_scale: float = 0.7,
    font_thickness: int = 2,
) -> np.ndarray:
    """Render coloured mask overlays with contours and ID labels onto a frame.

    This is the shared rendering path used by both the SAM3 tracking script's
    built-in overlay output and the standalone ``render_tracking_overlay.py``.

    Parameters
    ----------
    frame : np.ndarray
        RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
    masks : list[tuple[int, np.ndarray]]
        List of ``(object_id, binary_mask)`` pairs.  Each ``binary_mask`` has
        shape ``(H, W)`` and dtype ``bool`` or ``uint8`` (non-zero = foreground).
    alpha : float
        Overlay opacity (default 0.4).
    draw_contours : bool
        Draw contour outlines on mask edges (default True).
    draw_ids : bool
        Render object ID labels above each mask (default True).
    contour_thickness : int
        Pixel width of contour lines (default 2).
    font_scale : float
        Legacy text scale knob (default 0.7). Mapped to bitmap font scale.
    font_thickness : int
        Legacy text thickness knob (default 2). Mapped to bitmap font scale.

    Returns
    -------
    np.ndarray
        Copy of *frame* with overlays, same shape and dtype.
    """
    # Map legacy cv2 knobs to bitmap-font text scale so callers keep control.
    text_scale = max(
        1,
        int(round(max(0.1, float(font_scale)) * 3.0 + 0.5 * max(0, int(font_thickness) - 1))),
    )

    img_t = torch.from_numpy(frame.copy())
    masks_t = [(int(obj_id), torch.from_numpy(mask > 0)) for obj_id, mask in masks]
    result = overlay_instances(
        img_t,
        masks_t,
        alpha=alpha,
        draw_edges=draw_contours,
        draw_ids=draw_ids,
        edge_thickness=int(contour_thickness),
        text_scale=text_scale,
    )
    return result.numpy()
```

### torch_draw

Pure-torch drawing helpers for uint8 HWC images.

#### mask_edge

```python
mask_edge(mask, thickness=2)
```

Compute edge pixels from a binary mask.

Source code in `cuvis_ai/utils/torch_draw.py`

```python
@torch.no_grad()
def mask_edge(mask: torch.Tensor, thickness: int = 2) -> torch.Tensor:
    """Compute edge pixels from a binary mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape (H, W), got {tuple(mask.shape)}")

    mask_bool = mask.to(torch.bool)
    if not torch.any(mask_bool):
        return torch.zeros_like(mask_bool)

    t = max(1, int(thickness))
    kernel = 2 * t + 1
    inv = (~mask_bool).to(torch.float32).unsqueeze(0).unsqueeze(0)
    dilated_inv = F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=t)
    eroded = dilated_inv.eq(0).squeeze(0).squeeze(0)
    return mask_bool & ~eroded
```

#### draw_box

```python
draw_box(img, box_xyxy, color, thickness=2)
```

Draw rectangle edges in-place on a uint8 HWC image.

Source code in `cuvis_ai/utils/torch_draw.py`

```python
@torch.no_grad()
def draw_box(
    img: torch.Tensor,
    box_xyxy: tuple[int, int, int, int],
    color: torch.Tensor | tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw rectangle edges in-place on a uint8 HWC image."""
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    h, w = int(img.shape[0]), int(img.shape[1])
    if h == 0 or w == 0:
        return

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1 or y2 < y1:
        return

    t = max(1, int(thickness))
    color_t = _as_color_tensor(color, img.device)

    y_top_end = min(h, y1 + t)
    y_bottom_start = max(y1, y2 - t + 1)
    x_left_end = min(w, x1 + t)
    x_right_start = max(x1, x2 - t + 1)

    img[y1:y_top_end, x1 : x2 + 1, :] = color_t
    img[y_bottom_start : y2 + 1, x1 : x2 + 1, :] = color_t
    img[y1 : y2 + 1, x1:x_left_end, :] = color_t
    img[y1 : y2 + 1, x_right_start : x2 + 1, :] = color_t
```

#### draw_text

```python
draw_text(img, x, y, text, color, scale=2, bg=True)
```

Draw bitmap text in-place on a uint8 HWC image.

Source code in `cuvis_ai/utils/torch_draw.py`

```python
@torch.no_grad()
def draw_text(
    img: torch.Tensor,
    x: int,
    y: int,
    text: str,
    color: torch.Tensor | tuple[int, int, int],
    scale: int = 2,
    bg: bool = True,
) -> None:
    """Draw bitmap text in-place on a uint8 HWC image."""
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    glyph = _glyph(text, img.device)
    s = max(1, int(scale))
    if s > 1:
        glyph = glyph.repeat_interleave(s, dim=0).repeat_interleave(s, dim=1)

    gh, gw = int(glyph.shape[0]), int(glyph.shape[1])
    if gh == 0 or gw == 0:
        return

    h, w = int(img.shape[0]), int(img.shape[1])
    x_i, y_i = int(x), int(y)
    color_t = _as_color_tensor(color, img.device)

    if bg:
        pad = max(1, s)
        rx0 = max(0, x_i - pad)
        ry0 = max(0, y_i - pad)
        rx1 = min(w, x_i + gw + pad)
        ry1 = min(h, y_i + gh + pad)
        if rx1 > rx0 and ry1 > ry0:
            region = img[ry0:ry1, rx0:rx1, :].to(torch.float32)
            img[ry0:ry1, rx0:rx1, :] = torch.round(region * 0.25).to(torch.uint8)

    x0 = max(0, x_i)
    y0 = max(0, y_i)
    x1 = min(w, x_i + gw)
    y1 = min(h, y_i + gh)
    if x1 <= x0 or y1 <= y0:
        return

    gx0 = x0 - x_i
    gy0 = y0 - y_i
    gx1 = gx0 + (x1 - x0)
    gy1 = gy0 + (y1 - y0)

    mask_crop = glyph[gy0:gy1, gx0:gx1].to(torch.bool)
    if not torch.any(mask_crop):
        return

    region = img[y0:y1, x0:x1, :]
    region[mask_crop] = color_t
```

#### draw_downward_triangle

```python
draw_downward_triangle(
    img,
    tip_x,
    tip_y,
    width,
    height,
    color,
    *,
    outline_color=None,
    outline_thickness=1,
)
```

Draw a filled downward-pointing isosceles triangle in-place.

Source code in `cuvis_ai/utils/torch_draw.py`

```python
@torch.no_grad()
def draw_downward_triangle(
    img: torch.Tensor,
    tip_x: int,
    tip_y: int,
    width: int,
    height: int,
    color: torch.Tensor | tuple[int, int, int],
    *,
    outline_color: torch.Tensor | tuple[int, int, int] | None = None,
    outline_thickness: int = 1,
) -> None:
    """Draw a filled downward-pointing isosceles triangle in-place."""
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    h, w = int(img.shape[0]), int(img.shape[1])
    if h == 0 or w == 0:
        return

    tri_w = max(1, int(width))
    tri_h = max(1, int(height))
    tip_x_i = int(tip_x)
    tip_y_i = int(tip_y)
    half_w = tri_w / 2.0

    outer = (
        (tip_x_i, tip_y_i),
        (tip_x_i - half_w, tip_y_i - tri_h),
        (tip_x_i + half_w, tip_y_i - tri_h),
    )

    if outline_color is not None:
        _fill_triangle(img, outer, outline_color)

        t = max(0, int(outline_thickness))
        inner_w = tri_w - 2 * t
        inner_h = tri_h - 2 * t
        if inner_w > 0 and inner_h > 0:
            inner_half_w = inner_w / 2.0
            inner_tip_y = tip_y_i - t
            inner = (
                (tip_x_i, inner_tip_y),
                (tip_x_i - inner_half_w, inner_tip_y - inner_h),
                (tip_x_i + inner_half_w, inner_tip_y - inner_h),
            )
            _fill_triangle(img, inner, color)
        return

    _fill_triangle(img, outer, color)
```

#### id_to_color

```python
id_to_color(ids)
```

Map integer IDs to deterministic uint8 RGB colors (hybrid policy).

Source code in `cuvis_ai/utils/torch_draw.py`

```python
@torch.no_grad()
def id_to_color(ids: torch.Tensor) -> torch.Tensor:
    """Map integer IDs to deterministic uint8 RGB colors (hybrid policy)."""
    if ids.ndim != 1:
        raise ValueError(f"Expected ids shape (N,), got {tuple(ids.shape)}")

    ids_i64 = ids.to(torch.int64)
    n = int(ids_i64.numel())
    out = torch.empty((n, 3), dtype=torch.uint8, device=ids.device)
    if n == 0:
        return out

    palette = torch.tensor(_DEFAULT_PALETTE, dtype=torch.uint8, device=ids.device)
    palette_len = int(palette.shape[0])

    in_palette = (ids_i64 >= 0) & (ids_i64 < palette_len)
    if torch.any(in_palette):
        out[in_palette] = palette[ids_i64[in_palette]]

    out_of_palette = ~in_palette
    if torch.any(out_of_palette):
        hashed = ids_i64[out_of_palette] * 1103515245 + 12345
        raw = (
            torch.stack(
                [((hashed >> 16) & 0xFF), ((hashed >> 8) & 0xFF), (hashed & 0xFF)],
                dim=1,
            ).to(torch.float32)
            / 255.0
        )
        biased = (0.35 + 0.65 * raw).clamp(0.0, 1.0)
        out[out_of_palette] = torch.round(biased * 255.0).to(torch.uint8)

    return out
```

#### overlay_instances

```python
overlay_instances(
    image,
    masks,
    *,
    alpha=0.4,
    draw_edges=True,
    draw_ids=True,
    edge_thickness=2,
    text_scale=2,
)
```

Blend instance masks, draw optional edges, and draw optional object IDs.

Source code in `cuvis_ai/utils/torch_draw.py`

```python
@torch.no_grad()
def overlay_instances(
    image: torch.Tensor,
    masks: list[tuple[int, torch.Tensor]],
    *,
    alpha: float = 0.4,
    draw_edges: bool = True,
    draw_ids: bool = True,
    edge_thickness: int = 2,
    text_scale: int = 2,
) -> torch.Tensor:
    """Blend instance masks, draw optional edges, and draw optional object IDs."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(image.shape)} {image.dtype}"
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    out = image.clone()
    if not masks:
        return out

    edge_t = max(1, int(edge_thickness))
    txt_scale = max(1, int(text_scale))
    white = torch.tensor([255, 255, 255], dtype=torch.uint8, device=out.device)

    for obj_id, mask in masks:
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (H, W), got {tuple(mask.shape)}")
        if tuple(mask.shape) != tuple(out.shape[:2]):
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match image shape {tuple(out.shape[:2])}"
            )
        if mask.device != out.device:
            raise ValueError(f"Mask device {mask.device} must match image device {out.device}")

        fg = mask.to(torch.bool)
        if not torch.any(fg):
            continue

        color = id_to_color(torch.tensor([int(obj_id)], device=out.device, dtype=torch.int64))[0]

        if alpha > 0.0:
            current = out[fg].to(torch.float32)
            tint = color.to(torch.float32)
            out[fg] = (
                torch.round((1.0 - alpha) * current + alpha * tint).clamp(0, 255).to(torch.uint8)
            )

        if draw_edges:
            edges = mask_edge(fg, thickness=edge_t)
            out[edges] = color

        if draw_ids:
            ys, xs = torch.where(fg)
            if ys.numel() > 0:
                x_min = int(xs.min().item())
                y_min = int(ys.min().item())
                label = str(int(obj_id))
                label_mask = _glyph(label, out.device)
                if txt_scale > 1:
                    label_mask = label_mask.repeat_interleave(txt_scale, dim=0).repeat_interleave(
                        txt_scale, dim=1
                    )
                label_h = int(label_mask.shape[0])
                pad = max(1, txt_scale)
                text_x = max(0, x_min - 1)
                text_y = max(0, y_min - label_h - 2 * pad)
                draw_text(out, text_x, text_y, label, white, scale=txt_scale, bg=True)

    return out
```

#### draw_sparkline

```python
draw_sparkline(
    img, x1, y1, width, height, values, color, bg_alpha=0.5
)
```

Render a filled area sparkline chart on a uint8 HWC image (in-place).

Draws a mini filled area chart of `values` within the rectangle `(x1, y1, x1+width, y1+height)`. The values are min-max normalized internally; the chart is filled from the curve down to the bottom edge.

Parameters:

| Name       | Type              | Description                                            | Default    |
| ---------- | ----------------- | ------------------------------------------------------ | ---------- |
| `img`      | `Tensor`          | (H, W, 3) uint8 image, modified in-place.              | *required* |
| `x1`       | `int`             | Top-left corner of the sparkline region.               | *required* |
| `y1`       | `int`             | Top-left corner of the sparkline region.               | *required* |
| `width`    | `int`             | Dimensions of the sparkline region in pixels.          | *required* |
| `height`   | `int`             | Dimensions of the sparkline region in pixels.          | *required* |
| `values`   | `Tensor`          | (C,) float — the spectral signature or any 1-D signal. | *required* |
| `color`    | `Tensor or tuple` | RGB color for the filled area.                         | *required* |
| `bg_alpha` | `float`           | Background darkening factor (0=black, 1=no darkening). | `0.5`      |

Source code in `cuvis_ai/utils/torch_draw.py`

```python
@torch.no_grad()
def draw_sparkline(
    img: torch.Tensor,
    x1: int,
    y1: int,
    width: int,
    height: int,
    values: torch.Tensor,
    color: torch.Tensor | tuple[int, int, int],
    bg_alpha: float = 0.5,
) -> None:
    """Render a filled area sparkline chart on a uint8 HWC image (in-place).

    Draws a mini filled area chart of ``values`` within the rectangle
    ``(x1, y1, x1+width, y1+height)``. The values are min-max normalized
    internally; the chart is filled from the curve down to the bottom edge.

    Parameters
    ----------
    img : Tensor
        ``(H, W, 3)`` uint8 image, modified in-place.
    x1, y1 : int
        Top-left corner of the sparkline region.
    width, height : int
        Dimensions of the sparkline region in pixels.
    values : Tensor
        ``(C,)`` float — the spectral signature or any 1-D signal.
    color : Tensor or tuple
        RGB color for the filled area.
    bg_alpha : float
        Background darkening factor (0=black, 1=no darkening).
    """
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    h, w = int(img.shape[0]), int(img.shape[1])
    num_vals = int(values.numel())
    if num_vals < 2 or width < 2 or height < 2:
        return

    # Clamp region to image bounds
    rx1 = max(0, int(x1))
    ry1 = max(0, int(y1))
    rx2 = min(w, int(x1) + int(width))
    ry2 = min(h, int(y1) + int(height))
    if rx2 <= rx1 or ry2 <= ry1:
        return

    rw = rx2 - rx1
    rh = ry2 - ry1

    # Darken background region for readability
    region = img[ry1:ry2, rx1:rx2, :]
    darkened = (region.to(torch.float32) * bg_alpha).clamp(0, 255).to(torch.uint8)
    img[ry1:ry2, rx1:rx2, :] = darkened

    # Min-max normalize values to [0, 1]
    vals = values.to(torch.float32).detach()
    v_min = vals.min()
    v_max = vals.max()
    v_range = v_max - v_min
    if v_range < 1e-12:
        # Flat signal — draw a horizontal line at mid-height
        norm_vals = torch.full_like(vals, 0.5)
    else:
        norm_vals = (vals - v_min) / v_range

    # Map each column to a band index and compute y-position
    color_t = _as_color_tensor(color, img.device)
    for col in range(rw):
        # Map column to band index (linear interpolation)
        band_idx_f = col * (num_vals - 1) / max(1, rw - 1)
        band_lo = int(band_idx_f)
        band_hi = min(band_lo + 1, num_vals - 1)
        frac = band_idx_f - band_lo
        val = float(norm_vals[band_lo]) * (1.0 - frac) + float(norm_vals[band_hi]) * frac

        # y=0 at top of region, y=rh-1 at bottom
        # val=1 → top of region, val=0 → bottom
        curve_y = int(round((1.0 - val) * (rh - 1)))
        curve_y = max(0, min(curve_y, rh - 1))

        # Fill from curve_y down to bottom
        abs_x = rx1 + col
        abs_y_start = ry1 + curve_y
        if abs_y_start < ry2:
            img[abs_y_start:ry2, abs_x, :] = color_t
```

## Color And False-RGB Helpers

### color_spaces

Color space conversion utilities (PyTorch, differentiable).

#### srgb_to_linear

```python
srgb_to_linear(x)
```

Apply inverse sRGB EOTF (companding to linear).

Parameters:

| Name | Type     | Description                                       | Default    |
| ---- | -------- | ------------------------------------------------- | ---------- |
| `x`  | `Tensor` | sRGB values in [0, 1] with arbitrary batch shape. | *required* |

Returns:

| Type     | Description                               |
| -------- | ----------------------------------------- |
| `Tensor` | Linear-light values, same shape as input. |

Source code in `cuvis_ai/utils/color_spaces.py`

```python
def srgb_to_linear(x: Tensor) -> Tensor:
    """Apply inverse sRGB EOTF (companding to linear).

    Parameters
    ----------
    x : Tensor
        sRGB values in [0, 1] with arbitrary batch shape.

    Returns
    -------
    Tensor
        Linear-light values, same shape as input.
    """
    a = 0.055
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1.0 + a)).pow(2.4),
    )
```

#### linear_rgb_to_oklab

```python
linear_rgb_to_oklab(rgb)
```

Convert linear RGB to OKLab.

Parameters:

| Name  | Type     | Description                                                                                                                                                                   | Default    |
| ----- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `rgb` | `Tensor` | Linear RGB tensor with shape [..., 3] in range [0, 1]. Input should not have an sRGB gamma curve applied; use :func:srgb_to_linear first if working with display sRGB values. | *required* |

Returns:

| Type     | Description                                         |
| -------- | --------------------------------------------------- |
| `Tensor` | OKLab tensor [..., 3] where channels are (L, a, b). |

Source code in `cuvis_ai/utils/color_spaces.py`

```python
def linear_rgb_to_oklab(rgb: Tensor) -> Tensor:
    """Convert linear RGB to OKLab.

    Parameters
    ----------
    rgb : Tensor
        Linear RGB tensor with shape ``[..., 3]`` in range [0, 1].
        Input should **not** have an sRGB gamma curve applied;
        use :func:`srgb_to_linear` first if working with display sRGB values.

    Returns
    -------
    Tensor
        OKLab tensor ``[..., 3]`` where channels are (L, a, b).
    """
    # Linear RGB -> LMS
    M1 = rgb.new_tensor(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    lms = rgb @ M1.T  # [..., 3]

    # Cube root with sign handling (robust if values go slightly negative)
    lms_cbrt = torch.sign(lms) * torch.abs(lms).clamp_min(1e-12).pow(1.0 / 3.0)

    # LMS -> OKLab
    M2 = rgb.new_tensor(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return lms_cbrt @ M2.T  # [..., 3]
```

#### rgb_to_oklab

```python
rgb_to_oklab(rgb, assume_srgb=True)
```

Convert RGB to OKLab, optionally handling sRGB input.

Parameters:

| Name          | Type     | Description                                                                                                     | Default    |
| ------------- | -------- | --------------------------------------------------------------------------------------------------------------- | ---------- |
| `rgb`         | `Tensor` | RGB tensor [..., 3] in [0, 1].                                                                                  | *required* |
| `assume_srgb` | `bool`   | If True, apply inverse sRGB gamma first via :func:srgb_to_linear. If False, assume input is already linear RGB. | `True`     |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Tensor` | OKLab tensor [..., 3]. |

Source code in `cuvis_ai/utils/color_spaces.py`

```python
def rgb_to_oklab(rgb: Tensor, assume_srgb: bool = True) -> Tensor:
    """Convert RGB to OKLab, optionally handling sRGB input.

    Parameters
    ----------
    rgb : Tensor
        RGB tensor ``[..., 3]`` in [0, 1].
    assume_srgb : bool
        If ``True``, apply inverse sRGB gamma first via :func:`srgb_to_linear`.
        If ``False``, assume input is already linear RGB.

    Returns
    -------
    Tensor
        OKLab tensor ``[..., 3]``.
    """
    if assume_srgb:
        rgb = srgb_to_linear(rgb)
    return linear_rgb_to_oklab(rgb)
```

### false_rgb_sampling

Helpers for sampled-fixed false-RGB normalization initialization.

#### uniform_sample_positions

```python
uniform_sample_positions(
    total_frames, sample_fraction=0.05
)
```

Return deterministic uniformly spaced indices in `[0, total_frames)`.

Source code in `cuvis_ai/utils/false_rgb_sampling.py`

```python
def uniform_sample_positions(total_frames: int, sample_fraction: float = 0.05) -> list[int]:
    """Return deterministic uniformly spaced indices in ``[0, total_frames)``."""
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    sample_count = max(1, int(math.ceil(total_frames * float(sample_fraction))))
    if sample_count >= total_frames:
        return list(range(total_frames))
    if sample_count == 1:
        return [0]
    return [int((i * (total_frames - 1)) // (sample_count - 1)) for i in range(sample_count)]
```

#### build_statistical_sample_stream

```python
build_statistical_sample_stream(
    predict_ds, sample_positions
)
```

Yield sampled inputs in the format expected by `statistical_initialization`.

Source code in `cuvis_ai/utils/false_rgb_sampling.py`

```python
def build_statistical_sample_stream(
    predict_ds: Any,
    sample_positions: Iterable[int],
) -> Iterable[dict[str, Any]]:
    """Yield sampled inputs in the format expected by ``statistical_initialization``."""
    for pos in sample_positions:
        sample = predict_ds[int(pos)]
        cube = torch.as_tensor(sample["cube"], dtype=torch.float32)
        if cube.ndim != 3:
            raise ValueError(f"Expected sampled cube with shape [H, W, C], got {tuple(cube.shape)}")
        wavelengths = torch.as_tensor(sample["wavelengths"]).flatten().cpu().numpy()
        yield {
            "cube": cube.unsqueeze(0),  # [1, H, W, C]
            "wavelengths": wavelengths,
        }
```

#### initialize_false_rgb_sampled_fixed

```python
initialize_false_rgb_sampled_fixed(
    false_rgb_node, predict_ds, sample_fraction=0.05
)
```

Initialize a false-RGB selector statistically from a deterministic sample.

Source code in `cuvis_ai/utils/false_rgb_sampling.py`

```python
def initialize_false_rgb_sampled_fixed(
    false_rgb_node: Any,
    predict_ds: Any,
    sample_fraction: float = 0.05,
) -> list[int]:
    """Initialize a false-RGB selector statistically from a deterministic sample."""
    sample_positions = uniform_sample_positions(len(predict_ds), sample_fraction=sample_fraction)
    sample_stream = build_statistical_sample_stream(predict_ds, sample_positions)
    false_rgb_node.statistical_initialization(sample_stream)
    return sample_positions
```

## Numerical Utilities

### poisson_inpaint

GPU/CPU Poisson inpainting for channel-last images.

#### poisson_inpaint

```python
poisson_inpaint(image, mask, *, max_iter=1000, tol=1e-06)
```

Inpaint masked pixels by solving the Laplace equation.

Parameters:

| Name       | Type     | Description                                               | Default    |
| ---------- | -------- | --------------------------------------------------------- | ---------- |
| `image`    | `Tensor` | Tensor [H, W, C] on CPU or GPU.                           | *required* |
| `mask`     | `Tensor` | Tensor [H, W] where True marks unknown pixels to inpaint. | *required* |
| `max_iter` | `int`    | Maximum CG iterations.                                    | `1000`     |
| `tol`      | `float`  | CG residual tolerance.                                    | `1e-06`    |

Source code in `cuvis_ai/utils/poisson_inpaint.py`

```python
def poisson_inpaint(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Inpaint masked pixels by solving the Laplace equation.

    Parameters
    ----------
    image
        Tensor ``[H, W, C]`` on CPU or GPU.
    mask
        Tensor ``[H, W]`` where ``True`` marks unknown pixels to inpaint.
    max_iter
        Maximum CG iterations.
    tol
        CG residual tolerance.
    """
    if image.ndim != 3:
        raise ValueError(f"image must be [H, W, C], got shape {tuple(image.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H, W], got shape {tuple(mask.shape)}")
    if image.shape[:2] != tuple(mask.shape):
        raise ValueError(
            "image spatial shape and mask shape must match, got "
            f"{tuple(image.shape[:2])} vs {tuple(mask.shape)}"
        )
    if not image.is_floating_point():
        raise TypeError(f"image must be floating point, got {image.dtype}")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")

    original_dtype = image.dtype
    compute_dtype = image.dtype
    if image.dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32

    image_work = image.to(dtype=compute_dtype)
    mask_bool = mask.to(device=image.device, dtype=torch.bool)

    h, w, channels = image_work.shape
    flat_mask = mask_bool.reshape(-1)
    unknown_flat = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
    n_unknown = int(unknown_flat.numel())

    if n_unknown == 0:
        return image.clone()

    if n_unknown == h * w:
        raise ValueError("mask covers the whole image; Poisson inpainting has no known boundary")

    rows = torch.div(unknown_flat, w, rounding_mode="floor")
    cols = unknown_flat % w

    idx_map = torch.full((h * w,), -1, dtype=torch.long, device=image.device)
    idx_map[unknown_flat] = torch.arange(n_unknown, dtype=torch.long, device=image.device)

    diag = torch.zeros(n_unknown, dtype=compute_dtype, device=image.device)
    rhs = torch.zeros((n_unknown, channels), dtype=compute_dtype, device=image.device)
    image_flat = image_work.reshape(-1, channels)

    row_parts: list[torch.Tensor] = []
    col_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    has_known_boundary = False

    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        n_rows = rows + dr
        n_cols = cols + dc
        valid = (n_rows >= 0) & (n_rows < h) & (n_cols >= 0) & (n_cols < w)
        if not torch.any(valid):
            continue

        diag = diag + valid.to(dtype=compute_dtype)

        local_rows = torch.nonzero(valid, as_tuple=False).squeeze(1)
        neigh_flat = n_rows[valid] * w + n_cols[valid]
        neigh_unknown = idx_map[neigh_flat]

        is_unknown_neighbor = neigh_unknown >= 0
        if torch.any(is_unknown_neighbor):
            off_rows = local_rows[is_unknown_neighbor]
            off_cols = neigh_unknown[is_unknown_neighbor]
            off_vals = torch.full(
                (off_rows.numel(),),
                -1.0,
                dtype=compute_dtype,
                device=image.device,
            )
            row_parts.append(off_rows)
            col_parts.append(off_cols)
            val_parts.append(off_vals)

        is_known_neighbor = ~is_unknown_neighbor
        if torch.any(is_known_neighbor):
            has_known_boundary = True
            known_rows = local_rows[is_known_neighbor]
            known_flat = neigh_flat[is_known_neighbor]
            rhs.index_add_(0, known_rows, image_flat[known_flat])

    if not has_known_boundary:
        raise ValueError("mask has no known neighboring pixels; Poisson system is singular")

    diag_rows = torch.arange(n_unknown, dtype=torch.long, device=image.device)
    row_parts.append(diag_rows)
    col_parts.append(diag_rows)
    val_parts.append(diag)

    mat_rows = torch.cat(row_parts, dim=0)
    mat_cols = torch.cat(col_parts, dim=0)
    mat_vals = torch.cat(val_parts, dim=0)

    matrix = torch.sparse_coo_tensor(
        indices=torch.stack((mat_rows, mat_cols), dim=0),
        values=mat_vals,
        size=(n_unknown, n_unknown),
        dtype=compute_dtype,
        device=image.device,
    ).coalesce()

    solved = _cg_solve(matrix, rhs, max_iter=max_iter, tol=tol)

    result = image_work.clone()
    result.reshape(-1, channels)[unknown_flat] = solved
    return result.to(dtype=original_dtype)
```

### welford

Numerically stable streaming statistics using Welford's online algorithm.

Provides a reusable `WelfordAccumulator` that incrementally computes mean, variance, covariance, and correlation from batches of data. The accumulator is an `nn.Module` so that `.to(device)` propagates to its internal buffers automatically when the parent node is moved.

Reference: Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products." Technometrics, 4(3), 419-420.

```text
Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979). "Updating formulae
and a pairwise algorithm for computing sample variances."
```

#### WelfordAccumulator

```python
WelfordAccumulator(n_features, *, track_covariance=False)
```

Bases: `Module`

Numerically stable streaming mean / variance / covariance.

All internal accumulation happens in **float64** for numerical stability. Property getters return **float32** tensors.

The buffers are registered with `persistent=False` so they are **excluded** from `state_dict()` — only the parent node's own buffers (`mu`, `cov`, …) are serialised. This is correct because the accumulator is transient training state that is consumed by `finalize()` inside `statistical_initialization()`.

Parameters:

| Name               | Type   | Description                                                                                                                                                 | Default    |
| ------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `n_features`       | `int`  | Number of features (channels) per sample.                                                                                                                   | *required* |
| `track_covariance` | `bool` | If True, maintain the full (C, C) covariance matrix (O(C²) memory and compute per update). If False (default), only per-feature variance is tracked (O(C)). | `False`    |

Source code in `cuvis_ai/utils/welford.py`

```python
def __init__(self, n_features: int, *, track_covariance: bool = False) -> None:
    super().__init__()
    self._n_features = n_features
    self._track_cov = track_covariance

    self.register_buffer("_n", torch.tensor(0, dtype=torch.long), persistent=False)
    self.register_buffer(
        "_mean", torch.zeros(n_features, dtype=torch.float64), persistent=False
    )
    if track_covariance:
        self.register_buffer(
            "_M2",
            torch.zeros(n_features, n_features, dtype=torch.float64),
            persistent=False,
        )
    else:
        self.register_buffer(
            "_M2", torch.zeros(n_features, dtype=torch.float64), persistent=False
        )
```

##### count

```python
count
```

Total number of samples accumulated so far.

##### mean

```python
mean
```

Per-feature mean, shape `(C,)`, float32.

##### var

```python
var
```

Per-feature sample variance, shape `(C,)`, float32.

##### std

```python
std
```

Per-feature standard deviation, shape `(C,)`, float32.

##### cov

```python
cov
```

Sample covariance matrix, shape `(C, C)`, float32.

Raises:

| Type           | Description                                                                        |
| -------------- | ---------------------------------------------------------------------------------- |
| `RuntimeError` | If track_covariance was not enabled or fewer than 2 samples have been accumulated. |

##### corr

```python
corr
```

Absolute correlation matrix, shape `(C, C)`, float32.

Raises:

| Type           | Description                                                                        |
| -------------- | ---------------------------------------------------------------------------------- |
| `RuntimeError` | If track_covariance was not enabled or fewer than 2 samples have been accumulated. |

##### reset

```python
reset()
```

Zero all accumulators so the instance can be reused.

Source code in `cuvis_ai/utils/welford.py`

```python
def reset(self) -> None:
    """Zero all accumulators so the instance can be reused."""
    self._n.zero_()
    self._mean.zero_()
    self._M2.zero_()
```

##### update

```python
update(X)
```

Incorporate a batch of samples.

Parameters:

| Name | Type     | Description                                                                                                                                                                                  | Default    |
| ---- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `X`  | `Tensor` | Sample matrix of shape (N, C) where N is the number of samples and C equals n_features. A 1-D tensor of shape (N,) is accepted when n_features == 1 and is reshaped to (N, 1) automatically. | *required* |

Source code in `cuvis_ai/utils/welford.py`

```python
@torch.no_grad()
def update(self, X: Tensor) -> None:
    """Incorporate a batch of samples.

    Parameters
    ----------
    X : Tensor
        Sample matrix of shape ``(N, C)`` where *N* is the number of
        samples and *C* equals ``n_features``.  A 1-D tensor of shape
        ``(N,)`` is accepted when ``n_features == 1`` and is reshaped
        to ``(N, 1)`` automatically.
    """
    if X.ndim == 1:
        X = X.unsqueeze(-1)

    X = X.to(dtype=torch.float64)
    m = X.shape[0]
    if m == 0:
        return

    mean_b = X.mean(dim=0)  # (C,)

    if self._track_cov:
        centered = X - mean_b
        M2_b = centered.T @ centered  # (C, C)
    else:
        M2_b = ((X - mean_b) ** 2).sum(dim=0)  # (C,)

    n = int(self._n.item())
    if n == 0:
        self._n.fill_(m)
        self._mean.copy_(mean_b)
        self._M2.copy_(M2_b)
    else:
        tot = n + m
        delta = mean_b - self._mean
        self._mean.add_(delta * (m / tot))
        if self._track_cov:
            self._M2.add_(M2_b + torch.outer(delta, delta) * (n * m / tot))
        else:
            self._M2.add_(M2_b + delta**2 * (n * m / tot))
        self._n.fill_(tot)
```

### deep_svdd_factory

Deep SVDD Channel Configuration Utilities.

This module provides utilities for inferring channel counts after bandpass filtering for Deep SVDD networks. This is useful for automatically configuring network architectures based on the data pipeline's preprocessing steps.

See Also

cuvis_ai.node.preprocessors : Bandpass filtering nodes cuvis_ai.anomaly.deep_svdd : Deep SVDD anomaly detection

#### ChannelConfig

```python
ChannelConfig(num_channels, in_channels)
```

Configuration for network channel counts.

Stores the number of input and output channels for network layers, typically determined after bandpass filtering.

Attributes:

| Name           | Type  | Description                              |
| -------------- | ----- | ---------------------------------------- |
| `num_channels` | `int` | Total number of channels in the network. |
| `in_channels`  | `int` | Number of input channels to the network. |

#### infer_channels_after_bandpass

```python
infer_channels_after_bandpass(datamodule, bandpass_cfg)
```

Infer post-bandpass channel count from a sample batch.

Parameters:

| Name           | Type     | Description                                                                       | Default    |
| -------------- | -------- | --------------------------------------------------------------------------------- | ---------- |
| `datamodule`   | `object` | Datamodule with a train_dataloader() method returning batches with "wavelengths". | *required* |
| `bandpass_cfg` | `object` | Config with min_wavelength_nm and max_wavelength_nm fields.                       | *required* |

Returns:

| Type            | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| `ChannelConfig` | num_channels and in_channels set to the filtered channel count. |

Source code in `cuvis_ai/utils/deep_svdd_factory.py`

```python
def infer_channels_after_bandpass(datamodule, bandpass_cfg) -> ChannelConfig:
    """Infer post-bandpass channel count from a sample batch.

    Parameters
    ----------
    datamodule : object
        Datamodule with a train_dataloader() method returning batches with "wavelengths".
    bandpass_cfg : object
        Config with min_wavelength_nm and max_wavelength_nm fields.

    Returns
    -------
    ChannelConfig
        num_channels and in_channels set to the filtered channel count.
    """
    sample_batch = next(iter(datamodule.train_dataloader()))
    wavelengths = sample_batch["wavelengths"]
    keep_mask = wavelengths >= bandpass_cfg.min_wavelength_nm
    if bandpass_cfg.max_wavelength_nm is not None:
        keep_mask = keep_mask & (wavelengths <= bandpass_cfg.max_wavelength_nm)
    num_channels_after_bandpass = int(keep_mask.sum().item())
    return ChannelConfig(
        num_channels=num_channels_after_bandpass, in_channels=num_channels_after_bandpass
    )
```

## Related Pages

- [Client Connections & Sessions](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/deployment/client-connections/index.md)
- [Client Workflows & Error Handling](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/deployment/client-workflows/index.md)
- [Profiling & Performance](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.1/workflows/profiling/index.md)
