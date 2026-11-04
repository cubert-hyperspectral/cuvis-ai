"""CLI: check a deploy folder against its canonical pipeline on real frames.

Loads both the canonical pipeline and its deploy yaml the way ``restore-pipeline``
does, runs each on the same frames and reports, per output port of the exported span:
the max abs diff against the canonical node's output, the pixel decisions differing
when a decider port exists downstream, and the per-frame median time. Frames come from
a data module's predict split (``--data-module`` / ``--data-arg``, the way
``export-pipeline`` takes them) or from ``.npz`` files (``--frame``, one array per
batch key).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import yaml
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai_core.utils.restore import predict_batches, restore_pipeline


def _load_frame(path: Path, device: str | None) -> dict[str, torch.Tensor]:
    """One frame from an ``.npz`` file: one array per batch key (``cube``,
    ``wavelengths``, ...), the way ``restore-pipeline`` / ``export-pipeline`` read
    examples."""
    with np.load(path) as archive:
        frame = {key: torch.from_numpy(archive[key]) for key in archive.files}
    if device:
        frame = {k: v.to(device) for k, v in frame.items()}
    return frame


def _default_canonical_node(deploy_yaml_path: Path, span_node: str) -> str:
    """The node a deploy yaml's span replaced, read from the sidecar next to the ONNX
    file the span node names (``<onnx_path>.json``), via ``span.outputs[0].port``."""
    config = yaml.safe_load(deploy_yaml_path.read_text(encoding="utf-8"))
    nodes = {n["name"]: n for n in config.get("nodes", [])}
    span = nodes.get(span_node)
    if span is None:
        raise click.ClickException(
            f"{deploy_yaml_path}: no node named {span_node!r} to read the sidecar from."
        )
    onnx_path = span.get("hparams", {}).get("onnx_path")
    if not onnx_path:
        raise click.ClickException(
            f"{deploy_yaml_path}: node {span_node!r} has no onnx_path hparam."
        )
    sidecar_path = (deploy_yaml_path.parent / onnx_path).with_suffix(".onnx.json")
    if not sidecar_path.is_file():
        raise click.ClickException(f"{sidecar_path}: sidecar not found next to the deploy yaml.")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    outputs = sidecar.get("span", {}).get("outputs") or []
    if not outputs:
        raise click.ClickException(f"{sidecar_path}: the sidecar names no span outputs.")
    port = outputs[0]["port"]
    node_name = port.split(".", 1)[0]
    return node_name


def _run(pipeline, batch: dict[str, Any], device: str) -> tuple[dict, float]:
    """One forward pass; returns the pipeline outputs and the wall time in seconds."""
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
    if device == "cuda":
        torch.cuda.synchronize()
    return outputs, time.perf_counter() - t0


def _output_ports(deploy_yaml_path: Path, span_node: str) -> list[str]:
    """The exported span's output port names (bare, e.g. ``scores``), read from the
    sidecar the way :func:`_default_canonical_node` locates it."""
    config = yaml.safe_load(deploy_yaml_path.read_text(encoding="utf-8"))
    nodes = {n["name"]: n for n in config.get("nodes", [])}
    span = nodes[span_node]
    onnx_path = span["hparams"]["onnx_path"]
    sidecar_path = (deploy_yaml_path.parent / onnx_path).with_suffix(".onnx.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return [entry["name"] for entry in sidecar["span"]["outputs"]]


def _find_decider_port(pipeline, span_node: str) -> str | None:
    """The first downstream node with a ``decisions`` output port, if any; its node
    name, for comparing pixel decisions between the canonical and deploy pipelines."""
    for node in pipeline.nodes:
        if node.name == span_node:
            continue
        if "decisions" in getattr(node, "OUTPUT_SPECS", {}):
            return node.name
    return None


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max abs diff between two tensors, as floats regardless of dtype."""
    a64 = a.detach().to(torch.float64) if a.is_floating_point() else a.detach()
    b64 = b.detach().to(torch.float64) if b.is_floating_point() else b.detach()
    return float((a64.to(torch.float64) - b64.to(torch.float64)).abs().max())


@click.command(name="compare-deploy")
@click.option(
    "--canonical",
    "canonical_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The canonical pipeline yaml.",
)
@click.option(
    "--deploy",
    "deploy_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The deploy yaml (the one export-pipeline wrote next to the ONNX file).",
)
@click.option(
    "--plugins-dir",
    "plugins_dirs",
    multiple=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="A directory of plugin manifests. Repeatable.",
)
@click.option(
    "--data-module",
    default=None,
    help="A data module name (its DATA_MODULE_NAME, e.g. cu3s) whose predict split "
    "provides the frames; its plugin is loaded from --plugins-dir.",
)
@click.option(
    "--data-arg",
    "data_args",
    multiple=True,
    metavar="KEY=VALUE",
    help="An argument of the selected data module. Repeatable.",
)
@click.option(
    "--frame",
    "frame_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A frame: an .npz file with one array per batch key (cube, wavelengths, ...). Repeatable.",
)
@click.option(
    "--num-frames",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="How many frames to compare, from --data-module or the --frame list.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Where both pipelines and the frames run.",
)
@click.option(
    "--span-node",
    default="exported_span",
    show_default=True,
    help="The deploy yaml's exported-span node name.",
)
@click.option(
    "--canonical-node",
    default=None,
    help="The canonical node whose outputs the span replaces. Default: read from the "
    "sidecar's span.outputs.",
)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the json report.",
)
def main(
    canonical_path: Path,
    deploy_path: Path,
    plugins_dirs: tuple[Path, ...],
    data_module: str | None,
    data_args: tuple[str, ...],
    frame_files: tuple[Path, ...],
    num_frames: int,
    device: str,
    span_node: str,
    canonical_node: str | None,
    out_path: Path,
) -> None:
    """Compare a deploy folder against its canonical pipeline on the same frames."""
    if not frame_files and not data_module:
        raise click.UsageError(
            "Pass --frame FILE (repeatable) or --data-module NAME to provide frames."
        )

    parsed_args: dict[str, str] = {}
    for pair in data_args:
        if "=" not in pair:
            raise click.UsageError(f"--data-arg must be KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        parsed_args[key.strip()] = value

    plugin_dirs: list[str | Path] | None = [str(p) for p in plugins_dirs] or None
    load_device = None if device == "auto" else device

    canonical = restore_pipeline(canonical_path, device=device, plugins_dirs=plugin_dirs)
    deploy = restore_pipeline(deploy_path, device=device, plugins_dirs=plugin_dirs)

    resolved_canonical_node = canonical_node or _default_canonical_node(deploy_path, span_node)
    output_ports = _output_ports(deploy_path, span_node)
    decider_node = _find_decider_port(deploy, span_node)

    if frame_files:
        frames = [_load_frame(p, load_device) for p in frame_files[:num_frames]]
    else:
        frames = predict_batches(
            canonical_path,
            data_module or "",
            data_args=parsed_args,
            plugins_dirs=plugin_dirs,
            device=load_device,
        )
        frames = list(frames)[:num_frames]

    rows: list[dict[str, Any]] = []
    port_max_diff: dict[str, float] = dict.fromkeys(output_ports, 0.0)
    decisions_differ_total = 0
    times_canonical: list[float] = []
    times_deploy: list[float] = []

    for i, batch in enumerate(frames):
        canon_out, t_canon = _run(canonical, batch, device)
        deploy_out, t_deploy = _run(deploy, batch, device)

        row: dict[str, Any] = {
            "frame": i,
            "t_canonical_s": round(t_canon, 4),
            "t_deploy_s": round(t_deploy, 4),
        }
        for port in output_ports:
            canon_value = canon_out.get((resolved_canonical_node, port))
            deploy_value = deploy_out.get((span_node, port))
            if canon_value is None or deploy_value is None:
                continue
            diff = _max_abs_diff(canon_value, deploy_value)
            row[f"{port}_max_abs_diff"] = diff
            port_max_diff[port] = max(port_max_diff[port], diff)

        if decider_node is not None:
            canon_decisions = canon_out.get((decider_node, "decisions"))
            deploy_decisions = deploy_out.get((decider_node, "decisions"))
            if canon_decisions is not None and deploy_decisions is not None:
                differ = int((canon_decisions != deploy_decisions).sum())
                row["decisions_differ"] = differ
                decisions_differ_total += differ

        rows.append(row)
        times_canonical.append(t_canon)
        times_deploy.append(t_deploy)

    summary = {
        "canonical": str(canonical_path),
        "deploy": str(deploy_path),
        "canonical_node": resolved_canonical_node,
        "span_node": span_node,
        "decider_node": decider_node,
        "frames": len(rows),
        "device": device,
        "max_abs_diff": port_max_diff,
        "decisions_differ_total": decisions_differ_total,
        "median_s": {
            "canonical": round(statistics.median(times_canonical), 4) if times_canonical else None,
            "deploy": round(statistics.median(times_deploy), 4) if times_deploy else None,
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    click.echo(f"Frames compared: {summary['frames']}")
    for port, diff in port_max_diff.items():
        click.echo(f"  {port}: max abs diff {diff:.2e}")
    if decider_node is not None:
        click.echo(f"  decisions ({decider_node}): {decisions_differ_total} pixels differ")
    click.echo(
        f"  median time: canonical {summary['median_s']['canonical']}s, "
        f"deploy {summary['median_s']['deploy']}s"
    )
    click.echo(f"Report: {out_path}")


if __name__ == "__main__":
    main()
