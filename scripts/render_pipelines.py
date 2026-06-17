"""Render a transparent PNG of every pipeline YAML in configs/pipeline/.

Independent of the orchestrator smoke — this is a pure host-venv
operation: load each YAML, build the pipeline via PipelineBuilder
against the locally-installed plugins, and ask the existing
PipelineVisualizer for a graphviz-rendered PNG with
``bgcolor=transparent``.

Output lands under ``pipeline_renders/`` mirroring the source
directory structure, so the diagrams sit next to a parallel
breadcrumb of the YAMLs.

Run from the repo root:

    uv run python scripts/render_pipelines.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
PIPELINE_DIR = CONFIGS_DIR / "pipeline"
OUTPUT_ROOT = REPO_ROOT / "pipeline_renders"


@dataclass
class Result:
    yaml: str
    step: str  # "build" | "render" | "ok"
    duration_s: float
    output: str | None = None
    error: str | None = None


def _gather_yamls() -> list[Path]:
    return sorted(PIPELINE_DIR.rglob("*.yaml"))


def _register_plugins(plugin_names: list[str], yaml_path: Path) -> None:
    """Ensure the YAML's declared plugins are registered in the NodeRegistry.

    Reads the host's ``configs/plugins/<name>.yaml`` manifest entries
    so the import path matches what the gRPC server would do. The
    host venv must already have each plugin's Python package
    installed (this script does not compose its own venv).
    """
    from cuvis_ai_schemas.plugin import load_plugin_manifest

    from cuvis_ai_core.utils.node_registry import NodeRegistry

    registry = NodeRegistry()
    for plugin_name in plugin_names:
        manifest_file = CONFIGS_DIR / "plugins" / f"{plugin_name}.yaml"
        if not manifest_file.exists():
            logger.warning(
                f"{yaml_path.name}: plugin manifest '{plugin_name}.yaml' not found; "
                "skipping registration — the import may still succeed if installed"
            )
            continue
        manifest = load_plugin_manifest(manifest_file)
        if manifest.name != plugin_name:
            logger.warning(
                f"{yaml_path.name}: manifest {manifest_file.name} declares name "
                f"'{manifest.name}', expected '{plugin_name}'"
            )
            continue
        try:
            registry.register_plugin(plugin_name, manifest.model_dump())
        except Exception as exc:
            logger.warning(
                f"{yaml_path.name}: register_plugin('{plugin_name}') raised "
                f"{type(exc).__name__}: {exc} — continuing"
            )


def render_one(yaml_path: Path) -> Result:
    from cuvis_ai_core.pipeline.factory import PipelineBuilder
    from cuvis_ai_core.pipeline.visualizer import PipelineVisualizer
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    started = time.perf_counter()
    try:
        yaml_dict = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        plugin_names = list(yaml_dict.get("plugins", []) or [])
        _register_plugins(plugin_names, yaml_path)

        pipeline = PipelineBuilder(node_registry=NodeRegistry()).build_from_config(yaml_dict)
    except Exception as exc:
        return Result(
            yaml=str(yaml_path),
            step="build",
            duration_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    try:
        rel = yaml_path.relative_to(PIPELINE_DIR).with_suffix(".png")
        output_path = OUTPUT_ROOT / rel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer = PipelineVisualizer(pipeline)
        visualizer.render_graphviz(
            output_path=output_path,
            format="png",
            graph_attributes={"bgcolor": "transparent"},
        )
    except Exception as exc:
        return Result(
            yaml=str(yaml_path),
            step="render",
            duration_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    return Result(
        yaml=str(yaml_path),
        step="ok",
        duration_s=time.perf_counter() - started,
        output=str(output_path),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--filter", default="")
    args = parser.parse_args(argv)

    yamls = _gather_yamls()
    if args.filter:
        yamls = [y for y in yamls if args.filter in str(y).replace("\\", "/")]
    if args.limit:
        yamls = yamls[: args.limit]

    logger.info(f"Rendering {len(yamls)} pipeline YAML(s) -> {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []
    for idx, yaml_path in enumerate(yamls, 1):
        rel = yaml_path.relative_to(REPO_ROOT)
        logger.info(f"[{idx:2d}/{len(yamls)}] {rel}")
        result = render_one(yaml_path)
        results.append(result)
        marker = "OK" if result.step == "ok" else f"FAIL@{result.step}"
        logger.info(f"[{idx:2d}/{len(yamls)}] {rel} -> {marker} ({result.duration_s:.1f}s)")
        if result.output:
            logger.info(f"        wrote {result.output}")
        if result.error:
            logger.error(f"        {result.error.splitlines()[0][:280]}")

    passed = sum(1 for r in results if r.step == "ok")
    failed = len(results) - passed
    logger.info(f"=== SUMMARY === {passed}/{len(results)} passed, {failed} failed")

    report = REPO_ROOT / "render_pipelines.report.json"
    report.write_text(
        json.dumps(
            [
                {
                    "yaml": str(Path(r.yaml).relative_to(REPO_ROOT)),
                    "step": r.step,
                    "duration_s": r.duration_s,
                    "output": r.output,
                    "error": r.error,
                }
                for r in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Report: {report}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
