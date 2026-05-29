"""One-shot smoke for the orchestrator path against every pipeline YAML.

Runs the full ``CreateSession → SetSessionSearchPaths → ResolveConfig →
LoadPipeline → CloseSession`` chain in-process (no separate gRPC
server) with real child-runtime spawning (no in-memory orchestrator).
Each YAML is its own composed venv + spawned child; the cache shared
across runs so a second invocation is fast.

Intent: prove the orchestrator path materialises every checked-in
pipeline before item 01 is declared merge-ready. We do not exercise
``Inference`` because it would require dataset access; ``LoadPipeline``
is enough to validate plugin resolve → env compose → child spawn →
pipeline build end-to-end.

Run from the cuvis-ai host repo root:

    uv run python scripts/smoke_22_yamls.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import yaml
from loguru import logger

# Override the run-cache root so the smoke doesn't pollute ~/.cuvis_runs/.
os.environ.setdefault("CUVIS_RUN_CACHE_DIR", str(Path(tempfile.gettempdir()) / "cuvis_smoke_cache"))

from cuvis_ai_core.grpc import orchestrator_bridge
from cuvis_ai_core.grpc.service import CuvisAIService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
PIPELINE_DIR = CONFIGS_DIR / "pipeline"


@dataclass
class Result:
    yaml: str
    step: str  # "resolve" | "load" | "render" | "ok"
    duration_s: float
    error: str | None = None
    render_path: str | None = None


def _gather_yamls() -> list[Path]:
    return sorted(PIPELINE_DIR.rglob("*.yaml"))


def _relative_path_arg(yaml_path: Path) -> str:
    """Return the path Hydra expects for ResolveConfig.

    Path is interpreted relative to the session's search paths
    (``configs/`` here), so we include the ``pipeline/`` segment.
    """
    return str(yaml_path.relative_to(CONFIGS_DIR).with_suffix("")).replace("\\", "/")


def run_one(service: CuvisAIService, yaml_path: Path) -> Result:
    context = MagicMock()
    started = time.perf_counter()

    session_resp = service.CreateSession(cuvis_ai_pb2.CreateSessionRequest(), context)
    session_id = session_resp.session_id

    try:
        # Search paths point at the configs/ root so Hydra can resolve
        # pipeline + plugin defaults exactly as the host server does.
        path_resp = service.SetSessionSearchPaths(
            cuvis_ai_pb2.SetSessionSearchPathsRequest(
                session_id=session_id,
                search_paths=[str(CONFIGS_DIR)],
                append=False,
            ),
            context,
        )
        if not path_resp.success:
            return Result(
                yaml=str(yaml_path),
                step="search_paths",
                duration_s=time.perf_counter() - started,
                error="SetSessionSearchPaths returned success=False",
            )

        # The 22 checked-in pipeline YAMLs do not use Hydra defaults
        # (verified via `grep -l "^defaults:"`), so we hand the raw
        # YAML straight to LoadPipeline rather than round-tripping
        # through ResolveConfig (which Hydra-packages by directory and
        # would need a yaml-per-config_root setup).
        config_bytes = json.dumps(yaml.safe_load(yaml_path.read_text(encoding="utf-8"))).encode(
            "utf-8"
        )

        # Step: build the pipeline through the orchestrator (compose + spawn).
        load_resp = service.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(
                    config_bytes=config_bytes,
                ),
            ),
            context,
        )
        if not load_resp.success:
            return Result(
                yaml=str(yaml_path),
                step="load",
                duration_s=time.perf_counter() - started,
                error=f"LoadPipeline returned success=False (code={context.set_code.call_args}, details={context.set_details.call_args})",
            )

        return Result(
            yaml=str(yaml_path),
            step="ok",
            duration_s=time.perf_counter() - started,
        )
    except Exception as exc:
        return Result(
            yaml=str(yaml_path),
            step="exception",
            duration_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
    finally:
        try:
            service.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=session_id),
                context,
            )
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N yamls (0 = all). Useful for incremental smoke.",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Substring match on the yaml relative path (e.g. 'rx').",
    )
    args = parser.parse_args(argv)

    # Force the real orchestrator (no in-memory swap from conftest).
    orchestrator_bridge.reset_orchestrator()

    yamls = _gather_yamls()
    if args.filter:
        yamls = [y for y in yamls if args.filter in str(y).replace("\\", "/")]
    if args.limit:
        yamls = yamls[: args.limit]

    logger.info(f"Discovered {len(yamls)} pipeline YAML(s) under {PIPELINE_DIR}")
    logger.info(f"CUVIS_RUN_CACHE_DIR={os.environ.get('CUVIS_RUN_CACHE_DIR')}")

    session_manager = SessionManager()
    service = CuvisAIService(session_manager)

    results: list[Result] = []
    for idx, yaml_path in enumerate(yamls, 1):
        rel = yaml_path.relative_to(REPO_ROOT)
        logger.info(f"[{idx:2d}/{len(yamls)}] starting {rel}")
        result = run_one(service, yaml_path)
        results.append(result)
        marker = "OK" if result.step == "ok" else f"FAIL@{result.step}"
        logger.info(f"[{idx:2d}/{len(yamls)}] {rel} -> {marker} ({result.duration_s:.1f}s)")
        if result.error:
            head = result.error.splitlines()[0]
            logger.error(f"        {head[:280]}")

    passed = sum(1 for r in results if r.step == "ok")
    failed = len(results) - passed
    logger.info(f"=== SUMMARY === {passed}/{len(results)} passed, {failed} failed")

    if failed:
        logger.info("--- Failures ---")
        for r in results:
            if r.step != "ok":
                rel = Path(r.yaml).relative_to(REPO_ROOT)
                logger.error(f"{rel}  [step={r.step}, dur={r.duration_s:.1f}s]")
                if r.error:
                    for line in r.error.splitlines()[:6]:
                        logger.error(f"  {line}")

    # Persist JSON for follow-up triage.
    report = REPO_ROOT / "smoke_22_yamls.report.json"
    report.write_text(
        json.dumps(
            [
                {
                    "yaml": str(Path(r.yaml).relative_to(REPO_ROOT)),
                    "step": r.step,
                    "duration_s": r.duration_s,
                    "error": r.error,
                }
                for r in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Report written: {report}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
