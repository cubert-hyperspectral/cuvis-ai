"""Backfill ``plugins:`` into every pipeline YAML under ``configs/pipeline/``.

One-off mechanical migration for ALL-5349 item 02 (Phase 2): walks every
pipeline YAML, runs the auto-resolver against ``configs/plugins/``, and
writes the resolved plugin set back as a top-level ``plugins:`` list
positioned directly after ``metadata:``.

Idempotent: a yaml that already has ``plugins:`` is skipped (with a log
line). Re-running the script on the same tree is a no-op once every
file has been backfilled.

Run from the cuvis-ai repo root::

    uv run python scripts/backfill_pipeline_plugins.py [--dry-run]

ALL-5349 Phase 4 note: after Phase 4 ships, ``resolve_pipeline_plugins``
hard-fails on a missing ``plugins:`` field. This script imports the
``suggest_plugins_field`` / ``reorder_pipeline_with_plugins`` helpers
from ``cuvis_ai_core.utils.plugin_fixer`` (same heuristic, non-fatal
path) so the backfill still works on un-migrated yamls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from cuvis_ai_schemas.pipeline import PipelineConfig
from loguru import logger

from cuvis_ai_core.utils.plugin_fixer import (
    reorder_pipeline_with_plugins,
    suggest_plugins_field,
)


def _backfill_one(yaml_path: Path, plugins_dir: Path, dry_run: bool) -> bool:
    """Backfill a single pipeline YAML. Returns True if the file was modified."""
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        logger.warning(f"Skipping {yaml_path}: top-level is not a mapping")
        return False

    if raw.get("plugins"):
        logger.debug(f"Skipping {yaml_path}: 'plugins:' already present")
        return False

    pipeline_config = PipelineConfig.load_from_file(yaml_path)
    plugin_names, rebuilt = suggest_plugins_field(pipeline_config, raw, [plugins_dir])
    # `suggest_plugins_field` already runs reorder under the hood, but
    # the contract pins canonical leading order so the call is
    # idempotent. Keep the explicit reorder for clarity / future-proofing.
    rebuilt = reorder_pipeline_with_plugins(rebuilt, plugin_names)

    if dry_run:
        logger.info(f"[dry-run] {yaml_path} → plugins: {plugin_names}")
        return True

    yaml_path.write_text(
        yaml.safe_dump(rebuilt, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    logger.success(f"Backfilled {yaml_path} → plugins: {plugin_names}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=Path("configs"),
        help="Directory containing 'pipeline/' and 'plugins/' subtrees (default: ./configs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing files",
    )
    args = parser.parse_args(argv)

    pipeline_root = args.configs_root / "pipeline"
    plugins_dir = args.configs_root / "plugins"

    if not pipeline_root.is_dir():
        logger.error(f"Pipeline root not found: {pipeline_root}")
        return 1
    if not plugins_dir.is_dir():
        logger.error(f"Plugins catalog dir not found: {plugins_dir}")
        return 1

    yaml_files = sorted(pipeline_root.rglob("*.yaml"))
    logger.info(f"Found {len(yaml_files)} pipeline YAML(s) under {pipeline_root}")

    modified = 0
    failed: list[tuple[Path, str]] = []
    for yaml_path in yaml_files:
        try:
            if _backfill_one(yaml_path, plugins_dir, args.dry_run):
                modified += 1
        except Exception as exc:  # noqa: BLE001 — surface all errors per file
            failed.append((yaml_path, str(exc)))
            logger.error(f"Failed {yaml_path}: {exc}")

    summary = "would modify" if args.dry_run else "modified"
    logger.info(f"{summary} {modified}/{len(yaml_files)} file(s)")
    if failed:
        logger.error(f"{len(failed)} file(s) failed:")
        for path, err in failed:
            logger.error(f"  {path}: {err}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
