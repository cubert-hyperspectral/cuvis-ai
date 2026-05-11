"""One-shot script: rewire internal docs links after the IA restructure.

Walks every Markdown file under ``docs/``, parses each relative link
``[text](path.md...)``, resolves it to an absolute docs-relative path,
looks it up in the move-mapping, and rewrites the link as a relative
path from the source file to the new target.

Run from the repo root::

    python scripts/rewire_internal_links.py

The script is idempotent — re-running it after success is a no-op.
This is one-shot infrastructure; delete after the restructure lands
unless we expect another major IA shuffle soon.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

MOVE_MAP: dict[str, str] = {
    # user-guide/
    "user-guide/installation.md": "get-started/installation.md",
    "user-guide/quickstart.md": "get-started/quickstart.md",
    "user-guide/configuration.md": "reference/configuration/index.md",
    # concepts/
    "concepts/overview.md": "concepts/index.md",
    "concepts/node-system-deep-dive.md": "concepts/node.md",
    "concepts/port-system-deep-dive.md": "concepts/port.md",
    "concepts/pipeline-lifecycle.md": "concepts/pipeline.md",
    "concepts/two-phase-training.md": "concepts/training.md",
    # how-to/
    "how-to/restore-pipeline-trainrun.md": "workflows/restore-pipeline.md",
    "how-to/build-pipeline-python.md": "workflows/build-pipeline-python.md",
    "how-to/build-pipeline-yaml.md": "workflows/build-pipeline-yaml.md",
    "how-to/monitoring-and-viz.md": "workflows/monitoring.md",
    "how-to/profiling.md": "workflows/profiling.md",
    "how-to/index.md": "workflows/index.md",
    "how-to/add-builtin-node.md": "reference/contributing/add-builtin-node.md",
    # node-catalog/
    "node-catalog/data-nodes.md": "catalogs/nodes/data-nodes.md",
    "node-catalog/preprocessing.md": "catalogs/nodes/preprocessing.md",
    "node-catalog/selectors.md": "catalogs/nodes/selectors.md",
    "node-catalog/statistical.md": "catalogs/nodes/statistical.md",
    "node-catalog/loss-metrics.md": "catalogs/nodes/loss-metrics.md",
    "node-catalog/visualization.md": "catalogs/nodes/visualization.md",
    "node-catalog/output.md": "catalogs/nodes/output.md",
    "node-catalog/utility.md": "catalogs/nodes/utility.md",
    "node-catalog/node-catalog-plugins.md": "catalogs/nodes/external.md",
    "node-catalog/index.md": "catalogs/nodes/index.md",
    # config/
    "config/index.md": "reference/configuration/index.md",
    "config/config-groups.md": "reference/configuration/config-groups.md",
    "config/trainrun-schema.md": "reference/configuration/trainrun-schema.md",
    "config/hydra-basics.md": "reference/configuration/hydra-basics.md",
    "config/hydra-inheritance.md": "reference/configuration/hydra-inheritance.md",
    "config/hydra-sweeps.md": "reference/configuration/hydra-sweeps.md",
    # api/
    "api/index.md": "reference/python-api/index.md",
    "api/pipeline.md": "reference/python-api/pipeline.md",
    "api/ports.md": "reference/python-api/ports.md",
    "api/utilities.md": "reference/python-api/utilities.md",
    # plugin-system/
    "plugin-system/development.md": "reference/plugin-development/guide.md",
    "plugin-system/index.md": "reference/plugin-development/overview.md",
    "plugin-system/overview.md": "reference/plugin-development/overview.md",
    # development/
    "development/contributing.md": "reference/contributing/contributing.md",
    "development/docstrings.md": "reference/contributing/docstrings.md",
    "development/documentation-guidelines.md": "reference/contributing/docs-style.md",
    "development/git-hooks.md": "reference/contributing/git-hooks.md",
    "development/index.md": "reference/contributing/index.md",
    # grpc/ + deployment/
    "grpc/index.md": "deployment/index.md",
    "grpc/overview.md": "deployment/overview.md",
    "grpc/api-session.md": "deployment/api/session.md",
    "grpc/api-config.md": "deployment/api/config.md",
    "grpc/api-pipeline.md": "deployment/api/pipeline.md",
    "grpc/api-training-inference.md": "deployment/api/training-inference.md",
    "grpc/api-types-errors.md": "deployment/api/types-errors.md",
    "grpc/client-connections.md": "deployment/client-connections.md",
    "grpc/client-workflows.md": "deployment/client-workflows.md",
    "grpc/sequence-diagrams.md": "deployment/sequence-diagrams.md",
    "deployment/grpc_deployment.md": "deployment/grpc-deployment.md",
    # use_cases/
    "use_cases/rx-statistical.md": "tutorials/statistical/rx-anomaly.md",
    "use_cases/channel-selector.md": "tutorials/statistical/channel-selector.md",
    "use_cases/blood-perfusion.md": "tutorials/statistical/blood-perfusion.md",
    "use_cases/deep-svdd-gradient.md": "tutorials/gradient/deep-svdd.md",
    "use_cases/adaclip-workflow.md": "tutorials/gradient/adaclip.md",
    "use_cases/grpc-workflow.md": "deployment/grpc-workflow.md",
    "use_cases/index.md": "tutorials/index.md",
}

LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s#]+)(#[^)\s]*)?\)")

# Reverse: every new path → its old docs-relative path. Lets us figure
# out where a *source* file used to live and resolve link targets in
# that historical context.
REVERSE_MOVE_MAP: dict[str, str] = {new: old for old, new in MOVE_MAP.items()}


def _resolve_link(source: Path, target: str) -> Path | None:
    """Resolve a relative Markdown link to an absolute path under DOCS_ROOT.

    Returns None for absolute (http://, mailto:, /…, #anchor) or non-md links.
    """
    if target.startswith(("http://", "https://", "mailto:", "/", "#", "data:")):
        return None
    if not target.endswith(".md"):
        return None
    abs_target = (source.parent / target).resolve()
    try:
        return abs_target.relative_to(DOCS_ROOT.resolve())
    except ValueError:
        return None


def _new_relative(source: Path, new_docs_relative: str) -> str:
    new_abs = (DOCS_ROOT / new_docs_relative).resolve()
    rel = os.path.relpath(new_abs, source.parent.resolve())
    return Path(rel).as_posix()


def rewire_file(path: Path) -> int:
    original = path.read_text(encoding="utf-8")
    changes = 0

    # Figure out where this source file used to live so we can resolve
    # relative links the way they were originally authored.
    source_docs_rel = path.resolve().relative_to(DOCS_ROOT.resolve()).as_posix()
    old_source_rel = REVERSE_MOVE_MAP.get(source_docs_rel, source_docs_rel)
    old_source_path = DOCS_ROOT / old_source_rel

    def repl(match: re.Match[str]) -> str:
        nonlocal changes
        text, target, anchor = match.group(1), match.group(2), match.group(3) or ""
        if target.startswith(("http://", "https://", "mailto:", "/", "#", "data:")):
            return match.group(0)
        if not target.endswith(".md"):
            return match.group(0)

        # 1. Resolve as authored from the CURRENT (new) source location.
        try:
            authored_now = _resolve_link(path, target)
        except (OSError, ValueError):
            authored_now = None

        # 2. If the authored target already resolves to a real file, keep it.
        if authored_now is not None and (DOCS_ROOT / authored_now).exists():
            return match.group(0)

        candidate_keys: list[str] = []
        if authored_now is not None:
            candidate_keys.append(authored_now.as_posix())

        # 3. Resolve as if the source file were still at its OLD location.
        try:
            authored_old_abs = (old_source_path.parent / target).resolve()
            authored_old_rel = authored_old_abs.relative_to(DOCS_ROOT.resolve()).as_posix()
            candidate_keys.append(authored_old_rel)
        except (OSError, ValueError):
            pass

        # 4. Strip leading "../" — last-resort match for paths authored
        #    relative to docs/ root.
        stripped = target
        while stripped.startswith("../"):
            stripped = stripped[3:]
        candidate_keys.append(stripped)

        for key in candidate_keys:
            if key in MOVE_MAP:
                new_target = _new_relative(path, MOVE_MAP[key])
                changes += 1
                return f"[{text}]({new_target}{anchor})"
        return match.group(0)

    new_content = LINK_RE.sub(repl, original)
    if changes:
        path.write_text(new_content, encoding="utf-8")
    return changes


def main() -> int:
    total = 0
    files_touched = 0
    for md in sorted(DOCS_ROOT.rglob("*.md")):
        changes = rewire_file(md)
        if changes:
            files_touched += 1
            total += changes
            print(f"  {md.relative_to(REPO_ROOT)}: {changes} link(s) rewired")
    print(f"\nRewired {total} link(s) across {files_touched} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
