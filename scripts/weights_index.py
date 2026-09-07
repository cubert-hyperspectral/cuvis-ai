"""Regenerate or verify ``cuvis_ai/configs/plugins/weights.index.json``.

The index is ``download-model list --json --plugins-dir cuvis_ai/configs/plugins`` written
deterministically (cuvis-ai-core's ``index_json``): the registry core builds from the
``weights:`` blocks of the plugin manifests plus its own built-in rows, with no plugin
imported. CuvisNEXT's installer generator reads it with CMake's ``string(JSON)`` and its
test fixture is the same bytes, so the file is committed and
``tests/plugins/test_weights_declarations.py`` fails when it is stale.

Usage::

    uv run python -m scripts.weights_index          # rewrite the file
    uv run python -m scripts.weights_index --check  # exit 1 when the committed file is stale

Run it after ``emit_metadata`` has refreshed a manifest's ``weights:`` block (a plugin
pin bump), before tagging a cuvis-ai release.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cuvis_ai_core.data.model_weights import ModelWeights, index_json

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = REPO_ROOT / "cuvis_ai" / "configs" / "plugins"
INDEX_PATH = PLUGINS_DIR / "weights.index.json"


def generate(plugins_dir: Path = PLUGINS_DIR) -> str:
    """Return the index text for ``plugins_dir``: core's built-in rows plus every manifest block.

    The registry is reset first so a plugin that happens to be importable in this
    environment does not turn its rows into ``source: plugin`` entries; the committed
    file describes what an environment without the plugins sees.
    """
    ModelWeights.reset()
    ModelWeights.load_manifests([plugins_dir])
    return index_json(ModelWeights.list_payload())


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: write the index, or with ``--check`` compare and report staleness."""
    parser = argparse.ArgumentParser(description="Regenerate or verify weights.index.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the committed file with a fresh generation instead of writing; exit 1 when stale.",
    )
    args = parser.parse_args(argv)
    fresh = generate()
    rel = INDEX_PATH.relative_to(REPO_ROOT).as_posix()
    if args.check:
        current = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else ""
        if current != fresh:
            print(f"{rel} is stale; run `uv run python -m scripts.weights_index`", file=sys.stderr)
            return 1
        print(f"{rel} is up to date")
        return 0
    INDEX_PATH.write_text(fresh, encoding="utf-8", newline="\n")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
