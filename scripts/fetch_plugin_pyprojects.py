"""Fetch each git-sourced plugin's ``pyproject.toml`` by tag for the registry audit.

Reads ``cuvis_ai/configs/plugins/*.yaml``; for every plugin entry with ``repo`` + ``tag``,
downloads its ``pyproject.toml`` from the GitHub raw URL into
``<out>/<name>@<tag>/`` (``--out``, default ``.plugin_pyprojects/``), the directory
``audit-plugin-deps --check plugins --pyproject-cache <out>`` reads. Local-``path`` and untagged entries are skipped (the
audit host-checks those separately, in their own repo). Fetch failures are
warnings, not errors: the audit reports an uncached plugin as a note.

Used by ``.github/workflows/registry_compat.yml``. Run from the repo root.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

import yaml

_GITHUB = re.compile(r"(?:git@github\.com:|https?://github\.com/)(.+?)(?:\.git)?$")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch each pinned plugin's pyproject.toml.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".plugin_pyprojects"),
        help="Directory that receives <name>@<tag>/pyproject.toml; pass the same "
        "directory to audit-plugin-deps --pyproject-cache.",
    )
    cache: Path = parser.parse_args(argv).out
    catalog = Path("cuvis_ai/configs/plugins")
    # One file = one plugin: the source lives in the top-level `name` / `repo` / `tag`
    # keys (not a nested `plugins:` mapping). Local-`path` and untagged entries are skipped.
    for manifest in sorted(catalog.glob("*.yaml")):
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        name, repo, tag = data.get("name"), data.get("repo"), data.get("tag")
        if not (repo and tag):
            print(f"skip {name or manifest.stem}: local-path or non-tagged plugin")
            continue
        match = _GITHUB.match(repo)
        if not match:
            print(f"skip {name}: unsupported repo url {repo}")
            continue
        url = f"https://raw.githubusercontent.com/{match.group(1)}/{tag}/pyproject.toml"
        dest = cache / f"{name}@{tag}"
        dest.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dest / "pyproject.toml")
            print(f"fetched {name}@{tag} <- {url}")
        except Exception as exc:  # noqa: BLE001 - audit warns on a missing pyproject
            print(f"WARN {name}@{tag}: fetch failed ({exc})")


if __name__ == "__main__":
    main()
