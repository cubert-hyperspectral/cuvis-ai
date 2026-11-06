"""Version policy for the mike docs on gh-pages: one folder per minor release.

A release deploys into its minor folder (`0.22`), titled with the full patch
version (`0.22.3`), so a patch release replaces its minor's docs. Two commands:

- `check-deploy X.Y.Z` runs before `mike deploy` and refuses a version older
  than the one already published for its minor, so a re-run of an old release
  cannot roll the docs back.
- `prune` removes the patch folders of older releases. It keeps the newest
  version of each minor and turns every other patch of that minor into a
  symlink alias of it, so the old URLs keep resolving. Without `--apply` it only
  prints the plan; with `--apply` it runs `mike delete` and `mike alias` on the
  local gh-pages branch (never `--push`) and verifies the result.

Usage
-----
    python scripts/docs_versions.py check-deploy 0.22.3 --versions-json versions.json
    python scripts/docs_versions.py prune --versions-json versions.json [--apply]

See `.github/actions/deploy-docs/action.yml` and
`.github/workflows/docs-prune-patch-versions.yml`.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from docs_postprocess_gh_pages import VERSION_RE, version_key

PATCH_RE = re.compile(r"^\d+\.\d+\.\d+$")


class DocsVersionsError(Exception):
    """A versions.json state or request this policy refuses."""


@dataclass(frozen=True)
class PruneGroup:
    """One minor: the version that stays and the patch folders that become its aliases."""

    keeper: str
    pruned: tuple[str, ...]


def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def _checked_versions(versions: list[dict]) -> list[str]:
    names = []
    for entry in versions:
        name = entry.get("version")
        if not isinstance(name, str) or not VERSION_RE.match(name):
            raise DocsVersionsError(f"versions.json has an invalid version entry: {name!r}")
        names.append(name)
    return names


def check_deploy(versions: list[dict], version: str) -> None:
    """Refuse to deploy `version` when its minor already holds a newer patch."""
    if not PATCH_RE.match(version):
        raise DocsVersionsError(f"{version!r} is not a plain X.Y.Z version")
    minor = _minor(version)
    published = []
    for entry in versions:
        name = entry.get("version")
        if name == minor:
            title = entry.get("title")
            if not isinstance(title, str) or not PATCH_RE.match(title) or _minor(title) != minor:
                raise DocsVersionsError(
                    f"the {minor} folder has title {title!r}, expected an {minor}.Z version"
                )
            published.append(title)
        elif isinstance(name, str) and PATCH_RE.match(name) and _minor(name) == minor:
            published.append(name)
    if not published:
        return
    newest = max(published, key=version_key)
    if version_key(version) < version_key(newest):
        raise DocsVersionsError(
            f"refusing to deploy {version}: {newest} is already published for minor {minor}"
        )


def plan_prune(versions: list[dict]) -> list[PruneGroup]:
    """Keep the newest version of each minor; every other one of that minor is pruned."""
    by_minor: dict[str, list[str]] = defaultdict(list)
    for name in _checked_versions(versions):
        by_minor[_minor(name)].append(name)
    aliases = {entry["version"]: entry.get("aliases", []) for entry in versions}

    groups = []
    for names in by_minor.values():
        keeper, *pruned = sorted(names, key=version_key, reverse=True)
        if not pruned:
            continue
        for name in pruned:
            if aliases[name]:
                raise DocsVersionsError(
                    f"{name} carries aliases {aliases[name]!r} but would be pruned "
                    f"in favour of {keeper}"
                )
        groups.append(PruneGroup(keeper, tuple(sorted(pruned, key=version_key))))
    return sorted(groups, key=lambda g: version_key(g.keeper), reverse=True)


def apply_plan(groups: list[PruneGroup]) -> None:
    """Delete the pruned folders and alias them to their keepers, on the local branch only."""
    if not groups:
        return
    _run(["mike", "delete", *(name for g in groups for name in g.pruned)])
    for g in groups:
        _run(["mike", "alias", "--alias-type", "symlink", g.keeper, *g.pruned])


def verify(versions: list[dict], groups: list[PruneGroup]) -> None:
    """One version per minor, and every pruned name is an alias of its keeper."""
    minors: dict[str, list[str]] = defaultdict(list)
    for name in _checked_versions(versions):
        minors[_minor(name)].append(name)
    doubled = {m: names for m, names in minors.items() if len(names) > 1}
    if doubled:
        raise DocsVersionsError(f"more than one version per minor: {doubled!r}")
    aliases = {entry["version"]: set(entry.get("aliases", [])) for entry in versions}
    for g in groups:
        missing = set(g.pruned) - aliases.get(g.keeper, set())
        if missing:
            raise DocsVersionsError(f"{g.keeper} lacks the aliases {sorted(missing)!r}")


def check_symlink_aliases(groups: list[PruneGroup], branch: str) -> None:
    """Every pruned name on `branch` is a git symlink to its keeper's folder."""
    for g in groups:
        for name in g.pruned:
            listed = _run(["git", "ls-tree", branch, "--", name], capture_output=True, text=True)
            mode = listed.stdout.split(maxsplit=1)[0] if listed.stdout else ""
            target = ""
            if mode == "120000":
                shown = _run(["git", "show", f"{branch}:{name}"], capture_output=True, text=True)
                target = shown.stdout.strip()
            if target != g.keeper:
                raise DocsVersionsError(
                    f"{name} on {branch} is not a symlink to {g.keeper} (mode {mode!r}, "
                    f"target {target!r})"
                )


def format_plan(versions: list[dict], groups: list[PruneGroup]) -> str:
    if not groups:
        return f"nothing to prune: {len(versions)} versions, one per minor"
    lines = [f"keep {g.keeper}; delete and alias to it: {' '.join(g.pruned)}" for g in groups]
    pruned = sum(len(g.pruned) for g in groups)
    lines.append(
        f"{pruned} patch folders to delete in {len(groups)} minors; "
        f"{len(versions) - pruned} versions remain"
    )
    return "\n".join(lines)


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


@click.group()
def cli() -> None:
    """Version policy for the mike docs on gh-pages."""


@cli.command("check-deploy")
@click.argument("version")
@click.option(
    "--versions-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="versions.json of the gh-pages branch the deploy writes to.",
)
def check_deploy_command(version: str, versions_json: Path) -> None:
    """Refuse VERSION when its minor already holds a newer patch."""
    try:
        check_deploy(_load(versions_json), version)
    except DocsVersionsError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{version} may be deployed into {_minor(version)}")


@cli.command("prune")
@click.option(
    "--versions-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="versions.json of the gh-pages branch before the prune.",
)
@click.option("--apply", is_flag=True, help="Run mike on the local branch (never pushes).")
@click.option("--branch", default="gh-pages", show_default=True, help="Branch mike writes.")
def prune_command(versions_json: Path, apply: bool, branch: str) -> None:
    """Print the prune plan; with --apply, run it and verify the branch."""
    versions = _load(versions_json)
    try:
        groups = plan_prune(versions)
        click.echo(format_plan(versions, groups))
        if not apply or not groups:
            return
        apply_plan(groups)
        shown = _run(["git", "show", f"{branch}:versions.json"], capture_output=True, text=True)
        after = json.loads(shown.stdout)
        verify(after, groups)
        check_symlink_aliases(groups, branch)
    except DocsVersionsError as exc:
        raise click.ClickException(str(exc)) from exc
    pruned = sum(len(g.pruned) for g in groups)
    click.echo(
        f"verified: {len(after)} versions on {branch}, one per minor; "
        f"{pruned} patch names are symlinks to their kept version"
    )


if __name__ == "__main__":
    cli()
