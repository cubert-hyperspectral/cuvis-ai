"""Generate the searchable node catalog at ``/catalogs/nodes/index.md``.

Two data sources, both used at mkdocs build time:

* **Built-in nodes** are imported live from the local ``cuvis_ai.node`` package
  (already installed in the docs venv) and introspected via
  ``cls.get_category()`` / ``cls.get_tags()``. This handles inheritance — a
  subclass that doesn't redefine ``_category`` still reports its parent's
  value.

* **Plugin nodes** are read statically from source via ``ast``. Each
  ``(plugin_path, dotted_class_name)`` entry in
  ``docs/data/plugin_sources.yaml`` resolves to a ``.py`` file we parse
  without importing it — so the docs build never pulls in torch /
  ultralytics / SAM3 / etc.

Output: a single ``catalogs/nodes/index.md`` rendered as a list of
collapsible rows. Each row's body either includes a mkdocstrings
``:::`` block (built-ins, full docstring + signature) or the AST-extracted
class docstring + GitHub source link (plugins). The per-category
sub-pages that used to live alongside this one have been removed —
this page is the entire catalog.
"""

from __future__ import annotations

import ast
import inspect
import logging
import pkgutil
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import mkdocs_gen_files
import yaml
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.extensions.ui.node_display import TAG_STYLES

log = logging.getLogger("generate_node_catalog")

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_SOURCES = REPO_ROOT / "docs" / "data" / "plugin_sources.yaml"
BUILTIN_PACKAGE = "cuvis_ai.node"


@dataclass
class NodeEntry:
    """One node, source-agnostic, ready to render as a collapsible row."""

    name: str
    dotted_path: str
    category: str
    tags: list[str]
    summary: str
    is_plugin: bool
    plugin_name: str | None = None
    repo_url: str | None = None
    full_docstring: str = ""
    search_text: str = field(default="", repr=False)


def _first_doc_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _is_node_subclass(obj: Any) -> bool:
    if not inspect.isclass(obj):
        return False
    try:
        from cuvis_ai_core.node.node import Node  # noqa: PLC0415

        return issubclass(obj, Node) and obj is not Node
    except Exception:
        return False


def collect_builtin_nodes() -> list[NodeEntry]:
    """Walk ``cuvis_ai.node`` and yield every concrete Node subclass."""
    entries: list[NodeEntry] = []
    seen: set[str] = set()

    try:
        pkg = import_module(BUILTIN_PACKAGE)
    except Exception as exc:  # pragma: no cover — docs venv must have cuvis_ai
        log.warning("could not import %s: %s", BUILTIN_PACKAGE, exc)
        return entries

    for mod_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{BUILTIN_PACKAGE}."):
        try:
            mod = import_module(mod_info.name)
        except Exception as exc:
            log.warning("skipping module %s: %s", mod_info.name, exc)
            continue
        for _name, obj in vars(mod).items():
            if not _is_node_subclass(obj):
                continue
            if obj.__module__ != mod_info.name:
                continue  # avoid re-export duplicates
            dotted = f"{obj.__module__}.{obj.__name__}"
            if dotted in seen:
                continue
            seen.add(dotted)
            try:
                cat = obj.get_category().value
            except Exception:
                cat = NodeCategory.UNSPECIFIED.value
            try:
                tags = sorted(t.value for t in obj.get_tags())
            except Exception:
                tags = []
            entries.append(
                NodeEntry(
                    name=obj.__name__,
                    dotted_path=dotted,
                    category=cat,
                    tags=tags,
                    summary=_first_doc_line(inspect.getdoc(obj)),
                    is_plugin=False,
                )
            )
    return entries


_VALID_CATEGORIES = {c.value for c in NodeCategory}
_VALID_TAGS = {t.value for t in NodeTag}


def _ast_enum_value(node: ast.AST, enum_name: str, valid: set[str]) -> str | None:
    """Resolve ``EnumName.MEMBER`` to the enum's ``.value`` string."""
    if not isinstance(node, ast.Attribute):
        return None
    if not (isinstance(node.value, ast.Name) and node.value.id == enum_name):
        return None
    member = node.attr
    try:
        enum_cls = NodeCategory if enum_name == "NodeCategory" else NodeTag
        value = enum_cls[member].value
    except KeyError:
        log.warning("unknown %s member: %s", enum_name, member)
        return None
    if value not in valid:
        return None
    return value


def _extract_tag_set(rhs: ast.AST) -> list[str]:
    """Pull NodeTag values out of ``frozenset({...})`` / ``{...}`` / list literals."""
    items: list[ast.AST]
    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id == "frozenset":
        if not rhs.args:
            return []
        inner = rhs.args[0]
        if isinstance(inner, (ast.Set, ast.List, ast.Tuple)):
            items = list(inner.elts)
        else:
            return []
    elif isinstance(rhs, (ast.Set, ast.List, ast.Tuple)):
        items = list(rhs.elts)
    else:
        return []
    out: list[str] = []
    for elt in items:
        v = _ast_enum_value(elt, "NodeTag", _VALID_TAGS)
        if v is not None:
            out.append(v)
    return sorted(set(out))


def _extract_class_metadata(class_node: ast.ClassDef) -> tuple[str, list[str], str, str]:
    category = NodeCategory.UNSPECIFIED.value
    tags: list[str] = []
    full_doc = ast.get_docstring(class_node) or ""
    summary = _first_doc_line(full_doc)
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                if target.id == "_category":
                    v = _ast_enum_value(stmt.value, "NodeCategory", _VALID_CATEGORIES)
                    if v is not None:
                        category = v
                elif target.id == "_tags":
                    tags = _extract_tag_set(stmt.value)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.value is not None
        ):
            if stmt.target.id == "_category":
                v = _ast_enum_value(stmt.value, "NodeCategory", _VALID_CATEGORIES)
                if v is not None:
                    category = v
            elif stmt.target.id == "_tags":
                tags = _extract_tag_set(stmt.value)
    return category, tags, summary, full_doc


def _resolve_plugin_source(plugin_path: Path, dotted_class: str) -> tuple[Path, str] | None:
    """Map ``pkg.sub.module.ClassName`` -> ``(<plugin_path>/pkg/sub/module.py, ClassName)``."""
    parts = dotted_class.split(".")
    if len(parts) < 2:
        return None
    class_name = parts[-1]
    module_parts = parts[:-1]
    file_path = plugin_path.joinpath(*module_parts).with_suffix(".py")
    if not file_path.exists():
        return None
    return file_path, class_name


def collect_plugin_nodes() -> list[NodeEntry]:
    entries: list[NodeEntry] = []
    if not PLUGIN_SOURCES.exists():
        log.info("no plugin_sources.yaml found at %s", PLUGIN_SOURCES)
        return entries
    spec = yaml.safe_load(PLUGIN_SOURCES.read_text(encoding="utf-8")) or {}
    for plugin in spec.get("plugins", []):
        plugin_name = plugin.get("name", "<unnamed>")
        rel_path = plugin.get("path")
        repo_url = plugin.get("repo_url")
        if not rel_path:
            log.warning("plugin %s missing path; skipping", plugin_name)
            continue
        plugin_root = (REPO_ROOT / rel_path).resolve()
        if not plugin_root.exists():
            log.warning("plugin %s path does not exist: %s", plugin_name, plugin_root)
            continue
        for dotted in plugin.get("classes", []):
            resolved = _resolve_plugin_source(plugin_root, dotted)
            if resolved is None:
                log.warning("plugin %s class %s: source file not found", plugin_name, dotted)
                continue
            file_path, class_name = resolved
            try:
                tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            except SyntaxError as exc:
                log.warning("plugin %s: cannot parse %s: %s", plugin_name, file_path, exc)
                continue
            class_node = next(
                (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name),
                None,
            )
            if class_node is None:
                log.warning(
                    "plugin %s: class %s not found in %s", plugin_name, class_name, file_path
                )
                continue
            category, tags, summary, full_doc = _extract_class_metadata(class_node)
            entries.append(
                NodeEntry(
                    name=class_name,
                    dotted_path=dotted,
                    category=category,
                    tags=tags,
                    summary=summary,
                    full_docstring=full_doc,
                    is_plugin=True,
                    plugin_name=plugin_name,
                    repo_url=repo_url,
                )
            )
    return entries


def _short_label(tag: str) -> str:
    try:
        return TAG_STYLES[NodeTag(tag)]["short_label"]
    except (KeyError, ValueError):
        return tag


def _render_card(entry: NodeEntry) -> str:
    chips = "".join(
        f'<span class="tag-chip" data-tag="{t}" title="{t}">{_short_label(t)}</span>'
        for t in entry.tags
    )
    plugin_pill = (
        f'<span class="plugin-pill" title="From plugin {entry.plugin_name}">{entry.plugin_name}</span>'
        if entry.is_plugin
        else ""
    )
    repo_link = (
        f'<a class="card-repo" href="{entry.repo_url}" title="Open plugin repo" rel="noopener">↗</a>'
        if entry.is_plugin and entry.repo_url
        else ""
    )
    summary_text = entry.summary.replace("<", "&lt;").replace(">", "&gt;") if entry.summary else ""
    module_path = entry.dotted_path.rsplit(".", 1)[0]

    summary_html = (
        f'<img class="row-icon" src="../../images/node-categories/{entry.category}.svg" '
        f'alt="{entry.category}" title="{entry.category}" loading="lazy">'
        f'<span class="row-header">'
        f'<code class="row-name">{entry.name}</code>'
        f'<span class="row-module">{module_path}</span>'
        f'<span class="row-category category-chip" data-category="{entry.category}">'
        f"{entry.category}</span>"
        f'<span class="row-tags">{chips}</span>'
        f'<span class="row-meta">{plugin_pill}{repo_link}</span>'
        f"</span>"
        f'<span class="row-summary">{summary_text}</span>'
    )

    body = _render_body(entry)
    return (
        f'<details class="node-row" markdown="1" '
        f'data-category="{entry.category}" '
        f'data-tags="{" ".join(entry.tags)}" '
        f'data-source="{"plugin" if entry.is_plugin else "builtin"}" '
        f'data-search="{entry.search_text}">\n'
        f"<summary>{summary_html}</summary>\n\n"
        f"{body}\n"
        f"</details>\n"
    )


def _render_body(entry: NodeEntry) -> str:
    if entry.is_plugin:
        doc = entry.full_docstring.strip() or entry.summary
        repo_line = (
            f"\n[View source on GitHub]({entry.repo_url}){{ .row-source }}\n"
            if entry.repo_url
            else ""
        )
        return f'<div class="row-body" markdown="1">\n\n{doc}\n{repo_line}\n</div>'
    return (
        f'<div class="row-body" markdown="1">\n\n'
        f"::: {entry.dotted_path}\n"
        f"    options:\n"
        f"      show_root_heading: true\n"
        f"      heading_level: 4\n\n"
        f"</div>"
    )


def _build_search_text(entry: NodeEntry) -> str:
    parts = [entry.name, entry.dotted_path, entry.category, entry.summary]
    parts.extend(entry.tags)
    if entry.plugin_name:
        parts.append(entry.plugin_name)
    return " ".join(parts).lower().replace('"', "")


def _render_index_page(entries: list[NodeEntry]) -> str:
    categories_present = sorted({e.category for e in entries})
    tags_present = sorted({t for e in entries for t in e.tags})

    cat_chips = "".join(
        f'<button type="button" class="filter-chip category-chip" data-category="{c}" title="{c}">'
        f'<img class="chip-icon" src="../../images/node-categories/{c}.svg" alt="" aria-hidden="true">'
        f'<span class="chip-label">{c}</span>'
        f"</button>"
        for c in categories_present
    )
    tag_chips = "".join(
        f'<button type="button" class="filter-chip tag-chip" data-tag="{t}" '
        f'title="{t}">{_short_label(t)}</button>'
        for t in tags_present
    )

    rows = "\n\n".join(_render_card(e) for e in entries)

    return f"""---
hide:
  - toc
---

# Nodes Catalog

Every node available in cuvis-ai pipelines, in one place. Built-in nodes ship
with the `cuvis_ai` package; plugin nodes come from separately-installable
manifests — see [Plugin Development](../../reference/plugin-development/overview.md).

<div class="node-filter">
<input type="search" id="node-filter-search" placeholder="Search by name, tag, module…" autocomplete="off">
<div class="node-filter-row">
<span class="filter-label">Category</span>
<div class="filter-chips" id="node-filter-categories">{cat_chips}</div>
</div>
<div class="node-filter-row">
<span class="filter-label">Tags</span>
<div class="filter-chips" id="node-filter-tags">{tag_chips}</div>
</div>
<div class="node-filter-meta">
<span id="node-filter-count"></span>
<button type="button" id="node-filter-reset" class="filter-reset">Clear filters</button>
</div>
</div>

<div id="node-catalog-grid" class="node-list" markdown="1">

{rows}

</div>
"""


def _count_datasets() -> int:
    """Number of dataset pages under ``docs/catalogs/datasets/`` (excluding index)."""
    datasets_dir = REPO_ROOT / "docs" / "catalogs" / "datasets"
    if not datasets_dir.exists():
        return 0
    return sum(1 for p in datasets_dir.glob("*.md") if p.name != "index.md")


def _render_catalogs_overview(entries: list[NodeEntry]) -> str:
    total = len(entries)
    categories = sorted({e.category for e in entries})
    n_categories = len(categories)
    n_plugins = sum(1 for e in entries if e.is_plugin)
    n_builtin = total - n_plugins
    n_datasets = _count_datasets()

    cat_grid = "".join(
        f'<a class="cat-tile" href="nodes/#category={c}" data-category="{c}">'
        f'<img class="cat-tile-icon" src="../images/node-categories/{c}.svg" alt="" aria-hidden="true">'
        f'<span class="cat-tile-label">{c}</span>'
        f"</a>"
        for c in categories
    )

    return f"""---
hide:
  - toc
---

# Catalogs

The cuvis-ai catalogs are the inventory of every building block available
for hyperspectral pipelines:

- **{total} nodes** across **{n_categories} categories** — {n_builtin} built-in
  + {n_plugins} from plugins — see the filterable list at
  [Catalogs → Nodes](nodes/index.md).
- **{n_datasets} datasets** (cu3s recordings + annotations) — see the index
  at [Catalogs → Datasets](datasets/index.md).

## Nodes by category

<div class="cat-grid">
{cat_grid}
</div>

[Open the full Nodes catalog →](nodes/index.md){{ .md-button .md-button--primary }}

## Datasets

Reference datasets used by tutorials and benchmarks. Each is published to
HuggingFace under [cubert-gmbh](https://huggingface.co/cubert-gmbh) and
mirrored locally on first run.

[Open the Datasets catalog →](datasets/index.md){{ .md-button }}
"""


def main() -> None:
    builtins = collect_builtin_nodes()
    plugins = collect_plugin_nodes()
    all_entries: list[NodeEntry] = sorted(builtins + plugins, key=lambda e: (e.category, e.name))
    for entry in all_entries:
        entry.search_text = _build_search_text(entry)

    log.info("built-in nodes: %d, plugin nodes: %d", len(builtins), len(plugins))

    page = _render_index_page(all_entries)
    with mkdocs_gen_files.open("catalogs/nodes/index.md", "w") as fh:
        fh.write(page)

    overview = _render_catalogs_overview(all_entries)
    with mkdocs_gen_files.open("catalogs/index.md", "w") as fh:
        fh.write(overview)


main()
