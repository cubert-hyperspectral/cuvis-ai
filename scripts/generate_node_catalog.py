"""Generate the searchable node catalog at ``/catalogs/nodes/index.md``.

Two data sources, both used at mkdocs build time:

* **Built-in nodes** are imported live from the local ``cuvis_ai.node`` package
  (already installed in the docs venv) and introspected via
  ``cls.get_category()`` / ``cls.get_tags()``. This handles inheritance — a
  subclass that doesn't redefine ``_category`` still reports its parent's
  value.

* **Plugin capabilities** (nodes and data modules) are read from the plugin
  manifest YAMLs in the repo's plugins directory — the same files the
  pipeline loader and the gRPC server consume. Each capability entry already
  carries its category, tags, doc summary, and port specs, so the docs build
  never installs or imports torch / ultralytics / SAM3 / etc. Manifest entries
  that mirror built-in classes (``cuvis_ai_builtin.yaml``) are skipped in
  favour of the live import above.

Output: a single ``catalogs/nodes/index.md`` rendered as a list of
collapsible rows. Each row's body either includes a mkdocstrings
``:::`` block (built-ins, full docstring + signature) or the manifest's doc
summary plus input/output port tables and a link to the plugin repo at its
pinned tag (plugins).
"""

from __future__ import annotations

import inspect
import logging
import pkgutil
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import mkdocs_gen_files
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.extensions.ui.node_display import TAG_STYLES
from cuvis_ai_schemas.plugin import GitPluginSource, NodePortSpec, load_plugin_manifest

log = logging.getLogger("generate_node_catalog")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _plugin_manifest_dir() -> Path:
    """Locate the plugin manifests directory across repo layouts.

    Manifests live at ``cuvis_ai/configs/plugins`` once configs are packaged
    with the library, and at ``configs/plugins`` before that.
    """
    candidates = (
        REPO_ROOT / "cuvis_ai" / "configs" / "plugins",
        REPO_ROOT / "configs" / "plugins",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"no plugin manifests directory found; tried {candidates}")


PLUGIN_MANIFEST_DIR = _plugin_manifest_dir()
BUILTIN_PACKAGE = "cuvis_ai.node"

_SOURCE_LABELS = {"builtin": "Built-in", "plugin": "Plugin", "data-module": "Data module"}


@dataclass
class NodeEntry:
    """One catalog entry (node or data module), source-agnostic."""

    name: str
    dotted_path: str
    category: str
    tags: list[str]
    summary: str
    is_plugin: bool
    plugin_name: str | None = None
    repo_url: str | None = None
    version: str | None = None
    kind: str = "node"
    data_module_name: str = ""
    extras: list[str] = field(default_factory=list)
    input_specs: dict[str, NodePortSpec] = field(default_factory=dict)
    output_specs: dict[str, NodePortSpec] = field(default_factory=dict)
    search_text: str = field(default="", repr=False)

    @property
    def source(self) -> str:
        """Facet value for the Source filter chips (``data-source`` attribute)."""
        if not self.is_plugin:
            return "builtin"
        return "data-module" if self.kind == "data_module" else "plugin"


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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


def _browse_url(repo: str) -> str | None:
    """Turn a git remote URL into a browsable https URL (or ``None``)."""
    url = repo
    if url.startswith("git@"):
        url = "https://" + url.removeprefix("git@").replace(":", "/", 1)
    url = url.removesuffix(".git")
    return url if url.startswith(("http://", "https://")) else None


def collect_plugin_nodes(exclude_dotted: set[str]) -> list[NodeEntry]:
    """Read every plugin manifest and yield its capabilities as entries.

    ``exclude_dotted`` holds the dotted paths already collected from the live
    built-in import, so manifest mirrors of built-in classes
    (``cuvis_ai_builtin.yaml``) don't produce duplicate rows.

    Manifest validation errors propagate and fail the docs build: a broken
    manifest must be fixed, not silently dropped from the catalog. Likewise,
    parsing more than the built-in mirror but producing zero entries raises —
    that is exactly the "0 from plugins" regression this collector replaces.
    """
    manifest_paths = sorted(PLUGIN_MANIFEST_DIR.glob("*.yaml"))
    if not manifest_paths:
        raise RuntimeError(f"no plugin manifests found under {PLUGIN_MANIFEST_DIR}")

    entries: list[NodeEntry] = []
    for manifest_path in manifest_paths:
        manifest = load_plugin_manifest(manifest_path)
        is_git = isinstance(manifest, GitPluginSource)
        repo_url = _browse_url(manifest.repo) if is_git else None
        version = manifest.tag if is_git else None
        for cap in manifest.capabilities:
            if cap.class_name in exclude_dotted:
                continue
            category = cap.category
            if category not in _VALID_CATEGORIES:
                log.warning(
                    "%s: unknown category %r on %s; using 'unspecified'",
                    manifest_path.name,
                    category,
                    cap.class_name,
                )
                category = NodeCategory.UNSPECIFIED.value
            tags = sorted(t for t in cap.tags if t in _VALID_TAGS)
            unknown_tags = sorted(set(cap.tags) - _VALID_TAGS)
            if unknown_tags:
                log.warning(
                    "%s: dropping unknown tags %s on %s",
                    manifest_path.name,
                    unknown_tags,
                    cap.class_name,
                )
            entries.append(
                NodeEntry(
                    name=cap.class_name.rsplit(".", 1)[-1],
                    dotted_path=cap.class_name,
                    category=category,
                    tags=tags,
                    summary=cap.doc_summary,
                    is_plugin=True,
                    plugin_name=manifest.name,
                    repo_url=repo_url,
                    version=version,
                    kind=cap.kind,
                    data_module_name=cap.data_module_name,
                    extras=list(cap.extras),
                    input_specs=dict(cap.input_specs),
                    output_specs=dict(cap.output_specs),
                )
            )

    if len(manifest_paths) > 1 and not entries:
        raise RuntimeError(
            f"parsed {len(manifest_paths)} plugin manifests under "
            f"{PLUGIN_MANIFEST_DIR} but collected zero plugin capabilities — "
            "the catalog would silently list 0 plugin nodes"
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
    plugin_title = f"From plugin {entry.plugin_name}"
    if entry.version:
        plugin_title += f" {entry.version}"
    plugin_pill = (
        f'<span class="plugin-pill" title="{plugin_title}">{entry.plugin_name}</span>'
        if entry.is_plugin
        else ""
    )
    datamodule_pill = (
        '<span class="datamodule-pill" title="Data module provided by a plugin (not a node)">'
        "data module</span>"
        if entry.kind == "data_module"
        else ""
    )
    repo_link = (
        f'<a class="card-repo" href="{entry.repo_url}" title="Open plugin repo" rel="noopener">↗</a>'
        if entry.is_plugin and entry.repo_url
        else ""
    )
    summary_text = _html_escape(entry.summary) if entry.summary else ""
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
        f'<span class="row-meta">{datamodule_pill}{plugin_pill}{repo_link}</span>'
        f"</span>"
        f'<span class="row-summary">{summary_text}</span>'
    )

    body = _render_body(entry)
    return (
        f'<details class="node-row" markdown="1" '
        f'data-category="{entry.category}" '
        f'data-tags="{" ".join(entry.tags)}" '
        f'data-source="{entry.source}" '
        f'data-search="{entry.search_text}">\n'
        f"<summary>{summary_html}</summary>\n\n"
        f"{body}\n"
        f"</details>\n"
    )


def _render_ports_table(title: str, specs: dict[str, NodePortSpec]) -> str:
    """Render one Inputs/Outputs table as raw HTML (the ``tables`` markdown
    extension is not enabled, and raw HTML also sidesteps escaping issues)."""
    if not specs:
        return ""
    rows = []
    for port_name, spec in specs.items():
        marks = ""
        if spec.optional:
            marks += ' <span class="port-mark">optional</span>'
        if spec.variadic:
            marks += ' <span class="port-mark">variadic</span>'
        dtype = spec.dtype or "any"
        shape = str(list(spec.shape)) if spec.shape else "any"
        rows.append(
            "<tr>"
            f"<td><code>{port_name}</code>{marks}</td>"
            f"<td><code>{dtype}</code></td>"
            f"<td><code>{shape}</code></td>"
            f"<td>{_html_escape(spec.description)}</td>"
            "</tr>"
        )
    return (
        f'<p class="ports-title">{title}</p>\n'
        '<table class="ports-table">\n'
        "<thead><tr><th>Port</th><th>Dtype</th><th>Shape</th><th>Description</th></tr></thead>\n"
        "<tbody>" + "".join(rows) + "</tbody>\n</table>"
    )


def _render_body(entry: NodeEntry) -> str:
    if not entry.is_plugin:
        return (
            f'<div class="row-body" markdown="1">\n\n'
            f"::: {entry.dotted_path}\n"
            f"    options:\n"
            f"      show_root_heading: true\n"
            f"      heading_level: 4\n\n"
            f"</div>"
        )

    parts: list[str] = []
    if entry.summary:
        parts.append(_html_escape(entry.summary))
    if entry.kind == "data_module":
        extras = ", ".join(f"<code>{e}</code>" for e in entry.extras) or "none"
        parts.append(
            f"<p>Data module <code>{entry.data_module_name}</code> — pip extras: {extras}.</p>"
        )
    else:
        for title, specs in (("Inputs", entry.input_specs), ("Outputs", entry.output_specs)):
            table = _render_ports_table(title, specs)
            if table:
                parts.append(table)
    if entry.repo_url:
        if entry.version:
            parts.append(
                f"[View plugin repo ({entry.version})]"
                f"({entry.repo_url}/tree/{entry.version}){{ .row-source }}"
            )
        else:
            parts.append(f"[View plugin repo]({entry.repo_url}){{ .row-source }}")
    return '<div class="row-body" markdown="1">\n\n' + "\n\n".join(parts) + "\n\n</div>"


def _build_search_text(entry: NodeEntry) -> str:
    parts = [entry.name, entry.dotted_path, entry.category, entry.summary]
    parts.extend(entry.tags)
    if entry.plugin_name:
        parts.append(entry.plugin_name)
    if entry.kind == "data_module":
        parts.append("data module")
        parts.append(entry.data_module_name)
        parts.extend(entry.extras)
    return " ".join(parts).lower().replace('"', "")


def _render_index_page(entries: list[NodeEntry]) -> str:
    categories_present = sorted({e.category for e in entries})
    tags_present = sorted({t for e in entries for t in e.tags})
    sources_present = [
        s for s in ("builtin", "plugin", "data-module") if s in {e.source for e in entries}
    ]

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
    source_chips = "".join(
        f'<button type="button" class="filter-chip source-chip" data-source="{s}" '
        f'title="{_SOURCE_LABELS[s]}">{_SOURCE_LABELS[s]}</button>'
        for s in sources_present
    )

    rows = "\n\n".join(_render_card(e) for e in entries)
    n = len(entries)

    # One-row toolbar: search + foldable facet buttons + prerendered count. The
    # count is server-rendered so the bar doesn't reflow before the filter JS
    # runs; "items" (not "nodes") because the catalog also lists data modules.
    facet_buttons = "".join(
        f'<button type="button" class="filter-group-toggle" data-panel="node-filter-{key}" '
        f'aria-expanded="false" aria-controls="node-filter-{key}">{label}'
        f'<span class="filter-group-badge" aria-hidden="true"></span></button>'
        for key, label in (("categories", "Category"), ("tags", "Tags"), ("sources", "Source"))
    )

    return f"""---
hide:
  - toc
---

# Nodes Catalog

Every node available in cuvis-ai pipelines, in one place. Built-in nodes ship
with the `cuvis_ai` package; plugin nodes and data modules come from
separately-installable plugin manifests — see
[Plugin Development](../../reference/plugin-development/overview.md).

<div class="node-filter">
<div class="node-filter-toolbar">
<input type="search" id="node-filter-search" aria-label="Search catalog items by name, tag, or module" placeholder="Search {n} items by name, tag, module…" autocomplete="off">
<div class="filter-group-buttons">{facet_buttons}</div>
<span id="node-filter-count">{n} items</span>
<span id="node-filter-status" class="node-filter-sr-only" aria-live="polite"></span>
<button type="button" id="node-filter-reset" class="filter-reset" hidden>Clear</button>
</div>
<div class="filter-active-strip" id="node-filter-active" hidden></div>
<div class="filter-chips filter-panel" id="node-filter-categories" hidden>{cat_chips}</div>
<div class="filter-chips filter-panel" id="node-filter-tags" hidden>{tag_chips}</div>
<div class="filter-chips filter-panel" id="node-filter-sources" hidden>{source_chips}</div>
</div>

<div id="node-catalog-grid" class="node-list" markdown="1">

{rows}

<div class="node-filter-empty" id="node-filter-empty" hidden>
<p id="node-filter-empty-msg">No items match your search and filters.</p>
<button type="button" class="filter-reset" id="node-filter-empty-reset">Clear search and filters</button>
</div>

</div>
"""


def _count_datasets() -> int:
    """Number of dataset pages under ``docs/catalogs/datasets/`` (excluding index)."""
    datasets_dir = REPO_ROOT / "docs" / "catalogs" / "datasets"
    if not datasets_dir.exists():
        return 0
    return sum(1 for p in datasets_dir.glob("*.md") if p.name != "index.md")


def _render_catalogs_overview(entries: list[NodeEntry]) -> str:
    categories = sorted({e.category for e in entries})
    n_categories = len(categories)
    n_data_modules = sum(1 for e in entries if e.kind == "data_module")
    n_plugin_nodes = sum(1 for e in entries if e.is_plugin and e.kind == "node")
    n_nodes = len(entries) - n_data_modules
    n_builtin = n_nodes - n_plugin_nodes
    n_datasets = _count_datasets()

    # Continuation lines must not start with "+", "-", or "*": markdown would
    # parse them as nested list bullets.
    data_module_bullet = (
        f"- **{n_data_modules} data modules** from plugins, listed in the same\n"
        f"  catalog and filterable via the Source chips.\n"
        if n_data_modules
        else ""
    )

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

- **{n_nodes} nodes** across **{n_categories} categories** — {n_builtin} built-in
  and {n_plugin_nodes} from plugins — see the filterable list at
  [Catalogs → Nodes](nodes/index.md).
{data_module_bullet}- **{n_datasets} datasets** (cu3s recordings + annotations) — see the index
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
    plugins = collect_plugin_nodes(exclude_dotted={e.dotted_path for e in builtins})
    all_entries: list[NodeEntry] = sorted(builtins + plugins, key=lambda e: (e.category, e.name))
    for entry in all_entries:
        entry.search_text = _build_search_text(entry)

    log.info("built-in nodes: %d, plugin capabilities: %d", len(builtins), len(plugins))

    page = _render_index_page(all_entries)
    with mkdocs_gen_files.open("catalogs/nodes/index.md", "w") as fh:
        fh.write(page)

    overview = _render_catalogs_overview(all_entries)
    with mkdocs_gen_files.open("catalogs/index.md", "w") as fh:
        fh.write(overview)


# mkdocs-gen-files executes this script via runpy.run_path, which sets
# __name__ to "<run_path>"; the guard lets pytest import the module without
# triggering a docs build.
if __name__ in {"__main__", "<run_path>"}:
    main()
