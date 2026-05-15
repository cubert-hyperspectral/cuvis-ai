"""mkdocs hooks: cross-cutting page transforms.

1. **RST ``.. math::`` → MathJax**. Numpy-style docstrings embed LaTeX
   via the RST directive; mkdocstrings passes it straight through, so
   it lands in the rendered page as a literal ``.. math::`` paragraph
   followed by a syntax-highlighted code fence. ``on_page_content``
   rewrites the pair as ``<div class="arithmatex">\\[ … \\]</div>`` so
   MathJax (configured in ``docs/javascripts/mathjax.js`` to scan
   ``.arithmatex``) typesets the equation.

2. **Hide TOC sidebar on catalog pages**. The Catalogs section is
   dense — overview tiles or 100+ collapsible node rows. Material's
   per-heading TOC sidebar would either repeat the page content or
   balloon to be unusable. ``on_page_markdown`` injects
   ``hide: [toc]`` into ``page.meta`` for every page under
   ``catalogs/`` so Material drops the sidebar.
"""

from __future__ import annotations

import html
import re

# Match:  <p>.. math::</p>  followed by the first highlight code block.
# mkdocstrings/Material renders the indented LaTeX as a fenced code block,
# which becomes <div class="highlight"><pre><code>...</code></pre></div>
# (sometimes wrapped in a <table class="highlighttable">).
_MATH_BLOCK_RE = re.compile(
    r"<p>\s*\.\.\s+math::\s*</p>\s*"
    r"<div class=\"[^\"]*\bhighlight\b[^\"]*\"[^>]*>"
    r"(?P<body>.*?)"
    r"</div>",
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(fragment: str) -> str:
    text = _TAG_RE.sub("", fragment)
    return html.unescape(text).strip()


def on_page_content(content: str, **kwargs: object) -> str:
    """mkdocs hook entry point — rewrites RST math blocks to MathJax."""

    def replace(match: re.Match[str]) -> str:
        latex = _strip_tags(match.group("body"))
        if not latex:
            return match.group(0)
        return f'<div class="arithmatex">\\[\n{latex}\n\\]</div>'

    return _MATH_BLOCK_RE.sub(replace, content)


# Catalog pages — index, nodes, datasets — are dense list/grid views;
# the per-heading TOC sidebar would either repeat the data already
# visible on the page (overview tiles, list rows) or balloon into an
# unusable 100-entry table. Force-hide it.
def on_page_markdown(markdown: str, page: object, **kwargs: object) -> str:
    url = getattr(page, "url", "") or ""
    src = getattr(getattr(page, "file", None), "src_path", "") or ""
    is_catalog = (
        url.startswith("catalogs/")
        or url.startswith("/catalogs/")
        or src.replace("\\", "/").startswith("catalogs/")
    )
    if is_catalog:
        meta = getattr(page, "meta", None)
        if isinstance(meta, dict):
            hide = meta.setdefault("hide", [])
            if isinstance(hide, list) and "toc" not in hide:
                hide.append("toc")
    return markdown
