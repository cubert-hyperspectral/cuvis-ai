"""Sync HuggingFace dataset READMEs into docs/catalogs/datasets/.

Internal maintainer tooling — not run in CI and not part of
``mkdocs build``. Invoke manually after an upstream README change on
HuggingFace. The generated Markdown is committed to the repo, so
``mkdocs build`` stays hermetic and the regenerated pages show up in
PR diffs for review.

Usage::

    uv run python scripts/sync_hf_readmes.py

The script fetches ``https://huggingface.co/datasets/<org>/<hf_repo>/raw/main/README.md``
for each demo dataset, strips the HuggingFace YAML frontmatter (MkDocs
would otherwise interpret it as page metadata), rewrites relative
image links to absolute HF ``resolve/main`` URLs so they render outside
the HF UI, prepends a small "Mirrored from HuggingFace" attribution
admonition, and writes the result to ``docs/catalogs/datasets/<display_slug>.md``.

The display slug can differ from the HF repo name — useful when the HF
repo carries a long descriptive name but the docs want a shorter label
(e.g. ``XMR_Demo_Industrial_FOD_Lentils`` ←
``XMR_Demo_Industrial_Foreign_Object_Detection_Lentils``). All in-body
URLs and the attribution admonition still cite the real HF repo.

Adding a new dataset
--------------------

1. Publish the dataset on HuggingFace under the ``cubert-gmbh`` org.
2. Append a ``(display_slug, hf_repo_name)`` pair to the ``DATASETS``
   tuple in this module. Use the same value twice when no short alias
   is needed.
3. Run ``uv run python scripts/sync_hf_readmes.py``.
4. Add a nav entry in ``mkdocs.yml`` under ``Catalogs → Datasets``
   pointing at ``catalogs/datasets/<display_slug>.md``.
5. Add a card to ``docs/catalogs/datasets/index.md``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ORG = "cubert-gmbh"
DATASETS: tuple[tuple[str, str], ...] = (
    ("XMR_Demo_Blood_Perfusion", "XMR_Demo_Blood_Perfusion"),
    (
        "XMR_Demo_Industrial_FOD_Lentils",
        "XMR_Demo_Industrial_Foreign_Object_Detection_Lentils",
    ),
    ("XMR_Demo_Object_Tracking", "XMR_Demo_Object_Tracking"),
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "catalogs" / "datasets"
RAW_URL = "https://huggingface.co/datasets/{org}/{name}/raw/main/README.md"
RESOLVE_URL_BASE = "https://huggingface.co/datasets/{org}/{name}/resolve/main/"
HF_PAGE_URL = "https://huggingface.co/datasets/{org}/{name}"

FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
RELATIVE_IMG_RE = re.compile(
    r"(!\[[^\]]*\]\()(?!https?://|data:|/)([^)\s]+)(\))",
)
RELATIVE_LINK_RE = re.compile(
    r"(?<!!)(\[[^\]]+\]\()(?!https?://|mailto:|#|/)([^)\s#]+)(\))",
)


def _fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": "cuvis-ai-docs-sync/1.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text, count=1)


def _absolutise_assets(text: str, hf_repo: str) -> str:
    base = RESOLVE_URL_BASE.format(org=ORG, name=hf_repo)
    text = RELATIVE_IMG_RE.sub(lambda m: f"{m.group(1)}{base}{m.group(2)}{m.group(3)}", text)
    text = RELATIVE_LINK_RE.sub(lambda m: f"{m.group(1)}{base}{m.group(2)}{m.group(3)}", text)
    return text


def _header(hf_repo: str) -> str:
    page = HF_PAGE_URL.format(org=ORG, name=hf_repo)
    return (
        f'!!! info "Mirrored from HuggingFace"\n'
        f"    This page mirrors the README of [`{ORG}/{hf_repo}`]({page}).\n\n"
    )


def _empty_body_fallback(hf_repo: str) -> str:
    page = HF_PAGE_URL.format(org=ORG, name=hf_repo)
    return (
        f"# {hf_repo}\n\n"
        f"The upstream dataset card on HuggingFace is currently sparse — "
        f"the dataset is available at [`{ORG}/{hf_repo}`]({page}) but the "
        f"README has no descriptive body yet. Refer to the linked HuggingFace "
        f"page for files, splits, and download instructions.\n"
    )


def sync_one(display_slug: str, hf_repo: str) -> Path:
    url = RAW_URL.format(org=ORG, name=hf_repo)
    print(f"  fetching {url}")
    body = _fetch(url)
    body = _strip_frontmatter(body)
    body = _absolutise_assets(body, hf_repo)
    if not body.strip():
        body = _empty_body_fallback(hf_repo)
    output = OUTPUT_DIR / f"{display_slug}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_header(hf_repo) + body, encoding="utf-8")
    print(f"  wrote {output.relative_to(REPO_ROOT)}")
    return output


def main() -> int:
    print(
        f"Syncing {len(DATASETS)} HuggingFace dataset READMEs to {OUTPUT_DIR.relative_to(REPO_ROOT)}/"
    )
    failures: list[tuple[str, str]] = []
    for display_slug, hf_repo in DATASETS:
        try:
            sync_one(display_slug, hf_repo)
        except (HTTPError, URLError) as exc:
            print(f"  FAILED {display_slug} ({hf_repo}): {exc}", file=sys.stderr)
            failures.append((display_slug, str(exc)))
    if failures:
        print(f"\n{len(failures)} dataset(s) failed:", file=sys.stderr)
        for name, err in failures:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1
    print("\nAll datasets synced successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
