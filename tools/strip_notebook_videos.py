"""Strip embedded video outputs from Jupyter notebooks to keep `.ipynb` size down.

Used by the pre-commit hook to silently auto-strip, and by the pre-push
hook as a hard verification step. Targets only base64-baked video
outputs (from `IPython.display.Video(embed=True)`) which bloat the
notebook by megabytes; small `<video src="path.mp4">` file references
from `embed=False` are kept.

Usage
-----
    python tools/strip_notebook_videos.py [--check] PATH [PATH ...]

`--check` exits non-zero if any of the files would be modified, without
writing them. Without `--check`, modifies in place and reports what was
stripped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Embedded videos render as `<source src="data:video/mp4;base64,...">`
# (or audio for the embed=True audio case); the small file-reference
# variant uses `<video src="path.mp4">` which we keep.
EMBED_DATA_URI = "data:video"


def _output_has_embedded_video(output: dict) -> bool:
    data = output.get("data", {})
    html = data.get("text/html", "")
    if isinstance(html, list):
        html = "".join(html)
    return EMBED_DATA_URI in html.lower()


def _output_has_video(output: dict) -> bool:
    """Backward-compat alias used by the CLI."""
    return _output_has_embedded_video(output)


def strip_videos(notebook_path: Path) -> int:
    """Strip video outputs from a notebook in place.

    Returns the number of outputs removed; 0 means the notebook was untouched.
    """
    raw = notebook_path.read_text(encoding="utf-8")
    notebook = json.loads(raw)

    removed = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        kept = [out for out in outputs if not _output_has_video(out)]
        diff = len(outputs) - len(kept)
        if diff:
            cell["outputs"] = kept
            removed += diff

    if removed:
        notebook_path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return removed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Notebook files to scan")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any file would be modified; do not write.",
    )
    args = parser.parse_args(argv)

    any_violations = False
    for path in args.paths:
        if not path.exists() or path.suffix != ".ipynb":
            continue
        if args.check:
            raw = path.read_text(encoding="utf-8")
            notebook = json.loads(raw)
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                if any(_output_has_video(out) for out in cell.get("outputs", [])):
                    print(f"video output present in {path}", file=sys.stderr)
                    any_violations = True
                    break
        else:
            removed = strip_videos(path)
            if removed:
                print(f"  stripped {removed} video output(s) from {path}")

    if args.check and any_violations:
        print(
            "\nRun `python tools/strip_notebook_videos.py <paths>` to fix, "
            "then amend or re-commit.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
