"""Generate API reference pages automatically."""

from pathlib import Path

import mkdocs_gen_files

# Root of the cuvis_ai package
nav = mkdocs_gen_files.Nav()
src = Path(__file__).parent.parent / "cuvis_ai"

# Iterate through all Python files in cuvis_ai package
for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src.parent).with_suffix("")
    doc_path = path.relative_to(src.parent).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    # Skip __pycache__ and test files
    if "__pycache__" in parts or "test_" in path.name:
        continue

    # Skip __init__.py files unless they have content
    if path.name == "__init__.py":
        # Only include if it's a top-level __init__ or has significant content
        if len(parts) > 1:
            continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write navigation structure
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
