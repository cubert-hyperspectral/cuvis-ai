# Documentation Tests

This directory contains automated tests for documentation quality and correctness.

## Link Validation Tests

The `test_doc_links.py` module uses pytest-check-links to validate all links in the documentation.

### Running Link Tests

**Test all documentation links:**
```bash
uv run pytest tests/docs/test_doc_links.py -m check_links --check-links -v
```

**Test specific file:**
```bash
uv run pytest tests/docs/test_doc_links.py::test_root_readme --check-links -v
```

**Generate HTML report:**
```bash
uv run pytest tests/docs/test_doc_links.py -m check_links --check-links --html=reports/link_check.html
```

### What Gets Tested

- ✅ All internal documentation links (relative paths)
- ✅ External links (HTTP/HTTPS URLs)
- ✅ Anchor links within documents
- ✅ Root files (README.md, CONTRIBUTING.md, CHANGELOG.md)

### Ignored Links

Some links are ignored to prevent false positives:
- `https://download.pytorch.org/*` - External, may be slow
- `https://cloud.cubert-gmbh.de/*` - Private cloud
- GitHub release tags (checked manually)

### Interpreting Results

**PASSED**: All links are valid and accessible
**FAILED**: Contains broken or invalid links - fix before committing

## List Spacing Tests

The `test_markdown_list_spacing.py` module checks that prose and labels are separated
from following Markdown list blocks by a blank line.

### Running List Spacing Tests

**Test list spacing in docs, examples, and API docstrings:**
```bash
uv run pytest tests/docs/test_markdown_list_spacing.py -v
```

### What Gets Tested

- `docs/**/*.md`
- `examples/**/*.md`
- `cuvis_ai/**/*.py` docstrings rendered by mkdocstrings

### Notes

- Fenced code blocks are ignored so YAML/Python/JSON examples are not flagged.
- The check only targets missing blank lines before bullet or numbered list blocks.

## Future Tests

Additional test modules planned:
- `test_cli_commands.py` - Validate CLI command examples
- `test_code_examples.py` - Test Python code blocks in documentation
- `test_search_functionality.py` - Verify search index quality
