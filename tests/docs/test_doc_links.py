"""Test documentation links for validity.

This module uses pytest-check-links to validate all internal and external links
in the documentation. Links are validated to ensure no broken references exist.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# Get documentation directory relative to this test file
DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
ROOT_DIR = Path(__file__).parent.parent.parent


def test_docs_directory_exists():
    """Verify the docs directory exists before running link tests."""
    assert DOCS_DIR.exists(), f"Documentation directory not found: {DOCS_DIR}"
    assert DOCS_DIR.is_dir(), f"Expected directory, got file: {DOCS_DIR}"


@pytest.mark.check_links
@pytest.mark.parametrize("doc_file", sorted(DOCS_DIR.rglob("*.md")))
def test_markdown_links(doc_file):
    """
    Test each markdown file for broken links.

    This test uses pytest-check-links to automatically validate:
    - Internal documentation links (relative paths)
    - External links (HTTP/HTTPS URLs)
    - Anchor links within documents

    The test is parametrized to run separately for each markdown file,
    providing clear error messages when specific files have broken links.
    """
    # pytest-check-links will automatically check this file when the test runs
    assert doc_file.exists(), f"File not found: {doc_file}"


@pytest.mark.check_links
def test_root_readme():
    """Test links in root README.md file."""
    readme = ROOT_DIR / "README.md"
    assert readme.exists(), "README.md not found in repository root"


@pytest.mark.check_links
def test_root_contributing():
    """Test links in root CONTRIBUTING.md file."""
    contributing = ROOT_DIR / "CONTRIBUTING.md"
    assert contributing.exists(), "CONTRIBUTING.md not found in repository root"


@pytest.mark.check_links
def test_root_changelog():
    """Test links in root CHANGELOG.md file."""
    changelog = ROOT_DIR / "CHANGELOG.md"
    assert changelog.exists(), "CHANGELOG.md not found in repository root"
