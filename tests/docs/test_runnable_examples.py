"""Test runnable Python code examples from documentation.

This module tests that runnable Python code examples from the documentation
are syntactically valid and can be compiled.
"""

import ast
from pathlib import Path

import pytest
from extract_code_examples import categorize_code, extract_python_blocks

# Documentation directory
DOCS_DIR = Path(__file__).parent.parent.parent / "docs"


def get_runnable_examples():
    """Get all runnable code examples from documentation.

    Returns:
        List of (file_path, line_number, code) tuples
    """
    examples = []

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        blocks = extract_python_blocks(md_file)
        for line_num, code in blocks:
            if categorize_code(code) == "runnable":
                examples.append((md_file, line_num, code))

    return examples


# Get all runnable examples for parametrization
RUNNABLE_EXAMPLES = get_runnable_examples()


@pytest.mark.parametrize(
    "md_file,line_num,code",
    RUNNABLE_EXAMPLES,
    ids=[f"{f.relative_to(DOCS_DIR.parent)}:{ln}" for f, ln, _ in RUNNABLE_EXAMPLES],
)
def test_code_syntax(md_file, line_num, code):
    """Test that runnable code examples have valid Python syntax.

    This test compiles the code to check for syntax errors.
    It does not execute the code, only validates syntax.
    """
    try:
        # Attempt to parse the code as an AST
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(
            f"Syntax error in {md_file.relative_to(DOCS_DIR.parent)}:{line_num}\n"
            f"Error: {e}\n"
            f"Code:\n{code}"
        )


def test_runnable_examples_count():
    """Verify we found runnable examples to test."""
    assert len(RUNNABLE_EXAMPLES) > 0, "No runnable examples found in documentation"
    # We expect at least 100 runnable examples based on analysis
    assert len(RUNNABLE_EXAMPLES) >= 100, (
        f"Expected at least 100 runnable examples, found {len(RUNNABLE_EXAMPLES)}"
    )


def test_no_common_mistakes():
    """Test for common mistakes in runnable examples."""
    issues = []

    for md_file, line_num, code in RUNNABLE_EXAMPLES:
        relative_path = md_file.relative_to(DOCS_DIR.parent)

        # Check for hardcoded Windows paths
        if "C:\\" in code or "D:\\" in code:
            issues.append(f"{relative_path}:{line_num} - Contains hardcoded Windows path")

        # Check for TODO/FIXME comments in runnable code
        if "TODO" in code or "FIXME" in code:
            issues.append(f"{relative_path}:{line_num} - Contains TODO/FIXME comment")

    if issues:
        pytest.fail(
            f"Found {len(issues)} common mistakes in runnable examples:\n"
            + "\n".join(f"  - {issue}" for issue in issues[:10])
            + (f"\n  ... and {len(issues) - 10} more" if len(issues) > 10 else "")
        )


@pytest.mark.slow
def test_import_statements():
    """Test that import statements in runnable examples are valid."""
    import_issues = []

    for md_file, line_num, code in RUNNABLE_EXAMPLES:
        relative_path = md_file.relative_to(DOCS_DIR.parent)

        # Extract import statements
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Check for common typos or non-existent modules
                        if "cuvis_ai_core" in alias.name and "cuvis_ai_core" not in alias.name:
                            import_issues.append(
                                f"{relative_path}:{line_num} - "
                                f"Possible typo in import: {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if (
                        node.module
                        and "cuvis_ai_core" in node.module
                        and "cuvis_ai_core" != node.module
                    ):
                        # Only flag if it looks like a typo
                        if node.module.startswith("cuvis_ai"):
                            import_issues.append(
                                f"{relative_path}:{line_num} - "
                                f"Possible typo in import from: {node.module}"
                            )
        except SyntaxError:
            # Already caught by test_code_syntax
            pass

    if import_issues:
        pytest.fail(
            f"Found {len(import_issues)} potential import issues:\n"
            + "\n".join(f"  - {issue}" for issue in import_issues[:10])
        )
