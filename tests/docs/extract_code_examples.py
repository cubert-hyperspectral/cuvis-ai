"""Extract and analyze Python code examples from documentation.

This script extracts all Python code blocks from markdown files and categorizes
them as runnable, fragments, or requiring external data.
"""

import re
from pathlib import Path

# Documentation directory
DOCS_DIR = Path(__file__).parent.parent.parent / "docs"


def extract_python_blocks(md_file: Path) -> list[tuple[int, str]]:
    """Extract Python code blocks from a markdown file.

    Args:
        md_file: Path to markdown file

    Returns:
        List of (line_number, code_content) tuples
    """
    content = md_file.read_text(encoding="utf-8")
    blocks = []

    # Match ```python or ```py code blocks
    pattern = r"```(?:python|py)\n(.*?)```"

    for match in re.finditer(pattern, content, re.DOTALL):
        code = match.group(1)
        # Find line number by counting newlines before match
        line_num = content[: match.start()].count("\n") + 1
        blocks.append((line_num, code))

    return blocks


def categorize_code(code: str) -> str:
    """Categorize code block by type.

    Args:
        code: Python code string

    Returns:
        Category: 'runnable', 'fragment', 'requires_data', or 'config'
    """
    # Check for fragments (contains ellipsis or incomplete syntax)
    if "..." in code or code.strip().endswith(":"):
        return "fragment"

    # Check if code starts with indentation (part of larger context)
    if code and code[0] in (" ", "\t"):
        return "fragment"

    # Check for data requirements
    data_indicators = [
        "cu3s",
        "CU3S",
        "Demo_",
        "lentils",
        "data/",
        "/data/",
        "dataset",
        ".load(",
        ".open(",
        "Path(",
    ]
    if any(indicator in code for indicator in data_indicators):
        return "requires_data"

    # Check for configuration files (YAML/config examples)
    config_indicators = ["yaml", "config:", "plugins:", "nodes:"]
    if any(indicator in code for indicator in config_indicators):
        return "config"

    # Check for common runnable patterns
    runnable_indicators = ["def ", "class ", "import ", "from ", "print(", "# Example", "# Usage"]
    if any(indicator in code for indicator in runnable_indicators):
        # Exclude imports of cuvis_ai modules that might not be available
        if "import cuvis" in code or "from cuvis" in code:
            return "requires_data"
        return "runnable"

    return "fragment"


def analyze_documentation() -> dict[str, dict[str, int]]:
    """Analyze all Python code blocks in documentation.

    Returns:
        Dictionary mapping categories to statistics
    """
    stats = {
        "total_files": 0,
        "total_blocks": 0,
        "runnable": 0,
        "fragment": 0,
        "requires_data": 0,
        "config": 0,
        "files": {},
    }

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        blocks = extract_python_blocks(md_file)
        if not blocks:
            continue

        stats["total_files"] += 1
        file_stats = {"total": len(blocks), "categories": {}}

        for _line_num, code in blocks:
            stats["total_blocks"] += 1
            category = categorize_code(code)
            stats[category] += 1
            file_stats["categories"][category] = file_stats["categories"].get(category, 0) + 1

        relative_path = md_file.relative_to(DOCS_DIR.parent)
        stats["files"][str(relative_path)] = file_stats

    return stats


def print_report(stats: dict[str, dict[str, int]]):
    """Print analysis report.

    Args:
        stats: Statistics dictionary from analyze_documentation()
    """
    print("=" * 70)
    print("PYTHON CODE BLOCK ANALYSIS")
    print("=" * 70)
    print(f"\nTotal files with Python code: {stats['total_files']}")
    print(f"Total Python code blocks: {stats['total_blocks']}")
    print("\nBreakdown by category:")
    print(
        f"  Runnable examples:        {stats['runnable']:3d} ({stats['runnable'] / stats['total_blocks'] * 100:5.1f}%)"
    )
    print(
        f"  Code fragments:           {stats['fragment']:3d} ({stats['fragment'] / stats['total_blocks'] * 100:5.1f}%)"
    )
    print(
        f"  Requires data/setup:      {stats['requires_data']:3d} ({stats['requires_data'] / stats['total_blocks'] * 100:5.1f}%)"
    )
    print(
        f"  Configuration examples:   {stats['config']:3d} ({stats['config'] / stats['total_blocks'] * 100:5.1f}%)"
    )

    print("\n" + "=" * 70)
    print("FILES WITH MOST CODE BLOCKS")
    print("=" * 70)

    # Sort files by total blocks
    sorted_files = sorted(stats["files"].items(), key=lambda x: x[1]["total"], reverse=True)[:10]

    for filepath, file_stats in sorted_files:
        print(f"\n{filepath} ({file_stats['total']} blocks)")
        for category, count in sorted(file_stats["categories"].items()):
            print(f"  - {category}: {count}")


if __name__ == "__main__":
    stats = analyze_documentation()
    print_report(stats)

    # Save detailed report
    report_file = Path(__file__).parent / "code_analysis_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        import sys

        old_stdout = sys.stdout
        sys.stdout = f
        print_report(stats)

        # Add detailed file breakdown
        print("\n\n" + "=" * 70)
        print("DETAILED FILE BREAKDOWN")
        print("=" * 70)
        for filepath, file_stats in sorted(stats["files"].items()):
            print(f"\n{filepath}")
            print(f"  Total blocks: {file_stats['total']}")
            for category, count in sorted(file_stats["categories"].items()):
                print(f"    {category}: {count}")

        sys.stdout = old_stdout

    print(f"\nDetailed report saved to: {report_file}")
