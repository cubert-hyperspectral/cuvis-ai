Status: Needs Review

This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

______________________________________________________________________

# CUV

IS.AI Documentation Guidelines

**Created:** 2026-02-05 **Last Updated:** 2026-02-05

______________________________________________________________________

## Table of Contents

1. [Documentation Philosophy](#documentation-philosophy)
1. [Documentation Architecture](#documentation-architecture)
1. [Docstring Guidelines](#docstring-guidelines)
1. [Maintaining Curated API Pages](#maintaining-curated-api-pages)
1. [Adding New Modules](#adding-new-modules)
1. [Missing Anchor Fix Strategy](#missing-anchor-fix-strategy)
1. [Build and Verification](#build-and-verification)

______________________________________________________________________

## Documentation Philosophy

### Following PyTorch's Model

Cuvis.AI follows the **PyTorch documentation model**:

> "Autosummary generates concise summary tables for modules, classes, and functions... making it easier for users to get an overview of the API. Autodoc generates a one pager documentation for all functions in a class which is often overwhelming and hard for users to read. **In most cases, autosummary is a better way of organizing API documentation.**"
>
> — [PyTorch Documentation Guidelines](https://github.com/pytorch/pytorch/wiki/Docstring-Guidelines)

### Core Principles

1. **Hybrid Approach**: Combine hand-written structure with auto-generated content
1. **User-Centered**: Organize documentation for discoverability, not just completeness
1. **Single Source of Truth**: Docstrings in code are the authoritative source
1. **Stay Current**: Documentation auto-updates when code docstrings change
1. **Meaningful Organization**: Group by functionality, not just alphabetically

______________________________________________________________________

## Documentation Architecture

### Two-Layer System

```text
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Curated Pages (Manual Structure)                  │
│  - docs/catalogs/nodes/*.md                                 │
│  - docs/reference/python-api/*.md                           │
│  - Organized categories, context, navigation                │
│  - Uses ::: directives to pull API content                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Python Docstrings (Auto-Generated Content)        │
│  - cuvis_ai/anomaly/rx_detector.py                          │
│  - cuvis_ai/node/losses.py                                  │
│  - Google-style docstrings                                  │
│  - Pulled at build time via mkdocstrings                    │
└─────────────────────────────────────────────────────────────┘
```

### What's Manual vs Auto-Generated

| Aspect             | Manual                             | Auto-Generated            |
| ------------------ | ---------------------------------- | ------------------------- |
| **Organization**   | ✅ Category headers, navigation    | ❌                        |
| **Context**        | ✅ Overview sections, explanations | ❌                        |
| **API Content**    | ❌                                 | ✅ Pulled from docstrings |
| **Parameter Docs** | ❌                                 | ✅ Pulled from docstrings |
| **Examples**       | ❌                                 | ✅ Pulled from docstrings |
| **Stays Current**  | ⚠️ When adding new modules         | ✅ Automatically          |

______________________________________________________________________

## Docstring Guidelines

### Style: Google Format

Cuvis.AI uses **Google-style docstrings** (same as PyTorch).

#### Module-Level Docstrings

**Purpose:** Provide overview, context, and references

````python
"""RX anomaly detection nodes for hyperspectral imaging.

This module implements the Reed-Xiaoli (RX) anomaly detection algorithm, a widely used
statistical method for detecting anomalies in hyperspectral images. The RX algorithm
computes squared Mahalanobis distance from the background distribution, treating
pixels with large distances as potential anomalies.

The module provides two variants:

- **RXGlobal**: Uses global statistics (mean, covariance) estimated from training data.
  Supports two-phase training: statistical initialization followed by optional gradient-based
  fine-tuning via unfreeze().

- **RXPerBatch**: Computes statistics independently for each batch on-the-fly without
  requiring initialization. Useful for real-time processing or when training data is unavailable.

Examples:
    Basic usage with global statistics:

    ```python
    from cuvis_ai.anomaly.rx_detector import RXGlobal

    detector = RXGlobal(
        in_channels=224,
        normalize=True,
        epsilon=1e-6
    )
    ```

Reference:
    Reed, I. S., & Yu, X. (1990). "Adaptive multiple-band CFAR detection of an optical
    pattern with unknown spectral distribution." IEEE Transactions on Acoustics, Speech,
    and Signal Processing, 38(10), 1760-1770.
"""
````

**Required Sections:**

- Brief description (1-2 sentences)
- Detailed explanation
- Available classes/functions overview
- Examples (optional but recommended)
- References (for research-based implementations)

#### Class-Level Docstrings

````python
class RXGlobal(Node):
    """RX anomaly detector using global statistics.

    Computes anomaly scores using the Reed-Xiaoli (RX) algorithm with global
    mean and covariance estimated during training. Supports two-phase training
    where statistical initialization is followed by optional gradient-based
    fine-tuning.

    Attributes:
        in_channels: Number of input spectral channels
        normalize: Whether to normalize anomaly scores
        epsilon: Small constant for numerical stability

    Examples:
        Create and initialize detector:

        ```python
        detector = RXGlobal(in_channels=224)
        detector.init(data_iterator)
        ```
    """
````

#### Method/Function Docstrings

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute RX anomaly scores.

    Args:
        x: Input tensor of shape (B, H, W, C) where:
            - B: batch size
            - H: height
            - W: width
            - C: channels (must equal in_channels)

    Returns:
        Anomaly score tensor of shape (B, H, W, 1). Higher scores
        indicate greater likelihood of anomaly.

    Raises:
        ValueError: If input channels don't match in_channels.
        RuntimeError: If detector hasn't been initialized.
    """
```

**Required Sections:**

- Brief description
- Args: All parameters with types and descriptions
- Returns: Return value with type and meaning
- Raises: Exceptions that can be raised (if applicable)
- Examples: Usage examples (optional but recommended)

### Docstring Best Practices

1. **Be Specific About Tensor Shapes**: Use notation like `(B, H, W, C)` with legend
1. **Explain Units**: If values have units (meters, seconds, etc.), specify them
1. **Link Related Components**: Reference related classes/functions
1. **Provide Context**: Explain when/why to use this over alternatives
1. **Include Equations**: For algorithms, show key equations in LaTeX
1. **Add Warnings**: Document gotchas, performance considerations, limitations

______________________________________________________________________

## Maintaining the Nodes Catalog

The Nodes catalog at [`docs/catalogs/nodes/index.md`](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/catalogs/nodes/index.md) is **generated at build time** by `scripts/generate_node_catalog.py` (registered as a `mkdocs-gen-files` script in `mkdocs.yml`). Each row in the rendered page is a collapsible `<details>` element keyed off the node class's metadata.

### How a node enters the catalog

For a built-in `cuvis_ai.node.<module>.ClassName`:

1. Add `_category = NodeCategory.<X>` and `_tags = frozenset({NodeTag.<...>})` on the class — the generator reads these via live `cls.get_category()` / `cls.get_tags()` calls (no doc edit required).
1. Make sure the class has a docstring whose first non-empty line summarises what it does. That line becomes the row's collapsed summary; the full docstring is rendered inside the row by mkdocstrings when expanded.

For a plugin class in a sibling repo:

1. Add the same `_category` / `_tags` assignments to the class.
1. Add an entry under `docs/data/plugin_sources.yaml` pointing at the plugin repo's on-disk path and listing the dotted class names. The generator reads the source via `ast` and never imports the plugin — so torch / ultralytics / SAM3 dependencies stay out of the docs venv.

### Don't edit the catalog page on disk

`docs/catalogs/nodes/index.md` is overridden at build time. Edits there are silently ignored. Change the generator (`scripts/generate_node_catalog.py`), the node's class attributes, or its docstring instead.

### After changes, re-run the build

```bash
uv run mkdocs build --strict
```

Note: `mkdocs serve` does **not** reliably auto-reload when only the generator script changes. Restart serve after editing the generator.

______________________________________________________________________

## Adding New Modules

### Checklist for New Modules

When adding a new module to Cuvis.AI:

- Write module-level docstring with overview and examples
- Write class/function docstrings following Google style
- Add to appropriate curated page (`docs/catalogs/nodes/*.md` or `docs/reference/python-api/*.md`)
- Place in correct category section
- Build docs locally to verify
- Check for broken cross-references
- Update related tutorial/guide if applicable

### Choosing the Right Curated Page

| Module Type          | Curated Page                                                                               | Example                 |
| -------------------- | ------------------------------------------------------------------------------------------ | ----------------------- |
| Node implementations | Auto-generated from `Node._category` + class docstring into `docs/catalogs/nodes/index.md` | RXDetector, DeepSVDD    |
| Training components  | Same — set `_category = NodeCategory.LOSS` or `NodeCategory.METRIC`                        | Loss functions, metrics |
| Data handling        | Same — set `_category = NodeCategory.SOURCE`                                               | Datasets, data loaders  |
| Pipeline building    | `docs/reference/python-api/pipeline.md`                                                    | Graph, Pipeline         |
| Port definitions     | `docs/reference/python-api/ports.md`                                                       | PortSpec, StreamType    |
| Utilities            | `docs/reference/python-api/utilities.md`                                                   | Helpers, factories      |

______________________________________________________________________

## Missing Anchor Fix Strategy

### Understanding the Issue

MkDocs generates anchors from headings:

- Heading: `## Data Loading with LentilsAnomalyDataNode`
- Anchor: `#data-loading-with-lentilsanomalydatanode`

Links break when:

1. Heading doesn't exist
1. Heading text doesn't match link
1. Heading uses unexpected formatting

### Anchor Naming Rules

MkDocs transforms headings to anchors by:

1. Converting to lowercase
1. Replacing spaces with hyphens
1. Removing special characters
1. Removing multiple consecutive hyphens

**Examples:**

| Heading                          | Anchor                         |
| -------------------------------- | ------------------------------ |
| `## DeepSVDD Nodes`              | `#deepsvdd-nodes`              |
| `## Two-Phase Training Workflow` | `#two-phase-training-workflow` |
| `## Step 1: Data Loading`        | `#step-1-data-loading`         |

______________________________________________________________________

## Build and Verification

### Standard Build Command

```bash
# Normal build (warnings displayed but not fatal)
uv run mkdocs build

# Strict build (warnings cause build failure)
uv run mkdocs build --strict
```

### Expected Warnings

After fixes, expect **14 warnings** for external file references:

```text
WARNING - Doc file contains a link '../../examples/grpc/...'
WARNING - Doc file contains a link '../../configs/plugins/...'
```

These are **acceptable** - they reference legitimate source files outside `docs/`.

### Verification Workflow

1. **Baseline**: Record current warning count

   ```bash
   uv run mkdocs build --strict 2>&1 | grep "WARNING" | wc -l
   ```

1. **After changes**: Rebuild and compare

   ```bash
   uv run mkdocs build --strict
   ```

1. **Serve locally** to visually verify:

   ```bash
   uv run mkdocs serve
   # Open http://127.0.0.1:8000
   ```

### Pre-Commit Checklist

Before committing documentation changes:

- `mkdocs build --strict` passes (or only expected warnings)
- Docstrings follow Google style
- New modules added to curated pages
- Internal links verified
- Examples tested (if code examples included)
- Spelling checked

______________________________________________________________________

## Quick Reference

### Docstring Template

````python
"""Brief one-line description.

Detailed multi-paragraph explanation of what this does, when to use it,
and how it fits into the larger system.

Args:
    param1: Description with type info
    param2: Description with type info

Returns:
    Description of return value with type

Raises:
    ErrorType: When this error occurs

Examples:
    Basic usage:

    ```python
    result = function(param1, param2)
    ```

See Also:
    - RelatedClass: For related functionality
"""
````

### Curated Page Template

```markdown
# API Section Name

Brief introduction to this API section.

## Category 1

Description of what belongs in this category.

### Component A
::: module.path.component_a
    options:
      show_root_heading: true
      heading_level: 4
```

______________________________________________________________________

**For questions or suggestions**, see: [Contributing Guide](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/reference/contributing/contributing/index.md)
