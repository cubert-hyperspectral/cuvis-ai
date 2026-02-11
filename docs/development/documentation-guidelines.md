!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# CUV

IS.AI Documentation Guidelines

**Created:** 2026-02-05
**Last Updated:** 2026-02-05

---

## Table of Contents

1. [Documentation Philosophy](#documentation-philosophy)
2. [Documentation Architecture](#documentation-architecture)
3. [Docstring Guidelines](#docstring-guidelines)
4. [Maintaining Curated API Pages](#maintaining-curated-api-pages)
5. [Adding New Modules](#adding-new-modules)
6. [Missing Anchor Fix Strategy](#missing-anchor-fix-strategy)
7. [Build and Verification](#build-and-verification)

---

## Documentation Philosophy

### Following PyTorch's Model

CUVIS.AI follows the **PyTorch documentation model**:

> "Autosummary generates concise summary tables for modules, classes, and functions... making it easier for users to get an overview of the API. Autodoc generates a one pager documentation for all functions in a class which is often overwhelming and hard for users to read. **In most cases, autosummary is a better way of organizing API documentation.**"
>
> — [PyTorch Documentation Guidelines](https://github.com/pytorch/pytorch/wiki/Docstring-Guidelines)

### Core Principles

1. **Hybrid Approach**: Combine hand-written structure with auto-generated content
2. **User-Centered**: Organize documentation for discoverability, not just completeness
3. **Single Source of Truth**: Docstrings in code are the authoritative source
4. **Stay Current**: Documentation auto-updates when code docstrings change
5. **Meaningful Organization**: Group by functionality, not just alphabetically

---

## Documentation Architecture

### Two-Layer System

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Curated Pages (Manual Structure)                  │
│  - docs/api/nodes.md                                        │
│  - docs/api/training.md                                     │
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

| Aspect | Manual | Auto-Generated |
|--------|--------|----------------|
| **Organization** | ✅ Category headers, navigation | ❌ |
| **Context** | ✅ Overview sections, explanations | ❌ |
| **API Content** | ❌ | ✅ Pulled from docstrings |
| **Parameter Docs** | ❌ | ✅ Pulled from docstrings |
| **Examples** | ❌ | ✅ Pulled from docstrings |
| **Stays Current** | ⚠️ When adding new modules | ✅ Automatically |

---

## Docstring Guidelines

### Style: Google Format

CUVIS.AI uses **Google-style docstrings** (same as PyTorch).

#### Module-Level Docstrings

**Purpose:** Provide overview, context, and references

```python
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
```

**Required Sections:**
- Brief description (1-2 sentences)
- Detailed explanation
- Available classes/functions overview
- Examples (optional but recommended)
- References (for research-based implementations)

#### Class-Level Docstrings

```python
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
```

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
2. **Explain Units**: If values have units (meters, seconds, etc.), specify them
3. **Link Related Components**: Reference related classes/functions
4. **Provide Context**: Explain when/why to use this over alternatives
5. **Include Equations**: For algorithms, show key equations in LaTeX
6. **Add Warnings**: Document gotchas, performance considerations, limitations

---

## Maintaining Curated API Pages

### Structure of Curated Pages

Curated pages (e.g., `docs/api/nodes.md`) have this structure:

```markdown
# Page Title

Brief introduction explaining what this API section covers.

## Category Name

Description of this category and when to use these components.

### Component Name

::: full.module.path
    options:
      show_root_heading: true
      heading_level: 4
```

### When to Update Curated Pages

Update curated pages when:

1. **New Module Added**: Add entry with `:::` directive
2. **Module Moved**: Update import path in `:::` directive
3. **Category Changed**: Move `:::` directive to new category section
4. **Module Deprecated**: Add deprecation notice, consider removing
5. **New Category Needed**: Add category header with description

### Example: Adding a New Module

**Step 1:** Create Python module with rich docstrings

```python
# cuvis_ai/anomaly/new_detector.py
"""New anomaly detection method.

This module implements...
"""
```

**Step 2:** Add to appropriate curated page

```markdown
<!-- docs/api/nodes.md -->

## Anomaly Detection Nodes

Statistical and deep learning methods for detecting anomalies in hyperspectral data.

### RX Detector
::: cuvis_ai.anomaly.rx_detector
    options:
      show_root_heading: true
      heading_level: 4

### New Detector  ← ADD THIS
::: cuvis_ai.anomaly.new_detector
    options:
      show_root_heading: true
      heading_level: 4
```

**Step 3:** Verify build

```bash
uv run mkdocs build --strict
```

---

## Adding New Modules

### Checklist for New Modules

When adding a new module to CUVIS.AI:

- [ ] Write module-level docstring with overview and examples
- [ ] Write class/function docstrings following Google style
- [ ] Add to appropriate curated page (`docs/api/nodes.md`, etc.)
- [ ] Place in correct category section
- [ ] Build docs locally to verify
- [ ] Check for broken cross-references
- [ ] Update related tutorial/guide if applicable

### Choosing the Right Curated Page

| Module Type | Curated Page | Example |
|-------------|--------------|---------|
| Node implementations | `docs/api/nodes.md` | RXDetector, DeepSVDD |
| Training components | `docs/api/training.md` | Loss functions, metrics |
| Data handling | `docs/api/data.md` | Datasets, data loaders |
| Pipeline building | `docs/api/pipeline.md` | Graph, Pipeline |
| Port definitions | `docs/api/ports.md` | PortSpec, StreamType |
| Utilities | `docs/api/utilities.md` | Helpers, factories |

---

## Missing Anchor Fix Strategy

### Understanding the Issue

MkDocs generates anchors from headings:

- Heading: `## Data Loading with LentilsAnomalyDataNode`
- Anchor: `#data-loading-with-lentilsanomalydatanode`

Links break when:
1. Heading doesn't exist
2. Heading text doesn't match link
3. Heading uses unexpected formatting

### Anchor Naming Rules

MkDocs transforms headings to anchors by:

1. Converting to lowercase
2. Replacing spaces with hyphens
3. Removing special characters
4. Removing multiple consecutive hyphens

**Examples:**

| Heading | Anchor |
|---------|--------|
| `## DeepSVDD Nodes` | `#deepsvdd-nodes` |
| `## Two-Phase Training Workflow` | `#two-phase-training-workflow` |
| `## Step 1: Data Loading` | `#step-1-data-loading` |

---

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

```
WARNING - Doc file contains a link '../../examples/grpc/...'
WARNING - Doc file contains a link '../../configs/plugins/...'
```

These are **acceptable** - they reference legitimate source files outside `docs/`.

### Verification Workflow

1. **Baseline**: Record current warning count
   ```bash
   uv run mkdocs build --strict 2>&1 | grep "WARNING" | wc -l
   ```

2. **After changes**: Rebuild and compare
   ```bash
   uv run mkdocs build --strict
   ```

3. **Serve locally** to visually verify:
   ```bash
   uv run mkdocs serve
   # Open http://127.0.0.1:8000
   ```

### Pre-Commit Checklist

Before committing documentation changes:

- [ ] `mkdocs build --strict` passes (or only expected warnings)
- [ ] Docstrings follow Google style
- [ ] New modules added to curated pages
- [ ] Internal links verified
- [ ] Examples tested (if code examples included)
- [ ] Spelling checked

---

## Quick Reference

### Docstring Template

```python
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
```

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

---

**For questions or suggestions**, see: [Contributing Guide](contributing.md)
