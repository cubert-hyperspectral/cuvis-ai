!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Docstring Standards

## Overview

This guide provides comprehensive standards for writing high-quality docstrings in CUVIS.AI. Following these standards ensures consistent, well-documented code that generates excellent API documentation.

---

## Docstring Style

CUVIS.AI uses **NumPy style** docstrings for consistency with the scientific Python ecosystem and compatibility with mkdocstrings.

### Why NumPy Style?

- **Readability**: Clear section headers and structured format
- **Compatibility**: Works seamlessly with mkdocstrings and Sphinx
- **Scientific Standard**: Widely adopted in NumPy, SciPy, scikit-learn, and PyTorch
- **Rich Features**: Supports all common documentation needs (parameters, returns, raises, examples, notes, references)

---

## Required Sections by Component Type

### For Modules

Every Python module should have a module-level docstring at the top of the file:

```python
"""
One-line module summary (under 80 characters).

Extended description of module purpose and contents.
Can span multiple paragraphs to provide context about
what the module contains and when to use it.

See Also
--------
related_module : Brief description
another_module : Brief description
"""
```

**Example:**

```python
"""
Anomaly Detection Nodes.

This module provides anomaly detection nodes for hyperspectral image analysis,
including both statistical methods (RX, LAD) and deep learning approaches
(Deep SVDD). Each node implements the BaseNode interface and can be composed
into processing pipelines.

See Also
--------
cuvis_ai.deciders : Binary decision nodes for classification
cuvis_ai.node.normalization : Preprocessing nodes
"""
```

### For Classes

All public classes must have comprehensive docstrings:

```python
class MyNode(BaseNode):
    """
    One-line class summary (under 80 characters).

    Extended description explaining what this class does, when to use it,
    and any important behavioral characteristics. Can span multiple
    paragraphs if needed.

    Parameters
    ----------
    param1 : type
        Description of param1. Explain what it controls and valid values.
    param2 : type, optional
        Description of param2. Mention default behavior.
        Default is ``default_value``.
    param3 : type or None, optional
        Description of param3. Explain None behavior.
        If None, the behavior is... Default is None.

    Attributes
    ----------
    attribute1 : type
        Description of public attribute and what it stores.
    attribute2 : type
        Description of public attribute.

    Raises
    ------
    ValueError
        If param1 is negative or out of valid range.
    RuntimeError
        If node is not initialized before processing.

    See Also
    --------
    RelatedNode : Brief description of relationship
    AnotherNode : Brief description of relationship

    Notes
    -----
    Additional implementation notes, algorithm details, or important
    considerations for users. Can include mathematical formulas,
    performance characteristics, or usage guidelines.

    References
    ----------
    .. [1] Author, "Paper Title," Journal, Year.
           URL or DOI if applicable.

    Examples
    --------
    Basic usage:

    >>> node = MyNode(param1=10)
    >>> result = node.forward(data)
    >>> print(result["output"])

    Advanced usage with initialization:

    >>> from cuvis_ai_core.training import StatisticalTrainer
    >>> node = MyNode(param1=20, param2=0.5)
    >>> pipeline.add_node(node)
    >>> trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    >>> trainer.fit()  # Initializes node
    >>> result = node.forward(test_data)
    """
```

### For Methods and Functions

All public methods must document parameters, returns, and exceptions:

```python
def forward(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Process input data through the node.

    Extended description if the method needs more explanation
    about what it does and how it works.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape ``(batch, channels, height, width)``.
        Values should be in range [0, 1] after normalization.
    mask : np.ndarray or None, optional
        Binary mask with shape ``(batch, 1, height, width)``.
        If None, no masking is applied. Default is None.

    Returns
    -------
    dict
        Dictionary containing:

        - 'output' : np.ndarray
            Processed data with shape ``(batch, out_channels, height, width)``.
        - 'scores' : np.ndarray
            Anomaly scores with shape ``(batch, height, width)``.
        - 'metadata' : dict
            Processing metadata including computation time and statistics.

    Raises
    ------
    ValueError
        If input data has incorrect shape or invalid values.
    RuntimeError
        If node is not initialized via ``statistical_initialization()``.

    See Also
    --------
    statistical_initialization : Initialize node with initialization data

    Notes
    -----
    This method processes data in batches for efficiency. For large
    datasets, consider using batch_size <= 32 to avoid memory issues.

    Examples
    --------
    Basic usage:

    >>> data = torch.randn(1, 3, 64, 64)
    >>> result = node.forward(data)
    >>> result['output'].shape
    torch.Size([1, 10, 64, 64])

    With masking:

    >>> mask = torch.ones(1, 1, 64, 64)
    >>> result = node.forward(data, mask=mask)
    """
```

### For Properties

Properties should have concise docstrings:

```python
@property
def requires_initial_fit(self) -> bool:
    """
    Whether this node requires statistical initialization.

    Returns
    -------
    bool
        True if ``statistical_initialization()`` must be called before
        ``forward()``, False otherwise.
    """
```

---

## Best Practices

### 1. Be Concise but Complete

- **First line**: Brief summary in one sentence (under 80 characters)
- **Blank line**: Always follow first line with blank line
- **Extended description**: Add more details if needed in subsequent paragraphs

**Good:**

```python
def process(data):
    """
    Process hyperspectral data through RX anomaly detection.

    Applies the RX algorithm to compute pixel-wise anomaly scores
    based on Mahalanobis distance from the background distribution.
    """
```

**Bad:**

```python
def process(data):
    """This function processes data."""  # Too vague
```

### 2. Document Parameters Thoroughly

For each parameter, include:

- **Type**: Clear type annotation
- **Purpose**: What the parameter controls
- **Valid range/values**: Constraints or valid options
- **Default behavior**: For optional parameters
- **Units**: If applicable (e.g., pixels, seconds, degrees)

**Good:**

```python
"""
Parameters
----------
threshold : float
    Anomaly score threshold in range [0, 1]. Pixels with scores
    above this value are classified as anomalous. Default is 0.5.
channels : list of int or None, optional
    Channel indices to process. If None, all channels are used.
    Default is None.
"""
```

**Bad:**

```python
"""
Parameters
----------
threshold : float
    The threshold.  # Incomplete - missing range, purpose, default
"""
```

### 3. Describe Return Values Clearly

For dictionary returns, document all keys and their meanings:

**Good:**

```python
"""
Returns
-------
dict
    Dictionary containing:

    - 'scores' : torch.Tensor
        Anomaly scores with shape ``(batch, height, width)``.
        Higher values indicate more anomalous pixels.
    - 'threshold' : float
        Adaptive threshold value used for classification.
    - 'decisions' : torch.Tensor
        Binary decisions (0=normal, 1=anomaly) with shape
        ``(batch, height, width)``.
"""
```

### 4. Add Meaningful Examples

Include examples that demonstrate:

- **Basic usage**: Simplest way to use the component
- **Common patterns**: Typical use cases
- **Edge cases**: How to handle special situations

Make examples **runnable** when possible:

```python
"""
Examples
--------
Basic RX detection:

>>> from cuvis_ai.anomaly import RXDetector
>>> detector = RXDetector()
>>> data = torch.randn(1, 150, 64, 64)  # (batch, channels, H, W)
>>> result = detector.forward(data)
>>> result['scores'].shape
torch.Size([1, 64, 64])

With custom parameters:

>>> from cuvis_ai_core.training import StatisticalTrainer
>>> detector = RXDetector(use_global_covariance=True)
>>> pipeline.add_node(detector)
>>> trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
>>> trainer.fit()  # Initializes detector with background data
>>> result = detector.forward(test_data)
"""
```

### 5. Cross-Reference Related Items

Use "See Also" to link related functionality:

```python
"""
See Also
--------
RXDetector : Reed-Xiaoli anomaly detector
LADDetector : Local anomaly density detector
cuvis_ai.deciders.BinaryDecider : Convert scores to binary decisions
"""
```

### 6. Document Exceptions

List exceptions that can be raised and explain when:

```python
"""
Raises
------
ValueError
    If ``data`` has fewer than 2 dimensions.
    If ``channels`` contains indices >= data.shape[1].
RuntimeError
    If ``statistical_initialization()`` was not called when
    ``requires_initial_fit=True``.
FileNotFoundError
    If checkpoint file specified in ``load_path`` does not exist.
"""
```

### 7. Add Notes for Important Details

Use Notes section for:

- Algorithm details
- Performance considerations
- Memory requirements
- Thread safety
- Version compatibility

```python
"""
Notes
-----
This implementation uses Welford's online algorithm for numerical
stability when computing covariance matrices. Memory usage is
O(C^2) where C is the number of channels.

For datasets with >500 channels, consider using PCA dimensionality
reduction first to improve performance.
"""
```

---

## Type Hints

### Use Type Hints Consistently

Always include type hints in function signatures:

```python
from typing import Optional, Dict, List, Any, Union, Tuple
import numpy as np
import torch

def process(
    self,
    data: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    channels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Process data with optional mask and channel selection."""
```

### Common Types

| Type | Usage |
|------|-------|
| `torch.Tensor` | PyTorch tensors |
| `np.ndarray` | NumPy arrays |
| `Dict[str, Any]` | Dictionaries with string keys |
| `List[T]` | Lists of type T |
| `Tuple[T1, T2]` | Tuples with specific types |
| `Optional[T]` | T or None |
| `Union[T1, T2]` | Either T1 or T2 |
| `Callable[[ArgTypes], ReturnType]` | Function types |

---

## Examples in Docstrings

### Doctest Format

Use doctest format for executable examples:

```python
"""
Examples
--------
>>> node = RXDetector()
>>> data = torch.randn(1, 150, 64, 64)
>>> result = node.forward(data)
>>> result['scores'].shape
torch.Size([1, 64, 64])
>>> result['scores'].min() >= 0
True
"""
```

### Narrative Examples

For more complex examples, use narrative style:

```python
"""
Examples
--------
Basic usage with statistical initialization:

>>> # Create detector and load background data
>>> from cuvis_ai_core.training import StatisticalTrainer
>>> detector = RXDetector()
>>> pipeline.add_node(detector)
>>> background = load_hyperspectral_data("background.npy")
>>>
>>> # Initialize with background statistics
>>> trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
>>> trainer.fit()  # Initializes detector
>>>
>>> # Process test data
>>> test_data = load_hyperspectral_data("test.npy")
>>> result = detector.forward(test_data)
>>>
>>> # Apply threshold for binary decisions
>>> decisions = result['scores'] > 0.5

Complete pipeline example:

>>> from cuvis_ai.anomaly import RXDetector
>>> from cuvis_ai.deciders import BinaryDecider
>>>
>>> # Build detection pipeline
>>> detector = RXDetector(use_global_covariance=True)
>>> decider = BinaryDecider(threshold=0.5)
>>>
>>> # Process data
>>> scores = detector.forward(data)
>>> decisions = decider.forward(scores)
"""
```

---

## Testing Docstrings

### Check Coverage

Use `interrogate` to measure docstring coverage:

```bash
# Check entire package
interrogate -v cuvis_ai/

# Check specific module
interrogate -vv cuvis_ai/anomaly/rx_detector.py

# Require 95% coverage
interrogate -v cuvis_ai/ --fail-under 95
```

### Run Doctests

Test that examples in docstrings actually work:

```bash
# Test single file
python -m doctest cuvis_ai/anomaly/rx_detector.py -v

# Test all files
pytest --doctest-modules cuvis_ai/
```

### Validate Style

Check docstring style compliance:

```bash
# Install pydocstyle
pip install pydocstyle

# Check style
pydocstyle cuvis_ai/
```

---

## Tools

### interrogate

Measures docstring coverage:

```bash
# Install
pip install interrogate

# Basic usage
interrogate -v cuvis_ai/

# Detailed report with missing items
interrogate -vv cuvis_ai/

# Generate badge
interrogate --generate-badge docs/badges/ cuvis_ai/

# Fail if coverage below threshold
interrogate -v cuvis_ai/ --fail-under 95
```

Configuration in `pyproject.toml`:

```toml
[tool.interrogate]
ignore-init-method = true
ignore-init-module = false
ignore-magic = false
ignore-semiprivate = false
ignore-private = false
ignore-property-decorators = false
ignore-module = false
ignore-nested-functions = false
ignore-nested-classes = true
ignore-setters = false
fail-under = 95
verbose = 1
```

### pydocstyle

Checks docstring style compliance:

```bash
# Install
pip install pydocstyle

# Check style
pydocstyle cuvis_ai/

# Check specific convention
pydocstyle --convention=numpy cuvis_ai/
```

### mkdocstrings

Generates API documentation from docstrings:

```bash
# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve

# Build with strict error checking
mkdocs build --strict
```

---

## Common Patterns

### Node Classes

Standard pattern for node docstrings:

```python
class MyDetectorNode(BaseNode):
    """
    Brief one-line description of what this node does.

    Extended description explaining the algorithm, when to use
    this node, and key characteristics.

    Parameters
    ----------
    param1 : type
        Description with valid range and purpose.
    param2 : type, optional
        Description with default behavior. Default is value.

    Attributes
    ----------
    requires_initial_fit : bool
        Whether statistical initialization is required.
    output_ports : Dict[str, Port]
        Output port specifications.

    See Also
    --------
    RelatedNode : Alternative approach
    PreprocessingNode : Recommended preprocessing

    Notes
    -----
    Implementation notes and algorithm details.

    References
    ----------
    .. [1] Algorithm paper citation.

    Examples
    --------
    >>> node = MyDetectorNode(param1=value)
    >>> result = node.forward(data)
    """
```

### forward() Methods

Standard pattern for forward method docstrings:

```python
def forward(self, **inputs: Any) -> Dict[str, Any]:
    """
    Process data through the node.

    Parameters
    ----------
    **inputs : Any
        Input data from connected ports. Expected keys:

        - 'data' : torch.Tensor
            Input data with shape ``(batch, channels, H, W)``.
        - 'mask' : torch.Tensor, optional
            Binary mask with shape ``(batch, 1, H, W)``.

    Returns
    -------
    dict
        Output data for connected ports:

        - 'output' : torch.Tensor
            Processed data with shape ``(batch, out_channels, H, W)``.
        - 'scores' : torch.Tensor
            Computed scores with shape ``(batch, H, W)``.

    Raises
    ------
    ValueError
        If required inputs are missing or have invalid shapes.

    Examples
    --------
    >>> result = node.forward(data=input_tensor)
    >>> result['output'].shape
    torch.Size([1, 10, 64, 64])
    """
```

---

## See Also

- [Numpy Docstring Guide](https://numpydoc.readthedocs.io/) - Official NumPy style guide
- [Node Development](../how-to/add-builtin-node.md) - How to create new nodes
- [Contributing Guidelines](contributing.md) - Contribution workflow
- [API Reference](../api/index.md) - Auto-generated API documentation

---

## Related Pages

- [Development Overview](index.md)
- [Contributing Guide](contributing.md)
- [Git Hooks](git-hooks.md)
