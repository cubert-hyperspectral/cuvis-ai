"""
CUVIS.AI - Hyperspectral Anomaly Detection Framework.

This package provides a comprehensive framework for building and deploying
hyperspectral anomaly detection pipelines using both statistical and deep
learning methods. It includes nodes for data loading, preprocessing,
feature selection, anomaly detection, and visualization.

The framework is built on a pipeline architecture where nodes can be
composed together to create complex workflows for hyperspectral analysis.

See Also
--------
cuvis_ai.anomaly : Statistical and deep learning anomaly detection nodes
cuvis_ai.node : Core node implementations for data processing
cuvis_ai.deciders : Binary decision nodes for classification
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cuvis_ai")
except PackageNotFoundError:
    # Package is not installed, likely in development mode
    __version__ = "dev"

__all__ = ["__version__"]
