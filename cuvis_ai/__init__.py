import importlib

from cuvis_ai._version import __version__, get_version

_submodules = [
    "anomaly",
    "data",
    "deciders",
    "distance",
    "node",
    "pipeline",
    "preprocessor",
    "supervised",
    "test",
    "transformation",
    "tv_transforms",
    "unsupervised",
    "utils",
]


def __dir__():
    return _submodules


def __getattr__(name):
    if name in _submodules:
        return importlib.import_module(f"cuvis_ai.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'cuvis_ai' has no attribute '{name}'")
