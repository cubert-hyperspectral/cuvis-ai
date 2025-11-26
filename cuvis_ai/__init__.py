from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cuvis_ai")
except PackageNotFoundError:
    # Package is not installed, likely in development mode
    __version__ = "dev"

__all__ = ["__version__"]
