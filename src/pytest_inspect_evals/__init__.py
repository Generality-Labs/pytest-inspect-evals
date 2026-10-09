"""pytest plugin with shared test gates, fixtures and helpers for Inspect AI evaluations"""

from importlib.metadata import PackageNotFoundError, version

# The version is set once, in pyproject.toml; this reads it back from the
# installed package metadata.
try:
    __version__ = version("pytest-inspect-evals")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "unknown"
