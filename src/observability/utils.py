from __future__ import annotations

# AI-assisted maintenance note:
# This module holds the smallest cross-cutting observability helpers.
#
# These functions intentionally stay tiny and generic because they are reused
# by runtime setup, callbacks, and reporting. Keeping them here avoids either:
# - duplicating simple dependency/filesystem checks in several modules, or
# - creating awkward import direction where unrelated modules depend on each
#   other just to borrow a two-line helper.

import importlib.util
from pathlib import Path


def _has_module(module_name: str) -> bool:
    """
    Check whether an optional dependency is importable.

    Context:
    the observability stack enables many features conditionally, so dependency
    discovery needs to be lightweight and side-effect free.
    """
    # We use importlib-based checks instead of direct imports when the goal is
    # simply "is this optional feature available?" That lets the package decide
    # whether to enable a capability without crashing eagerly when an optional
    # dependency is absent.
    return importlib.util.find_spec(module_name) is not None


def _ensure_parent(path: Path | None) -> None:
    """
    Create the parent directory for a file path when needed.

    Context:
    many observability artifacts are files rather than directories, so callers
    often need the parent created without assuming the file already exists.
    """
    # Many artifact paths in observability are file paths, not directory
    # paths. This helper ensures the parent folder exists before we attempt to
    # write the file itself.
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_dir(path: Path | None) -> None:
    """
    Create a directory path when needed.

    Context:
    this is the directory-oriented counterpart to `_ensure_parent(...)`.
    """
    # Counterpart to `_ensure_parent(...)` for fields that already represent a
    # directory rather than a file.
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
