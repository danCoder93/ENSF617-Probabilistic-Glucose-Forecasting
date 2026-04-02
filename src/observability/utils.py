from __future__ import annotations

# AI-assisted maintenance note:
# This module holds the smallest cross-cutting observability helpers.
#
# Responsibility boundary:
# - perform tiny dependency/file-system utility tasks shared by observability
#   setup, callbacks, and reporting
# - keep those tasks centralized so the larger modules can stay focused on
#   their higher-level behavior
#
# These functions intentionally stay tiny and generic because they are reused
# by runtime setup, callbacks, and reporting. Keeping them here avoids either:
# - duplicating simple dependency/filesystem checks in several modules, or
# - creating awkward import direction where unrelated modules depend on each
#   other just to borrow a two-line helper.
#
# What does *not* live here:
# - logger construction
# - callback behavior
# - report generation
# - tensor/batch normalization helpers
#
# Important disclaimer:
# because these helpers are intentionally tiny, the comments around them carry
# more architectural value than the code itself. Their purpose is not to be
# clever; it is to keep the rest of the observability package boring and
# decoupled.

import importlib.util
from pathlib import Path


# ============================================================================
# Optional Dependency Discovery
# ============================================================================
# Observability features are heavily optional, so callers often need to ask
# "is package X available?" without importing it eagerly.
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


# ============================================================================
# Filesystem Preparation
# ============================================================================
# Observability code writes a mix of individual files and whole directories.
# Keeping those cases separate avoids a pile of repeated `mkdir(...)` call
# sites in the larger runtime/reporting modules.
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
