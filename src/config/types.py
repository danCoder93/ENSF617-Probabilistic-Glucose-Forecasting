from __future__ import annotations

# AI-assisted maintenance note:
# This module holds small shared typing primitives for the configuration
# package. Keeping them isolated avoids import cycles between the domain
# specific config modules while still giving the rest of the package one common
# type vocabulary.

from pathlib import Path


# `Path` objects are the normalized internal representation used throughout the
# codebase, but several constructors intentionally accept plain strings because
# CLI code, notebooks, and tests often build configs from string paths first.
# Using one shared alias keeps that "accept strings, normalize to Path later"
# contract explicit in the type hints instead of relying only on `__post_init__`
# behavior.
PathInput = str | Path

