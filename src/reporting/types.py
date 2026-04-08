from __future__ import annotations

# AI-assisted maintenance note:
# This module defines the core in-memory reporting contracts used by the
# repository's post-run reporting layer.
#
# Why these contracts live in their own file:
# - they are shared by multiple reporting helpers and sinks
# - they are conceptually the most stable part of the reporting package
# - keeping them separate makes it easier to expand builders/exports/HTML/TensorBoard
#   sinks later without forcing those files to redefine the same types
#
# Responsibility boundary:
# - define the canonical shared-report dataclass
# - define the minimal dataloader contract needed by reporting builders
#
# What does *not* live here:
# - row-building logic
# - dataframe construction
# - CSV export
# - Plotly/TensorBoard rendering
#
# In other words, this file defines what the reporting layer *passes around*,
# not how those payloads are computed or rendered.

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


class TestDataloaderProvider(Protocol):
    """
    Minimal contract needed by post-run prediction export/reporting.

    Context:
    the reporting/export path only needs access to the held-out test batches,
    not the full concrete `AZT1DDataModule` API. Keeping the type boundary this
    small makes the helpers easier to test and more accurate about what they
    truly depend on.

    Design intent:
    this protocol deliberately exposes only one method. That keeps the shared
    reporting path honest about its dependency surface and avoids implicitly
    coupling the post-run artifact code to unrelated datamodule features.
    """

    def test_dataloader(self) -> Any:
        """Return the held-out test dataloader or iterable of test batches used for export."""
        ...


@dataclass(frozen=True)
class SharedReport:
    """
    Canonical in-memory report bundle for one post-run prediction analysis.

    Purpose:
    package the most common post-run reporting surfaces into one structured
    object so export and visualization sinks can share the same source of
    truth.

    Context:
    this is intentionally *not* a replacement for `EvaluationResult`.
    `EvaluationResult` remains the canonical detailed metric contract produced
    by the evaluation package. `SharedReport` is the packaging layer that turns
    raw predictions plus evaluation outputs into sink-friendly tables, scalars,
    lightweight narrative text, and figure-ready data.

    Design note:
    keeping these fields as plain dictionaries of familiar Python / pandas
    objects makes the report easy to inspect in notebooks, serialize in simple
    ways, and extend later without forcing a new complex dependency.
    """

    # Scalar summaries that are easy to log, compare, or serialize.
    #
    # Examples:
    # - top-line MAE/RMSE/bias values
    # - counts such as number of rows or number of subjects
    # - interval summary values when quantile forecasts are available
    scalars: dict[str, float | int | None] = field(default_factory=dict)

    # Canonical tabular surfaces for downstream exports and analysis.
    #
    # These tables are where most sinks should start. CSV export writes one of
    # these tables directly; future JSON or dashboard sinks can inspect them
    # without having to reconstruct rows from raw prediction tensors.
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Lightweight narrative summaries for dashboards, logs, or future text
    # sinks.
    #
    # The goal is not to generate a polished report essay here. Instead, this
    # field provides compact factual text that can be surfaced in TensorBoard,
    # summaries, or notebooks without making every caller hand-roll its own
    # textual description of the same artifacts.
    text: dict[str, str] = field(default_factory=dict)

    # Figure-ready placeholders or lightweight plot input structures.
    #
    # Keeping this field broad for now allows the package to evolve toward
    # richer sink-specific rendering without forcing a premature figure object
    # contract across the whole reporting layer.
    figures: dict[str, Any] = field(default_factory=dict)

    # Metadata describing how the report was built and what it contains.
    #
    # This field is useful for provenance and later sink logic. It lets callers
    # understand which quantiles were used, how many batches were packaged, and
    # whether the report includes structured evaluation outputs.
    metadata: dict[str, Any] = field(default_factory=dict)