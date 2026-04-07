from __future__ import annotations

# AI-assisted maintenance note:
# This module owns conversion of structured evaluation outputs into stable
# tabular surfaces for the repository's post-run reporting layer.
#
# Why this file exists:
# - grouped evaluation rows are already canonical on the evaluation side, but
#   downstream reporting sinks often prefer DataFrame surfaces
# - that translation used to live alongside unrelated builder logic in
#   `builders.py`
# - splitting it out keeps grouped-table packaging responsibilities explicit and
#   makes later table-specific enhancements easier to reason about
#
# Responsibility boundary:
# - flatten structured metric summaries into scalar maps when needed
# - convert grouped evaluation rows into stable DataFrame surfaces
# - apply lightweight table normalization/sorting helpers for deterministic
#   downstream use
#
# What does *not* live here:
# - row-level prediction table construction
# - new metric truth computation
# - dashboard/report narrative generation
# - sink-specific export logic
#
# In other words, this module packages already-canonical evaluation outputs into
# tabular or scalar reporting surfaces without becoming a second evaluation
# layer.

from typing import Any, Sequence

import pandas as pd

from evaluation import GroupedMetricRow, MetricSummary


def metric_summary_to_scalars(summary: MetricSummary | None) -> dict[str, float | int | None]:
    """
    Flatten a structured metric summary into a sink-friendly scalar dictionary.

    Context:
    grouped tables are useful for richer analysis, but many downstream sinks
    benefit from one flat scalar map.

    Why keep this helper separate:
    converting the structured evaluation dataclass into a plain scalar map is a
    packaging concern, not an evaluation concern. Keeping that translation here
    avoids leaking sink-specific naming choices back into the evaluation layer.
    """
    if summary is None:
        return {}

    scalars: dict[str, float | int | None] = {
        "count": summary.count,
        "mae": summary.mae,
        "rmse": summary.rmse,
        "bias": summary.bias,
        "overall_pinball_loss": summary.overall_pinball_loss,
        "mean_interval_width": summary.mean_interval_width,
        "empirical_interval_coverage": summary.empirical_interval_coverage,
    }

    # Preserve the evaluation layer's existing per-quantile pinball keys while
    # flattening them into the shared report's scalar namespace.
    for quantile_key, value in summary.pinball_loss_by_quantile.items():
        scalars[f"pinball_loss_{quantile_key}"] = value

    return scalars


def grouped_rows_to_frame(rows: Sequence[GroupedMetricRow]) -> pd.DataFrame:
    """
    Convert grouped evaluation rows into a stable tabular surface.

    Context:
    grouped rows are already canonical on the evaluation side, but sinks such as
    CSV/HTML/notebooks usually prefer a DataFrame-like surface.

    Design note:
    empty grouped outputs still return a DataFrame with a stable schema. That
    makes downstream code simpler because it can depend on the column contract
    even when the table contains zero rows.
    """
    if not rows:
        # Build the empty frame from an empty row list plus an explicit schema.
        # This is equivalent at runtime to passing `columns=[...]` directly, but
        # it is friendlier to static type checkers for the pandas constructor.
        return pd.DataFrame.from_records(
            [],
            columns=[
                "group_name",
                "group_value",
                "count",
                "mae",
                "rmse",
                "bias",
                "overall_pinball_loss",
                "mean_interval_width",
                "empirical_interval_coverage",
            ],
        )

    # Preserve the evaluation package as the source of truth for the actual row
    # values. This helper only repackages those values into a predictable frame.
    return pd.DataFrame(
        [
            {
                "group_name": row.group_name,
                "group_value": row.group_value,
                "count": row.count,
                "mae": row.mae,
                "rmse": row.rmse,
                "bias": row.bias,
                "overall_pinball_loss": row.overall_pinball_loss,
                "mean_interval_width": row.mean_interval_width,
                "empirical_interval_coverage": row.empirical_interval_coverage,
            }
            for row in rows
        ]
    )


def sorted_grouped_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return a grouped metrics frame sorted by its canonical group axis.

    Context:
    grouped evaluation tables use a repository-stable `group_value` column for
    the x-axis or grouping key. Sorting once here keeps textual summaries and
    table-oriented sinks deterministic and avoids duplicated normalization logic.
    """
    if frame.empty or "group_value" not in frame.columns:
        return frame.copy()

    # Sorting by `group_value` keeps horizon summaries monotonic and also makes
    # subject/range summaries deterministic for repeated runs and tests.
    return frame.copy().sort_values(by=["group_value"]).reset_index(drop=True)


def format_optional_metric(value: Any) -> str:
    """Format a scalar-like metric value for deterministic report text.

    Context:
    report text should stay concise and stable across sinks. This helper keeps
    optional numeric formatting consistent anywhere the reporting package needs
    to mention a metric that may legitimately be absent.
    """
    if value is None:
        return "unavailable"

    if pd.isna(value):
        return "unavailable"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
