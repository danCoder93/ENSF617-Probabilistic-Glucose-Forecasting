from __future__ import annotations

# AI-assisted maintenance note:
# This module contains filesystem-oriented export sinks for the repository's
# post-run reporting layer.
#
# Why this file exists:
# - the reporting package now distinguishes between "build one canonical shared
#   report" and "serialize part of that report to disk"
# - keeping export sinks separate from report builders makes it easier to reason
#   about what is source-of-truth computation versus what is persistence
# - this keeps CSV-writing logic from spreading across workflow code or richer
#   visualization sinks
#
# Responsibility boundary:
# - consume `SharedReport` or the inputs needed to build one
# - serialize analysis-friendly artifacts to disk
# - keep filesystem policy local to export helpers
#
# What does *not* live here:
# - shared-report row construction
# - grouped metric computation
# - Plotly / TensorBoard rendering
#
# In other words, this file writes already-packaged reporting surfaces out to
# persistent artifact files.

from pathlib import Path
from typing import Sequence

from torch import Tensor

from config import PathInput
from evaluation import EvaluationResult
from reporting.builders import build_shared_report
from reporting.types import SharedReport, TestDataloaderProvider


def export_prediction_table_from_report(
    *,
    shared_report: SharedReport,
    output_path: PathInput | None,
) -> Path | None:
    """Export the canonical shared-report prediction table to CSV.

    Purpose:
    provide the strict canonical sink path for prediction-table export once a
    `SharedReport` has already been built by the workflow or another caller.

    Context:
    the repository now prefers the "build once, consume many ways" reporting
    lifecycle:
    - evaluation computes metric truth
    - builders package that truth into `SharedReport`
    - sinks such as CSV, Plotly, and TensorBoard consume the already-built
      report rather than quietly recomputing it
    """
    if output_path is None:
        return None

    output_path = Path(output_path)

    # The canonical flat prediction table is the only table this helper writes.
    # If it is absent or empty, there is nothing useful to serialize.
    prediction_table = shared_report.tables.get("prediction_table")
    if prediction_table is None or prediction_table.empty:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(output_path, index=False)
    return output_path


def export_grouped_tables_from_report(
    *,
    shared_report: SharedReport,
    output_dir: PathInput | None,
) -> dict[str, Path]:
    """Export grouped canonical report tables into a directory of CSV files.

    Context:
    the canonical reporting payload already includes grouped tables for horizon,
    subject, and glucose-range views. Persisting those tables makes offline
    spreadsheet analysis easier while preserving the same source-of-truth path
    used by the HTML and TensorBoard sinks.

    Behavior:
    - returns an empty mapping when no output directory is configured
    - skips grouped tables that are absent or empty
    - writes only tables already present in `SharedReport.tables`
    - never recomputes grouped metrics from the flat prediction table
    """
    if output_dir is None:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: dict[str, Path] = {}

    # File names are intentionally stable so downstream tooling can rely on
    # them without needing to inspect the report contents first.
    for table_name, filename in (
        ("by_horizon", "by_horizon.csv"),
        ("by_subject", "by_subject.csv"),
        ("by_glucose_range", "by_glucose_range.csv"),
    ):
        table = shared_report.tables.get(table_name)
        if table is None or table.empty:
            continue

        output_path = output_dir / filename
        table.to_csv(output_path, index=False)
        written_paths[table_name] = output_path

    return written_paths


def export_prediction_table(
    *,
    datamodule: TestDataloaderProvider,
    predictions: Sequence[Tensor],
    quantiles: Sequence[float],
    output_path: PathInput | None,
    sampling_interval_minutes: int,
    evaluation_result: EvaluationResult | None = None,
) -> Path | None:
    """Export test predictions as a flat analysis-friendly CSV table.

    Context:
    the raw tensor dump preserves fidelity, while this table optimizes for
    plotting, inspection, spreadsheet-style analysis, and downstream reporting.

    Design note:
    this function now acts as a compatibility wrapper:
    - it builds the canonical shared report from raw inputs
    - it forwards the actual CSV write to `export_prediction_table_from_report(...)`
    - it preserves the historical call shape for older callers while keeping
      the real sink logic on the report-based path
    """
    shared_report = build_shared_report(
        datamodule=datamodule,
        predictions=predictions,
        quantiles=quantiles,
        sampling_interval_minutes=sampling_interval_minutes,
        evaluation_result=evaluation_result,
    )

    return export_prediction_table_from_report(
        shared_report=shared_report,
        output_path=output_path,
    )
