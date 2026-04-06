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
    """
    Export the canonical shared-report prediction table to CSV.

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

    Parameters:
        shared_report:
            The canonical in-memory reporting bundle whose `prediction_table`
            should be serialized.
        output_path:
            Destination CSV path. When `None`, the export is skipped.

    Returns:
        The written CSV path when an export occurs, otherwise `None`.

    Behavior:
        - returns early when no output path is configured
        - returns early when the canonical prediction table is empty
        - creates parent directories on demand
        - writes only the already-built canonical `prediction_table`
    """
    # This helper is the preferred export path because it consumes the already-
    # packaged shared report directly. That keeps the export sink aligned with
    # the repository's canonical reporting lifecycle instead of rebuilding the
    # report surface again from raw inputs.
    if output_path is None:
        return None

    output_path = Path(output_path)

    # The canonical flat prediction table is the only table this CSV sink
    # serializes. If the report does not contain rows, there is nothing useful
    # to write and the export is skipped cleanly.
    prediction_table = shared_report.tables.get("prediction_table")
    if prediction_table is None or prediction_table.empty:
        return None

    # Filesystem policy belongs in the sink, not in the report builder. The
    # builder should stay focused on in-memory packaging while this sink owns
    # parent-directory creation and final CSV serialization.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(output_path, index=False)
    return output_path


def export_prediction_table(
    *,
    datamodule: TestDataloaderProvider,
    predictions: Sequence[Tensor],
    quantiles: Sequence[float],
    output_path: PathInput | None,
    sampling_interval_minutes: int,
    evaluation_result: EvaluationResult | None = None,
) -> Path | None:
    """
    Export test predictions as a flat analysis-friendly CSV table.

    Context:
    the raw tensor dump preserves fidelity, while this table optimizes for
    plotting, inspection, spreadsheet-style analysis, and downstream reporting.

    Design note:
    this function now acts as a compatibility wrapper:
    - it builds the canonical shared report from raw inputs
    - it forwards the actual CSV write to
      `export_prediction_table_from_report(...)`
    - it preserves the historical call shape for older callers while keeping
      the real sink logic on the report-based path

    Parameters:
        datamodule:
            Provider of the held-out test dataloader whose batches align with
            the provided prediction tensors.
        predictions:
            Post-run prediction batches, typically produced by the workflow's
            `predict_test(...)` path.
        quantiles:
            Ordered forecast quantiles used to label exported prediction columns.
        output_path:
            Destination CSV path. When `None`, the export is skipped.
        sampling_interval_minutes:
            Step size used to reconstruct horizon timestamps in the canonical
            prediction table.
        evaluation_result:
            Optional structured evaluation output. This is forwarded into the
            shared-report builder so the export path can stay aligned with the
            same canonical post-run packaging surface used by other sinks.

    Returns:
        The written CSV path when an export occurs, otherwise `None`.

    Behavior:
        - returns early when no output path is configured
        - returns early when no prediction batches exist
        - builds the canonical `SharedReport`
        - delegates the actual CSV write to the strict report-based sink
    """
    # This wrapper deliberately preserves the old public call shape so existing
    # tests, scripts, and ad hoc callers do not have to switch all at once.
    # The important architectural change is that the actual write path now
    # happens through `export_prediction_table_from_report(...)`.
    if output_path is None:
        return None

    # If no prediction batches exist, building a shared report would only
    # produce an empty prediction table. Returning early keeps the historical
    # behavior unchanged and avoids unnecessary work.
    if not predictions:
        return None

    # Canonical report construction remains centralized in the reporting
    # builders module. This wrapper does not recreate row logic itself; it
    # simply asks the builder for the shared report and then forwards that
    # report into the strict sink path.
    report = build_shared_report(
        datamodule=datamodule,
        predictions=predictions,
        quantiles=quantiles,
        sampling_interval_minutes=sampling_interval_minutes,
        evaluation_result=evaluation_result,
    )

    # Delegate the final CSV write to the report-based sink so the real export
    # logic stays centralized in one place.
    return export_prediction_table_from_report(
        shared_report=report,
        output_path=output_path,
    )