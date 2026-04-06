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
from reporting.types import TestDataloaderProvider


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
    this function is intentionally a thin sink:
    - it delegates canonical table construction to `build_shared_report(...)`
    - it serializes the resulting prediction table to disk
    - it does not silently become a second computation layer

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
        - creates parent directories on demand
        - writes the canonical `prediction_table` from `SharedReport`
    """
    # This export deliberately denormalizes prediction results into one flat
    # row-per-horizon table because that format is easy to inspect in a
    # notebook, easy to plot with pandas/Plotly, and easy to archive as a run
    # artifact.
    #
    # It complements rather than replaces the raw tensor dump used elsewhere:
    # - raw `.pt` files preserve full tensor fidelity for PyTorch consumers
    # - this CSV prioritizes analysis convenience
    if output_path is None:
        return None

    output_path = Path(output_path)
    if not predictions:
        return None

    report = build_shared_report(
        datamodule=datamodule,
        predictions=predictions,
        quantiles=quantiles,
        sampling_interval_minutes=sampling_interval_minutes,
        evaluation_result=evaluation_result,
    )
    prediction_table = report.tables["prediction_table"]

    # Filesystem policy belongs in the sink, not in the shared-report builder.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(output_path, index=False)
    return output_path
