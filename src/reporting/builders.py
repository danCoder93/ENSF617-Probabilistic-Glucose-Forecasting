from __future__ import annotations

# AI-assisted maintenance note:
# This module builds the canonical in-memory reporting surfaces for the
# repository's post-run reporting layer.
#
# Why this file exists:
# - the repository now distinguishes between "build one shared report" and
#   "render/export that report through different sinks"
# - keeping report construction separate from CSV/HTML/TensorBoard logic makes
#   the source-of-truth path easier to reason about and test
# - it also prevents export helpers from quietly becoming a second computation
#   layer
#
# Responsibility boundary:
# - build the flat prediction table from raw prediction batches + test batches
# - translate structured evaluation outputs into scalar/table/text surfaces
# - package those surfaces into `SharedReport`
#
# What does *not* live here:
# - CSV export
# - Plotly HTML report generation
# - TensorBoard sink logic
#
# In other words, this file computes the canonical shared-report payload once,
# and downstream sinks should consume that payload rather than recomputing it.

from typing import Any, Sequence

import pandas as pd
import torch
from torch import Tensor

from evaluation import (
    EvaluationResult,
    GroupedMetricRow,
    MetricSummary,
    select_point_prediction,
)
from observability.tensors import _as_metadata_lists

from reporting.types import SharedReport, TestDataloaderProvider


def _build_prediction_rows(
    *,
    datamodule: TestDataloaderProvider,
    predictions: Sequence[Tensor],
    quantiles: Sequence[float],
    sampling_interval_minutes: int,
) -> list[dict[str, Any]]:
    """
    Build the canonical flat row-per-horizon prediction table in memory.

    Context:
    the repository already relied on this denormalized table shape for CSV
    export and lightweight reports. The reporting package keeps that shape
    intact, but centralizes the row construction so downstream sinks do not
    each invent their own variant of the same logic.

    Important behavior:
    - aligns prediction batches with the original test dataloader batches
    - attaches metadata such as subject ID and decoder timing
    - emits one row per forecast horizon step for easy pandas use

    Why this shape is intentionally denormalized:
    a row-per-horizon table is not the most storage-efficient representation,
    but it is the most convenient shape for quick pandas aggregation, grouped
    plotting, CSV export, and manual inspection. That tradeoff is worthwhile in
    the reporting layer because clarity and reuse matter more here than raw
    compactness.
    """
    rows: list[dict[str, Any]] = []

    # Reuse the held-out test loader so prediction rows stay aligned with the
    # exact source metadata that produced them.
    test_loader = datamodule.test_dataloader()

    # Column names are derived deterministically from the configured quantiles.
    quantile_columns = [f"pred_q{int(round(q * 100)):02d}" for q in quantiles]

    for batch_index, (prediction_batch, batch) in enumerate(zip(predictions, test_loader)):
        # Post-run reporting works on CPU copies so downstream pandas/reporting
        # code stays device-agnostic and does not accidentally hold onto
        # accelerator memory.
        prediction_cpu = prediction_batch.detach().cpu()

        # Keep one representative point forecast column in addition to the
        # quantile columns. The repo's current default is the median.
        point_prediction_cpu = select_point_prediction(
            prediction_cpu,
            quantiles,
            quantile=0.5,
        )

        target = batch["target"]
        if isinstance(target, Tensor):
            target_cpu = target.detach().cpu()
        else:
            # Preserve compatibility with lighter or slightly older code paths
            # that may provide target-like values in non-tensor form.
            target_cpu = torch.as_tensor(target)

        # Normalize trailing singleton target dimensions into the canonical
        # [batch, horizon] table-friendly shape.
        if target_cpu.ndim == 3 and target_cpu.shape[-1] == 1:
            target_cpu = target_cpu.squeeze(-1)

        batch_size = int(prediction_cpu.shape[0])

        # Normalize metadata into predictable sample-level lists.
        metadata = _as_metadata_lists(batch["metadata"], batch_size)

        for sample_index in range(batch_size):
            subject_id = str(metadata.get("subject_id", ["unknown"])[sample_index])
            decoder_start = pd.Timestamp(
                str(
                    metadata.get("decoder_start", ["1970-01-01 00:00:00"])[sample_index]
                )
            )

            for horizon_index in range(int(prediction_cpu.shape[1])):
                timestamp = decoder_start + pd.Timedelta(
                    minutes=sampling_interval_minutes * horizon_index
                )
                row = {
                    "prediction_batch_index": batch_index,
                    "sample_index_within_batch": sample_index,
                    "subject_id": subject_id,
                    "decoder_start": str(metadata.get("decoder_start", [""])[sample_index]),
                    "decoder_end": str(metadata.get("decoder_end", [""])[sample_index]),
                    "timestamp": timestamp.isoformat(),
                    "horizon_index": horizon_index,
                    "target": float(target_cpu[sample_index, horizon_index].item()),
                }

                for quantile_index, column_name in enumerate(quantile_columns):
                    row[column_name] = float(
                        prediction_cpu[
                            sample_index,
                            horizon_index,
                            quantile_index,
                        ].item()
                    )

                row["median_prediction"] = float(
                    point_prediction_cpu[sample_index, horizon_index].item()
                )

                # Residual sign convention stays stable across the repo:
                # prediction - target.
                row["residual"] = row["median_prediction"] - row["target"]

                # Expose simple interval width directly when at least two
                # quantiles are available.
                if len(quantile_columns) >= 2:
                    row["prediction_interval_width"] = (
                        row[quantile_columns[-1]] - row[quantile_columns[0]]
                    )

                rows.append(row)

    return rows


def _metric_summary_to_scalars(summary: MetricSummary | None) -> dict[str, float | int | None]:
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

    for quantile_key, value in summary.pinball_loss_by_quantile.items():
        scalars[f"pinball_loss_{quantile_key}"] = value

    return scalars


def _grouped_rows_to_frame(rows: Sequence[GroupedMetricRow]) -> pd.DataFrame:
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
        return pd.DataFrame(
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
            ]
        )

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


def _build_report_text(
    *,
    prediction_table: pd.DataFrame,
    evaluation_result: EvaluationResult | None,
    quantiles: Sequence[float],
) -> dict[str, str]:
    """
    Build lightweight narrative text summaries for the shared report.

    Context:
    this reporting layer keeps text generation deliberately small and factual.
    The aim is to provide concise human-readable interpretation surfaces for
    later sinks without turning the reporting package into a heavyweight
    natural-language report system.
    """
    text: dict[str, str] = {}

    sample_count = len(prediction_table)
    subject_count = (
        int(prediction_table["subject_id"].nunique()) if "subject_id" in prediction_table else 0
    )
    horizon_count = (
        int(prediction_table["horizon_index"].nunique())
        if "horizon_index" in prediction_table
        else 0
    )
    text["dataset_overview"] = (
        "Shared report covers "
        f"{sample_count} forecast rows across {subject_count} subject(s) "
        f"and {horizon_count} horizon step(s)."
    )

    if evaluation_result is not None:
        summary = evaluation_result.summary

        coverage_text = (
            "unavailable"
            if summary.empirical_interval_coverage is None
            else f"{summary.empirical_interval_coverage:.4f}"
        )
        interval_text = (
            "unavailable"
            if summary.mean_interval_width is None
            else f"{summary.mean_interval_width:.4f}"
        )
        text["metric_overview"] = (
            "Detailed evaluation summary: "
            f"MAE={summary.mae:.4f}, RMSE={summary.rmse:.4f}, "
            f"bias={summary.bias:.4f}, "
            f"overall_pinball_loss={summary.overall_pinball_loss:.4f}, "
            f"mean_interval_width={interval_text}, "
            f"empirical_interval_coverage={coverage_text}."
        )
    else:
        text["metric_overview"] = (
            "Detailed evaluation summary was not available, so the shared report "
            "contains only prediction-table-derived reporting surfaces."
        )

    text["quantile_overview"] = (
        "Quantile configuration for this shared report: "
        + ", ".join(f"{float(q):.3f}" for q in quantiles)
    )

    return text


def build_shared_report(
    *,
    datamodule: TestDataloaderProvider,
    predictions: Sequence[Tensor],
    quantiles: Sequence[float],
    sampling_interval_minutes: int,
    evaluation_result: EvaluationResult | None = None,
) -> SharedReport:
    """
    Build the canonical in-memory shared report for one post-run prediction run.

    Purpose:
    compute once, package once, and let downstream sinks reuse the same report
    surfaces without redoing row construction or grouped-table assembly.

    Context:
    this is the core reporting-package builder. It does not replace CSV/HTML/
    TensorBoard outputs; it sits underneath them so those outputs can remain
    thin consumers of one shared report object.
    """
    if not predictions:
        return SharedReport(
            scalars={},
            tables={
                "prediction_table": pd.DataFrame(),
                "by_horizon": pd.DataFrame(),
                "by_subject": pd.DataFrame(),
                "by_glucose_range": pd.DataFrame(),
            },
            text={
                "dataset_overview": "Shared report is empty because no prediction batches were provided.",
                "metric_overview": "No evaluation summary is available because no prediction batches were provided.",
                "quantile_overview": "Quantile configuration is unavailable because no prediction batches were provided.",
            },
            figures={},
            metadata={
                "num_prediction_batches": 0,
                "quantiles": tuple(float(q) for q in quantiles),
                "sampling_interval_minutes": sampling_interval_minutes,
                "has_evaluation_result": evaluation_result is not None,
            },
        )

    rows = _build_prediction_rows(
        datamodule=datamodule,
        predictions=predictions,
        quantiles=quantiles,
        sampling_interval_minutes=sampling_interval_minutes,
    )
    prediction_table = pd.DataFrame(rows)

    # Preserve the evaluation package as the canonical metric truth.
    by_horizon = _grouped_rows_to_frame(
        () if evaluation_result is None else evaluation_result.by_horizon
    )
    by_subject = _grouped_rows_to_frame(
        () if evaluation_result is None else evaluation_result.by_subject
    )
    by_glucose_range = _grouped_rows_to_frame(
        () if evaluation_result is None else evaluation_result.by_glucose_range
    )

    scalars = _metric_summary_to_scalars(
        None if evaluation_result is None else evaluation_result.summary
    )
    scalars["num_prediction_batches"] = len(predictions)
    scalars["num_prediction_rows"] = len(prediction_table)
    scalars["num_subjects"] = (
        int(prediction_table["subject_id"].nunique()) if "subject_id" in prediction_table else 0
    )
    scalars["num_horizons"] = (
        int(prediction_table["horizon_index"].nunique())
        if "horizon_index" in prediction_table
        else 0
    )

    text = _build_report_text(
        prediction_table=prediction_table,
        evaluation_result=evaluation_result,
        quantiles=quantiles,
    )

    # Keep figure payloads intentionally lightweight for now. Concrete sinks can
    # consume these report tables/metadata and grow richer over time.
    figures: dict[str, Any] = {
        "available_plot_inputs": {
            "has_prediction_table": not prediction_table.empty,
            "has_by_horizon": not by_horizon.empty,
            "has_by_subject": not by_subject.empty,
            "has_by_glucose_range": not by_glucose_range.empty,
        }
    }

    metadata: dict[str, Any] = {
        "num_prediction_batches": len(predictions),
        "quantiles": tuple(float(q) for q in quantiles),
        "sampling_interval_minutes": sampling_interval_minutes,
        "has_evaluation_result": evaluation_result is not None,
        "prediction_table_columns": tuple(str(column) for column in prediction_table.columns),
    }

    return SharedReport(
        scalars=scalars,
        tables={
            "prediction_table": prediction_table,
            "by_horizon": by_horizon,
            "by_subject": by_subject,
            "by_glucose_range": by_glucose_range,
        },
        text=text,
        figures=figures,
        metadata=metadata,
    )