from __future__ import annotations

# AI-assisted maintenance note:
# This module builds the canonical in-memory reporting surfaces for the
# repository's post-run reporting layer.
#
# Why this file exists:
# - the repository distinguishes between "build one shared report" and
#   "render/export that report through different sinks"
# - this module is the small orchestrator that assembles those canonical
#   surfaces after predictions and structured evaluation already exist
#
# Responsibility boundary:
# - orchestrate canonical prediction-row construction
# - translate grouped evaluation outputs into tabular surfaces
# - package scalar/text/metadata/figure-ready surfaces into `SharedReport`
#
# What does *not* live here:
# - TensorBoard sink logic
# - Plotly HTML generation
# - CSV/JSON export policy
# - workflow orchestration
#
# In other words, this module builds the shared in-memory report once so sinks
# can consume it consistently.

from typing import Any, Mapping, Sequence

import pandas as pd
from torch import Tensor

from evaluation import EvaluationResult

from reporting.prediction_rows import build_prediction_rows
from reporting.report_tables import grouped_rows_to_frame, metric_summary_to_scalars
from reporting.report_text import build_report_text
from reporting.types import SharedReport, TestDataloaderProvider


def build_shared_report(
    *,
    datamodule: TestDataloaderProvider,
    predictions: Sequence[Tensor],
    quantiles: Sequence[float],
    sampling_interval_minutes: int,
    evaluation_result: EvaluationResult | None = None,
    data_summary: Mapping[str, Any] | None = None,
) -> SharedReport:
    """Build the canonical in-memory shared report for one post-run prediction run.

    Important compatibility rule:
    this function keeps the existing shared-report contract intact and only
    extends it with optional data-summary-aware text/metadata packaging when the
    workflow provides a DataModule summary.
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
                "dashboard_overview": "Dashboard overview is unavailable because no prediction batches were provided.",
                "health_warning_overview": "No high-level health summary is available because no prediction batches were provided.",
                "horizon_overview": "Forecast-horizon grouped metrics are unavailable because no prediction batches were provided.",
                "probabilistic_overview": "Probabilistic grouped interpretation is unavailable because no prediction batches were provided.",
                "subject_variability_overview": "Subject-level grouped metrics are unavailable because no prediction batches were provided.",
                "glucose_range_overview": "Glucose-range grouped metrics are unavailable because no prediction batches were provided.",
                "data_summary_overview": "Data-summary overview is unavailable because no prediction batches were provided.",
            },
            figures={},
            metadata={
                "num_prediction_batches": 0,
                "quantiles": tuple(float(q) for q in quantiles),
                "sampling_interval_minutes": sampling_interval_minutes,
                "has_evaluation_result": evaluation_result is not None,
                "has_data_summary": bool(data_summary),
            },
        )

    rows = build_prediction_rows(
        datamodule=datamodule,
        predictions=predictions,
        quantiles=quantiles,
        sampling_interval_minutes=sampling_interval_minutes,
    )
    prediction_table = pd.DataFrame(rows)

    by_horizon = grouped_rows_to_frame(
        () if evaluation_result is None else evaluation_result.by_horizon
    )
    by_subject = grouped_rows_to_frame(
        () if evaluation_result is None else evaluation_result.by_subject
    )
    by_glucose_range = grouped_rows_to_frame(
        () if evaluation_result is None else evaluation_result.by_glucose_range
    )

    scalars = metric_summary_to_scalars(
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

    if not prediction_table.empty:
        if "residual" in prediction_table.columns:
            residual_series = prediction_table["residual"]
            if not bool(residual_series.isna().all()):
                scalars["mean_abs_residual"] = float(residual_series.abs().mean())
                scalars["residual_std"] = float(residual_series.std(ddof=0))

        if "prediction_interval_width" in prediction_table.columns:
            width_series = prediction_table["prediction_interval_width"]
            if not bool(width_series.isna().all()):
                scalars["mean_prediction_interval_width_from_rows"] = float(
                    width_series.mean()
                )
                scalars["near_zero_interval_fraction"] = float(
                    (width_series.abs() <= 1e-8).mean()
                )

        if "target" in prediction_table.columns:
            target_series = prediction_table["target"]
            if not bool(target_series.isna().all()):
                scalars["target_mean"] = float(target_series.mean())
                scalars["target_std"] = float(target_series.std(ddof=0))

        if "median_prediction" in prediction_table.columns:
            prediction_series = prediction_table["median_prediction"]
            if not bool(prediction_series.isna().all()):
                scalars["prediction_mean"] = float(prediction_series.mean())
                scalars["prediction_std"] = float(prediction_series.std(ddof=0))

    text = build_report_text(
        prediction_table=prediction_table,
        evaluation_result=evaluation_result,
        quantiles=quantiles,
        by_horizon=by_horizon,
        by_subject=by_subject,
        by_glucose_range=by_glucose_range,
        data_summary=data_summary,
    )

    figures: dict[str, Any] = {
        "available_plot_inputs": {
            "has_prediction_table": not prediction_table.empty,
            "has_by_horizon": not by_horizon.empty,
            "has_by_subject": not by_subject.empty,
            "has_by_glucose_range": not by_glucose_range.empty,
        }
    }

    # Keep the raw data summary out of top-level text/scalar flattening. It is
    # stored as metadata only so sinks can reference it later without turning
    # builders into a second summary-computation layer.
    metadata: dict[str, Any] = {
        "num_prediction_batches": len(predictions),
        "quantiles": tuple(float(q) for q in quantiles),
        "sampling_interval_minutes": sampling_interval_minutes,
        "has_evaluation_result": evaluation_result is not None,
        "prediction_table_columns": tuple(str(column) for column in prediction_table.columns),
        "has_data_summary": bool(data_summary),
    }
    if data_summary:
        metadata["data_summary"] = data_summary

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
