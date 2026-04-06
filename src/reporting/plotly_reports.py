from __future__ import annotations

# AI-assisted maintenance note:
# This module contains the lightweight Plotly HTML reporting sink for the
# repository's post-run reporting layer.
#
# Why this file exists:
# - the reporting package now distinguishes between canonical report building
#   and sink-specific rendering
# - Plotly/HTML report generation is useful, but it is not the source of truth
#   for metrics or packaged report structure
# - separating the HTML sink into its own file makes it much easier to later
#   add TensorBoard and other sinks without mixing all rendering paths together
#
# Responsibility boundary:
# - consume a prebuilt `SharedReport` when available
# - fall back to a prediction-table CSV only for compatibility / lighter flows
# - render a small set of immediately useful HTML diagnostics
#
# What does *not* live here:
# - canonical metric computation
# - shared-report row construction
# - filesystem export of generic tabular artifacts
# - runtime callback logging
#
# In other words, this file is a presentation sink. It should visualize already
# packaged reporting surfaces, not quietly become a second metric engine.

from pathlib import Path
from typing import Any

import pandas as pd

from config import PathInput
from evaluation import EvaluationResult
from observability.utils import _has_module

from reporting.types import SharedReport


def _build_horizon_metrics_frame(
    *,
    prediction_table: pd.DataFrame,
    evaluation_result: EvaluationResult | None,
) -> pd.DataFrame:
    """
    Build the horizon-metrics frame used by the lightweight Plotly sink.

    Context:
    the repository already has a canonical structured evaluation surface. When
    that surface is available, the Plotly sink should prefer it. However, the
    HTML reports have historically also worked from the flat prediction table,
    so this helper preserves that compatibility fallback.

    Preference order:
    1. use grouped horizon metrics from `EvaluationResult`
    2. otherwise derive a simpler horizon summary from the prediction table

    Why this helper exists:
    the fallback logic is sink-specific. It belongs in the HTML-report module,
    not in the canonical shared-report builder or evaluation package.
    """
    if evaluation_result is not None and evaluation_result.by_horizon:
        # Structured grouped evaluation is the preferred source because it is
        # the repository's canonical detailed-metric surface.
        return pd.DataFrame(
            {
                "horizon_index": [row.group_value for row in evaluation_result.by_horizon],
                "mae": [row.mae for row in evaluation_result.by_horizon],
                "rmse": [row.rmse for row in evaluation_result.by_horizon],
                "mean_interval_width": [
                    row.mean_interval_width for row in evaluation_result.by_horizon
                ],
            }
        )

    # The fallback path intentionally keeps the report sink usable even in
    # lighter workflows where only the flat prediction table is available.
    #
    # We compute a small set of horizon-wise diagnostics because "how does
    # error grow as forecasting looks farther ahead?" is one of the most useful
    # first-pass views for sequence forecasting.
    grouped = prediction_table.assign(abs_error=lambda data: data["residual"].abs()).groupby(
        "horizon_index",
        as_index=False,
    )

    aggregation: dict[str, Any] = {
        "mae": ("abs_error", "mean"),
        "rmse": ("residual", lambda values: float((values.pow(2).mean()) ** 0.5)),
    }
    if "prediction_interval_width" in prediction_table.columns:
        aggregation["mean_interval_width"] = ("prediction_interval_width", "mean")

    return grouped.agg(**aggregation)


def generate_plotly_reports(
    prediction_table_path: PathInput | None,
    *,
    report_dir: PathInput | None,
    max_subjects: int,
    evaluation_result: EvaluationResult | None = None,
    shared_report: SharedReport | None = None,
) -> dict[str, Path]:
    """
    Generate lightweight Plotly HTML reports from the exported prediction table
    or a precomputed shared report.

    Purpose:
    leave behind a few immediately useful human-facing diagnostics after a run
    without requiring a custom notebook for basic inspection.

    Context:
    this sink is intentionally lightweight:
    - it is useful for quick post-run browsing
    - it is not the canonical source of metric truth
    - it should remain optional and best-effort
    - it should consume the reporting layer rather than redefining it

    Preferred input:
    - when `shared_report` is provided, use its canonical packaged tables
    - otherwise, fall back to reading the exported prediction-table CSV

    Returns:
        A mapping from report name to written HTML path.

    Behavior:
    - returns an empty mapping when Plotly is unavailable
    - returns an empty mapping when there is no usable prediction table
    - writes a small number of stable report artifacts into `report_dir`
    """
    # These reports are intentionally first-pass diagnostics rather than a full
    # experiment reporting system. The aim is to generate a few reliable HTML
    # artifacts automatically so each run leaves behind something visual and
    # shareable.
    if report_dir is None:
        return {}

    report_dir = Path(report_dir)

    # Plotly is optional by design. Missing the dependency should degrade
    # gracefully instead of breaking the rest of the workflow.
    if not _has_module("plotly"):
        return {}

    frame: pd.DataFrame
    if shared_report is not None:
        # Prefer the canonical in-memory shared report when it already exists.
        # This avoids redundant disk reads and keeps the sink aligned with the
        # reporting package's "build once, render many ways" design.
        frame = shared_report.tables.get("prediction_table", pd.DataFrame()).copy()
    else:
        if prediction_table_path is None:
            return {}

        prediction_table_path = Path(prediction_table_path)
        if not prediction_table_path.exists():
            return {}

        # This path-based fallback preserves the historical usage style so
        # lighter or transitional workflows do not have to provide a
        # `SharedReport` object yet.
        frame = pd.read_csv(prediction_table_path)

    if frame.empty:
        return {}

    import plotly.express as px
    import plotly.graph_objects as go

    report_dir.mkdir(parents=True, exist_ok=True)
    report_paths: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # 1. Residual histogram
    # ------------------------------------------------------------------
    # Residual distribution is one of the fastest sanity checks for:
    # - broad bias direction
    # - outlier behavior
    # - whether errors are tightly concentrated or widely spread
    #
    # It is intentionally cheap to compute, so every run can leave behind at
    # least one immediately useful error view.
    residual_histogram = px.histogram(
        frame,
        x="residual",
        nbins=50,
        title="Residual Distribution",
    )
    residual_histogram_path = report_dir / "residual_histogram.html"
    residual_histogram.write_html(str(residual_histogram_path))
    report_paths["residual_histogram"] = residual_histogram_path

    # ------------------------------------------------------------------
    # 2. Horizon metrics figure
    # ------------------------------------------------------------------
    # Prefer the structured evaluation result when it exists because it is the
    # repository's canonical detailed-metric surface. The fallback still keeps
    # the HTML sink useful when only flat prediction rows are available.
    horizon_metrics = _build_horizon_metrics_frame(
        prediction_table=frame,
        evaluation_result=evaluation_result,
    )

    horizon_metrics_fig = go.Figure()
    horizon_metrics_fig.add_trace(
        go.Scatter(
            x=horizon_metrics["horizon_index"],
            y=horizon_metrics["mae"],
            mode="lines+markers",
            name="MAE",
        )
    )
    horizon_metrics_fig.add_trace(
        go.Scatter(
            x=horizon_metrics["horizon_index"],
            y=horizon_metrics["rmse"],
            mode="lines+markers",
            name="RMSE",
        )
    )

    # When interval-width information exists, surface it on a secondary axis so
    # uncertainty-width growth can be visually compared with point-error growth
    # without collapsing unlike scales into one axis.
    if "mean_interval_width" in horizon_metrics.columns and not horizon_metrics[
        "mean_interval_width"
    ].isna().all():
        horizon_metrics_fig.add_trace(
            go.Scatter(
                x=horizon_metrics["horizon_index"],
                y=horizon_metrics["mean_interval_width"],
                mode="lines+markers",
                name="Mean Interval Width",
                yaxis="y2",
            )
        )
        horizon_metrics_fig.update_layout(
            yaxis2=dict(
                title="Interval Width",
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )

    horizon_metrics_fig.update_layout(title="Error Metrics By Forecast Horizon")
    horizon_metrics_path = report_dir / "horizon_metrics.html"
    horizon_metrics_fig.write_html(str(horizon_metrics_path))
    report_paths["horizon_metrics"] = horizon_metrics_path

    # ------------------------------------------------------------------
    # 3. Forecast overview figure
    # ------------------------------------------------------------------
    # This is the most qualitative sink artifact. It overlays:
    # - true target values
    # - median prediction
    # - simple prediction interval band when available
    #
    # Subject count is capped intentionally so the output stays readable and
    # does not balloon in size for larger runs.
    overview_fig = go.Figure()

    subject_ids = list(dict.fromkeys(frame["subject_id"].tolist()))[:max_subjects]
    filtered = frame[frame["subject_id"].isin(subject_ids)].copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
    filtered.sort_values(["subject_id", "timestamp"], inplace=True)

    for subject_id in subject_ids:
        subject_frame = filtered[filtered["subject_id"] == subject_id]
        if subject_frame.empty:
            continue

        # Plot the observed target first so all forecast traces are interpreted
        # relative to the real glucose trajectory.
        overview_fig.add_trace(
            go.Scatter(
                x=subject_frame["timestamp"],
                y=subject_frame["target"],
                mode="lines",
                name=f"{subject_id} target",
            )
        )
        overview_fig.add_trace(
            go.Scatter(
                x=subject_frame["timestamp"],
                y=subject_frame["median_prediction"],
                mode="lines",
                name=f"{subject_id} median",
            )
        )

        quantile_columns = sorted(
            column for column in subject_frame.columns if column.startswith("pred_q")
        )
        if len(quantile_columns) >= 2:
            lower = quantile_columns[0]
            upper = quantile_columns[-1]

            # Plotly interval fill uses two traces: an upper bound followed by a
            # lower bound filled to the previous trace. Keeping this explicit
            # makes the intent easier to understand during maintenance.
            overview_fig.add_trace(
                go.Scatter(
                    x=subject_frame["timestamp"],
                    y=subject_frame[upper],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            overview_fig.add_trace(
                go.Scatter(
                    x=subject_frame["timestamp"],
                    y=subject_frame[lower],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    opacity=0.2,
                    name=f"{subject_id} interval",
                )
            )

    overview_fig.update_layout(title="Forecast Overview")
    overview_path = report_dir / "forecast_overview.html"
    overview_fig.write_html(str(overview_path))
    report_paths["forecast_overview"] = overview_path

    return report_paths
