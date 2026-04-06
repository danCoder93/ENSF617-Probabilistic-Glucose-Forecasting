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

import pandas as pd

from config import PathInput
from observability.utils import _has_module

from reporting.types import SharedReport


def _plot_title(title: str, subtitle: str) -> str:
    """Return a compact human-readable Plotly title string.

    Context:
    Phase 6 focuses on interpretation polish rather than new metrics. A small
    title helper keeps the sink's pages more readable and more consistent with
    the TensorBoard terminology without changing the canonical reporting
    contract.
    """
    return f"{title}<br><sup>{subtitle}</sup>"


def _build_horizon_metrics_frame(
    *,
    shared_report: SharedReport | None,
) -> pd.DataFrame:
    """
    Build the horizon-metrics frame used by the lightweight Plotly sink.

    Context:
    the repository now treats grouped horizon metrics as canonical reporting
    surfaces that belong upstream in evaluation + shared-report packaging.

    Canonical rule for this sink:
    - horizon metrics must come from `SharedReport.tables["by_horizon"]`
    - this sink does *not* derive grouped horizon metrics from the flat
      prediction table
    - when canonical grouped horizon data is unavailable, the horizon-metrics
      artifact is omitted rather than recomputed here

    Why this helper exists:
    the sink still needs one small normalization layer because the grouped
    report table uses repository-stable column names such as `group_value`,
    while the Plotly figure wants an explicit horizon-axis column.
    """
    if shared_report is None:
        return pd.DataFrame()

    by_horizon = shared_report.tables.get("by_horizon", pd.DataFrame()).copy()
    if by_horizon.empty:
        return pd.DataFrame()

    required_columns = {"group_value", "mae", "rmse"}
    if not required_columns.issubset(set(by_horizon.columns)):
        return pd.DataFrame()

    horizon_metrics = by_horizon.rename(columns={"group_value": "horizon_index"}).copy()
    # Use explicit keyword arguments here because pandas' overloads are easier
    # for static analysis to resolve when both `by=` and `ascending=` are named.
    horizon_metrics = horizon_metrics.sort_values(
        by=["horizon_index"],
        ascending=True,
    ).reset_index(drop=True)

    # Keep only the columns the figures currently understand so the sink stays
    # resilient to future shared-report expansion.
    desired_columns = [
        column
        for column in (
            "horizon_index",
            "mae",
            "rmse",
            "bias",
            "overall_pinball_loss",
            "mean_interval_width",
            "empirical_interval_coverage",
        )
        if column in horizon_metrics.columns
    ]
    return horizon_metrics.loc[:, desired_columns]


def _build_grouped_metrics_frame(
    *,
    shared_report: SharedReport | None,
    table_name: str,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return one canonical grouped table after minimal Plotly normalization.

    Context:
    subject and glucose-range reports are already packaged upstream. The Plotly
    sink only needs a light validation and sorting layer before rendering them.
    """
    if shared_report is None:
        return pd.DataFrame()

    frame = shared_report.tables.get(table_name, pd.DataFrame()).copy()
    if frame.empty:
        return pd.DataFrame()

    if not set(required_columns).issubset(set(frame.columns)):
        return pd.DataFrame()

    # As above, prefer explicit keyword arguments to keep pandas sorting calls
    # friendlier to Pyright/Pylance static analysis.
    frame = frame.sort_values(
        by=["group_value"],
        ascending=True,
    ).reset_index(drop=True)
    return frame


def generate_plotly_reports(
    prediction_table_path: PathInput | None,
    *,
    report_dir: PathInput | None,
    max_subjects: int,
    shared_report: SharedReport | None = None,
) -> dict[str, Path]:
    """
    Generate lightweight Plotly HTML reports from the canonical shared report
    or a compatibility prediction-table CSV.

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

    Important canonical rule:
    - residual and forecast-overview figures may render from the flat
      prediction table because they are direct visualizations of row-level
      prediction data
    - grouped horizon, subject, and glucose-range figures must render only from
      canonical grouped report tables packaged inside `SharedReport`
    - when a required grouped table is unavailable, the corresponding artifact
      is omitted rather than recomputed here
    """
    if report_dir is None:
        return {}

    report_dir = Path(report_dir)

    if not _has_module("plotly"):
        return {}

    frame: pd.DataFrame
    if shared_report is not None:
        frame = shared_report.tables.get("prediction_table", pd.DataFrame()).copy()
    else:
        if prediction_table_path is None:
            return {}

        prediction_table_path = Path(prediction_table_path)
        if not prediction_table_path.exists():
            return {}

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
    residual_histogram = px.histogram(
        frame,
        x="residual",
        nbins=50,
        title=_plot_title(
            "Residual Distribution",
            "Row-level forecast errors across the held-out prediction table.",
        ),
    )
    residual_histogram.update_layout(
        xaxis_title="Residual",
        yaxis_title="Count",
    )
    residual_histogram_path = report_dir / "residual_histogram.html"
    residual_histogram.write_html(str(residual_histogram_path))
    report_paths["residual_histogram"] = residual_histogram_path

    # ------------------------------------------------------------------
    # 2. Horizon metrics figure
    # ------------------------------------------------------------------
    horizon_metrics = _build_horizon_metrics_frame(shared_report=shared_report)
    if not horizon_metrics.empty:
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

        mean_interval_width = horizon_metrics.get("mean_interval_width")
        if mean_interval_width is not None and not bool(mean_interval_width.isna().all()):
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

        horizon_metrics_fig.update_layout(
            title=_plot_title(
                "Forecast Horizon Error Metrics",
                "MAE and RMSE by step ahead, with optional interval width when available.",
            ),
            xaxis_title="Horizon Index",
            yaxis_title="Error Metric",
        )
        horizon_metrics_path = report_dir / "horizon_metrics.html"
        horizon_metrics_fig.write_html(str(horizon_metrics_path))
        report_paths["horizon_metrics"] = horizon_metrics_path

        # Bias and pinball loss are kept in a separate artifact so the main
        # horizon page stays readable while still surfacing directional and
        # probabilistic quality behavior from the same canonical grouped table.
        if "bias" in horizon_metrics.columns or "overall_pinball_loss" in horizon_metrics.columns:
            horizon_bias_fig = go.Figure()

            if "bias" in horizon_metrics.columns:
                horizon_bias_fig.add_trace(
                    go.Scatter(
                        x=horizon_metrics["horizon_index"],
                        y=horizon_metrics["bias"],
                        mode="lines+markers",
                        name="Bias",
                    )
                )

            pinball_series = horizon_metrics.get("overall_pinball_loss")
            if pinball_series is not None and not bool(pinball_series.isna().all()):
                horizon_bias_fig.add_trace(
                    go.Scatter(
                        x=horizon_metrics["horizon_index"],
                        y=horizon_metrics["overall_pinball_loss"],
                        mode="lines+markers",
                        name="Overall Pinball Loss",
                        yaxis="y2",
                    )
                )
                horizon_bias_fig.update_layout(
                    yaxis2=dict(
                        title="Pinball Loss",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    )
                )

            horizon_bias_fig.update_layout(
                title=_plot_title(
                    "Forecast Horizon Bias And Pinball Loss",
                    "Directional error and probabilistic loss across the prediction horizon.",
                ),
                xaxis_title="Horizon Index",
                yaxis_title="Bias",
            )
            horizon_bias_path = report_dir / "horizon_bias.html"
            horizon_bias_fig.write_html(str(horizon_bias_path))
            report_paths["horizon_bias"] = horizon_bias_path

        # Coverage gets its own page because it often benefits from being read
        # as a calibration-focused artifact rather than mixed into error plots.
        coverage_series = horizon_metrics.get("empirical_interval_coverage")
        if coverage_series is not None and not bool(coverage_series.isna().all()):
            horizon_coverage_fig = go.Figure()
            horizon_coverage_fig.add_trace(
                go.Scatter(
                    x=horizon_metrics["horizon_index"],
                    y=horizon_metrics["empirical_interval_coverage"],
                    mode="lines+markers",
                    name="Empirical Coverage",
                )
            )
            horizon_coverage_fig.update_layout(
                title=_plot_title(
                    "Forecast Horizon Coverage",
                    "Empirical interval coverage across the prediction horizon.",
                ),
                xaxis_title="Horizon Index",
                yaxis_title="Coverage",
            )
            horizon_coverage_path = report_dir / "horizon_coverage.html"
            horizon_coverage_fig.write_html(str(horizon_coverage_path))
            report_paths["horizon_coverage"] = horizon_coverage_path

    # ------------------------------------------------------------------
    # 3. Subject-level grouped metrics
    # ------------------------------------------------------------------
    by_subject = _build_grouped_metrics_frame(
        shared_report=shared_report,
        table_name="by_subject",
        required_columns=("group_value", "mae"),
    )
    if not by_subject.empty:
        subject_fig = go.Figure()
        subject_fig.add_trace(
            go.Bar(
                x=by_subject["group_value"].astype(str),
                y=by_subject["mae"],
                name="MAE",
            )
        )
        if "bias" in by_subject.columns and not bool(by_subject["bias"].isna().all()):
            subject_fig.add_trace(
                go.Scatter(
                    x=by_subject["group_value"].astype(str),
                    y=by_subject["bias"],
                    mode="lines+markers",
                    name="Bias",
                    yaxis="y2",
                )
            )
            subject_fig.update_layout(
                yaxis2=dict(
                    title="Bias",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                )
            )
        subject_fig.update_layout(
            title=_plot_title(
                "Subject-Level Metrics",
                "Average error by subject, with directional bias when available.",
            ),
            xaxis_title="Subject",
            yaxis_title="MAE",
        )
        subject_path = report_dir / "subject_metrics.html"
        subject_fig.write_html(str(subject_path))
        report_paths["subject_metrics"] = subject_path

    # ------------------------------------------------------------------
    # 4. Glucose-range grouped metrics
    # ------------------------------------------------------------------
    by_glucose_range = _build_grouped_metrics_frame(
        shared_report=shared_report,
        table_name="by_glucose_range",
        required_columns=("group_value", "mae"),
    )
    if not by_glucose_range.empty:
        glucose_range_fig = go.Figure()
        glucose_range_fig.add_trace(
            go.Bar(
                x=by_glucose_range["group_value"].astype(str),
                y=by_glucose_range["mae"],
                name="MAE",
            )
        )
        coverage_series = by_glucose_range.get("empirical_interval_coverage")
        if coverage_series is not None and not bool(coverage_series.isna().all()):
            glucose_range_fig.add_trace(
                go.Scatter(
                    x=by_glucose_range["group_value"].astype(str),
                    y=by_glucose_range["empirical_interval_coverage"],
                    mode="lines+markers",
                    name="Empirical Coverage",
                    yaxis="y2",
                )
            )
            glucose_range_fig.update_layout(
                yaxis2=dict(
                    title="Coverage",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                )
            )
        glucose_range_fig.update_layout(
            title=_plot_title(
                "Glucose-Range Metrics",
                "Average error by glycemic regime, with coverage when available.",
            ),
            xaxis_title="Glucose Range",
            yaxis_title="MAE",
        )
        glucose_range_path = report_dir / "glucose_range_metrics.html"
        glucose_range_fig.write_html(str(glucose_range_path))
        report_paths["glucose_range_metrics"] = glucose_range_path

    # ------------------------------------------------------------------
    # 5. Forecast overview figure
    # ------------------------------------------------------------------
    overview_fig = go.Figure()
    subject_ids = list(dict.fromkeys(frame["subject_id"].tolist()))[:max_subjects]
    filtered = frame[frame["subject_id"].isin(subject_ids)].copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
    filtered = filtered.sort_values(
        by=["subject_id", "timestamp"],
        ascending=True,
    )

    quantile_columns = sorted(
        column for column in frame.columns if str(column).startswith("pred_q")
    )
    for subject_id in subject_ids:
        subject_frame = filtered.loc[filtered["subject_id"] == subject_id, :]
        if len(subject_frame) == 0:
            continue

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

        if len(quantile_columns) >= 2:
            lower = quantile_columns[0]
            upper = quantile_columns[-1]
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

    overview_fig.update_layout(
        title=_plot_title(
            "Forecast Overview",
            "Target trajectory, median prediction, and prediction interval for a sample of subjects.",
        ),
        xaxis_title="Timestamp",
        yaxis_title="Value",
    )
    overview_path = report_dir / "forecast_overview.html"
    overview_fig.write_html(str(overview_path))
    report_paths["forecast_overview"] = overview_path
    return report_paths
