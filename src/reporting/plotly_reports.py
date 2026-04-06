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

    # The canonical grouped horizon table is packaged by `build_shared_report`
    # from the structured evaluation result. If it is absent here, the sink
    # intentionally refuses to act as a backup metrics engine.
    by_horizon = shared_report.tables.get("by_horizon", pd.DataFrame()).copy()
    if by_horizon.empty:
        return pd.DataFrame()

    # The shared-report grouped-table contract uses `group_value` as the stable
    # axis field across grouped surfaces. For the horizon figure we rename that
    # field into a more explicit plotting-oriented column while preserving the
    # original metric columns.
    required_columns = {"group_value", "mae", "rmse"}
    if not required_columns.issubset(set(by_horizon.columns)):
        return pd.DataFrame()

    horizon_metrics = by_horizon.rename(
        columns={"group_value": "horizon_index"}
    ).copy()

    # Keep only the columns the figure actually understands so the sink remains
    # resilient to future expansion of the grouped report schema.
    desired_columns = [
        column
        for column in (
            "horizon_index",
            "mae",
            "rmse",
            "mean_interval_width",
        )
        if column in horizon_metrics.columns
    ]
    return horizon_metrics.loc[:, desired_columns]


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
    - the horizon-metrics figure must render only from the canonical grouped
      `by_horizon` table packaged inside `SharedReport`
    - when that grouped table is unavailable, the sink omits the
      `horizon_metrics` artifact rather than recomputing grouped metrics

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
    # Under the stricter canonical reporting rule, grouped horizon metrics must
    # come from the shared report's `by_horizon` table. The sink no longer
    # derives grouped metrics from flat prediction rows.
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

        # When interval-width information exists, surface it on a secondary axis
        # so uncertainty-width growth can be visually compared with point-error
        # growth without collapsing unlike scales into one axis.
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
    # Use keyword argument for static type-checker friendliness
    filtered.sort_values(by=["subject_id", "timestamp"], inplace=True)

    # Quantile columns are global schema-level columns of the prediction
        # table, so derive them from the full frame once per loop iteration
        # rather than relying on the narrowed subject slice's inferred type.
    quantile_columns = sorted(
        column for column in frame.columns if str(column).startswith("pred_q")
    )   
    
    for subject_id in subject_ids:
        # `loc[...]` keeps the intent explicit for both readers and static type
        # checkers: we are selecting rows from the canonical prediction-table
        # DataFrame for one subject.
        subject_frame = filtered.loc[filtered["subject_id"] == subject_id, :]
        if len(subject_frame) == 0:
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
