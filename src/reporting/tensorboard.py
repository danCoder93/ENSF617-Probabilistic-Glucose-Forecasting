from __future__ import annotations

# AI-assisted maintenance note:
# This module contains the TensorBoard sink for the repository's post-run
# reporting layer.
#
# Why this file exists:
# - the repository already has strong runtime TensorBoard observability through
#   callbacks, model text/graph logging, parameter telemetry, and prediction
#   figures
# - what was still missing was a post-run TensorBoard consumer for the
#   canonical shared-report surface built after prediction + structured
#   evaluation
# - placing that sink in the `reporting` package keeps the lifecycle boundary
#   explicit:
#     * `observability` handles live runtime visibility during training
#     * `reporting` handles post-run packaging and rendering once predictions
#       already exist
#
# Responsibility boundary:
# - consume an already-built `SharedReport`
# - normalize logger/trainer inputs down to TensorBoard-compatible experiments
# - log post-run scalars, text blocks, compact table previews, and a few
#   lightweight matplotlib figures
#
# What does *not* live here:
# - canonical metric computation
# - shared-report row construction
# - CSV export
# - Plotly HTML generation
# - live callback logic
#
# In other words, this file is a post-run sink. It should render the canonical
# shared-report surface into TensorBoard, not quietly become a second
# computation layer.

from io import StringIO
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from reporting.types import SharedReport


# ============================================================================
# TensorBoard Experiment Discovery
# ============================================================================
# The post-run reporting path cannot assume it is always handed a full Lightning
# `Trainer`. In practice, the workflow may provide:
# - the `Trainer` itself
# - one logger object
# - a list/tuple of loggers
# - `trainer_observability.logger` from the runtime observability bundle
#
# This module therefore includes its own small normalization helpers instead of
# forcing the workflow to reshape those inputs before calling into the reporting
# sink.
def _coerce_loggers(logger_or_trainer: Any) -> list[Any]:
    """
    Normalize a trainer/logger input into a simple logger list.

    Context:
    the reporting sink is called after the core workflow has already run, so it
    may not always receive a full Lightning `Trainer`. Keeping this normalization
    local makes the sink easier to reuse from workflows, tests, and notebooks.

    Accepted shapes:
    - a Lightning-like trainer with `.loggers`
    - a Lightning-like trainer with `.logger`
    - one logger instance
    - a list/tuple/set of logger instances
    - `None`

    Design note:
    this helper intentionally does *not* validate that the returned objects are
    TensorBoard loggers. That is handled by `_tensorboard_experiments(...)`
    below so the logger-shape normalization and TensorBoard-capability filtering
    remain separate concerns.
    """
    if logger_or_trainer is None:
        return []

    # Lightning may expose multiple loggers through `.loggers`. This is the
    # richest shape because it preserves the full configured logger set.
    loggers = getattr(logger_or_trainer, "loggers", None)
    if loggers is not None:
        return list(loggers)

    # Single-logger trainers often expose `.logger` instead.
    logger = getattr(logger_or_trainer, "logger", None)
    if logger is not None:
        return [logger]

    # The workflow may also pass one logger directly or a small collection of
    # loggers from the observability artifact bundle.
    if isinstance(logger_or_trainer, (list, tuple, set)):
        return list(logger_or_trainer)

    return [logger_or_trainer]


def _tensorboard_experiments(logger_or_trainer: Any) -> list[Any]:
    """
    Return TensorBoard-compatible experiment backends from a trainer/logger input.

    Context:
    only some Lightning logger backends expose TensorBoard-style experiment
    objects with methods such as:
    - `add_scalar(...)`
    - `add_text(...)`
    - `add_figure(...)`

    Why this helper exists:
    the post-run reporting sink should degrade gracefully when a run used a CSV
    logger or no logger at all. Filtering compatibility here keeps the public
    sink API simple while still remaining safe across logger backends.
    """
    experiments: list[Any] = []
    for logger in _coerce_loggers(logger_or_trainer):
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_scalar"):
            experiments.append(experiment)
    return experiments


# ============================================================================
# Text / Table Formatting Helpers
# ============================================================================
# TensorBoard's text surface is useful for compact narrative summaries and
# markdown-style table previews. The helpers below keep that formatting logic
# centralized so the public sink stays focused on the reporting lifecycle.
def _metadata_text(metadata: Mapping[str, Any]) -> str:
    """
    Convert shared-report metadata into a compact deterministic text block.

    Context:
    metadata is useful for provenance and later interpretation, but it is not
    always convenient to inspect as raw Python objects. This helper renders the
    small metadata surface into a stable markdown-like text payload for
    TensorBoard.
    """
    if not metadata:
        return "Shared-report metadata is empty."

    lines = ["Shared-report metadata:"]
    for key in sorted(metadata):
        value = metadata[key]
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _frame_preview_text(
    frame: pd.DataFrame,
    *,
    name: str,
    max_rows: int,
) -> str:
    """
    Render one DataFrame as a compact markdown/code-style preview block.

    Context:
    full tables are often too large for TensorBoard text panes. A stable,
    capped preview is usually more useful than dumping every row.

    Design choice:
    this helper prefers a plain text-table preview rather than HTML so the
    payload remains easy to read in TensorBoard's text tab and simple to reason
    about during maintenance.
    """
    if frame.empty:
        return f"{name}: empty table."

    preview = frame.head(max_rows)
    buffer = StringIO()
    preview.to_string(buf=buffer, index=False)
    row_suffix = (
        f"showing first {len(preview)} of {len(frame)} row(s)"
        if len(frame) > len(preview)
        else f"showing all {len(frame)} row(s)"
    )
    return f"{name} ({row_suffix})\n\n```\n{buffer.getvalue()}\n```"


# ============================================================================
# Scalar / Text / Table Logging
# ============================================================================
# These helpers each log one aspect of the shared report. Keeping them separate
# makes the public sink easier to read and keeps each responsibility explicit.
def _log_shared_report_scalars(
    *,
    experiments: Sequence[Any],
    shared_report: SharedReport,
    global_step: int,
    namespace: str,
) -> None:
    """
    Log scalar entries from `SharedReport.scalars` into TensorBoard.

    Context:
    these are the most dashboard-friendly report outputs. They are already
    packaged by the reporting builders, so this sink should only mirror them
    into TensorBoard rather than recomputing them.
    """
    if not experiments or not shared_report.scalars:
        return

    for key, value in shared_report.scalars.items():
        # Skip missing scalar values rather than forcing a placeholder into
        # TensorBoard. This keeps the scalar surface focused on real numeric
        # values only.
        if value is None:
            continue

        tag = f"{namespace}/scalars/{key}"
        for experiment in experiments:
            add_scalar = getattr(experiment, "add_scalar", None)
            if callable(add_scalar):
                add_scalar(tag, value, global_step=global_step)


def _log_shared_report_text(
    *,
    experiments: Sequence[Any],
    shared_report: SharedReport,
    global_step: int,
    namespace: str,
) -> None:
    """
    Log narrative text blocks and metadata summaries into TensorBoard.

    Context:
    the reporting package intentionally builds small factual text summaries in
    `SharedReport.text`. TensorBoard text panes are a natural home for those
    summaries because they make the report more interpretable without requiring
    a separate document or notebook.
    """
    if not experiments:
        return

    for key, value in shared_report.text.items():
        tag = f"{namespace}/text/{key}"
        for experiment in experiments:
            add_text = getattr(experiment, "add_text", None)
            if callable(add_text):
                add_text(tag, value, global_step=global_step)

    metadata_text = _metadata_text(shared_report.metadata)
    for experiment in experiments:
        add_text = getattr(experiment, "add_text", None)
        if callable(add_text):
            add_text(f"{namespace}/text/metadata", metadata_text, global_step=global_step)


def _log_shared_report_tables(
    *,
    experiments: Sequence[Any],
    shared_report: SharedReport,
    global_step: int,
    namespace: str,
    max_rows: int,
) -> None:
    """
    Log compact preview text for each shared-report table.

    Context:
    TensorBoard does not provide a full spreadsheet-style table viewer. The
    pragmatic compromise is to log small previews as text blocks so users can
    quickly inspect the schema and the first few rows of each table.
    """
    if not experiments or not shared_report.tables:
        return

    for table_name, frame in shared_report.tables.items():
        preview_text = _frame_preview_text(
            frame,
            name=table_name,
            max_rows=max_rows,
        )
        tag = f"{namespace}/tables/{table_name}"
        for experiment in experiments:
            add_text = getattr(experiment, "add_text", None)
            if callable(add_text):
                add_text(tag, preview_text, global_step=global_step)


# ============================================================================
# Figure Builders
# ============================================================================
# The figure builders below intentionally consume the already-packaged shared
# report tables. They do *not* revisit raw model outputs or test batches.
# That preserves the architecture boundary:
# - build once in `reporting.builders`
# - render many ways in sink modules such as this one
def _build_residual_histogram_figure(shared_report: SharedReport) -> Any | None:
    """
    Build a matplotlib residual histogram from the shared prediction table.

    Context:
    the residual histogram is a fast first-pass diagnostic for:
    - broad bias direction
    - outlier spread
    - whether error appears tightly concentrated or widely dispersed
    """
    prediction_table = shared_report.tables.get("prediction_table", pd.DataFrame())
    if prediction_table.empty or "residual" not in prediction_table.columns:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    figure = plt.figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.hist(prediction_table["residual"].dropna())
    axes.set_title("Residual Distribution")
    axes.set_xlabel("Residual")
    axes.set_ylabel("Count")
    figure.tight_layout()
    return figure


def _normalized_grouped_frame(
    shared_report: SharedReport,
    *,
    table_name: str,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    """Return one canonical grouped report table after minimal sink normalization.

    Context:
    the shared report stores grouped tables in a repository-stable schema. Sink
    code still benefits from one small helper that validates presence, sorts the
    canonical group axis, and returns an isolated copy for plotting.
    """
    frame = shared_report.tables.get(table_name, pd.DataFrame()).copy()
    if frame.empty:
        return pd.DataFrame()

    if not set(required_columns).issubset(set(frame.columns)):
        return pd.DataFrame()

    # Sorting on the canonical grouped axis keeps line plots monotonic for
    # horizon views and deterministic for subgroup bar charts.
    frame.sort_values(by=["group_value"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _build_horizon_metrics_figure(shared_report: SharedReport) -> Any | None:
    """
    Build a matplotlib horizon-metrics figure from shared-report tables.

    Context:
    horizon-wise error behavior is one of the most informative post-run views
    for sequence forecasting because it reveals whether performance degrades
    sharply or gradually across the prediction horizon.
    """
    by_horizon = _normalized_grouped_frame(
        shared_report,
        table_name="by_horizon",
        required_columns=("group_value", "mae", "rmse"),
    )
    if by_horizon.empty:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    figure = plt.figure()
    axes = figure.add_subplot(1, 1, 1)

    # `group_value` is the canonical grouped-axis field produced by the shared
    # reporting builder for horizon-grouped tables.
    horizon_index = by_horizon["group_value"]
    axes.plot(horizon_index, by_horizon["mae"], marker="o", label="MAE")
    axes.plot(horizon_index, by_horizon["rmse"], marker="o", label="RMSE")
    axes.set_title("Error Metrics By Forecast Horizon")
    axes.set_xlabel("Horizon Index")
    axes.set_ylabel("Metric Value")
    axes.legend()
    figure.tight_layout()
    return figure


def _build_horizon_uncertainty_figure(shared_report: SharedReport) -> Any | None:
    """Build a horizon-wise uncertainty and calibration figure.

    Context:
    the canonical `by_horizon` table already carries uncertainty-oriented fields
    such as interval width and empirical coverage. Surfacing them here turns
    TensorBoard into a stronger probabilistic forecasting surface without
    changing evaluation semantics.
    """
    by_horizon = _normalized_grouped_frame(
        shared_report,
        table_name="by_horizon",
        required_columns=("group_value",),
    )
    if by_horizon.empty:
        return None

    has_width = "mean_interval_width" in by_horizon.columns and not bool(
        by_horizon["mean_interval_width"].isna().all()
    )
    has_coverage = "empirical_interval_coverage" in by_horizon.columns and not bool(
        by_horizon["empirical_interval_coverage"].isna().all()
    )

    if not has_width and not has_coverage:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    figure = plt.figure()
    axes = figure.add_subplot(1, 1, 1)
    horizon_index = by_horizon["group_value"]

    # Width and coverage can have unlike scales, so a secondary axis keeps both
    # readable without rescaling or silently discarding one of them.
    secondary_axes = None
    if has_width:
        axes.plot(
            horizon_index,
            by_horizon["mean_interval_width"],
            marker="o",
            label="Mean Interval Width",
        )

    if has_coverage:
        secondary_axes = axes.twinx()
        secondary_axes.plot(
            horizon_index,
            by_horizon["empirical_interval_coverage"],
            marker="o",
            label="Empirical Coverage",
        )
        secondary_axes.set_ylabel("Coverage")

    axes.set_title("Uncertainty By Forecast Horizon")
    axes.set_xlabel("Horizon Index")
    axes.set_ylabel("Interval Width")

    # Matplotlib keeps legends per-axis, so merge handles explicitly when a
    # secondary axis exists.
    handles, labels = axes.get_legend_handles_labels()
    if secondary_axes is not None:
        secondary_handles, secondary_labels = secondary_axes.get_legend_handles_labels()
        handles += secondary_handles
        labels += secondary_labels
    if handles:
        axes.legend(handles, labels)

    figure.tight_layout()
    return figure


def _build_horizon_bias_figure(shared_report: SharedReport) -> Any | None:
    """Build a horizon-wise bias and pinball-loss figure.

    Context:
    the grouped horizon table already carries signed bias and pinball-loss
    summaries. Logging them separately keeps the main horizon error plot clean
    while still surfacing important probabilistic and directional behavior.
    """
    by_horizon = _normalized_grouped_frame(
        shared_report,
        table_name="by_horizon",
        required_columns=("group_value",),
    )
    if by_horizon.empty:
        return None

    has_bias = "bias" in by_horizon.columns and not bool(by_horizon["bias"].isna().all())
    has_pinball = "overall_pinball_loss" in by_horizon.columns and not bool(
        by_horizon["overall_pinball_loss"].isna().all()
    )

    if not has_bias and not has_pinball:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    figure = plt.figure()
    axes = figure.add_subplot(1, 1, 1)
    horizon_index = by_horizon["group_value"]

    if has_bias:
        axes.plot(horizon_index, by_horizon["bias"], marker="o", label="Bias")

    secondary_axes = None
    if has_pinball:
        secondary_axes = axes.twinx()
        secondary_axes.plot(
            horizon_index,
            by_horizon["overall_pinball_loss"],
            marker="o",
            label="Overall Pinball Loss",
        )
        secondary_axes.set_ylabel("Pinball Loss")

    axes.set_title("Bias And Pinball Loss By Forecast Horizon")
    axes.set_xlabel("Horizon Index")
    axes.set_ylabel("Bias")

    handles, labels = axes.get_legend_handles_labels()
    if secondary_axes is not None:
        secondary_handles, secondary_labels = secondary_axes.get_legend_handles_labels()
        handles += secondary_handles
        labels += secondary_labels
    if handles:
        axes.legend(handles, labels)

    figure.tight_layout()
    return figure


def _build_grouped_bar_figure(
    shared_report: SharedReport,
    *,
    table_name: str,
    metric_name: str,
    title: str,
    ylabel: str,
) -> Any | None:
    """Build a compact grouped bar chart for one canonical grouped table.

    Context:
    subject-level and glucose-range tables are already packaged canonically. A
    small generic helper keeps the sink consistent when turning those grouped
    tables into simple bar charts for TensorBoard.
    """
    frame = _normalized_grouped_frame(
        shared_report,
        table_name=table_name,
        required_columns=("group_value", metric_name),
    )
    if frame.empty:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    figure = plt.figure()
    axes = figure.add_subplot(1, 1, 1)

    # Converting the canonical group values to strings keeps category labels
    # stable regardless of whether the grouping key is numeric, textual, or a
    # pandas interval-like object.
    group_labels = frame["group_value"].astype(str)
    axes.bar(group_labels, frame[metric_name])
    axes.set_title(title)
    axes.set_xlabel("Group")
    axes.set_ylabel(ylabel)

    # Larger grouped surfaces can have longer labels, so a light rotation keeps
    # them readable without changing the underlying grouped contract.
    for label in axes.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    figure.tight_layout()
    return figure


def _build_forecast_overview_figure(
    shared_report: SharedReport,
    *,
    max_subjects: int,
) -> Any | None:
    """
    Build a compact forecast-overview figure from the shared prediction table.

    Context:
    this is the most qualitative TensorBoard figure in the post-run sink. It is
    not meant to be an exhaustive all-subject dashboard; instead it provides a
    small visual sample of:
    - true target trajectory
    - median prediction
    - simple prediction interval band when available

    Design note:
    subject count is capped so the figure remains readable and does not explode
    into a dense unreadable plot on larger runs.
    """
    prediction_table = shared_report.tables.get("prediction_table", pd.DataFrame())
    if prediction_table.empty:
        return None

    required_columns = {"subject_id", "timestamp", "target", "median_prediction"}
    if not required_columns.issubset(set(prediction_table.columns)):
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    frame = prediction_table.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame.sort_values(by=["subject_id", "timestamp"], inplace=True)

    subject_ids = list(dict.fromkeys(frame["subject_id"].tolist()))[:max_subjects]
    if not subject_ids:
        return None

    figure = plt.figure()
    axes = figure.add_subplot(1, 1, 1)

    quantile_columns = sorted(
        column for column in frame.columns if str(column).startswith("pred_q")
    )

    for subject_id in subject_ids:
        subject_frame = frame.loc[frame["subject_id"] == subject_id, :]
        if subject_frame.empty:
            continue

        # Plot the true trajectory first so the forecast traces are interpreted
        # relative to the observed target.
        axes.plot(
            subject_frame["timestamp"],
            subject_frame["target"],
            label=f"{subject_id} target",
        )
        axes.plot(
            subject_frame["timestamp"],
            subject_frame["median_prediction"],
            label=f"{subject_id} median",
        )

        # When interval columns exist, show the widest simple band between the
        # first and last quantile columns. This mirrors the lightweight interval
        # interpretation already used in the flat prediction-table schema.
        if len(quantile_columns) >= 2:
            lower = subject_frame[quantile_columns[0]]
            upper = subject_frame[quantile_columns[-1]]
            axes.fill_between(
                subject_frame["timestamp"],
                lower,
                upper,
                alpha=0.2,
            )

    axes.set_title("Forecast Overview")
    axes.set_xlabel("Timestamp")
    axes.set_ylabel("Value")
    axes.legend()
    figure.tight_layout()
    return figure


def _iter_report_figures(
    shared_report: SharedReport,
    *,
    max_subjects: int,
) -> Iterable[tuple[str, Any]]:
    """
    Yield the small built-in matplotlib figures supported by this sink.

    Context:
    keeping figure enumeration in one helper makes the public sink simpler and
    keeps figure selection policy explicit in one place.
    """
    residual_histogram = _build_residual_histogram_figure(shared_report)
    if residual_histogram is not None:
        yield "residual_histogram", residual_histogram

    horizon_metrics = _build_horizon_metrics_figure(shared_report)
    if horizon_metrics is not None:
        yield "horizon_metrics", horizon_metrics

    horizon_uncertainty = _build_horizon_uncertainty_figure(shared_report)
    if horizon_uncertainty is not None:
        yield "horizon_uncertainty", horizon_uncertainty

    horizon_bias = _build_horizon_bias_figure(shared_report)
    if horizon_bias is not None:
        yield "horizon_bias", horizon_bias

    subject_mae = _build_grouped_bar_figure(
        shared_report,
        table_name="by_subject",
        metric_name="mae",
        title="MAE By Subject",
        ylabel="MAE",
    )
    if subject_mae is not None:
        yield "subject_mae", subject_mae

    glucose_range_mae = _build_grouped_bar_figure(
        shared_report,
        table_name="by_glucose_range",
        metric_name="mae",
        title="MAE By Glucose Range",
        ylabel="MAE",
    )
    if glucose_range_mae is not None:
        yield "glucose_range_mae", glucose_range_mae

    glucose_range_coverage = _build_grouped_bar_figure(
        shared_report,
        table_name="by_glucose_range",
        metric_name="empirical_interval_coverage",
        title="Empirical Coverage By Glucose Range",
        ylabel="Coverage",
    )
    if glucose_range_coverage is not None:
        yield "glucose_range_coverage", glucose_range_coverage

    forecast_overview = _build_forecast_overview_figure(
        shared_report,
        max_subjects=max_subjects,
    )
    if forecast_overview is not None:
        yield "forecast_overview", forecast_overview


def _log_shared_report_figures(
    *,
    experiments: Sequence[Any],
    shared_report: SharedReport,
    global_step: int,
    namespace: str,
    max_subjects: int,
) -> None:
    """
    Log a small set of matplotlib figures derived from the shared report.

    Context:
    figure generation is intentionally conservative here:
    - it uses already-packaged report tables
    - it generates only a few stable high-value figures
    - it degrades gracefully if matplotlib is unavailable
    """
    if not experiments:
        return

    for figure_name, figure in _iter_report_figures(
        shared_report,
        max_subjects=max_subjects,
    ):
        tag = f"{namespace}/figures/{figure_name}"
        for experiment in experiments:
            add_figure = getattr(experiment, "add_figure", None)
            if callable(add_figure):
                add_figure(tag, figure, global_step=global_step)

        # Close figures after logging so the post-run sink does not accumulate
        # matplotlib state during longer workflows or repeated notebook use.
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception:
            pass


# ============================================================================
# Public TensorBoard Sink
# ============================================================================
def log_shared_report_to_tensorboard(
    *,
    shared_report: SharedReport,
    logger_or_trainer: Any,
    global_step: int = 0,
    namespace: str = "report",
    max_table_rows: int = 20,
    max_forecast_subjects: int = 5,
) -> bool:
    """
    Log a canonical shared report into TensorBoard-compatible logger backends.

    Purpose:
    make TensorBoard a first-class post-run reporting consumer without changing
    training logic, evaluation truth, or the shared-report construction path.

    Context:
    this function is intended to be called by the workflow layer after:
    - prediction batches exist
    - structured evaluation has been computed
    - `build_shared_report(...)` has already packaged the post-run surfaces

    Parameters:
        shared_report:
            The canonical in-memory report bundle to log.
        logger_or_trainer:
            A Lightning trainer, one logger, or a logger collection. The sink
            will normalize this input down to TensorBoard-compatible experiments.
        global_step:
            TensorBoard step used for all logged post-run report artifacts.
            The default of `0` is appropriate for end-of-run reporting because
            these artifacts are not emitted as a time series during training.
        namespace:
            Top-level TensorBoard namespace under which report artifacts are
            grouped. Keeping the namespace configurable makes it easier to add
            multiple report families later without changing the sink internals.
        max_table_rows:
            Maximum number of rows included in each table preview text block.
        max_forecast_subjects:
            Maximum number of subjects surfaced in the forecast overview figure.

    Returns:
        `True` when at least one TensorBoard-compatible experiment was found and
        the sink attempted logging; otherwise `False`.

    Failure behavior:
    this sink is intentionally best-effort:
    - if no TensorBoard logger is active, it returns `False`
    - if a specific artifact type cannot be rendered, the rest of the sink
      continues
    - it should never become a reason for the training workflow to fail
    """
    experiments = _tensorboard_experiments(logger_or_trainer)
    if not experiments:
        return False

    # Scalars, text, tables, and figures are logged in separate passes so each
    # artifact family remains conceptually independent and easier to maintain.
    _log_shared_report_scalars(
        experiments=experiments,
        shared_report=shared_report,
        global_step=global_step,
        namespace=namespace,
    )
    _log_shared_report_text(
        experiments=experiments,
        shared_report=shared_report,
        global_step=global_step,
        namespace=namespace,
    )
    _log_shared_report_tables(
        experiments=experiments,
        shared_report=shared_report,
        global_step=global_step,
        namespace=namespace,
        max_rows=max_table_rows,
    )
    _log_shared_report_figures(
        experiments=experiments,
        shared_report=shared_report,
        global_step=global_step,
        namespace=namespace,
        max_subjects=max_forecast_subjects,
    )
    return True
