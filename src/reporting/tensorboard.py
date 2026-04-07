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
# - impose a dashboard-first TensorBoard information architecture without
#   changing the underlying metric truth already computed elsewhere
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
#
# Dashboard-first enhancement note:
# Earlier versions of this module logged report outputs under a single broad
# namespace. That preserved information, but it made TensorBoard feel like a raw
# artifact dump rather than a readable dashboard.
#
# The current version keeps the same reporting inputs and figure/table support,
# but reorganizes them into clearer presentation layers:
# - `dashboard/*` for the front-door views a human should inspect first
# - `text/*` for orientation, interpretation, and compact narrative summaries
# - `report/*` for broader report-style previews that remain useful but should
#   not dominate the first TensorBoard screen
#
# Important compatibility rule:
# This module still consumes the same canonical `SharedReport`. The enhancement
# is about curation, naming, and presentation—not about changing the truth of
# the run.

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
# Namespace / Tag Policy Helpers
# ============================================================================
# The helpers below encode the dashboard-first TensorBoard taxonomy used by this
# sink.
#
# Why make the taxonomy explicit here:
# - it keeps naming policy centralized instead of scattering string literals
#   across scalar/text/figure logging paths
# - it preserves backward reasoning about what should appear in the front door
#   versus what should remain a broader report preview
# - it makes future refinements easier because the presentation contract is easy
#   to inspect in one place
#
# Important boundary:
# These helpers reorganize already-computed report outputs. They do not change
# metric truth or recompute evaluation.

def _dashboard_base(namespace: str) -> str:
    """Return the dashboard-first base namespace for curated front-door views.

    Context:
    callers may still choose a broader top-level namespace such as `report` for
    compatibility. The dashboard surface should remain clearly separated from
    that broader namespace so the TensorBoard front door feels intentional
    rather than like a raw dump of every artifact type.
    """
    return f"{namespace}/dashboard"



def _text_base(namespace: str) -> str:
    """Return the base namespace used for narrative orientation panels.

    Context:
    text blocks are first-class parts of the dashboard design because they make
    the run easier to interpret. They still deserve a dedicated text namespace
    so they remain easy to browse in TensorBoard's text tab.
    """
    return f"{namespace}/text"



def _report_base(namespace: str) -> str:
    """Return the broader report namespace for previews and secondary surfaces.

    Context:
    not every report artifact belongs in the first dashboard impression.
    Compact table previews and additional report text are still valuable, but
    they are better treated as a secondary report surface than as primary
    dashboard cards.
    """
    return f"{namespace}/report"



def _scalar_dashboard_tag(key: str, *, namespace: str) -> str:
    """Map one canonical scalar key onto a curated dashboard TensorBoard tag.

    Context:
    `SharedReport.scalars` already stores canonical scalar truth. The dashboard
    layer's job is to present that truth through human-readable groups that make
    TensorBoard easier to scan.

    Mapping policy:
    - known training/forecast/uncertainty/calibration/health metrics are placed
      into explicit dashboard sections
    - unknown scalar keys fall back to a general dashboard/report bucket so the
      sink remains forward-compatible with new scalar fields
    """
    dashboard = _dashboard_base(namespace)

    # Training-quality and forecast-error summaries should be the front door.
    if key in {"mae", "rmse", "mape", "median_ae", "bias", "overall_pinball_loss"}:
        return f"{dashboard}/forecast/{key}"

    # Uncertainty-width summaries deserve their own dashboard family because the
    # repo is probabilistic and interval behavior is a first-class concern.
    if key in {"mean_interval_width", "median_interval_width"}:
        return f"{dashboard}/uncertainty/{key}"

    # Coverage-style metrics are calibration signals and should be grouped
    # separately from generic uncertainty width.
    if "coverage" in key:
        return f"{dashboard}/calibration/{key}"

    # Count-like or diagnostic summary scalars still matter, but they are better
    # framed as health indicators than as forecast-quality metrics.
    if any(token in key for token in ("count", "warning", "nonfinite", "nan", "dead", "dominance")):
        return f"{dashboard}/health/{key}"

    # Prediction-table-derived residual aggregates are useful front-door health
    # signals even when they do not fit the stricter metric families above.
    if any(token in key for token in ("residual", "error", "abs_")):
        return f"{dashboard}/health/{key}"

    # Unknown-but-valid scalar additions should still surface somewhere curated
    # instead of silently disappearing. A general bucket keeps the sink forward-
    # compatible while remaining more organized than one flat scalar namespace.
    return f"{dashboard}/report/{key}"



def _table_preview_tag(table_name: str, *, namespace: str) -> str:
    """Return the text tag used for one compact table preview.

    Context:
    table previews are useful for schema discovery and quick inspection, but
    they are not the first thing a human should see on the TensorBoard front
    page. We therefore keep them under the broader report namespace.
    """
    return f"{_report_base(namespace)}/tables/{_table_display_title(table_name)}"



def _report_text_tag(text_key: str, *, namespace: str) -> str:
    """Return the tag used for one broader report text panel.

    Context:
    the canonical text sections remain valuable even after adding a curated
    dashboard-first layer. These report text entries are kept under the report
    namespace so they remain easy to browse without competing directly with the
    smaller orientation panels.
    """
    return f"{_report_base(namespace)}/text/{_text_section_title(text_key)}"



def _dashboard_text_items(shared_report: SharedReport) -> list[tuple[str, str]]:
    """Build the small curated dashboard text surface from the shared report.

    Purpose:
    provide a handful of interpretation-first text panels that make TensorBoard
    easier to read before a user drills into the broader report namespace.

    Why this helper exists:
    the shared report already contains several canonical narrative sections, but
    the dashboard front door benefits from a smaller, more guided subset:
    - an index/orientation panel
    - a compact metadata/provenance panel
    - a small set of the most explanatory report sections when they exist

    Important boundary:
    this helper only reorganizes existing text content and metadata. It does not
    invent new evaluation truth.
    """
    items: list[tuple[str, str]] = []

    # Orientation first: the dashboard should advertise what interpretive text
    # is available before the user drills into the longer sections.
    items.append(("Overview Index", _report_text_index(shared_report)))

    # Metadata is part of the front door because provenance and run context are
    # often the first questions users ask when revisiting an artifact.
    items.append(("Run Metadata", _metadata_text(shared_report.metadata)))

    # Prefer the most explanatory existing canonical panels as dashboard text.
    for key, value in _ordered_report_text_items(shared_report):
        if key in {
            "dataset_overview",
            "metric_overview",
            "data_summary_overview",
            "probabilistic_overview",
            "horizon_overview",
        }:
            items.append((_text_section_title(key), value))

    return items



def _dashboard_figure_tag(figure_name: str, *, namespace: str) -> str:
    """Map one internal figure key onto a dashboard-friendly figure tag.

    Context:
    figures are some of the highest-value front-door artifacts because they can
    summarize multidimensional report behavior more clearly than many separate
    scalar cards.

    Mapping policy:
    - example-style visuals go under `dashboard/examples`
    - horizon/uncertainty/calibration views go under their more specific
      dashboard sections
    - anything else still remains on the dashboard, but in a generic report-like
      bucket so the sink stays forward-compatible
    """
    dashboard = _dashboard_base(namespace)

    if figure_name == "forecast_overview":
        return f"{dashboard}/examples/{_figure_tag_name(figure_name)}"
    if figure_name == "residual_histogram":
        return f"{dashboard}/examples/{_figure_tag_name(figure_name)}"
    if figure_name in {"horizon_metrics", "subject_mae", "subject_rmse", "glucose_range_mae"}:
        return f"{dashboard}/forecast/{_figure_tag_name(figure_name)}"
    if figure_name in {"horizon_uncertainty", "glucose_range_interval_width"}:
        return f"{dashboard}/uncertainty/{_figure_tag_name(figure_name)}"
    if figure_name in {"glucose_range_coverage"}:
        return f"{dashboard}/calibration/{_figure_tag_name(figure_name)}"
    if figure_name in {"horizon_bias", "subject_bias", "glucose_range_bias"}:
        return f"{dashboard}/health/{_figure_tag_name(figure_name)}"
    return f"{dashboard}/report/{_figure_tag_name(figure_name)}"


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



def _table_display_title(table_name: str) -> str:
    """
    Convert an internal shared-report table key into a more readable title.

    Context:
    table keys are intentionally concise and code-friendly. TensorBoard text
    previews are easier to scan when the title shown to readers uses a
    human-readable label that explains what the table represents.
    """
    titles = {
        "prediction_table": "Prediction-Level Analysis Table",
        "by_horizon": "Forecast-Horizon Summary Table",
        "by_subject": "Subject-Level Summary Table",
        "by_glucose_range": "Glucose-Range Summary Table",
    }
    return titles.get(table_name, table_name.replace("_", " ").title())



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
    display_title = _table_display_title(name)
    if frame.empty:
        return f"{display_title}: empty table."

    preview = frame.head(max_rows)
    buffer = StringIO()
    preview.to_string(buf=buffer, index=False)
    row_suffix = (
        f"showing first {len(preview)} of {len(frame)} row(s)"
        if len(frame) > len(preview)
        else f"showing all {len(frame)} row(s)"
    )
    return f"{display_title} ({row_suffix})\n\n```\n{buffer.getvalue()}\n```"



def _ordered_report_text_items(shared_report: SharedReport) -> list[tuple[str, str]]:
    """
    Return report text items in a deterministic interpretation-first order.

    Context:
    `SharedReport.text` is already canonical and should remain the source of
    truth for narrative summaries. The TensorBoard sink still benefits from a
    stable ordering policy so the most interpretive panels surface first in the
    text tab instead of being shown in arbitrary dictionary order.

    Design note:
    this helper does not synthesize new metric content. It only controls the
    presentation order of already-packaged text blocks.
    """
    preferred_order = [
        "dataset_overview",
        "metric_overview",
        "data_summary_overview",
        "quantile_overview",
        "horizon_overview",
        "probabilistic_overview",
        "subject_variability_overview",
        "glucose_range_overview",
    ]
    remaining_keys = sorted(
        key for key in shared_report.text.keys() if key not in set(preferred_order)
    )

    ordered_keys = [key for key in preferred_order if key in shared_report.text]
    ordered_keys.extend(remaining_keys)

    return [(key, shared_report.text[key]) for key in ordered_keys]



def _report_text_index(shared_report: SharedReport) -> str:
    """
    Build a lightweight index describing which canonical text panels are present.

    Context:
    TensorBoard text panes are flatter than a document viewer. A compact index
    at the top makes it easier to understand which interpretation sections exist
    in the current shared report before drilling into each individual panel.

    Design note:
    the index is intentionally descriptive only. It should not become another
    computation surface or duplicate the contents of the individual sections.
    """
    ordered_items = _ordered_report_text_items(shared_report)
    if not ordered_items:
        return "No canonical report text panels are available."

    lines = ["Available report text panels:"]
    for key, _ in ordered_items:
        lines.append(f"- {key}")
    return "\n".join(lines)



def _text_section_title(text_key: str) -> str:
    """
    Convert an internal report-text key into a more readable section title.

    Context:
    The TensorBoard tag surface is easier to scan when text entries use
    interpretation-oriented labels rather than raw internal key names.
    """
    titles = {
        "data_summary_overview": "Data Summary Overview",
        "dataset_overview": "Dataset Overview",
        "metric_overview": "Metric Overview",
        "quantile_overview": "Quantile Overview",
        "horizon_overview": "Forecast Horizon Overview",
        "probabilistic_overview": "Probabilistic Forecast Overview",
        "subject_variability_overview": "Subject Variability Overview",
        "glucose_range_overview": "Glucose-Range Overview",
        "metadata": "Metadata",
        "index": "Index",
    }
    return titles.get(text_key, text_key.replace("_", " ").title())



def _figure_tag_name(figure_name: str) -> str:
    """
    Convert an internal figure key into a more readable TensorBoard tag name.

    Context:
    Internal figure keys are kept concise for maintenance, while TensorBoard
    users benefit from tags that read like dashboard sections rather than code
    identifiers.
    """
    titles = {
        "residual_histogram": "Residual Distribution",
        "horizon_metrics": "Forecast Horizon Error Metrics",
        "horizon_uncertainty": "Forecast Horizon Uncertainty",
        "horizon_bias": "Forecast Horizon Bias And Pinball Loss",
        "subject_mae": "Subject-Level MAE",
        "subject_bias": "Subject-Level Bias",
        "subject_rmse": "Subject-Level RMSE",
        "glucose_range_mae": "Glucose-Range MAE",
        "glucose_range_bias": "Glucose-Range Bias",
        "glucose_range_interval_width": "Glucose-Range Interval Width",
        "glucose_range_coverage": "Glucose-Range Coverage",
        "forecast_overview": "Forecast Overview",
    }
    return titles.get(figure_name, figure_name.replace("_", " ").title())


# ============================================================================
# Scalar / Text / Table Logging
# ============================================================================
# These helpers each log one aspect of the shared report. Keeping them separate
# makes the public sink easier to read and keeps each responsibility explicit.
def _log_dashboard_scalars(
    *,
    experiments: Sequence[Any],
    shared_report: SharedReport,
    global_step: int,
    namespace: str,
) -> None:
    """Log curated shared-report scalars into the dashboard namespace.

    Context:
    these are the most dashboard-friendly report outputs. They are already
    packaged by the reporting builders, so this sink should only mirror them
    into TensorBoard rather than recomputing them.

    Enhancement note:
    this helper replaces the older flat `scalars/*` presentation with a curated
    dashboard hierarchy so the TensorBoard front door better reflects the kinds
    of questions users ask first:
    - how good is the forecast?
    - how wide are the intervals?
    - how well calibrated are they?
    - does the run look healthy?
    """
    if not experiments or not shared_report.scalars:
        return

    for key, value in shared_report.scalars.items():
        # Skip missing scalar values rather than forcing a placeholder into
        # TensorBoard. This keeps the scalar surface focused on real numeric
        # values only.
        if value is None:
            continue

        tag = _scalar_dashboard_tag(key, namespace=namespace)
        for experiment in experiments:
            add_scalar = getattr(experiment, "add_scalar", None)
            if callable(add_scalar):
                add_scalar(tag, value, global_step=global_step)



def _log_dashboard_text(
    *,
    experiments: Sequence[Any],
    shared_report: SharedReport,
    global_step: int,
    namespace: str,
) -> None:
    """Log a small curated front-door text surface for the dashboard.

    Context:
    a dashboard-first TensorBoard layout needs lightweight interpretation and
    provenance text near the front door. The canonical report text remains
    available in the broader report namespace, but this helper selects the most
    explanatory subset for the first-pass view.
    """
    if not experiments:
        return

    for section_title, payload in _dashboard_text_items(shared_report):
        tag = f"{_text_base(namespace)}/overview/{section_title}"
        for experiment in experiments:
            add_text = getattr(experiment, "add_text", None)
            if callable(add_text):
                add_text(tag, payload, global_step=global_step)



def _log_shared_report_text(
    *,
    experiments: Sequence[Any],
    shared_report: SharedReport,
    global_step: int,
    namespace: str,
) -> None:
    """Log the broader canonical report-text surface into TensorBoard.

    Context:
    the dashboard text panels are intentionally small and guided. This helper
    preserves the richer report-style text surface for users who want to inspect
    the full narrative packaging already built by the shared report.
    """
    if not experiments:
        return

    report_text_index = _report_text_index(shared_report)
    for experiment in experiments:
        add_text = getattr(experiment, "add_text", None)
        if callable(add_text):
            add_text(
                _report_text_tag("index", namespace=namespace),
                report_text_index,
                global_step=global_step,
            )

    for key, value in _ordered_report_text_items(shared_report):
        tag = _report_text_tag(key, namespace=namespace)
        for experiment in experiments:
            add_text = getattr(experiment, "add_text", None)
            if callable(add_text):
                add_text(tag, value, global_step=global_step)

    metadata_text = _metadata_text(shared_report.metadata)
    for experiment in experiments:
        add_text = getattr(experiment, "add_text", None)
        if callable(add_text):
            add_text(
                _report_text_tag("metadata", namespace=namespace),
                metadata_text,
                global_step=global_step,
            )



def _log_shared_report_tables(
    *,
    experiments: Sequence[Any],
    shared_report: SharedReport,
    global_step: int,
    namespace: str,
    max_rows: int,
) -> None:
    """Log compact preview text for each shared-report table.

    Context:
    TensorBoard does not provide a full spreadsheet-style table viewer. The
    pragmatic compromise is to log small previews as text blocks so users can
    quickly inspect the schema and the first few rows of each table.

    Dashboard-placement note:
    table previews remain intentionally outside the main dashboard front door.
    They are valuable for inspection, but they are secondary report artifacts,
    not the first thing that should dominate the TensorBoard experience.
    """
    if not experiments or not shared_report.tables:
        return

    for table_name, frame in shared_report.tables.items():
        preview_text = _frame_preview_text(
            frame,
            name=table_name,
            max_rows=max_rows,
        )
        tag = _table_preview_tag(table_name, namespace=namespace)
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
    axes.set_title("Residual Distribution: Error Spread Across Prediction Rows")
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
    frame = frame.sort_values(by=["group_value"]).reset_index(drop=True)
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
    axes.set_title("Forecast Horizon Error Metrics: MAE And RMSE By Step Ahead")
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

    axes.set_title("Forecast Horizon Uncertainty: Interval Width And Coverage")
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

    axes.set_title("Forecast Horizon Bias: Directional Error And Pinball Loss")
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



def _group_axis_label(table_name: str) -> str:
    """
    Return a readable x-axis label for grouped bar figures.

    Context:
    Grouped report tables summarize different entities. Using one generic
    “Group” label works functionally, but a more explicit label makes the
    resulting TensorBoard figures easier to interpret at a glance.
    """
    labels = {
        "by_subject": "Subject",
        "by_glucose_range": "Glucose Range",
    }
    return labels.get(table_name, "Group")



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
    axes.set_xlabel(_group_axis_label(table_name))
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
    frame = frame.sort_values(by=["subject_id", "timestamp"])

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

    axes.set_title("Forecast Overview: Target, Median Prediction, And Interval Band")
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
        title="Subject-Level MAE: Average Error By Subject",
        ylabel="MAE",
    )
    if subject_mae is not None:
        yield "subject_mae", subject_mae

    # Bias is already packaged canonically by evaluation/builders, so the sink
    # only needs to surface it for subject-level directional interpretation.
    subject_bias = _build_grouped_bar_figure(
        shared_report,
        table_name="by_subject",
        metric_name="bias",
        title="Subject-Level Bias: Over- Or Under-Prediction By Subject",
        ylabel="Bias",
    )
    if subject_bias is not None:
        yield "subject_bias", subject_bias

    # RMSE provides a simple spread-oriented companion to MAE for subject-level
    # inspection without introducing any new grouped metric computation here.
    subject_rmse = _build_grouped_bar_figure(
        shared_report,
        table_name="by_subject",
        metric_name="rmse",
        title="Subject-Level RMSE: Error Spread By Subject",
        ylabel="RMSE",
    )
    if subject_rmse is not None:
        yield "subject_rmse", subject_rmse

    glucose_range_mae = _build_grouped_bar_figure(
        shared_report,
        table_name="by_glucose_range",
        metric_name="mae",
        title="Glucose-Range MAE: Average Error By Glycemic Regime",
        ylabel="MAE",
    )
    if glucose_range_mae is not None:
        yield "glucose_range_mae", glucose_range_mae

    # Bias by glucose range helps show whether specific glycemic regimes skew
    # high or low even when top-line MAE stays acceptable.
    glucose_range_bias = _build_grouped_bar_figure(
        shared_report,
        table_name="by_glucose_range",
        metric_name="bias",
        title="Glucose-Range Bias: Directional Error By Glycemic Regime",
        ylabel="Bias",
    )
    if glucose_range_bias is not None:
        yield "glucose_range_bias", glucose_range_bias

    # Interval width by glucose range gives TensorBoard a direct view of how
    # predictive uncertainty changes across clinically different regimes.
    glucose_range_interval_width = _build_grouped_bar_figure(
        shared_report,
        table_name="by_glucose_range",
        metric_name="mean_interval_width",
        title="Glucose-Range Interval Width: Uncertainty By Glycemic Regime",
        ylabel="Interval Width",
    )
    if glucose_range_interval_width is not None:
        yield "glucose_range_interval_width", glucose_range_interval_width

    glucose_range_coverage = _build_grouped_bar_figure(
        shared_report,
        table_name="by_glucose_range",
        metric_name="empirical_interval_coverage",
        title="Glucose-Range Coverage: Calibration By Glycemic Regime",
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



def _log_dashboard_figures(
    *,
    experiments: Sequence[Any],
    shared_report: SharedReport,
    global_step: int,
    namespace: str,
    max_subjects: int,
) -> None:
    """Log curated report figures into dashboard-oriented namespaces.

    Context:
    figures are among the most interpretable post-run surfaces because they can
    communicate multi-dimensional forecast behavior more effectively than a long
    flat list of scalar cards.

    Enhancement note:
    the figure-generation logic itself remains unchanged. The enhancement here is
    the presentation layer: figure tags now reflect whether the figure is about
    examples, forecast quality, uncertainty, calibration, or health.
    """
    if not experiments:
        return

    for figure_name, figure in _iter_report_figures(
        shared_report,
        max_subjects=max_subjects,
    ):
        tag = _dashboard_figure_tag(figure_name, namespace=namespace)
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

    Dashboard-first enhancement summary:
    this function now performs multiple conceptually separate passes so the
    resulting TensorBoard experience is easier to navigate:
    - dashboard scalars for first-pass quantitative inspection
    - dashboard text for orientation and provenance
    - dashboard figures for high-value visual interpretation
    - broader report text and table previews for deeper inspection

    Important compatibility note:
    the function name and call contract stay unchanged so the workflow layer can
    continue using the same integration point while benefiting from the improved
    organization.
    """
    experiments = _tensorboard_experiments(logger_or_trainer)
    if not experiments:
        return False

    # The logging passes are intentionally separated by artifact family so the
    # presentation contract remains easy to reason about and future patches can
    # adjust one surface without disturbing the others.
    _log_dashboard_scalars(
        experiments=experiments,
        shared_report=shared_report,
        global_step=global_step,
        namespace=namespace,
    )
    _log_dashboard_text(
        experiments=experiments,
        shared_report=shared_report,
        global_step=global_step,
        namespace=namespace,
    )
    _log_dashboard_figures(
        experiments=experiments,
        shared_report=shared_report,
        global_step=global_step,
        namespace=namespace,
        max_subjects=max_forecast_subjects,
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
    return True
