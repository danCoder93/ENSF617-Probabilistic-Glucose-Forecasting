from __future__ import annotations

# AI-assisted maintenance note:
# This module owns observability artifacts that are generated *after* model
# predictions already exist.
#
# Responsibility boundary:
# - build one canonical in-memory shared report for post-run predictions
# - export prediction tensors into an analysis-friendly CSV table
# - generate lightweight Plotly HTML reports from that shared report/table
#
# Why keep this apart from callbacks:
# - callbacks run inside the live Trainer loop
# - these helpers operate after the run on persisted prediction outputs
#
# Keeping that lifecycle boundary explicit makes it easier to reason about what
# can fail without affecting training itself and what artifacts are expected to
# exist only after prediction generation completes.
#
# Phase 1 reporting architecture note:
# this file now contains both the canonical post-run report builder and the
# concrete export/report sinks that consume it. The design goal is to improve
# internal structure without forcing a wider package split yet.
#
# In other words:
# - evaluation remains the source of metric truth
# - this module packages that truth into one reusable report surface
# - CSV / HTML outputs are thin consumers of that packaged report
#
# This keeps the current public API stable for the rest of the repository while
# still moving the implementation toward a shared-reporting architecture.
#
# Why the file is intentionally a bit verbose:
# this repository has been moving toward more explicit, tutorial-style comments
# in infrastructure-heavy files. Reporting code is especially easy to misread
# later because it sits between model outputs, evaluation summaries, pandas
# tables, and visualization sinks. The detailed comments below are therefore
# deliberate: they document both *what* happens and *why* the boundaries look
# the way they do.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd
import torch
from torch import Tensor

from config import PathInput
from evaluation import EvaluationResult, GroupedMetricRow, MetricSummary, select_point_prediction
from observability.tensors import _as_metadata_lists
from observability.utils import _has_module


class TestDataloaderProvider(Protocol):
    """
    Minimal contract needed by post-run prediction export/reporting.

    Context:
    the reporting/export path only needs access to the held-out test batches,
    not the full concrete `AZT1DDataModule` API. Keeping the type boundary this
    small makes the helpers easier to test and more accurate about what they
    truly depend on.

    Design intent:
    this protocol deliberately exposes only one method. That keeps the shared
    reporting path honest about its dependency surface and avoids implicitly
    coupling the post-run artifact code to unrelated datamodule features.
    """

    def test_dataloader(self) -> Any:
        """Return the held-out test dataloader or iterable of test batches used for export."""
        ...


# ============================================================================
# Shared Reporting Contracts
# ============================================================================
# These dataclasses define the canonical in-memory reporting surface for the
# post-run observability path. The intent is to compute one stable report once
# and let multiple sinks (CSV, HTML, future TensorBoard/JSON/XML) consume it
# without recomputing tables or summaries.
#
# The architectural distinction is:
# - `EvaluationResult` answers: "what are the canonical evaluation metrics?"
# - `SharedReport` answers: "how should those metrics and predictions be
#   packaged for observability sinks and human inspection?"


@dataclass(frozen=True)
class SharedReport:
    """
    Canonical in-memory report bundle for one post-run prediction analysis.

    Purpose:
    package the most common post-run reporting surfaces into one structured
    object so export and visualization sinks can share the same source of
    truth.

    Context:
    this is intentionally *not* a replacement for `EvaluationResult`.
    `EvaluationResult` remains the canonical detailed metric contract produced
    by the evaluation package. `SharedReport` is the packaging layer that turns
    raw predictions plus evaluation outputs into sink-friendly tables, scalars,
    lightweight narrative text, and figure-ready data.

    Design note:
    keeping these fields as plain dictionaries of familiar Python / pandas
    objects makes the report easy to inspect in notebooks, serialize in simple
    ways, and extend later without forcing a new complex dependency.
    """

    # Scalar summaries that are easy to log, compare, or serialize.
    #
    # Examples:
    # - top-line MAE/RMSE/bias values
    # - counts such as number of rows or number of subjects
    # - interval summary values when quantile forecasts are available
    scalars: dict[str, float | int | None] = field(default_factory=dict)

    # Canonical tabular surfaces for downstream exports and analysis.
    #
    # These tables are where most sinks should start. CSV export writes one of
    # these tables directly; future JSON or dashboard sinks can inspect them
    # without having to reconstruct rows from raw prediction tensors.
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Lightweight narrative summaries for dashboards, logs, or future text
    # sinks.
    #
    # The goal is not to generate a polished report essay here. Instead, this
    # field provides compact factual text that can be surfaced in TensorBoard,
    # summaries, or notebooks without making every caller hand-roll its own
    # textual description of the same artifacts.
    text: dict[str, str] = field(default_factory=dict)

    # Figure-ready placeholders or lightweight plot input structures.
    #
    # Phase 1 intentionally keeps this field loose because Plotly generation
    # still happens in the concrete sink below. The important point is that the
    # file now has an explicit home for future figure-domain data rather than
    # hiding that information implicitly in sink-specific code.
    figures: dict[str, Any] = field(default_factory=dict)

    # Metadata describing how the report was built and what it contains.
    #
    # This field is useful for provenance and later sink logic. It lets callers
    # understand which quantiles were used, how many batches were packaged, and
    # whether the report includes structured evaluation outputs.
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Shared Reporting Builders
# ============================================================================
# The helpers below compute canonical in-memory reporting surfaces once. Export
# and visualization functions later in this file should consume these outputs
# rather than rebuilding the same tables independently.
#
# That is the core Phase 1 enhancement in this file: keep the outward features
# similar, but move the actual table/summary construction into one reusable
# computation path.


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
    export and lightweight Plotly reports. Phase 1 keeps that shape intact, but
    centralizes the row construction so downstream sinks do not each invent
    their own variant of the same logic.

    Important behavior:
    - aligns prediction batches with the original test dataloader batches
    - attaches metadata such as subject ID and decoder timing
    - emits one row per forecast horizon step for easy pandas/Plotly use

    Why this shape is intentionally denormalized:
    a row-per-horizon table is not the most storage-efficient representation,
    but it is the most convenient shape for quick pandas aggregation, grouped
    plotting, CSV export, and manual inspection. That tradeoff is worthwhile in
    the observability layer because clarity and reuse matter more here than raw
    compactness.
    """
    rows: list[dict[str, Any]] = []

    # The shared reporting path reuses the held-out test loader so it can align
    # each prediction batch with the exact source batch metadata that produced
    # it. This is how subject identifiers and decoder window boundaries stay
    # attached to the exported/reporting rows.
    test_loader = datamodule.test_dataloader()

    # Column names are derived once from the quantile configuration so every row
    # uses the same deterministic schema. Keeping this stable matters because
    # later sinks and notebooks often key off these exact column names.
    quantile_columns = [f"pred_q{int(round(q * 100)):02d}" for q in quantiles]

    # The exported/reporting table intentionally lines up predictions with the
    # original test dataloader batches so metadata such as subject ID and
    # decoder start can be attached row by row.
    for batch_index, (prediction_batch, batch) in enumerate(zip(predictions, test_loader)):
        # Post-run reporting works on CPU copies so the downstream pandas/Plotly
        # path is device-agnostic and does not accidentally hold onto GPU/MPS
        # memory longer than necessary.
        prediction_cpu = prediction_batch.detach().cpu()

        # The reporting table keeps one explicit median/point forecast column in
        # addition to the quantile columns. This mirrors the repo's current
        # convention of treating the 0.5 quantile as the default point forecast
        # surface for quick inspection.
        point_prediction_cpu = select_point_prediction(
            prediction_cpu,
            quantiles,
            quantile=0.5,
        )

        target = batch["target"]
        if isinstance(target, Tensor):
            target_cpu = target.detach().cpu()
        else:
            # Some dataset/test paths may expose target-like values in a form
            # that is not already a Tensor. `torch.as_tensor(...)` preserves the
            # current behavior of accepting those paths without forcing every
            # caller to pre-normalize them.
            target_cpu = torch.as_tensor(target)

        # Some code paths preserve a trailing singleton target dimension.
        # The flat reporting table expects `[batch, horizon]`, so we normalize
        # that case here once instead of making every consumer remember it.
        if target_cpu.ndim == 3 and target_cpu.shape[-1] == 1:
            target_cpu = target_cpu.squeeze(-1)

        batch_size = int(prediction_cpu.shape[0])

        # Metadata can arrive in nested or mixed shapes depending on how the
        # upstream batch was assembled. `_as_metadata_lists(...)` is the shared
        # normalization helper used elsewhere in observability to make sample-
        # level indexing predictable.
        metadata = _as_metadata_lists(batch["metadata"], batch_size)

        for sample_index in range(batch_size):
            # Metadata defaults are intentionally conservative. Reporting should
            # remain usable even when a field is missing, and a placeholder is
            # more diagnosable than failing with an obscure key/index error.
            subject_id = str(metadata.get("subject_id", ["unknown"])[sample_index])
            decoder_start = pd.Timestamp(
                str(
                    metadata.get("decoder_start", ["1970-01-01 00:00:00"])[sample_index]
                )
            )

            for horizon_index in range(int(prediction_cpu.shape[1])):
                # The canonical reporting table is intentionally one row per
                # forecast horizon step rather than one row per sample window.
                # That denormalized shape is what makes later grouped metric
                # analysis, CSV export, and Plotly usage straightforward.
                timestamp = decoder_start + pd.Timedelta(
                    minutes=sampling_interval_minutes * horizon_index
                )
                row = {
                    # Batch/sample indices are retained on purpose. They are not
                    # just bookkeeping; they help trace suspicious rows back to
                    # their original prediction batch when debugging.
                    "prediction_batch_index": batch_index,
                    "sample_index_within_batch": sample_index,
                    "subject_id": subject_id,
                    "decoder_start": str(metadata.get("decoder_start", [""])[sample_index]),
                    "decoder_end": str(metadata.get("decoder_end", [""])[sample_index]),
                    "timestamp": timestamp.isoformat(),
                    "horizon_index": horizon_index,
                    "target": float(target_cpu[sample_index, horizon_index].item()),
                }

                # Each quantile is expanded into its own scalar column. This is
                # intentionally redundant relative to the original prediction
                # tensor layout, but it makes CSV/Plotly/pandas consumers much
                # easier to write and inspect.
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

                # Residual follows the repo's existing convention:
                # prediction - target. Keeping the sign stable matters because
                # downstream plots and grouped metrics interpret bias direction
                # using that choice.
                row["residual"] = row["median_prediction"] - row["target"]

                # When at least two quantiles are available, expose the widest
                # simple interval width directly in the table. This keeps common
                # uncertainty diagnostics easy to compute and plot later.
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
    grouped tables are useful for rich analysis, but sinks such as run summaries
    or future dashboard integrations often benefit from one flat scalar map.

    Why keep this helper separate:
    converting the structured evaluation dataclass into a plain scalar map is a
    packaging concern, not an evaluation concern. Keeping that translation here
    avoids leaking sink-specific naming choices back into the evaluation layer.
    """
    if summary is None:
        return {}

    # Top-line scalar names are intentionally short and stable because they are
    # likely to be reused in logs, JSON summaries, or TensorBoard scalar names.
    scalars: dict[str, float | int | None] = {
        "count": summary.count,
        "mae": summary.mae,
        "rmse": summary.rmse,
        "bias": summary.bias,
        "overall_pinball_loss": summary.overall_pinball_loss,
        "mean_interval_width": summary.mean_interval_width,
        "empirical_interval_coverage": summary.empirical_interval_coverage,
    }

    # Quantile-specific losses stay namespaced so they remain easy to inspect in
    # JSON, TensorBoard scalar groups, or simple textual summaries later.
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

    # The explicit dictionary construction here is intentionally repetitive. It
    # keeps the exported column order stable and makes the table schema obvious
    # during maintenance.
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
    Phase 1 keeps text generation deliberately small and factual. The aim is to
    provide concise human-readable interpretation surfaces for later sinks
    without turning this module into a heavyweight natural-language report
    system.
    """
    text: dict[str, str] = {}

    # These small counts provide a quick sanity snapshot of what the report
    # actually covers. They are useful both for dashboards and for debugging
    # suspiciously small or unexpectedly empty outputs.
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

        # The text fallback strings keep the narrative deterministic when the
        # evaluation summary legitimately lacks interval-oriented fields.
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
        # This branch intentionally documents *why* the report text is thinner.
        # The absence of an evaluation result is not necessarily an error; it is
        # a legitimate lighter-weight workflow mode that still deserves a clear
        # textual explanation.
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
    this function is Phase 1's key architectural enhancement. It does not
    replace the current CSV/HTML outputs; it sits underneath them so those
    outputs can gradually become thinner consumers of one shared report object.
    """
    if not predictions:
        # Returning an explicit empty report keeps downstream behavior easier to
        # reason about than returning `None`. Sinks can inspect the stable
        # report contract and decide what to do, while callers do not have to
        # special-case the absence of a report object itself.
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

    # Grouped evaluation tables come from the structured evaluation result when
    # it is available. That preserves the evaluation package as the canonical
    # metric source of truth rather than recomputing grouped metrics here.
    by_horizon = _grouped_rows_to_frame(
        () if evaluation_result is None else evaluation_result.by_horizon
    )
    by_subject = _grouped_rows_to_frame(
        () if evaluation_result is None else evaluation_result.by_subject
    )
    by_glucose_range = _grouped_rows_to_frame(
        () if evaluation_result is None else evaluation_result.by_glucose_range
    )

    # The scalar surface combines structured evaluation summaries with a few
    # report-shape counts that are useful regardless of whether a richer metric
    # package is available.
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

    # Phase 1 keeps figures as figure-ready lightweight payloads rather than
    # fully materialized plot objects. The current Plotly sink still owns final
    # plot construction, but now it can source its input from one shared object.
    figures: dict[str, Any] = {
        "available_plot_inputs": {
            "has_prediction_table": not prediction_table.empty,
            "has_by_horizon": not by_horizon.empty,
            "has_by_subject": not by_subject.empty,
            "has_by_glucose_range": not by_glucose_range.empty,
        }
    }

    # Metadata serves as lightweight provenance for the packaged report. This is
    # especially useful later when multiple sinks or summaries need to know how
    # the report was built without re-inspecting all tables.
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


# ============================================================================
# Post-Run Prediction Export
# ============================================================================
# The helpers below operate after prediction batches have already been
# produced. They turn the canonical shared report/table into analysis-friendly
# persisted artifacts.
#
# Important design choice:
# file-writing functions below should stay thin. Their job is to serialize or
# render, not to quietly become a second hidden computation layer.


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
    plotting, inspection, and report generation.

    Phase 1 note:
    this function now delegates table construction to `build_shared_report(...)`
    so the CSV sink consumes the same canonical in-memory report surface used by
    other post-run reporting helpers.
    """
    # This export deliberately denormalizes prediction results into one flat
    # row-per-horizon table because that format is easy to inspect in a
    # notebook, easy to plot with Plotly/pandas, and easy to archive as a run
    # artifact.
    #
    # It complements the raw tensor dump written elsewhere in the workflow:
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

    # Directory creation stays in the sink because filesystem policy belongs to
    # the serialization layer, not the in-memory report builder.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(output_path, index=False)
    return output_path


# ============================================================================
# Post-Run Report Generation
# ============================================================================
# These reports are intentionally lightweight first-pass visual artifacts. They
# are meant to make a run inspectable immediately, not to replace a full
# analytics notebook.
#
# The current Plotly path is intentionally lightweight and optional:
# - it should never become the canonical metric source of truth
# - it should stay usable even in lighter workflows
# - it should remain easy to turn off in constrained environments


def _build_horizon_metrics_frame(
    *,
    prediction_table: pd.DataFrame,
    evaluation_result: EvaluationResult | None,
) -> pd.DataFrame:
    """
    Build the horizon-metrics frame used by the lightweight Plotly sink.

    Context:
    this helper preserves the existing preference order:
    - use the structured evaluation result when available
    - fall back to prediction-table-derived aggregation otherwise

    That behavior keeps current features intact while moving the aggregation
    logic into a small reusable helper.
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

    # The fallback path intentionally keeps the pre-existing convenience
    # behavior: if only the flat prediction table is available, the report sink
    # can still produce a useful horizon diagnostics plot.
    grouped = prediction_table.assign(abs_error=lambda data: data["residual"].abs()).groupby(
        "horizon_index",
        as_index=False,
    )

    # Grouping by horizon index gives us a simple answer to one of the most
    # important forecasting diagnostics questions:
    # "How does error behave as we predict farther into the future?"
    #
    # That horizon-wise view is often more informative than one single global
    # metric because short-horizon and long-horizon behavior can differ a lot.
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

    Context:
    these reports are intended to make each run immediately inspectable
    without requiring a separate notebook.

    Phase 1 note:
    callers may now provide `shared_report` directly so the HTML sink can
    consume the canonical in-memory reporting surface. The prediction-table-path
    input is retained for backwards compatibility and lighter workflows.
    """
    # These reports are intentionally lightweight first-pass diagnostics, not a
    # complete experiment-reporting system. The aim is to generate a few useful
    # HTML artifacts automatically from the flat prediction table so every run
    # leaves behind something visual and shareable.
    if report_dir is None:
        return {}
    report_dir = Path(report_dir)
    if not _has_module("plotly"):
        return {}

    frame: pd.DataFrame
    if shared_report is not None:
        # Prefer the canonical in-memory shared report when it is already
        # available. This avoids redundant disk reads and fits the new Phase 1
        # design goal of treating sinks as consumers rather than recomputers.
        frame = shared_report.tables.get("prediction_table", pd.DataFrame()).copy()
    else:
        if prediction_table_path is None:
            return {}
        prediction_table_path = Path(prediction_table_path)
        if not prediction_table_path.exists():
            return {}

        # The path-based fallback preserves the previous usage style so lighter
        # workflows and existing callers do not have to change immediately.
        frame = pd.read_csv(prediction_table_path)

    if frame.empty:
        return {}

    import plotly.express as px
    import plotly.graph_objects as go

    report_dir.mkdir(parents=True, exist_ok=True)
    report_paths: dict[str, Path] = {}

    # Residual distribution is a fast top-level diagnostic for both bias shape
    # and outlier behavior. It is intentionally simple and cheap to compute, so
    # every run can leave behind at least one immediately useful error view.
    residual_histogram = px.histogram(
        frame,
        x="residual",
        nbins=50,
        title="Residual Distribution",
    )
    residual_histogram_path = report_dir / "residual_histogram.html"
    residual_histogram.write_html(str(residual_histogram_path))
    report_paths["residual_histogram"] = residual_histogram_path

    # Prefer the structured evaluation result when it exists because it is the
    # repository's canonical detailed-metric surface. We still keep the table
    # fallback so report generation remains usable in lighter workflows that
    # only have the flat prediction table available.
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

    # If interval-width information exists, expose it on a secondary axis so we
    # can compare point-error growth and uncertainty-width growth in one compact
    # diagnostic without visually collapsing the scales into each other.
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

    overview_fig = go.Figure()

    # Subject limiting is intentional. These HTML artifacts are meant to be
    # quick, lightweight overviews, not exhaustive all-subject dashboards. A
    # capped subject subset keeps the output readable and avoids bloating the
    # auto-generated report for large runs.
    subject_ids = list(dict.fromkeys(frame["subject_id"].tolist()))[:max_subjects]
    filtered = frame[frame["subject_id"].isin(subject_ids)].copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
    filtered.sort_values(["subject_id", "timestamp"], inplace=True)

    for subject_id in subject_ids:
        subject_frame = filtered[filtered["subject_id"] == subject_id]
        if subject_frame.empty:
            continue

        # Plot the observed target first so the prediction traces are visually
        # interpreted relative to the true glucose trajectory.
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

            # Plotly interval fill uses two traces: first an invisible upper
            # bound, then the lower bound filled to the previous trace. The
            # explicit trace pair here is therefore intentional rather than
            # redundant.
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
