from __future__ import annotations

# AI-assisted maintenance note:
# This module owns the lightweight narrative text surfaces packaged into the
# repository's canonical post-run shared report.
#
# Why this file exists:
# - the reporting package already distinguishes between canonical report
#   construction and sink-specific rendering/export
# - text generation is easier to maintain when it lives in one focused module
#   instead of being scattered across builders, workflow code, and sink modules
# - dashboard/TensorBoard/report sinks can then consume one stable set of text
#   surfaces without each sink inventing its own prose
#
# Responsibility boundary:
# - derive concise text summaries from already-canonical report inputs
# - keep wording factual and deterministic
# - package compact dataset/evaluation/probabilistic/grouped-table summaries
# - optionally package compact data-summary text when a DataModule summary is
#   available upstream
#
# What does *not* live here:
# - metric computation
# - row construction
# - grouped-table assembly
# - TensorBoard/HTML/CSV sink policy
#
# In other words, this file packages interpretation text from canonical inputs.
# It does not compute new truth.

from typing import Any, Mapping, Sequence

import pandas as pd

from evaluation import EvaluationResult


def format_optional_metric(value: Any) -> str:
    """Format a scalar-like metric value for deterministic report text."""
    if value is None:
        return "unavailable"

    if pd.isna(value):
        return "unavailable"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def sorted_grouped_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a grouped metrics frame sorted by its canonical group axis."""
    if frame.empty or "group_value" not in frame.columns:
        return frame.copy()

    return frame.copy().sort_values(by=["group_value"]).reset_index(drop=True)


def _safe_nested_value(mapping: Mapping[str, Any], *path: str) -> Any:
    """Return a nested mapping value when the path exists, otherwise `None`."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _format_subject_count_text(data_summary: Mapping[str, Any]) -> str | None:
    """Build one compact subject/sample/window count sentence when available."""
    subject_count = _safe_nested_value(data_summary, "dataset", "num_subjects")
    row_count = _safe_nested_value(data_summary, "dataset", "num_rows")
    split_window_counts = _safe_nested_value(data_summary, "splits", "window_counts")

    parts: list[str] = []
    if row_count is not None:
        parts.append(f"{row_count} cleaned row(s)")
    if subject_count is not None:
        parts.append(f"{subject_count} subject(s)")

    text = ""
    if parts:
        text = "Data summary covers " + ", ".join(parts) + "."

    if isinstance(split_window_counts, Mapping) and split_window_counts:
        split_parts = [
            f"{split_name}={split_window_counts[split_name]}"
            for split_name in ("train", "val", "test")
            if split_name in split_window_counts
        ]
        if split_parts:
            if text:
                text += " "
            text += "Window counts by split: " + ", ".join(split_parts) + "."

    return text or None


def _format_target_summary_text(data_summary: Mapping[str, Any]) -> str | None:
    """Build one compact target-distribution sentence when available."""
    target_stats = _safe_nested_value(data_summary, "target", "summary")
    if not isinstance(target_stats, Mapping):
        return None

    parts: list[str] = []
    for label, key in (
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
    ):
        if key in target_stats:
            parts.append(f"{label}={format_optional_metric(target_stats[key])}")

    if not parts:
        return None

    return "Target distribution summary: " + ", ".join(parts) + "."


def _format_missingness_summary_text(data_summary: Mapping[str, Any]) -> str | None:
    """Build one compact missingness/data-health sentence when available."""
    missing_summary = _safe_nested_value(data_summary, "data_quality", "missing_fraction_by_column")
    if not isinstance(missing_summary, Mapping) or not missing_summary:
        return None

    # Rank the most-missing columns first so the summary remains compact and
    # still surfaces the most useful data-quality signal.
    ranked = sorted(
        ((str(column), value) for column, value in missing_summary.items()),
        key=lambda item: float(item[1]) if item[1] is not None else float("-inf"),
        reverse=True,
    )[:3]

    if not ranked:
        return None

    top_columns = ", ".join(
        f"{column}={format_optional_metric(value)}" for column, value in ranked
    )
    return "Highest missing-data fractions: " + top_columns + "."


def build_data_summary_overview(data_summary: Mapping[str, Any] | None) -> str:
    """Build one compact data-summary overview block from DataModule output.

    Context:
    the DataModule owns dataset-summary truth through `describe_data()`. This
    helper does not recompute that truth; it only turns an already-collected
    summary into sink-friendly narrative text for the shared report.
    """
    if not data_summary:
        return (
            "Data-summary overview is unavailable because no DataModule "
            "summary was attached to this shared report."
        )

    parts: list[str] = []

    count_text = _format_subject_count_text(data_summary)
    if count_text:
        parts.append(count_text)

    target_text = _format_target_summary_text(data_summary)
    if target_text:
        parts.append(target_text)

    missingness_text = _format_missingness_summary_text(data_summary)
    if missingness_text:
        parts.append(missingness_text)

    if not parts:
        return (
            "A DataModule summary was attached to this shared report, but no "
            "compact data-summary text fields could be derived from its current shape."
        )

    return " ".join(parts)


def build_horizon_overview(by_horizon: pd.DataFrame) -> str:
    """Build a compact interpretation of horizon-wise error behavior."""
    if by_horizon.empty:
        return (
            "Forecast-horizon grouped metrics are unavailable, so the shared "
            "report cannot summarize how error changes across the step-ahead "
            "axis for this run."
        )

    frame = sorted_grouped_frame(by_horizon)
    first_row = frame.iloc[0]
    last_row = frame.iloc[-1]

    text = (
        "Across the forecast horizon, MAE changes from "
        f"{format_optional_metric(first_row.get('mae'))} at "
        f"horizon={first_row.get('group_value')} to "
        f"{format_optional_metric(last_row.get('mae'))} at "
        f"horizon={last_row.get('group_value')}, while RMSE changes from "
        f"{format_optional_metric(first_row.get('rmse'))} to "
        f"{format_optional_metric(last_row.get('rmse'))}."
    )

    if "bias" in frame.columns:
        absolute_bias = frame["bias"].abs()
        if not bool(absolute_bias.isna().all()):
            bias_row = frame.loc[absolute_bias.idxmax()]
            text += (
                " The largest absolute directional error appears at "
                f"horizon={bias_row.get('group_value')} with bias="
                f"{format_optional_metric(bias_row.get('bias'))}."
            )

    return text


def build_probabilistic_overview(
    *,
    evaluation_result: EvaluationResult | None,
    by_horizon: pd.DataFrame,
) -> str:
    """Build a compact probabilistic-quality interpretation block."""
    if evaluation_result is None:
        return (
            "Probabilistic evaluation detail is unavailable because no structured "
            "evaluation result was supplied to the shared-report builder."
        )

    summary = evaluation_result.summary
    text = (
        "Probabilistic overview: overall_pinball_loss="
        f"{format_optional_metric(summary.overall_pinball_loss)}, "
        "mean_interval_width="
        f"{format_optional_metric(summary.mean_interval_width)}, "
        "empirical_interval_coverage="
        f"{format_optional_metric(summary.empirical_interval_coverage)}."
    )

    if not by_horizon.empty and "empirical_interval_coverage" in by_horizon.columns:
        coverage_series = by_horizon["empirical_interval_coverage"]
        if not bool(coverage_series.isna().all()):
            coverage_min_row = by_horizon.loc[coverage_series.idxmin()]
            coverage_max_row = by_horizon.loc[coverage_series.idxmax()]
            text += (
                " Horizon-level empirical coverage ranges from "
                f"{format_optional_metric(coverage_min_row.get('empirical_interval_coverage'))} "
                f"at horizon={coverage_min_row.get('group_value')} to "
                f"{format_optional_metric(coverage_max_row.get('empirical_interval_coverage'))} "
                f"at horizon={coverage_max_row.get('group_value')}."
            )

    if not by_horizon.empty and "mean_interval_width" in by_horizon.columns:
        width_series = by_horizon["mean_interval_width"]
        if not bool(width_series.isna().all()):
            width_min_row = by_horizon.loc[width_series.idxmin()]
            width_max_row = by_horizon.loc[width_series.idxmax()]
            text += (
                " Horizon-level interval width ranges from "
                f"{format_optional_metric(width_min_row.get('mean_interval_width'))} "
                f"at horizon={width_min_row.get('group_value')} to "
                f"{format_optional_metric(width_max_row.get('mean_interval_width'))} "
                f"at horizon={width_max_row.get('group_value')}."
            )

    return text


def build_subject_variability_overview(by_subject: pd.DataFrame) -> str:
    """Build a compact summary of subject-level variability."""
    if by_subject.empty:
        return (
            "Subject-level grouped metrics are unavailable, so the shared report "
            "cannot summarize cross-subject variability for this run."
        )

    frame = sorted_grouped_frame(by_subject)
    mae_worst_row = frame.loc[frame["mae"].idxmax()]
    mae_best_row = frame.loc[frame["mae"].idxmin()]

    text = (
        "Across subjects, the lowest MAE appears at subject="
        f"{mae_best_row.get('group_value')} with MAE="
        f"{format_optional_metric(mae_best_row.get('mae'))}, while the highest "
        "MAE appears at subject="
        f"{mae_worst_row.get('group_value')} with MAE="
        f"{format_optional_metric(mae_worst_row.get('mae'))}."
    )

    if "bias" in frame.columns:
        absolute_bias = frame["bias"].abs()
        if not bool(absolute_bias.isna().all()):
            bias_row = frame.loc[absolute_bias.idxmax()]
            text += (
                " The largest absolute subject bias appears at subject="
                f"{bias_row.get('group_value')} with bias="
                f"{format_optional_metric(bias_row.get('bias'))}."
            )

    return text


def build_glucose_range_overview(by_glucose_range: pd.DataFrame) -> str:
    """Build a compact summary of glucose-range performance differences."""
    if by_glucose_range.empty:
        return (
            "Glucose-range grouped metrics are unavailable, so the shared report "
            "cannot summarize range-specific behavior for this run."
        )

    frame = sorted_grouped_frame(by_glucose_range)
    mae_worst_row = frame.loc[frame["mae"].idxmax()]
    mae_best_row = frame.loc[frame["mae"].idxmin()]

    text = (
        "Across glucose ranges, the lowest MAE appears at range="
        f"{mae_best_row.get('group_value')} with MAE="
        f"{format_optional_metric(mae_best_row.get('mae'))}, while the highest "
        "MAE appears at range="
        f"{mae_worst_row.get('group_value')} with MAE="
        f"{format_optional_metric(mae_worst_row.get('mae'))}."
    )

    if "empirical_interval_coverage" in frame.columns:
        coverage_series = frame["empirical_interval_coverage"]
        if not bool(coverage_series.isna().all()):
            coverage_low_row = frame.loc[coverage_series.idxmin()]
            coverage_high_row = frame.loc[coverage_series.idxmax()]
            text += (
                " Empirical interval coverage ranges from "
                f"{format_optional_metric(coverage_low_row.get('empirical_interval_coverage'))} "
                f"at range={coverage_low_row.get('group_value')} to "
                f"{format_optional_metric(coverage_high_row.get('empirical_interval_coverage'))} "
                f"at range={coverage_high_row.get('group_value')}."
            )

    return text


def build_dashboard_overview_text(
    *,
    prediction_table: pd.DataFrame,
    evaluation_result: EvaluationResult | None,
    by_horizon: pd.DataFrame,
) -> str:
    """Build one compact dashboard-first orientation block."""
    sample_count = len(prediction_table)
    subject_count = (
        int(prediction_table["subject_id"].nunique())
        if "subject_id" in prediction_table
        else 0
    )
    horizon_count = (
        int(prediction_table["horizon_index"].nunique())
        if "horizon_index" in prediction_table
        else 0
    )

    if evaluation_result is None:
        return (
            "Dashboard overview: "
            f"{sample_count} forecast row(s), {subject_count} subject(s), and "
            f"{horizon_count} horizon step(s) are available, but structured "
            "evaluation metrics are unavailable for this run."
        )

    summary = evaluation_result.summary
    text = (
        "Dashboard overview: "
        f"{sample_count} forecast row(s), {subject_count} subject(s), and "
        f"{horizon_count} horizon step(s). "
        f"Top-line MAE={format_optional_metric(summary.mae)}, "
        f"RMSE={format_optional_metric(summary.rmse)}, "
        f"bias={format_optional_metric(summary.bias)}, "
        f"mean_interval_width={format_optional_metric(summary.mean_interval_width)}, "
        f"empirical_interval_coverage={format_optional_metric(summary.empirical_interval_coverage)}."
    )

    if not by_horizon.empty:
        sorted_horizon = sorted_grouped_frame(by_horizon)
        first_row = sorted_horizon.iloc[0]
        last_row = sorted_horizon.iloc[-1]
        text += (
            " Horizon MAE spans from "
            f"{format_optional_metric(first_row.get('mae'))} at "
            f"horizon={first_row.get('group_value')} to "
            f"{format_optional_metric(last_row.get('mae'))} at "
            f"horizon={last_row.get('group_value')}."
        )

    return text


def build_health_warning_text(
    *,
    prediction_table: pd.DataFrame,
    evaluation_result: EvaluationResult | None,
) -> str:
    """Build a compact warning/anomaly summary from canonical report inputs."""
    warning_parts: list[str] = []

    if not prediction_table.empty:
        if "residual" in prediction_table.columns:
            residual_series = prediction_table["residual"]
            if not bool(residual_series.isna().all()):
                mean_abs_residual = float(residual_series.abs().mean())
                warning_parts.append(
                    "mean_abs_residual="
                    f"{format_optional_metric(mean_abs_residual)}"
                )

        if "prediction_interval_width" in prediction_table.columns:
            width_series = prediction_table["prediction_interval_width"]
            if not bool(width_series.isna().all()):
                near_zero_fraction = float((width_series.abs() <= 1e-8).mean())
                warning_parts.append(
                    "near_zero_interval_fraction="
                    f"{format_optional_metric(near_zero_fraction)}"
                )

        numeric_frame = prediction_table.select_dtypes(include=["number"])
        if not numeric_frame.empty:
            # CHANGE: Use DataFrame.isna() directly instead of applymap(pd.notna).
            # This keeps the health-warning summary compatible with the pandas
            # version in the environment and still answers the same question:
            # how many numeric cells are missing/non-finite at a high level.
            nonfinite_count = int(numeric_frame.isna().sum().sum())
            if nonfinite_count > 0:
                warning_parts.append(f"nonfinite_numeric_cells={nonfinite_count}")

    if evaluation_result is not None:
        summary = evaluation_result.summary
        coverage = summary.empirical_interval_coverage
        if coverage is not None and not pd.isna(coverage):
            if float(coverage) < 0.0 or float(coverage) > 1.0:
                warning_parts.append(
                    "coverage_out_of_range="
                    f"{format_optional_metric(coverage)}"
                )

    if not warning_parts:
        return (
            "No immediate high-level report warnings were derived from the "
            "canonical shared-report surfaces."
        )

    return "High-level report health summary: " + "; ".join(warning_parts) + "."


def build_report_text(
    *,
    prediction_table: pd.DataFrame,
    evaluation_result: EvaluationResult | None,
    quantiles: Sequence[float],
    by_horizon: pd.DataFrame,
    by_subject: pd.DataFrame,
    by_glucose_range: pd.DataFrame,
    data_summary: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build lightweight narrative text summaries for the shared report."""
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
        f"{sample_count} forecast row(s) across {subject_count} subject(s) "
        f"and {horizon_count} horizon step(s)."
    )

    if evaluation_result is not None:
        summary = evaluation_result.summary

        coverage_text = format_optional_metric(summary.empirical_interval_coverage)
        interval_text = format_optional_metric(summary.mean_interval_width)
        text["metric_overview"] = (
            "Top-level evaluation summary: "
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

    text["dashboard_overview"] = build_dashboard_overview_text(
        prediction_table=prediction_table,
        evaluation_result=evaluation_result,
        by_horizon=by_horizon,
    )
    text["health_warning_overview"] = build_health_warning_text(
        prediction_table=prediction_table,
        evaluation_result=evaluation_result,
    )
    text["horizon_overview"] = build_horizon_overview(by_horizon)
    text["probabilistic_overview"] = build_probabilistic_overview(
        evaluation_result=evaluation_result,
        by_horizon=by_horizon,
    )
    text["subject_variability_overview"] = build_subject_variability_overview(by_subject)
    text["glucose_range_overview"] = build_glucose_range_overview(by_glucose_range)

    # The data-summary text surface is intentionally additive. When the workflow
    # did not attach a DataModule summary, the rest of the shared-report text
    # contract remains unchanged.
    text["data_summary_overview"] = build_data_summary_overview(data_summary)

    return text