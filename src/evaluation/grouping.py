from __future__ import annotations

# AI-assisted maintenance note:
# This module owns grouped aggregation helpers for detailed evaluation.
#
# Responsibility boundary:
# - define the default glucose-range buckets
# - accumulate row-level metrics into grouped summaries
# - keep grouping logic independent from plotting and logger concerns
#
# Why this split helps:
# grouped evaluation has a different maintenance profile than primitive metric
# formulas, so it is clearer to keep "how do we aggregate rows?" separate from
# "how do we compute one metric?"
#
# Row contract reminder:
# the evaluator flattens prediction batches into one row per
# sample/horizon-point before calling this module. That means grouping code can
# stay in ordinary Python numeric terms instead of needing to understand the
# original tensor layouts directly.

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Sequence

from evaluation.types import GroupedMetricRow


DEFAULT_GLUCOSE_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("lt_70", None, 70.0),
    ("70_to_180", 70.0, 180.0),
    ("gt_180", 180.0, None),
)
# These default ranges are intentionally lightweight baseline buckets rather
# than a full clinical taxonomy. They provide one useful first-pass slice for
# glucose-forecast evaluation without implying that the project's evaluation
# story is clinically complete.


@dataclass
class _GroupedAccumulator:
    count: int = 0
    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    bias_sum: float = 0.0
    pinball_sum: float = 0.0
    interval_width_sum: float = 0.0
    interval_width_count: int = 0
    coverage_sum: float = 0.0
    coverage_count: int = 0

    def update(self, row: dict[str, Any]) -> None:
        # This accumulator intentionally keeps only the running totals needed by
        # the current grouped-metric surface. That keeps grouped aggregation
        # cheap and easy to inspect.
        #
        # In other words:
        # - the evaluator does the tensor-to-row flattening
        # - this accumulator does simple numeric folding
        # - reporting later consumes the already-aggregated grouped rows
        self.count += 1
        self.abs_error_sum += float(row["abs_error"])
        self.squared_error_sum += float(row["squared_error"])
        self.bias_sum += float(row["residual"])
        self.pinball_sum += float(row["pinball_loss"])

        interval_width = row.get("interval_width")
        if interval_width is not None:
            self.interval_width_sum += float(interval_width)
            self.interval_width_count += 1

        is_covered = row.get("is_covered")
        if is_covered is not None:
            self.coverage_sum += float(is_covered)
            self.coverage_count += 1

    def to_row(self, *, group_name: str, group_value: str | int) -> GroupedMetricRow:
        # Returning an explicit zeroed row for the empty case keeps the grouped
        # result shape predictable, even though current callers normally only
        # build accumulators for seen groups.
        if self.count == 0:
            return GroupedMetricRow(
                group_name=group_name,
                group_value=group_value,
                count=0,
                mae=0.0,
                rmse=0.0,
                bias=0.0,
                overall_pinball_loss=0.0,
            )

        # The grouped metrics intentionally mirror the top-level summary fields
        # where practical so consumers can compare "global" and "slice-level"
        # results without translating between different metric schemas.
        return GroupedMetricRow(
            group_name=group_name,
            group_value=group_value,
            count=self.count,
            mae=self.abs_error_sum / self.count,
            rmse=sqrt(self.squared_error_sum / self.count),
            bias=self.bias_sum / self.count,
            overall_pinball_loss=self.pinball_sum / self.count,
            mean_interval_width=(
                self.interval_width_sum / self.interval_width_count
                if self.interval_width_count > 0
                else None
            ),
            empirical_interval_coverage=(
                self.coverage_sum / self.coverage_count
                if self.coverage_count > 0
                else None
            ),
        )


def glucose_range_label(
    target_value: float,
    *,
    glucose_bands: Sequence[tuple[str, float | None, float | None]] = DEFAULT_GLUCOSE_BANDS,
) -> str:
    """
    Assign one target value to a configured glucose-range bucket.

    Context:
    this is used by grouped evaluation to provide a simple range-aware view of
    performance without hardwiring glucose-specific slicing into the primitive
    metric functions.
    """

    # Each band is interpreted as:
    # - lower bound inclusive when present
    # - upper bound exclusive when present
    #
    # That keeps adjacent buckets non-overlapping and deterministic.
    for label, lower, upper in glucose_bands:
        if lower is not None and target_value < lower:
            continue
        if upper is not None and target_value >= upper:
            continue
        return label
    return "unbounded"


def grouped_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    group_name: str,
) -> tuple[GroupedMetricRow, ...]:
    """
    Aggregate flat evaluation rows into grouped metric rows.

    Context:
    the evaluator first reduces prediction batches into flat per-sample /
    per-horizon rows. This helper then folds those rows into one grouped view.
    """

    accumulators: dict[str | int, _GroupedAccumulator] = {}
    for row in rows:
        group_value = row[group_name]
        if group_value not in accumulators:
            accumulators[group_value] = _GroupedAccumulator()
        accumulators[group_value].update(row)

    # Group order currently follows first appearance in the flattened rows. For
    # the current use cases that is acceptable and keeps the implementation
    # simple, but callers that need a different presentation order should sort
    # the returned rows explicitly at the reporting layer.
    return tuple(
        accumulator.to_row(group_name=group_name, group_value=group_value)
        for group_value, accumulator in accumulators.items()
    )
