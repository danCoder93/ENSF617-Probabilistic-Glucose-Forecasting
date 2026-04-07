from __future__ import annotations

# AI-assisted maintenance note:
# This module defines the structured data contracts used by the evaluation
# package.
#
# Responsibility boundary:
# - describe what one normalized evaluation batch looks like
# - describe what scalar and grouped evaluation outputs look like
# - stay framework-light so downstream code can serialize or inspect results
#   easily
#
# What does not live here:
# - metric formulas
# - tensor normalization logic
# - reporting or plotting code
#
# Why these contracts matter:
# the evaluation package deliberately uses a small number of explicit
# dataclasses instead of passing around loose dictionaries. That makes the
# boundary between "normalized evaluation data" and "rendered artifacts"
# easier to inspect, type-check, and serialize.

from dataclasses import dataclass, field
from typing import Any, Mapping

from torch import Tensor


@dataclass(frozen=True)
class EvaluationBatch:
    """
    Canonical in-memory view of one prediction/target batch for evaluation.

    Context:
    this is the handoff object between low-level tensor normalization and the
    higher-level evaluator/aggregation code.
    """

    # Canonical tensor layout conventions:
    # - `predictions`: `[batch, horizon, quantiles]`
    # - `target`: `[batch, horizon]`
    # - `point_prediction`: `[batch, horizon]`
    #
    # Keeping these shapes explicit in one dataclass is important because the
    # rest of the evaluation package assumes it can aggregate across batches
    # without re-checking "does this batch still have a trailing singleton
    # target dimension?" or "is this point forecast already detached from the
    # quantile stack?" on every code path.
    predictions: Tensor
    target: Tensor
    point_prediction: Tensor
    quantiles: tuple[float, ...]
    point_quantile: float
    # Metadata is normalized into one tuple per field so sample-level indexing
    # stays predictable regardless of whether the original batch metadata came
    # from tensors, lists, tuples, or scalar values.
    metadata: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSummary:
    """
    Scalar summary metrics for one evaluated dataset slice.

    Context:
    these fields are intended to be easy to log, serialize, compare across
    runs, and embed in top-level artifact summaries.
    """

    # `count` is the number of evaluated target points, not the number of
    # batches. That makes the summary stable across different dataloader batch
    # sizes and more meaningful when comparing runs with the same held-out
    # window count but different loader settings.
    count: int
    mae: float
    rmse: float
    bias: float
    overall_pinball_loss: float
    # Quantile-specific losses are keyed by string rather than raw float so the
    # result is safer to serialize into JSON and easier to consume from simple
    # dashboards or text summaries.
    pinball_loss_by_quantile: Mapping[str, float] = field(default_factory=dict)
    mean_interval_width: float | None = None
    empirical_interval_coverage: float | None = None


@dataclass(frozen=True)
class GroupedMetricRow:
    """
    One grouped metric row, such as one forecast horizon or one subject.

    Context:
    grouped rows make it possible to preserve richer evaluation detail without
    inventing a different ad hoc schema for each grouping dimension.
    """

    # This one row shape is intentionally reused for horizon-, subject-, and
    # glucose-band-level summaries. That keeps the reporting layer simple: it
    # can render "a grouped table" without caring which exact grouping
    # dimension produced it.
    group_name: str
    group_value: str | int
    count: int
    mae: float
    rmse: float
    bias: float
    overall_pinball_loss: float
    mean_interval_width: float | None = None
    empirical_interval_coverage: float | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """
    Structured detailed evaluation output for one prediction run.

    Context:
    this is the canonical detailed-evaluation payload returned by the
    evaluation package and later consumed by reporting or artifact code.
    """

    # The summary is the quick top-line view; the grouped tuples preserve the
    # richer "where is the model performing differently?" slices without
    # forcing every consumer to recompute them from raw predictions.
    summary: MetricSummary
    by_horizon: tuple[GroupedMetricRow, ...] = ()
    by_subject: tuple[GroupedMetricRow, ...] = ()
    by_glucose_range: tuple[GroupedMetricRow, ...] = ()
    # These fields document how the point forecast and quantile stack were
    # interpreted when the result was built, which matters when comparing runs
    # that may change the configured quantile set later.
    point_quantile: float = 0.5
    quantiles: tuple[float, ...] = ()
