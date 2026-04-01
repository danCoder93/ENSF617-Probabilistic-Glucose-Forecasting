from __future__ import annotations

# AI-assisted maintenance note:
# This module assembles the evaluation package into end-to-end detailed
# evaluation results.
#
# Responsibility boundary:
# - accept normalized or raw batch-aligned prediction outputs
# - compute scalar summary metrics
# - compute grouped horizon / subject / glucose-range metrics
# - return one stable `EvaluationResult` contract for downstream consumers
#
# What does not live here:
# - model forward or loss logic
# - logger side effects
# - plotting or file export behavior
#
# Important disclaimer:
# this evaluator is intended to support research/development visibility into
# forecasting quality. It does not by itself make the repository's outputs
# clinically validated or production-ready.
#
# Architectural overview:
# - `evaluation.core` normalizes one batch at a time
# - this module concatenates those normalized batches across the run
# - tensor-level summaries are computed for the full dataset slice
# - row-level summaries are derived for grouped views like horizon or subject
#
# That hybrid approach is deliberate:
# - tensor aggregation is concise for global metrics
# - row aggregation is easier to extend for metadata-aware slices

from itertools import zip_longest
from typing import Any, Iterable, Mapping, Sequence, cast

import torch
from torch import Tensor

from evaluation.core import build_evaluation_batch, quantile_key
from evaluation.grouping import grouped_metrics, glucose_range_label
from evaluation.metrics import (
    empirical_interval_coverage,
    mean_absolute_error,
    mean_bias,
    mean_prediction_interval_width,
    pinball_loss,
    pinball_loss_by_quantile,
    root_mean_squared_error,
)
from evaluation.types import EvaluationBatch, EvaluationResult, MetricSummary


def _empty_result(
    *,
    quantiles: Sequence[float],
    point_quantile: float,
) -> EvaluationResult:
    # Keeping an explicit empty-result constructor makes the "no predictions"
    # case deterministic and easy for outer workflow code to serialize.
    #
    # This is especially useful in the top-level workflow, where test or
    # prediction stages may be skipped intentionally and we still want summary
    # serialization to stay shape-stable.
    return EvaluationResult(
        summary=MetricSummary(
            count=0,
            mae=0.0,
            rmse=0.0,
            bias=0.0,
            overall_pinball_loss=0.0,
            pinball_loss_by_quantile={
                quantile_key(quantile): 0.0 for quantile in quantiles
            },
            mean_interval_width=None,
            empirical_interval_coverage=None,
        ),
        point_quantile=point_quantile,
        quantiles=tuple(float(value) for value in quantiles),
    )


def _rows_from_batch(batch: EvaluationBatch) -> list[dict[str, Any]]:
    # The evaluator's grouped views all start from one denormalized row-per-
    # horizon representation. That keeps grouping simple and avoids forcing the
    # grouping layer to understand tensor layouts directly.
    #
    # Why flatten at all?
    # - grouped horizon metrics need one record per forecast step
    # - subject/range metrics need ordinary scalar metadata values
    # - keeping those views row-based makes future grouped slices much easier
    #   to add than if every grouping helper had to index into tensors itself
    rows: list[dict[str, Any]] = []
    batch_size = int(batch.predictions.shape[0])
    horizon_size = int(batch.predictions.shape[1])

    for sample_index in range(batch_size):
        # Subject metadata is optional in the normalized batch contract, so the
        # evaluator falls back to `"unknown"` instead of dropping the row.
        # That keeps grouped views complete even for prediction sources that do
        # not provide subject labels.
        subject_values = batch.metadata.get("subject_id")
        subject_id = (
            "unknown"
            if subject_values is None
            else str(subject_values[sample_index])
        )
        for horizon_index in range(horizon_size):
            target_value = float(batch.target[sample_index, horizon_index].item())
            point_prediction = float(
                batch.point_prediction[sample_index, horizon_index].item()
            )
            residual = point_prediction - target_value
            quantile_predictions = batch.predictions[sample_index, horizon_index]
            quantile_errors = target_value - quantile_predictions
            # Per-row pinball loss is kept as a scalar mean across quantiles so
            # grouped aggregations can summarize probabilistic quality without
            # exploding the grouped row schema into one column per quantile.
            #
            # Quantile-specific global summaries still exist separately in the
            # top-level `MetricSummary`.
            per_quantile_pinball = torch.maximum(
                (
                    quantile_predictions.new_tensor(batch.quantiles)
                    - 1.0
                )
                * quantile_errors,
                quantile_predictions.new_tensor(batch.quantiles) * quantile_errors,
            )

            interval_width: float | None = None
            is_covered: float | None = None
            if quantile_predictions.shape[-1] >= 2:
                # The row-level grouped views use the same "outermost quantiles
                # define the interval" convention as the primitive interval
                # metrics. That keeps grouped sharpness/coverage summaries
                # aligned with the top-line scalar summary.
                lower = float(quantile_predictions[0].item())
                upper = float(quantile_predictions[-1].item())
                interval_width = upper - lower
                is_covered = float(lower <= target_value <= upper)

            rows.append(
                {
                    "subject_id": subject_id,
                    "horizon_index": horizon_index,
                    "glucose_range": glucose_range_label(target_value),
                    "target": target_value,
                    "point_prediction": point_prediction,
                    "residual": residual,
                    "abs_error": abs(residual),
                    "squared_error": residual * residual,
                    "pinball_loss": float(per_quantile_pinball.mean().item()),
                    "interval_width": interval_width,
                    "is_covered": is_covered,
                }
            )
    return rows


def evaluate_batch(
    *,
    predictions: Tensor,
    target: Tensor,
    quantiles: Sequence[float],
    metadata: Mapping[str, Any] | None = None,
    point_quantile: float = 0.5,
) -> EvaluationResult:
    """
    Evaluate one prediction/target batch and return structured metrics.

    Context:
    this is the smallest public entrypoint for callers that already have one
    aligned prediction tensor and one aligned target tensor in memory.
    """

    # This thin wrapper exists so callers with one already-aligned batch do not
    # have to manually construct a singleton sequence just to reuse the common
    # multi-batch aggregation path.
    batch = build_evaluation_batch(
        predictions=predictions,
        target=target,
        quantiles=quantiles,
        metadata=metadata,
        point_quantile=point_quantile,
    )
    return _evaluate_batches([batch], quantiles=quantiles, point_quantile=point_quantile)


def evaluate_prediction_batches(
    *,
    predictions: Sequence[Tensor],
    batches: Iterable[Mapping[str, Any]],
    quantiles: Sequence[float],
    point_quantile: float = 0.5,
) -> EvaluationResult:
    """
    Evaluate a sequence of prediction tensors aligned with source batches.

    Context:
    this is the main workflow entrypoint for held-out evaluation because
    Lightning prediction typically returns one tensor per batch while the
    source dataloader still holds the aligned targets and metadata.
    """

    built_batches: list[EvaluationBatch] = []
    missing = object()
    for prediction_batch, batch in zip_longest(predictions, batches, fillvalue=missing):
        if prediction_batch is missing or batch is missing:
            raise ValueError(
                "Predictions and source batches must have the same number of batches."
            )
        # `zip_longest(...)` is intentionally paired with explicit casts after
        # the sentinel guard so static analysis can see the narrowed runtime
        # contract clearly.
        typed_prediction_batch = cast(Tensor, prediction_batch)
        typed_batch = cast(Mapping[str, Any], batch)
        built_batches.append(
            build_evaluation_batch(
                # Detaching onto CPU here keeps the final evaluation result as a
                # pure analysis artifact. The evaluator does not need gradients,
                # and moving data off-device early avoids accidental coupling to
                # the training/inference runtime state.
                predictions=typed_prediction_batch.detach().cpu(),
                target=typed_batch["target"],
                quantiles=quantiles,
                metadata=typed_batch.get("metadata"),
                point_quantile=point_quantile,
            )
        )
    return _evaluate_batches(
        built_batches,
        quantiles=quantiles,
        point_quantile=point_quantile,
    )


def _evaluate_batches(
    batches: Sequence[EvaluationBatch],
    *,
    quantiles: Sequence[float],
    point_quantile: float,
) -> EvaluationResult:
    # `_evaluate_batches(...)` is the shared aggregation core behind the two
    # public entrypoints above.
    #
    # It intentionally does two parallel reductions:
    # - concatenate tensors for compact global metrics
    # - flatten rows for flexible grouped metrics
    if not batches:
        return _empty_result(
            quantiles=quantiles,
            point_quantile=point_quantile,
        )

    # Tensor concatenation preserves the natural forecasting shapes for global
    # metrics like MAE, RMSE, overall pinball, and interval summaries.
    all_predictions = torch.cat([batch.predictions for batch in batches], dim=0)
    all_targets = torch.cat([batch.target for batch in batches], dim=0)
    all_points = torch.cat([batch.point_prediction for batch in batches], dim=0)
    # Row flattening feeds the generic grouped aggregation helpers.
    all_rows = [row for batch in batches for row in _rows_from_batch(batch)]

    per_quantile = {
        quantile_key(quantile): float(loss.item())
        for quantile, loss in pinball_loss_by_quantile(
            all_predictions,
            all_targets,
            quantiles,
        ).items()
    }
    interval_width = mean_prediction_interval_width(all_predictions)
    coverage = empirical_interval_coverage(all_predictions, all_targets)

    summary = MetricSummary(
        count=int(all_targets.numel()),
        mae=float(mean_absolute_error(all_points, all_targets).item()),
        rmse=float(root_mean_squared_error(all_points, all_targets).item()),
        bias=float(mean_bias(all_points, all_targets).item()),
        overall_pinball_loss=float(
            pinball_loss(all_predictions, all_targets, quantiles).item()
        ),
        pinball_loss_by_quantile=per_quantile,
        mean_interval_width=(
            None if interval_width is None else float(interval_width.item())
        ),
        empirical_interval_coverage=(
            None if coverage is None else float(coverage.item())
        ),
    )

    # The returned `EvaluationResult` is intentionally self-contained: callers
    # should not need access to the raw batch tensors just to build summaries,
    # grouped tables, or top-level JSON artifacts afterward.
    return EvaluationResult(
        summary=summary,
        by_horizon=grouped_metrics(all_rows, group_name="horizon_index"),
        by_subject=grouped_metrics(all_rows, group_name="subject_id"),
        by_glucose_range=grouped_metrics(all_rows, group_name="glucose_range"),
        point_quantile=point_quantile,
        quantiles=tuple(float(value) for value in quantiles),
    )
