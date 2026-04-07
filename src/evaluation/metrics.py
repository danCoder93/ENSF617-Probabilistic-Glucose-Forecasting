from __future__ import annotations

# AI-assisted maintenance note:
# This module owns primitive evaluation metrics for the forecasting project.
#
# Responsibility boundary:
# - implement reusable metric formulas
# - stay independent of Lightning logging, plotting, and file I/O
# - operate on already-normalized tensors from `evaluation.core`
#
# What does not live here:
# - grouped aggregation
# - metadata-aware slicing
# - report/table generation
#
# Design philosophy:
# these helpers should be boring, explicit, and easy to compare against the
# mathematical definitions used in the paper/codebase discussion. Keeping the
# formulas small and dependency-light also makes them straightforward to unit
# test and reuse from both the model-side live metrics and the held-out
# evaluator.

from typing import Sequence

import torch
from torch import Tensor


def mean_absolute_error(prediction: Tensor, target: Tensor) -> Tensor:
    """
    Mean absolute error over a point forecast.

    Context:
    this is the most interpretable scalar point-forecast error for quick run
    comparisons.
    """

    # Inputs are expected to already be shape-aligned point forecasts and
    # targets. This helper intentionally does not hide shape mismatches with
    # implicit squeezing or broadcasting.
    return torch.mean(torch.abs(prediction - target))


def root_mean_squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    """
    Root mean squared error over a point forecast.

    Context:
    RMSE complements MAE by emphasizing larger misses more strongly.
    """

    # RMSE is computed directly from squared residuals rather than reusing MAE
    # machinery so the intent stays obvious to readers scanning the code.
    return torch.sqrt(torch.mean(torch.square(prediction - target)))


def mean_bias(prediction: Tensor, target: Tensor) -> Tensor:
    """
    Signed mean residual over a point forecast.

    Context:
    this reveals whether the representative point forecast tends to over- or
    under-predict systematically.
    """

    # Positive bias means over-prediction on average; negative bias means
    # under-prediction on average.
    return torch.mean(prediction - target)


def pinball_loss(
    predictions: Tensor,
    target: Tensor,
    quantiles: Sequence[float],
) -> Tensor:
    """
    Mean pinball loss across all quantiles.

    Context:
    this mirrors the model's probabilistic supervision objective, but is kept
    here so evaluation and training can share one consistent definition.
    """

    # The quantile tensor is built on the prediction tensor's device/dtype so
    # the loss can run without extra device transfers or dtype mismatches.
    quantile_tensor = predictions.new_tensor(tuple(float(value) for value in quantiles)).view(
        1,
        1,
        -1,
    )
    # Error shape:
    # - `target.unsqueeze(-1)`: `[batch, horizon, 1]`
    # - `predictions`: `[batch, horizon, quantiles]`
    # - result: `[batch, horizon, quantiles]`
    #
    # Each quantile channel then receives the standard asymmetric pinball
    # penalty for its own residuals.
    errors = target.unsqueeze(-1) - predictions
    return torch.maximum((quantile_tensor - 1.0) * errors, quantile_tensor * errors).mean()


def pinball_loss_by_quantile(
    predictions: Tensor,
    target: Tensor,
    quantiles: Sequence[float],
) -> dict[float, Tensor]:
    """
    Mean pinball loss for each configured quantile independently.

    Context:
    per-quantile losses are more diagnostic than one global average when
    assessing calibration-like behavior across the predictive distribution.
    """

    # This helper mirrors `pinball_loss(...)` but keeps the quantile channels
    # separated so later reports can spot asymmetries across the predictive
    # distribution instead of only seeing one collapsed average.
    quantile_values = tuple(float(value) for value in quantiles)
    errors = target.unsqueeze(-1) - predictions
    losses: dict[float, Tensor] = {}
    for index, quantile in enumerate(quantile_values):
        quantile_errors = errors[..., index]
        quantile_loss = torch.maximum(
            (quantile - 1.0) * quantile_errors,
            quantile * quantile_errors,
        ).mean()
        losses[quantile] = quantile_loss
    return losses


def mean_prediction_interval_width(predictions: Tensor) -> Tensor | None:
    """
    Mean width of the outer prediction interval, if multiple quantiles exist.

    Context:
    interval width is a simple sharpness signal for probabilistic forecasts.
    """

    # The current evaluation surface interprets the outermost configured
    # quantiles as the prediction interval bounds. That is intentionally simple
    # and matches the project's existing `(low, median, high)` style outputs.
    if predictions.shape[-1] < 2:
        return None
    return (predictions[..., -1] - predictions[..., 0]).mean()


def empirical_interval_coverage(predictions: Tensor, target: Tensor) -> Tensor | None:
    """
    Empirical coverage of the outer prediction interval, if it exists.

    Context:
    this is the simplest first-pass calibration-style summary for the outer
    predictive interval.
    """

    if predictions.shape[-1] < 2:
        return None
    # Coverage is computed against the same outer quantile pair used for the
    # interval-width summary, so sharpness and first-pass calibration are
    # talking about the same interval definition.
    lower = predictions[..., 0]
    upper = predictions[..., -1]
    return ((target >= lower) & (target <= upper)).float().mean()
