from __future__ import annotations

# AI-assisted maintenance note:
# `evaluation` is the canonical home for model-quality metric computation and
# structured evaluation outputs.
#
# Why keep this file:
# - callers can use concise imports like `from evaluation import
#   evaluate_prediction_batches`
# - the public facade keeps the rest of the repository decoupled from the
#   package's internal file split
# - this mirrors the convenience style already used by `config` and
#   `observability`
#
# Internal layout:
# - `types.py` defines the structured evaluation result contracts
# - `core.py` normalizes tensors, quantiles, and metadata into canonical forms
# - `metrics.py` owns primitive metric definitions
# - `grouping.py` owns grouped aggregation helpers
# - `evaluator.py` assembles those pieces into end-to-end evaluation results
#
# Architectural intent:
# the rest of the repository should rarely need to care which exact evaluation
# submodule owns a helper. Most callers only need two mental buckets:
# - "I need a primitive metric or tensor-normalization helper"
# - "I need a full structured evaluation result"
#
# Re-exporting the common entrypoints here keeps `main.py`, reporting code, and
# tests readable while still letting the internal package stay split by
# responsibility.

from evaluation.core import (
    build_evaluation_batch,
    normalize_batch_metadata,
    normalize_target_tensor,
    quantile_key,
    select_point_prediction,
)
from evaluation.evaluator import evaluate_batch, evaluate_prediction_batches
from evaluation.grouping import DEFAULT_GLUCOSE_BANDS, glucose_range_label, grouped_metrics
from evaluation.metrics import (
    empirical_interval_coverage,
    mean_absolute_error,
    mean_bias,
    mean_prediction_interval_width,
    pinball_loss,
    pinball_loss_by_quantile,
    root_mean_squared_error,
)
from evaluation.types import EvaluationBatch, EvaluationResult, GroupedMetricRow, MetricSummary

__all__ = [
    "DEFAULT_GLUCOSE_BANDS",
    "EvaluationBatch",
    "EvaluationResult",
    "GroupedMetricRow",
    "MetricSummary",
    "build_evaluation_batch",
    "empirical_interval_coverage",
    "evaluate_batch",
    "evaluate_prediction_batches",
    "glucose_range_label",
    "grouped_metrics",
    "mean_absolute_error",
    "mean_bias",
    "mean_prediction_interval_width",
    "normalize_batch_metadata",
    "normalize_target_tensor",
    "pinball_loss",
    "pinball_loss_by_quantile",
    "quantile_key",
    "root_mean_squared_error",
    "select_point_prediction",
]
