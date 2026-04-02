from __future__ import annotations

# AI-assisted maintenance note:
# `observability` is now a small package rather than one giant module.
#
# Why keep this file:
# - existing callers already use imports like `from observability import
#   setup_observability, BatchAuditCallback`
# - that public import surface is convenient for `train.py`, `main.py`, tests,
#   and notebooks
# - this package-level facade lets us preserve those imports while moving the
#   implementation into smaller files with tighter responsibility boundaries
#
# Internal layout:
# - `runtime.py` assembles loggers/profilers/path artifacts for a run
# - `logging_utils.py` holds logger/hparameter helper functions shared by
#   callbacks and training orchestration
# - `tensors.py` centralizes nested tensor/batch normalization helpers
# - `callbacks.py` is the stable public facade and callback assembly point
# - `debug_callbacks.py`, `system_callbacks.py`, `parameter_callbacks.py`, and
#   `prediction_callbacks.py` hold the split callback implementations
# - `reporting.py` owns post-run exports and HTML report generation
#
# Responsibility boundary:
# - expose the repository's public observability surface in one place
# - let orchestration code import logging/profiler/callback/reporting helpers
#   without caring about the internal package split
# - keep the internal modules free to evolve without forcing widespread import
#   churn elsewhere in the codebase
#
# What does *not* live here:
# - the implementation details of callbacks, logging, profiling, or reporting
# - training orchestration
# - runtime environment policy
#
# The goal is not to hide the split; it is to make each part easier to inspect
# without forcing the rest of the repository to care about the file boundaries.


# ============================================================================
# Public Observability Surface
# ============================================================================
# Re-export the commonly used runtime helpers, callbacks, and reporting tools
# so the rest of the repository can continue using short package-level imports.

from observability.callbacks import (
    ActivationStatsCallback,
    BatchAuditCallback,
    GradientStatsCallback,
    ModelTensorBoardCallback,
    ParameterHistogramCallback,
    ParameterScalarTelemetryCallback,
    PredictionFigureCallback,
    SystemTelemetryCallback,
    build_observability_callbacks,
)
from observability.logging_utils import (
    log_hyperparameters,
    log_metrics_to_loggers,
)
from observability.reporting import (
    export_prediction_table,
    generate_plotly_reports,
)
from observability.runtime import (
    ObservabilityArtifacts,
    build_lightning_logger,
    build_profiler,
    setup_observability,
    setup_text_logger,
)


# `__all__` is the stable import contract for package-level callers like
# `train.py`, `main.py`, and the observability-focused tests.
__all__ = [
    "ActivationStatsCallback",
    "BatchAuditCallback",
    "GradientStatsCallback",
    "ModelTensorBoardCallback",
    "ObservabilityArtifacts",
    "ParameterHistogramCallback",
    "ParameterScalarTelemetryCallback",
    "PredictionFigureCallback",
    "SystemTelemetryCallback",
    "build_lightning_logger",
    "build_observability_callbacks",
    "build_profiler",
    "export_prediction_table",
    "generate_plotly_reports",
    "log_hyperparameters",
    "log_metrics_to_loggers",
    "setup_observability",
    "setup_text_logger",
]
