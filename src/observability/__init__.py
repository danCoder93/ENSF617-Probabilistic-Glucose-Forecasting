from __future__ import annotations

# AI-assisted maintenance note:
# `observability` is the stable package-level facade for the repository's
# logging, callback, profiler, runtime-observability, and post-run reporting
# utilities.
#
# Why keep this file:
# - existing callers already use concise imports like
#   `from observability import setup_observability, BatchAuditCallback`
# - that short public surface is convenient for `train.py`, `main.py`, tests,
#   notebooks, and one-off debugging scripts
# - this facade lets the internal package evolve into smaller, more focused
#   files without forcing import churn everywhere else in the repository
#
# Internal layout:
# - `runtime.py` assembles loggers, profilers, output paths, and text logging
#   helpers for one run
# - `logging_utils.py` holds smaller metric / hparameter helpers that are
#   shared by orchestration code and callbacks
# - `tensors.py` centralizes nested tensor and metadata normalization helpers
# - `callbacks.py` is the stable public callback assembly surface
# - `debug_callbacks.py`, `system_callbacks.py`, `parameter_callbacks.py`, and
#   `prediction_callbacks.py` hold the concrete callback implementations
# - `reporting.py` now owns both:
#   1. the shared post-run reporting layer that packages prediction/evaluation
#      outputs into a canonical in-memory report surface, and
#   2. the concrete export / visualization sinks that serialize or render that
#      report into CSV and lightweight HTML artifacts
#
# Responsibility boundary:
# - expose the repository's public observability surface in one place
# - let orchestration code import reporting/runtime/callback helpers without
#   needing to care about the package's internal file split
# - keep the rest of the codebase insulated from internal observability
#   refactors as long as the package-level contract remains stable
#
# What does *not* live here:
# - the implementation details of callbacks, logging, profiling, or reporting
# - training orchestration
# - runtime environment policy
# - model-quality metric computation itself
#
# Architectural note:
# the evaluation package remains the canonical home for metric computation and
# grouped evaluation truth. The reporting layer in `reporting.py` consumes that
# structured evaluation output and packages it for downstream sinks. This file
# intentionally re-exports both layers' public observability-facing entrypoints
# without pretending they are the same concern.
#
# The goal is not to hide the split; it is to keep the rest of the repository
# easy to read while preserving explicit responsibility boundaries inside the
# implementation modules.


# ============================================================================
# Public Observability Surface
# ============================================================================
# Re-export the commonly used runtime helpers, callbacks, logging helpers, and
# post-run reporting tools so the rest of the repository can continue using
# short package-level imports.

from observability.callbacks import (
    ActivationStatsCallback,
    BatchAuditCallback,
    GradientStatsCallback,
    ModelTensorBoardCallback,
    ParameterHistogramCallback,
    ParameterScalarTelemetryCallback,
    PredictionFigureCallback,
    PredictionSanityCallback,
    SystemTelemetryCallback,
    build_observability_callbacks,
)
from observability.logging_utils import (
    log_hyperparameters,
    log_metrics_to_loggers,
)
from observability.reporting import (
    SharedReport,
    build_shared_report,
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

# `__all__` is the stable package-level import contract for callers like
# `train.py`, `main.py`, tests, and notebook workflows.
#
# Important compatibility note:
# we keep the long-standing exports such as `export_prediction_table` and
# `generate_plotly_reports` so existing call sites remain valid, while also
# surfacing the newer shared-reporting primitives introduced for the Phase 1
# reporting architecture refactor.
__all__ = [
    "ActivationStatsCallback",
    "BatchAuditCallback",
    "GradientStatsCallback",
    "ModelTensorBoardCallback",
    "ObservabilityArtifacts",
    "ParameterHistogramCallback",
    "ParameterScalarTelemetryCallback",
    "PredictionFigureCallback",
    "PredictionSanityCallback",
    "SharedReport",
    "SystemTelemetryCallback",
    "build_lightning_logger",
    "build_observability_callbacks",
    "build_profiler",
    "build_shared_report",
    "export_prediction_table",
    "generate_plotly_reports",
    "log_hyperparameters",
    "log_metrics_to_loggers",
    "setup_observability",
    "setup_text_logger",
]
