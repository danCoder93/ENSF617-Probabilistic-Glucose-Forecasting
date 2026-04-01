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
# - `callbacks.py` contains the Lightning callbacks plus callback assembly
# - `reporting.py` owns post-run exports and HTML report generation
#
# The goal is not to hide the split; it is to make each part easier to inspect
# without forcing the rest of the repository to care about the new file
# boundaries.

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
