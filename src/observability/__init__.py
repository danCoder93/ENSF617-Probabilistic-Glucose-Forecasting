from __future__ import annotations

# AI-assisted maintenance note:
# `observability` is the stable package-level facade for the repository's
# runtime-observability stack.
#
# Why this file exists:
# - existing callers already use concise imports like
#   `from observability import setup_observability, BatchAuditCallback`
# - that short public surface remains convenient for `train.py`, `main.py`,
#   tests, notebooks, and one-off debugging scripts
# - this facade lets the internal runtime-observability package evolve into
#   smaller, more focused files without forcing unnecessary import churn
#   everywhere else in the repository
#
# Architectural clarification:
# the repository now keeps three related concerns separate on purpose:
#
# 1. `observability`
#    Live runtime visibility during training/evaluation, including:
#    - logger/profiler setup
#    - callback assembly
#    - system telemetry
#    - model graph/text surfaces
#    - gradient/activation/parameter/prediction-sanity instrumentation
#
# 2. `reporting`
#    Post-run packaging and presentation once predictions already exist,
#    including:
#    - canonical `SharedReport` construction
#    - tabular prediction export
#    - lightweight HTML report sinks
#
# 3. `evaluation`
#    Canonical metric computation and grouped evaluation truth
#
# In other words:
# - evaluation computes model-quality truth
# - reporting packages and renders post-run artifacts from that truth
# - observability handles live runtime visibility while the run is happening
#
# Important boundary for this phase:
# this file intentionally does *not* eagerly import the reporting layer.
# Post-run reporting now belongs to the dedicated `reporting` package and
# is exposed here only through a lazy compatibility bridge.
#
# What does *not* live here:
# - concrete callback implementation details
# - report-building or report-rendering logic
# - training orchestration
# - canonical metric computation
#
# This file exists to define the stable runtime-observability import surface,
# not to mix runtime observability with every post-run artifact concern.

from importlib import import_module
from typing import TYPE_CHECKING, Any

# ============================================================================
# Public Runtime-Observability Surface
# ============================================================================
# Re-export the commonly used runtime helpers, callbacks, and smaller logging
# utilities so the rest of the repository can continue using short
# package-level imports.

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
from observability.runtime import (
    ObservabilityArtifacts,
    build_lightning_logger,
    build_profiler,
    setup_observability,
    setup_text_logger,
)

if TYPE_CHECKING:
    # CHANGE: These imports are for static analysis only.
    # They let Pylance see the compatibility-exported reporting names without
    # eagerly importing `reporting` at runtime and recreating the circular import.
    from reporting import (
        SharedReport,
        build_shared_report,
        export_grouped_tables_from_report,
        export_prediction_table,
        export_prediction_table_from_report,
        generate_plotly_reports,
        log_shared_report_to_tensorboard,
    )

# Compatibility surface:
# the canonical post-run reporting implementation now lives in `reporting`.
# We intentionally resolve these names lazily so importing lightweight
# observability modules such as `observability.tensors` does not force the
# `reporting` package to import while it is still partially initialized.
_REPORTING_EXPORT_NAMES = {
    "SharedReport",
    "build_shared_report",
    "export_grouped_tables_from_report",
    "export_prediction_table",
    "export_prediction_table_from_report",
    "generate_plotly_reports",
    "log_shared_report_to_tensorboard",
}


def __getattr__(name: str) -> Any:
    """Lazily expose reporting-package symbols through `observability`."""
    if name in _REPORTING_EXPORT_NAMES:
        reporting_module = import_module("reporting")
        return getattr(reporting_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# `__all__` is the stable package-level import contract for callers like
# `train.py`, `main.py`, tests, and notebook workflows.
#
# Migration note:
# post-run reporting helpers are still available here for compatibility, but
# their canonical implementation lives in the dedicated `reporting` package.
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
    "SystemTelemetryCallback",
    "build_lightning_logger",
    "build_observability_callbacks",
    "build_profiler",
    "log_hyperparameters",
    "log_metrics_to_loggers",
    "setup_observability",
    "setup_text_logger",
    "SharedReport",
    "build_shared_report",
    "export_grouped_tables_from_report",
    "export_prediction_table",
    "export_prediction_table_from_report",
    "generate_plotly_reports",
    "log_shared_report_to_tensorboard",
]