from __future__ import annotations

# AI-assisted maintenance note:
# This module is now the public callback assembly facade for observability.
#
# Why keep this file:
# - callers and tests already import callback classes from
#   `observability.callbacks`
# - the concrete callback implementations now live in smaller responsibility-
#   focused modules, but the public import surface stays stable here
# - keeping assembly centralized still makes it easy to answer "which
#   observability callbacks are active for this run?"

import logging

from pytorch_lightning.callbacks import (
    Callback,
    DeviceStatsMonitor,
    LearningRateMonitor,
    RichProgressBar,
)

from config import ObservabilityConfig
from observability.debug_callbacks import (
    ActivationStatsCallback,
    BatchAuditCallback,
    GradientStatsCallback,
)
from observability.parameter_callbacks import (
    ParameterHistogramCallback,
    ParameterScalarTelemetryCallback,
)
from observability.prediction_callbacks import PredictionFigureCallback
from observability.system_callbacks import (
    ModelTensorBoardCallback,
    SystemTelemetryCallback,
)
from observability.utils import _has_module


# ============================================================================
# Callback Assembly
# ============================================================================
# This is the single place where the repository turns `ObservabilityConfig`
# flags into an actual Lightning callback list. Keeping this centralized makes
# it much easier to answer questions like "which observability features are
# active for this run?" without searching through trainer construction code.
def build_observability_callbacks(
    config: ObservabilityConfig,
    *,
    text_logger: logging.Logger | None = None,
) -> list[Callback]:
    """
    Translate `ObservabilityConfig` flags into a Lightning callback list.

    Context:
    this is the single assembly point for the repo's custom and built-in
    observability callbacks.
    """

    # Keep callback assembly centralized so trainer construction stays readable
    # and all observability feature flags live behind one consistent policy.
    callbacks: list[Callback] = []
    # Ordering note:
    # - model-visualization and built-in Lightning telemetry callbacks come
    #   first because they describe the overall run and model structure
    # - deeper debug callbacks come after that
    # - figure/report-style callbacks come later because they are more
    #   presentation-oriented than foundational
    #
    # The exact order is not mission-critical for every callback, but keeping
    # it intentional makes the resulting callback stack easier to reason
    # about.

    if config.enable_model_graph or config.enable_model_text or config.enable_torchview:
        callbacks.append(ModelTensorBoardCallback(config, text_logger=text_logger))
    if config.enable_learning_rate_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if config.enable_device_stats:
        callbacks.append(DeviceStatsMonitor())
    if config.enable_rich_progress_bar and _has_module("rich"):
        callbacks.append(RichProgressBar())
    if config.enable_batch_audit:
        callbacks.append(BatchAuditCallback(config, text_logger=text_logger))
    if config.enable_gradient_stats:
        callbacks.append(GradientStatsCallback(config))
    if config.enable_system_telemetry:
        callbacks.append(SystemTelemetryCallback(config, text_logger=text_logger))
    if config.enable_activation_stats:
        callbacks.append(ActivationStatsCallback(config))
    if config.enable_parameter_scalars:
        callbacks.append(ParameterScalarTelemetryCallback(config))
    if config.enable_parameter_histograms:
        callbacks.append(ParameterHistogramCallback(config))
    if config.enable_prediction_figures:
        callbacks.append(PredictionFigureCallback(config))

    return callbacks
