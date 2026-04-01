from __future__ import annotations

"""
AI-assisted implementation note:
This test module protects the observability package split itself.

The goal is not to re-test every callback behavior in detail. Those behaviors
are already covered more indirectly through the training-wrapper and main
workflow tests when the full project dependencies are available.

Instead, this file focuses on one specific regression risk introduced by the
refactor:
- the old single-file `observability.py` implementation was replaced by a
  package
- callers still rely on imports like `from observability import
  setup_observability, BatchAuditCallback`
- the package facade therefore needs its public re-exports to remain stable
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from observability import (
    ActivationStatsCallback,
    BatchAuditCallback,
    GradientStatsCallback,
    ModelTensorBoardCallback,
    ObservabilityArtifacts,
    ParameterHistogramCallback,
    ParameterScalarTelemetryCallback,
    PredictionFigureCallback,
    SystemTelemetryCallback,
    build_lightning_logger,
    build_observability_callbacks,
    build_profiler,
    export_prediction_table,
    generate_plotly_reports,
    log_hyperparameters,
    log_metrics_to_loggers,
    setup_observability,
    setup_text_logger,
)
from observability.callbacks import build_observability_callbacks as callbacks_build
from observability.logging_utils import log_hyperparameters as logging_log_hparams
from observability.logging_utils import log_metrics_to_loggers as logging_log_metrics
from observability.reporting import export_prediction_table as reporting_export
from observability.reporting import generate_plotly_reports as reporting_reports
from observability.runtime import (
    ObservabilityArtifacts as RuntimeObservabilityArtifacts,
)
from observability.runtime import (
    build_lightning_logger as runtime_build_logger,
)
from observability.runtime import build_profiler as runtime_build_profiler
from observability.runtime import setup_observability as runtime_setup
from observability.runtime import setup_text_logger as runtime_setup_text_logger


def test_observability_package_reexports_runtime_and_reporting_api() -> None:
    assert ObservabilityArtifacts is RuntimeObservabilityArtifacts
    assert setup_observability is runtime_setup
    assert setup_text_logger is runtime_setup_text_logger
    assert build_lightning_logger is runtime_build_logger
    assert build_profiler is runtime_build_profiler
    assert log_metrics_to_loggers is logging_log_metrics
    assert log_hyperparameters is logging_log_hparams
    assert build_observability_callbacks is callbacks_build
    assert export_prediction_table is reporting_export
    assert generate_plotly_reports is reporting_reports


def test_observability_package_reexports_callback_types() -> None:
    callback_types = {
        BatchAuditCallback,
        GradientStatsCallback,
        SystemTelemetryCallback,
        ActivationStatsCallback,
        ModelTensorBoardCallback,
        ParameterScalarTelemetryCallback,
        ParameterHistogramCallback,
        PredictionFigureCallback,
    }

    assert len(callback_types) == 8
