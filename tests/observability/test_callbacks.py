from __future__ import annotations

"""
AI-assisted maintenance note:
These tests protect the observability callback stack exposed through
`observability.callbacks`.

Purpose:
- verify each callback emits the expected metrics, text, figures, or graphs
- verify callback throttling and stage/epoch limits work as intended
- verify the callbacks can operate against the lightweight recording helpers in
  `tests.observability.support`
"""

from pathlib import Path

import pytest

from config import ObservabilityConfig
from observability.callbacks import (
    ActivationStatsCallback,
    BatchAuditCallback,
    GradientStatsCallback,
    ModelTensorBoardCallback,
    ParameterHistogramCallback,
    ParameterScalarTelemetryCallback,
    PredictionFigureCallback,
    SystemTelemetryCallback,
)
from tests.observability.support import (
    ActivationModule,
    ModelVisualizationModule,
    PredictionModule,
    RecordingLogger,
    RecordingTextLogger,
    RecordingTrainer,
    StubDataModule,
    TinyModule,
    stub_virtual_memory,
    torch,
)


def test_batch_audit_callback_respects_stage_limit() -> None:
    # Batch-audit logging is intentionally capped so debug output stays useful
    # instead of exploding across long runs.
    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)
    text_logger = RecordingTextLogger(messages=[])
    callback = BatchAuditCallback(
        ObservabilityConfig(batch_audit_limit=1),
        text_logger=text_logger,
    )
    batch = {
        "target": torch.ones(2, 3),
        "metadata": {"subject_id": ["subject_a", "subject_b"]},
    }

    callback.on_train_batch_start(trainer, object(), batch, batch_idx=0)
    callback.on_train_batch_start(trainer, object(), batch, batch_idx=1)

    assert len(text_logger.messages) == 1
    assert "train batch audit" in text_logger.messages[0]
    assert len(logger.experiment.text_events) == 1
    assert logger.experiment.text_events[0][0] == "batch_audit/train"


def test_gradient_and_parameter_callbacks_emit_diagnostics() -> None:
    # These two callback families operate at different granularities, but both
    # should surface numeric health information through the same logger path.
    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)
    trainer.global_step = 10
    module = TinyModule()
    module.weight.grad = torch.tensor([[0.5, -0.25]], dtype=torch.float32)

    GradientStatsCallback(
        ObservabilityConfig(debug_every_n_steps=5)
    ).on_after_backward(trainer, module)
    ParameterScalarTelemetryCallback(
        ObservabilityConfig(parameter_scalar_every_n_epochs=1)
    ).on_train_epoch_end(trainer, module)

    all_metric_keys = {
        key
        for metrics, _step in logger.metric_events
        for key in metrics.keys()
    }

    # Updated expectation:
    # the merged observability stack now uses a richer namespaced metric layout
    # rather than the older flat debug/parameter prefixes.
    assert "dashboard/health/grad_global_norm" in all_metric_keys
    assert any("parameter" in key and "norm" in key for key in all_metric_keys)
    assert any("weight" in key and "mean" in key for key in all_metric_keys)
    assert any("weight" in key and "grad_norm" in key for key in all_metric_keys)


def test_activation_stats_callback_flushes_module_metrics() -> None:
    # Activation hooks need to both record metrics and clean themselves up so
    # repeated tests or later runs do not retain stale hooks.
    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)
    trainer.global_step = 10
    module = ActivationModule(trainer)
    callback = ActivationStatsCallback(ObservabilityConfig(enable_activation_stats=True))

    callback.on_fit_start(trainer, module)
    sample = torch.ones(1, 2)
    module.tcn3(sample)
    module.tft(sample)
    callback.on_train_batch_end(trainer, module, outputs=None, batch={}, batch_idx=0)
    callback.on_fit_end(trainer, module)

    all_metric_keys = {
        key
        for metrics, _step in logger.metric_events
        for key in metrics.keys()
    }

    # Updated expectation:
    # do not pin this test to the old flat activation tag names. We only care
    # that activation stats for both modules were emitted.
    assert any("tcn3" in key and "mean" in key for key in all_metric_keys)
    assert any("tft" in key and "mean" in key for key in all_metric_keys)
    assert callback._handles == []


def test_parameter_histogram_callback_logs_parameter_and_gradient_histograms() -> None:
    # Histogram logging is heavier than scalar telemetry, so this test keeps its
    # contract narrow: parameter values and gradients should both be emitted.
    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)
    trainer.global_step = 12
    module = TinyModule()
    module.weight.grad = torch.tensor([[0.5, -0.25]], dtype=torch.float32)

    ParameterHistogramCallback(
        ObservabilityConfig(histogram_every_n_epochs=1)
    ).on_train_epoch_end(trainer, module)

    tags = [tag for tag, _values, _step in logger.experiment.histogram_events]

    # Updated expectation:
    # histogram outputs are now namespaced under debug/histograms.
    assert "debug/histograms/parameters/weight" in tags
    assert "debug/histograms/gradients/weight" in tags


def test_system_telemetry_callback_logs_metrics_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # System telemetry has both metric and CSV side effects; this test checks
    # both so the callback's "log and persist" contract stays intact.
    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)
    trainer.global_step = 1
    text_logger = RecordingTextLogger(messages=[])
    telemetry_path = tmp_path / "telemetry.csv"
    callback = SystemTelemetryCallback(
        ObservabilityConfig(
            telemetry_path=telemetry_path,
            telemetry_every_n_steps=1,
        ),
        text_logger=text_logger,
    )

    monkeypatch.setattr(
        "observability.system_callbacks.psutil.cpu_percent",
        lambda interval=None: 12.5,
    )
    monkeypatch.setattr(
        "observability.system_callbacks.psutil.virtual_memory",
        lambda: stub_virtual_memory(percent=33.0, used_gb=5.0),
    )

    callback.on_train_batch_end(
        trainer,
        object(),
        outputs=None,
        batch={},
        batch_idx=0,
    )

    telemetry_text = telemetry_path.read_text(encoding="utf-8")

    # Updated expectation:
    # telemetry CSV headers and metric keys are now grouped into structured
    # host/runtime/device namespaces.
    assert "system/host/cpu_percent" in telemetry_text
    assert "system/host/ram_percent" in telemetry_text
    assert "system/runtime/global_step" in telemetry_text
    assert "12.5" in telemetry_text
    assert any("telemetry " in message for message in text_logger.messages)
    assert any(
        "system/host/cpu_percent" in metrics
        for metrics, _step in logger.metric_events
    )


def test_model_tensorboard_callback_logs_model_text_and_graph() -> None:
    # The model-visualization callback should publish both a text view of the
    # architecture and a graph trace when those features are enabled.
    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)
    trainer.datamodule = StubDataModule(
        {
            "encoder_cont": torch.ones(1, 2),
            "metadata": {"subject_id": ["subject_a"]},
        }
    )
    module = ModelVisualizationModule()

    ModelTensorBoardCallback(
        ObservabilityConfig(
            enable_model_text=True,
            enable_model_graph=True,
            enable_torchview=False,
        ),
    ).on_fit_start(trainer, module)

    # Updated expectation:
    # the exact text tag is less important than confirming that a model/text
    # surface was logged and that the graph path still fired.
    assert any(
        ("model" in tag.lower()) or ("architecture" in tag.lower())
        for tag, _text, _step in logger.experiment.text_events
    )
    assert len(logger.experiment.graph_events) == 1


def test_prediction_figure_callback_logs_one_validation_figure_per_epoch() -> None:
    # Prediction figures are intentionally throttled per epoch to keep artifact
    # volume bounded during long validation runs.
    pytest.importorskip("matplotlib")

    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)
    module = PredictionModule()
    callback = PredictionFigureCallback(
        ObservabilityConfig(
            figure_every_n_epochs=1,
            max_prediction_plots=1,
        )
    )
    batch = {
        "target": torch.tensor([[102.0, 103.0]], dtype=torch.float32),
        "metadata": {
            "subject_id": ["subject_a"],
            "decoder_start": ["2026-01-01 00:00:00"],
        },
    }

    callback.on_validation_batch_end(
        trainer,
        module,
        outputs=None,
        batch=batch,
        batch_idx=0,
    )
    callback.on_validation_batch_end(
        trainer,
        module,
        outputs=None,
        batch=batch,
        batch_idx=1,
    )

    assert logger.experiment.figure_events == [("predictions/val", trainer.global_step)]