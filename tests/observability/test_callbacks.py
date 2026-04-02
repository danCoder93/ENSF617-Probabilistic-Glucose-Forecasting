from __future__ import annotations

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
    assert "debug/grad_total_norm" in all_metric_keys
    assert "debug/parameter_total_norm" in all_metric_keys
    assert "parameter_scalars/weight/mean" in all_metric_keys
    assert "parameter_scalars/weight/grad_norm" in all_metric_keys


def test_activation_stats_callback_flushes_module_metrics() -> None:
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
    assert "activation/tcn3_mean" in all_metric_keys
    assert "activation/tft_mean" in all_metric_keys
    assert callback._handles == []


def test_parameter_histogram_callback_logs_parameter_and_gradient_histograms() -> None:
    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)
    trainer.global_step = 12
    module = TinyModule()
    module.weight.grad = torch.tensor([[0.5, -0.25]], dtype=torch.float32)

    ParameterHistogramCallback(
        ObservabilityConfig(histogram_every_n_epochs=1)
    ).on_train_epoch_end(trainer, module)

    tags = [tag for tag, _values, _step in logger.experiment.histogram_events]
    assert "parameters/weight" in tags
    assert "gradients/weight" in tags


def test_system_telemetry_callback_logs_metrics_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert "telemetry/cpu_percent" in telemetry_text
    assert "12.5" in telemetry_text
    assert any("telemetry " in message for message in text_logger.messages)
    assert any(
        "telemetry/cpu_percent" in metrics
        for metrics, _step in logger.metric_events
    )


def test_model_tensorboard_callback_logs_model_text_and_graph() -> None:
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

    assert any(
        tag == "model/architecture"
        for tag, _text, _step in logger.experiment.text_events
    )
    assert len(logger.experiment.graph_events) == 1


def test_prediction_figure_callback_logs_one_validation_figure_per_epoch() -> None:
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
