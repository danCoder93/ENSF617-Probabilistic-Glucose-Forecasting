from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

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
from observability.logging_utils import (
    _log_text_to_loggers,
    log_hyperparameters,
    log_metrics_to_loggers,
)
from observability.runtime import setup_observability


class RecordingExperiment:
    def __init__(self) -> None:
        self.text_events: list[tuple[str, str, int]] = []
        self.graph_events: list[tuple[Any, Any]] = []
        self.histogram_events: list[tuple[str, Any, int]] = []
        self.figure_events: list[tuple[str, int]] = []

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        del tag, value, global_step

    def add_text(self, tag: str, text: str, global_step: int) -> None:
        self.text_events.append((tag, text, global_step))

    def add_graph(self, module: Any, input_to_model: Any) -> None:
        self.graph_events.append((module, input_to_model))

    def add_histogram(self, tag: str, values: Any, global_step: int) -> None:
        self.histogram_events.append((tag, values, global_step))

    def add_figure(self, tag: str, figure: Any, global_step: int) -> None:
        del figure
        self.figure_events.append((tag, global_step))


class RecordingLogger:
    def __init__(self) -> None:
        self.experiment = RecordingExperiment()
        self.metric_events: list[tuple[dict[str, float], int]] = []
        self.hparam_events: list[dict[str, str | int | float | bool]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metric_events.append((metrics, step))

    def log_hyperparams(self, params: dict[str, str | int | float | bool]) -> None:
        self.hparam_events.append(params)


@dataclass
class RecordingTextLogger(logging.Logger):
    messages: list[str]

    def __post_init__(self) -> None:
        super().__init__("recording-text-logger")

    def info(
        self,
        msg: object,
        *args: object,
        exc_info: object | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: object | None = None,
    ) -> None:
        del exc_info, stack_info, stacklevel, extra
        rendered = str(msg) % args if args else str(msg)
        self.messages.append(rendered)


class RecordingTrainer:
    def __init__(self, logger: RecordingLogger) -> None:
        self.logger = logger
        self.loggers = [logger]
        self.global_step = 5
        self.current_epoch = 0
        self.sanity_checking = False
        self.datamodule: Any = None


class TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0]], dtype=torch.float32))


class ActivationModule(torch.nn.Module):
    def __init__(self, trainer: RecordingTrainer) -> None:
        super().__init__()
        self._trainer = trainer
        self.tcn3 = torch.nn.Linear(2, 2)
        self.tcn5 = torch.nn.Linear(2, 2)
        self.tcn7 = torch.nn.Linear(2, 2)
        self.tft = torch.nn.Linear(2, 2)
        self.grn = torch.nn.Linear(2, 2)
        self.fcn = torch.nn.Linear(2, 2)
        self.train()


class ModelVisualizationModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.device = torch.device("cpu")
        self.quantiles = (0.1, 0.5, 0.9)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.linear(batch["encoder_cont"])


class StubDataModule:
    def __init__(self, batch: dict[str, Any]) -> None:
        self._batch = batch

    def train_dataloader(self) -> list[dict[str, Any]]:
        return [self._batch]


class PredictionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.quantiles = (0.1, 0.5, 0.9)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        del batch
        return torch.tensor(
            [[[95.0, 100.0, 105.0], [96.0, 101.0, 106.0]]],
            dtype=torch.float32,
        )

    def _target_tensor(self, batch: dict[str, Tensor]) -> Tensor:
        return batch["target"]


def test_logging_helpers_publish_metrics_text_and_hparams() -> None:
    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)

    log_metrics_to_loggers(trainer, {"debug/example": 1.25}, step=7)
    _log_text_to_loggers(trainer, "batch_audit/train", "sample batch")
    log_hyperparameters(
        trainer,
        {
            "config": {
                "epochs": 3,
                "output_dir": Path("artifacts/run"),
            },
            "flags": {"debug": True, "seed": None},
        },
    )

    assert logger.metric_events == [({"debug/example": 1.25}, 7)]
    assert logger.experiment.text_events == [
        ("batch_audit/train", "sample batch", trainer.global_step)
    ]
    assert logger.hparam_events == [
        {
            "config/epochs": 3,
            "config/output_dir": "artifacts/run",
            "flags/debug": True,
            "flags/seed": "None",
        }
    ]


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
        lambda: SimpleNamespace(percent=33.0, used=5 * (1024.0 ** 3)),
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

    assert any(tag == "model/architecture" for tag, _text, _step in logger.experiment.text_events)
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


def test_setup_observability_creates_text_logger_and_profiler(tmp_path: Path) -> None:
    config = ObservabilityConfig(
        enable_tensorboard=False,
        enable_csv_fallback_logger=True,
        enable_profiler=True,
        profiler_type="simple",
        log_dir=tmp_path / "logs",
        text_log_path=tmp_path / "run.log",
        telemetry_path=tmp_path / "telemetry.csv",
        profiler_path=tmp_path / "profiler",
    )

    artifacts = setup_observability(config)
    assert artifacts.logger is not None
    assert artifacts.logger_dir == tmp_path / "logs"
    assert artifacts.text_logger is not None
    assert artifacts.profiler is not None

    artifacts.text_logger.info("runtime ready")
    assert artifacts.text_log_path is not None
    assert "runtime ready" in artifacts.text_log_path.read_text(encoding="utf-8")
