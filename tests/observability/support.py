from __future__ import annotations

from dataclasses import dataclass
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


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


def stub_virtual_memory(percent: float, used_gb: float) -> SimpleNamespace:
    return SimpleNamespace(percent=percent, used=used_gb * (1024.0 ** 3))
