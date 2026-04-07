from __future__ import annotations

"""
AI-assisted maintenance note:
This helper module supports the observability test suite with intentionally
small recording fakes and tiny torch modules.

Purpose:
- capture what callbacks and logger helpers emit without needing a real
  TensorBoard server or a full Lightning training loop
- give the observability tests explicit stand-ins for the narrow runtime
  surfaces they care about

Context:
these helpers are deliberately incomplete. They only implement the attributes
and methods exercised by the tests so failures point at observability behavior
rather than at unrelated framework setup.
"""

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
    """
    Minimal TensorBoard-like experiment object that records emitted events.

    Context:
    several observability callbacks talk to `logger.experiment`. The tests only
    need to know which calls happened, not to render real dashboards, so this
    fake stores the emitted payloads in plain Python lists.
    """

    def __init__(self) -> None:
        """Initialize empty event buffers for each experiment API the tests inspect."""
        self.text_events: list[tuple[str, str, int]] = []
        self.graph_events: list[tuple[Any, Any]] = []
        self.histogram_events: list[tuple[str, Any, int]] = []
        self.figure_events: list[tuple[str, int]] = []

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        """Accept scalar logging calls without recording them when a test does not need them."""
        del tag, value, global_step

    def add_text(self, tag: str, text: str, global_step: int) -> None:
        """Record emitted text events for assertions about human-readable diagnostics."""
        self.text_events.append((tag, text, global_step))

    def add_graph(self, module: Any, input_to_model: Any) -> None:
        """Record model-graph requests so tests can verify graph logging happened."""
        self.graph_events.append((module, input_to_model))

    def add_histogram(self, tag: str, values: Any, global_step: int) -> None:
        """Record histogram emissions for parameter and gradient callback tests."""
        self.histogram_events.append((tag, values, global_step))

    def add_figure(self, tag: str, figure: Any, global_step: int) -> None:
        """Record figure logging requests while discarding the heavy figure object itself."""
        del figure
        self.figure_events.append((tag, global_step))


class RecordingLogger:
    """
    Minimal Lightning-style logger that records metrics and hyperparameters.

    Context:
    observability helpers often need both a high-level logger interface and an
    attached `experiment` object. This fake provides both surfaces in the
    smallest form the tests need.
    """

    def __init__(self) -> None:
        """Create the logger along with a paired recording experiment surface."""
        self.experiment = RecordingExperiment()
        self.metric_events: list[tuple[dict[str, float], int]] = []
        self.hparam_events: list[dict[str, str | int | float | bool]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Record scalar metric payloads exactly as the logger helper emitted them."""
        self.metric_events.append((metrics, step))

    def log_hyperparams(self, params: dict[str, str | int | float | bool]) -> None:
        """Record hyperparameter payloads for later assertions."""
        self.hparam_events.append(params)


@dataclass
class RecordingTextLogger(logging.Logger):
    """
    Tiny text logger that stores formatted `.info(...)` messages in memory.

    Context:
    the observability package writes several human-readable diagnostics to a
    standard logger. Capturing the rendered messages keeps those tests simple
    and avoids writing temporary log files.
    """

    messages: list[str]

    def __post_init__(self) -> None:
        """Initialize the underlying logging base class with a stable test name."""
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
        """Render the log record exactly once and append the final string to `messages`."""
        del exc_info, stack_info, stacklevel, extra
        rendered = str(msg) % args if args else str(msg)
        self.messages.append(rendered)


class RecordingTrainer:
    """
    Lightweight stand-in for the Trainer attributes callback tests inspect.

    Context:
    most callback tests do not need a real epoch loop. They only need the
    logger handles, global-step counters, and a couple of state flags that the
    callback hooks read during execution.
    """

    def __init__(self, logger: RecordingLogger) -> None:
        """Expose the minimal Trainer-like state used by the observability callbacks."""
        self.logger = logger
        self.loggers = [logger]
        self.global_step = 5
        self.current_epoch = 0
        self.sanity_checking = False
        self.datamodule: Any = None


class TinyModule(torch.nn.Module):
    """Single-parameter module used for simple parameter and gradient telemetry checks."""

    def __init__(self) -> None:
        """Create one small parameter tensor so scalar and histogram callbacks have something to inspect."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0]], dtype=torch.float32))


class ActivationModule(torch.nn.Module):
    """
    Tiny module exposing the named submodules expected by activation-stat callbacks.

    Context:
    the activation callback looks for branch names that mirror the real fused
    model. This stub preserves those names without needing to construct the full
    forecasting architecture in every observability unit test.
    """

    def __init__(self, trainer: RecordingTrainer) -> None:
        """Populate the branch names that the activation callback registers hooks against."""
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
    """
    Small forward-pass module for model-graph visualization tests.

    Context:
    the torchview/TensorBoard graph callback only needs a module with a valid
    `forward(...)` signature over one representative batch key.
    """

    def __init__(self) -> None:
        """Provide one linear layer plus the minimal metadata expected by the callback stack."""
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.device = torch.device("cpu")
        self.quantiles = (0.1, 0.5, 0.9)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Project the synthetic encoder tensor so graph-building code can trace a real op."""
        return self.linear(batch["encoder_cont"])


class StubDataModule:
    """
    Minimal DataModule exposing only the training dataloader surface under test.

    Context:
    model-visualization callbacks sample one batch from the training loader to
    build a graph. This stub keeps that behavior focused and deterministic.
    """

    def __init__(self, batch: dict[str, Any]) -> None:
        """Store the one batch that should be returned by the fake training loader."""
        self._batch = batch

    def train_dataloader(self) -> list[dict[str, Any]]:
        """Return the stored batch as a single-item iterable loader stand-in."""
        return [self._batch]


class PredictionModule(torch.nn.Module):
    """
    Tiny probabilistic model stub for prediction-figure callback tests.

    Context:
    the callback needs quantile-shaped outputs and a `_target_tensor(...)`
    helper, but it does not need the real fused model architecture.
    """

    def __init__(self) -> None:
        """Expose the same minimal prediction metadata the callback expects from the real model."""
        super().__init__()
        self.device = torch.device("cpu")
        self.quantiles = (0.1, 0.5, 0.9)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one fixed quantile forecast so figure tests stay deterministic."""
        del batch
        return torch.tensor(
            [[[95.0, 100.0, 105.0], [96.0, 101.0, 106.0]]],
            dtype=torch.float32,
        )

    def _target_tensor(self, batch: dict[str, Tensor]) -> Tensor:
        """Mirror the real model helper by exposing the target tensor expected by plotting code."""
        return batch["target"]


def stub_virtual_memory(percent: float, used_gb: float) -> SimpleNamespace:
    """
    Build a psutil-like virtual-memory object from lightweight numeric inputs.

    Context:
    system-telemetry tests only care about percentage usage and converted byte
    counts, so this helper avoids depending on a real `psutil` payload.
    """
    return SimpleNamespace(percent=percent, used=used_gb * (1024.0 ** 3))
