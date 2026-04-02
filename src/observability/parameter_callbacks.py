from __future__ import annotations

# AI-assisted maintenance note:
# These callbacks focus specifically on parameter distributions and magnitudes.
#
# Why keep them together:
# - they operate over the same named-parameter iteration
# - one callback emits cheap scalar summaries while the other emits heavier
#   histogram detail
# - reading them together makes the scalar-versus-histogram tradeoff explicit

from typing import Any

import torch
from pytorch_lightning.callbacks import Callback

from config import ObservabilityConfig
from observability.logging_utils import _tensorboard_experiments, log_metrics_to_loggers


class ParameterScalarTelemetryCallback(Callback):
    """
    Log compact per-parameter scalar summaries at epoch boundaries.

    Purpose:
    emit compact scalar summaries for every parameter tensor.

    Context:
    these scalars complement TensorBoard histograms by making it quick to spot
    broad trends in parameter magnitude and gradient magnitude without opening
    distribution plots for every tensor first.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        self.config = config

    @staticmethod
    def _tag_name(parameter_name: str, stat_name: str) -> str:
        return f"parameter_scalars/{parameter_name.replace('.', '/')}/{stat_name}"

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        # Epoch-end logging keeps these summaries stable and affordable.
        # Logging per parameter on every step would be too heavy for normal
        # runs.
        if not self.config.enable_parameter_scalars:
            return
        if (trainer.current_epoch + 1) % self.config.parameter_scalar_every_n_epochs != 0:
            return

        metrics: dict[str, float] = {}
        for name, parameter in pl_module.named_parameters():
            values = parameter.detach().float()
            metrics[self._tag_name(name, "mean")] = float(values.mean().item())
            metrics[self._tag_name(name, "std")] = float(
                values.std(unbiased=False).item()
            )
            metrics[self._tag_name(name, "norm")] = float(torch.norm(values).item())
            metrics[self._tag_name(name, "max_abs")] = float(
                torch.max(torch.abs(values)).item()
            )
            if parameter.grad is not None:
                grad = parameter.grad.detach().float()
                metrics[self._tag_name(name, "grad_norm")] = float(
                    torch.norm(grad).item()
                )
                metrics[self._tag_name(name, "grad_max_abs")] = float(
                    torch.max(torch.abs(grad)).item()
                )
        # The scalar view is intentionally compact:
        # - it is cheap enough to log for every parameter tensor at epoch
        #   boundaries
        # - it gives a quick scan surface in TensorBoard
        # - users can then open the heavier histogram views only when a scalar
        #   trend suggests something suspicious

        if metrics:
            log_metrics_to_loggers(trainer, metrics, step=trainer.global_step)


class ParameterHistogramCallback(Callback):
    """
    Log full parameter and gradient distributions to TensorBoard histograms.

    Purpose:
    expose richer parameter and gradient distribution detail than scalar
    summaries alone can provide.

    Context:
    histogram logging provides richer visibility than scalar summaries, but it
    is correspondingly heavier. For that reason this callback emits histograms
    at configurable epoch intervals rather than every optimization step.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        self.config = config

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        # Histograms are intentionally recorded at epoch granularity because
        # the payload size is much larger than scalar metrics.
        if not self.config.enable_parameter_histograms:
            return
        if (trainer.current_epoch + 1) % self.config.histogram_every_n_epochs != 0:
            return

        for experiment in _tensorboard_experiments(trainer):
            add_histogram = getattr(experiment, "add_histogram", None)
            if not callable(add_histogram):
                continue
            for name, parameter in pl_module.named_parameters():
                # Histogram logging is the expensive "deep dive" counterpart to
                # parameter scalar telemetry. Scalars tell you that a tensor may
                # be drifting; histograms show the full distribution shape.
                add_histogram(
                    f"parameters/{name}",
                    parameter.detach().cpu(),
                    global_step=trainer.global_step,
                )
                if parameter.grad is not None:
                    add_histogram(
                        f"gradients/{name}",
                        parameter.grad.detach().cpu(),
                        global_step=trainer.global_step,
                    )
