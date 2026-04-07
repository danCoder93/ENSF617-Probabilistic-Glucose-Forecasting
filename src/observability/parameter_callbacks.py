from __future__ import annotations

# AI-assisted maintenance note:
# These callbacks focus specifically on parameter distributions and magnitudes.
#
# Why keep them together:
# - they operate over the same named-parameter iteration
# - one callback emits cheap scalar summaries while the other emits heavier
#   histogram detail
# - reading them together makes the scalar-versus-histogram tradeoff explicit
#
# Dashboard-first enhancement note:
# Earlier revisions of this module already exposed useful parameter telemetry,
# but the resulting TensorBoard surface could still become cluttered because
# every parameter tensor produced several scalar series at the same hierarchy
# level. The current version preserves that visibility while making the
# drill-down structure clearer:
# - scalar parameter telemetry is grouped under `debug/parameters/...`
# - histogram telemetry is grouped under `debug/histograms/...`
#
# Important compatibility rule:
# These callbacks still log the same *kind* of underlying parameter evidence.
# The enhancement here is about clearer namespace organization and better
# comments, not about changing model behavior or removing deep parameter
# inspection.

from typing import Any

import torch
from pytorch_lightning.callbacks import Callback

from config import ObservabilityConfig
from observability.logging_utils import _tensorboard_experiments, log_metrics_to_loggers


def _parameter_scalar_tag(parameter_name: str, stat_name: str) -> str:
    """Return the canonical debug scalar namespace for one parameter statistic.

    Context:
        Parameter scalar telemetry is intended to be a drill-down surface rather
        than a dashboard front-door surface. Grouping these tags under a clear
        debug namespace makes TensorBoard browsing more readable while keeping
        the per-parameter detail intact.
    """
    return f"debug/parameters/{parameter_name.replace('.', '/')}/{stat_name}"


def _parameter_histogram_tag(parameter_name: str) -> str:
    """Return the canonical histogram namespace for one parameter tensor.

    Context:
        Histogram logging is the heavier companion to scalar telemetry. It is
        useful when investigating distribution shape, but it should still live
        under a clearly collapsible debug-oriented namespace.
    """
    return f"debug/histograms/parameters/{parameter_name}"


def _gradient_histogram_tag(parameter_name: str) -> str:
    """Return the canonical histogram namespace for one parameter-gradient tensor."""
    return f"debug/histograms/gradients/{parameter_name}"


class ParameterScalarTelemetryCallback(Callback):
    """
    Log compact per-parameter scalar summaries at epoch boundaries.

    Purpose:
    emit compact scalar summaries for every parameter tensor.

    Context:
    these scalars complement TensorBoard histograms by making it quick to spot
    broad trends in parameter magnitude and gradient magnitude without opening
    distribution plots for every tensor first.

    Important presentation note:
    this callback intentionally logs into the debug namespace rather than the
    dashboard namespace. The scalar payload is still valuable, but it is too
    dense and parameter-granular to serve as a front-door dashboard surface.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        """Store the observability policy that governs scalar parameter logging frequency."""
        self.config = config

    @staticmethod
    def _tag_name(parameter_name: str, stat_name: str) -> str:
        """Build the TensorBoard-friendly scalar tag used for one parameter statistic.

        Why this helper exists:
            Centralizing tag construction keeps the namespace policy easy to
            update and ensures that all scalar parameter series follow the same
            drill-down layout.
        """
        return _parameter_scalar_tag(parameter_name, stat_name)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Log compact parameter and gradient summary scalars at eligible epoch boundaries."""
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

            # Parameter-value summaries provide a quick numerical-health view of
            # the actual learned tensor:
            # - mean and std show location/spread
            # - norm shows overall magnitude
            # - max_abs shows whether extreme values are emerging
            metrics[self._tag_name(name, "mean")] = float(values.mean().item())
            metrics[self._tag_name(name, "std")] = float(
                values.std(unbiased=False).item()
            )
            metrics[self._tag_name(name, "norm")] = float(torch.norm(values).item())
            metrics[self._tag_name(name, "max_abs")] = float(
                torch.max(torch.abs(values)).item()
            )

            # Gradient summaries complement parameter-value summaries by
            # answering whether the tensor is still receiving meaningful update
            # signal at the current logging epoch.
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
        #
        # Dashboard policy note:
        # We intentionally do not promote these series into `dashboard/*`.
        # Even when the underlying values are useful, the per-parameter density
        # is too high for a front-door dashboard experience.
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

    Important presentation note:
    histogram logging is explicitly a deep-dive surface. It is therefore kept in
    a debug-oriented hierarchy rather than mixed with dashboard-first report
    content.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        """Store the observability policy that governs histogram logging frequency."""
        self.config = config

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Log parameter and gradient histograms to TensorBoard at eligible epoch boundaries."""
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
                    _parameter_histogram_tag(name),
                    parameter.detach().cpu(),
                    global_step=trainer.global_step,
                )

                # Gradient histograms are logged separately from parameter
                # histograms so later TensorBoard inspection can distinguish
                # "what values does this tensor currently hold?" from
                # "what update distribution is currently reaching it?"
                if parameter.grad is not None:
                    add_histogram(
                        _gradient_histogram_tag(name),
                        parameter.grad.detach().cpu(),
                        global_step=trainer.global_step,
                    )
