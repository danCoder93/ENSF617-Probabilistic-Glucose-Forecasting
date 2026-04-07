from __future__ import annotations

# AI-assisted maintenance note:
# These callbacks focus on lightweight debugging and numerical-health signals.
#
# Why group them together:
# - they all emit compact diagnostics rather than user-facing reports
# - they share the same "sample, summarize, log" pattern
# - keeping the debug-oriented callbacks side-by-side makes it easier to tune
#   how noisy or expensive deeper instrumentation should be

import json
import logging
from typing import Any

import torch
from pytorch_lightning.callbacks import Callback

from config import ObservabilityConfig
from observability.logging_utils import _log_text_to_loggers, log_metrics_to_loggers
from observability.tensors import _flatten_tensor_output, _summarize_batch, _tensor_stats


class BatchAuditCallback(Callback):
    """
    Log a small number of representative batch summaries for debugging.

    Purpose:
    capture the structure of a few representative batches.

    Context:
    the callback records tensor shapes, dtypes, devices, and lightweight value
    statistics for a capped number of train/validation/test batches. This
    makes data-contract issues visible early without flooding the logs with one
    entry per batch for an entire run.
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        text_logger: logging.Logger | None = None,
    ) -> None:
        """Store batch-audit policy, optional text logger, and per-stage audit counters."""
        self.config = config
        self.text_logger = text_logger
        self._seen_counts = {"train": 0, "val": 0, "test": 0}

    def _maybe_log_batch(self, trainer: Any, stage: str, batch: Any) -> None:
        """Emit one summarized batch snapshot when the stage-specific audit cap allows it."""
        # Audit only a small number of batches per stage to keep the text logs
        # and TensorBoard text pane useful rather than overwhelming.
        if not self.config.enable_batch_audit:
            return
        if self._seen_counts[stage] >= self.config.batch_audit_limit:
            return

        summary = json.dumps(_summarize_batch(batch), indent=2)
        if self.text_logger is not None:
            self.text_logger.info("%s batch audit\n%s", stage, summary)
        _log_text_to_loggers(trainer, f"batch_audit/{stage}", summary)
        self._seen_counts[stage] += 1

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Training hook that records an early batch summary when auditing is enabled."""
        del pl_module, batch_idx
        self._maybe_log_batch(trainer, "train", batch)

    def on_validation_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Validation hook that records an early batch summary when auditing is enabled."""
        del pl_module, batch_idx, dataloader_idx
        self._maybe_log_batch(trainer, "val", batch)

    def on_test_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test hook that records an early batch summary when auditing is enabled."""
        del pl_module, batch_idx, dataloader_idx
        self._maybe_log_batch(trainer, "test", batch)


class GradientStatsCallback(Callback):
    """
    Emit sampled gradient and parameter health summaries during training.

    Purpose:
    track whether gradients and parameters look numerically healthy during
    training.

    Context:
    the callback focuses on compact diagnostics such as total norm, max
    absolute value, and non-finite gradient counts so TensorBoard can expose
    numerical-instability patterns without the cost of logging full tensors on
    every step.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        """Store the debug-sampling policy used for gradient and parameter-health logging."""
        self.config = config

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        """Summarize gradient and parameter health immediately after backpropagation."""
        # This hook runs after gradients exist but before the next optimizer
        # step. That makes it the right place to summarize gradient health
        # without having to modify model code.
        if not self.config.enable_gradient_stats:
            return
        if trainer.sanity_checking:
            return
        if trainer.global_step % self.config.debug_every_n_steps != 0:
            return

        total_norm_sq = 0.0
        max_abs = 0.0
        grad_parameter_count = 0
        nonfinite_grad_parameters = 0
        parameter_norm_sq = 0.0
        parameter_max_abs = 0.0

        for parameter in pl_module.parameters():
            detached_parameter = parameter.detach()
            parameter_norm = float(torch.norm(detached_parameter).item())
            parameter_norm_sq += parameter_norm * parameter_norm
            parameter_max_abs = max(
                parameter_max_abs,
                float(torch.max(torch.abs(detached_parameter)).item()),
            )

            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            grad_parameter_count += 1
            if not torch.isfinite(grad).all():
                nonfinite_grad_parameters += 1
            param_norm = float(torch.norm(grad).item())
            total_norm_sq += param_norm * param_norm
            max_abs = max(max_abs, float(torch.max(torch.abs(grad)).item()))

        metrics = {
            "debug/grad_total_norm": total_norm_sq ** 0.5,
            "debug/grad_max_abs": max_abs,
            "debug/nonfinite_grad_parameters": float(nonfinite_grad_parameters),
            "debug/grad_parameter_count": float(grad_parameter_count),
            "debug/parameter_total_norm": parameter_norm_sq ** 0.5,
            "debug/parameter_max_abs": parameter_max_abs,
        }
        log_metrics_to_loggers(trainer, metrics, step=trainer.global_step)


class ActivationStatsCallback(Callback):
    """
    Sample forward-activation statistics from the major fused-model blocks.

    Purpose:
    sample forward activations from the major fused-model blocks.

    Context:
    the callback attaches lightweight forward hooks to the main architectural
    components and periodically records summary statistics for their outputs.
    It is intended for deeper debugging, so it remains off by default in the
    baseline observability mode.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        """Initialize activation-stat policy plus the hook/metric state used during fit."""
        self.config = config
        self._handles: list[Any] = []
        self._pending_metrics: dict[str, float] = {}

    def _register_hook(self, pl_module: Any, module_name: str) -> None:
        """Attach one forward hook to a named high-level module if that module exists."""
        # We register on named high-level modules rather than every submodule
        # to keep the activation output readable and tied to the architecture
        # the user actually thinks about.
        module = getattr(pl_module, module_name, None)
        if module is None:
            return

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            # The hook only stages metrics into `_pending_metrics`; we flush
            # them at batch end so logging happens in one place and uses the
            # trainer's current global step consistently.
            trainer = getattr(pl_module, "_trainer", None)
            if trainer is None or trainer.sanity_checking or not pl_module.training:
                return
            if trainer.global_step % self.config.debug_every_n_steps != 0:
                return

            tensor = _flatten_tensor_output(output)
            if tensor is None:
                return

            stats = _tensor_stats(tensor)
            for stat_name, value in stats.items():
                self._pending_metrics[f"activation/{module_name}_{stat_name}"] = value
            # We overwrite the staged metrics for a module each time its hook
            # fires within the same batch. For these high-level modules that is
            # acceptable because the goal is one representative activation
            # summary per module per logged step, not a full call-by-call
            # trace.

        self._handles.append(module.register_forward_hook(hook))

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Register activation hooks on the major fused-model blocks at fit start."""
        del trainer
        if not self.config.enable_activation_stats:
            return
        for module_name in ("tcn3", "tcn5", "tcn7", "tft", "grn", "fcn"):
            self._register_hook(pl_module, module_name)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Flush any staged activation metrics once the training batch has finished."""
        del pl_module, outputs, batch, batch_idx
        if not self._pending_metrics:
            return
        # Flushing at batch end gives all hooked modules one shared global step
        # and avoids interleaving many tiny logger writes during the forward
        # pass itself.
        log_metrics_to_loggers(trainer, self._pending_metrics, step=trainer.global_step)
        self._pending_metrics = {}

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Remove any registered activation hooks so later runs do not retain stale instrumentation."""
        del trainer, pl_module
        for handle in self._handles:
            handle.remove()
        self._handles = []
