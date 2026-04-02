from __future__ import annotations

# AI-assisted maintenance note:
# This module contains the Lightning callback layer for observability.
#
# Why group the callbacks together:
# - they all express runtime hook policy for the Trainer lifecycle
# - they share logger/tensor helper functions heavily
# - reading them side-by-side makes ordering and sampling strategy easier to
#   understand than if each callback were scattered into its own tiny file
#
# This file therefore aims for a middle ground:
# - smaller than the original catch-all module
# - still cohesive enough that "observability callback behavior" can be read in
#   one place

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import psutil
import torch
from pytorch_lightning.callbacks import (
    Callback,
    DeviceStatsMonitor,
    LearningRateMonitor,
    RichProgressBar,
)

from config import ObservabilityConfig
from observability.logging_utils import (
    _log_text_to_loggers,
    _tensorboard_experiments,
    log_metrics_to_loggers,
)
from observability.tensors import (
    _as_metadata_lists,
    _flatten_tensor_output,
    _move_batch_to_device,
    _summarize_batch,
    _tensor_only_structure,
    _tensor_stats,
)
from observability.utils import _ensure_parent, _has_module

try:
    from torchview import draw_graph
except ImportError:  # pragma: no cover - optional dependency until installed
    draw_graph = None  # type: ignore[assignment]


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
        self.config = config
        self.text_logger = text_logger
        self._seen_counts = {"train": 0, "val": 0, "test": 0}

    def _maybe_log_batch(self, trainer: Any, stage: str, batch: Any) -> None:
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
        self.config = config

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
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


class SystemTelemetryCallback(Callback):
    """
    Record host and device telemetry alongside model metrics.

    Purpose:
    log runtime system telemetry alongside training metrics.

    Context:
    this callback logs CPU, RAM, and GPU memory telemetry to the active
    Lightning logger and mirrors the same metrics to a CSV file for later
    offline inspection. GPU utilization percentage is best-effort and depends
    on optional NVML bindings being available at runtime.
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        text_logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.text_logger = text_logger
        self._telemetry_header_written = False

    def _gpu_metrics(self) -> dict[str, float]:
        # CUDA memory stats are always available when CUDA is active, but
        # utilization percentage depends on optional NVML bindings. We return a
        # consistent metric dictionary either way so downstream logging stays
        # stable.
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_runtime = getattr(torch, "mps", None)
        if (
            mps_backend is not None
            and bool(getattr(mps_backend, "is_available", lambda: False)())
            and mps_runtime is not None
        ):
            allocated = float(
                getattr(mps_runtime, "current_allocated_memory", lambda: 0)()
            )
            reserved = float(
                getattr(mps_runtime, "driver_allocated_memory", lambda: 0)()
            )
            return {
                "telemetry/gpu_memory_allocated_mb": allocated / (1024.0 * 1024.0),
                "telemetry/gpu_memory_reserved_mb": reserved / (1024.0 * 1024.0),
                "telemetry/gpu_utilization_percent": 0.0,
            }

        if not torch.cuda.is_available():
            return {
                "telemetry/gpu_memory_allocated_mb": 0.0,
                "telemetry/gpu_memory_reserved_mb": 0.0,
                "telemetry/gpu_utilization_percent": 0.0,
            }

        metrics = {
            "telemetry/gpu_memory_allocated_mb": (
                torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            ),
            "telemetry/gpu_memory_reserved_mb": (
                torch.cuda.memory_reserved() / (1024.0 * 1024.0)
            ),
            "telemetry/gpu_utilization_percent": 0.0,
        }

        if _has_module("pynvml"):
            try:  # pragma: no cover - hardware/library dependent
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(
                    torch.cuda.current_device()
                )
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["telemetry/gpu_utilization_percent"] = float(utilization.gpu)
            except Exception:
                pass

        return metrics

    def _append_csv_row(self, row: Mapping[str, float]) -> None:
        # The CSV output mirrors what is logged live. This is useful when the
        # user wants to inspect runtime telemetry after the run without opening
        # TensorBoard.
        if self.config.telemetry_path is None:
            return
        # The callback keeps `ObservabilityConfig` as its source of truth, but
        # converts path-like config values into a concrete `Path` right before
        # filesystem access so static typing and runtime behavior stay aligned.
        telemetry_path = Path(self.config.telemetry_path)
        _ensure_parent(telemetry_path)
        with telemetry_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            if not self._telemetry_header_written:
                # The first observed row defines the header order. Keeping the
                # same fieldnames thereafter makes the CSV stable and easy to
                # load as a normal tabular artifact after training.
                writer.writeheader()
                self._telemetry_header_written = True
            writer.writerow(row)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if not self.config.enable_system_telemetry:
            return
        if trainer.sanity_checking:
            return
        if trainer.global_step % self.config.telemetry_every_n_steps != 0:
            return

        memory = psutil.virtual_memory()
        metrics = {
            "telemetry/cpu_percent": float(psutil.cpu_percent(interval=None)),
            "telemetry/ram_percent": float(memory.percent),
            "telemetry/ram_used_gb": float(memory.used / (1024.0 ** 3)),
            "telemetry/global_step": float(trainer.global_step),
            "telemetry/current_epoch": float(trainer.current_epoch),
        }
        # The metrics dictionary is intentionally a mix of host telemetry and
        # run-position metadata:
        # - CPU/RAM/GPU fields describe system health
        # - global step / current epoch describe *when* that health snapshot
        #   was taken within training
        #
        # Keeping both together makes it much easier to correlate resource
        # spikes with specific training phases during later analysis.
        metrics.update(self._gpu_metrics())
        log_metrics_to_loggers(trainer, metrics, step=trainer.global_step)
        self._append_csv_row(metrics)

        if self.text_logger is not None:
            self.text_logger.info("telemetry %s", json.dumps(metrics, sort_keys=True))


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
        self.config = config
        self._handles: list[Any] = []
        self._pending_metrics: dict[str, float] = {}

    def _register_hook(self, pl_module: Any, module_name: str) -> None:
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
        del pl_module, outputs, batch, batch_idx
        if not self._pending_metrics:
            return
        # Flushing at batch end gives all hooked modules one shared global step
        # and avoids interleaving many tiny logger writes during the forward
        # pass itself.
        log_metrics_to_loggers(trainer, self._pending_metrics, step=trainer.global_step)
        self._pending_metrics = {}

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ModelTensorBoardCallback(Callback):
    """
    Push model architecture visualizations and text into TensorBoard.

    Purpose:
    expose the model itself, not just its scalar metrics, inside the
    TensorBoard experience.

    Context:
    this callback covers three complementary model-visualization surfaces:
    - plain-text architecture via `repr(pl_module)`
    - Lightning execution graph tracing when `add_graph(...)` succeeds
    - optional `torchview` rendering for a more presentation-oriented diagram

    All visualization work is best-effort. Failures to trace the model or
    render a graph are logged and ignored so training can continue normally.
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        text_logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.text_logger = text_logger
        self._graph_logged = False
        self._torchview_logged = False

    def _sample_tensor_batch(self, trainer: Any, pl_module: Any) -> Any:
        # We intentionally use a single sampled train batch for graph/model
        # visualization. This avoids perturbing the training loop while still
        # giving Lightning and torchview realistic tensor shapes.
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None

        try:
            batch = next(iter(datamodule.train_dataloader()))
        except Exception as exc:
            if self.text_logger is not None:
                self.text_logger.info(
                    "unable to sample batch for model visualization: %s", exc
                )
            return None

        batch_on_device = _move_batch_to_device(
            _tensor_only_structure(batch),
            pl_module.device,
        )
        # Two normalizations happen before graphing:
        # - `_tensor_only_structure(...)` drops metadata fields that graph
        #   tools cannot consume
        # - `_move_batch_to_device(...)` ensures the sample lives on the same
        #   device as the model so tracing does not fail due to device mismatch
        return batch_on_device

    def _log_torchview(
        self,
        trainer: Any,
        pl_module: Any,
        batch_on_device: Any,
    ) -> None:
        # torchview is complementary to TensorBoard, not a replacement for it.
        # The goal here is to create one static architecture artifact per run
        # that can also be surfaced back into TensorBoard as an image/text
        # pair.
        if not self.config.enable_torchview or self._torchview_logged:
            return
        if draw_graph is None or batch_on_device is None:
            return

        # The torchview artifact path is configured externally and may arrive
        # as a string. We normalize it once here before interacting with the
        # Graphviz render API or the filesystem.
        base_path = (
            None if self.config.torchview_path is None else Path(self.config.torchview_path)
        )
        if base_path is None:
            return
        base_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            graph = draw_graph(
                pl_module,
                input_data=batch_on_device,
                depth=self.config.torchview_depth,
                roll=self.config.torchview_roll,
                expand_nested=self.config.torchview_expand_nested,
                device=str(pl_module.device),
            )
            render_path = graph.visual_graph.render(
                filename=base_path.name,
                directory=str(base_path.parent),
                format="png",
                cleanup=True,
            )
            self._torchview_logged = True
        except Exception as exc:
            if self.text_logger is not None:
                self.text_logger.info("torchview rendering failed: %s", exc)
            return

        png_path = Path(render_path)
        source_text = getattr(getattr(graph, "visual_graph", None), "source", None)
        # We surface the torchview result back into TensorBoard in two forms:
        # - DOT/source text for inspection and debugging
        # - rendered PNG image for quick visual browsing
        #
        # This gives the user something useful even in environments where the
        # raw Graphviz artifact itself is not opened directly.
        for experiment in _tensorboard_experiments(trainer):
            add_text = getattr(experiment, "add_text", None)
            if callable(add_text) and source_text is not None:
                add_text("model/torchview_dot", source_text, global_step=0)

            add_image = getattr(experiment, "add_image", None)
            if callable(add_image) and _has_module("matplotlib"):
                try:
                    import matplotlib.image as mpimg

                    image = mpimg.imread(png_path)
                    add_image(
                        "model/torchview",
                        image,
                        global_step=0,
                        dataformats="HWC",
                    )
                except Exception as exc:
                    if self.text_logger is not None:
                        self.text_logger.info(
                            "torchview TensorBoard image logging failed: %s", exc
                        )

        if self.text_logger is not None:
            self.text_logger.info("saved torchview model diagram to %s", png_path)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        # We do model visualization at fit start because:
        # - the model is fully constructed
        # - the datamodule is already attached
        # - we only need one representative sample input for graphing
        # - repeating this every epoch would add noise and overhead
        experiments = _tensorboard_experiments(trainer)
        if not experiments:
            return

        if self.config.enable_model_text:
            model_text = repr(pl_module)
            for experiment in experiments:
                add_text = getattr(experiment, "add_text", None)
                if callable(add_text):
                    add_text("model/architecture", model_text, global_step=0)
            if self.text_logger is not None:
                self.text_logger.info("model architecture\n%s", model_text)

        batch_on_device = self._sample_tensor_batch(trainer, pl_module)
        self._log_torchview(trainer, pl_module, batch_on_device)
        # Torchview and Lightning graph logging are intentionally decoupled:
        # - torchview aims for a clearer architecture diagram
        # - `add_graph(...)` aims for a TensorBoard-native traced graph view
        #
        # One can succeed even if the other fails, so each path is attempted
        # independently.

        if not self.config.enable_model_graph or self._graph_logged:
            return
        if batch_on_device is None:
            return
        for experiment in experiments:
            add_graph = getattr(experiment, "add_graph", None)
            if not callable(add_graph):
                continue
            try:
                add_graph(pl_module, input_to_model=(batch_on_device,))
                self._graph_logged = True
            except Exception as exc:
                if self.text_logger is not None:
                    self.text_logger.info("graph logging failed: %s", exc)


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


class PredictionFigureCallback(Callback):
    """
    Log a few qualitative forecast examples directly into TensorBoard.

    Purpose:
    push qualitative forecast examples directly into TensorBoard.

    Context:
    the callback renders small matplotlib figures comparing targets, median
    predictions, and prediction intervals for representative validation/test
    examples. It is intentionally scoped to a handful of examples per epoch so
    it acts as a monitoring aid rather than a full report generator.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        self.config = config
        self._logged_validation_epochs: set[int] = set()
        self._logged_test_epochs: set[int] = set()

    def _should_log(self, trainer: Any, stage: str) -> bool:
        # We only log one small figure set per stage per eligible epoch. The
        # goal is qualitative inspection, not exhaustive visualization of every
        # batch.
        if not self.config.enable_prediction_figures:
            return False
        if not _has_module("matplotlib"):
            return False
        target_set = (
            self._logged_validation_epochs if stage == "val" else self._logged_test_epochs
        )
        if trainer.current_epoch in target_set:
            return False
        if (trainer.current_epoch + 1) % self.config.figure_every_n_epochs != 0:
            return False
        target_set.add(trainer.current_epoch)
        return True

    def _log_prediction_figure(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Mapping[str, Any],
        stage: str,
    ) -> None:
        # This uses the current model directly on the observed batch so the
        # figure reflects the live state of the run at that point in training.
        if not self._should_log(trainer, stage):
            return

        import matplotlib.pyplot as plt

        with torch.no_grad():
            predictions = pl_module(batch).detach().cpu()
        target = pl_module._target_tensor(batch).detach().cpu()
        metadata = _as_metadata_lists(batch["metadata"], int(predictions.shape[0]))
        # We intentionally run the live model on the current batch rather than
        # reusing some cached predictions because the point of this callback is
        # to show what the model currently believes at this stage of training.

        max_plots = min(self.config.max_prediction_plots, int(predictions.shape[0]))
        figure, axes = plt.subplots(
            max_plots,
            1,
            figsize=(10, max(4, 3 * max_plots)),
            squeeze=False,
        )
        median_index = min(
            range(len(pl_module.quantiles)),
            key=lambda index: abs(float(pl_module.quantiles[index]) - 0.5),
        )

        for plot_index in range(max_plots):
            axis = axes[plot_index][0]
            horizon = list(range(int(predictions.shape[1])))
            # Each subplot shows one forecast window:
            # - black line: ground-truth target trajectory
            # - blue line: median / representative point forecast
            # - shaded band: outer prediction interval if multiple quantiles
            #   exist
            #
            # This gives TensorBoard a quick qualitative "does this forecast
            # look reasonable?" surface alongside the scalar metrics.
            axis.plot(
                horizon,
                target[plot_index].tolist(),
                label="target",
                color="black",
                linewidth=2,
            )
            axis.plot(
                horizon,
                predictions[plot_index, :, median_index].tolist(),
                label="median prediction",
                color="tab:blue",
            )
            if predictions.shape[-1] >= 2:
                lower = predictions[plot_index, :, 0].tolist()
                upper = predictions[plot_index, :, -1].tolist()
                axis.fill_between(
                    horizon,
                    lower,
                    upper,
                    color="tab:blue",
                    alpha=0.2,
                    label="prediction interval",
                )
            subject_id = str(metadata.get("subject_id", ["unknown"])[plot_index])
            decoder_start = str(metadata.get("decoder_start", [""])[plot_index])
            axis.set_title(f"{stage} subject={subject_id} decoder_start={decoder_start}")
            axis.set_xlabel("Horizon Step")
            axis.set_ylabel("Glucose")
            axis.legend(loc="best")

        figure.tight_layout()
        for experiment in _tensorboard_experiments(trainer):
            add_figure = getattr(experiment, "add_figure", None)
            if callable(add_figure):
                add_figure(
                    f"predictions/{stage}",
                    figure,
                    global_step=trainer.global_step,
                )
        plt.close(figure)

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch_idx, dataloader_idx
        self._log_prediction_figure(trainer, pl_module, batch, "val")

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch_idx, dataloader_idx
        self._log_prediction_figure(trainer, pl_module, batch, "test")


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
    if config.enable_rich_progress_bar:
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
