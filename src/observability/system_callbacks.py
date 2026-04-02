from __future__ import annotations

# AI-assisted maintenance note:
# These callbacks focus on run-wide system visibility rather than on parameter-
# specific or prediction-specific instrumentation.
#
# Why group them together:
# - both callbacks describe the environment around the model, not the training
#   signal itself
# - they share filesystem/logging helpers and depend on optional visualization
#   integrations
# - keeping them separate from the smaller debug callbacks reduces the amount
#   of unrelated code a reader has to hold in their head at once

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import psutil
import torch
from pytorch_lightning.callbacks import Callback

from config import ObservabilityConfig
from observability.logging_utils import (
    _tensorboard_experiments,
    log_metrics_to_loggers,
)
from observability.tensors import (
    _move_batch_to_device,
    _tensor_only_structure,
)
from observability.utils import _ensure_parent, _has_module

try:
    from torchview import draw_graph
except ImportError:  # pragma: no cover - optional dependency until installed
    draw_graph = None  # type: ignore[assignment]


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
        """Store telemetry policy and optional plain-text logging surface for the callback."""
        self.config = config
        self.text_logger = text_logger
        self._telemetry_header_written = False

    def _gpu_metrics(self) -> dict[str, float]:
        """
        Collect best-effort GPU or MPS telemetry in the same metric shape used by host logging.

        Context:
        returning one stable dictionary keeps downstream metric logging and CSV
        export simple even when only some backend-specific telemetry is available.
        """
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
        """Append one telemetry snapshot to the configured CSV artifact when that output is enabled."""
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
        """Sample and publish host/device telemetry at the configured step interval."""
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
        """Store visualization policy and the one-shot state used by model-graph logging."""
        self.config = config
        self.text_logger = text_logger
        self._graph_logged = False
        self._torchview_logged = False

    def _sample_tensor_batch(self, trainer: Any, pl_module: Any) -> Any:
        """Sample one tensor-only training batch suitable for graph/architecture visualization."""
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
        """Render the optional torchview artifact and surface it back into TensorBoard when possible."""
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
            None
            if self.config.torchview_path is None
            else Path(self.config.torchview_path)
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
        """Emit one-time model text, graph, and optional torchview artifacts at fit start."""
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
