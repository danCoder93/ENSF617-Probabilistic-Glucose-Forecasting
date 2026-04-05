from __future__ import annotations

# AI-assisted maintenance note:
# These callbacks focus on run-wide system visibility rather than on parameter-
# specific or prediction-specific instrumentation.
#
# Why group them together:
# - both callbacks describe the environment around the model, not the training
#   signal itself
# - both callbacks produce observability artifacts that are useful outside the
#   immediate training step
# - both callbacks depend on helper utilities for logger discovery, tensor
#   normalization, and filesystem-safe artifact writing
# - keeping them separate from smaller debug callbacks reduces the amount of
#   unrelated code a reader has to hold in their head at once
#
# Design boundary:
# - this file orchestrates observability work
# - it does not define model architecture
# - it does not define data preprocessing
# - it does not define optimizer or training-step logic
#
# Practical reading guide:
# - `SystemTelemetryCallback` answers: "what was the machine doing while the
#   run progressed?"
# - `ModelTensorBoardCallback` answers: "what model did we run, and can we
#   render its structure as text/graph artifacts?"

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import psutil
import torch
from pytorch_lightning.callbacks import Callback
from observability.model_visualization import TorchviewFusedAdapter, warmup_visualization_model

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
    """Record host and device telemetry alongside model metrics.

    Purpose:
        Log runtime system telemetry alongside training metrics so the user can
        correlate model behavior with machine behavior.

    Context:
        This callback emits lightweight host/device telemetry during training
        and mirrors that telemetry into two places:
        - active Lightning loggers for live inspection
        - a CSV artifact for offline inspection after the run

        The callback is intentionally best-effort:
        - missing GPU backends should not break training
        - missing NVML bindings should not break training
        - unsupported utilization APIs should degrade gracefully to zeros

    Why this callback exists:
        Scalar metrics like train loss or val loss tell us whether the model is
        learning, but they do not explain whether the machine was memory-bound,
        under-utilized, or unstable. This callback fills that gap.
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        text_logger: logging.Logger | None = None,
    ) -> None:
        """Store telemetry policy and optional plain-text logging surface.

        Parameters:
            config:
                Observability settings controlling whether telemetry is enabled,
                how often it is sampled, and where CSV artifacts are written.
            text_logger:
                Optional human-readable logger used for plain-text run logs in
                addition to structured logger backends.
        """
        self.config = config
        self.text_logger = text_logger

        # We write the telemetry CSV incrementally over the course of training.
        # This flag ensures the header is written exactly once, using the first
        # observed row as the canonical column order.
        self._telemetry_header_written = False

    def _gpu_metrics(self) -> dict[str, float]:
        """Collect best-effort GPU or MPS telemetry in a stable metric shape.

        Returns:
            A dictionary with the same keys regardless of backend availability.

        Why the return shape is fixed:
            Downstream logging is much simpler when telemetry always uses the
            same metric names. Even when a backend cannot provide utilization,
            returning zeros is preferable to changing the schema mid-run.

        Backend behavior:
            - Apple Silicon / MPS:
                Use MPS memory APIs when available. Utilization is reported as
                `0.0` because a reliable utilization percentage is not exposed
                in the same way as CUDA+NVML.
            - CUDA without NVML:
                Report CUDA memory stats and leave utilization at `0.0`.
            - CUDA with NVML:
                Add GPU utilization percentage when supported.
            - CPU-only:
                Return zeros for all GPU-related metrics.

        Failure behavior:
            Any optional-backend failure should degrade to partial telemetry,
            never to a training error.
        """
        # CUDA memory stats are always available when CUDA is active, but
        # utilization percentage depends on optional NVML bindings. We return a
        # consistent metric dictionary either way so downstream logging stays
        # stable and CSV columns do not drift across environments.
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_runtime = getattr(torch, "mps", None)

        if (
            mps_backend is not None
            and bool(getattr(mps_backend, "is_available", lambda: False)())
            and mps_runtime is not None
        ):
            # On Apple Silicon we use the runtime memory counters that are
            # exposed through torch.mps. These values are still useful for
            # capacity debugging even though utilization percentage is not
            # available in the same backend-neutral form as CUDA+NVML.
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
            # CPU-only runs should still produce a stable telemetry schema so
            # downstream dashboards and CSV readers do not need special cases.
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
                # NVML access should never be allowed to interrupt the run.
                # Memory telemetry is still useful even without utilization.
                pass

        return metrics

    def _append_csv_row(self, row: Mapping[str, float]) -> None:
        """Append one telemetry snapshot to the configured CSV artifact.

        Why CSV is written in addition to live logger backends:
            TensorBoard and other logger backends are convenient for live
            inspection, but a simple CSV is often easier to load later for:
            - quick spreadsheet inspection
            - custom plotting
            - comparing telemetry against other run artifacts
            - attaching plain tabular evidence to bug reports

        Behavior:
            - no-op when CSV output is disabled
            - creates parent directories on demand
            - writes header once using the first observed row
        """
        # The CSV output mirrors what is logged live. This is useful when the
        # user wants to inspect runtime telemetry after the run without opening
        # TensorBoard or relying on any specific experiment-tracking backend.
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
        """Sample and publish host/device telemetry at the configured interval.

        Hook choice:
            `on_train_batch_end` is a good place to sample telemetry because:
            - the model has just completed useful work
            - trainer step/epoch counters are current
            - the callback can sample at a predictable step cadence
            - it avoids interfering with forward-path control flow

        Intentional non-use of some hook parameters:
            The hook signature is fixed by Lightning even though this callback
            does not need the actual model outputs or batch payload.
        """
        del pl_module, outputs, batch, batch_idx

        # Respect the observability configuration first. This keeps the callback
        # cheap to keep registered even in runs where telemetry is disabled.
        if not self.config.enable_system_telemetry:
            return

        # Skip sanity checking to avoid mixing pre-training validation probes
        # with real training telemetry in the same time series.
        if trainer.sanity_checking:
            return

        # Sample only on the configured cadence so the callback remains low
        # overhead even during long runs.
        if trainer.global_step % self.config.telemetry_every_n_steps != 0:
            return

        memory = psutil.virtual_memory()
        metrics = {
            "telemetry/cpu_percent": float(psutil.cpu_percent(interval=None)),
            "telemetry/ram_percent": float(memory.percent),
            "telemetry/ram_used_gb": float(memory.used / (1024.0**3)),
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

        # The plain-text log is useful as a forensic fallback when richer
        # backends are unavailable, disabled, or incomplete in saved artifacts.
        if self.text_logger is not None:
            self.text_logger.info("telemetry %s", json.dumps(metrics, sort_keys=True))


class ModelTensorBoardCallback(Callback):
    """Push model architecture text and graph artifacts into observability sinks.

    Purpose:
        Expose the model itself, not just scalar metrics, inside the run's
        observability outputs.

    Context:
        This callback covers three complementary model-visibility surfaces:
        - plain-text architecture via `repr(pl_module)`
        - TensorBoard-native graph tracing via `add_graph(...)`
        - optional `torchview` rendering for a cleaner presentation-oriented
          static architecture diagram

        All visualization work is best-effort. Failures to trace the model or
        render a graph are logged and ignored so training can continue normally.

    Why this callback exists:
        When debugging a complex model, scalar metrics alone are not enough.
        We also want evidence of:
        - what module structure was actually instantiated
        - whether the graph can be traced
        - whether the model surface seen by observability tools is stable
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        text_logger: logging.Logger | None = None,
    ) -> None:
        """Store visualization policy and one-shot callback state.

        State flags:
            `_graph_logged`
                Prevents repeated TensorBoard graph tracing.
            `_torchview_logged`
                Prevents repeated torchview rendering.

        Why one-shot behavior matters:
            Model architecture is effectively static within a run, so repeated
            graph export adds overhead and artifact noise without providing
            additional insight.
        """
        self.config = config
        self.text_logger = text_logger
        self._graph_logged = False
        self._torchview_logged = False

    def _sample_tensor_batch(self, trainer: Any, pl_module: Any) -> Any:
        """Sample one tensor-only training batch for model visualization.

        Why a real batch is needed:
            Graphing tools need concrete input data to infer:
            - tensor shapes
            - execution paths
            - connectivity between submodules

        Why a training batch is used:
            - it matches the real model contract
            - it reflects true dataset preprocessing and collation behavior
            - it avoids divergence between synthetic dummy inputs and actual
              runtime inputs

        Normalization steps:
            1. `_tensor_only_structure(batch)`
               Removes metadata and other non-tensor values that graph tools
               are not designed to consume.
            2. `_move_batch_to_device(...)`
               Ensures the sample is on the same device as the model before
               tracing, avoiding device mismatch failures.

        Failure behavior:
            If the datamodule is missing or batch sampling fails, this returns
            `None` and the caller should skip graph export gracefully.
        """
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
                    "unable to sample batch for model visualization: %s",
                    exc,
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
        """Render the optional torchview artifact and surface it into loggers.

        What this path does:
            - wraps the model in a thin visualization adapter
            - runs a single graph-building pass using one sampled batch
            - renders a PNG via Graphviz
            - optionally mirrors the result into TensorBoard as text/image

        Why torchview is treated separately from `add_graph(...)`:
            TensorBoard graph tracing and torchview solve related but different
            problems:
            - `add_graph(...)` is a TensorBoard-native tracing surface
            - `torchview` aims to generate a cleaner architecture diagram

            Either one may succeed when the other fails, so they are attempted
            independently.

        Why the adapter is used:
            The wrapped Lightning module may be a more complicated tracing
            surface than visualization tools prefer. The adapter presents a
            narrower plain-`nn.Module` surface without changing real model
            behavior.

        Failure behavior:
            This path is strictly best-effort:
            - if torchview is disabled, missing, or fails, training continues
            - failures are logged for debugging
            - no exception is allowed to escape into training control flow
        """
        # torchview is complementary to TensorBoard, not a replacement for it.
        # The goal here is to create one static architecture artifact per run
        # that can also be surfaced back into TensorBoard as an image/text pair.
        if not self.config.enable_torchview or self._torchview_logged:
            return
        if draw_graph is None or batch_on_device is None:
            return

        # The torchview artifact path is configured externally and may arrive as
        # a string. We normalize it once here before interacting with the
        # Graphviz render API or the filesystem.
        base_path = (
            None
            if self.config.torchview_path is None
            else Path(self.config.torchview_path)
        )
        if base_path is None:
            return

        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Wrap the Lightning module in a small visualization-facing adapter.
        #
        # This keeps the actual model architecture untouched while giving
        # torchview a simpler tracing surface. If tracing still fails after
        # this point, the likely cause is the model/input contract itself
        # rather than callback orchestration.
        graph_model = TorchviewFusedAdapter(pl_module).to(pl_module.device)

        # Perform a one-time warmup forward pass before invoking torchview graph capture.
        #
        # Rationale:
        #   - Some submodules (e.g., lazy layers or dynamically constructed components)
        #     finalize their internal structure only after the first forward pass.
        #   - Torchview internally traces the model multiple times and expects a stable
        #     module graph across invocations. If the model mutates between those passes,
        #     tracing fails with graph mismatch errors.
        #   - Running an explicit warmup here ensures that any deferred initialization
        #     happens *before* graph capture begins, improving trace stability.
        #
        # Design notes:
        #   - The warmup runs under torch.no_grad() and eval() mode inside the helper,
        #     so it does not affect gradients, optimizer state, or training behavior.
        #   - The original training/eval state is restored after the warmup completes.
        #
        # Failure handling:
        #   - Warmup is best-effort only. Any failure is logged and we skip visualization
        #     entirely rather than interrupting training.
        #   - This preserves the callback’s non-intrusive design: observability must
        #     never break the training workflow.
        try:
            warmup_visualization_model(graph_model, batch_on_device)
        except Exception as exc:
            if self.text_logger is not None:
                self.text_logger.info("torchview warmup failed: %s", exc)
            return

        # Attempt to generate a torchview computational graph for the model.
        #
        # Rationale:
        #   - draw_graph(...) performs a trace-based traversal of the model using the
        #     provided example batch. It builds a visual representation of module
        #     structure, tensor flow, and nested submodules.
        #   - This complements text-based summaries by giving a structural view of how
        #     inputs propagate through the model.
        #
        # Important:
        #   - This step depends on tracing stability. If the model contains dynamic
        #     control flow, shape-dependent branching, or recently-initialized lazy
        #     modules, graph capture may fail even after warmup.
        #   - The adapter (TorchviewFusedAdapter) ensures a simplified output contract,
        #     but does not eliminate all trace-time risks.
        try:
            graph = draw_graph(
                graph_model,
                input_data=batch_on_device,
                depth=self.config.torchview_depth,          # limit traversal depth to control graph size/complexity
                roll=self.config.torchview_roll,            # collapse repeated modules for readability
                expand_nested=self.config.torchview_expand_nested,  # optionally expand nested modules in detail
                device=str(pl_module.device),               # ensure tracing occurs on the correct device
            )

            # Render the traced graph to a PNG file using Graphviz.
            #
            # Behavior:
            #   - graph.visual_graph is a Graphviz Digraph object generated by torchview
            #   - render(...) writes both the image and (temporarily) the source .dot file
            #   - cleanup=True removes intermediate files after rendering
            #
            # Output:
            #   - A PNG image is saved under the configured model_viz artifact directory
            #   - The returned render_path points to the generated file
            render_path = graph.visual_graph.render(
                filename=base_path.name,                    # base filename for the rendered artifact
                directory=str(base_path.parent),            # target directory for model_viz artifacts
                format="png",                              # output format (PNG for portability)
                cleanup=True,                              # remove intermediate Graphviz files (.dot, etc.)
            )

            # Mark visualization as successfully generated so it is not retried.
            # The callback is designed to log the graph once per run to avoid redundant work.
            self._torchview_logged = True

        # Failure handling:
        #   - Any exception during tracing or rendering is caught and logged.
        #   - Common failure causes include:
        #       * unstable tracing graphs across invocations
        #       * unsupported Python control flow in forward(...)
        #       * missing Graphviz backend (dot executable)
        #   - Visualization is optional, so we fail gracefully and continue training.
        except Exception as exc:
            if self.text_logger is not None:
                self.text_logger.info("torchview rendering failed: %s", exc)
            return


        # Normalize the render path to a Path object for downstream processing.
        png_path = Path(render_path)

        # Extract raw Graphviz source (DOT format) if available.
        #
        # Purpose:
        #   - Provides a text-based representation of the graph for debugging,
        #     reproducibility, or artifact logging alongside the PNG.
        #   - Useful when visual rendering succeeds but deeper inspection is needed.
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
                            "torchview TensorBoard image logging failed: %s",
                            exc,
                        )

        if self.text_logger is not None:
            self.text_logger.info("saved torchview model diagram to %s", png_path)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Emit one-time model text, graph, and optional torchview artifacts.

        Why this hook is used:
            `on_fit_start` is late enough that:
            - the model is fully constructed
            - trainer state exists
            - the datamodule is attached
            - device placement has been resolved

            But it is early enough that:
            - visualization happens before the run has progressed far
            - artifacts represent the initial model structure for that run
            - repeated per-epoch export is avoided

        Execution order:
            1. Discover TensorBoard experiments, if any.
            2. Sample one normalized batch for tracing.
            3. Attempt torchview rendering regardless of TensorBoard presence.
            4. If TensorBoard exists, log model text.
            5. If enabled, attempt TensorBoard-native graph tracing.

        Important design choice:
            Torchview is intentionally not gated by TensorBoard availability.
            The rendered PNG artifact is useful on its own, so we should still
            produce it in runs that do not attach a TensorBoard logger.

        Failure behavior:
            Visualization errors should not stop training. This callback is for
            observability only.
        """
        # We do model visualization at fit start because:
        # - the model is fully constructed
        # - the datamodule is already attached
        # - we only need one representative sample input for graphing
        # - repeating this every epoch would add noise and overhead
        experiments = _tensorboard_experiments(trainer)

        batch_on_device = self._sample_tensor_batch(trainer, pl_module)

        # Torchview should not depend on TensorBoard logger availability.
        #
        # We still surface the rendered artifact into TensorBoard when a
        # compatible experiment logger exists, but the PNG artifact itself is
        # useful even without TensorBoard. Keeping this call outside the
        # `experiments` gate avoids silently skipping torchview in runs that do
        # not attach a TensorBoard logger.
        self._log_torchview(trainer, pl_module, batch_on_device)

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