from __future__ import annotations

# AI-assisted implementation note:
# This module centralizes the repository's observability/runtime diagnostics
# surface for Lightning training runs.
#
# What this module is responsible for:
# - constructing the active experiment logger (`TensorBoardLogger` when
#   available, otherwise a safe CSV fallback)
# - wiring optional profiler support through Lightning's native profiler API
# - exposing callbacks for telemetry, batch auditing, activation summaries,
#   parameter summaries, model visualization, and TensorBoard figures
# - exporting post-run prediction tables and lightweight Plotly HTML reports
#
# Important disclaimers:
# - the observability stack is intentionally best-effort and dependency-aware;
#   optional extras like TensorBoard, Plotly, matplotlib, torchview, or NVML
#   should enhance a run when installed but should not make the core training
#   workflow unusable when absent
# - several diagnostics are sampled rather than logged on every step because a
#   "log everything at full fidelity" policy can make Colab and notebook runs
#   prohibitively slow
# - these utilities are meant to support research/debugging visibility, not to
#   act as a replacement for rigorous validation, unit tests, or model-quality
#   guarantees

import csv
import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import psutil
import torch
from torch import Tensor

from pytorch_lightning.callbacks import (
    Callback,
    DeviceStatsMonitor,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler

from data.datamodule import AZT1DDataModule
from config import ObservabilityConfig, PathInput

try:
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:  # pragma: no cover - Lightning import issues surface elsewhere
    TensorBoardLogger = None  # type: ignore[assignment]

try:
    from torchview import draw_graph
except ImportError:  # pragma: no cover - optional dependency until installed
    draw_graph = None  # type: ignore[assignment]


# ============================================================================
# Runtime Artifact Bundle
# ============================================================================
# This small dataclass is the handoff point between observability setup and the
# rest of the training workflow.
#
# The pattern in this file is:
# 1. build loggers/profiler/text-log paths from `ObservabilityConfig`
# 2. store the resulting runtime objects in `ObservabilityArtifacts`
# 3. let the trainer wrapper consume that one bundle rather than reaching back
#    into several separate helper functions later
@dataclass(frozen=True)
class ObservabilityArtifacts:
    """
    Small bundle describing the active observability surface for a run.

    Purpose:
    capture the assembled logger/profiler/text-log state for a run in one
    object.

    Context:
    this object is returned by `setup_observability(...)` and handed to the
    training wrapper so the rest of the workflow can treat logging and
    profiling as one cohesive runtime concern rather than reaching into several
    separate helpers.

    It intentionally captures:
    - the active Lightning logger, if any
    - the optional plain-text file logger
    - the main filesystem locations for logs and telemetry
    - the active profiler instance when profiling is enabled
    """
    logger: Any
    text_logger: logging.Logger | None
    logger_dir: Path | None
    text_log_path: Path | None
    telemetry_path: Path | None
    torchview_path: Path | None = None
    profiler: Any = None


# ============================================================================
# Low-Level Utility Helpers
# ============================================================================
# These helpers are intentionally tiny and generic. They exist to keep the
# higher-level setup/callback code readable by moving repetitive checks and
# filesystem normalization into one place.
def _has_module(module_name: str) -> bool:
    """
    Check whether an optional dependency is importable.

    Context:
    the observability stack enables many features conditionally, so dependency
    discovery needs to be lightweight and side-effect free.
    """
    # We use importlib-based checks instead of direct imports when the goal is
    # simply "is this optional feature available?" That lets the file decide
    # whether to enable a capability without crashing eagerly when an optional
    # dependency is absent.
    return importlib.util.find_spec(module_name) is not None


def _ensure_parent(path: Path | None) -> None:
    """
    Create the parent directory for a file path when needed.

    Context:
    many observability artifacts are files rather than directories, so callers
    often need the parent created without assuming the file already exists.
    """
    # Many artifact paths in this module are file paths, not directory paths.
    # This helper ensures the parent folder exists before we attempt to write
    # the file itself.
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_dir(path: Path | None) -> None:
    """
    Create a directory path when needed.

    Context:
    this is the directory-oriented counterpart to `_ensure_parent(...)`.
    """
    # Counterpart to `_ensure_parent(...)` for fields that already represent a
    # directory rather than a file.
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Logger / Profiler Setup
# ============================================================================
# This block is responsible for constructing the runtime observability surface
# from `ObservabilityConfig`.
#
# In practice that means:
# - file logger for plain-text lifecycle/debug notes
# - Lightning logger for metrics/figures/hparams
# - optional Lightning profiler
# - a final bundle object returned to the training wrapper
def setup_text_logger(config: ObservabilityConfig) -> logging.Logger | None:
    """
    Build the optional plain-text run logger.

    Context:
    text logs complement TensorBoard by recording lifecycle and diagnostic
    messages that are awkward to inspect as scalars alone.
    """
    # Text logging complements TensorBoard rather than competing with it.
    # The file logger is useful for lifecycle notes and failures that would be
    # awkward to inspect only through scalar dashboards.
    if not config.enable_text_logging or config.text_log_path is None:
        return None

    # `ObservabilityConfig` accepts convenient string inputs from notebooks,
    # CLIs, and tests, but the file-logger implementation should only deal in
    # concrete `Path` objects once control enters this helper.
    text_log_path = Path(config.text_log_path)
    _ensure_parent(text_log_path)
    logger_name = f"ensf617.observability.{text_log_path}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == text_log_path
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(text_log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def build_lightning_logger(config: ObservabilityConfig) -> tuple[Any, Path | None]:
    """
    Build the active Lightning logger and return its resolved log directory.

    Context:
    TensorBoard is preferred when available, but CSV fallback keeps the training
    stack operational in minimal environments.
    """
    # We prefer Lightning's native TensorBoard integration because `self.log`,
    # callback metrics, hyperparameter logging, graphs, figures, and profiler
    # data all already know how to flow there naturally.
    #
    # The CSV fallback keeps training operational in stripped-down
    # environments, which is especially useful during local smoke tests or
    # partial dependency installs.
    # Normalize the public path-like config field once at the boundary so the
    # rest of the logger-construction code can safely use `Path` methods.
    log_dir = None if config.log_dir is None else Path(config.log_dir)
    _ensure_dir(log_dir)

    if config.enable_tensorboard and TensorBoardLogger is not None and _has_module("tensorboard"):
        if log_dir is None:
            return TensorBoardLogger(save_dir=".", name="lightning_logs"), None

        return (
            TensorBoardLogger(
                save_dir=str(log_dir.parent),
                name=log_dir.name,
                version="",
            ),
            log_dir,
        )

    if config.enable_csv_fallback_logger:
        fallback_dir = log_dir or Path("logs")
        _ensure_dir(fallback_dir)
        return (
            CSVLogger(
                save_dir=str(fallback_dir.parent),
                name=fallback_dir.name,
                version="",
            ),
            fallback_dir,
        )

    return None, log_dir


def build_profiler(config: ObservabilityConfig) -> Any:
    """
    Build the requested Lightning profiler instance when profiling is enabled.

    Context:
    profiler construction is centralized here so trainer assembly does not need
    to branch on profiler type or filesystem setup details.
    """
    # Profiler support is intentionally opt-in because even Lightning's native
    # profilers can add noticeable overhead. The config chooses the profiler
    # family while this helper keeps the instantiation details in one place.
    if not config.enable_profiler:
        return None

    profiler_type = config.profiler_type
    # Profiler output directories also originate from the path-like config
    # surface, so we normalize here before any filesystem operations.
    profiler_path = None if config.profiler_path is None else Path(config.profiler_path)
    if profiler_path is not None:
        profiler_path.mkdir(parents=True, exist_ok=True)

    if profiler_type == "simple":
        return SimpleProfiler(
            dirpath=None if profiler_path is None else str(profiler_path),
            filename="simple_profiler",
        )
    if profiler_type == "advanced":
        return AdvancedProfiler(
            dirpath=None if profiler_path is None else str(profiler_path),
            filename="advanced_profiler.txt",
        )
    return PyTorchProfiler(
        dirpath=None if profiler_path is None else str(profiler_path),
        filename="pytorch_profiler",
    )


def setup_observability(config: ObservabilityConfig) -> ObservabilityArtifacts:
    """
    Assemble the runtime observability surface for one run.

    Context:
    this bundles logger, text logger, profiler, and normalized artifact paths
    into one object consumed by the trainer wrapper.
    """
    # Assemble the entire runtime observability surface once near trainer
    # construction time so later code can stay declarative and simply consume
    # the resulting logger / profiler / path bundle.
    #
    # `ObservabilityConfig` accepts path-like inputs (`str | Path`) for
    # convenience, but `ObservabilityArtifacts` represents the normalized
    # runtime view, so we coerce those fields to concrete `Path` objects here.
    text_logger = setup_text_logger(config)
    lightning_logger, logger_dir = build_lightning_logger(config)
    profiler = build_profiler(config)
    return ObservabilityArtifacts(
        logger=lightning_logger,
        text_logger=text_logger,
        logger_dir=logger_dir,
        text_log_path=None if config.text_log_path is None else Path(config.text_log_path),
        telemetry_path=None if config.telemetry_path is None else Path(config.telemetry_path),
        torchview_path=None if config.torchview_path is None else Path(config.torchview_path),
        profiler=profiler,
    )


def _active_loggers(trainer: Any) -> list[Any]:
    """
    Return the trainer's active logger objects as a normalized list.

    Context:
    Lightning can expose either `trainer.logger` or `trainer.loggers`, so the
    rest of this module should not have to care which shape is active.
    """
    # Lightning may expose a single logger as `trainer.logger` or multiple
    # loggers as `trainer.loggers`. This helper normalizes both cases so the
    # rest of the module can treat "active loggers" as a simple list.
    loggers = getattr(trainer, "loggers", None)
    if loggers is not None:
        return list(loggers)
    logger = getattr(trainer, "logger", None)
    return [] if logger is None else [logger]


def _tensorboard_experiments(trainer: Any) -> list[Any]:
    """
    Return TensorBoard-compatible experiment backends from the active loggers.

    Context:
    only some logger backends expose TensorBoard methods like `add_text`,
    `add_histogram`, or `add_figure`.
    """
    # Not every active logger is a TensorBoard logger. This helper filters the
    # active logger set down to logger backends whose `.experiment` object
    # supports TensorBoard-style methods like `add_scalar`, `add_text`, and
    # `add_histogram`.
    experiments: list[Any] = []
    for logger in _active_loggers(trainer):
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_scalar"):
            experiments.append(experiment)
    return experiments


def log_metrics_to_loggers(trainer: Any, metrics: Mapping[str, float], step: int) -> None:
    """
    Push a precomputed metric mapping to every compatible active logger.

    Context:
    custom callbacks often compute metrics outside `LightningModule.self.log`,
    but they still need one shared path to the configured loggers.
    """
    # This helper is used by custom callbacks that compute metrics outside the
    # model's `self.log(...)` path. It pushes a ready-made metric dictionary to
    # every active logger that exposes Lightning's `log_metrics(...)` API.
    for logger in _active_loggers(trainer):
        log_metrics = getattr(logger, "log_metrics", None)
        if callable(log_metrics):
            log_metrics(dict(metrics), step=step)


def _log_text_to_loggers(trainer: Any, tag: str, text: str) -> None:
    """
    Publish a text payload to TensorBoard-compatible logger backends.

    Context:
    this is used for structured debug artifacts such as batch audits that are
    more readable as text than as scalar metrics.
    """
    # TensorBoard has a useful text surface for structured debug payloads such
    # as batch audits. This helper publishes text only to compatible
    # TensorBoard-backed experiments and skips other logger types quietly.
    for experiment in _tensorboard_experiments(trainer):
        add_text = getattr(experiment, "add_text", None)
        if callable(add_text):
            add_text(tag, text, global_step=getattr(trainer, "global_step", 0))


def _flatten_for_hparams(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, str | int | float | bool]:
    """
    Flatten nested config-like payloads into scalar logger-friendly entries.

    Context:
    hyperparameter backends usually expect a flat mapping rather than nested
    dictionaries or Python-specific objects.
    """
    # Hyperparameter logging is much more readable when nested config objects
    # are flattened into stable slash-delimited keys such as
    # `config/data/encoder_length`. This also avoids surprising behavior in
    # logger backends that expect scalar-like values rather than nested dicts.
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in payload.items():
        joined_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_for_hparams(value, prefix=joined_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[joined_key] = "None" if value is None else value
        elif isinstance(value, Path):
            flattened[joined_key] = str(value)
        else:
            flattened[joined_key] = json.dumps(value, sort_keys=True)
    return flattened


def log_hyperparameters(
    trainer: Any,
    payload: Mapping[str, Any],
) -> None:
    """
    Log a nested hyperparameter payload to the active logger set.

    Context:
    this helper keeps hyperparameter logging consistent across TensorBoard and
    any other configured Lightning logger backends.
    """
    # Flattening here keeps the logged hparams readable in TensorBoard and
    # compatible with logger backends that expect simple scalar/string values.
    flattened = _flatten_for_hparams(payload)
    for logger in _active_loggers(trainer):
        log_hparams = getattr(logger, "log_hyperparams", None)
        if callable(log_hparams):
            log_hparams(flattened)


# ============================================================================
# Tensor / Batch Normalization Helpers
# ============================================================================
# The custom callbacks below often need to inspect nested batch dictionaries or
# nested model outputs. These helpers normalize that work so each callback can
# stay focused on "what to log" rather than "how to recursively walk arbitrary
# nested structures."
def _flatten_tensor_output(output: Any) -> Tensor | None:
    """
    Find the first tensor payload inside a nested model output structure.

    Context:
    several debug callbacks only need one representative tensor for summary
    statistics, even when the real output is nested.
    """
    # Many model outputs in PyTorch ecosystems are nested structures rather
    # than a single tensor. For summary statistics we only need the first
    # actual tensor payload we can find.
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _flatten_tensor_output(item)
            if tensor is not None:
                return tensor
    if isinstance(output, Mapping):
        for item in output.values():
            tensor = _flatten_tensor_output(item)
            if tensor is not None:
                return tensor
    return None


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """
    Recursively move every tensor in a nested batch structure to one device.

    Context:
    model graph logging and `torchview` rendering need example inputs located on
    the same device as the model.
    """
    # TensorBoard graph logging and torchview rendering need example inputs on
    # the same device as the model. This helper recursively mirrors a nested
    # tensor structure onto the target device while leaving non-tensor metadata
    # untouched.
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)
    return batch


def _tensor_only_structure(batch: Any) -> Any:
    """
    Remove non-tensor values from a nested batch structure.

    Context:
    some visualization utilities only understand tensors, so metadata must be
    filtered out before those tools see the batch.
    """
    # Some visualization utilities do not know what to do with strings,
    # timestamps, or nested metadata objects. This helper removes everything
    # except tensors from a batch structure so the visualization path only sees
    # model-consumable inputs.
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, Mapping):
        filtered: dict[Any, Any] = {}
        for key, value in batch.items():
            nested = _tensor_only_structure(value)
            if nested is not None:
                filtered[key] = nested
        return filtered if filtered else None
    if isinstance(batch, list):
        filtered_list = [nested for value in batch if (nested := _tensor_only_structure(value)) is not None]
        return filtered_list if filtered_list else None
    if isinstance(batch, tuple):
        filtered_tuple = tuple(
            nested for value in batch if (nested := _tensor_only_structure(value)) is not None
        )
        return filtered_tuple if filtered_tuple else None
    return None


def _tensor_stats(tensor: Tensor) -> dict[str, float]:
    """
    Compute a compact health summary for one tensor.

    Context:
    these statistics are reused across multiple debug callbacks, so keeping them
    centralized ensures consistent logging behavior.
    """
    # Centralized tensor summary logic keeps the callback metrics consistent.
    # Every place in this file that wants a "quick health snapshot" of a tensor
    # uses the same mean/std/min/max/finite accounting.
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    total_count = detached.numel()
    if finite_count == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "finite_fraction": 0.0,
            "nan_count": float(torch.isnan(detached).sum().item()),
            "inf_count": float(torch.isinf(detached).sum().item()),
        }

    finite_values = detached[finite_mask]
    return {
        "mean": float(finite_values.mean().item()),
        "std": float(finite_values.std(unbiased=False).item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "finite_fraction": finite_count / total_count,
        "nan_count": float(torch.isnan(detached).sum().item()),
        "inf_count": float(torch.isinf(detached).sum().item()),
    }


def _summarize_batch(batch: Any) -> Any:
    """
    Convert a nested batch structure into a JSON-friendly summary object.

    Context:
    batch-audit logging needs tensor shape, dtype, and basic statistics without
    trying to serialize raw tensors directly.
    """
    # This turns an arbitrary nested batch structure into a JSON-friendly
    # summary object that can be logged as text. Tensors become shape/dtype/
    # stats dictionaries; non-tensor values are passed through as-is.
    if isinstance(batch, Tensor):
        return {
            "shape": list(batch.shape),
            "dtype": str(batch.dtype),
            "device": str(batch.device),
            "stats": _tensor_stats(batch),
        }
    if isinstance(batch, Mapping):
        return {str(key): _summarize_batch(value) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [_summarize_batch(value) for value in batch]
    return batch


def _as_metadata_lists(metadata: Mapping[str, Any], batch_size: int) -> dict[str, list[Any]]:
    """
    Normalize metadata values into per-sample Python lists.

    Context:
    prediction export and figure generation need metadata that can be indexed
    row by row regardless of whether the original source was scalar, tuple, or tensor.
    """
    # Prediction exports and figure callbacks need metadata in a predictable
    # row-oriented shape. This helper converts tensor/list/scalar metadata into
    # "list of per-sample values" form so downstream code can index it
    # uniformly.
    normalized: dict[str, list[Any]] = {}
    for key, value in metadata.items():
        if isinstance(value, Tensor):
            normalized[key] = value.detach().cpu().tolist()
        elif isinstance(value, list):
            normalized[key] = value
        elif isinstance(value, tuple):
            normalized[key] = list(value)
        else:
            normalized[key] = [value for _ in range(batch_size)]
    return normalized


class BatchAuditCallback(Callback):
    """
    Log a small number of representative batch summaries for debugging.

    Purpose:
    capture the structure of a few representative batches.

    Context:
    the callback records tensor shapes, dtypes, devices, and lightweight value
    statistics for a capped number of train/validation/test batches. This makes
    data-contract issues visible early without flooding the logs with one entry
    per batch for an entire run.
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
        self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int
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
        # We register on named high-level modules rather than every submodule to
        # keep the activation output readable and tied to the architecture the
        # user actually thinks about.
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
            # summary per module per logged step, not a full call-by-call trace.

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
    This callback covers three complementary model-visualization surfaces:
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
                self.text_logger.info("unable to sample batch for model visualization: %s", exc)
            return None

        batch_on_device = _move_batch_to_device(
            _tensor_only_structure(batch),
            pl_module.device,
        )
        # Two normalizations happen before graphing:
        # - `_tensor_only_structure(...)` drops metadata fields that graph tools
        #   cannot consume
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
        # that can also be surfaced back into TensorBoard as an image/text pair.
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
                        self.text_logger.info("torchview TensorBoard image logging failed: %s", exc)

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
        # Epoch-end logging keeps these summaries stable and affordable. Logging
        # per parameter on every step would be too heavy for normal runs.
        if not self.config.enable_parameter_scalars:
            return
        if (trainer.current_epoch + 1) % self.config.parameter_scalar_every_n_epochs != 0:
            return

        metrics: dict[str, float] = {}
        for name, parameter in pl_module.named_parameters():
            values = parameter.detach().float()
            metrics[self._tag_name(name, "mean")] = float(values.mean().item())
            metrics[self._tag_name(name, "std")] = float(values.std(unbiased=False).item())
            metrics[self._tag_name(name, "norm")] = float(torch.norm(values).item())
            metrics[self._tag_name(name, "max_abs")] = float(
                torch.max(torch.abs(values)).item()
            )
            if parameter.grad is not None:
                grad = parameter.grad.detach().float()
                metrics[self._tag_name(name, "grad_norm")] = float(torch.norm(grad).item())
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
        # Histograms are intentionally recorded at epoch granularity because the
        # payload size is much larger than scalar metrics.
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
            # - shaded band: outer prediction interval if multiple quantiles exist
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


#
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
    # it intentional makes the resulting callback stack easier to reason about.

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


# ============================================================================
# Post-Run Prediction Export
# ============================================================================
# The helpers below operate after prediction batches have already been produced.
# They turn raw tensors into analysis-friendly artifacts.
def export_prediction_table(
    *,
    datamodule: AZT1DDataModule,
    predictions: Sequence[Tensor],
    quantiles: Sequence[float],
    output_path: PathInput | None,
    sampling_interval_minutes: int,
) -> Path | None:
    """
    Export test predictions as a flat analysis-friendly CSV table.

    Context:
    the raw tensor dump preserves fidelity, while this table optimizes for
    plotting, inspection, and report generation.
    """
    # This export deliberately denormalizes prediction results into one flat
    # row-per-horizon table because that format is easy to inspect in a
    # notebook, easy to plot with Plotly/pandas, and easy to archive as a run
    # artifact.
    #
    # It complements the raw tensor dump written elsewhere in the workflow:
    # - raw `.pt` files preserve full tensor fidelity for PyTorch consumers
    # - this CSV prioritizes analysis convenience
    if output_path is None:
        return None
    output_path = Path(output_path)
    if not predictions:
        return None

    rows: list[dict[str, Any]] = []
    test_loader = datamodule.test_dataloader()
    quantile_columns = [f"pred_q{int(round(q * 100)):02d}" for q in quantiles]
    median_index = min(
        range(len(quantiles)),
        key=lambda index: abs(float(quantiles[index]) - 0.5),
    )

    # The exported table intentionally lines up predictions with the original
    # test dataloader batches so metadata such as subject ID and decoder start
    # can be attached row by row.
    for batch_index, (prediction_batch, batch) in enumerate(zip(predictions, test_loader)):
        prediction_cpu = prediction_batch.detach().cpu()
        target = batch["target"]
        if isinstance(target, Tensor):
            target_cpu = target.detach().cpu()
        else:
            target_cpu = torch.as_tensor(target)
        if target_cpu.ndim == 3 and target_cpu.shape[-1] == 1:
            target_cpu = target_cpu.squeeze(-1)

        batch_size = int(prediction_cpu.shape[0])
        metadata = _as_metadata_lists(batch["metadata"], batch_size)

        for sample_index in range(batch_size):
            subject_id = str(metadata.get("subject_id", ["unknown"])[sample_index])
            decoder_start = pd.Timestamp(
                str(metadata.get("decoder_start", ["1970-01-01 00:00:00"])[sample_index])
            )
            for horizon_index in range(int(prediction_cpu.shape[1])):
                # The export is intentionally one row per forecast horizon step
                # rather than one row per sample window. That denormalized shape
                # is what makes later plotting and grouped metric analysis with
                # pandas/Plotly straightforward.
                timestamp = decoder_start + pd.Timedelta(
                    minutes=sampling_interval_minutes * horizon_index
                )
                row = {
                    "prediction_batch_index": batch_index,
                    "sample_index_within_batch": sample_index,
                    "subject_id": subject_id,
                    "decoder_start": str(metadata.get("decoder_start", [""])[sample_index]),
                    "decoder_end": str(metadata.get("decoder_end", [""])[sample_index]),
                    "timestamp": timestamp.isoformat(),
                    "horizon_index": horizon_index,
                    "target": float(target_cpu[sample_index, horizon_index].item()),
                }
                for quantile_index, column_name in enumerate(quantile_columns):
                    row[column_name] = float(
                        prediction_cpu[sample_index, horizon_index, quantile_index].item()
                    )
                row["median_prediction"] = float(
                    prediction_cpu[sample_index, horizon_index, median_index].item()
                )
                row["residual"] = row["median_prediction"] - row["target"]
                if len(quantile_columns) >= 2:
                    row["prediction_interval_width"] = (
                        row[quantile_columns[-1]] - row[quantile_columns[0]]
                    )
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


# ============================================================================
# Post-Run Report Generation
# ============================================================================
# These reports are intentionally lightweight first-pass visual artifacts. They
# are meant to make a run inspectable immediately, not to replace a full
# analytics notebook.
def generate_plotly_reports(
    prediction_table_path: PathInput | None,
    *,
    report_dir: PathInput | None,
    max_subjects: int,
) -> dict[str, Path]:
    """
    Generate lightweight Plotly HTML reports from the exported prediction table.

    Context:
    these reports are intended to make each run immediately inspectable without
    requiring a separate notebook.
    """
    # These reports are intentionally lightweight first-pass diagnostics, not a
    # complete experiment-reporting system. The aim is to generate a few useful
    # HTML artifacts automatically from the flat prediction table so every run
    # leaves behind something visual and shareable.
    if prediction_table_path is None or report_dir is None:
        return {}
    prediction_table_path = Path(prediction_table_path)
    report_dir = Path(report_dir)
    if not prediction_table_path.exists():
        return {}
    if not _has_module("plotly"):
        return {}

    import plotly.express as px
    import plotly.graph_objects as go

    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(prediction_table_path)
    if frame.empty:
        return {}

    report_paths: dict[str, Path] = {}

    residual_histogram = px.histogram(
        frame,
        x="residual",
        nbins=50,
        title="Residual Distribution",
    )
    residual_histogram_path = report_dir / "residual_histogram.html"
    residual_histogram.write_html(str(residual_histogram_path))
    report_paths["residual_histogram"] = residual_histogram_path

    grouped = frame.assign(abs_error=lambda data: data["residual"].abs()).groupby(
        "horizon_index",
        as_index=False,
    )
    # Grouping by horizon index gives us a simple answer to one of the most
    # important forecasting diagnostics questions:
    # "How does error behave as we predict farther into the future?"
    #
    # That horizon-wise view is often more informative than one single global
    # metric because short-horizon and long-horizon behavior can differ a lot.
    aggregation: dict[str, Any] = {
        "mae": ("abs_error", "mean"),
        "rmse": ("residual", lambda values: float((values.pow(2).mean()) ** 0.5)),
    }
    if "prediction_interval_width" in frame.columns:
        aggregation["mean_interval_width"] = ("prediction_interval_width", "mean")
    horizon_metrics = grouped.agg(**aggregation)
    horizon_metrics_fig = go.Figure()
    horizon_metrics_fig.add_trace(
        go.Scatter(
            x=horizon_metrics["horizon_index"],
            y=horizon_metrics["mae"],
            mode="lines+markers",
            name="MAE",
        )
    )
    horizon_metrics_fig.add_trace(
        go.Scatter(
            x=horizon_metrics["horizon_index"],
            y=horizon_metrics["rmse"],
            mode="lines+markers",
            name="RMSE",
        )
    )
    if "mean_interval_width" in horizon_metrics:
        horizon_metrics_fig.add_trace(
            go.Scatter(
                x=horizon_metrics["horizon_index"],
                y=horizon_metrics["mean_interval_width"],
                mode="lines+markers",
                name="Mean Interval Width",
                yaxis="y2",
            )
        )
        horizon_metrics_fig.update_layout(
            yaxis2=dict(
                title="Interval Width",
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )
    horizon_metrics_fig.update_layout(title="Error Metrics By Forecast Horizon")
    horizon_metrics_path = report_dir / "horizon_metrics.html"
    horizon_metrics_fig.write_html(str(horizon_metrics_path))
    report_paths["horizon_metrics"] = horizon_metrics_path

    overview_fig = go.Figure()
    subject_ids = list(dict.fromkeys(frame["subject_id"].tolist()))[:max_subjects]
    filtered = frame[frame["subject_id"].isin(subject_ids)].copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
    filtered.sort_values(["subject_id", "timestamp"], inplace=True)

    for subject_id in subject_ids:
        subject_frame = filtered[filtered["subject_id"] == subject_id]
        if subject_frame.empty:
            continue
        overview_fig.add_trace(
            go.Scatter(
                x=subject_frame["timestamp"],
                y=subject_frame["target"],
                mode="lines",
                name=f"{subject_id} target",
            )
        )
        overview_fig.add_trace(
            go.Scatter(
                x=subject_frame["timestamp"],
                y=subject_frame["median_prediction"],
                mode="lines",
                name=f"{subject_id} median",
            )
        )
        quantile_columns = sorted(
            column for column in subject_frame.columns if column.startswith("pred_q")
        )
        if len(quantile_columns) >= 2:
            lower = quantile_columns[0]
            upper = quantile_columns[-1]
            overview_fig.add_trace(
                go.Scatter(
                    x=subject_frame["timestamp"],
                    y=subject_frame[upper],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            overview_fig.add_trace(
                go.Scatter(
                    x=subject_frame["timestamp"],
                    y=subject_frame[lower],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    opacity=0.2,
                    name=f"{subject_id} interval",
                )
            )

    overview_fig.update_layout(title="Forecast Overview")
    overview_path = report_dir / "forecast_overview.html"
    overview_fig.write_html(str(overview_path))
    report_paths["forecast_overview"] = overview_path

    return report_paths
