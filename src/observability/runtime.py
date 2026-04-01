from __future__ import annotations

# AI-assisted maintenance note:
# This module assembles the runtime observability surface for a training run.
#
# Responsibility boundary:
# - build the plain-text file logger
# - build the Lightning logger
# - build the optional profiler
# - normalize the resulting runtime objects and paths into one bundle
#
# What does *not* live here:
# - callback implementations
# - tensor/batch inspection helpers
# - post-run CSV/HTML reporting
#
# That separation matters because these runtime objects are needed very early
# during trainer construction, while callbacks and reports are consumed later.

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)

from config import ObservabilityConfig
from observability.utils import _ensure_dir, _ensure_parent, _has_module

try:
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:  # pragma: no cover - Lightning import issues surface elsewhere
    TensorBoardLogger = None  # type: ignore[assignment]


# ============================================================================
# Runtime Artifact Bundle
# ============================================================================
# This small dataclass is the handoff point between observability setup and the
# rest of the training workflow.
#
# The pattern in this package is:
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

    if (
        config.enable_tensorboard
        and TensorBoardLogger is not None
        and _has_module("tensorboard")
    ):
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
    profiler_path = (
        None if config.profiler_path is None else Path(config.profiler_path)
    )
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
        text_log_path=(
            None if config.text_log_path is None else Path(config.text_log_path)
        ),
        telemetry_path=(
            None if config.telemetry_path is None else Path(config.telemetry_path)
        ),
        torchview_path=(
            None if config.torchview_path is None else Path(config.torchview_path)
        ),
        profiler=profiler,
    )
