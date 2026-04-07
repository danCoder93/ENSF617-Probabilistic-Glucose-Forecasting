from __future__ import annotations

# AI-assisted maintenance note:
# This module owns Trainer- and checkpoint-related configuration. It keeps
# execution-policy concerns separate from the data/model contracts so callers
# can reason about runtime behavior without paging through the architectural
# config definitions.

from dataclasses import dataclass
from pathlib import Path

from config.types import PathInput


@dataclass
class SnapshotConfig:
    """
    Configuration for optional Lightning checkpoint snapshots.

    Purpose:
    answer the run-management question:
    "if we want to save model state during training, what exactly should be
    saved, where should it go, and how should Lightning decide which snapshots
    matter?"

    Context:
    it maps directly onto Lightning's `ModelCheckpoint` callback, but keeps the
    project's preferred checkpoint policy in one typed place instead of
    scattering callback arguments across scripts and notebooks.
    """

    # Whether snapshotting is enabled at all for the run. Turning this off lets
    # a caller do quick experiments without writing checkpoints to disk.
    enabled: bool = True

    # Optional target directory for saved snapshots. When left as `None`, the
    # training wrapper lets Lightning fall back to its default root-dir logic.
    dirpath: PathInput | None = None

    # Filename template passed to `ModelCheckpoint`. The default keeps the epoch
    # number and validation loss in the path because those are usually the most
    # useful pieces of information when inspecting saved checkpoints manually.
    filename: str = "epoch={epoch:02d}-val_loss={val_loss:.4f}"

    # Metric name used to rank checkpoints when validation-driven snapshotting
    # is active. This should match the metric logged by the LightningModule.
    monitor: str = "val_loss"

    # Whether lower metric values are better (`min`) or higher ones are better
    # (`max`).
    mode: str = "min"

    # Number of top-ranked checkpoints to keep. `1` means keep only the current
    # best checkpoint; `0` disables ranked checkpoint saving; `-1` means keep
    # every checkpoint Lightning would otherwise save.
    save_top_k: int = 1

    # Whether to additionally keep Lightning's special "last" checkpoint that
    # tracks the most recent training state regardless of metric ranking.
    save_last: bool = True

    # Whether the checkpoint should contain only model weights or the full
    # Lightning training state. Weights-only snapshots are smaller; full
    # checkpoints are better for exact training resumption.
    save_weights_only: bool = False

    def __post_init__(self) -> None:
        """
        Normalize snapshot paths and validate the checkpoint-ranking policy.

        Context:
        snapshot configuration maps directly onto Lightning callback behavior,
        so validating it here keeps invalid run-control policies out of the
        training wrapper.
        """
        # Normalize path-like inputs once so the rest of the training code can
        # assume `Path | None` semantics rather than handling raw strings too.
        if self.dirpath is not None:
            self.dirpath = Path(self.dirpath)

        # Lightning allows `save_top_k=-1` as the "save all checkpoints" mode,
        # so values below that are always invalid.
        if self.save_top_k < -1:
            raise ValueError("save_top_k must be >= -1")

        # `ModelCheckpoint` expects a ranking direction. Guarding it here keeps
        # invalid values from surfacing later only when training is launched.
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")


@dataclass
class TrainConfig:
    """
    Configuration for PyTorch Lightning training runs.

    Purpose:
    represent the Trainer-side execution policy for a run:
    device placement, numerical precision, epoch limits, sanity checks,
    progress display, and other loop-level behavior that sits above the model
    itself.

    Context:
    keeping these values in a typed config gives `main.py` and notebooks a
    single object to tune, log, and pass around instead of rebuilding long
    `Trainer(...)` argument lists at every call site.
    """

    # Lightning accelerator selection. Typical values are `"auto"`, `"cpu"`,
    # `"gpu"`, or `"mps"` depending on the environment.
    accelerator: str = "auto"

    # Device selection passed straight to Lightning. This may be `"auto"`, an
    # integer device count, or an explicit list of device indices.
    devices: str | int | list[int] = "auto"

    # Trainer precision policy. This stays flexible because Lightning accepts
    # both integer styles like `32` and string styles like `"16-mixed"`.
    precision: int | str = 32

    # Optional gradient clipping threshold passed to Lightning.
    gradient_clip_val: float | None = None

    # Number of gradient accumulation steps before each optimizer update.
    accumulate_grad_batches: int = 1

    # Distributed/runtime strategy passed through to Lightning.
    strategy: str = "auto"

    # Whether to enable sync batch norm when running across multiple devices.
    sync_batchnorm: bool = False

    # Optional PyTorch backend/runtime tuning knobs.
    matmul_precision: str | None = None
    allow_tf32: bool | None = None
    cudnn_benchmark: bool | None = None
    intraop_threads: int | None = None
    interop_threads: int | None = None
    mps_high_watermark_ratio: float | None = None
    mps_low_watermark_ratio: float | None = None
    enable_mps_fallback: bool | None = None

    # Optional model compilation controls. Kept off by default because compile
    # support can vary across model architectures and debugging workflows.
    compile_model: bool = False
    compile_mode: str | None = None
    compile_fullgraph: bool = False

    # Hard upper bound on epochs for the run. Early stopping may still end the
    # run sooner if validation monitoring is enabled.
    max_epochs: int = 20

    # Whether Lightning should request deterministic behavior where possible.
    # This can improve reproducibility, sometimes at a runtime cost.
    deterministic: bool = False

    # Logging frequency in optimizer steps.
    log_every_n_steps: int = 10

    # Number of validation sanity batches Lightning should run before the first
    # training epoch. This often catches shape or metric issues early.
    num_sanity_val_steps: int = 2

    # Lightning's one-batch smoke-test mode for debugging the full training
    # loop quickly.
    fast_dev_run: bool = False

    # Optional trainer limits for train/validation/test loops. Lightning
    # interprets floats as fractions of the loader and ints as explicit batch
    # counts.
    limit_train_batches: int | float = 1.0
    limit_val_batches: int | float = 1.0
    limit_test_batches: int | float = 1.0

    # Progress/summary toggles. These are useful to keep configurable because
    # notebook runs and automated jobs often want different console behavior.
    enable_progress_bar: bool = True
    enable_model_summary: bool = True

    # Optional default root directory that Lightning can use for logs,
    # checkpoints, and other run artifacts when the logger/callback stack does
    # not override them more specifically.
    default_root_dir: PathInput | None = None

    # Early stopping patience expressed in validation checks. `None` disables
    # early stopping entirely.
    early_stopping_patience: int | None = 5

    def __post_init__(self) -> None:
        """
        Normalize Trainer-facing path inputs and validate loop/runtime tuning fields.

        Context:
        this keeps the Trainer wrapper focused on orchestration by ensuring the
        typed runtime policy is already coherent before Lightning objects are
        assembled.
        """
        # Normalize path-like values once so downstream orchestration can treat
        # this field consistently as a `Path | None`.
        if self.default_root_dir is not None:
            self.default_root_dir = Path(self.default_root_dir)

        # Basic scalar validation keeps obvious run-configuration mistakes close
        # to config construction time instead of surfacing only after the
        # Trainer has already been assembled.
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if self.log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be > 0")
        if self.num_sanity_val_steps < 0:
            raise ValueError("num_sanity_val_steps must be >= 0")
        if self.gradient_clip_val is not None and self.gradient_clip_val < 0.0:
            raise ValueError("gradient_clip_val must be >= 0.0 or None")
        if self.accumulate_grad_batches <= 0:
            raise ValueError("accumulate_grad_batches must be > 0")
        if self.matmul_precision is not None and self.matmul_precision not in {
            "highest",
            "high",
            "medium",
        }:
            raise ValueError("matmul_precision must be one of 'highest', 'high', or 'medium'")
        if self.intraop_threads is not None and self.intraop_threads <= 0:
            raise ValueError("intraop_threads must be > 0 or None")
        if self.interop_threads is not None and self.interop_threads <= 0:
            raise ValueError("interop_threads must be > 0 or None")
        if self.mps_high_watermark_ratio is not None and self.mps_high_watermark_ratio <= 0.0:
            raise ValueError("mps_high_watermark_ratio must be > 0.0 or None")
        if self.mps_low_watermark_ratio is not None and self.mps_low_watermark_ratio <= 0.0:
            raise ValueError("mps_low_watermark_ratio must be > 0.0 or None")
        if self.compile_mode is not None and not self.compile_model:
            raise ValueError("compile_mode requires compile_model=True")

        # Lightning treats integer batch limits and fractional batch limits
        # differently, so we accept both types explicitly and validate each on
        # its own terms.
        for name, value in (
            ("limit_train_batches", self.limit_train_batches),
            ("limit_val_batches", self.limit_val_batches),
            ("limit_test_batches", self.limit_test_batches),
        ):
            if isinstance(value, float):
                if value <= 0.0:
                    raise ValueError(f"{name} must be > 0.0")
            elif isinstance(value, int):
                if value <= 0:
                    raise ValueError(f"{name} must be > 0")
            else:
                raise ValueError(f"{name} must be an int or float")

        # Negative patience would not make semantic sense for early stopping.
        if self.early_stopping_patience is not None and self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0 or None")
