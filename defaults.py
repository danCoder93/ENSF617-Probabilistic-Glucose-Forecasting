from __future__ import annotations

# AI-assisted implementation note:
# This module centralizes the repository's "starter" runtime configuration for
# the top-level script and notebook entrypoints.
#
# Context:
# - `main.py` and `main.ipynb` should share one source of truth for baseline
#   data/model/training settings.
# - keeping those defaults in one place reduces drift between the CLI and the
#   notebook workflow
# - callers can still override any value, but they start from the same coherent
#   baseline rather than copying config code into multiple surfaces
#
# Important disclaimer:
# - these defaults are convenience-oriented research defaults, not claims of
#   optimal performance
# - they are intended to make the repository runnable end to end with minimal
#   setup, but real experiments may require tuning batch size, epochs, model
#   width, split policy, and checkpointing behavior
# - this repository is a forecasting research codebase and should not be treated
#   as a clinically validated decision-support system

import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
# The project uses direct imports like `from data...` and `from config...`.
# Adding `src/` here means any consumer that imports `defaults.py` from the
# repository root gets the same import behavior as the tests and notebooks.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    Config,
    DataConfig,
    ObservabilityConfig,
    SnapshotConfig,
    TCNConfig,
    TFTConfig,
    TrainConfig,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "main_run"
# This URL points at the public AZT1D download used elsewhere in the repo.
# Keeping it here avoids hard-coding the same long string in both the script
# and the notebook.
DEFAULT_AZT1D_URL = (
    "https://data.mendeley.com/public-files/datasets/"
    "gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded"
)


def build_default_config(
    *,
    dataset_url: str | None = DEFAULT_AZT1D_URL,
    raw_dir: Path = ROOT_DIR / "data" / "raw",
    cache_dir: Path = ROOT_DIR / "data" / "cache",
    extracted_dir: Path = ROOT_DIR / "data" / "extracted",
    processed_dir: Path = ROOT_DIR / "data" / "processed",
    processed_file_name: str = "azt1d_processed.csv",
    encoder_length: int = 168,
    prediction_length: int = 12,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    rebuild_processed: bool = False,
    redownload: bool = False,
    tcn_channels: Sequence[int] = (64, 64, 128),
    tcn_kernel_size: int = 3,
    tcn_dilations: Sequence[int] = (1, 2, 4),
    tcn_dropout: float = 0.1,
    tft_hidden_size: int = 128,
    tft_n_head: int = 4,
    tft_dropout: float = 0.1,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> Config:
    # This builder intentionally returns a complete top-level config that is
    # "good enough to start" for the script and notebook entrypoints.
    #
    # It should be read as a baseline experiment recipe, not as the single
    # canonical or best-performing setup for every dataset slice or hardware
    # environment.
    # Data defaults are chosen to make the first local run predictable:
    # - repository-relative folders keep artifacts contained inside the repo
    # - `num_workers=0` is conservative but avoids multiprocessing surprises
    #   in notebooks and on macOS
    # - `rebuild_processed` / `redownload` default to False so repeated runs
    #   do not redo expensive work unless the caller asks for it
    data_config = DataConfig(
        dataset_url=dataset_url,
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        extracted_dir=extracted_dir,
        processed_dir=processed_dir,
        processed_file_name=processed_file_name,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        rebuild_processed=rebuild_processed,
        redownload=redownload,
    )
    # Model defaults intentionally stay modest rather than aggressive. They are
    # meant to create a valid end-to-end run, not to encode a final tuning
    # result for the project.
    tft_config = TFTConfig(
        hidden_size=tft_hidden_size,
        n_head=tft_n_head,
        dropout=tft_dropout,
        quantiles=tuple(quantiles),
        encoder_length=encoder_length,
        example_length=encoder_length + prediction_length,
    )
    # The TCN branch defaults mirror the current fused-model design: a small
    # multi-scale stack with three dilation stages and a single output channel
    # per forecast target.
    tcn_config = TCNConfig(
        num_inputs=1,
        num_channels=tuple(tcn_channels),
        kernel_size=tcn_kernel_size,
        dilations=tuple(tcn_dilations),
        dropout=tcn_dropout,
        prediction_length=prediction_length,
        output_size=1,
    )
    return Config(data=data_config, tft=tft_config, tcn=tcn_config)


def build_default_train_config(
    *,
    max_epochs: int = 20,
    accelerator: str = "auto",
    devices: str | int | list[int] = "auto",
    precision: int | str = 32,
    deterministic: bool = False,
    log_every_n_steps: int = 10,
    num_sanity_val_steps: int = 2,
    fast_dev_run: bool = False,
    limit_train_batches: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
    limit_test_batches: int | float = 1.0,
    enable_progress_bar: bool = True,
    enable_model_summary: bool = True,
    default_root_dir: Path | None = DEFAULT_OUTPUT_DIR,
    early_stopping_patience: int | None = 5,
) -> TrainConfig:
    # Trainer defaults prioritize an approachable local workflow:
    # - `devices="auto"` lets Lightning choose CPU/GPU when available
    # - `num_workers=0` remains in the data config by default to avoid the
    #   common notebook/macOS worker pitfalls on first run
    # - modest epoch counts and early stopping keep first experiments from
    #   running indefinitely
    # - full-batch limits default to `1.0` so the entrypoints do not silently
    #   run in a debug-style truncated mode unless the user opts into that
    return TrainConfig(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        deterministic=deterministic,
        log_every_n_steps=log_every_n_steps,
        num_sanity_val_steps=num_sanity_val_steps,
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
        default_root_dir=default_root_dir,
        early_stopping_patience=early_stopping_patience,
    )


def build_default_snapshot_config(
    *,
    enabled: bool = True,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
    dirpath: Path | None = None,
    filename: str = "epoch={epoch:02d}-val_loss={val_loss:.4f}",
    monitor: str = "val_loss",
    mode: str = "min",
    save_top_k: int = 1,
    save_last: bool = True,
    save_weights_only: bool = False,
) -> SnapshotConfig:
    # Snapshotting is enabled by default because the top-level entrypoints are
    # meant to feel like full experiment surfaces rather than one-off scripts.
    #
    # The default directory is nested under the chosen output folder so a run's
    # summary, checkpoints, and predictions stay grouped together.
    #
    # We keep a "best" checkpoint plus "last" by default because they answer
    # two different questions:
    # - "what checkpoint scored best on validation?"
    # - "what was the final model state at the end of training?"
    checkpoint_dir = dirpath
    if checkpoint_dir is None and output_dir is not None:
        checkpoint_dir = output_dir / "checkpoints"
    return SnapshotConfig(
        enabled=enabled,
        dirpath=checkpoint_dir,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
        save_weights_only=save_weights_only,
    )


def build_default_observability_config(
    *,
    mode: str = "baseline",
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
    enable_tensorboard: bool = True,
    enable_text_logging: bool = True,
    enable_csv_fallback_logger: bool = True,
    enable_learning_rate_monitor: bool = True,
    enable_device_stats: bool = True,
    enable_rich_progress_bar: bool = True,
    enable_system_telemetry: bool = True,
    enable_parameter_histograms: bool = True,
    enable_parameter_scalars: bool = True,
    enable_prediction_figures: bool = True,
    enable_model_graph: bool = True,
    enable_model_text: bool = True,
    enable_torchview: bool = True,
    enable_profiler: bool = False,
    enable_gradient_stats: bool = True,
    enable_activation_stats: bool | None = None,
    enable_batch_audit: bool = True,
    enable_prediction_exports: bool = True,
    enable_plot_reports: bool = True,
    debug_every_n_steps: int = 10,
    telemetry_every_n_steps: int = 10,
    histogram_every_n_epochs: int = 1,
    parameter_scalar_every_n_epochs: int = 1,
    figure_every_n_epochs: int = 1,
    max_prediction_plots: int = 2,
    profiler_type: str = "simple",
    torchview_depth: int = 4,
    torchview_roll: bool = True,
    torchview_expand_nested: bool = True,
) -> ObservabilityConfig:
    # This helper intentionally defines a "strong default visibility" profile
    # for normal experiments:
    # - TensorBoard stays on by default because Lightning already integrates
    #   with it naturally for scalars, graphs, figures, and profiler outputs
    # - text logs and telemetry CSVs are enabled so a run still leaves behind
    #   readable artifacts outside the TensorBoard UI
    # - the heavier debugging hooks remain configurable, with activation hooks
    #   automatically promoted in debug/trace modes
    #
    # Important disclaimer:
    # - these defaults favor observability and debuggability over absolute
    #   minimal overhead
    # - callers running on very constrained Colab sessions may still want to
    #   reduce figure frequency, histogram frequency, or optional callbacks
    # Observability defaults are designed to give strong diagnostics without
    # making the baseline run path too heavy. Trace-level internals remain
    # opt-in because they can slow notebook and Colab runs substantially.
    if enable_activation_stats is None:
        enable_activation_stats = mode in {"debug", "trace"}

    log_dir = None if output_dir is None else output_dir / "logs"
    text_log_path = None if output_dir is None else output_dir / "run.log"
    telemetry_path = None if output_dir is None else output_dir / "telemetry.csv"
    prediction_table_path = (
        None if output_dir is None else output_dir / "test_predictions.csv"
    )
    report_dir = None if output_dir is None else output_dir / "reports"
    profiler_path = None if output_dir is None else output_dir / "profiler"
    torchview_path = None if output_dir is None else output_dir / "model_viz" / "fused_model"

    return ObservabilityConfig(
        mode=mode,
        enable_tensorboard=enable_tensorboard,
        enable_text_logging=enable_text_logging,
        enable_csv_fallback_logger=enable_csv_fallback_logger,
        enable_learning_rate_monitor=enable_learning_rate_monitor,
        enable_device_stats=enable_device_stats,
        enable_rich_progress_bar=enable_rich_progress_bar,
        enable_system_telemetry=enable_system_telemetry,
        enable_parameter_histograms=enable_parameter_histograms,
        enable_parameter_scalars=enable_parameter_scalars,
        enable_prediction_figures=enable_prediction_figures,
        enable_model_graph=enable_model_graph,
        enable_model_text=enable_model_text,
        enable_torchview=enable_torchview,
        enable_profiler=enable_profiler,
        enable_gradient_stats=enable_gradient_stats,
        enable_activation_stats=enable_activation_stats,
        enable_batch_audit=enable_batch_audit,
        enable_prediction_exports=enable_prediction_exports,
        enable_plot_reports=enable_plot_reports,
        log_dir=log_dir,
        text_log_path=text_log_path,
        telemetry_path=telemetry_path,
        prediction_table_path=prediction_table_path,
        report_dir=report_dir,
        profiler_path=profiler_path,
        torchview_path=torchview_path,
        debug_every_n_steps=debug_every_n_steps,
        telemetry_every_n_steps=telemetry_every_n_steps,
        histogram_every_n_epochs=histogram_every_n_epochs,
        parameter_scalar_every_n_epochs=parameter_scalar_every_n_epochs,
        figure_every_n_epochs=figure_every_n_epochs,
        max_prediction_plots=max_prediction_plots,
        profiler_type=profiler_type,
        torchview_depth=torchview_depth,
        torchview_roll=torchview_roll,
        torchview_expand_nested=torchview_expand_nested,
    )
