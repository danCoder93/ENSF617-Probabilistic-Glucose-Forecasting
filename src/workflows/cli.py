from __future__ import annotations

# AI-assisted maintenance note:
# This module is the command-line layer for the repository's main workflow.
#
# Design intent:
# - keep `argparse` and user-facing print behavior separate from the reusable
#   Python-callable workflows
# - turn flat CLI flags into the structured config objects used elsewhere
# - preserve one stable entrypoint surface while allowing the underlying
#   workflow implementation to live in smaller modules

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

from config import Config, ObservabilityConfig, SnapshotConfig, TrainConfig
from defaults import (
    DEFAULT_AZT1D_URL,
    DEFAULT_OUTPUT_DIR,
    ROOT_DIR,
    build_default_config,
    build_default_observability_config,
    build_default_snapshot_config,
    build_default_train_config,
)
from environment import (
    DEVICE_PROFILE_CHOICES,
    DeviceProfileResolution,
    RuntimeDiagnostic,
    RuntimeEnvironment,
    collect_runtime_diagnostics,
    detect_runtime_environment,
    has_error_diagnostics,
    resolve_device_profile,
)
from workflows.helpers import (
    _apply_early_apple_silicon_environment_defaults,
    _collect_explicit_cli_overrides,
    _json_ready,
    _normalize_optional_string,
    _parse_csv_floats,
    _parse_csv_ints,
    _parse_devices,
    _parse_limit,
    _print_runtime_diagnostics,
)
from workflows.training import (
    run_environment_benchmark_workflow,
    run_training_workflow,
)
from workflows.types import EnvironmentBenchmarkArtifacts, MainRunArtifacts


@dataclass(frozen=True)
class _CliResolvedConfiguration:
    """
    Structured configuration bundle derived from parsed CLI flags.

    Purpose:
    keep the `main(...)` function focused on control flow rather than on
    carrying many parallel config objects and runtime-resolution outputs.

    Context:
    the CLI path has to carry both the typed config objects and the
    environment/profile-resolution metadata derived from them. Grouping that
    handoff state into one dataclass keeps `main(...)` readable.
    """

    # Typed configuration objects that feed the reusable workflow layer.
    config: Config
    train_config: TrainConfig
    snapshot_config: SnapshotConfig
    observability_config: ObservabilityConfig
    output_dir: Path

    # Runtime/profile information derived from the current machine plus the CLI.
    runtime_environment: RuntimeEnvironment
    profile_resolution: DeviceProfileResolution
    preflight_diagnostics: tuple[RuntimeDiagnostic, ...]


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register dataset and loader-shape CLI arguments on the parser.

    Purpose:
    keep the flat parser construction readable by grouping the data-oriented
    flags in one helper.

    Context:
    the CLI intentionally stays in one module, but splitting argument
    registration by concern keeps the entrypoint easier to maintain as the
    option surface grows.
    """

    # Dataset download URL.
    # Expected value: HTTP(S) URL string or empty-like value that will later be
    # normalized through `_normalize_optional_string(...)`.
    # Change this when switching to a different raw dataset source.
    parser.add_argument("--dataset-url", default=DEFAULT_AZT1D_URL)

    # Directory for the raw downloaded archive or source file.
    # Expected value: valid filesystem path.
    # Default is usually correct unless you want to relocate repository data.
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / "data" / "raw"))

    # Directory for cached intermediate download artifacts.
    # Expected value: valid filesystem path.
    # Useful when repeated local runs should avoid re-fetching unchanged files.
    parser.add_argument("--cache-dir", default=str(ROOT_DIR / "data" / "cache"))

    # Directory for extracted raw contents after unpacking.
    # Expected value: valid filesystem path.
    # Keep this separate from `raw-dir` and `processed-dir` to preserve a clean
    # pipeline boundary between download, extraction, and preprocessing.
    parser.add_argument("--extract-dir", default=str(ROOT_DIR / "data" / "extracted"))

    # Directory for processed model-ready tabular data.
    # Expected value: valid filesystem path.
    # The downstream workflow reports the resolved processed file path from this
    # directory during run summary printing.
    parser.add_argument(
        "--processed-dir",
        default=str(ROOT_DIR / "data" / "processed"),
    )

    # Output filename for the processed dataset inside `processed-dir`.
    # Expected value: filename string, typically ending in `.csv`.
    # Change this when producing alternate processed variants side-by-side.
    parser.add_argument("--processed-file-name", default="azt1d_processed.csv")

    # Root directory for run outputs such as summaries, reports, logs, and
    # possibly predictions.
    # Expected value: valid filesystem path.
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    # Optional manual checkpoint directory override.
    # Expected value: valid filesystem path or omitted/None.
    # When omitted, snapshot defaults decide where checkpoints should live.
    parser.add_argument("--checkpoint-dir", default=None)

    # Historical context length given to the forecasting model.
    # Expected value: positive integer.
    # In practice this should be > 0 and normally larger than
    # `prediction-length` so the model sees enough history.
    parser.add_argument("--encoder-length", type=int, default=168)

    # Forecast horizon length.
    # Expected value: positive integer.
    # In practice this should be > 0 and represent the number of future steps
    # the model is trained to predict per example window.
    parser.add_argument("--prediction-length", type=int, default=12)

    # Mini-batch size used by the data loader.
    # Expected value: positive integer.
    # Larger values may improve throughput but increase memory pressure.
    parser.add_argument("--batch-size", type=int, default=64)

    # Number of worker subprocesses for data loading.
    # Expected value: integer >= 0.
    # `0` keeps loading in the main process, which is often simpler and more
    # stable for debugging or smaller local runs.
    parser.add_argument("--num-workers", type=int, default=0)

    # Number of prefetched batches per worker.
    # Expected value: integer >= 1 or None.
    # This only matters when worker processes are enabled and can improve input
    # pipeline throughput at the cost of extra memory.
    parser.add_argument("--prefetch-factor", type=int, default=None)

    # Whether the data loader should pin host memory for faster host-to-device
    # transfer on some accelerators.
    # Expected values: explicit `--pin-memory` or `--no-pin-memory`.
    # Default remains `None` here so later config assembly can distinguish
    # "user did not specify" from an explicit True/False override.
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Whether worker subprocesses should stay alive across epochs.
    # Expected values: explicit `--persistent-workers` or
    # `--no-persistent-workers`.
    # Usually only meaningful when `num-workers > 0`.
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register Trainer/runtime-related CLI arguments on the parser.

    Purpose:
    collect device, precision, compilation, and loop-control flags in one
    focused section.

    Context:
    these options are the most environment-sensitive knobs in the entrypoint,
    so grouping them makes the relationship to device-profile resolution more
    obvious during maintenance.
    """

    # Number of training epochs.
    # Expected value: positive integer.
    # Increase for fuller convergence; reduce for smoke tests and iteration.
    parser.add_argument("--max-epochs", type=int, default=20)

    # High-level runtime preset for the current machine class.
    # Expected value: one of the repository-defined device profile choices.
    # This is later resolved against the actual detected runtime environment,
    # not blindly applied as a fixed hard-coded configuration.
    parser.add_argument(
        "--device-profile",
        default="auto",
        choices=DEVICE_PROFILE_CHOICES,
    )

    # Low-level Lightning accelerator request.
    # Expected values often include strings like `auto`, `cpu`, `gpu`, or
    # backend-specific values depending on Lightning support.
    # Leave as `auto` unless you need to force a specific execution path.
    parser.add_argument("--accelerator", default="auto")

    # Device selection for the Trainer.
    # Expected value: string that is later parsed by `_parse_devices(...)`.
    # Common forms are `auto`, integer-like strings, or accelerator-specific
    # device lists depending on downstream support.
    parser.add_argument("--devices", default="auto")

    # Numerical precision mode.
    # Expected value: either integer-like strings such as `32` or Lightning
    # precision mode strings like `16-mixed`.
    # `_build_cli_configuration(...)` preserves either an `int` or `str`
    # depending on what the user passed.
    parser.add_argument("--precision", default="32")

    # Gradient clipping threshold.
    # Expected value: float >= 0 or None.
    # Use this to reduce gradient explosions; leave unset to disable clipping.
    parser.add_argument("--gradient-clip-val", type=float, default=None)

    # Gradient accumulation factor.
    # Expected value: integer >= 1.
    # Useful when effective batch size should be larger than what device memory
    # allows in a single optimizer step.
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)

    # Distributed or execution strategy name.
    # Expected value: strategy string supported by Lightning or `auto`.
    # Most single-device local runs should keep the default.
    parser.add_argument("--strategy", default="auto")

    # Whether synchronized batch normalization should be enabled.
    # Expected values: explicit `--sync-batchnorm` or `--no-sync-batchnorm`.
    # Mostly relevant for multi-device distributed training.
    parser.add_argument(
        "--sync-batchnorm",
        dest="sync_batchnorm",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Matrix-multiplication precision hint.
    # Expected value: backend-supported string or None.
    # Useful for performance tuning on hardware where matmul precision tradeoffs
    # matter.
    parser.add_argument("--matmul-precision", default=None)

    # Whether TensorFloat-32 is allowed on supported CUDA hardware.
    # Expected values: explicit `--allow-tf32` or `--no-allow-tf32`.
    # Can improve throughput with a possible numerical tradeoff.
    parser.add_argument(
        "--allow-tf32",
        dest="allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Whether cuDNN benchmark mode is enabled.
    # Expected values: explicit `--cudnn-benchmark` or
    # `--no-cudnn-benchmark`.
    # Can improve speed for stable input shapes, but may reduce determinism.
    parser.add_argument(
        "--cudnn-benchmark",
        dest="cudnn_benchmark",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Torch intra-op thread count.
    # Expected value: integer >= 1 or None.
    # This controls parallelism inside a single operation on CPU backends.
    parser.add_argument("--intraop-threads", type=int, default=None)

    # Torch inter-op thread count.
    # Expected value: integer >= 1 or None.
    # This controls parallelism across independent operations on CPU backends.
    parser.add_argument("--interop-threads", type=int, default=None)

    # MPS allocator high watermark ratio.
    # Expected value: positive float or None.
    # Apple Silicon specific tuning knob; usually leave unset unless debugging
    # memory pressure or allocator behavior.
    parser.add_argument("--mps-high-watermark-ratio", type=float, default=None)

    # MPS allocator low watermark ratio.
    # Expected value: positive float or None.
    # Apple Silicon specific tuning knob paired with the high watermark.
    parser.add_argument("--mps-low-watermark-ratio", type=float, default=None)

    # Whether unsupported MPS ops may fall back to CPU.
    # Expected values: explicit `--enable-mps-fallback` or
    # `--no-enable-mps-fallback`.
    # This is useful on Apple Silicon when some kernels are missing on MPS.
    parser.add_argument(
        "--enable-mps-fallback",
        dest="enable_mps_fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Whether Torch compile should be enabled.
    # Expected values: explicit `--compile-model` or `--no-compile-model`.
    # Compilation can improve speed in some environments, but may increase
    # startup time or trigger backend-specific issues.
    parser.add_argument(
        "--compile-model",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Torch compile mode.
    # Expected value: backend-supported mode string or None.
    # Only meaningful when compile is enabled.
    parser.add_argument("--compile-mode", default=None)

    # Whether compile should request full-graph capture.
    # Expected value: flag presence only.
    # This is more restrictive and may fail on dynamic code paths.
    parser.add_argument("--compile-fullgraph", action="store_true")

    # Optimizer learning rate.
    # Expected value: positive float.
    # Small changes here often matter more than many architecture tweaks.
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    # Optimizer weight decay coefficient.
    # Expected value: float >= 0.
    # Set to zero to disable weight decay regularization.
    parser.add_argument("--weight-decay", type=float, default=0.0)

    # Optimizer selection name.
    # Expected value: optimizer key understood by the downstream training
    # workflow.
    parser.add_argument("--optimizer", default="adam")

    # Global random seed for reproducibility-related setup.
    # Expected value: integer.
    parser.add_argument("--seed", type=int, default=42)

    # Training batch limit.
    # Expected value: integer-like or float-like string.
    # Parsed later by `_parse_limit(...)` because Lightning supports both whole
    # batch counts and fractional dataset limits.
    parser.add_argument("--limit-train-batches", default="1.0")

    # Validation batch limit.
    # Expected value: integer-like or float-like string.
    parser.add_argument("--limit-val-batches", default="1.0")

    # Test batch limit.
    # Expected value: integer-like or float-like string.
    parser.add_argument("--limit-test-batches", default="1.0")

    # Early stopping patience measured in validation checks/epochs depending on
    # downstream callback behavior.
    # Expected value: integer >= 0.
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    # Optional checkpoint path to resume or initialize the fit phase from.
    # Expected value: path string or None.
    parser.add_argument("--fit-ckpt-path", default=None)

    # Checkpoint selection for evaluation/testing.
    # Expected value: path string, `best`, or None-like value depending on
    # downstream workflow handling.
    parser.add_argument("--eval-ckpt-path", default="best")

    # Whether deterministic execution should be requested.
    # Expected value: flag presence only.
    # This may improve reproducibility at some performance cost.
    parser.add_argument("--deterministic", action="store_true")

    # Lightning fast-dev-run toggle.
    # Expected value: flag presence only.
    # Useful for smoke tests because it runs a very short end-to-end pass.
    parser.add_argument("--fast-dev-run", action="store_true")

    # Whether the standard progress bar should be enabled.
    # Expected values: explicit `--progress-bar` or `--no-progress-bar`.
    # Default remains tri-state so profile or config logic can decide when the
    # user does not specify it.
    parser.add_argument(
        "--progress-bar",
        dest="enable_progress_bar",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Whether the richer progress bar variant should be enabled.
    # Expected values: explicit `--rich-progress-bar` or
    # `--no-rich-progress-bar`.
    parser.add_argument(
        "--rich-progress-bar",
        dest="enable_rich_progress_bar",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Whether device statistics collection/reporting should be enabled.
    # Expected values: explicit `--device-stats` or `--no-device-stats`.
    parser.add_argument(
        "--device-stats",
        dest="enable_device_stats",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Whether preflight/runtime diagnostic errors should abort the run.
    # Expected values: explicit `--fail-on-preflight-errors` or
    # `--no-fail-on-preflight-errors`.
    # This is passed straight through to the workflow entrypoint later.
    parser.add_argument(
        "--fail-on-preflight-errors",
        dest="fail_on_preflight_errors",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register top-level model-shape CLI arguments on the parser.

    Purpose:
    expose the small set of architecture controls that are commonly adjusted
    in local experiments without surfacing every nested config field.
    """

    # TCN hidden channel sizes as a comma-separated list.
    # Expected value: CSV integer string such as `64,64,128`.
    # Parsed later by `_parse_csv_ints(...)`.
    parser.add_argument("--tcn-channels", default="64,64,128")

    # TCN dilation schedule as a comma-separated list.
    # Expected value: CSV integer string such as `1,2,4`.
    # Values should normally be positive integers.
    parser.add_argument("--tcn-dilations", default="1,2,4")

    # TCN convolution kernel size.
    # Expected value: positive integer, typically small.
    # Very small kernels are common for temporal convolutions.
    parser.add_argument("--tcn-kernel-size", type=int, default=3)

    # TFT hidden representation size.
    # Expected value: positive integer.
    # This affects model capacity and memory cost.
    parser.add_argument("--tft-hidden-size", type=int, default=128)

    # Number of attention heads in the TFT block.
    # Expected value: positive integer.
    # In practice this should divide compatible internal dimensions expected by
    # the downstream TFT implementation.
    parser.add_argument("--tft-n-head", type=int, default=4)

    # Quantile levels for probabilistic forecasting.
    # Expected value: CSV float string such as `0.1,0.5,0.9`.
    # In practice each value should lie strictly between 0 and 1 and the list
    # is usually ordered from low to high quantile.
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")


def _add_behavior_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register high-level workflow-behavior toggles on the parser.

    Purpose:
    keep the operational switches for diagnostics, benchmarking, checkpoints,
    and held-out evaluation grouped together.
    """

    # Whether processed data should be regenerated even if an existing processed
    # artifact is already present.
    # Expected value: flag presence only.
    # Use this when preprocessing logic changes or cached processed outputs are
    # suspected to be stale.
    parser.add_argument("--rebuild-processed", action="store_true")

    # Whether the raw dataset should be downloaded again even if a cached copy
    # already exists.
    # Expected value: flag presence only.
    # Use this when cached raw artifacts are missing, corrupted, or intentionally
    # being refreshed.
    parser.add_argument("--redownload", action="store_true")

    # Whether to run only the environment benchmark workflow and skip the full
    # training/evaluation pipeline.
    # Expected value: flag presence only.
    parser.add_argument("--run-benchmark-only", action="store_true")

    # Number of train batches used during benchmark-only mode.
    # Expected value: positive integer.
    # Keep this relatively small so the benchmark remains quick and focused on
    # environment characterization rather than full model convergence.
    parser.add_argument("--benchmark-train-batches", type=int, default=10)

    # Whether to run only runtime diagnostics and then exit before training.
    # Expected value: flag presence only.
    # Useful for checking environment/profile resolution without paying the cost
    # of a full workflow run.
    parser.add_argument("--run-diagnostics-only", action="store_true")

    # Whether to skip the held-out test phase after fitting.
    # Expected value: flag presence only.
    # Useful for quicker exploratory runs where only fitting behavior matters.
    parser.add_argument("--skip-test", action="store_true")

    # Whether to skip prediction export/generation after fit/test.
    # Expected value: flag presence only.
    # Useful when metrics are enough and prediction artifacts are not needed.
    parser.add_argument("--skip-predict", action="store_true")

    # Whether serialized prediction outputs should be suppressed.
    # Expected value: flag presence only.
    # The workflow still may run prediction-related logic unless it is also
    # skipped; this specifically controls saving prediction artifacts.
    parser.add_argument("--no-save-predictions", action="store_true")

    # Whether checkpoint saving should be disabled entirely.
    # Expected value: flag presence only.
    # Useful for quick local experiments where disk artifacts are unnecessary.
    parser.add_argument("--disable-checkpoints", action="store_true")

    # Whether checkpoints should store weights only rather than fuller training
    # state where supported.
    # Expected value: flag presence only.
    # This usually reduces artifact size but may limit resume behavior.
    parser.add_argument("--save-weights-only", action="store_true")


def _add_observability_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register observability and debugging CLI arguments on the parser.

    Purpose:
    gather logging, telemetry, report, and visualization switches in one
    clearly observability-focused section.

    Dashboard-first context:
    The repository now organizes TensorBoard and reporting outputs into
    distinct conceptual layers:

    - dashboard/* : small, front-door summary metrics and health signals
    - text/*      : interpretation and orientation panels
    - debug/*     : detailed drill-down diagnostics (gradients, params, etc.)
    - system/*    : runtime/device telemetry

    The CLI flags below still control the same underlying features, but those
    features are now surfaced in a more structured way inside TensorBoard.
    """

    # Overall observability preset or mode name.
    # Expected value: repository-supported mode string.
    # This acts as a high-level observability baseline before individual feature
    # toggles are applied.
    parser.add_argument("--observability-mode", default="baseline")

    # Whether TensorBoard logging should be disabled.
    # Expected value: flag presence only.
    #
    # When enabled, TensorBoard now contains multiple structured surfaces:
    # - dashboard summaries (front-door view)
    # - text panels (model + report interpretation)
    # - debug metrics (gradients, parameters, activations)
    # - system telemetry (CPU/GPU/memory/runtime signals)
    parser.add_argument("--disable-tensorboard", action="store_true")

    # Whether generated plot/report artifacts should be disabled.
    # Expected value: flag presence only.
    parser.add_argument("--disable-plot-reports", action="store_true")

    # Whether system telemetry collection should be disabled.
    # Expected value: flag presence only.
    #
    # These signals are logged under `system/*` and include runtime/device
    # metrics such as memory usage and execution behavior. Disable this if
    # telemetry overhead or artifact volume is undesirable.
    parser.add_argument("--disable-system-telemetry", action="store_true")

    # Whether gradient statistics collection should be disabled.
    # Expected value: flag presence only.
    #
    # Gradient statistics are logged under `debug/gradients/*` and provide
    # drill-down visibility into optimization health. These are not part of
    # the front-door dashboard view.
    # Gradient stats are useful for debugging optimization behavior but may add
    # overhead and extra artifacts.
    parser.add_argument("--disable-gradient-stats", action="store_true")

    # Whether activation statistics should be enabled.
    # Expected value: flag presence only.
    #
    # Activation summaries are logged under `debug/activations/*` and are
    # useful for deep debugging of internal model behavior. These are
    # higher-cost, drill-down diagnostics.
    # This is useful for deeper debugging of internal model behavior and can add
    # runtime and storage overhead.
    parser.add_argument("--enable-activation-stats", action="store_true")

    # Whether parameter histogram logging should be disabled.
    #
    # Histograms are logged under `debug/parameters/*` and provide detailed
    # distribution views of weights and gradients. These are drill-down
    # diagnostics rather than dashboard summaries.
    # Expected value: flag presence only.
    parser.add_argument("--disable-parameter-histograms", action="store_true")

    # Whether parameter scalar logging should be disabled.
    #
    # Parameter scalars (e.g., norms) are logged under `debug/parameters/*`
    # and are intended for drill-down inspection rather than front-door
    # dashboard metrics.
    # Expected value: flag presence only.
    parser.add_argument("--disable-parameter-scalars", action="store_true")

    # Whether prediction figures should be disabled.
    #
    # Prediction figures provide qualitative forecast inspection and may
    # feed both dashboard-level summaries and saved report artifacts.
    # Expected value: flag presence only.
    # Useful when numeric outputs are enough and figure generation is not needed.
    parser.add_argument("--disable-prediction-figures", action="store_true")

    # Whether model graph export/logging should be disabled.
    #
    # These are structural model visualizations and are not part of the
    # dashboard summary. They complement deeper inspection workflows.
    # Expected value: flag presence only.
    parser.add_argument("--disable-model-graph", action="store_true")

    # Whether textual model summaries should be disabled.
    #
    # Model text summaries are typically logged under `text/model/*` and
    # provide orientation into architecture structure.
    # Expected value: flag presence only.
    parser.add_argument("--disable-model-text", action="store_true")

    # Whether torchview diagram generation should be disabled.
    #
    # Torchview outputs are treated as `debug/model/*` artifacts and provide
    # a static visualization of model structure for deeper inspection.
    # Expected value: flag presence only.
    # Torchview can be helpful for structure inspection but may be unnecessary in
    # fast iteration loops.
    parser.add_argument("--disable-torchview", action="store_true")

    # Maximum traversal depth used when generating torchview visualizations.
    # Expected value: positive integer.
    # Larger values reveal more nested detail but can produce much bigger graphs.
    parser.add_argument("--torchview-depth", type=int, default=4)

    # Whether a profiler should be enabled.
    # Expected value: flag presence only.
    # Profiling is useful for performance debugging and usually adds overhead.
    parser.add_argument("--enable-profiler", action="store_true")

    # Profiler backend/type selection.
    # Expected value: repository-supported profiler type string.
    parser.add_argument("--profiler-type", default="simple")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the repository's top-level CLI parser.

    Purpose:
    expose a practical command-line surface for common experiments while
    leaving deeper config detail inside the typed config layer.

    Context:
    this entrypoint intentionally does not mirror every internal dataclass
    field one-to-one. The parser focuses on the options that matter most for
    interactive local use.
    """

    # The CLI intentionally exposes the most common knobs for local
    # experimentation without trying to mirror every field from every nested
    # dataclass. If the project later needs a richer config surface, this can
    # evolve toward config-file support.
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the fused glucose forecasting model from one "
            "entrypoint."
        )
    )

    _add_data_arguments(parser)
    _add_runtime_arguments(parser)
    _add_model_arguments(parser)
    _add_behavior_arguments(parser)
    _add_observability_arguments(parser)

    # This parser intentionally does not try to expose every possible nested
    # config field. The goal is a practical top-level entrypoint, not a
    # one-to-one command-line mirror of every internal dataclass.
    return parser


def _build_cli_configuration(
    args: argparse.Namespace,
    *,
    explicit_overrides: set[str],
) -> _CliResolvedConfiguration:
    """
    Convert parsed CLI values into structured config and runtime-resolution state.

    Purpose:
    perform the one-time translation from flat CLI primitives into the typed
    config objects and profile-resolution outputs used by the reusable workflow
    layer.

    Context:
    separating this step from `main(...)` keeps the control-flow logic much
    easier to scan and makes the configuration assembly boundary explicit.

    Important inputs:
    - `args` is the raw `argparse` namespace for the current CLI invocation
    - `explicit_overrides` records which destinations the user explicitly set,
      so environment profiles can behave as defaults rather than forced
      overrides
    """

    # This is the point where flat command-line arguments become the structured
    # config objects used by the rest of the repository.
    output_dir = Path(args.output_dir)

    # Build the top-level data/model config first. This remains close to the
    # repository's default-config factory so that the CLI path and notebook path
    # still share the same baseline semantics where possible.
    config = build_default_config(
        dataset_url=_normalize_optional_string(args.dataset_url),
        raw_dir=Path(args.raw_dir),
        cache_dir=Path(args.cache_dir),
        extracted_dir=Path(args.extract_dir),
        processed_dir=Path(args.processed_dir),
        processed_file_name=args.processed_file_name,
        encoder_length=args.encoder_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=False if args.pin_memory is None else args.pin_memory,
        persistent_workers=(
            False if args.persistent_workers is None else args.persistent_workers
        ),
        rebuild_processed=args.rebuild_processed,
        redownload=args.redownload,
        tcn_channels=_parse_csv_ints(args.tcn_channels),
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_dilations=_parse_csv_ints(args.tcn_dilations),
        tft_hidden_size=args.tft_hidden_size,
        tft_n_head=args.tft_n_head,
        quantiles=_parse_csv_floats(args.quantiles),
    )

    # Lightning accepts both integer precision modes like `32` and string modes
    # like `16-mixed`, so we preserve whichever style the user intended.
    precision_value: int | str
    precision_value = int(args.precision) if args.precision.isdigit() else args.precision

    # Build the Trainer-oriented config separately from the data/model config.
    # This split mirrors the fact that many runtime knobs are environment-
    # sensitive and may be adjusted later by profile resolution.
    train_config = build_default_train_config(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        precision=precision_value,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy=args.strategy,
        sync_batchnorm=False if args.sync_batchnorm is None else args.sync_batchnorm,
        matmul_precision=args.matmul_precision,
        allow_tf32=args.allow_tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        intraop_threads=args.intraop_threads,
        interop_threads=args.interop_threads,
        mps_high_watermark_ratio=args.mps_high_watermark_ratio,
        mps_low_watermark_ratio=args.mps_low_watermark_ratio,
        enable_mps_fallback=args.enable_mps_fallback,
        compile_model=False if args.compile_model is None else args.compile_model,
        compile_mode=_normalize_optional_string(args.compile_mode),
        compile_fullgraph=args.compile_fullgraph,
        deterministic=args.deterministic,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=_parse_limit(args.limit_train_batches),
        limit_val_batches=_parse_limit(args.limit_val_batches),
        limit_test_batches=_parse_limit(args.limit_test_batches),
        enable_progress_bar=(
            True if args.enable_progress_bar is None else args.enable_progress_bar
        ),
        default_root_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
    )

    # Some Apple Silicon runtime defaults need to be applied before the broader
    # device-profile resolution step so the later environment logic starts from
    # a sane baseline for MPS-oriented local runs.
    _apply_early_apple_silicon_environment_defaults(
        requested_device_profile=args.device_profile,
        train_config=train_config,
    )

    # Snapshot and observability config are kept separate because they control
    # artifact production and logging/report behavior rather than core training
    # semantics.
    snapshot_config = build_default_snapshot_config(
        enabled=not args.disable_checkpoints,
        output_dir=output_dir,
        dirpath=None if args.checkpoint_dir is None else Path(args.checkpoint_dir),
        save_weights_only=args.save_weights_only,
    )

    observability_config = build_default_observability_config(
        mode=args.observability_mode,
        output_dir=output_dir,
        enable_tensorboard=not args.disable_tensorboard,
        enable_device_stats=(
            True if args.enable_device_stats is None else args.enable_device_stats
        ),
        enable_rich_progress_bar=(
            True
            if args.enable_rich_progress_bar is None
            else args.enable_rich_progress_bar
        ),
        enable_system_telemetry=not args.disable_system_telemetry,
        enable_gradient_stats=not args.disable_gradient_stats,
        enable_activation_stats=args.enable_activation_stats,
        enable_parameter_histograms=not args.disable_parameter_histograms,
        enable_parameter_scalars=not args.disable_parameter_scalars,
        enable_prediction_figures=not args.disable_prediction_figures,
        enable_model_graph=not args.disable_model_graph,
        enable_model_text=not args.disable_model_text,
        enable_torchview=not args.disable_torchview,
        enable_profiler=args.enable_profiler,
        enable_plot_reports=not args.disable_plot_reports,
        profiler_type=args.profiler_type,
        torchview_depth=args.torchview_depth,
    )

    # Detect the actual runtime environment of the current machine, then
    # resolve the requested profile against:
    # - the machine we are actually on
    # - the current train/data/observability config
    # - which CLI values were explicitly set by the user
    #
    # The explicit override set is important because profile defaults should
    # only fill in gaps; they should not silently overwrite user intent.
    runtime_environment = detect_runtime_environment()

    profile_resolution = resolve_device_profile(
        requested_profile=args.device_profile,
        environment=runtime_environment,
        train_config=train_config,
        data_config=config.data,
        observability_config=observability_config,
        explicit_overrides=explicit_overrides,
    )

    # Build the final resolved config bundle that will be passed into the
    # workflow layer. Data config may change during profile resolution, while
    # the current TCN/TFT sub-configs remain the ones assembled above.
    resolved_config = Config(
        data=profile_resolution.data_config,
        tft=config.tft,
        tcn=config.tcn,
    )
    resolved_train_config = profile_resolution.train_config
    resolved_observability_config = profile_resolution.observability_config

    # Collect preflight diagnostics after profile resolution so the diagnostic
    # messages reflect the final effective settings rather than the raw CLI
    # request alone.
    preflight_diagnostics = tuple(
        collect_runtime_diagnostics(
            requested_profile=profile_resolution.requested_profile,
            resolved_profile=profile_resolution.resolved_profile,
            environment=runtime_environment,
            train_config=resolved_train_config,
            data_config=resolved_config.data,
            observability_config=resolved_observability_config,
        )
    )

    return _CliResolvedConfiguration(
        config=resolved_config,
        train_config=resolved_train_config,
        snapshot_config=snapshot_config,
        observability_config=resolved_observability_config,
        output_dir=output_dir,
        runtime_environment=runtime_environment,
        profile_resolution=profile_resolution,
        preflight_diagnostics=preflight_diagnostics,
    )


def _print_benchmark_artifacts(
    benchmark_artifacts: EnvironmentBenchmarkArtifacts,
) -> None:
    """
    Print the user-facing summary for benchmark-only runs.

    Purpose:
    keep benchmark CLI presentation consistent and separate from the heavier
    main workflow control flow.
    """

    print("Environment benchmark summary:")
    print(json.dumps(_json_ready(benchmark_artifacts.summary), indent=2))

    if benchmark_artifacts.summary_path is not None:
        print(f"Saved benchmark summary to: {benchmark_artifacts.summary_path}")


def _print_run_artifacts(
    artifacts: MainRunArtifacts,
    *,
    config: Config,
) -> None:
    """
    Print the concise user-facing summary for a full workflow run.

    Purpose:
    surface the most useful artifact locations and high-level outcomes without
    requiring the caller to open the JSON summary file immediately.

    Context:
    the detailed state already lives in `run_summary.json`; this printer is a
    compact terminal companion rather than the canonical source of record.
    """

    # The printed summary is intentionally concise. Detailed state is written
    # to the JSON summary file and, when requested, the raw prediction tensor
    # file.
    print(f"Processed data: {config.data.processed_file_path}")
    print(
        "Resolved environment: "
        f"{artifacts.runtime_environment.system} "
        f"{artifacts.runtime_environment.machine}"
    )
    print(f"Validation windows available: {artifacts.fit.has_validation_data}")
    print(f"Test windows available: {artifacts.fit.has_test_data}")
    print(
        "Disclaimer: this run uses repository defaults unless you override them, "
        "and the resulting forecasts are for research workflows rather than "
        "clinical decision-making."
    )

    if artifacts.fit.best_checkpoint_path:
        print(f"Best checkpoint: {artifacts.fit.best_checkpoint_path}")

    if artifacts.test_metrics is not None:
        print("Test metrics:")
        print(json.dumps(_json_ready(artifacts.test_metrics), indent=2))

    if artifacts.predictions_path is not None:
        print(f"Saved test predictions to: {artifacts.predictions_path}")

    if artifacts.prediction_table_path is not None:
        print(f"Saved prediction table to: {artifacts.prediction_table_path}")

    if artifacts.report_paths:
        print("Saved Plotly reports:")
        print(json.dumps(_json_ready(artifacts.report_paths), indent=2))

    if artifacts.telemetry_path is not None:
        print(f"Saved telemetry log to: {artifacts.telemetry_path}")

    if artifacts.text_log_path is not None:
        print(f"Saved text log to: {artifacts.text_log_path}")

    # A few observability outputs currently live inside the summary structure
    # instead of as direct top-level artifact fields, so we surface them here
    # from the summary dictionary for convenience.
    profiler_path = artifacts.summary.get("observability", {}).get("profiler_path")
    if profiler_path is not None:
        print(f"Profiler output path: {profiler_path}")

    torchview_path = artifacts.summary.get("observability_config", {}).get(
        "torchview_path"
    )
    if torchview_path is not None:
        print(f"Torchview output base path: {torchview_path}")

    if artifacts.summary_path is not None:
        print(f"Saved run summary to: {artifacts.summary_path}")


def main(
    argv: Sequence[str] | None = None,
) -> MainRunArtifacts | EnvironmentBenchmarkArtifacts | None:
    """
    Parse CLI arguments and dispatch the requested top-level workflow.

    Purpose:
    preserve `python main.py ...` as the stable entrypoint while delegating the
    heavier orchestration to the reusable workflow modules.

    Context:
    tests and notebooks call the lower-level workflow functions directly, but
    the CLI still needs one place that owns argument parsing, diagnostic-only
    handling, and terminal-oriented output.

    Important input:
    `argv` is optional so tests can drive the CLI deterministically without
    mutating global process arguments.
    """

    # `main(...)` keeps argument parsing separate from the reusable workflow
    # functions. That split makes the same training flow callable from tests
    # and notebooks without forcing everything through `argparse`.
    parser = build_argument_parser()

    # Preserve the ability for tests to inject argv explicitly while normal CLI
    # usage still reads from the process command line.
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)

    # Collect explicit CLI overrides before config/profile resolution so the
    # resolver can tell which values came from the user versus which ones are
    # still eligible for profile-provided defaults.
    explicit_overrides = _collect_explicit_cli_overrides(parser, raw_argv)

    resolved = _build_cli_configuration(
        args,
        explicit_overrides=explicit_overrides,
    )

    # At this point we have three layers of structured configuration:
    # - `config` for data/model semantics
    # - `train_config` for Trainer behavior
    # - `snapshot_config` for checkpoint policy
    # - `observability_config` for logging, telemetry, visualization, and
    #   debug/report artifacts

    # Diagnostics are printed either when:
    # - the user explicitly requested diagnostics-only mode, or
    # - the preflight checks contain no errors
    #
    # The main reason for this guard is to avoid printing a misleading
    # "everything looks fine" diagnostic banner when preflight has already
    # found blocking issues that will be surfaced through the workflow path.
    if args.run_diagnostics_only or not has_error_diagnostics(
        resolved.preflight_diagnostics
    ):
        print(
            "Device profile: "
            f"{resolved.profile_resolution.requested_profile} -> "
            f"{resolved.profile_resolution.resolved_profile}"
        )
        _print_runtime_diagnostics(resolved.preflight_diagnostics)

    if args.run_diagnostics_only:
        print("Diagnostics-only mode enabled; training was not started.")
        return None

    if args.run_benchmark_only:
        # Benchmark-only mode exercises a short training-oriented workload to
        # characterize the current environment without running the full fit/test
        # pipeline.
        benchmark_artifacts = run_environment_benchmark_workflow(
            resolved.config,
            train_config=resolved.train_config,
            snapshot_config=resolved.snapshot_config,
            observability_config=resolved.observability_config,
            requested_device_profile=resolved.profile_resolution.requested_profile,
            resolved_device_profile=resolved.profile_resolution.resolved_profile,
            applied_profile_defaults=resolved.profile_resolution.applied_defaults,
            runtime_environment=resolved.runtime_environment,
            preflight_diagnostics=resolved.preflight_diagnostics,
            output_dir=resolved.output_dir,
            benchmark_train_batches=args.benchmark_train_batches,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
        )
        _print_benchmark_artifacts(benchmark_artifacts)
        return benchmark_artifacts

    try:
        # Full workflow mode executes fit, optional evaluation, optional
        # prediction export, and artifact/report generation from one entrypoint.
        artifacts = run_training_workflow(
            resolved.config,
            train_config=resolved.train_config,
            snapshot_config=resolved.snapshot_config,
            observability_config=resolved.observability_config,
            requested_device_profile=resolved.profile_resolution.requested_profile,
            resolved_device_profile=resolved.profile_resolution.resolved_profile,
            applied_profile_defaults=resolved.profile_resolution.applied_defaults,
            runtime_environment=resolved.runtime_environment,
            preflight_diagnostics=resolved.preflight_diagnostics,
            fail_on_preflight_errors=(
                True
                if args.fail_on_preflight_errors is None
                else args.fail_on_preflight_errors
            ),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            fit_ckpt_path=_normalize_optional_string(args.fit_ckpt_path),
            eval_ckpt_path=_normalize_optional_string(args.eval_ckpt_path),
            output_dir=resolved.output_dir,
            seed=args.seed,
            skip_test=args.skip_test,
            skip_predict=args.skip_predict,
            save_predictions=not args.no_save_predictions,
        )
    except RuntimeError as exc:
        message = str(exc)

        # Preserve the current behavior for workflow/preflight failures:
        # print the user-facing message to stderr and re-raise so upstream
        # callers or tests still see the original exception semantics.
        if (
            "Runtime preflight checks failed " in message
            or "Training workflow failed during " in message
        ):
            print(message, file=sys.stderr)
            raise
            # raise SystemExit(1) from exc

        raise

    # The returned artifact bundle is the same shape notebooks and tests
    # receive from `run_training_workflow(...)`, so CLI and non-CLI entrypoints
    # share one result contract.
    _print_run_artifacts(artifacts, config=resolved.config)
    return artifacts