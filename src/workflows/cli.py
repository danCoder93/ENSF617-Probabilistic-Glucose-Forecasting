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
from typing import Any, Sequence

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
    # Group the flat parser construction into sections so the CLI shape stays
    # readable as the entrypoint grows.
    parser.add_argument("--dataset-url", default=DEFAULT_AZT1D_URL)
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / "data" / "raw"))
    parser.add_argument("--cache-dir", default=str(ROOT_DIR / "data" / "cache"))
    parser.add_argument("--extract-dir", default=str(ROOT_DIR / "data" / "extracted"))
    parser.add_argument("--processed-dir", default=str(ROOT_DIR / "data" / "processed"))
    parser.add_argument("--processed-file-name", default="azt1d_processed.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--encoder-length", type=int, default=168)
    parser.add_argument("--prediction-length", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument(
        "--device-profile",
        default="auto",
        choices=DEVICE_PROFILE_CHOICES,
    )
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--gradient-clip-val", type=float, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument(
        "--sync-batchnorm",
        dest="sync_batchnorm",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--matmul-precision", default=None)
    parser.add_argument(
        "--allow-tf32",
        dest="allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--cudnn-benchmark",
        dest="cudnn_benchmark",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--intraop-threads", type=int, default=None)
    parser.add_argument("--interop-threads", type=int, default=None)
    parser.add_argument("--mps-high-watermark-ratio", type=float, default=None)
    parser.add_argument("--mps-low-watermark-ratio", type=float, default=None)
    parser.add_argument(
        "--enable-mps-fallback",
        dest="enable_mps_fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compile-model",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-batches", default="1.0")
    parser.add_argument("--limit-val-batches", default="1.0")
    parser.add_argument("--limit-test-batches", default="1.0")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--fit-ckpt-path", default=None)
    parser.add_argument("--eval-ckpt-path", default="best")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument(
        "--progress-bar",
        dest="enable_progress_bar",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--rich-progress-bar",
        dest="enable_rich_progress_bar",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--device-stats",
        dest="enable_device_stats",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
    parser.add_argument("--tcn-channels", default="64,64,128")
    parser.add_argument("--tcn-dilations", default="1,2,4")
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--tft-hidden-size", type=int, default=128)
    parser.add_argument("--tft-n-head", type=int, default=4)
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")


def _add_behavior_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register high-level workflow-behavior toggles on the parser.

    Purpose:
    keep the operational switches for diagnostics, benchmarking, checkpoints,
    and held-out evaluation grouped together.
    """
    parser.add_argument("--rebuild-processed", action="store_true")
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--run-benchmark-only", action="store_true")
    parser.add_argument("--benchmark-train-batches", type=int, default=10)
    parser.add_argument("--run-diagnostics-only", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--disable-checkpoints", action="store_true")
    parser.add_argument("--save-weights-only", action="store_true")


def _add_observability_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register observability and debugging CLI arguments on the parser.

    Purpose:
    gather logging, telemetry, report, and visualization switches in one
    clearly observability-focused section.
    """
    parser.add_argument("--observability-mode", default="baseline")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--disable-plot-reports", action="store_true")
    parser.add_argument("--disable-system-telemetry", action="store_true")
    parser.add_argument("--disable-gradient-stats", action="store_true")
    parser.add_argument("--enable-activation-stats", action="store_true")
    parser.add_argument("--disable-parameter-histograms", action="store_true")
    parser.add_argument("--disable-parameter-scalars", action="store_true")
    parser.add_argument("--disable-prediction-figures", action="store_true")
    parser.add_argument("--disable-model-graph", action="store_true")
    parser.add_argument("--disable-model-text", action="store_true")
    parser.add_argument("--disable-torchview", action="store_true")
    parser.add_argument("--torchview-depth", type=int, default=4)
    parser.add_argument("--enable-profiler", action="store_true")
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
            "Train and evaluate the fused glucose forecasting model from one entrypoint."
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
    config objects and profile-resolution outputs used by the reusable
    workflow layer.

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

    precision_value: int | str
    # Lightning accepts both integer precision modes like `32` and string modes
    # like `16-mixed`, so we preserve whichever style the user intended.
    precision_value = int(args.precision) if args.precision.isdigit() else args.precision
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
    _apply_early_apple_silicon_environment_defaults(
        requested_device_profile=args.device_profile,
        train_config=train_config,
    )

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

    runtime_environment = detect_runtime_environment()
    profile_resolution = resolve_device_profile(
        requested_profile=args.device_profile,
        environment=runtime_environment,
        train_config=train_config,
        data_config=config.data,
        observability_config=observability_config,
        explicit_overrides=explicit_overrides,
    )
    resolved_config = Config(
        data=profile_resolution.data_config,
        tft=config.tft,
        tcn=config.tcn,
    )
    resolved_train_config = profile_resolution.train_config
    resolved_observability_config = profile_resolution.observability_config
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    explicit_overrides = _collect_explicit_cli_overrides(parser, raw_argv)
    resolved = _build_cli_configuration(args, explicit_overrides=explicit_overrides)

    # At this point we have three layers of structured configuration:
    # - `config` for data/model semantics
    # - `train_config` for Trainer behavior
    # - `snapshot_config` for checkpoint policy
    # - `observability_config` for logging, telemetry, visualization, and
    #   debug/report artifacts
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
        if (
            "Runtime preflight checks failed " in message
            or "Training workflow failed during " in message
        ):
            print(message, file=sys.stderr)
            raise SystemExit(1) from None
        raise

    # The returned artifact bundle is the same shape notebooks and tests
    # receive from `run_training_workflow(...)`, so CLI and non-CLI entrypoints
    # share one result contract.
    _print_run_artifacts(artifacts, config=resolved.config)
    return artifacts
