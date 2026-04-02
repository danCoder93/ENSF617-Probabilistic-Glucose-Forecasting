from __future__ import annotations

# AI-assisted maintenance note:
# This module owns the reusable top-level workflows that sit above
# `src/train.py`.
#
# Responsibility boundary:
# - translate a prepared project config into one full run
# - coordinate evaluation, prediction export, reports, and run summaries
# - keep CLI and notebook callers on one shared orchestration path
#
# What does *not* live here:
# - argument parsing
# - low-level Trainer assembly details
# - model forward/loss/optimizer behavior

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from config import (
    Config,
    ObservabilityConfig,
    SnapshotConfig,
    TrainConfig,
    config_to_dict,
)
from defaults import (
    DEFAULT_OUTPUT_DIR,
    build_default_observability_config,
    build_default_snapshot_config,
    build_default_train_config,
)
from environment import (
    RuntimeDiagnostic,
    RuntimeEnvironment,
    analyze_runtime_failure,
    apply_runtime_environment_overrides,
    collect_runtime_diagnostics,
    detect_runtime_environment,
    format_runtime_diagnostics,
    has_error_diagnostics,
    resolve_device_profile,
)
from workflows.helpers import (
    _apply_early_apple_silicon_environment_defaults,
    _json_ready,
    _resolve_eval_ckpt_path,
)
from workflows.types import EnvironmentBenchmarkArtifacts, MainRunArtifacts

if TYPE_CHECKING:
    import torch

    from evaluation import EvaluationResult
    from train import CheckpointSelection, FitArtifacts, FusedModelTrainer
else:
    CheckpointSelection = Any
    EvaluationResult = Any
    FitArtifacts = Any
    FusedModelTrainer = Any

try:
    from pytorch_lightning import seed_everything
except ImportError:  # pragma: no cover - import failure is surfaced at runtime
    # The rest of the workflow still imports Lightning-backed project modules,
    # so a missing install will fail later during real execution. Keeping the
    # fallback here simply lets static analysis and lightweight file parsing
    # succeed without pretending training can run without Lightning.
    seed_everything = None


def _build_run_summary(
    *,
    config: Config,
    train_config: TrainConfig,
    snapshot_config: SnapshotConfig,
    observability_config: ObservabilityConfig,
    requested_device_profile: str,
    resolved_device_profile: str,
    applied_profile_defaults: Mapping[str, Any],
    runtime_environment: RuntimeEnvironment,
    preflight_diagnostics: Sequence[RuntimeDiagnostic],
    fit_artifacts: FitArtifacts,
    eval_ckpt_path: CheckpointSelection,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    output_dir: Path | None,
    test_metrics: list[Mapping[str, float]] | None,
    test_evaluation: EvaluationResult | None,
    predictions: list[torch.Tensor] | None,
    predictions_path: Path | None,
    prediction_table_path: Path | None,
    report_paths: Mapping[str, Path],
    logger_dir: Path | None,
    text_log_path: Path | None,
    telemetry_path: Path | None,
    runtime_tuning: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the compact JSON-ready summary for one workflow execution.

    Purpose:
    capture the important run decisions and artifact locations in one stable
    dictionary that can be written to disk and inspected later.

    Context:
    this repository intentionally keeps run summaries lightweight rather than
    treating them as a full experiment-tracking system, so the summary focuses
    on reproducibility and artifact discovery rather than raw trainer internals.

    Important inputs:
    - the various `*_config` arguments preserve the run's declared policy
    - `fit_artifacts`, evaluation outputs, and artifact paths capture what the
      run actually produced
    - `runtime_tuning` records the environment-sensitive tweaks applied during
      execution so later readers can understand how defaults were adapted
    """
    # The summary is intentionally lightweight and human-readable:
    # enough to reproduce the broad run setup and locate the major artifacts,
    # but not a full experiment tracker replacement.
    #
    # We store the *declared* config and run decisions here rather than the
    # full model object or Trainer internals so the summary stays portable and
    # easy to inspect in plain text.
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "config": config_to_dict(config),
        "train_config": _json_ready(train_config),
        "snapshot_config": _json_ready(snapshot_config),
        "observability_config": _json_ready(observability_config),
        "optimizer": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer_name": optimizer_name,
        },
        "device_profile": {
            "requested": requested_device_profile,
            "resolved": resolved_device_profile,
            "applied_defaults": _json_ready(dict(applied_profile_defaults)),
        },
        "environment": {
            "runtime": _json_ready(runtime_environment),
            "preflight_diagnostics": _json_ready(tuple(preflight_diagnostics)),
            "runtime_tuning": _json_ready(dict(runtime_tuning or {})),
        },
        "fit": {
            "has_validation_data": fit_artifacts.has_validation_data,
            "has_test_data": fit_artifacts.has_test_data,
            "best_checkpoint_path": fit_artifacts.best_checkpoint_path or None,
        },
        "evaluation": {
            "resolved_eval_ckpt_path": str(eval_ckpt_path)
            if eval_ckpt_path not in (None, "best", "last")
            else eval_ckpt_path,
            "test_metrics": _json_ready(test_metrics),
            "test_evaluation": _json_ready(test_evaluation),
        },
        "predictions": {
            "num_batches": 0 if predictions is None else len(predictions),
            "batch_shapes": []
            if predictions is None
            else [list(tensor.shape) for tensor in predictions],
            "saved_to": str(predictions_path) if predictions_path is not None else None,
            "table_path": (
                str(prediction_table_path)
                if prediction_table_path is not None
                else None
            ),
        },
        "observability": {
            "logger_dir": str(logger_dir) if logger_dir is not None else None,
            "text_log_path": str(text_log_path) if text_log_path is not None else None,
            "telemetry_path": str(telemetry_path) if telemetry_path is not None else None,
            "profiler_path": (
                str(observability_config.profiler_path)
                if observability_config.profiler_path is not None
                else None
            ),
            "report_paths": {name: str(path) for name, path in report_paths.items()},
        },
    }


def _reset_environment_benchmark_state(environment: RuntimeEnvironment) -> None:
    """
    Best-effort reset of backend memory state before a benchmark run.

    Purpose:
    reduce the chance that stale allocations from earlier work in the same
    process distort the short benchmark measurement.

    Context:
    the benchmark workflow reuses the normal training stack, so this helper
    clears whatever memory counters or caches the active backend exposes
    without pretending every platform supports the same reset semantics.
    """
    # Benchmark mode tries to measure the short run itself rather than stale
    # memory from earlier activity in the same Python process. The reset is
    # best-effort because different backends expose different levels of memory
    # introspection and cache control.
    try:
        import torch
    except ImportError:
        return

    if environment.cuda_available and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    if environment.mps_available:
        mps_runtime = getattr(torch, "mps", None)
        if mps_runtime is not None and hasattr(mps_runtime, "empty_cache"):
            try:
                mps_runtime.empty_cache()
            except Exception:
                pass


def _collect_environment_benchmark_memory(
    environment: RuntimeEnvironment,
) -> dict[str, float | None]:
    """
    Collect a compact cross-backend memory summary for benchmark reporting.

    Purpose:
    present CPU, CUDA, and MPS benchmark runs through one comparable memory
    surface instead of exposing many backend-specific raw counters.

    Context:
    benchmark summaries are meant for quick environment comparisons, so the
    memory contract here favors readability and consistency over exhaustive
    low-level profiling detail.
    """
    # The benchmark summary intentionally reports a small cross-backend memory
    # shape rather than backend-specific raw counters. That keeps CPU, CUDA,
    # and MPS results comparable in one compact JSON surface.
    metrics: dict[str, float | None] = {
        "device_peak_memory_mb": None,
        "device_reserved_memory_mb": None,
        "process_rss_mb": None,
    }

    if environment.cuda_available:
        try:
            import torch

            if torch.cuda.is_available():
                metrics["device_peak_memory_mb"] = (
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                )
                metrics["device_reserved_memory_mb"] = (
                    torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
                )
        except Exception:
            pass
    elif environment.mps_available:
        try:
            import torch

            mps_runtime = getattr(torch, "mps", None)
            if mps_runtime is not None:
                metrics["device_peak_memory_mb"] = (
                    float(
                        getattr(mps_runtime, "current_allocated_memory", lambda: 0)()
                    )
                    / (1024.0 * 1024.0)
                )
                metrics["device_reserved_memory_mb"] = (
                    float(
                        getattr(mps_runtime, "driver_allocated_memory", lambda: 0)()
                    )
                    / (1024.0 * 1024.0)
                )
        except Exception:
            pass

    try:
        import psutil

        metrics["process_rss_mb"] = (
            psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        )
    except Exception:
        pass

    return metrics


def run_environment_benchmark_workflow(
    config: Config,
    *,
    train_config: TrainConfig,
    snapshot_config: SnapshotConfig,
    observability_config: ObservabilityConfig,
    requested_device_profile: str,
    resolved_device_profile: str,
    applied_profile_defaults: Mapping[str, Any],
    runtime_environment: RuntimeEnvironment,
    preflight_diagnostics: Sequence[RuntimeDiagnostic],
    output_dir: Path | None,
    benchmark_train_batches: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_name: str = "adam",
    trainer_class: type[Any] | None = None,
) -> EnvironmentBenchmarkArtifacts:
    """
    Run a short training-only benchmark tuned for environment comparison.

    Purpose:
    reuse the real training workflow while trimming away heavier reporting and
    evaluation steps so hardware/runtime configurations can be compared more
    directly.

    Context:
    this benchmark is intentionally not a separate micro-benchmark framework.
    It exercises the same top-level orchestration stack with a deliberately
    smaller observability and loop budget.

    Important inputs:
    - the incoming config objects describe the "real" run policy before the
      benchmark-specific reductions are applied
    - `benchmark_train_batches` controls the intentionally short training span
    - `runtime_environment` and `preflight_diagnostics` are preserved so the
      summary explains which machine/runtime was actually measured
    """

    if benchmark_train_batches <= 0:
        raise ValueError("benchmark_train_batches must be > 0")

    # Benchmark mode reuses the real training workflow on purpose. The goal is
    # not to invent a second "micro-benchmark only" codepath, but to run the
    # same environment resolution, backend tuning, and Trainer assembly while
    # trimming away the heavier reporting/evaluation stages.
    benchmark_train_config = replace(
        train_config,
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_train_batches=benchmark_train_batches,
        limit_val_batches=1,
        limit_test_batches=1,
        enable_progress_bar=False,
    )
    benchmark_snapshot_config = replace(snapshot_config, enabled=False)
    # Benchmark runs intentionally disable most observability extras so the
    # timing reflects training throughput more than artifact generation.
    benchmark_observability_config = replace(
        observability_config,
        enable_tensorboard=False,
        enable_text_logging=False,
        enable_csv_fallback_logger=False,
        enable_learning_rate_monitor=False,
        enable_device_stats=False,
        enable_rich_progress_bar=False,
        enable_system_telemetry=False,
        enable_parameter_histograms=False,
        enable_parameter_scalars=False,
        enable_prediction_figures=False,
        enable_model_graph=False,
        enable_model_text=False,
        enable_torchview=False,
        enable_profiler=False,
        enable_batch_audit=False,
        enable_prediction_exports=False,
        enable_plot_reports=False,
    )

    _reset_environment_benchmark_state(runtime_environment)
    start_time = perf_counter()
    artifacts = run_training_workflow(
        config,
        train_config=benchmark_train_config,
        snapshot_config=benchmark_snapshot_config,
        observability_config=benchmark_observability_config,
        requested_device_profile=requested_device_profile,
        resolved_device_profile=resolved_device_profile,
        applied_profile_defaults=applied_profile_defaults,
        runtime_environment=runtime_environment,
        preflight_diagnostics=preflight_diagnostics,
        fail_on_preflight_errors=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        output_dir=output_dir,
        skip_test=True,
        skip_predict=True,
        save_predictions=False,
        trainer_class=trainer_class,
    )
    duration_seconds = perf_counter() - start_time
    benchmark_memory = _collect_environment_benchmark_memory(runtime_environment)
    actual_train_batches = (
        artifacts.fit.train_batches_processed
        if artifacts.fit.train_batches_processed is not None
        else benchmark_train_batches
    )
    # This is an estimate rather than a promise of exact sample count. It is
    # still useful for quick environment comparison, but it should not be read
    # as a full profiler-grade accounting metric.
    samples_processed = actual_train_batches * config.data.batch_size

    summary = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "requested_device_profile": requested_device_profile,
        "resolved_device_profile": resolved_device_profile,
        "applied_profile_defaults": _json_ready(dict(applied_profile_defaults)),
        "environment": _json_ready(runtime_environment),
        "preflight_diagnostics": _json_ready(tuple(preflight_diagnostics)),
        "runtime_tuning": _json_ready(
            artifacts.summary.get("environment", {}).get("runtime_tuning", {})
        ),
        "train_config": _json_ready(benchmark_train_config),
        "benchmark": {
            "requested_train_batches": benchmark_train_batches,
            "actual_train_batches": actual_train_batches,
            "batch_size": config.data.batch_size,
            "samples_processed_estimate": samples_processed,
            "duration_seconds": duration_seconds,
            "batches_per_second": actual_train_batches / duration_seconds
            if duration_seconds > 0.0
            else None,
            "samples_per_second": samples_processed / duration_seconds
            if duration_seconds > 0.0
            else None,
        },
        "memory": benchmark_memory,
        "run_summary_path": (
            str(artifacts.summary_path) if artifacts.summary_path is not None else None
        ),
    }

    summary_path: Path | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "benchmark_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return EnvironmentBenchmarkArtifacts(summary=summary, summary_path=summary_path)


def run_training_workflow(
    config: Config,
    *,
    train_config: TrainConfig | None = None,
    snapshot_config: SnapshotConfig | None = None,
    observability_config: ObservabilityConfig | None = None,
    requested_device_profile: str = "auto",
    resolved_device_profile: str | None = None,
    applied_profile_defaults: Mapping[str, Any] | None = None,
    runtime_environment: RuntimeEnvironment | None = None,
    preflight_diagnostics: Sequence[RuntimeDiagnostic] | None = None,
    fail_on_preflight_errors: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_name: str = "adam",
    fit_ckpt_path: str | Path | None = None,
    eval_ckpt_path: CheckpointSelection = "best",
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
    seed: int | None = 42,
    skip_test: bool = False,
    skip_predict: bool = False,
    save_predictions: bool = True,
    trainer_class: type[Any] | None = None,
) -> MainRunArtifacts:
    """
    Execute the repository's shared train/evaluate/predict workflow.

    Purpose:
    provide one reusable Python-callable orchestration surface for CLI runs,
    tests, and notebooks.

    Context:
    this function sits above `FusedModelTrainer`, adding environment-profile
    resolution, summary writing, prediction export, and post-fit evaluation so
    those concerns do not have to be rebuilt at each entrypoint.

    Important inputs:
    - `config` carries the semantic data/model configuration for the run
    - the optional `*_config` arguments allow callers to override runtime,
      checkpoint, and observability policy without re-implementing the
      workflow
    - `requested_device_profile` and related runtime arguments let both CLI
      callers and direct Python callers use the same environment-aware tuning
      path
    """
    # This function is the shared orchestration surface used by both the CLI
    # script and the notebook. Keeping the workflow in one callable helps avoid
    # the classic problem where notebook logic quietly diverges from script
    # logic over time.
    #
    # Workflow outline:
    # 1. seed the run when possible
    # 2. materialize the output folder
    # 3. build the DataModule and training wrapper
    # 4. fit the model
    # 5. optionally test and predict on the held-out split
    # 6. persist a compact run summary plus optional prediction tensors
    if seed is not None and seed_everything is not None:
        seed_everything(seed, workers=True)
    # If Lightning is unavailable, we do not raise here ourselves. The later
    # model/trainer imports and construction path will already fail in a more
    # direct way. This conditional simply avoids pretending we can seed a
    # library that is not installed.

    output_dir = None if output_dir is None else Path(output_dir)
    if output_dir is not None:
        # The top-level workflow owns output directory creation so lower layers
        # can assume the destination exists before they try to write summaries,
        # predictions, or checkpoints.
        output_dir.mkdir(parents=True, exist_ok=True)

    effective_train_config = train_config or build_default_train_config(
        default_root_dir=output_dir,
    )
    # Apple Silicon is the one place where some useful runtime controls need to
    # be exported extremely early so the later Torch import/runtime init sees
    # them. The full profile resolver still runs below; this is only the
    # bootstrap bridge that lets those profile defaults take effect in time.
    _apply_early_apple_silicon_environment_defaults(
        requested_device_profile=requested_device_profile,
        train_config=effective_train_config,
    )
    effective_snapshot_config = snapshot_config or build_default_snapshot_config(
        output_dir=output_dir,
    )
    effective_observability_config = observability_config or build_default_observability_config(
        output_dir=output_dir,
    )
    apply_runtime_environment_overrides(train_config=effective_train_config)
    effective_runtime_environment = runtime_environment or detect_runtime_environment()
    effective_config = config
    effective_applied_profile_defaults = (
        dict(applied_profile_defaults) if applied_profile_defaults is not None else {}
    )
    if resolved_device_profile is None:
        # Direct Python callers should receive the same environment-aware
        # defaults as the CLI path. Resolving profiles here keeps the reusable
        # workflow honest instead of making environment tuning "CLI only."
        profile_resolution = resolve_device_profile(
            requested_profile=requested_device_profile,
            environment=effective_runtime_environment,
            train_config=effective_train_config,
            data_config=effective_config.data,
            observability_config=effective_observability_config,
        )
        effective_resolved_device_profile = profile_resolution.resolved_profile
        effective_config = Config(
            data=profile_resolution.data_config,
            tft=effective_config.tft,
            tcn=effective_config.tcn,
        )
        effective_train_config = profile_resolution.train_config
        effective_observability_config = profile_resolution.observability_config
        effective_applied_profile_defaults = dict(profile_resolution.applied_defaults)
        apply_runtime_environment_overrides(train_config=effective_train_config)
    else:
        # If the caller already resolved the profile externally, this workflow
        # trusts that result and avoids second-guessing it.
        effective_resolved_device_profile = resolved_device_profile
    effective_preflight_diagnostics = tuple(
        preflight_diagnostics
        if preflight_diagnostics is not None
        else collect_runtime_diagnostics(
            requested_profile=requested_device_profile,
            resolved_profile=effective_resolved_device_profile,
            environment=effective_runtime_environment,
            train_config=effective_train_config,
            data_config=effective_config.data,
            observability_config=effective_observability_config,
        )
    )
    # If the caller supplied explicit configs we respect them verbatim.
    # Otherwise we derive defaults that are aligned with the chosen output dir.
    #
    # Important disclaimer:
    # - this workflow treats observability as first-class run configuration
    # - the defaults are intentionally visibility-friendly, but callers still
    #   remain free to dial them down for very constrained hardware sessions

    if fail_on_preflight_errors and has_error_diagnostics(effective_preflight_diagnostics):
        raise RuntimeError(
            "Runtime preflight checks failed "
            f"for device profile {requested_device_profile} -> "
            f"{effective_resolved_device_profile}.\n"
            f"{format_runtime_diagnostics(effective_preflight_diagnostics)}"
        )

    import torch

    from data.datamodule import AZT1DDataModule
    from evaluation import evaluate_prediction_batches
    from observability import export_prediction_table, generate_plotly_reports
    from train import FusedModelTrainer as DefaultFusedModelTrainer

    if trainer_class is None:
        trainer_class = DefaultFusedModelTrainer

    datamodule = AZT1DDataModule(effective_config.data)
    # The training wrapper deliberately sits between the top-level workflow and
    # Lightning. That keeps this module focused on orchestration while
    # `src/train.py` owns fit/test/predict policy.
    trainer = trainer_class(
        effective_config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        trainer_config=effective_train_config,
        snapshot_config=effective_snapshot_config,
        observability_config=effective_observability_config,
    )
    trainer_observability = getattr(trainer, "observability_artifacts", None)
    text_logger = getattr(trainer_observability, "text_logger", None)
    if text_logger is not None:
        text_logger.info("starting training workflow")

    try:
        fit_artifacts = trainer.fit(datamodule, ckpt_path=fit_ckpt_path)
    except Exception as exc:
        runtime_failure_diagnostics = analyze_runtime_failure(
            exc,
            requested_profile=requested_device_profile,
            resolved_profile=effective_resolved_device_profile,
            environment=effective_runtime_environment,
        )
        raise RuntimeError(
            "Training workflow failed during fit() "
            f"for device profile {requested_device_profile} -> "
            f"{effective_resolved_device_profile}.\n"
            f"Preflight diagnostics:\n{format_runtime_diagnostics(effective_preflight_diagnostics)}\n\n"
            f"Failure analysis:\n{format_runtime_diagnostics(runtime_failure_diagnostics)}"
        ) from exc
    # Evaluation may target `"best"`, `"last"`, an explicit checkpoint path,
    # or the current in-memory model. We normalize that choice once here so
    # test and prediction stay consistent.
    resolved_eval_ckpt_path = _resolve_eval_ckpt_path(fit_artifacts, eval_ckpt_path)

    test_metrics: list[Mapping[str, float]] | None = None
    test_evaluation: EvaluationResult | None = None
    test_predictions: list[torch.Tensor] | None = None
    predictions_path: Path | None = None
    prediction_table_path: Path | None = None
    report_paths: dict[str, Path] = {}

    if fit_artifacts.has_test_data and not skip_test:
        try:
            test_metrics = trainer.test(datamodule, ckpt_path=resolved_eval_ckpt_path)
        except Exception as exc:
            runtime_failure_diagnostics = analyze_runtime_failure(
                exc,
                requested_profile=requested_device_profile,
                resolved_profile=effective_resolved_device_profile,
                environment=effective_runtime_environment,
            )
            raise RuntimeError(
                "Training workflow failed during test() "
                f"for device profile {requested_device_profile} -> "
                f"{effective_resolved_device_profile}.\n"
                f"Preflight diagnostics:\n{format_runtime_diagnostics(effective_preflight_diagnostics)}\n\n"
                f"Failure analysis:\n{format_runtime_diagnostics(runtime_failure_diagnostics)}"
            ) from exc
    # We intentionally treat test evaluation and raw prediction collection as
    # separate toggles. Some workflows want metrics only, while others want the
    # raw tensors for notebook analysis or custom visualization.

    if fit_artifacts.has_test_data and not skip_predict:
        try:
            test_predictions = trainer.predict_test(
                datamodule,
                ckpt_path=resolved_eval_ckpt_path,
            )
        except Exception as exc:
            runtime_failure_diagnostics = analyze_runtime_failure(
                exc,
                requested_profile=requested_device_profile,
                resolved_profile=effective_resolved_device_profile,
                environment=effective_runtime_environment,
            )
            raise RuntimeError(
                "Training workflow failed during predict() "
                f"for device profile {requested_device_profile} -> "
                f"{effective_resolved_device_profile}.\n"
                f"Preflight diagnostics:\n{format_runtime_diagnostics(effective_preflight_diagnostics)}\n\n"
                f"Failure analysis:\n{format_runtime_diagnostics(runtime_failure_diagnostics)}"
            ) from exc
        # Detailed evaluation currently hangs off the prediction path rather
        # than the `trainer.test(...)` path because the richer evaluator needs
        # raw forecast tensors plus the aligned source batches, not just
        # Lightning's already-reduced scalar metrics.
        if test_predictions is None:
            raise RuntimeError("predict_test returned no prediction batches.")
        quantiles = getattr(fit_artifacts.model, "quantiles", (0.1, 0.5, 0.9))
        test_evaluation = evaluate_prediction_batches(
            predictions=test_predictions,
            batches=datamodule.test_dataloader(),
            quantiles=quantiles,
        )
        if output_dir is not None and save_predictions:
            # Predictions are saved as raw tensors on purpose. That preserves
            # the model's direct output without forcing an opinionated export
            # schema before the project decides how it wants downstream
            # analysis, plotting, or calibration code to consume them.
            predictions_path = output_dir / "test_predictions.pt"
            torch.save(
                [tensor.detach().cpu() for tensor in test_predictions],
                predictions_path,
            )
        if effective_observability_config.enable_prediction_exports:
            # The CSV export is intentionally additive rather than a replacement
            # for the raw tensor artifact. We keep both because they serve
            # different debugging/research use cases.
            prediction_table_path = export_prediction_table(
                datamodule=datamodule,
                predictions=test_predictions,
                quantiles=quantiles,
                output_path=effective_observability_config.prediction_table_path,
                sampling_interval_minutes=effective_config.data.sampling_interval_minutes,
            )
        if effective_observability_config.enable_plot_reports:
            # Plotly reports are generated from the flat prediction table so the
            # reporting path stays decoupled from the in-memory prediction
            # tensor structure.
            report_paths = generate_plotly_reports(
                prediction_table_path,
                report_dir=effective_observability_config.report_dir,
                max_subjects=effective_observability_config.max_forecast_subjects_per_report,
                evaluation_result=test_evaluation,
            )

    runtime_tuning_report = getattr(trainer, "runtime_tuning_report", None)
    runtime_tuning_applied = (
        dict(getattr(runtime_tuning_report, "applied", {}))
        if runtime_tuning_report is not None
        else {}
    )
    runtime_tuning_skipped = (
        dict(getattr(runtime_tuning_report, "skipped", {}))
        if runtime_tuning_report is not None
        else {}
    )
    runtime_tuning_summary = {
        "applied": runtime_tuning_applied,
        "skipped": runtime_tuning_skipped,
    }

    summary = _build_run_summary(
        config=effective_config,
        train_config=effective_train_config,
        snapshot_config=effective_snapshot_config,
        observability_config=effective_observability_config,
        requested_device_profile=requested_device_profile,
        resolved_device_profile=effective_resolved_device_profile,
        applied_profile_defaults=effective_applied_profile_defaults,
        runtime_environment=effective_runtime_environment,
        preflight_diagnostics=effective_preflight_diagnostics,
        fit_artifacts=fit_artifacts,
        eval_ckpt_path=resolved_eval_ckpt_path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        output_dir=output_dir,
        test_metrics=test_metrics,
        test_evaluation=test_evaluation,
        predictions=test_predictions,
        predictions_path=predictions_path,
        prediction_table_path=prediction_table_path,
        report_paths=report_paths,
        logger_dir=getattr(trainer_observability, "logger_dir", None),
        text_log_path=getattr(trainer_observability, "text_log_path", None),
        telemetry_path=getattr(trainer_observability, "telemetry_path", None),
        runtime_tuning=runtime_tuning_summary,
    )

    summary_path: Path | None = None
    if output_dir is not None:
        # JSON keeps the run summary easy to diff, archive, or inspect from
        # outside Python.
        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return MainRunArtifacts(
        fit=fit_artifacts,
        test_metrics=test_metrics,
        test_evaluation=test_evaluation,
        test_predictions=test_predictions,
        summary=summary,
        summary_path=summary_path,
        predictions_path=predictions_path,
        prediction_table_path=prediction_table_path,
        report_paths=report_paths,
        telemetry_path=getattr(trainer_observability, "telemetry_path", None),
        logger_dir=getattr(trainer_observability, "logger_dir", None),
        text_log_path=getattr(trainer_observability, "text_log_path", None),
        requested_device_profile=requested_device_profile,
        resolved_device_profile=effective_resolved_device_profile,
        applied_profile_defaults=effective_applied_profile_defaults,
        runtime_environment=effective_runtime_environment,
        preflight_diagnostics=effective_preflight_diagnostics,
    )
