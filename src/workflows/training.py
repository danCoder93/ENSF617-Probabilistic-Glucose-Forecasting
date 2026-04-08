from __future__ import annotations

# AI-assisted maintenance note:
# This module owns the reusable top-level workflows that sit above
# `src/train.py`.
#
# Responsibility boundary:
# - translate a prepared project config into one full run
# - coordinate evaluation, post-run shared-report packaging, export sinks,
#   lightweight reports, TensorBoard post-run report logging, and compact run
#   summaries
# - keep CLI and notebook callers on one shared orchestration path
#
# What does *not* live here:
# - argument parsing
# - low-level Trainer assembly details
# - model forward/loss/optimizer behavior

import json
import csv
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from config import (
    Config,
    ConfigurationValidationError,
    ObservabilityConfig,
    SnapshotConfig,
    TrainConfig,
    config_to_dict,
    validate_runtime_configuration,
    ConfigurationValidationError,
)
from config.types import PathInput
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
    synchronize_runtime_device,
)
from workflows.helpers import (
    _apply_early_apple_silicon_environment_defaults,
    _json_ready,
    _resolve_eval_ckpt_path,
)
from workflows.types import EnvironmentBenchmarkArtifacts, MainRunArtifacts

if TYPE_CHECKING:
    import logging
    import torch

    from evaluation import EvaluationResult
    from reporting.types import SharedReport
    from train import CheckpointSelection, FitArtifacts, FusedModelTrainer
else:
    CheckpointSelection = Any
    EvaluationResult = Any
    FitArtifacts = Any
    FusedModelTrainer = Any
    SharedReport = Any
    logging = Any

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
    data_summary: Mapping[str, Any] | None = None,
    data_summary_path: Path | None = None,
    metrics_summary_path: Path | None = None,
    grouped_metrics_paths: Mapping[str, Path] | None = None,
    report_index_path: Path | None = None,
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
        # CHANGE: Put the dataset summary directly in the run summary.
        # Clear idea here: if someone opens one JSON file, they should immediately
        # understand what data the run actually used.
        "data": {
            "summary_path": str(data_summary_path) if data_summary_path is not None else None,
            "descriptive_stats": _json_ready(data_summary),
        },
        # CHANGE: Add a dedicated metrics section that points to the clean
        # metrics artifact and the grouped metric table exports.
        "metrics": {
            "summary_path": (
                str(metrics_summary_path) if metrics_summary_path is not None else None
            ),
            "grouped_paths": {
                name: str(path)
                for name, path in (grouped_metrics_paths or {}).items()
            },
            "test_metrics": _json_ready(test_metrics),
            "test_evaluation": _json_ready(test_evaluation),
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
            "report_index_path": (
                str(report_index_path) if report_index_path is not None else None
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

    # CUDA and MPS expose different cache/stat-reset APIs. We use whichever is
    # available and ignore failures because benchmark hygiene should help the
    # run, not become another reason the workflow fails.
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

    # CUDA and MPS use different Torch runtime surfaces for memory inspection.
    # We normalize both into the same final metric names so later benchmark
    # summaries remain easy to compare.
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

    # Process RSS complements device memory stats by answering the broader
    # process-memory question regardless of accelerator backend.
    try:
        import psutil

        metrics["process_rss_mb"] = (
            psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        )
    except Exception:
        pass

    return metrics


def _coerce_table_rows(value: Any) -> list[dict[str, Any]]:
    """
    Convert a grouped evaluation payload into a flat list of row dictionaries.

    Purpose:
    make downstream CSV export simple even when the incoming evaluation structure
    is nested, partially typed, or a mix of dataclasses and plain dictionaries.
    """
    value = _json_ready(value)

    if value is None:
        return []

    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                rows.append(item)
            else:
                rows.append({"value": item})
        return rows

    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        scalar_keys = {
            key: val
            for key, val in value.items()
            if not isinstance(val, (dict, list))
        }

        nested_found = False
        for key, val in value.items():
            if isinstance(val, list):
                nested_found = True
                for item in val:
                    if isinstance(item, dict):
                        rows.append({"group": key, **scalar_keys, **item})
                    else:
                        rows.append({"group": key, **scalar_keys, "value": item})
            elif isinstance(val, dict):
                nested_found = True
                rows.append({"group": key, **scalar_keys, **val})

        if nested_found:
            return rows

        return [value]

    return [{"value": value}]


def _write_csv_rows(output_path: Path, rows: Sequence[Mapping[str, Any]]) -> Path | None:
    """
    Write a sequence of dictionaries to CSV.

    Purpose:
    keep artifact export lightweight and dependency-free so the workflow can
    persist report-friendly tables without introducing a pandas requirement here.
    """
    if not rows:
        return None

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )

    return output_path


def _extract_grouped_evaluation_tables(
    evaluation_result: EvaluationResult | None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Pull grouped evaluation sections into flat row tables.

    Purpose:
    turn structured evaluation outputs into report-friendly tables that can be
    exported directly as CSV artifacts.
    """
    if evaluation_result is None:
        return {}

    payload = _json_ready(evaluation_result)
    if not isinstance(payload, dict):
        return {}

    candidate_keys = {
        "metrics_by_horizon": "metrics_by_horizon",
        "by_horizon": "metrics_by_horizon",
        "horizon_metrics": "metrics_by_horizon",
        "metrics_by_subject": "metrics_by_subject",
        "by_subject": "metrics_by_subject",
        "subject_metrics": "metrics_by_subject",
        "metrics_by_glucose_range": "metrics_by_glucose_range",
        "by_glucose_range": "metrics_by_glucose_range",
        "glucose_range_metrics": "metrics_by_glucose_range",
    }

    extracted: dict[str, list[dict[str, Any]]] = {}
    for incoming_key, artifact_name in candidate_keys.items():
        if incoming_key in payload:
            rows = _coerce_table_rows(payload[incoming_key])
            if rows:
                extracted[artifact_name] = rows

    return extracted


def _log_post_run_shared_report_to_tensorboard(
    *,
    shared_report: SharedReport,
    logger_or_trainer: Any,
    max_forecast_subjects: int,
    text_logger: logging.Logger | None = None,
) -> bool:
    """
    Best-effort bridge from workflow-level post-run artifacts into TensorBoard.

    Purpose:
    centralize the workflow's call into the reporting package's TensorBoard
    sink so the post-run logging policy stays readable and easy to evolve.

    Context:
    the workflow is the first layer that can see:
    - the canonical shared report
    - the active experiment logger surface
    - the final run-stage decision that prediction/evaluation are complete

    That makes it the correct place to perform the post-run TensorBoard handoff
    without pushing reporting policy down into callbacks or model code.

    Important compatibility rule:
    this helper does not change metric truth, prediction flow, or evaluation
    timing. It only standardizes the handoff into the reporting sink.

    Namespace rule:
    the reporting sink now internally organizes outputs into clearer
    dashboard/text/report surfaces. We intentionally keep the top-level
    namespace argument stable here so existing run layouts remain broadly
    recognizable while the sink itself improves the internal information
    architecture underneath that root.
    """
    from reporting import log_shared_report_to_tensorboard

    # The sink itself is already best-effort. This helper simply keeps the
    # workflow's intent explicit and provides a small optional text-log note so
    # later readers can tell whether a TensorBoard-compatible logger was active.
    logged = log_shared_report_to_tensorboard(
        shared_report=shared_report,
        logger_or_trainer=logger_or_trainer,
        global_step=0,
        namespace="report",
        max_table_rows=20,
        max_forecast_subjects=max_forecast_subjects,
    )

    if text_logger is not None:
        if logged:
            text_logger.info(
                "logged post-run shared report to TensorBoard using dashboard-first report sink"
            )
        else:
            text_logger.info(
                "skipped post-run TensorBoard report logging because no compatible TensorBoard experiment backend was active"
            )

    return logged


def _export_post_run_shared_report_artifacts(
    *,
    shared_report: SharedReport,
    report_dir: PathInput | None,
    text_logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """
    Best-effort bridge from the canonical shared report into structured artifacts.

    Purpose:
    centralize the workflow's call into the reporting package's structured
    export sink so CSV/JSON artifact export stays aligned with the same
    post-run package used by TensorBoard and Plotly.

    Context:
    this helper exists for the same architectural reason as the TensorBoard
    handoff helper:
    - the workflow owns the final resolved report directory
    - the workflow owns the canonical shared report
    - the workflow owns the final artifact map surfaced in summaries

    Output shape:
    the reporting sink writes a mixed-format bundle under:
    `<report_dir>/artifacts/shared_report/`

    Important compatibility rule:
    this helper is best-effort and sink-only. It does not change evaluation
    truth, prediction export semantics, or table-building logic.
    """
    from reporting.structured_exports import export_shared_report_artifacts

    exported_paths = export_shared_report_artifacts(
        shared_report=shared_report,
        report_dir=report_dir,
    )

    if text_logger is not None:
        if exported_paths:
            text_logger.info(
                "exported structured shared-report artifacts to %s",
                report_dir,
            )
        else:
            text_logger.info(
                "skipped structured shared-report artifact export because report_dir was not configured"
            )

    return exported_paths


def _collect_datamodule_data_summary(
    *,
    datamodule: Any,
    text_logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """
    Best-effort handoff into the DataModule-owned dataset summary surface.

    Purpose:
    ask the DataModule for its already-canonical dataset description without
    moving dataset-summary computation into the workflow.

    Context:
    the workflow is allowed to orchestrate post-run collection of the summary,
    but the DataModule remains the source of truth for what that summary means
    and how it is computed.

    Important compatibility rule:
    failures here must never break training, evaluation, or prediction export.
    Dataset-summary collection is an additive reporting enhancement only.
    """
    describe_data = getattr(datamodule, "describe_data", None)
    if describe_data is None:
        if text_logger is not None:
            text_logger.info(
                "skipped dataset summary collection because the active DataModule does not expose describe_data()"
            )
        return None

    try:
        data_summary = describe_data()
    except Exception as exc:
        if text_logger is not None:
            text_logger.warning(
                "failed to collect dataset summary from DataModule.describe_data(); continuing without data summary export: %s",
                exc,
            )
        return None

    if text_logger is not None:
        text_logger.info("collected dataset summary from DataModule.describe_data()")

    return data_summary


def _export_datamodule_data_summary(
    *,
    data_summary: Mapping[str, Any] | None,
    report_dir: PathInput | None,
    text_logger: logging.Logger | None = None,
) -> Path | None:
    """
    Best-effort export of the DataModule-provided dataset summary.

    Purpose:
    write the DataModule-owned summary to a stable JSON artifact so later
    reporting sinks and manual inspection can reuse it.

    Output shape:
    this workflow currently writes the summary to:
    `<report_dir>/data_summary.json`

    Design note:
    the export location is intentionally simple for now. A later structured-
    export patch can move this into a richer subfolder layout without changing
    the fact that the DataModule remains the summary source of truth.
    """
    if data_summary is None:
        if text_logger is not None:
            text_logger.info(
                "skipped dataset summary export because no dataset summary was collected"
            )
        return None

    if report_dir is None:
        if text_logger is not None:
            text_logger.info(
                "skipped dataset summary export because report_dir was not configured"
            )
        return None

    resolved_report_dir = Path(report_dir)
    resolved_report_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_report_dir / "data_summary.json"

    try:
        output_path.write_text(
            json.dumps(_json_ready(dict(data_summary)), indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        if text_logger is not None:
            text_logger.warning(
                "failed to export dataset summary JSON; continuing without data summary artifact: %s",
                exc,
            )
        return None

    if text_logger is not None:
        text_logger.info("exported dataset summary JSON to %s", output_path)

    return output_path


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
    #
    # The prediction-sanity flag is also disabled here because benchmark mode is
    # about keeping the loop lightweight; the goal is environment comparison,
    # not semantic forecast forensics.
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
        enable_prediction_sanity=False,
        enable_prediction_figures=False,
        enable_model_graph=False,
        enable_model_text=False,
        enable_torchview=False,
        enable_profiler=False,
        enable_batch_audit=False,
        enable_prediction_exports=False,
        enable_plot_reports=False,
    )

    # Synchronization around the timed section matters because accelerator
    # backends can queue work asynchronously. Reset/synchronize first so the
    # measured interval reflects completed benchmark work more closely.
    _reset_environment_benchmark_state(runtime_environment)
    synchronize_runtime_device(environment=runtime_environment)
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
    synchronize_runtime_device(environment=runtime_environment)
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

    # The benchmark summary mirrors the normal run summary style but focuses on
    # the small subset of fields that matter most for throughput comparison.
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

def _run_post_run_analysis_scripts(
    *,
    artifact_dir: Path,
) -> dict[str, str]:
    """
    Run optional post-run analysis scripts after the core workflow finishes.

    Design choice:
    - these scripts are best-effort post-processing only
    - failures here should not invalidate a successful train/test/predict run
    - each script receives the same artifact directory so outputs stay aligned
    """
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"

    script_names = [
        "build_metrics_analysis.py",
        "build_threshold_accuracy_analysis.py",
        "build_persistence_baseline.py",
        "build_event_aware_analysis.py",
        "build_run_health_summary.py",
    ]

    results: dict[str, str] = {}

    for script_name in script_names:
        script_path = scripts_dir / script_name
        label = script_path.stem

        if not script_path.exists():
            results[label] = "missing"
            print(f"[post-run] Skipping missing script: {script_path}")
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--artifact-dir",
            str(artifact_dir),
        ]

        try:
            completed = subprocess.run(
                cmd,
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            results[label] = "ok"
            print(f"[post-run] Completed: {script_name}")
            if completed.stdout.strip():
                print(completed.stdout.strip())
            if completed.stderr.strip():
                print(completed.stderr.strip())

        except subprocess.CalledProcessError as exc:
            results[label] = f"failed ({exc.returncode})"
            print(f"[post-run] FAILED: {script_name} (exit code {exc.returncode})")
            if exc.stdout.strip():
                print(exc.stdout.strip())
            if exc.stderr.strip():
                print(exc.stderr.strip())

        except Exception as exc:
            results[label] = f"error ({type(exc).__name__})"
            print(f"[post-run] ERROR running {script_name}: {exc}")

    return results

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
    resolution, summary writing, prediction export, structured evaluation, and
    post-run reporting so those concerns do not have to be rebuilt at each
    entrypoint.

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
    # 6. build the canonical post-run shared report when predictions exist
    # 7. mirror that shared report into configured post-run sinks
    # 8. serialize optional prediction/report artifacts
    # 9. persist a compact run summary
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

    # Apply low-level runtime overrides before profile resolution so both direct
    # callers and profile logic start from the same effective baseline.
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

        # Re-apply environment overrides after profile resolution because the
        # profile may have changed runtime-facing settings such as precision or
        # worker defaults.
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

    try:
        validate_runtime_configuration(
            train_config=effective_train_config,
            data_config=effective_config.data,
            observability_config=effective_observability_config,
            snapshot_config=effective_snapshot_config,
            resolved_profile=effective_resolved_device_profile,
            has_validation_data=None,
        )
    except ConfigurationValidationError as exc:
        raise RuntimeError(
            "Training workflow failed during configuration validation "
            f"for device profile {requested_device_profile} -> "
            f"{effective_resolved_device_profile}.\n{exc}"
        ) from exc

    import torch

    from data.datamodule import AZT1DDataModule
    from evaluation import evaluate_prediction_batches
    from reporting import (
        build_shared_report,
        export_prediction_table_from_report,
        generate_plotly_reports,
    )
    from train import FusedModelTrainer as DefaultFusedModelTrainer

    if trainer_class is None:
        trainer_class = DefaultFusedModelTrainer

    datamodule = AZT1DDataModule(effective_config.data)

    data_summary: dict[str, Any] | None = None
    data_summary_path: Path | None = None

    # CHANGE: Try to capture dataset observability early, but never let that
    # block the run. Training is the primary path. Artifact collection is best-effort.
    try:
        datamodule.prepare_data()
        datamodule.setup(stage="fit")
        data_summary = datamodule.describe_data()
    except Exception:
        data_summary = None

    # CHANGE: Save the dataset summary as its own artifact when available.
    if output_dir is not None and data_summary is not None:
        data_summary_path = output_dir / "data_summary.json"
        data_summary_path.write_text(
            json.dumps(_json_ready(data_summary), indent=2),
            encoding="utf-8",
        )

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

    # The runtime observability bundle may provide several useful paths and
    # logger surfaces. We pull them out once here because the workflow needs:
    # - a plain-text logger for lifecycle notes
    # - the active experiment logger for post-run reporting sinks
    text_logger = getattr(trainer_observability, "text_logger", None)
    active_logger = getattr(trainer_observability, "logger", None)
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

    # Evaluation may target "best", "last", an explicit checkpoint path, or
    # the current in-memory model. We normalize that choice once here so test
    # and prediction stay consistent.
    resolved_eval_ckpt_path = _resolve_eval_ckpt_path(fit_artifacts, eval_ckpt_path)

    test_metrics: list[Mapping[str, float]] | None = None
    test_evaluation: EvaluationResult | None = None
    test_predictions: list[torch.Tensor] | None = None
    data_summary: dict[str, Any] | None = None
    predictions_path: Path | None = None
    prediction_table_path: Path | None = None
    metrics_summary_path: Path | None = None
    report_index_path: Path | None = None
    grouped_metrics_paths: dict[str, Path] = {}
    report_paths: dict[str, Path] = {}

    # `shared_report` is the canonical in-memory packaging layer for post-run
    # reporting. It is intentionally optional here so the workflow can preserve
    # the older artifact-only behavior when prediction generation is skipped or
    # when no held-out predictions are available.
    shared_report = None

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
    # separate toggles. Some workflows want scalar test metrics only, while
    # others want the raw tensors and structured post-run report surfaces.
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

        # Structured evaluation remains the canonical source of metric truth.
        # The reporting package consumes this evaluation output later; it does
        # not replace or recompute it here.
        test_evaluation = evaluate_prediction_batches(
            predictions=test_predictions,
            batches=datamodule.test_dataloader(),
            quantiles=quantiles,
        )

        # Build the canonical in-memory shared report once after raw predictions
        # and structured evaluation both exist.
        #
        # This is the key lifecycle boundary of the reporting architecture:
        # - evaluation computes metric truth
        # - reporting packages that truth into one reusable surface
        # - sinks such as TensorBoard, CSV, and Plotly consume that surface or
        #   stay tightly aligned with it
        #
        # Important compatibility note:
        # the surrounding workflow still preserves the long-standing raw tensor
        # save, CSV export, and Plotly HTML generation paths. The shared report
        # is an enhancement layer underneath those sinks rather than a new
        # mandatory artifact contract for callers.
        shared_report = build_shared_report(
            datamodule=datamodule,
            predictions=test_predictions,
            quantiles=quantiles,
            sampling_interval_minutes=effective_config.data.sampling_interval_minutes,
            evaluation_result=test_evaluation,
            data_summary=data_summary,
        )

        # Collect a best-effort dataset summary from the DataModule after setup
        # has already happened through the normal fit/test/predict lifecycle.
        #
        # Important architectural rule:
        # the workflow only asks for the summary and optionally exports it. The
        # DataModule still owns the underlying summary computation and remains
        # the source of truth for what the dataset summary contains.
        data_summary = _collect_datamodule_data_summary(
            datamodule=datamodule,
            text_logger=text_logger,
        )

        data_summary_path = _export_datamodule_data_summary(
            data_summary=data_summary,
            report_dir=effective_observability_config.report_dir,
            text_logger=text_logger,
        )
        if data_summary_path is not None:
            report_paths["data_summary"] = data_summary_path

        # Mirror the canonical shared report into TensorBoard-compatible logger
        # backends when such a backend is active for the run.
        #
        # Important lifecycle note:
        # this happens only after prediction and structured evaluation complete.
        # That keeps the split explicit:
        # - callbacks/loggers handle live runtime observability
        # - the reporting package handles post-run packaging and rendering
        #
        # Dashboard-first enhancement note:
        # the TensorBoard sink itself now performs internal curation into
        # dashboard/text/report layers. The workflow does not duplicate that
        # curation logic here; it simply performs the one canonical handoff once
        # the shared report exists.
        #
        # Important safety note:
        # the sink remains intentionally best-effort.
        # - if the active logger is not TensorBoard-compatible, logging is
        #   skipped cleanly
        # - if a particular artifact family cannot be rendered, the rest of the
        #   workflow continues
        # - the call does not alter training, prediction, or evaluation logic
        _log_post_run_shared_report_to_tensorboard(
            shared_report=shared_report,
            logger_or_trainer=active_logger,
            max_forecast_subjects=effective_observability_config.max_forecast_subjects_per_report,
            text_logger=text_logger,
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
            #
            # The workflow already built the canonical `shared_report` above, so
            # the strict export path now consumes that same in-memory report
            # directly instead of rebuilding reporting surfaces from raw inputs.
            # This keeps CSV export aligned with the same "build once, consume
            # many ways" pattern already used by the TensorBoard and Plotly
            # reporting sinks.
            prediction_table_path = export_prediction_table_from_report(
                shared_report=shared_report,
                output_path=effective_observability_config.prediction_table_path,
            )

        # Export a machine-readable mixed CSV/JSON artifact bundle from the same
        # canonical shared report used by the other post-run sinks.
        #
        # Why this happens here:
        # - the workflow already owns the resolved report directory
        # - the workflow already owns the canonical `SharedReport`
        # - the workflow already collects `report_paths` for later summary
        #   writing and CLI/notebook visibility
        #
        # Important policy:
        # this is additive to, not a replacement for, the existing raw tensor
        # and single prediction-table exports.
        structured_export_paths = _export_post_run_shared_report_artifacts(
            shared_report=shared_report,
            report_dir=effective_observability_config.report_dir,
            text_logger=text_logger,
        )
        report_paths.update(structured_export_paths)

        if effective_observability_config.enable_plot_reports:
            plotly_report_paths = generate_plotly_reports(
                prediction_table_path,
                report_dir=effective_observability_config.report_dir,
                max_subjects=effective_observability_config.max_forecast_subjects_per_report,
                shared_report=shared_report,
            )
            report_paths.update(plotly_report_paths)

    # CHANGE: Write a clean metrics artifact so results are easy to inspect
    # without digging through the full run summary.
    metrics_summary: dict[str, Any] | None = None
    if test_metrics is not None or test_evaluation is not None:
        metrics_summary = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "resolved_eval_ckpt_path": (
                str(resolved_eval_ckpt_path)
                if resolved_eval_ckpt_path not in (None, "best", "last")
                else resolved_eval_ckpt_path
            ),
            "test_metrics": _json_ready(test_metrics),
            "test_evaluation": _json_ready(test_evaluation),
            "prediction_table_path": (
                str(prediction_table_path)
                if prediction_table_path is not None
                else None
            ),
            "report_paths": {name: str(path) for name, path in report_paths.items()},
        }

    if output_dir is not None and metrics_summary is not None:
        metrics_summary_path = output_dir / "metrics_summary.json"
        metrics_summary_path.write_text(
            json.dumps(metrics_summary, indent=2),
            encoding="utf-8",
        )

    # CHANGE: Export grouped evaluation tables as flat CSV files so the results
    # are easier to inspect, plot, and reuse in the report.
    grouped_metric_tables = _extract_grouped_evaluation_tables(test_evaluation)
    if output_dir is not None and grouped_metric_tables:
        for artifact_name, rows in grouped_metric_tables.items():
            output_path = output_dir / f"{artifact_name}.csv"
            written_path = _write_csv_rows(output_path, rows)
            if written_path is not None:
                grouped_metrics_paths[artifact_name] = written_path

    # CHANGE: Write one artifact index so every important output from the run
    # can be found from a single file. This keeps reporting and debugging cleaner.
    report_index: dict[str, Any] | None = None
    if output_dir is not None:
        report_index = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "output_dir": str(output_dir),
            "run_summary_path": str(output_dir / "run_summary.json"),
            "data_summary_path": (
                str(data_summary_path) if data_summary_path is not None else None
            ),
            "metrics_summary_path": (
                str(metrics_summary_path) if metrics_summary_path is not None else None
            ),
            "grouped_metrics_paths": {
                name: str(path) for name, path in grouped_metrics_paths.items()
            },
            "prediction_artifacts": {
                "predictions_path": (
                    str(predictions_path) if predictions_path is not None else None
                ),
                "prediction_table_path": (
                    str(prediction_table_path)
                    if prediction_table_path is not None
                    else None
                ),
            },
            "report_paths": {name: str(path) for name, path in report_paths.items()},
            "observability": {
                "logger_dir": (
                    str(getattr(trainer_observability, "logger_dir", None))
                    if getattr(trainer_observability, "logger_dir", None) is not None
                    else None
                ),
                "text_log_path": (
                    str(getattr(trainer_observability, "text_log_path", None))
                    if getattr(trainer_observability, "text_log_path", None) is not None
                    else None
                ),
                "telemetry_path": (
                    str(getattr(trainer_observability, "telemetry_path", None))
                    if getattr(trainer_observability, "telemetry_path", None) is not None
                    else None
                ),
            },
        }

        report_index_path = output_dir / "report_index.json"
        report_index_path.write_text(
            json.dumps(report_index, indent=2),
            encoding="utf-8",
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

    # Persist both the applied and skipped runtime-tuning decisions so later
    # readers can understand not only what changed, but also what the runtime
    # layer considered and declined to change.
    runtime_tuning_summary = {
        "applied": runtime_tuning_applied,
        "skipped": runtime_tuning_skipped,
    }

    # Build the final run summary only after all optional post-run sinks have
    # had a chance to populate their artifact paths and reporting outputs.
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
        data_summary=data_summary,
        data_summary_path=data_summary_path,
        metrics_summary_path=metrics_summary_path,
        grouped_metrics_paths=grouped_metrics_paths,
        report_index_path=report_index_path,
    )

    summary_path: Path | None = None
    if output_dir is not None:
        # JSON keeps the run summary easy to diff, archive, or inspect from
        # outside Python.
        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Run best-effort post-run analysis scripts only after the core summary
        # and artifact bundle already exist on disk.
        post_run_analysis_results = _run_post_run_analysis_scripts(
            artifact_dir=output_dir,
        )

        # Persist the script statuses into the run summary and rewrite it so the
        # final summary reflects both the core workflow and any post-run
        # analysis outcomes.
        summary["post_run_analysis"] = post_run_analysis_results
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("Post-run analysis script results:")
        print(post_run_analysis_results)

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