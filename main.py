from __future__ import annotations

# AI-assisted implementation note:
# This script provides the repository's primary runnable entrypoint for the
# fused glucose forecasting pipeline.
#
# Design goal:
# - let a user run one script and have the code handle data preparation,
#   configuration, training, held-out testing, and artifact writing
# - keep the orchestration logic here thin by delegating model behavior to
#   `FusedModel`, data behavior to `AZT1DDataModule`, and default experiment
#   settings to `defaults.py`
#
# Important disclaimers:
# - this script is meant for research/development workflows, not for clinical
#   use
# - the default hyperparameters are baseline starting points and may need
#   tuning for meaningful experiments
# - successful execution still depends on the local Python environment having
#   the required packages installed, including `torch` and
#   `pytorch_lightning`

import argparse
import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from defaults import (
    DEFAULT_AZT1D_URL,
    DEFAULT_OUTPUT_DIR,
    ROOT_DIR,
    build_default_config,
    build_default_observability_config,
    build_default_snapshot_config,
    build_default_train_config,
)

from config import (
    Config,
    ObservabilityConfig,
    SnapshotConfig,
    TrainConfig,
    config_to_dict,
)
from environment import (
    DEVICE_PROFILE_CHOICES,
    RuntimeDiagnostic,
    RuntimeEnvironment,
    analyze_runtime_failure,
    collect_runtime_diagnostics,
    detect_runtime_environment,
    format_runtime_diagnostics,
    has_error_diagnostics,
    infer_device_profile,
    resolve_device_profile,
)

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


@dataclass(frozen=True)
class MainRunArtifacts:
    """
    Stable summary of the major artifacts produced by a top-level run.

    Purpose:
    provide one stable object describing the major outputs of a top-level
    workflow execution.

    Context:
    the CLI, tests, and notebooks all receive the same named result contract,
    which keeps downstream inspection much clearer than relying on positional
    tuples or peeking into mutable workflow internals.

    Evaluation note:
    `test_metrics` reflects Lightning's reduced scalar test output, while
    `test_evaluation` carries the repository's richer structured detailed
    evaluation derived from raw test predictions plus aligned targets/metadata.
    """
    fit: FitArtifacts
    test_metrics: list[Mapping[str, float]] | None
    test_evaluation: EvaluationResult | None
    test_predictions: list[torch.Tensor] | None
    summary: dict[str, Any]
    summary_path: Path | None
    predictions_path: Path | None
    prediction_table_path: Path | None
    report_paths: dict[str, Path]
    telemetry_path: Path | None
    logger_dir: Path | None
    text_log_path: Path | None
    requested_device_profile: str
    resolved_device_profile: str
    applied_profile_defaults: dict[str, Any]
    runtime_environment: RuntimeEnvironment
    preflight_diagnostics: tuple[RuntimeDiagnostic, ...]


def _json_ready(value: Any) -> Any:
    # Run summaries are persisted as JSON, so this helper normalizes a mix of
    # Paths, dataclasses, lists, and plain mappings into JSON-friendly values.
    #
    # The goal is not to serialize every arbitrary Python object; it is simply
    # to make this script's own artifact metadata readable and portable.
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if is_dataclass(value):
        return {
            field_info.name: _json_ready(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    return value


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    # Several CLI flags accept comma-separated lists because that is friendlier
    # to shell usage than repeating a flag or forcing JSON syntax.
    #
    # Example:
    # `--tcn-channels 64,64,128`
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    # Mirrors `_parse_csv_ints(...)`, but for values like quantiles.
    # Example:
    # `--quantiles 0.1,0.5,0.9`
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_devices(value: str) -> str | int | list[int]:
    # Lightning accepts several device formats:
    # - `"auto"` for automatic placement
    # - a single int like `1` for one device / one process
    # - a list like `[0, 1]` for explicit device indices
    #
    # This helper lets the CLI stay simple while still mapping cleanly onto the
    # richer Trainer API.
    cleaned = value.strip()
    if cleaned == "auto":
        return "auto"
    if "," in cleaned:
        return [int(part.strip()) for part in cleaned.split(",") if part.strip()]
    try:
        return int(cleaned)
    except ValueError:
        return cleaned


def _parse_limit(value: str) -> int | float:
    # Lightning interprets integers and floats differently for loop limits:
    # - `10` means "run exactly 10 batches"
    # - `0.25` means "run 25% of the loader"
    #
    # Parsing here preserves that distinction from the command line.
    cleaned = value.strip()
    if any(character in cleaned for character in (".", "e", "E")):
        return float(cleaned)
    return int(cleaned)


def _normalize_optional_string(value: str | None) -> str | None:
    # The CLI passes many values through as strings. This helper makes flags
    # like `--dataset-url none` or `--fit-ckpt-path null` behave the way users
    # usually mean them: "treat this as no value".
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "null"}:
        return None
    return cleaned


def _collect_explicit_cli_overrides(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> set[str]:
    # The profile resolver needs to know which flags were explicitly provided
    # so it can treat the device profile as a source of defaults rather than an
    # unconditional override.
    explicit_overrides: set[str] = set()
    option_actions = getattr(parser, "_option_string_actions", {})
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None:
            explicit_overrides.add(action.dest)
    return explicit_overrides


def _print_runtime_diagnostics(
    diagnostics: Sequence[RuntimeDiagnostic],
) -> None:
    if not diagnostics:
        return
    print("Runtime diagnostics:")
    print(format_runtime_diagnostics(diagnostics))


def _resolve_eval_ckpt_path(
    fit_artifacts: FitArtifacts,
    eval_ckpt_path: CheckpointSelection,
) -> CheckpointSelection:
    # A top-level workflow should degrade gracefully when there is no
    # validation-ranked "best" checkpoint to reload. In that case we fall back
    # to the in-memory model weights from the just-finished fit run.
    if eval_ckpt_path == "best" and not fit_artifacts.best_checkpoint_path:
        return None
    return eval_ckpt_path


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
) -> dict[str, Any]:
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
    effective_snapshot_config = snapshot_config or build_default_snapshot_config(
        output_dir=output_dir,
    )
    effective_observability_config = observability_config or build_default_observability_config(
        output_dir=output_dir,
    )
    effective_runtime_environment = runtime_environment or detect_runtime_environment()
    effective_resolved_device_profile = resolved_device_profile or infer_device_profile(
        requested_device_profile,
        effective_runtime_environment,
    )
    effective_preflight_diagnostics = tuple(
        preflight_diagnostics
        if preflight_diagnostics is not None
        else collect_runtime_diagnostics(
            requested_profile=requested_device_profile,
            resolved_profile=effective_resolved_device_profile,
            environment=effective_runtime_environment,
            train_config=effective_train_config,
            data_config=config.data,
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

    datamodule = AZT1DDataModule(config.data)
    # The training wrapper deliberately sits between the top-level workflow and
    # Lightning. That keeps `main.py` focused on orchestration while
    # `src/train.py` owns fit/test/predict policy.
    trainer = trainer_class(
        config,
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
                sampling_interval_minutes=config.data.sampling_interval_minutes,
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

    summary = _build_run_summary(
        config=config,
        train_config=effective_train_config,
        snapshot_config=effective_snapshot_config,
        observability_config=effective_observability_config,
        requested_device_profile=requested_device_profile,
        resolved_device_profile=effective_resolved_device_profile,
        applied_profile_defaults=applied_profile_defaults or {},
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
        applied_profile_defaults=dict(applied_profile_defaults or {}),
        runtime_environment=effective_runtime_environment,
        preflight_diagnostics=effective_preflight_diagnostics,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    # The CLI intentionally exposes the most common knobs for local
    # experimentation without trying to mirror every field from every nested
    # dataclass. If the project later needs a richer config surface, this can
    # evolve toward config-file support.
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the fused glucose forecasting model from one entrypoint."
        )
    )
    # Data location / preprocessing flags.
    parser.add_argument("--dataset-url", default=DEFAULT_AZT1D_URL)
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / "data" / "raw"))
    parser.add_argument("--cache-dir", default=str(ROOT_DIR / "data" / "cache"))
    parser.add_argument("--extract-dir", default=str(ROOT_DIR / "data" / "extracted"))
    parser.add_argument("--processed-dir", default=str(ROOT_DIR / "data" / "processed"))
    parser.add_argument("--processed-file-name", default="azt1d_processed.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint-dir", default=None)
    # Sequence and data loader sizing.
    parser.add_argument("--encoder-length", type=int, default=168)
    parser.add_argument("--prediction-length", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
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
    # Trainer/runtime controls.
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument(
        "--device-profile",
        default="auto",
        choices=DEVICE_PROFILE_CHOICES,
    )
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32")
    # Optimizer controls.
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--seed", type=int, default=42)
    # Model-shape controls.
    parser.add_argument("--tcn-channels", default="64,64,128")
    parser.add_argument("--tcn-dilations", default="1,2,4")
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--tft-hidden-size", type=int, default=128)
    parser.add_argument("--tft-n-head", type=int, default=4)
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")
    # Loop debugging / truncation controls.
    parser.add_argument("--limit-train-batches", default="1.0")
    parser.add_argument("--limit-val-batches", default="1.0")
    parser.add_argument("--limit-test-batches", default="1.0")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    # Checkpoint-selection controls.
    parser.add_argument("--fit-ckpt-path", default=None)
    parser.add_argument("--eval-ckpt-path", default="best")
    # Behavior toggles.
    parser.add_argument("--rebuild-processed", action="store_true")
    parser.add_argument("--redownload", action="store_true")
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
    parser.add_argument("--run-diagnostics-only", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--disable-checkpoints", action="store_true")
    parser.add_argument("--save-weights-only", action="store_true")
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
    # This parser intentionally does not try to expose every possible nested
    # config field. The goal is a practical top-level entrypoint, not a
    # one-to-one command-line mirror of every internal dataclass.
    return parser


def main(argv: Sequence[str] | None = None) -> MainRunArtifacts | None:
    # `main(...)` keeps argument parsing separate from the reusable workflow
    # function above. That split makes the same training flow callable from
    # tests and notebooks without forcing everything through `argparse`.
    parser = build_argument_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    explicit_overrides = _collect_explicit_cli_overrides(parser, raw_argv)

    output_dir = Path(args.output_dir)
    # Build the nested project config from CLI primitives. This is the point
    # where "flat command-line arguments" become the structured config objects
    # used throughout the rest of the codebase.
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
    config = Config(
        data=profile_resolution.data_config,
        tft=config.tft,
        tcn=config.tcn,
    )
    train_config = profile_resolution.train_config
    observability_config = profile_resolution.observability_config
    preflight_diagnostics = collect_runtime_diagnostics(
        requested_profile=profile_resolution.requested_profile,
        resolved_profile=profile_resolution.resolved_profile,
        environment=runtime_environment,
        train_config=train_config,
        data_config=config.data,
        observability_config=observability_config,
    )
    # At this point we have three layers of structured configuration:
    # - `config` for data/model semantics
    # - `train_config` for Trainer behavior
    # - `snapshot_config` for checkpoint policy
    # - `observability_config` for logging, telemetry, visualization, and
    #   debug/report artifacts

    if args.run_diagnostics_only or not has_error_diagnostics(preflight_diagnostics):
        print(
            "Device profile: "
            f"{profile_resolution.requested_profile} -> {profile_resolution.resolved_profile}"
        )
        _print_runtime_diagnostics(preflight_diagnostics)

    if args.run_diagnostics_only:
        print("Diagnostics-only mode enabled; training was not started.")
        return None

    try:
        artifacts = run_training_workflow(
            config,
            train_config=train_config,
            snapshot_config=snapshot_config,
            observability_config=observability_config,
            requested_device_profile=profile_resolution.requested_profile,
            resolved_device_profile=profile_resolution.resolved_profile,
            applied_profile_defaults=profile_resolution.applied_defaults,
            runtime_environment=runtime_environment,
            preflight_diagnostics=preflight_diagnostics,
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
            output_dir=output_dir,
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

    # The printed summary is intentionally concise. Detailed state is written to
    # the JSON summary file and, when requested, the raw prediction tensor file.
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
    torchview_path = artifacts.summary.get("observability_config", {}).get("torchview_path")
    if torchview_path is not None:
        print(f"Torchview output base path: {torchview_path}")
    if artifacts.summary_path is not None:
        print(f"Saved run summary to: {artifacts.summary_path}")

    return artifacts


if __name__ == "__main__":
    main()
