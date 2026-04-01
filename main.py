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
from typing import Any, Mapping, Sequence

import torch

from defaults import (
    DEFAULT_AZT1D_URL,
    DEFAULT_OUTPUT_DIR,
    ROOT_DIR,
    build_default_config,
    build_default_snapshot_config,
    build_default_train_config,
)

from data.datamodule import AZT1DDataModule
from train import CheckpointSelection, FitArtifacts, FusedModelTrainer
from utils.config import Config, SnapshotConfig, TrainConfig, config_to_dict

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
    # This dataclass gives callers one stable object to inspect after a run.
    # That makes the notebook experience nicer than having to remember tuple
    # positions, and it keeps the CLI/workflow code explicit about which
    # artifacts are expected to exist.
    fit: FitArtifacts
    test_metrics: list[Mapping[str, float]] | None
    test_predictions: list[torch.Tensor] | None
    summary: dict[str, Any]
    summary_path: Path | None
    predictions_path: Path | None


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
    fit_artifacts: FitArtifacts,
    eval_ckpt_path: CheckpointSelection,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    output_dir: Path | None,
    test_metrics: list[Mapping[str, float]] | None,
    predictions: list[torch.Tensor] | None,
    predictions_path: Path | None,
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
        "optimizer": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer_name": optimizer_name,
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
        },
        "predictions": {
            "num_batches": 0 if predictions is None else len(predictions),
            "batch_shapes": []
            if predictions is None
            else [list(tensor.shape) for tensor in predictions],
            "saved_to": str(predictions_path) if predictions_path is not None else None,
        },
    }


def run_training_workflow(
    config: Config,
    *,
    train_config: TrainConfig | None = None,
    snapshot_config: SnapshotConfig | None = None,
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
    trainer_class: type[FusedModelTrainer] = FusedModelTrainer,
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
    # If the caller supplied explicit configs we respect them verbatim.
    # Otherwise we derive defaults that are aligned with the chosen output dir.

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
    )

    fit_artifacts = trainer.fit(datamodule, ckpt_path=fit_ckpt_path)
    # Evaluation may target `"best"`, `"last"`, an explicit checkpoint path,
    # or the current in-memory model. We normalize that choice once here so
    # test and prediction stay consistent.
    resolved_eval_ckpt_path = _resolve_eval_ckpt_path(fit_artifacts, eval_ckpt_path)

    test_metrics: list[Mapping[str, float]] | None = None
    test_predictions: list[torch.Tensor] | None = None
    predictions_path: Path | None = None

    if fit_artifacts.has_test_data and not skip_test:
        test_metrics = trainer.test(datamodule, ckpt_path=resolved_eval_ckpt_path)
    # We intentionally treat test evaluation and raw prediction collection as
    # separate toggles. Some workflows want metrics only, while others want the
    # raw tensors for notebook analysis or custom visualization.

    if fit_artifacts.has_test_data and not skip_predict:
        test_predictions = trainer.predict_test(
            datamodule,
            ckpt_path=resolved_eval_ckpt_path,
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

    summary = _build_run_summary(
        config=config,
        train_config=effective_train_config,
        snapshot_config=effective_snapshot_config,
        fit_artifacts=fit_artifacts,
        eval_ckpt_path=resolved_eval_ckpt_path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        output_dir=output_dir,
        test_metrics=test_metrics,
        predictions=test_predictions,
        predictions_path=predictions_path,
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
        test_predictions=test_predictions,
        summary=summary,
        summary_path=summary_path,
        predictions_path=predictions_path,
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
    # Trainer/runtime controls.
    parser.add_argument("--max-epochs", type=int, default=20)
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
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--disable-checkpoints", action="store_true")
    parser.add_argument("--save-weights-only", action="store_true")
    # This parser intentionally does not try to expose every possible nested
    # config field. The goal is a practical top-level entrypoint, not a
    # one-to-one command-line mirror of every internal dataclass.
    return parser


def main(argv: Sequence[str] | None = None) -> MainRunArtifacts:
    # `main(...)` keeps argument parsing separate from the reusable workflow
    # function above. That split makes the same training flow callable from
    # tests and notebooks without forcing everything through `argparse`.
    parser = build_argument_parser()
    args = parser.parse_args(argv)

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
        default_root_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
    )
    snapshot_config = build_default_snapshot_config(
        enabled=not args.disable_checkpoints,
        output_dir=output_dir,
        dirpath=None if args.checkpoint_dir is None else Path(args.checkpoint_dir),
        save_weights_only=args.save_weights_only,
    )
    # At this point we have three layers of structured configuration:
    # - `config` for data/model semantics
    # - `train_config` for Trainer behavior
    # - `snapshot_config` for checkpoint policy

    artifacts = run_training_workflow(
        config,
        train_config=train_config,
        snapshot_config=snapshot_config,
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
    # The returned artifact bundle is the same shape notebooks and tests
    # receive from `run_training_workflow(...)`, so CLI and non-CLI entrypoints
    # share one result contract.

    # The printed summary is intentionally concise. Detailed state is written to
    # the JSON summary file and, when requested, the raw prediction tensor file.
    print(f"Processed data: {config.data.processed_file_path}")
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
    if artifacts.summary_path is not None:
        print(f"Saved run summary to: {artifacts.summary_path}")

    return artifacts


if __name__ == "__main__":
    main()
