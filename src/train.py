from __future__ import annotations

# AI-assisted implementation note:
# This file was drafted with AI assistance and then reviewed/adapted for this
# project. It provides the repository's PyTorch Lightning training wrapper for
# the fused TCN + TFT forecasting stack.
#
# Design intent:
# - keep model-specific learning behavior inside `FusedModel`
# - keep data preparation and batching inside `AZT1DDataModule`
# - keep Lightning `Trainer` orchestration, checkpoint snapshots, validation,
#   test, and prediction flow in one reusable module that can be called from
#   `main.py` or notebooks
#
# Scope:
# - this file does not redefine the training step, loss, or optimizer logic
# - this file does coordinate how a prepared DataModule, a bound model config,
#   callbacks, checkpoints, and Lightning's Trainer interact during a run

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Sized, cast
from dataclasses import dataclass

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch import Tensor

from data.datamodule import AZT1DDataModule
from models.fused_model import FusedModel
from utils.config import Config, SnapshotConfig, TrainConfig


CheckpointAlias = Literal["best", "last"]
CheckpointSelection = CheckpointAlias | str | Path | None

@dataclass(frozen=True)
class FitArtifacts:
    # Lightweight record returned by `fit(...)` so outer scripts can inspect
    # the exact model/trainer/runtime-config combination used for the run
    # without poking at mutable trainer-wrapper internals.
    model: FusedModel
    runtime_config: Config
    trainer: Trainer
    has_validation_data: bool
    has_test_data: bool
    best_checkpoint_path: str


@dataclass(frozen=True)
class TrainingRunArtifacts:
    # Convenience container for the common "train, then evaluate, then collect
    # held-out predictions" workflow. Keeping this explicit makes notebook
    # experimentation less error-prone than returning a loosely structured
    # tuple whose ordering has to be remembered by the caller.
    fit: FitArtifacts
    test_metrics: list[Mapping[str, float]] | None
    test_predictions: list[Tensor] | None


def _dataset_size(dataset: Sized | None) -> int:
    # Small helper to keep all "does this split actually have any windows?"
    # checks consistent across fit/validation/test orchestration.
    #
    # Returning `0` for `None` keeps the caller logic simple because the
    # DataModule starts with datasets unset and only materializes them during
    # `setup()`.
    return 0 if dataset is None else len(dataset)


class FusedModelTrainer:
    """
    Thin orchestration layer around Lightning's Trainer for this repository.

    Responsibility boundary:
    - `FusedModel` owns forward/loss/optimizer behavior
    - `AZT1DDataModule` owns data preparation and loaders
    - this class owns the Trainer lifecycle: fit, validation, test, prediction,
      and optional checkpoint snapshots

    This keeps `main.py` or notebooks free to handle experiment-specific setup
    without rebuilding the same Lightning orchestration glue each time.
    """

    def __init__(
        self,
        config: Config,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_name: str = "adam",
        trainer_config: TrainConfig | None = None,
        snapshot_config: SnapshotConfig | None = None,
        logger: Any = None,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        # `config` is the declarative project config. It is intentionally kept
        # unmodified here because the runtime-bound variant depends on the
        # DataModule's discovered metadata and is therefore produced later.
        self.config = config

        # These optimizer settings are passed straight through to the
        # `FusedModel`, where Lightning expects optimizer construction to live.
        # The wrapper stores them only so it can recreate the model
        # deterministically after binding the runtime config.
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name

        # `TrainConfig` and `SnapshotConfig` capture the Trainer-side policy for
        # the run: devices, limits, precision, progress UI, checkpoints, and
        # early stopping. Default instances keep the wrapper immediately usable
        # in notebooks without forcing the caller to construct every config by
        # hand for a first experiment.
        self.trainer_config = trainer_config or TrainConfig()
        self.snapshot_config = snapshot_config or SnapshotConfig()

        # `logger` and custom callbacks are left fully injectable because those
        # often vary from one experiment surface to another even when the
        # underlying model/data config stays the same.
        self.logger = logger
        self.callbacks = tuple(callbacks)

        # The fields below are populated lazily as the wrapper progresses
        # through `fit(...)`. Keeping them on the instance allows later calls to
        # `test(...)` or `predict_test(...)` to reuse the same in-memory run
        # state when appropriate.
        self.model: FusedModel | None = None
        self.runtime_config: Config | None = None
        self.trainer: Trainer | None = None
        self.best_checkpoint_path: str = ""

    def _prepare_datamodule(self, datamodule: AZT1DDataModule) -> None:
        # We prepare/setup eagerly because runtime categorical metadata must be
        # known before `FusedModel` can be constructed with a bound TFT config.
        #
        # In a typical Lightning project, the Trainer can call DataModule hooks
        # itself during `fit(...)`. We do a pre-pass here intentionally because
        # this repository binds runtime-discovered categorical cardinalities and
        # feature metadata into the model config *before* instantiating the
        # `FusedModel`.
        #
        # That means this wrapper has a real dependency on the post-setup
        # DataModule state, not just on the DataModule object existing.
        datamodule.prepare_data()
        datamodule.setup()

    def build_model(self, datamodule: AZT1DDataModule) -> FusedModel:
        # Step 1: make sure the DataModule has discovered all runtime metadata
        # needed by the TFT branch.
        self._prepare_datamodule(datamodule)

        # Step 2: bind the declarative top-level config to the actual data
        # contract seen after `setup()`. This is where categorical cardinalities
        # and fallback feature specs become concrete.
        runtime_config = datamodule.bind_model_config(self.config)

        # Step 3: instantiate the LightningModule with the bound config plus the
        # optimizer hyperparameters owned by the model-side training contract.
        model = FusedModel(
            runtime_config,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer_name=self.optimizer_name,
        )

        self.runtime_config = runtime_config
        self.model = model
        return model

    def has_validation_data(self, datamodule: AZT1DDataModule) -> bool:
        # Exposed as a helper because notebook/main callers may want to inspect
        # split availability before actually launching training.
        self._prepare_datamodule(datamodule)
        return _dataset_size(datamodule.val_dataset) > 0

    def has_test_data(self, datamodule: AZT1DDataModule) -> bool:
        # Same idea as `has_validation_data(...)`: keep split-availability
        # checks in one place rather than repeating DataModule setup and length
        # logic in outer scripts.
        self._prepare_datamodule(datamodule)
        return _dataset_size(datamodule.test_dataset) > 0

    def build_callbacks(self, *, has_validation_data: bool) -> list[Callback]:
        # Start from any user-supplied callbacks so the wrapper remains
        # extensible; the project-specific callbacks are appended after that so
        # they can coexist with logger-specific or experiment-specific hooks.
        callbacks = list(self.callbacks)
        snapshot_config = self.snapshot_config

        if snapshot_config.enabled:
            # `ModelCheckpoint` accepts `None` to fall back to Lightning's
            # default root dir behavior. We preserve that behavior here rather
            # than forcing every caller to specify a snapshot folder.
            dirpath = (
                str(snapshot_config.dirpath)
                if snapshot_config.dirpath is not None
                else None
            )

            if has_validation_data:
                # When validation exists, checkpointing should usually be tied to
                # a validation metric. That gives us a meaningful "best" model
                # alias instead of saving only the final epoch.
                #
                # The filename pattern defaults to including epoch and
                # validation loss because those are the two most useful pieces
                # of information when browsing saved snapshots by hand.
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=dirpath,
                        filename=snapshot_config.filename,
                        monitor=snapshot_config.monitor,
                        mode=snapshot_config.mode,
                        save_top_k=snapshot_config.save_top_k,
                        save_last=snapshot_config.save_last,
                        save_weights_only=snapshot_config.save_weights_only,
                    )
                )
            else:
                # Without validation data there is nothing meaningful to
                # monitor, so we fall back to "last snapshot only" semantics.
                #
                # We deliberately avoid pretending a "best" checkpoint exists in
                # this case because there is no monitored validation signal to
                # rank snapshots by.
                #
                # `save_top_k=0` + `save_last=True` expresses exactly that
                # policy: do not maintain a ranked set of top checkpoints, but
                # still keep the latest weights if snapshotting is enabled.
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=dirpath,
                        filename="last",
                        save_top_k=0,
                        save_last=snapshot_config.save_last,
                        save_weights_only=snapshot_config.save_weights_only,
                    )
                )

        if has_validation_data and self.trainer_config.early_stopping_patience is not None:
            # Early stopping is only sensible when validation data exists and
            # the caller has not disabled it via `early_stopping_patience=None`.
            #
            # We reuse the same monitor/mode as the snapshot config so the
            # "which model should we save?" and "when should training stop?"
            # decisions stay aligned.
            callbacks.append(
                EarlyStopping(
                    monitor=self.snapshot_config.monitor,
                    mode=self.snapshot_config.mode,
                    patience=self.trainer_config.early_stopping_patience,
                )
            )

        return callbacks

    def build_trainer(self, *, has_validation_data: bool) -> Trainer:
        # Keep Trainer construction data-driven so `main.py` or notebooks can
        # configure runs by passing a typed `TrainConfig` rather than rebuilding
        # this argument list each time.
        config = self.trainer_config

        # We assemble kwargs in a dictionary first for two reasons:
        # 1. it keeps the mapping from `TrainConfig` fields to Lightning
        #    arguments visually obvious
        # 2. it makes the one conditional field (`default_root_dir`) easier to
        #    inject without duplicating the entire constructor call
        trainer_kwargs: dict[str, Any] = {
            "accelerator": config.accelerator,
            "devices": config.devices,
            "precision": config.precision,
            "max_epochs": config.max_epochs,
            "deterministic": config.deterministic,
            "log_every_n_steps": config.log_every_n_steps,
            "num_sanity_val_steps": (
                config.num_sanity_val_steps if has_validation_data else 0
            ),
            "fast_dev_run": config.fast_dev_run,
            "limit_train_batches": config.limit_train_batches,
            "limit_val_batches": config.limit_val_batches,
            "limit_test_batches": config.limit_test_batches,
            "enable_progress_bar": config.enable_progress_bar,
            "enable_model_summary": config.enable_model_summary,
            "logger": self.logger,
            "callbacks": self.build_callbacks(has_validation_data=has_validation_data),
        }
        if config.default_root_dir is not None:
            # Lightning accepts `str | PathLike`, but we normalize to plain
            # strings here for consistency with the rest of the wrapper.
            trainer_kwargs["default_root_dir"] = str(config.default_root_dir)

        # From this point on, Lightning owns the actual epoch loop mechanics:
        # gradient steps, validation scheduling, accelerator/device placement,
        # precision behavior, callback dispatch, and loop bookkeeping.
        trainer = Trainer(**trainer_kwargs)
        self.trainer = trainer
        return trainer

    def fit(
        self,
        datamodule: AZT1DDataModule,
        *,
        ckpt_path: str | Path | None = None,
    ) -> FitArtifacts:
        # `fit(...)` is the primary orchestration entry point:
        # 1. build the bound model from the prepared DataModule
        # 2. decide whether validation/test splits are actually populated
        # 3. build the Lightning Trainer with the right callback policy
        # 4. hand control of the epoch loop to Lightning
        model = self.build_model(datamodule)
        has_validation_data = _dataset_size(datamodule.val_dataset) > 0
        has_test_data = _dataset_size(datamodule.test_dataset) > 0
        trainer = self.build_trainer(has_validation_data=has_validation_data)

        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(ckpt_path) if ckpt_path is not None else None,
        )

        # Cache the best checkpoint path after fitting so later calls to
        # `test(..., ckpt_path="best")` or `predict_test(..., ckpt_path="best")`
        # can reuse Lightning's ranking result without the outer caller needing
        # to inspect callback state directly.
        #
        # This cached path is especially helpful in notebooks where the user may
        # call `fit(...)` in one cell and `test(...)` in another.
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        self.best_checkpoint_path = (
            checkpoint_callback.best_model_path
            if isinstance(checkpoint_callback, ModelCheckpoint)
            else ""
        )

        return FitArtifacts(
            model=model,
            runtime_config=self.runtime_config or self.config,
            trainer=trainer,
            has_validation_data=has_validation_data,
            has_test_data=has_test_data,
            best_checkpoint_path=self.best_checkpoint_path,
        )

    def _require_fit_state(self) -> tuple[FusedModel, Trainer]:
        # The current implementation supports in-memory evaluation/prediction
        # after a training run. This guard makes that contract explicit instead
        # of failing later with a less readable `NoneType` error.
        #
        # It also documents an important current limitation of the wrapper:
        # pure checkpoint-only evaluation without a prior `fit()` call is not
        # fully supported yet.
        if self.model is None or self.trainer is None:
            raise RuntimeError(
                "fit() must be called before using in-memory evaluation or prediction."
            )
        return self.model, self.trainer

    def _resolve_checkpoint_reference(self, ckpt_path: CheckpointSelection) -> str | None:
        # This helper accepts three checkpoint styles:
        # - `None`: use the current in-memory model
        # - `"best"` / `"last"`: let Lightning resolve a standard alias
        # - explicit filesystem path: load that checkpoint directly
        #
        # Normalizing the selection in one place keeps `test(...)` and
        # `predict_test(...)` behavior aligned and avoids duplicating alias
        # validation logic.
        if ckpt_path is None:
            return None
        if ckpt_path in ("best", "last"):
            if self.trainer is None:
                raise RuntimeError(
                    "Checkpoint aliases 'best' and 'last' are only available after fit()."
                )
            if ckpt_path == "best" and not self.best_checkpoint_path:
                # If there was no validation-monitored checkpoint, Lightning has
                # no meaningful "best" model to recover. Raising here avoids a
                # confusing downstream failure inside the Trainer.
                raise RuntimeError(
                    "No best checkpoint snapshot is available. "
                    "Run fit() with validation snapshots enabled, or pass "
                    "`ckpt_path=None` to evaluate the current in-memory weights."
                )
            alias: CheckpointAlias = "best" if ckpt_path == "best" else "last"
            return alias
        return str(Path(ckpt_path))

    def _model_for_evaluation(self, resolved_ckpt_path: str | None) -> FusedModel:
        # If the caller asked for in-memory evaluation, or for Lightning's
        # built-in `"best"` / `"last"` aliases, we keep using the model already
        # attached to the current training session.
        #
        # For an explicit checkpoint path we create a fresh model instance via
        # Lightning's checkpoint loader so the evaluation path can read weights
        # from disk instead of relying on the current in-memory state.
        #
        # `FusedModel` already supports checkpoint-friendly config restoration,
        # so `load_from_checkpoint(...)` can rebuild the model constructor
        # arguments from the saved hyperparameters.
        if resolved_ckpt_path is None or resolved_ckpt_path in ("best", "last"):
            model, _ = self._require_fit_state()
            return model
        return FusedModel.load_from_checkpoint(resolved_ckpt_path)

    def test(
        self,
        datamodule: AZT1DDataModule,
        *,
        ckpt_path: CheckpointSelection = "best",
    ) -> list[Mapping[str, float]]:
        # We re-run DataModule preparation here so test-time calls from a
        # notebook remain robust even if they happen in a later cell after the
        # original fit call.
        #
        # The preparation path is designed to be idempotent: `prepare_data()`
        # should skip redundant work when the processed dataset already exists,
        # and `setup()` should simply refresh the in-memory split datasets.
        #
        # Current limitation:
        # this method still assumes the wrapper has an in-memory Trainer/model
        # from a prior `fit()` call. Explicit checkpoint paths can already be
        # used to choose which weights to evaluate, but the surrounding Trainer
        # session is not yet rebuilt from scratch for a fully standalone
        # "evaluate-only" workflow.
        self._prepare_datamodule(datamodule)
        if _dataset_size(datamodule.test_dataset) == 0:
            raise RuntimeError("Cannot run test() because the DataModule has no test windows.")

        resolved_ckpt_path = self._resolve_checkpoint_reference(ckpt_path)
        model = self._model_for_evaluation(resolved_ckpt_path)
        _, trainer = self._require_fit_state()

        # Lightning handles the test loop itself; this wrapper just chooses the
        # model/checkpoint source and provides the correct DataModule.
        return trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=resolved_ckpt_path,
        )

    def predict_test(
        self,
        datamodule: AZT1DDataModule,
        *,
        ckpt_path: CheckpointSelection = "best",
    ) -> list[Tensor]:
        # `predict_test(...)` mirrors `test(...)`, but routes through
        # `trainer.predict(...)` so the caller gets the raw probabilistic output
        # tensors from `FusedModel.predict_step(...)`.
        #
        # Current limitation:
        # like `test(...)`, this prediction path still expects the wrapper to
        # have an in-memory Trainer/model from an earlier `fit()` call. It is
        # therefore ideal for the common "train, then predict on held-out test
        # windows" workflow, but not yet a fully standalone checkpoint-only
        # inference entry point.
        self._prepare_datamodule(datamodule)
        if _dataset_size(datamodule.test_dataset) == 0:
            raise RuntimeError(
                "Cannot run predict_test() because the DataModule has no test windows."
            )

        resolved_ckpt_path = self._resolve_checkpoint_reference(ckpt_path)
        model = self._model_for_evaluation(resolved_ckpt_path)
        _, trainer = self._require_fit_state()

        # We pass the explicit test dataloader rather than the whole DataModule
        # because this method is specifically about producing predictions for
        # the held-out test split, not for every possible prediction loader a
        # future DataModule might expose.
        #
        # The returned list is Lightning's standard prediction output shape:
        # one tensor per prediction batch, where each tensor contains the raw
        # quantile forecasts emitted by `FusedModel.predict_step(...)`.
        predictions = trainer.predict(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=resolved_ckpt_path,
        )

        # Lightning types `predict(...)` broadly because it supports multiple
        # dataloaders and arbitrary prediction-step return shapes. In this
        # wrapper we call it with exactly one test dataloader, and
        # `FusedModel.predict_step(...)` returns a `Tensor`, so the concrete
        # runtime shape we expect here is `list[Tensor]`.
        if predictions is None:
            return []
        return cast(list[Tensor], predictions)

    def fit_test_predict(
        self,
        datamodule: AZT1DDataModule,
        *,
        fit_ckpt_path: str | Path | None = None,
        eval_ckpt_path: CheckpointSelection = "best",
    ) -> TrainingRunArtifacts:
        # Convenience method for the common workflow used in experiments:
        # train the model, evaluate the held-out test split, and collect raw
        # test predictions in one call.
        #
        # This method is intentionally thin composition over the public
        # `fit(...)`, `test(...)`, and `predict_test(...)` methods so there is
        # still one canonical implementation for each stage rather than a second
        # hidden training path.
        fit_artifacts = self.fit(datamodule, ckpt_path=fit_ckpt_path)

        if not fit_artifacts.has_test_data:
            # Returning `None` payloads keeps the result explicit while avoiding
            # an exception for datasets/split policies that legitimately produce
            # no test windows.
            return TrainingRunArtifacts(
                fit=fit_artifacts,
                test_metrics=None,
                test_predictions=None,
            )

        # If the caller asked for `"best"` but the fit run did not produce a
        # validation-ranked checkpoint, fall back to the current in-memory
        # weights so the common "train, then test" workflow still completes.
        resolved_eval_ckpt_path: CheckpointSelection = eval_ckpt_path
        if eval_ckpt_path == "best" and not fit_artifacts.best_checkpoint_path:
            resolved_eval_ckpt_path = None

        test_metrics = self.test(datamodule, ckpt_path=resolved_eval_ckpt_path)
        test_predictions = self.predict_test(
            datamodule,
            ckpt_path=resolved_eval_ckpt_path,
        )
        return TrainingRunArtifacts(
            fit=fit_artifacts,
            test_metrics=test_metrics,
            test_predictions=test_predictions,
        )
