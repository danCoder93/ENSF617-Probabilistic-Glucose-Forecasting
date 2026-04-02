from __future__ import annotations

"""
AI-assisted maintenance note:
These tests protect the trainer wrapper's construction-time responsibilities.

Purpose:
- verify runtime data metadata is bound into the model config before model creation
- verify callback and Trainer assembly reflect the typed runtime configuration
- verify dataloader settings are preserved where they materially affect runtime behavior

Context:
these tests focus on setup and construction policy, not on the actual epoch
loop. Execution-path behavior lives in the companion `test_trainer_execution.py`
module.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pytorch_lightning")
import train as train_module
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from data.datamodule import AZT1DDataModule
from observability import (
    BatchAuditCallback,
    GradientStatsCallback,
    ModelTensorBoardCallback,
    ParameterHistogramCallback,
    ParameterScalarTelemetryCallback,
    PredictionFigureCallback,
    SystemTelemetryCallback,
)
from train import FusedModelTrainer
from config import DataConfig, SnapshotConfig, TrainConfig
from tests.support import build_base_config, build_minimal_data_config


def test_build_model_binds_runtime_tft_metadata(
    write_processed_csv,
    build_data_config,
) -> None:
    # The trainer wrapper must bind runtime-discovered categorical metadata into
    # the TFT config before the model is instantiated. If that contract breaks,
    # the model and DataModule no longer agree on feature cardinalities.
    csv_path = write_processed_csv(subject_ids=("subject_a", "subject_b"), steps_per_subject=40)
    data_config = build_data_config(csv_path, batch_size=2)
    datamodule = AZT1DDataModule(data_config)
    trainer = FusedModelTrainer(build_base_config(data_config))

    model = trainer.build_model(datamodule)

    assert trainer.runtime_config is not None
    assert trainer.runtime_config.tft.static_categorical_inp_lens == (2,)
    assert trainer.runtime_config.tft.temporal_observed_categorical_inp_lens == (4, 10)
    assert trainer.runtime_config.tft.temporal_target_size == 1
    assert trainer.runtime_config.tft.encoder_length == data_config.encoder_length
    assert trainer.runtime_config.tft.example_length == (
        data_config.encoder_length + data_config.prediction_length
    )
    assert model.config == trainer.runtime_config


def test_has_validation_and_test_data_reflect_prepared_splits(
    write_processed_csv,
    build_data_config,
) -> None:
    # Split-availability helpers are thin, but they encode the wrapper's policy
    # of eagerly preparing the DataModule before answering "does this split exist?"
    csv_path = write_processed_csv(steps_per_subject=80)
    data_config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(data_config)
    trainer = FusedModelTrainer(build_base_config(data_config))

    assert trainer.has_validation_data(datamodule) is True
    assert trainer.has_test_data(datamodule) is True


def test_build_callbacks_monitor_validation_loss_when_validation_exists() -> None:
    # With validation data available, checkpointing and early stopping should
    # both key off the validation-loss contract used elsewhere in the repo.
    trainer = FusedModelTrainer(
        build_base_config(build_minimal_data_config()),
        trainer_config=TrainConfig(early_stopping_patience=3),
    )

    callbacks = trainer.build_callbacks(has_validation_data=True)

    checkpoint = next(
        callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
    )
    early_stopping = next(
        callback for callback in callbacks if isinstance(callback, EarlyStopping)
    )

    assert checkpoint.monitor == "val_loss"
    assert checkpoint.save_top_k == 1
    assert checkpoint.save_last is True
    assert checkpoint.save_weights_only is False
    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == 3
    assert any(isinstance(callback, LearningRateMonitor) for callback in callbacks)
    assert any(isinstance(callback, DeviceStatsMonitor) for callback in callbacks)
    if importlib.util.find_spec("rich") is not None:
        assert any(isinstance(callback, RichProgressBar) for callback in callbacks)
    else:
        assert not any(isinstance(callback, RichProgressBar) for callback in callbacks)
    assert any(isinstance(callback, BatchAuditCallback) for callback in callbacks)
    assert any(isinstance(callback, GradientStatsCallback) for callback in callbacks)
    assert any(isinstance(callback, SystemTelemetryCallback) for callback in callbacks)
    assert any(isinstance(callback, ModelTensorBoardCallback) for callback in callbacks)
    assert any(
        isinstance(callback, ParameterScalarTelemetryCallback) for callback in callbacks
    )
    assert any(isinstance(callback, ParameterHistogramCallback) for callback in callbacks)
    assert any(isinstance(callback, PredictionFigureCallback) for callback in callbacks)


def test_build_callbacks_can_snapshot_weights_only_without_validation() -> None:
    # Without validation data, the wrapper should downgrade from ranked
    # checkpointing to simple "last snapshot" semantics rather than pretending a
    # meaningful monitored best checkpoint exists.
    trainer = FusedModelTrainer(
        build_base_config(build_minimal_data_config()),
        snapshot_config=SnapshotConfig(
            enabled=True,
            dirpath=Path("snapshots"),
            save_weights_only=True,
        ),
        trainer_config=TrainConfig(early_stopping_patience=None),
    )

    callbacks = trainer.build_callbacks(has_validation_data=False)

    checkpoint = next(
        callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
    )

    assert isinstance(checkpoint, ModelCheckpoint)
    assert checkpoint.monitor is None
    assert checkpoint.save_top_k == 0
    assert checkpoint.save_last is True
    assert checkpoint.save_weights_only is True
    assert not any(isinstance(callback, EarlyStopping) for callback in callbacks)


def test_build_trainer_passes_runtime_tuning_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test treats the Trainer constructor as a translation boundary from
    # typed repo config to Lightning kwargs. A fake Trainer keeps that mapping
    # easy to inspect directly.
    observed_kwargs: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            observed_kwargs.update(kwargs)

    monkeypatch.setattr(train_module, "Trainer", FakeTrainer)

    trainer = FusedModelTrainer(
        build_base_config(
            DataConfig(
                dataset_url=None,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )
        ),
        trainer_config=TrainConfig(
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            gradient_clip_val=1.0,
            accumulate_grad_batches=2,
            strategy="auto",
            sync_batchnorm=False,
            matmul_precision="high",
            allow_tf32=True,
            cudnn_benchmark=True,
        ),
    )

    trainer.build_trainer(has_validation_data=True)

    assert observed_kwargs["gradient_clip_val"] == 1.0
    assert observed_kwargs["accumulate_grad_batches"] == 2
    assert observed_kwargs["strategy"] == "auto"
    assert observed_kwargs["sync_batchnorm"] is False


def test_datamodule_includes_prefetch_factor_only_when_workers_are_enabled(
    write_processed_csv,
    build_data_config,
) -> None:
    # Prefetch-related loader kwargs are only valid when workers are enabled.
    # This test protects that contract on the real DataModule output.
    csv_path = write_processed_csv()
    data_config = build_data_config(
        csv_path,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    datamodule = AZT1DDataModule(data_config)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()

    assert train_loader.prefetch_factor == 4
