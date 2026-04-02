from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("pytorch_lightning")
import train as train_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from data.datamodule import AZT1DDataModule
from models.fused_model import FusedModel
from observability import (
    BatchAuditCallback,
    GradientStatsCallback,
    ModelTensorBoardCallback,
    ParameterHistogramCallback,
    ParameterScalarTelemetryCallback,
    PredictionFigureCallback,
    SystemTelemetryCallback,
)
from train import FitArtifacts, FusedModelTrainer
from config import Config, DataConfig, SnapshotConfig, TCNConfig, TFTConfig, TrainConfig


def _build_base_config(data_config: DataConfig) -> Config:
    return Config(
        data=data_config,
        tft=TFTConfig(),
        tcn=TCNConfig(num_inputs=1, num_channels=(4,)),
    )


def _build_minimal_data_config() -> DataConfig:
    return DataConfig(
        dataset_url=None,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def test_build_model_binds_runtime_tft_metadata(
    write_processed_csv,
    build_data_config,
) -> None:
    csv_path = write_processed_csv(subject_ids=("subject_a", "subject_b"), steps_per_subject=40)
    data_config = build_data_config(csv_path, batch_size=2)
    datamodule = AZT1DDataModule(data_config)
    trainer = FusedModelTrainer(_build_base_config(data_config))

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


def test_build_callbacks_monitor_validation_loss_when_validation_exists() -> None:
    trainer = FusedModelTrainer(
        _build_base_config(_build_minimal_data_config()),
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
    assert any(isinstance(callback, RichProgressBar) for callback in callbacks)
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
    trainer = FusedModelTrainer(
        _build_base_config(_build_minimal_data_config()),
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


def test_checkpoint_alias_requires_fit_before_evaluation(
    write_processed_csv,
    build_data_config,
) -> None:
    csv_path = write_processed_csv()
    data_config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(data_config)
    trainer = FusedModelTrainer(_build_base_config(data_config))

    with pytest.raises(RuntimeError, match="only available after fit"):
        trainer.test(datamodule, ckpt_path="best")


def test_fit_test_predict_falls_back_to_in_memory_weights_without_best_checkpoint(
    write_processed_csv,
    build_data_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = write_processed_csv()
    data_config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(data_config)
    trainer = FusedModelTrainer(_build_base_config(data_config))

    fit_artifacts = FitArtifacts(
        # Use a real model instance so the test fixture stays compatible with
        # the typed `FitArtifacts` contract and does not rely on broad object
        # placeholders that static analysis rejects.
        model=FusedModel(_build_base_config(data_config)),
        runtime_config=_build_base_config(data_config),
        trainer=cast(Trainer, object()),
        has_validation_data=False,
        has_test_data=True,
        best_checkpoint_path="",
    )
    observed_ckpt_paths: list[object] = []

    def fake_fit(
        observed_datamodule: AZT1DDataModule,
        *,
        ckpt_path: str | Path | None = None,
    ) -> FitArtifacts:
        del observed_datamodule, ckpt_path
        return fit_artifacts

    def fake_test(
        observed_datamodule: AZT1DDataModule,
        *,
        ckpt_path: object = "best",
    ) -> list[dict[str, float]]:
        del observed_datamodule
        observed_ckpt_paths.append(ckpt_path)
        return [{"test_loss": 0.5}]

    def fake_predict_test(
        observed_datamodule: AZT1DDataModule,
        *,
        ckpt_path: object = "best",
    ) -> list[object]:
        del observed_datamodule
        observed_ckpt_paths.append(ckpt_path)
        return []

    monkeypatch.setattr(trainer, "fit", fake_fit)
    monkeypatch.setattr(trainer, "test", fake_test)
    monkeypatch.setattr(trainer, "predict_test", fake_predict_test)

    artifacts = trainer.fit_test_predict(datamodule)

    assert observed_ckpt_paths == [None, None]
    assert artifacts.test_metrics == [{"test_loss": 0.5}]


def test_build_trainer_passes_runtime_tuning_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_kwargs: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            observed_kwargs.update(kwargs)

    monkeypatch.setattr(train_module, "Trainer", FakeTrainer)

    trainer = FusedModelTrainer(
        _build_base_config(
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


def test_fit_falls_back_to_eager_model_when_compile_raises(
    write_processed_csv,
    build_data_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = write_processed_csv()
    data_config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(data_config)
    trainer = FusedModelTrainer(
        _build_base_config(data_config),
        trainer_config=TrainConfig(compile_model=True),
    )

    class FakeTrainer:
        num_training_batches = 1

        def fit(
            self,
            *,
            model: object,
            datamodule: object,
            ckpt_path: str | None = None,
        ) -> None:
            del datamodule, ckpt_path
            observed_models.append(model)

    observed_models: list[object] = []

    def fake_compile_model(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("compile exploded")

    monkeypatch.setattr(train_module, "maybe_compile_model", fake_compile_model)
    monkeypatch.setattr(
        trainer,
        "build_trainer",
        lambda has_validation_data: cast(Trainer, FakeTrainer()),
    )

    artifacts = trainer.fit(datamodule)

    assert observed_models == [trainer.model]
    assert "compile_model" in trainer.runtime_tuning_report.skipped
    assert artifacts.train_batches_processed == 1
