from __future__ import annotations

"""
AI-assisted maintenance note:
These tests protect the trainer wrapper's execution-time orchestration policy.

Purpose:
- verify checkpoint aliases are validated against available fit state
- verify fit/test/predict composition falls back sensibly when no best checkpoint exists
- verify optional compilation failures degrade to eager execution rather than aborting the run

Context:
the focus here is the wrapper's control flow around Lightning, not the details
of the model's forward or loss implementation.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

pytest.importorskip("pytorch_lightning")
import train as train_module
from pytorch_lightning import Trainer

from data.datamodule import AZT1DDataModule
from models.fused_model import FusedModel
from train import FitArtifacts, FusedModelTrainer
from config import TrainConfig
from tests.support import build_base_config


def test_checkpoint_alias_requires_fit_before_evaluation(
    write_processed_csv,
    build_data_config,
) -> None:
    # Alias-based evaluation only makes sense once the wrapper has a live
    # Trainer session and any checkpoint callback state produced by `fit()`.
    csv_path = write_processed_csv(steps_per_subject=80)
    data_config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(data_config)
    trainer = FusedModelTrainer(build_base_config(data_config))

    with pytest.raises(RuntimeError, match="only available after fit"):
        trainer.test(datamodule, ckpt_path="best")


def test_resolve_checkpoint_reference_normalizes_aliases_and_explicit_paths(
    tmp_path: Path,
    write_processed_csv,
    build_data_config,
) -> None:
    # This helper is the normalization boundary for every evaluation call, so
    # these assertions keep alias handling and explicit paths aligned.
    csv_path = write_processed_csv()
    data_config = build_data_config(csv_path)
    trainer = FusedModelTrainer(build_base_config(data_config))
    trainer.trainer = cast(Trainer, object())

    with pytest.raises(RuntimeError, match="No best checkpoint snapshot"):
        trainer._resolve_checkpoint_reference("best")

    trainer.best_checkpoint_path = str(tmp_path / "best.ckpt")

    assert trainer._resolve_checkpoint_reference(None) is None
    assert trainer._resolve_checkpoint_reference("best") == "best"
    assert trainer._resolve_checkpoint_reference("last") == "last"
    assert trainer._resolve_checkpoint_reference(tmp_path / "manual.ckpt") == str(
        tmp_path / "manual.ckpt"
    )


def test_fit_test_predict_falls_back_to_in_memory_weights_without_best_checkpoint(
    write_processed_csv,
    build_data_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The most common workflow is "fit, then evaluate." If that run never
    # produced a validation-ranked best checkpoint, the wrapper should keep
    # going with the current in-memory weights instead of failing.
    csv_path = write_processed_csv()
    data_config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(data_config)
    trainer = FusedModelTrainer(build_base_config(data_config))

    fit_artifacts = FitArtifacts(
        model=cast(FusedModel, SimpleNamespace(quantiles=(0.1, 0.5, 0.9))),
        runtime_config=build_base_config(data_config),
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


def test_fit_falls_back_to_eager_model_when_compile_raises(
    write_processed_csv,
    build_data_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Compilation is an optimization layer, not part of model correctness. A
    # compile failure should therefore be observable but non-fatal when the
    # eager model is otherwise valid.
    csv_path = write_processed_csv()
    data_config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(data_config)
    trainer = FusedModelTrainer(
        build_base_config(data_config),
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
