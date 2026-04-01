from __future__ import annotations

from typing import cast

import pytest
import torch

from main import build_default_config, run_training_workflow
from train import FitArtifacts, FusedModelTrainer
from utils.config import Config


pytest.importorskip("pytorch_lightning")
from pytorch_lightning import Trainer


class FakeTrainer(FusedModelTrainer):
    last_instance: FakeTrainer | None = None

    def __init__(self, config: Config, **kwargs: object) -> None:
        self.config = config
        self.kwargs = kwargs
        self.test_ckpt_path: object = None
        self.predict_ckpt_path: object = None
        FakeTrainer.last_instance = self

    def fit(self, datamodule: object, *, ckpt_path: str | None = None) -> FitArtifacts:
        del datamodule, ckpt_path
        return FitArtifacts(
            model=cast(object, object()),
            runtime_config=self.config,
            trainer=cast(Trainer, object()),
            has_validation_data=False,
            has_test_data=True,
            best_checkpoint_path="",
        )

    def test(self, datamodule: object, *, ckpt_path: object = "best") -> list[dict[str, float]]:
        del datamodule
        self.test_ckpt_path = ckpt_path
        return [{"test_loss": 0.123}]

    def predict_test(self, datamodule: object, *, ckpt_path: object = "best") -> list[torch.Tensor]:
        del datamodule
        self.predict_ckpt_path = ckpt_path
        return [torch.ones(2, 3, 1)]


def test_run_training_workflow_falls_back_to_in_memory_eval_and_writes_artifacts(
    tmp_path,
    write_processed_csv,
) -> None:
    csv_path = write_processed_csv(steps_per_subject=12)
    config = build_default_config(
        dataset_url=None,
        processed_dir=csv_path.parent,
        processed_file_name=csv_path.name,
        raw_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        extracted_dir=tmp_path / "extracted",
        encoder_length=4,
        prediction_length=2,
        num_workers=0,
    )

    artifacts = run_training_workflow(
        config,
        output_dir=tmp_path / "artifacts",
        trainer_class=FakeTrainer,
    )

    fake_trainer = FakeTrainer.last_instance
    assert fake_trainer is not None
    assert fake_trainer.test_ckpt_path is None
    assert fake_trainer.predict_ckpt_path is None
    assert artifacts.test_metrics == [{"test_loss": 0.123}]
    assert artifacts.predictions_path is not None
    assert artifacts.predictions_path.exists()
    assert artifacts.summary_path is not None
    assert artifacts.summary_path.exists()
    loaded_predictions = torch.load(artifacts.predictions_path)
    assert len(loaded_predictions) == 1
    assert tuple(loaded_predictions[0].shape) == (2, 3, 1)
