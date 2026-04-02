from __future__ import annotations

"""
AI-assisted implementation note:
This test file validates the top-level workflow in `main.py` using a narrow
fake trainer surface. The fake trainer intentionally mirrors the public method
signatures of `FusedModelTrainer` closely so the tests stay compatible with the
real orchestration contract and do not trigger static-analysis override errors.
"""

from pathlib import Path
from typing import Mapping, cast

import pandas as pd
import pytest
import torch

from main import build_default_config, run_training_workflow
from models.fused_model import FusedModel
from train import CheckpointSelection, FitArtifacts, FusedModelTrainer
from config import Config
from environment import RuntimeDiagnostic, RuntimeEnvironment


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

    def fit(
        self,
        datamodule: object,
        *,
        ckpt_path: str | Path | None = None,
    ) -> FitArtifacts:
        del datamodule, ckpt_path
        # Use a real `FusedModel` instance here so the fake trainer returns a
        # fully type-correct `FitArtifacts` object without relying on broad
        # `object()` placeholders that upset static analysis.
        return FitArtifacts(
            model=FusedModel(self.config),
            runtime_config=self.config,
            trainer=cast(Trainer, object()),
            has_validation_data=False,
            has_test_data=True,
            best_checkpoint_path="",
        )

    def test(
        self,
        datamodule: object,
        *,
        ckpt_path: CheckpointSelection = "best",
    ) -> list[Mapping[str, float]]:
        del datamodule
        self.test_ckpt_path = ckpt_path
        return [{"test_loss": 0.123}]

    def predict_test(
        self,
        datamodule: object,
        *,
        ckpt_path: CheckpointSelection = "best",
    ) -> list[torch.Tensor]:
        del datamodule
        self.predict_ckpt_path = ckpt_path
        # Keep this fake output aligned with the test config:
        # batch=2, prediction_length=2, quantiles=3.
        return [torch.ones(2, 2, 3)]


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
    runtime_environment = RuntimeEnvironment(
        platform="test-platform",
        system="Linux",
        release="test-release",
        machine="x86_64",
        python_version="3.12.0",
        is_colab=False,
        is_slurm=False,
        torch_available=True,
        pytorch_lightning_available=True,
        tensorboard_available=True,
        torchview_available=True,
        torch_version="2.0.0",
        accelerator_api_available=False,
        accelerator_available=False,
        accelerator_type=None,
        accelerator_device_count=0,
        cuda_available=False,
        cuda_device_count=0,
        cuda_device_name=None,
        cuda_visible_devices=None,
        mps_built=False,
        mps_available=False,
        slurm_job_id=None,
        slurm_cpus_per_task=None,
        slurm_gpus=None,
        slurm_detected_by_lightning=False,
    )
    preflight_diagnostics = (
        RuntimeDiagnostic(
            severity="info",
            code="auto_profile_resolved",
            message="`auto` resolved to `local-cpu` for this environment.",
        ),
    )

    artifacts = run_training_workflow(
        config,
        output_dir=tmp_path / "artifacts",
        trainer_class=FakeTrainer,
        requested_device_profile="auto",
        resolved_device_profile="local-cpu",
        applied_profile_defaults={"accelerator": "cpu", "devices": 1},
        runtime_environment=runtime_environment,
        preflight_diagnostics=preflight_diagnostics,
    )

    fake_trainer = FakeTrainer.last_instance
    assert fake_trainer is not None
    assert fake_trainer.test_ckpt_path is None
    assert fake_trainer.predict_ckpt_path is None
    assert artifacts.test_metrics == [{"test_loss": 0.123}]
    assert artifacts.predictions_path is not None
    assert artifacts.predictions_path.exists()
    assert artifacts.prediction_table_path is not None
    assert artifacts.prediction_table_path.exists()
    assert artifacts.summary_path is not None
    assert artifacts.summary_path.exists()
    assert artifacts.requested_device_profile == "auto"
    assert artifacts.resolved_device_profile == "local-cpu"
    assert artifacts.applied_profile_defaults == {"accelerator": "cpu", "devices": 1}
    assert artifacts.summary["device_profile"]["resolved"] == "local-cpu"
    assert artifacts.summary["device_profile"]["applied_defaults"]["accelerator"] == "cpu"
    loaded_predictions = torch.load(artifacts.predictions_path)
    assert len(loaded_predictions) == 1
    assert tuple(loaded_predictions[0].shape) == (2, 2, 3)
    prediction_table = pd.read_csv(artifacts.prediction_table_path)
    assert not prediction_table.empty
    assert "median_prediction" in prediction_table.columns
    assert artifacts.test_evaluation is not None
    assert artifacts.test_evaluation.summary.count == 4
    assert artifacts.report_paths == {}
