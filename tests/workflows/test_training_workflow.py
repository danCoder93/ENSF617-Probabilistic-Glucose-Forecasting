from __future__ import annotations

"""
AI-assisted maintenance note:
These tests protect the top-level reusable training workflow in
`workflows.training`.

Purpose:
- verify the workflow writes the expected artifact set
- verify profile-default application is reflected in the trainer/runtime config
- verify preflight diagnostics can stop the workflow before training begins

Context:
the tests intentionally replace the real trainer wrapper with a small fake so
they can focus on workflow composition, artifact plumbing, and checkpoint
selection policy rather than on Lightning execution details.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pandas as pd
import pytest
import torch

pytest.importorskip("pytorch_lightning")

from defaults import build_default_config
from pytorch_lightning import Trainer
from models.fused_model import FusedModel
from train import CheckpointSelection, FitArtifacts, FusedModelTrainer
from environment import RuntimeDiagnostic
from tests.support import build_runtime_environment
from workflows.training import run_training_workflow


class FakeTrainer(FusedModelTrainer):
    """
    Small workflow-level trainer fake that records evaluation checkpoint choices.

    Context:
    these tests are not about the internal Trainer wrapper. They only need a
    trainer-shaped object that returns deterministic fit/test/predict payloads
    so the workflow's orchestration logic can be asserted directly.
    """

    last_instance: FakeTrainer | None = None

    def __init__(self, config, **kwargs: object) -> None:
        """Capture the runtime config and keyword arguments the workflow passed to the trainer."""
        self.config = config
        self.kwargs: dict[str, Any] = dict(kwargs)
        self.test_ckpt_path: object = None
        self.predict_ckpt_path: object = None
        FakeTrainer.last_instance = self

    def fit(
        self,
        datamodule: object,
        *,
        ckpt_path: str | Path | None = None,
    ) -> FitArtifacts:
        """Simulate a successful fit while still preparing the DataModule like the real workflow would."""
        del ckpt_path
        datamodule.prepare_data()
        datamodule.setup()
        return FitArtifacts(
            model=cast(FusedModel, SimpleNamespace(quantiles=(0.1, 0.5, 0.9))),
            runtime_config=self.config,
            trainer=cast(Trainer, object()),
            has_validation_data=False,
            has_test_data=True,
            best_checkpoint_path="",
            train_batches_processed=2,
        )

    def test(
        self,
        datamodule: object,
        *,
        ckpt_path: CheckpointSelection = "best",
    ) -> list[Mapping[str, float]]:
        """Record which checkpoint reference the workflow asked to evaluate."""
        del datamodule
        self.test_ckpt_path = ckpt_path
        return [{"test_loss": 0.123}]

    def predict_test(
        self,
        datamodule: object,
        *,
        ckpt_path: CheckpointSelection = "best",
    ) -> list[torch.Tensor]:
        """Return deterministic quantile-shaped batches while recording checkpoint selection."""
        self.predict_ckpt_path = ckpt_path
        predictions: list[torch.Tensor] = []
        for batch in datamodule.test_dataloader():
            batch_size, prediction_length = batch["target"].shape
            predictions.append(torch.ones(batch_size, prediction_length, 3))
        return predictions


def test_run_training_workflow_falls_back_to_in_memory_eval_and_writes_artifacts(
    tmp_path,
    write_processed_csv,
) -> None:
    # This is the broadest happy-path workflow scenario in the file:
    # train, evaluate, export predictions, derive analysis tables, and write a
    # final summary bundle. The fake trainer keeps the assertions focused on
    # orchestration rather than on model execution.
    csv_path = write_processed_csv(steps_per_subject=80)
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
        runtime_environment=build_runtime_environment(),
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
    assert len(loaded_predictions) >= 1
    assert loaded_predictions[0].shape[-1] == 3
    prediction_table = pd.read_csv(artifacts.prediction_table_path)
    assert not prediction_table.empty
    assert "median_prediction" in prediction_table.columns
    assert artifacts.test_evaluation is not None
    assert artifacts.test_evaluation.summary.count > 0
    assert set(artifacts.report_paths) == {
        "forecast_overview",
        "horizon_metrics",
        "residual_histogram",
    }


def test_run_training_workflow_applies_profile_defaults_when_unresolved(
    tmp_path,
    write_processed_csv,
) -> None:
    # When profile resolution is left to the workflow, it should still produce
    # one concrete runtime config before trainer construction. This test checks
    # that those defaults actually flow into the trainer-facing config.
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
        requested_device_profile="auto",
        runtime_environment=build_runtime_environment(),
        skip_test=True,
        skip_predict=True,
        save_predictions=False,
    )

    fake_trainer = FakeTrainer.last_instance
    assert fake_trainer is not None
    assert artifacts.resolved_device_profile == "local-cpu"
    trainer_config = cast(Any, fake_trainer.kwargs["trainer_config"])
    assert trainer_config.accelerator == "cpu"
    assert trainer_config.compile_model is True
    assert fake_trainer.config.data.num_workers == 1


def test_run_training_workflow_fails_fast_on_preflight_errors(tmp_path) -> None:
    # Error-severity diagnostics should be able to stop the workflow before any
    # expensive training work starts when the caller opts into fail-fast mode.
    config = build_default_config(
        dataset_url=None,
        processed_dir=tmp_path / "processed",
        raw_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        extracted_dir=tmp_path / "extracted",
        num_workers=0,
    )

    with pytest.raises(RuntimeError, match="Runtime preflight checks failed"):
        run_training_workflow(
            config,
            output_dir=tmp_path / "artifacts",
            trainer_class=FakeTrainer,
            requested_device_profile="local-cuda",
            resolved_device_profile="local-cuda",
            runtime_environment=build_runtime_environment(),
            preflight_diagnostics=(
                RuntimeDiagnostic(
                    severity="error",
                    code="cuda_unavailable",
                    message="GPU acceleration was requested, but CUDA is not available.",
                ),
            ),
            fail_on_preflight_errors=True,
        )
