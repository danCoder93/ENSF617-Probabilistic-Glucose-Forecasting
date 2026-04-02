from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest
import torch

pytest.importorskip("pytorch_lightning")
from pytorch_lightning import Trainer

from defaults import (
    build_default_config,
    build_default_observability_config,
    build_default_snapshot_config,
    build_default_train_config,
)
from environment import RuntimeDiagnostic
from models.fused_model import FusedModel
from train import FitArtifacts, FusedModelTrainer
from tests.support import build_runtime_environment
from workflows.training import run_environment_benchmark_workflow


class FakeTrainer(FusedModelTrainer):
    def __init__(self, config, **kwargs: object) -> None:
        self.config = config
        self.kwargs: dict[str, Any] = dict(kwargs)

    def fit(
        self,
        datamodule: object,
        *,
        ckpt_path: str | Path | None = None,
    ) -> FitArtifacts:
        del datamodule, ckpt_path
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
        ckpt_path="best",
    ) -> list[Mapping[str, float]]:
        del datamodule, ckpt_path
        return [{"test_loss": 0.123}]

    def predict_test(
        self,
        datamodule: object,
        *,
        ckpt_path="best",
    ) -> list[torch.Tensor]:
        del datamodule, ckpt_path
        return [torch.ones(2, 2, 3)]


def test_run_environment_benchmark_workflow_writes_summary(
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
    preflight_diagnostics = (
        RuntimeDiagnostic(
            severity="info",
            code="auto_profile_resolved",
            message="`auto` resolved to `local-cpu` for this environment.",
        ),
    )

    artifacts = run_environment_benchmark_workflow(
        config,
        train_config=build_default_train_config(),
        snapshot_config=build_default_snapshot_config(output_dir=tmp_path / "artifacts"),
        observability_config=build_default_observability_config(
            output_dir=tmp_path / "artifacts"
        ),
        requested_device_profile="auto",
        resolved_device_profile="local-cpu",
        applied_profile_defaults={"accelerator": "cpu", "devices": 1},
        runtime_environment=build_runtime_environment(),
        preflight_diagnostics=preflight_diagnostics,
        output_dir=tmp_path / "artifacts",
        benchmark_train_batches=2,
        trainer_class=FakeTrainer,
    )

    assert artifacts.summary["benchmark"]["requested_train_batches"] == 2
    assert artifacts.summary["benchmark"]["actual_train_batches"] == 2
    assert artifacts.summary["benchmark"]["batch_size"] == 64
    assert artifacts.summary_path is not None
    assert artifacts.summary_path.exists()


def test_run_environment_benchmark_workflow_synchronizes_cuda_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sync_calls: list[str] = []

    monkeypatch.setattr(
        "workflows.training.synchronize_runtime_device",
        lambda *, environment: sync_calls.append(environment.accelerator_type or "none"),
    )

    artifacts = run_environment_benchmark_workflow(
        build_default_config(
            dataset_url=None,
            raw_dir=tmp_path / "raw",
            cache_dir=tmp_path / "cache",
            extracted_dir=tmp_path / "extracted",
            processed_dir=tmp_path / "processed",
            num_workers=0,
        ),
        train_config=build_default_train_config(),
        snapshot_config=build_default_snapshot_config(output_dir=tmp_path / "artifacts"),
        observability_config=build_default_observability_config(
            output_dir=tmp_path / "artifacts"
        ),
        requested_device_profile="local-cuda",
        resolved_device_profile="local-cuda",
        applied_profile_defaults={"accelerator": "gpu"},
        runtime_environment=build_runtime_environment(
            cuda_available=True,
            cuda_device_count=1,
            accelerator_type="cuda",
        ),
        preflight_diagnostics=(),
        output_dir=tmp_path / "artifacts",
        benchmark_train_batches=2,
        trainer_class=FakeTrainer,
    )

    assert artifacts.summary_path is not None
    assert sync_calls == ["cuda", "cuda"]


def test_run_environment_benchmark_workflow_requires_positive_batch_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="benchmark_train_batches must be > 0"):
        run_environment_benchmark_workflow(
            build_default_config(
                dataset_url=None,
                raw_dir=tmp_path / "raw",
                cache_dir=tmp_path / "cache",
                extracted_dir=tmp_path / "extracted",
                processed_dir=tmp_path / "processed",
                num_workers=0,
            ),
            train_config=build_default_train_config(),
            snapshot_config=build_default_snapshot_config(output_dir=tmp_path / "artifacts"),
            observability_config=build_default_observability_config(
                output_dir=tmp_path / "artifacts"
            ),
            requested_device_profile="auto",
            resolved_device_profile="local-cpu",
            applied_profile_defaults={},
            runtime_environment=build_runtime_environment(),
            preflight_diagnostics=(),
            output_dir=tmp_path / "artifacts",
            benchmark_train_batches=0,
            trainer_class=FakeTrainer,
        )
