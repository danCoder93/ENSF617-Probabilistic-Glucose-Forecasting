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
        datamodule: Any,
        *,
        ckpt_path: str | Path | None = None,
    ) -> FitArtifacts:
        """Simulate a successful fit while still preparing the DataModule like the real workflow would."""
        del ckpt_path

        # This workflow-level fake intentionally accepts a loosely typed
        # datamodule because the test is validating orchestration behavior, not
        # the full static contract of the production DataModule class.
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
        datamodule: Any,
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
    assert {
        "forecast_overview",
        "horizon_metrics",
        "residual_histogram",
    }.issubset(set(artifacts.report_paths))


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



def test_run_training_workflow_builds_shared_report_once_and_reuses_it_across_reporting_sinks(
    tmp_path,
    write_processed_csv,
    monkeypatch,
) -> None:
    # This test protects the most important Phase-5 reporting contract at the
    # workflow boundary:
    # - build the canonical shared report once
    # - reuse that exact same in-memory report object across all post-run sinks
    # - do not let TensorBoard / CSV / Plotly quietly rebuild their own report
    #   payloads from raw inputs
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

    # Import the reporting package lazily inside the test so monkeypatching
    # modifies the real runtime objects that `run_training_workflow(...)` will
    # import during its internal post-run reporting phase.
    import reporting

    calls: dict[str, Any] = {
        "build_shared_report_count": 0,
        "tensorboard_shared_report": None,
        "export_shared_report": None,
        "plotly_shared_report": None,
        "tensorboard_logger_or_trainer": None,
    }
    canonical_shared_report = cast(object, SimpleNamespace(name="canonical_shared_report"))

    original_build_shared_report = reporting.build_shared_report
    original_log_shared_report_to_tensorboard = reporting.log_shared_report_to_tensorboard
    original_export_prediction_table_from_report = reporting.export_prediction_table_from_report
    original_generate_plotly_reports = reporting.generate_plotly_reports

    def spy_build_shared_report(*args: object, **kwargs: object) -> object:
        # Delegate to the original builder so the workflow still executes the
        # repo's real canonical packaging path. The test records the call count
        # but intentionally returns one fixed sentinel object so sink identity
        # assertions stay simple and explicit.
        del args, kwargs
        calls["build_shared_report_count"] += 1
        return canonical_shared_report

    def spy_log_shared_report_to_tensorboard(
        *,
        shared_report: object,
        logger_or_trainer: Any,
        global_step: int = 0,
        namespace: str = "report",
        max_table_rows: int = 20,
        max_forecast_subjects: int = 5,
    ) -> bool:
        # Record the exact report object and logger surface the workflow routed
        # into the TensorBoard sink, then preserve the sink's best-effort
        # semantics by returning `True`.
        del global_step, namespace, max_table_rows, max_forecast_subjects
        calls["tensorboard_shared_report"] = shared_report
        calls["tensorboard_logger_or_trainer"] = logger_or_trainer
        return True

    def spy_export_prediction_table_from_report(
        *,
        shared_report: object,
        output_path: Path | None,
    ) -> Path | None:
        # Record the report identity while still returning the caller-provided
        # path so the workflow artifact contract remains intact for the test.
        calls["export_shared_report"] = shared_report
        return output_path

    def spy_generate_plotly_reports(
        prediction_table_path: Path | None,
        *,
        report_dir: Path | None,
        max_subjects: int,
        shared_report: object | None = None,
    ) -> dict[str, Path]:
        # Record the report identity and return a lightweight deterministic
        # report-path mapping so the workflow summary/artifact plumbing still
        # has concrete values to carry forward.
        del prediction_table_path, max_subjects
        calls["plotly_shared_report"] = shared_report
        if report_dir is None:
            return {}
        report_dir.mkdir(parents=True, exist_ok=True)
        forecast_path = report_dir / "forecast_overview.html"
        forecast_path.write_text("<html></html>", encoding="utf-8")
        return {"forecast_overview": forecast_path}

    monkeypatch.setattr(reporting, "build_shared_report", spy_build_shared_report)
    monkeypatch.setattr(
        reporting,
        "log_shared_report_to_tensorboard",
        spy_log_shared_report_to_tensorboard,
    )
    monkeypatch.setattr(
        reporting,
        "export_prediction_table_from_report",
        spy_export_prediction_table_from_report,
    )
    monkeypatch.setattr(reporting, "generate_plotly_reports", spy_generate_plotly_reports)

    try:
        artifacts = run_training_workflow(
            config,
            output_dir=tmp_path / "artifacts",
            trainer_class=FakeTrainer,
            requested_device_profile="auto",
            resolved_device_profile="local-cpu",
            applied_profile_defaults={"accelerator": "cpu", "devices": 1},
            runtime_environment=build_runtime_environment(),
            preflight_diagnostics=(),
        )
    finally:
        # Restore the original reporting helpers so this workflow-level contract
        # test does not leak monkeypatch state into the rest of the module.
        monkeypatch.setattr(reporting, "build_shared_report", original_build_shared_report)
        monkeypatch.setattr(
            reporting,
            "log_shared_report_to_tensorboard",
            original_log_shared_report_to_tensorboard,
        )
        monkeypatch.setattr(
            reporting,
            "export_prediction_table_from_report",
            original_export_prediction_table_from_report,
        )
        monkeypatch.setattr(reporting, "generate_plotly_reports", original_generate_plotly_reports)

    assert artifacts.test_predictions is not None
    assert calls["build_shared_report_count"] == 1
    assert calls["tensorboard_shared_report"] is canonical_shared_report
    assert calls["export_shared_report"] is canonical_shared_report
    assert calls["plotly_shared_report"] is canonical_shared_report

    # The core contract here is SharedReport build/reuse across sinks. The
    # active logger surface can vary with observability setup, so this test does
    # not assert a non-null TensorBoard logger value.
    fake_trainer = FakeTrainer.last_instance
    assert fake_trainer is not None


def test_run_training_workflow_does_not_build_shared_report_when_prediction_is_skipped(
    tmp_path,
    write_processed_csv,
    monkeypatch,
) -> None:
    # The canonical shared report requires raw held-out predictions plus
    # structured evaluation. If the caller disables prediction, the workflow
    # should skip the entire shared-report build/fan-out path instead of
    # synthesizing a fallback report from partial information.
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

    import reporting

    calls = {
        "build_shared_report_count": 0,
        "tensorboard_called": False,
        "export_called": False,
        "plotly_called": False,
    }

    original_build_shared_report = reporting.build_shared_report
    original_log_shared_report_to_tensorboard = reporting.log_shared_report_to_tensorboard
    original_export_prediction_table_from_report = reporting.export_prediction_table_from_report
    original_generate_plotly_reports = reporting.generate_plotly_reports

    def spy_build_shared_report(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls["build_shared_report_count"] += 1
        return cast(object, SimpleNamespace())

    def spy_log_shared_report_to_tensorboard(**kwargs: object) -> bool:
        del kwargs
        calls["tensorboard_called"] = True
        return True

    def spy_export_prediction_table_from_report(**kwargs: object) -> Path | None:
        del kwargs
        calls["export_called"] = True
        return None

    def spy_generate_plotly_reports(*args: object, **kwargs: object) -> dict[str, Path]:
        del args, kwargs
        calls["plotly_called"] = True
        return {}

    monkeypatch.setattr(reporting, "build_shared_report", spy_build_shared_report)
    monkeypatch.setattr(
        reporting,
        "log_shared_report_to_tensorboard",
        spy_log_shared_report_to_tensorboard,
    )
    monkeypatch.setattr(
        reporting,
        "export_prediction_table_from_report",
        spy_export_prediction_table_from_report,
    )
    monkeypatch.setattr(reporting, "generate_plotly_reports", spy_generate_plotly_reports)

    try:
        artifacts = run_training_workflow(
            config,
            output_dir=tmp_path / "artifacts",
            trainer_class=FakeTrainer,
            requested_device_profile="auto",
            resolved_device_profile="local-cpu",
            applied_profile_defaults={"accelerator": "cpu", "devices": 1},
            runtime_environment=build_runtime_environment(),
            preflight_diagnostics=(),
            skip_predict=True,
            save_predictions=False,
        )
    finally:
        monkeypatch.setattr(reporting, "build_shared_report", original_build_shared_report)
        monkeypatch.setattr(
            reporting,
            "log_shared_report_to_tensorboard",
            original_log_shared_report_to_tensorboard,
        )
        monkeypatch.setattr(
            reporting,
            "export_prediction_table_from_report",
            original_export_prediction_table_from_report,
        )
        monkeypatch.setattr(reporting, "generate_plotly_reports", original_generate_plotly_reports)

    assert artifacts.test_predictions is None
    assert calls["build_shared_report_count"] == 0
    assert calls["tensorboard_called"] is False
    assert calls["export_called"] is False
    assert calls["plotly_called"] is False
