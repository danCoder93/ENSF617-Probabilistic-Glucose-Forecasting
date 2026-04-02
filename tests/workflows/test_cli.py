from __future__ import annotations

"""
AI-assisted maintenance note:
These tests protect the command-line layer in `workflows.cli`.

Purpose:
- verify CLI parsing preserves intended value types
- verify device-profile resolution and diagnostics are threaded into the final
  runtime configuration
- verify diagnostics-only mode exits before the training workflow is invoked
"""

from dataclasses import replace
from pathlib import Path

from config import Config, ObservabilityConfig, TrainConfig
from environment import DeviceProfileResolution, RuntimeDiagnostic
from tests.support import build_minimal_data_config, build_runtime_environment
from workflows.cli import _build_cli_configuration, build_argument_parser, main as cli_main


def test_build_cli_configuration_parses_values_and_profile_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # This test treats CLI parsing plus profile resolution as one translation
    # boundary from raw argv strings to the structured workflow configuration.
    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "--dataset-url",
            "none",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--devices",
            "0,1",
            "--precision",
            "16-mixed",
            "--limit-train-batches",
            "0.25",
            "--allow-tf32",
            "--torchview-depth",
            "6",
            "--checkpoint-dir",
            str(tmp_path / "ckpts"),
        ]
    )

    monkeypatch.setattr(
        "workflows.cli.detect_runtime_environment",
        lambda: build_runtime_environment(),
    )

    def fake_resolve_profile(
        *,
        requested_profile: str,
        environment,
        train_config: TrainConfig,
        data_config,
        observability_config: ObservabilityConfig,
        explicit_overrides: set[str] | None = None,
    ) -> DeviceProfileResolution:
        del environment, explicit_overrides
        return DeviceProfileResolution(
            requested_profile=requested_profile,
            resolved_profile="local-cpu",
            train_config=replace(train_config, accelerator="cpu", devices=1),
            data_config=replace(data_config, num_workers=1),
            observability_config=replace(observability_config, enable_device_stats=False),
            applied_defaults={"accelerator": "cpu", "devices": 1},
        )

    monkeypatch.setattr("workflows.cli.resolve_device_profile", fake_resolve_profile)
    monkeypatch.setattr(
        "workflows.cli.collect_runtime_diagnostics",
        lambda **kwargs: (
            RuntimeDiagnostic(severity="info", code="resolved", message="resolved"),
        ),
    )

    resolved = _build_cli_configuration(
        args,
        explicit_overrides={"devices", "precision", "allow_tf32"},
    )

    assert resolved.output_dir == tmp_path / "artifacts"
    assert resolved.config.data.dataset_url is None
    assert resolved.config.data.num_workers == 1
    assert resolved.train_config.accelerator == "cpu"
    assert resolved.train_config.devices == 1
    assert resolved.train_config.precision == "16-mixed"
    assert resolved.train_config.limit_train_batches == 0.25
    assert resolved.train_config.allow_tf32 is True
    assert resolved.snapshot_config.dirpath == tmp_path / "ckpts"
    assert resolved.observability_config.enable_device_stats is False
    assert resolved.observability_config.torchview_depth == 6
    assert resolved.preflight_diagnostics[0].code == "resolved"


def test_cli_main_in_diagnostics_only_mode_skips_training(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    # Diagnostics-only mode is intentionally a non-training code path. This
    # assertion keeps the CLI honest about stopping after environment checks and
    # human-readable reporting.
    monkeypatch.setattr(
        "workflows.cli.detect_runtime_environment",
        lambda: build_runtime_environment(),
    )

    def fake_resolve_profile(
        *,
        requested_profile: str,
        environment,
        train_config: TrainConfig,
        data_config,
        observability_config: ObservabilityConfig,
        explicit_overrides: set[str] | None = None,
    ) -> DeviceProfileResolution:
        del environment, explicit_overrides
        return DeviceProfileResolution(
            requested_profile=requested_profile,
            resolved_profile="local-cpu",
            train_config=replace(train_config, accelerator="cpu", devices=1),
            data_config=data_config,
            observability_config=observability_config,
            applied_defaults={"accelerator": "cpu"},
        )

    monkeypatch.setattr("workflows.cli.resolve_device_profile", fake_resolve_profile)
    monkeypatch.setattr(
        "workflows.cli.collect_runtime_diagnostics",
        lambda **kwargs: (
            RuntimeDiagnostic(severity="info", code="ready", message="ready"),
        ),
    )
    monkeypatch.setattr(
        "workflows.cli.run_training_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    result = cli_main(
        [
            "--run-diagnostics-only",
            "--dataset-url",
            "none",
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    captured = capsys.readouterr()
    assert result is None
    assert "Diagnostics-only mode enabled" in captured.out
