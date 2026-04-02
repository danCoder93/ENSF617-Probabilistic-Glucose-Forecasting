from __future__ import annotations

# These tests protect the root default-builder helpers that keep the script and
# notebook entrypoints aligned.
#
# The goal is not to prove those defaults are "best", but to ensure the helper
# functions return one internally consistent baseline configuration.

from pathlib import Path

from defaults import (
    build_default_config,
    build_default_observability_config,
    build_default_snapshot_config,
    build_default_train_config,
)


def test_build_default_config_keeps_data_and_model_lengths_in_sync(tmp_path: Path) -> None:
    # The default builder is one coherence boundary for encoder/horizon lengths,
    # TCN horizon settings, and TFT example length.
    config = build_default_config(
        dataset_url=None,
        processed_dir=tmp_path / "processed",
        processed_file_name="dataset.csv",
        encoder_length=24,
        prediction_length=6,
        tcn_channels=(8, 16),
        quantiles=(0.1, 0.5, 0.9),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    assert config.data.processed_file_path == tmp_path / "processed" / "dataset.csv"
    assert config.tft.encoder_length == 24
    assert config.tft.example_length == 30
    assert config.tcn.prediction_length == 6
    assert config.tcn.num_channels == (8, 16)
    assert config.tft.quantiles == (0.1, 0.5, 0.9)


def test_build_default_train_snapshot_and_observability_configs_share_output_layout(
    tmp_path: Path,
) -> None:
    # Output-layout consistency matters because the default builders are meant
    # to create one tidy run directory structure across logs, reports, and
    # checkpoints.
    train_config = build_default_train_config(default_root_dir=tmp_path)
    snapshot_config = build_default_snapshot_config(output_dir=tmp_path)
    observability_config = build_default_observability_config(
        output_dir=tmp_path,
        enable_activation_stats=False,
    )

    assert train_config.default_root_dir == tmp_path
    assert snapshot_config.dirpath == tmp_path / "checkpoints"
    assert observability_config.log_dir == tmp_path / "logs"
    assert observability_config.text_log_path == tmp_path / "run.log"
    assert observability_config.telemetry_path == tmp_path / "telemetry.csv"
    assert observability_config.prediction_table_path == tmp_path / "test_predictions.csv"
    assert observability_config.report_dir == tmp_path / "reports"


def test_build_default_observability_config_enables_activation_stats_for_debug_modes() -> None:
    # Debug and trace modes intentionally trade more runtime cost for deeper
    # visibility, and activation hooks are one of the main policy differences.
    baseline = build_default_observability_config(output_dir=None, mode="baseline")
    debug = build_default_observability_config(output_dir=None, mode="debug")
    trace = build_default_observability_config(output_dir=None, mode="trace")

    assert baseline.enable_activation_stats is False
    assert debug.enable_activation_stats is True
    assert trace.enable_activation_stats is True
