from __future__ import annotations

from pathlib import Path

import pytest

from config import ObservabilityConfig, SnapshotConfig, TrainConfig


def test_observability_config_normalizes_paths_and_validates_mode(
    tmp_path: Path,
) -> None:
    config = ObservabilityConfig(
        mode="debug",
        log_dir=str(tmp_path / "logs"),
        text_log_path=str(tmp_path / "run.log"),
        telemetry_path=str(tmp_path / "telemetry.csv"),
        prediction_table_path=str(tmp_path / "predictions.csv"),
        report_dir=str(tmp_path / "reports"),
    )

    assert config.log_dir == tmp_path / "logs"
    assert config.text_log_path == tmp_path / "run.log"
    assert config.telemetry_path == tmp_path / "telemetry.csv"
    assert config.prediction_table_path == tmp_path / "predictions.csv"
    assert config.report_dir == tmp_path / "reports"

    with pytest.raises(ValueError, match="mode must be one of"):
        ObservabilityConfig(mode="nope")


def test_train_config_validates_runtime_tuning_contract() -> None:
    with pytest.raises(ValueError, match="compile_mode requires compile_model"):
        TrainConfig(compile_model=False, compile_mode="default")

    with pytest.raises(ValueError, match="matmul_precision"):
        TrainConfig(matmul_precision="fast")


def test_snapshot_config_normalizes_dirpath_and_validates_mode(tmp_path: Path) -> None:
    config = SnapshotConfig(
        dirpath=str(tmp_path / "checkpoints"),
        mode="max",
        save_top_k=2,
    )

    assert config.dirpath == tmp_path / "checkpoints"
    assert config.mode == "max"
    assert config.save_top_k == 2

    with pytest.raises(ValueError, match="mode must be either"):
        SnapshotConfig(mode="median")
