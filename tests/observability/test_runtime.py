from __future__ import annotations

# These tests protect the observability runtime-setup helpers that assemble the
# logger, profiler, and text logging surfaces for a run.

from pathlib import Path

import pytest

pytest.importorskip("pytorch_lightning")

from config import ObservabilityConfig
from observability.runtime import build_lightning_logger, setup_observability


def test_build_lightning_logger_falls_back_to_csv_when_tensorboard_is_disabled(
    tmp_path: Path,
) -> None:
    # When TensorBoard is disabled, the runtime setup should still leave the run
    # with a usable scalar logger via the CSV fallback path.
    logger, logger_dir = build_lightning_logger(
        ObservabilityConfig(
            enable_tensorboard=False,
            enable_csv_fallback_logger=True,
            log_dir=tmp_path / "logs",
        )
    )

    assert logger is not None
    assert logger_dir == tmp_path / "logs"


def test_setup_observability_creates_text_logger_and_profiler(tmp_path: Path) -> None:
    # This is the broadest runtime-setup smoke test in the file: logger,
    # profiler, and text-log surfaces should all come back wired and usable.
    config = ObservabilityConfig(
        enable_tensorboard=False,
        enable_csv_fallback_logger=True,
        enable_profiler=True,
        profiler_type="simple",
        log_dir=tmp_path / "logs",
        text_log_path=tmp_path / "run.log",
        telemetry_path=tmp_path / "telemetry.csv",
        profiler_path=tmp_path / "profiler",
    )

    artifacts = setup_observability(config)
    assert artifacts.logger is not None
    assert artifacts.logger_dir == tmp_path / "logs"
    assert artifacts.text_logger is not None
    assert artifacts.profiler is not None

    artifacts.text_logger.info("runtime ready")
    assert artifacts.text_log_path is not None
    assert "runtime ready" in artifacts.text_log_path.read_text(encoding="utf-8")
