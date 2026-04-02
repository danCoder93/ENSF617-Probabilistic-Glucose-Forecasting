from __future__ import annotations

# These tests protect the small logging helpers shared across the observability
# package.

from pathlib import Path

from observability.logging_utils import (
    _log_text_to_loggers,
    log_hyperparameters,
    log_metrics_to_loggers,
)
from tests.observability.support import RecordingLogger, RecordingTrainer


def test_logging_helpers_publish_metrics_text_and_hparams() -> None:
    # The helpers flatten several different payload types into the logger
    # surface, so one compact test keeps those surfaces aligned.
    logger = RecordingLogger()
    trainer = RecordingTrainer(logger)

    log_metrics_to_loggers(trainer, {"debug/example": 1.25}, step=7)
    _log_text_to_loggers(trainer, "batch_audit/train", "sample batch")
    log_hyperparameters(
        trainer,
        {
            "config": {
                "epochs": 3,
                "output_dir": Path("artifacts/run"),
            },
            "flags": {"debug": True, "seed": None},
        },
    )

    assert logger.metric_events == [({"debug/example": 1.25}, 7)]
    assert logger.experiment.text_events == [
        ("batch_audit/train", "sample batch", trainer.global_step)
    ]
    assert logger.hparam_events == [
        {
            "config/epochs": 3,
            "config/output_dir": "artifacts/run",
            "flags/debug": True,
            "flags/seed": "None",
        }
    ]
