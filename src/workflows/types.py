from __future__ import annotations

# AI-assisted maintenance note:
# These dataclasses are the stable artifact contracts for the repository's
# top-level execution flows.
#
# Why keep them separate:
# - both the CLI wrapper and notebook/tests rely on these result objects
# - moving them out of the root entrypoint keeps the workflow modules smaller
# - a dedicated home makes it clear these types describe orchestration results,
#   not model or data-domain entities

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from environment import RuntimeDiagnostic, RuntimeEnvironment

if TYPE_CHECKING:
    import torch

    from evaluation import EvaluationResult
    from train import FitArtifacts


@dataclass(frozen=True)
class MainRunArtifacts:
    """
    Stable summary of the major artifacts produced by a top-level run.

    Purpose:
    provide one stable object describing the major outputs of a top-level
    workflow execution.

    Context:
    the CLI, tests, and notebooks all receive the same named result contract,
    which keeps downstream inspection much clearer than relying on positional
    tuples or peeking into mutable workflow internals.

    Evaluation note:
    `test_metrics` reflects Lightning's reduced scalar test output, while
    `test_evaluation` carries the repository's richer structured detailed
    evaluation derived from raw test predictions plus aligned targets/metadata.

    Field notes:
    - `fit` captures the trainer-side result contract from `src/train.py`
    - `summary` is the JSON-ready in-memory mirror of `run_summary.json`
    - `predictions_path` and `prediction_table_path` intentionally separate
      raw tensor export from flat tabular export because those artifacts serve
      different downstream workflows
    """
    # Training/evaluation state from the reusable workflow itself.
    fit: FitArtifacts
    test_metrics: list[Mapping[str, float]] | None
    test_evaluation: EvaluationResult | None
    test_predictions: list[torch.Tensor] | None

    # Persisted or JSON-ready summary artifacts describing the run.
    summary: dict[str, Any]
    summary_path: Path | None
    predictions_path: Path | None
    prediction_table_path: Path | None
    report_paths: dict[str, Path]

    # Observability/runtime artifact locations surfaced back to callers.
    telemetry_path: Path | None
    logger_dir: Path | None
    text_log_path: Path | None

    # Environment/profile resolution metadata that explains how the run was tuned.
    requested_device_profile: str
    resolved_device_profile: str
    applied_profile_defaults: dict[str, Any]
    runtime_environment: RuntimeEnvironment
    preflight_diagnostics: tuple[RuntimeDiagnostic, ...]


@dataclass(frozen=True)
class EnvironmentBenchmarkArtifacts:
    """
    Summary of one short environment-focused benchmark run.

    Context:
    benchmark mode intentionally returns a much smaller artifact contract than
    the full training workflow because the primary output is the comparison
    summary rather than checkpoints, prediction exports, or detailed reports.
    """
    # JSON-ready benchmark summary plus the optional persisted copy on disk.
    summary: dict[str, Any]
    summary_path: Path | None
