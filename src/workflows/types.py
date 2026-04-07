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
#
# Responsibility boundary:
# - define the named result objects returned by the top-level workflows
# - preserve a stable, explicit contract for callers outside the workflow
#   implementation modules
# - keep the workflow return surface readable and inspectable without exposing
#   lower-level trainer internals directly
#
# What does *not* live here:
# - the implementation of training, evaluation, prediction export, or reporting
# - filesystem-writing logic
# - model configuration or runtime policy
#
# Phase 1 reporting architecture note:
# the workflow now builds a shared post-run reporting surface internally, but
# these result dataclasses intentionally remain conservative in this phase.
#
# In particular:
# - `MainRunArtifacts` continues to expose the persisted artifact paths that
#   downstream callers already expect
# - we do not yet widen this contract to carry a full in-memory shared report
#   object because that would be a bigger public-surface change than Phase 1
#   needs
# - instead, the workflow keeps using the existing outward-facing artifact
#   fields while the reporting internals become more structured behind the
#   scenes
#
# That design keeps the Phase 1 refactor enhancement-focused:
# improve the reporting path internally without forcing notebooks, tests, or
# other callers to immediately adapt to a larger returned artifact contract.

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

    Reporting note:
    the Phase 1 reporting refactor improves how post-run prediction tables and
    report artifacts are *constructed* internally, but this outward-facing
    contract still exposes the same persisted artifact surfaces as before.
    That is intentional: callers should be able to keep using this dataclass
    without needing to understand the new internal shared-reporting layer.

    Field notes:
    - `fit` captures the trainer-side result contract from `src/train.py`
    - `summary` is the JSON-ready in-memory mirror of `run_summary.json`
    - `predictions_path` and `prediction_table_path` intentionally separate
      raw tensor export from flat tabular export because those artifacts serve
      different downstream workflows
    - `report_paths` stays intentionally generic because lightweight HTML or
      other rendered report artifacts may evolve over time without requiring a
      brand-new top-level return contract for each new sink
    """
    # =========================================================================
    # Training / evaluation state
    # =========================================================================
    # These fields describe what the shared workflow computed in memory during
    # the run itself. They are useful for immediate Python-side inspection in
    # notebooks, tests, or higher-level orchestration code.
    fit: FitArtifacts
    test_metrics: list[Mapping[str, float]] | None
    test_evaluation: EvaluationResult | None
    test_predictions: list[torch.Tensor] | None

    # =========================================================================
    # Persisted or JSON-ready summary artifacts
    # =========================================================================
    # These fields describe the run outputs that are either already written to
    # disk or are directly serializable. Keeping them explicit makes it easy for
    # callers to discover the major artifacts without reverse-engineering the
    # workflow's output directory layout.
    summary: dict[str, Any]
    summary_path: Path | None
    predictions_path: Path | None
    prediction_table_path: Path | None
    report_paths: dict[str, Path]

    # =========================================================================
    # Observability/runtime artifact locations
    # =========================================================================
    # These are surfaced separately from the main report artifacts because they
    # belong to the broader observability/runtime layer rather than the model's
    # held-out prediction analysis outputs specifically.
    telemetry_path: Path | None
    logger_dir: Path | None
    text_log_path: Path | None

    # =========================================================================
    # Environment/profile resolution metadata
    # =========================================================================
    # These fields explain how the run was tuned and on which resolved runtime
    # profile it actually executed. That context is important when comparing
    # outputs across local CPU, Apple Silicon, CUDA, Slurm, or other runtime
    # environments.
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

    Design note:
    unlike `MainRunArtifacts`, this dataclass is intentionally narrow because
    benchmark mode is meant to answer quick environment-comparison questions
    rather than serve as a full experiment-reporting surface.
    """
    # JSON-ready benchmark summary plus the optional persisted copy on disk.
    summary: dict[str, Any]
    summary_path: Path | None
