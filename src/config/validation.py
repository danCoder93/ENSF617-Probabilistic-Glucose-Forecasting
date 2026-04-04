# Cross-configuration validation utilities for the training workflow.
#
# This file is intentionally limited to rules that require comparing:
# - multiple config objects together, or
# - resolved config against runtime/workflow context.
#
# Local invariants that can be validated within a single config object should
# stay in that config's own `__post_init__` method instead of being duplicated
# here.
#
# Practical rule of thumb used for this file:
# - Single-config truth  -> belongs in that config dataclass
# - Cross-config truth   -> belongs in this file
#
# Examples of what belongs here:
# - observability rich progress bar requires the Trainer progress bar
# - early stopping requires validation data to actually exist
# - ranked checkpoint saving requires validation data
# - resolved runtime profile should stay consistent with the selected
#   accelerator after all defaults/overrides are applied
#
# AI-assisted maintenance note (April 3, 2026):
# This file structure and explanatory comments were refined with AI assistance
# under user direction. The goal is to make cross-config runtime behavior
# easier to audit and debug, not to silently redefine training policy.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config.data import DataConfig
from config.observability import ObservabilityConfig
from config.runtime import SnapshotConfig, TrainConfig


# Represents one concrete cross-configuration problem.
#
# Why use a structured object instead of raw strings?
# - keeps validation rules easy to read
# - lets us collect multiple issues before raising
# - makes future extension easier if we later want severity/code/suggestion
#
# `field_path` should point to the most relevant user-facing config path.
@dataclass(frozen=True)
class ValidationIssue:
    field_path: str
    message: str

    def format(self) -> str:
        return f"{self.field_path}: {self.message}"


# Raised when one or more cross-config validation rules fail.
#
# We intentionally aggregate all discovered issues into one exception so the
# user can fix multiple configuration problems in one pass instead of rerunning
# repeatedly and discovering them one-by-one.
class ConfigurationValidationError(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = tuple(issues)

        lines = ["Invalid runtime configuration:"]
        lines.extend(f"- {issue.format()}" for issue in self.issues)

        super().__init__("\n".join(lines))


def collect_runtime_configuration_issues(
    *,
    train_config: TrainConfig | None,
    data_config: DataConfig | None,
    observability_config: ObservabilityConfig | None,
    snapshot_config: SnapshotConfig | None = None,
    resolved_profile: str | None = None,
    has_validation_data: bool | None = None,
) -> list[ValidationIssue]:
    """
    Collect cross-config compatibility issues without raising immediately.

    This function should only contain rules that need information from more than
    one config object or from runtime/workflow context.
    """
    issues: list[ValidationIssue] = []

    # NOTE:
    # `data_config` is currently not heavily used in the first set of rules, but
    # it is intentionally accepted here because this module is meant to be the
    # single home for cross-config validation across all major config groups.
    #
    # Keeping it in the signature now avoids future churn at call sites when a
    # true train/data or observability/data compatibility rule is added later.
    _ = data_config

    # ------------------------------------------------------------------
    # Rule group 1: Progress bar compatibility
    # ------------------------------------------------------------------
    #
    # Why this belongs here:
    # - `train_config.enable_progress_bar` controls the base Lightning Trainer
    #   progress bar.
    # - `observability_config.enable_rich_progress_bar` controls whether the
    #   rich progress bar callback is requested.
    #
    # Each field is independently valid, but the combination:
    #   enable_rich_progress_bar=True
    #   enable_progress_bar=False
    # is invalid.
    #
    # That makes this a textbook cross-config rule.
    if (observability_config is None or train_config is None):
        return []
    if (
        observability_config.enable_rich_progress_bar
        and not train_config.enable_progress_bar
    ):
        profile_hint = ""
        if resolved_profile is not None:
            profile_hint = (
                f" The resolved profile '{resolved_profile}' may disable the base "
                "Trainer progress bar by default, so enabling only the rich "
                "progress bar creates an invalid final configuration."
            )

        issues.append(
            ValidationIssue(
                field_path="observability.enable_rich_progress_bar",
                message=(
                    "Rich progress bar requires train.enable_progress_bar=True."
                    f"{profile_hint}"
                ),
            )
        )

    # ------------------------------------------------------------------
    # Rule group 2: Validation-dependent Trainer behavior
    # ------------------------------------------------------------------
    #
    # These checks only run when the caller already knows the effective runtime
    # truth about whether validation data exists for this run.
    #
    # Why these belong here:
    # - the truth of "has validation data" is not purely a local property of
    #   TrainConfig or SnapshotConfig
    # - it depends on the actual resolved workflow/data setup
    #
    # So these are not good candidates for `__post_init__`.
    if has_validation_data is False:
        # Lightning sanity validation steps require a validation loop.
        if train_config.num_sanity_val_steps > 0:
            issues.append(
                ValidationIssue(
                    field_path="train.num_sanity_val_steps",
                    message=(
                        "num_sanity_val_steps must be 0 when no validation data "
                        "is available for the current run."
                    ),
                )
            )

        # Early stopping is only meaningful when there is validation feedback
        # available to monitor.
        if train_config.early_stopping_patience is not None:
            issues.append(
                ValidationIssue(
                    field_path="train.early_stopping_patience",
                    message=(
                        "early_stopping_patience requires validation data for "
                        "the current run."
                    ),
                )
            )

        # Ranked checkpointing such as top-k by monitored metric requires
        # validation metrics to exist. Without validation, the workflow should
        # generally fall back to last-checkpoint-only behavior instead.
        if snapshot_config is not None and snapshot_config.enabled:
            if snapshot_config.save_top_k not in {0}:
                issues.append(
                    ValidationIssue(
                        field_path="snapshot.save_top_k",
                        message=(
                            "Ranked checkpoint saving requires validation data. "
                            "Use save_top_k=0 when running without validation."
                        ),
                    )
                )

    # ------------------------------------------------------------------
    # Rule group 3: Profile-to-runtime consistency
    # ------------------------------------------------------------------
    #
    # Profiles are a source of resolved runtime context. They are not merely a
    # property of TrainConfig itself.
    #
    # These checks are intentionally conservative:
    # - they should only catch clearly contradictory or misleading final states
    # - they should not turn the profile system into an overly strict policy
    #   layer that blocks harmless experimentation
    #
    # In other words, these checks are here to catch "this final setup does not
    # make sense for the chosen profile", not to forbid every manual override.
    if resolved_profile == "apple-silicon":
        if train_config.accelerator not in {"mps", "auto"}:
            issues.append(
                ValidationIssue(
                    field_path="profile.train.accelerator",
                    message=(
                        "apple-silicon profile should resolve to "
                        "train.accelerator='mps' or leave it as 'auto'."
                    ),
                )
            )

    if resolved_profile == "slurm-cuda":
        if train_config.accelerator not in {"gpu", "auto"}:
            issues.append(
                ValidationIssue(
                    field_path="profile.train.accelerator",
                    message=(
                        "slurm-cuda profile should resolve to "
                        "train.accelerator='gpu' or leave it as 'auto'."
                    ),
                )
            )

    if resolved_profile == "slurm-cpu":
        if train_config.accelerator not in {"cpu", "auto"}:
            issues.append(
                ValidationIssue(
                    field_path="profile.train.accelerator",
                    message=(
                        "slurm-cpu profile should resolve to "
                        "train.accelerator='cpu' or leave it as 'auto'."
                    ),
                )
            )

    return issues


def validate_runtime_configuration(
    *,
    train_config: TrainConfig | None,
    data_config: DataConfig | None,
    observability_config: ObservabilityConfig | None,
    snapshot_config: SnapshotConfig | None = None,
    resolved_profile: str | None = None,
    has_validation_data: bool | None = None,
) -> None:
    """
    Validate final cross-config compatibility and raise a combined error if any
    issue is found.
    """
    # Keep this function intentionally small.
    #
    # Design choice:
    # - `collect_runtime_configuration_issues(...)` owns the actual rules
    # - this function is only the public "validate and raise" entry point
    #
    # This keeps the code easier to test:
    # - tests can assert on the collected issue list directly
    # - production code can call the simple raising wrapper
    issues = collect_runtime_configuration_issues(
        train_config=train_config,
        data_config=data_config,
        observability_config=observability_config,
        snapshot_config=snapshot_config,
        resolved_profile=resolved_profile,
        has_validation_data=has_validation_data,
    )

    # Raise one combined exception only after all rules have been checked.
    if issues:
        raise ConfigurationValidationError(issues)