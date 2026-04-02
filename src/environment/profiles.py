from __future__ import annotations

# AI-assisted maintenance note:
# This module maps detected environments onto project-specific runtime
# profiles. It owns the environment-aware defaulting logic, but does not own
# hardware detection or diagnostic explanation.
#
# Responsibility boundary:
# - choose a concrete profile from the detected environment
# - apply profile defaults to runtime-facing config objects
# - preserve explicit user overrides
#
# What does *not* live here:
# - probing torch/Lightning/backend state
# - deciding whether a resolved setup is valid
# - formatting diagnostics for users
#
# Important disclaimer:
# these profiles are repository-specific convenience policies. They are meant
# to provide safer starting points for common environments, not to claim that
# the chosen settings are universally optimal for performance or stability.

from dataclasses import replace
from typing import Any

from config.data import DataConfig
from config.observability import ObservabilityConfig
from config.runtime import TrainConfig
from environment.types import DeviceProfileResolution, RuntimeEnvironment


# ============================================================================
# Profile Selection
# ============================================================================
# The `auto` profile is intentionally conservative in its priority order:
# 1. notebook / scheduler environments with strong external context
# 2. Apple Silicon MPS
# 3. local CUDA
# 4. plain local CPU
# This keeps explicit special environments from being mistaken for generic
# local runs.
def infer_device_profile(
    requested_profile: str,
    environment: RuntimeEnvironment,
) -> str:
    """
    Resolve `auto` into one concrete device profile for the current environment.
    """

    if requested_profile != "auto":
        return requested_profile

    if environment.is_colab:
        return "colab-cuda" if environment.cuda_available else "colab-cpu"
    if environment.is_slurm:
        return "slurm-cuda" if environment.cuda_available else "slurm-cpu"
    if (
        environment.accelerator_type == "mps"
        or (environment.mps_available and environment.system == "Darwin")
    ):
        return "apple-silicon"
    if environment.accelerator_type == "cuda" or environment.cuda_available:
        return "local-cuda"
    return "local-cpu"


def _slurm_worker_default(environment: RuntimeEnvironment) -> int:
    if environment.slurm_cpus_per_task is None:
        return 4
    return max(0, min(environment.slurm_cpus_per_task - 1, 8))


def resolve_device_profile(
    *,
    requested_profile: str,
    environment: RuntimeEnvironment,
    train_config: TrainConfig,
    data_config: DataConfig,
    observability_config: ObservabilityConfig,
    explicit_overrides: set[str] | None = None,
) -> DeviceProfileResolution:
    """
    Apply one high-level device profile to the runtime-facing config objects.

    Context:
    this function is the translation layer from "one user-facing runtime
    choice" to concrete config updates across Trainer, DataLoader, and
    observability policy.
    """

    overrides = explicit_overrides or set()
    resolved_profile = infer_device_profile(requested_profile, environment)
    train_updates: dict[str, Any] = {}
    data_updates: dict[str, Any] = {}
    observability_updates: dict[str, Any] = {}
    applied_defaults: dict[str, Any] = {}

    def maybe_update(
        destination: dict[str, Any],
        field_name: str,
        value: Any,
        *,
        cli_dest: str | None = None,
    ) -> None:
        # Explicit user inputs remain authoritative. Profiles only provide
        # defaults for settings the caller did not specify directly.
        override_name = field_name if cli_dest is None else cli_dest
        if override_name in overrides:
            return
        destination[field_name] = value
        applied_defaults[field_name] = value

    # The profile table is intentionally explicit rather than data-driven. That
    # makes it easier to answer:
    # - what does each environment profile actually do?
    # - which knobs differ between profiles?
    # - where should a future profile-specific tweak be made?
    if resolved_profile == "local-cpu":
        maybe_update(train_updates, "accelerator", "cpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", 32)
        maybe_update(data_updates, "num_workers", 0)
        maybe_update(data_updates, "pin_memory", False)
        maybe_update(data_updates, "persistent_workers", False)
    elif resolved_profile == "local-cuda":
        maybe_update(train_updates, "accelerator", "gpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", "16-mixed")
        maybe_update(data_updates, "num_workers", 4)
        maybe_update(data_updates, "pin_memory", True)
        maybe_update(data_updates, "persistent_workers", True)
    elif resolved_profile == "colab-cpu":
        maybe_update(train_updates, "accelerator", "cpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", 32)
        maybe_update(data_updates, "num_workers", 0)
        maybe_update(data_updates, "pin_memory", False)
        maybe_update(data_updates, "persistent_workers", False)
        maybe_update(observability_updates, "enable_torchview", False)
    elif resolved_profile == "colab-cuda":
        maybe_update(train_updates, "accelerator", "gpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", "16-mixed")
        maybe_update(data_updates, "num_workers", 2)
        maybe_update(data_updates, "pin_memory", True)
        maybe_update(data_updates, "persistent_workers", True)
        maybe_update(observability_updates, "enable_torchview", False)
    elif resolved_profile == "slurm-cpu":
        maybe_update(train_updates, "accelerator", "cpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", 32)
        maybe_update(train_updates, "enable_progress_bar", False)
        maybe_update(data_updates, "num_workers", _slurm_worker_default(environment))
        maybe_update(data_updates, "pin_memory", False)
        maybe_update(data_updates, "persistent_workers", True)
        maybe_update(observability_updates, "enable_rich_progress_bar", False)
    elif resolved_profile == "slurm-cuda":
        maybe_update(train_updates, "accelerator", "gpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", "16-mixed")
        maybe_update(train_updates, "enable_progress_bar", False)
        maybe_update(data_updates, "num_workers", _slurm_worker_default(environment))
        maybe_update(data_updates, "pin_memory", True)
        maybe_update(data_updates, "persistent_workers", True)
        maybe_update(observability_updates, "enable_rich_progress_bar", False)
    elif resolved_profile == "apple-silicon":
        maybe_update(train_updates, "accelerator", "mps")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", 32)
        maybe_update(data_updates, "num_workers", 0)
        maybe_update(data_updates, "pin_memory", False)
        maybe_update(data_updates, "persistent_workers", False)
        maybe_update(observability_updates, "enable_device_stats", False)

    resolved_train_config = (
        replace(train_config, **train_updates) if train_updates else train_config
    )
    resolved_data_config = (
        replace(data_config, **data_updates) if data_updates else data_config
    )
    resolved_observability_config = (
        replace(observability_config, **observability_updates)
        if observability_updates
        else observability_config
    )

    return DeviceProfileResolution(
        requested_profile=requested_profile,
        resolved_profile=resolved_profile,
        train_config=resolved_train_config,
        data_config=resolved_data_config,
        observability_config=resolved_observability_config,
        applied_defaults=applied_defaults,
    )
