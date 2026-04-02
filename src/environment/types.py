from __future__ import annotations

# AI-assisted maintenance note:
# This module holds the shared dataclasses and constants used by the runtime
# environment package. Keeping the shared types here lets detection, profile
# resolution, and diagnostics depend on one stable contract without creating
# ownership ambiguity.
#
# Responsibility boundary:
# - define stable data contracts shared by the runtime environment package
# - keep those contracts lightweight and JSON-friendly so they can appear in
#   run summaries, notebook inspection, and test fixtures
#
# What does *not* live here:
# - backend probing logic
# - profile selection policy
# - diagnostic generation
#
# Important disclaimer:
# these dataclasses are operational metadata, not a replacement for raw
# third-party runtime objects. They intentionally capture only the subset of
# environment state that this repository needs for profile selection,
# validation, and reporting.

from dataclasses import dataclass
from typing import Any

from config.data import DataConfig
from config.observability import ObservabilityConfig
from config.runtime import TrainConfig


# ============================================================================
# Shared Runtime Contracts
# ============================================================================
# These types are intentionally centralized because they are referenced by:
# - backend detection
# - profile resolution
# - preflight diagnostics
# - run summaries / notebook inspection
# Keeping them here avoids one submodule becoming the "implied owner" of the
# package-wide data model.
DEVICE_PROFILE_CHOICES = (
    "auto",
    "local-cpu",
    "local-cuda",
    "colab-cpu",
    "colab-cuda",
    "slurm-cpu",
    "slurm-cuda",
    "apple-silicon",
)


@dataclass(frozen=True)
class RuntimeEnvironment:
    """
    Snapshot of the host/runtime environment used to select device profiles.

    Context:
    this is the normalized environment view that the rest of the package uses
    after the detection layer has already finished probing PyTorch, Lightning,
    and common environment variables.
    """

    platform: str
    system: str
    release: str
    machine: str
    python_version: str
    is_colab: bool
    is_slurm: bool
    torch_available: bool
    pytorch_lightning_available: bool
    tensorboard_available: bool
    torchview_available: bool
    torch_version: str | None
    accelerator_api_available: bool
    accelerator_available: bool
    accelerator_type: str | None
    accelerator_device_count: int
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_visible_devices: str | None
    mps_built: bool
    mps_available: bool
    slurm_job_id: str | None
    slurm_cpus_per_task: int | None
    slurm_gpus: str | None
    slurm_detected_by_lightning: bool


@dataclass(frozen=True)
class RuntimeDiagnostic:
    """
    Structured runtime diagnostic produced by preflight checks or failure analysis.

    Context:
    the repository uses these diagnostics both before launching training and
    after environment-looking failures so CLI runs, notebooks, and summaries
    can all present the same message shape.
    """

    severity: str
    code: str
    message: str
    suggestion: str | None = None


@dataclass(frozen=True)
class DeviceProfileResolution:
    """
    Result of applying a requested device profile to runtime-facing configs.

    Context:
    profile resolution never mutates the caller's config objects in place. It
    returns the resolved profile name plus the updated runtime-facing config
    objects and the exact defaults that were applied.
    """

    requested_profile: str
    resolved_profile: str
    train_config: TrainConfig
    data_config: DataConfig
    observability_config: ObservabilityConfig
    applied_defaults: dict[str, Any]
