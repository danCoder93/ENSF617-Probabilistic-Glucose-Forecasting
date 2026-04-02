from __future__ import annotations

# AI-assisted maintenance note:
# This package groups runtime detection, profile resolution, diagnostics, and
# runtime tuning into one operational surface that sits alongside, rather than
# inside, the pure configuration dataclasses.
#
# Design intent:
# - let callers import one compact operational API from `environment`
# - keep the internal split by responsibility (`types`, `detection`,
#   `profiles`, `diagnostics`, `tuning`) without forcing every caller to know it
# - make runtime-facing decisions discoverable in one package rather than
#   scattering them between `main.py`, `train.py`, and config modules
#
# Internal layout:
# - `types.py` defines the normalized runtime/environment dataclasses
# - `detection.py` probes the current host and backend surfaces
# - `profiles.py` maps one detected environment to repository-specific defaults
# - `diagnostics.py` validates and explains likely runtime mismatches
# - `tuning.py` applies low-level backend knobs after policy has been chosen
#
# Responsibility boundary:
# - answer "what machine/runtime am I on?"
# - answer "which runtime defaults should we apply here?"
# - answer "does this chosen setup look valid?"
# - apply the final low-level backend tuning actions when asked
#
# What does *not* live here:
# - the declarative configuration dataclasses themselves
# - model architecture or data semantics
# - the top-level training/evaluation workflow orchestration
#
# Important disclaimer:
# this `__init__` file is intentionally just a re-export surface. The real
# implementation details live in the submodules so ownership remains clear.


# ============================================================================
# Public Runtime Surface
# ============================================================================
# This explicit re-export list keeps the runtime package ergonomic for callers
# while making the boundary between "public environment API" and "internal
# submodule details" easy to inspect.

from environment.detection import detect_runtime_environment
from environment.diagnostics import (
    analyze_runtime_failure,
    collect_runtime_diagnostics,
    format_runtime_diagnostics,
    has_error_diagnostics,
)
from environment.profiles import infer_device_profile, resolve_device_profile
from environment.tuning import (
    RuntimeTuningReport,
    apply_runtime_environment_overrides,
    apply_runtime_tuning,
    maybe_compile_model,
    synchronize_runtime_device,
)
from environment.types import (
    DEVICE_PROFILE_CHOICES,
    DeviceProfileResolution,
    RuntimeDiagnostic,
    RuntimeEnvironment,
)


# `__all__` documents the stable runtime-facing surface the rest of the
# repository should depend on from this package.
__all__ = [
    "DEVICE_PROFILE_CHOICES",
    "DeviceProfileResolution",
    "RuntimeDiagnostic",
    "RuntimeEnvironment",
    "RuntimeTuningReport",
    "analyze_runtime_failure",
    "apply_runtime_environment_overrides",
    "apply_runtime_tuning",
    "collect_runtime_diagnostics",
    "detect_runtime_environment",
    "format_runtime_diagnostics",
    "has_error_diagnostics",
    "infer_device_profile",
    "maybe_compile_model",
    "resolve_device_profile",
    "synchronize_runtime_device",
]
