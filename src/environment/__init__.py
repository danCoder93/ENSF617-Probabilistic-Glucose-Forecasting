from __future__ import annotations

# AI-assisted maintenance note:
# This package groups runtime environment detection, profile resolution, and
# diagnostics into one operational surface that sits alongside, rather than
# inside, the pure configuration dataclasses.
#
# Design intent:
# - let callers import one compact operational API from `environment`
# - keep the internal split by responsibility (`types`, `detection`,
#   `profiles`, `diagnostics`) without forcing every caller to know it
#
# Important disclaimer:
# this `__init__` file is intentionally just a re-export surface. The real
# implementation details live in the submodules so ownership remains clear.

from environment.detection import detect_runtime_environment
from environment.diagnostics import (
    analyze_runtime_failure,
    collect_runtime_diagnostics,
    format_runtime_diagnostics,
    has_error_diagnostics,
)
from environment.profiles import infer_device_profile, resolve_device_profile
from environment.types import (
    DEVICE_PROFILE_CHOICES,
    DeviceProfileResolution,
    RuntimeDiagnostic,
    RuntimeEnvironment,
)

__all__ = [
    "DEVICE_PROFILE_CHOICES",
    "DeviceProfileResolution",
    "RuntimeDiagnostic",
    "RuntimeEnvironment",
    "analyze_runtime_failure",
    "collect_runtime_diagnostics",
    "detect_runtime_environment",
    "format_runtime_diagnostics",
    "has_error_diagnostics",
    "infer_device_profile",
    "resolve_device_profile",
]
