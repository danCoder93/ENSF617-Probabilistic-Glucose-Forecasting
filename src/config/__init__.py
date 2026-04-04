from __future__ import annotations

# AI-assisted maintenance note:
# This package is the canonical home for the repository's declarative
# configuration contracts.
#
# Why keep this file:
# - callers across training, evaluation, tests, and notebooks already expect
#   concise imports like `from config import Config, TrainConfig`
# - the package-level facade keeps those imports stable while the underlying
#   config implementation stays split into smaller focused modules
# - the re-export surface gives the repository one obvious public place to look
#   for "what knobs exist?" without forcing every caller to know the file layout
#
# Responsibility boundary:
# - expose the public configuration dataclasses and serialization helpers
# - preserve a stable import surface for the rest of the repository
# - keep the internal config package layout swappable without changing callers
#
# What does *not* live here:
# - validation logic details for each config class
# - runtime environment detection or profile selection
# - training orchestration or model behavior
#
# Internal layout:
# - `data.py` owns dataset- and loader-facing config
# - `model.py` owns top-level model architecture config objects
# - `runtime.py` owns Trainer/checkpoint execution policy
# - `observability.py` owns logging/reporting/profiler policy
# - `serde.py` owns checkpoint-friendly dict serialization helpers
# - `types.py` holds small shared type aliases used across config modules
#
# Important disclaimer:
# this file is intentionally a public facade, not the place where substantive
# config behavior should accumulate. Rich validation and normalization belong in
# the underlying modules so ownership stays clear.


# ============================================================================
# Public Config Surface
# ============================================================================
# The imports below are intentionally explicit rather than wildcard-based so
# maintainers can answer "what exactly does `from config import ...` promise?"
# by reading one short file.

from config.data import DataConfig
from config.model import Config, TCNConfig, TFTConfig
from config.observability import ObservabilityConfig
from config.runtime import SnapshotConfig, TrainConfig
from config.serde import (
    config_from_dict,
    config_to_dict,
    data_config_from_dict,
    data_config_to_dict,
    tcn_config_from_dict,
    tcn_config_to_dict,
    tft_config_from_dict,
    tft_config_to_dict,
)
from config.types import PathInput
# ------------------------------------------------------------------
# Cross-configuration validation exports
# ------------------------------------------------------------------
# These are intentionally limited to the public-facing validation API.
# Internal helpers (e.g., issue collectors) are not exported to avoid
# coupling external code to internal validation mechanics.

from config.validation import (
    ConfigurationValidationError,
    validate_runtime_configuration,
)


# `__all__` is the contract for the package-level facade above. Keeping it
# aligned with the re-export list makes the public config surface easier to
# inspect and keeps accidental helper leakage out of auto-complete.
__all__ = [
    "Config",
    "DataConfig",
    "ObservabilityConfig",
    "PathInput",
    "SnapshotConfig",
    "TCNConfig",
    "TFTConfig",
    "TrainConfig",
    "config_from_dict",
    "config_to_dict",
    "data_config_from_dict",
    "data_config_to_dict",
    "tcn_config_from_dict",
    "tcn_config_to_dict",
    "tft_config_from_dict",
    "tft_config_to_dict",
    # existing exports...
    "ConfigurationValidationError",
    "validate_runtime_configuration",
]
