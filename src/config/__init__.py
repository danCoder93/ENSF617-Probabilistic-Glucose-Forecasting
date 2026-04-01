from __future__ import annotations

# AI-assisted maintenance note:
# This package is the canonical home for the repository's configuration
# contracts. It re-exports the public config API from smaller focused modules
# so callers can keep convenient imports like `from config import Config,
# TrainConfig` without having to know the internal file split.

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
]
