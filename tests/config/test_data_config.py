from __future__ import annotations

# These tests protect the narrow `DataConfig` contract used by the data layer.
#
# They focus on two responsibilities:
# - normalizing user-facing path inputs into stable `Path` objects
# - rejecting split/dataloader settings that would make the data contract
#   inconsistent before the rest of the pipeline starts

from pathlib import Path
from typing import cast

import pytest

from config import DataConfig
from utils.tft_utils import DataTypes, FeatureSpec, InputTypes


def _feature_specs(*specs: FeatureSpec) -> tuple[FeatureSpec, ...]:
    # Keeping the tuple helper local makes the tests read in terms of semantic
    # feature declarations instead of casts.
    return cast(tuple[FeatureSpec, ...], specs)


def test_data_config_normalizes_paths_and_preserves_processed_file_contract(
    tmp_path: Path,
) -> None:
    # Path normalization is a core part of the config contract because the data
    # layer accepts user-friendly strings but downstream code expects `Path`.
    features = _feature_specs(
        FeatureSpec("subject_id", InputTypes.STATIC, DataTypes.CATEGORICAL),
    )

    config = DataConfig(
        raw_dir=str(tmp_path / "raw"),
        cache_dir=str(tmp_path / "cache"),
        extracted_dir=str(tmp_path / "extracted"),
        processed_dir=str(tmp_path / "processed"),
        processed_file_name="dataset.csv",
        features=features,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    assert isinstance(config.raw_dir, Path)
    assert isinstance(config.cache_dir, Path)
    assert isinstance(config.extracted_dir, Path)
    assert isinstance(config.processed_dir, Path)
    assert config.features == features
    assert config.processed_file_path == tmp_path / "processed" / "dataset.csv"

    with pytest.raises(ValueError, match="prefetch_factor"):
        DataConfig(prefetch_factor=0)


def test_data_config_rejects_invalid_split_contracts() -> None:
    # Split policy mistakes should fail at config construction time rather than
    # surfacing later as confusing empty-window or leakage issues.
    with pytest.raises(ValueError, match="sum to 1.0"):
        DataConfig(train_ratio=0.8, val_ratio=0.15, test_ratio=0.15)

    with pytest.raises(ValueError, match="cannot both be True"):
        DataConfig(split_by_subject=True, split_within_subject=True)
