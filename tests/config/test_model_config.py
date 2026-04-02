from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from config import (
    Config,
    DataConfig,
    TCNConfig,
    TFTConfig,
    config_from_dict,
    config_to_dict,
)
from utils.tft_utils import DataTypes, FeatureSpec, InputTypes


def _feature_specs(*specs: FeatureSpec) -> tuple[FeatureSpec, ...]:
    return cast(tuple[FeatureSpec, ...], specs)


def test_tcn_config_accepts_sequence_inputs_used_by_older_call_sites() -> None:
    config = TCNConfig(
        num_inputs=3,
        num_channels=(8, 16),
        dilations=(1, 2),
        prediction_length=6,
        output_size=2,
    )

    assert config.num_channels == (8, 16)
    assert config.dilations == (1, 2)
    assert config.prediction_length == 6
    assert config.output_size == 2
    assert config.use_norm == "layer_norm"


def test_tcn_config_rejects_normalization_modes_not_supported_by_refactor() -> None:
    with pytest.raises(ValueError, match="layer_norm"):
        TCNConfig(
            num_inputs=2,
            num_channels=(8, 8),
            dilations=(1, 2),
            use_norm="weight_norm",
        )


def test_tft_config_derives_counts_from_feature_schema_and_runtime_metadata() -> None:
    features = _feature_specs(
        FeatureSpec("subject_id", InputTypes.STATIC, DataTypes.CATEGORICAL),
        FeatureSpec("age_years", InputTypes.STATIC, DataTypes.CONTINUOUS),
        FeatureSpec("hour", InputTypes.KNOWN, DataTypes.CONTINUOUS),
        FeatureSpec("device_mode", InputTypes.OBSERVED, DataTypes.CATEGORICAL),
        FeatureSpec("carbs_g", InputTypes.OBSERVED, DataTypes.CONTINUOUS),
        FeatureSpec("glucose_mg_dl", InputTypes.TARGET, DataTypes.CONTINUOUS),
    )

    config = TFTConfig(
        features=features,
        static_categorical_inp_lens=(10,),
        temporal_known_categorical_inp_lens=(),
        temporal_observed_categorical_inp_lens=(4,),
        num_aux_future_features=3,
    )

    assert config.static_continuous_inp_size == 1
    assert config.temporal_known_continuous_inp_size == 1
    assert config.temporal_observed_continuous_inp_size == 1
    assert config.temporal_target_size == 1
    assert config.num_static_vars == 2
    assert config.num_future_vars == 4
    assert config.num_historic_vars == 7
    assert config.layer_norm_eps == 1e-3


def test_tft_config_supports_runtime_rebinding_used_by_datamodule_and_fused_model() -> None:
    features = _feature_specs(
        FeatureSpec("subject_id", InputTypes.STATIC, DataTypes.CATEGORICAL),
        FeatureSpec("hour", InputTypes.KNOWN, DataTypes.CONTINUOUS),
        FeatureSpec("glucose_mg_dl", InputTypes.TARGET, DataTypes.CONTINUOUS),
    )

    base_config = TFTConfig(features=features)
    bound_config = replace(
        base_config,
        static_categorical_inp_lens=(12,),
        encoder_length=24,
        example_length=30,
        num_aux_future_features=3,
    )

    assert base_config.static_categorical_inp_lens == ()
    assert bound_config.static_categorical_inp_lens == (12,)
    assert bound_config.encoder_length == 24
    assert bound_config.example_length == 30
    assert bound_config.num_future_vars == 4
    assert bound_config.num_historic_vars == 5


def test_tft_config_validates_layer_norm_epsilon() -> None:
    with pytest.raises(ValueError, match="layer_norm_eps must be > 0.0"):
        TFTConfig(layer_norm_eps=0.0)


def test_top_level_config_groups_data_tft_and_tcn_contracts() -> None:
    config = Config(
        data=DataConfig(num_workers=0, pin_memory=False, persistent_workers=False),
        tft=TFTConfig(),
        tcn=TCNConfig(),
    )

    assert isinstance(config.data, DataConfig)
    assert isinstance(config.tft, TFTConfig)
    assert isinstance(config.tcn, TCNConfig)


def test_top_level_config_round_trips_through_checkpoint_friendly_dict() -> None:
    features = _feature_specs(
        FeatureSpec("subject_id", InputTypes.STATIC, DataTypes.CATEGORICAL),
        FeatureSpec("hour", InputTypes.KNOWN, DataTypes.CONTINUOUS),
        FeatureSpec("glucose_mg_dl", InputTypes.TARGET, DataTypes.CONTINUOUS),
    )

    original = Config(
        data=DataConfig(
            dataset_url=None,
            features=features,
            raw_dir=Path("custom/raw"),
            processed_dir=Path("custom/processed"),
            num_workers=0,
            prefetch_factor=3,
            pin_memory=False,
            persistent_workers=False,
        ),
        tft=TFTConfig(
            features=features,
            static_categorical_inp_lens=(5,),
            encoder_length=24,
            example_length=30,
        ),
        tcn=TCNConfig(
            num_inputs=2,
            num_channels=(8, 16),
            dilations=(1, 2),
        ),
    )

    payload = config_to_dict(original)
    restored = config_from_dict(payload)

    assert payload["data"]["raw_dir"] == "custom/raw"
    assert payload["data"]["prefetch_factor"] == 3
    assert payload["tft"]["features"][0]["feature_type"] == "STATIC"
    assert restored == original
