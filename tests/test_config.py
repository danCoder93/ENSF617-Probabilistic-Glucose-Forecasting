"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates the shared config contracts used by the refactored
data and model layers.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from utils.config import (
    Config,
    DataConfig,
    TCNConfig,
    TFTConfig,
    config_from_dict,
    config_to_dict,
)
from utils.tft_utils import DataTypes, FeatureSpec, InputTypes


# ============================================================
# Config tests
# ============================================================
# Purpose:
#   Verify that the shared config layer remains compatible with
#   both the older declarative construction style and the newer
#   runtime-bound data/model integration path.
# ============================================================


def _feature_specs(*specs: FeatureSpec) -> tuple[FeatureSpec, ...]:
    """
    Return a typed tuple of FeatureSpec entries.

    Keeping this helper local makes the test literals a little clearer and
    gives static analyzers a concrete tuple type instead of inferring from
    inline list/tuple expressions.
    """
    return cast(tuple[FeatureSpec, ...], specs)


def test_data_config_normalizes_paths_and_preserves_processed_file_contract(
    tmp_path: Path,
) -> None:
    # Callers in scripts, notebooks, and tests may pass either strings or
    # `Path` objects. Normalizing them in one place keeps the rest of the
    # codebase free from repetitive path coercion.
    features = _feature_specs(
        FeatureSpec("subject_id", InputTypes.STATIC, DataTypes.CATEGORICAL),
    )
    path_overrides: dict[str, Any] = {
        "raw_dir": str(tmp_path / "raw"),
        "cache_dir": str(tmp_path / "cache"),
        "extracted_dir": str(tmp_path / "extracted"),
        "processed_dir": str(tmp_path / "processed"),
    }

    config = DataConfig(
        processed_file_name="dataset.csv",
        features=features,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        **path_overrides,
    )

    assert isinstance(config.raw_dir, Path)
    assert isinstance(config.cache_dir, Path)
    assert isinstance(config.extracted_dir, Path)
    assert isinstance(config.processed_dir, Path)
    assert config.features == (
        FeatureSpec("subject_id", InputTypes.STATIC, DataTypes.CATEGORICAL),
    )
    assert config.processed_file_path == tmp_path / "processed" / "dataset.csv"


def test_tcn_config_accepts_sequence_inputs_used_by_older_call_sites() -> None:
    # Older call sites commonly supplied plain Python lists for channel widths
    # and dilations. Keeping that construction style working helps the config
    # remain compatible while the implementation underneath has been narrowed.
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
    # The lean TCN branch now hard-codes layer norm internally. This test makes
    # that narrower contract explicit so unsupported legacy options do not
    # silently appear to work.
    with pytest.raises(ValueError, match="layer_norm"):
        TCNConfig(
            num_inputs=2,
            num_channels=(8, 8),
            dilations=(1, 2),
            use_norm="weight_norm",
        )


def test_tft_config_derives_counts_from_feature_schema_and_runtime_metadata() -> None:
    # The TFT config is now the bridge between declarative feature semantics and
    # runtime metadata discovered by the data pipeline, so both sources need to
    # be reflected in the derived counts.
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
    assert config.num_historic_vars == 6
    assert config.layer_norm_eps == 1e-3


def test_tft_config_supports_runtime_rebinding_used_by_datamodule_and_fused_model() -> None:
    # The newer pipeline uses `replace(...)` to bind sequence lengths,
    # categorical cardinalities, and auxiliary future channels after the data
    # contract is known. This should stay easy and safe to do.
    features = _feature_specs(
        FeatureSpec("subject_id", InputTypes.STATIC, DataTypes.CATEGORICAL),
        FeatureSpec("hour", InputTypes.KNOWN, DataTypes.CONTINUOUS),
        FeatureSpec("glucose_mg_dl", InputTypes.TARGET, DataTypes.CONTINUOUS),
    )

    base_config = TFTConfig(
        features=features
    )

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
    # The GRN refactor now lets TFT-owned normalization share one epsilon value
    # from config, so invalid values should fail before model construction.
    with pytest.raises(ValueError, match="layer_norm_eps must be > 0.0"):
        TFTConfig(layer_norm_eps=0.0)


def test_top_level_config_groups_data_tft_and_tcn_contracts() -> None:
    # The top-level config object is intentionally lightweight, but a direct
    # test still helps protect the shared constructor shape used across the
    # training/bootstrap path.
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
    assert payload["tft"]["features"][0]["feature_type"] == "STATIC"
    assert restored == original
