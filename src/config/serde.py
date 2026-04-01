from __future__ import annotations

# AI-assisted maintenance note:
# This module isolates serialization and deserialization helpers from the
# dataclass definitions themselves. Keeping the serde layer separate makes the
# individual config modules smaller and makes it easier to reason about what
# parts of the configuration contract are purely runtime objects versus what is
# persisted in checkpoints and summaries.

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from config.data import DataConfig
from config.model import Config, TCNConfig, TFTConfig
from utils.tft_utils import DataTypes, FeatureSpec, InputTypes


def _serialize_feature_spec(feature: FeatureSpec) -> dict[str, str]:
    """
    Convert one `FeatureSpec` into a plain checkpoint-friendly dictionary.

    Context:
    the serialized form should stay human-readable and not depend on Python
    object pickling or enum integer values.
    """
    # `FeatureSpec` is a namedtuple carrying enum values. Converting it to a
    # plain string-keyed payload keeps checkpoint metadata human-readable and
    # avoids depending on Python-specific object reconstruction behavior.
    return {
        "name": feature.name,
        "feature_type": InputTypes(feature.feature_type).name,
        "feature_embed_type": DataTypes(feature.feature_embed_type).name,
    }


def _deserialize_feature_spec(payload: Mapping[str, Any]) -> FeatureSpec:
    """
    Rebuild a typed `FeatureSpec` from its plain serialized representation.

    Context:
    this is the inverse of `_serialize_feature_spec(...)` and restores the enum
    types needed by config validation and model binding logic.
    """
    # Rebuild the original strongly typed feature declaration from the plain
    # checkpoint payload. Using enum names rather than raw integer values makes
    # the stored representation more stable and easier to inspect by eye.
    return FeatureSpec(
        str(payload["name"]),
        InputTypes[str(payload["feature_type"])],
        DataTypes[str(payload["feature_embed_type"])],
    )


def _serialize_dataclass_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize dataclass field values into JSON/checkpoint-friendly primitives.

    Context:
    config dataclasses contain `Path`, `FeatureSpec`, and tuple values that are
    convenient in Python but should be flattened before persistence.
    """
    serialized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            # `Path` objects are convenient in live Python code but should be
            # flattened to strings for checkpoint portability across machines
            # and notebook runtimes.
            serialized[key] = str(value)
        elif isinstance(value, FeatureSpec):
            serialized[key] = _serialize_feature_spec(value)
        elif isinstance(value, tuple):
            # Config dataclasses use tuples heavily for "immutable in practice"
            # semantics. Converting them to plain lists keeps the serialized
            # payload JSON/YAML/checkpoint friendly, while the corresponding
            # `*_from_dict(...)` helpers restore the typed tuple form.
            serialized[key] = [
                _serialize_feature_spec(item) if isinstance(item, FeatureSpec) else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


def data_config_to_dict(config: DataConfig) -> dict[str, Any]:
    """
    Serialize `DataConfig` into a plain dictionary.

    Context:
    only declared dataclass fields are emitted so the serialized form stays
    aligned with the public config contract.
    """
    # Serialize only the declared dataclass fields so checkpoint metadata stays
    # aligned with the public config contract rather than accidentally capturing
    # computed properties or future incidental attributes.
    payload = {field_info.name: getattr(config, field_info.name) for field_info in fields(DataConfig)}
    return _serialize_dataclass_payload(payload)


def tft_config_to_dict(config: TFTConfig) -> dict[str, Any]:
    """
    Serialize `TFTConfig` into a plain dictionary.

    Context:
    the output includes both declarative fields and derived counts so
    checkpoint reloads preserve the exact bound runtime contract.
    """
    # `TFTConfig` contains both declarative inputs and derived counts. We keep
    # the full dataclass field set so checkpoint reloads preserve the exact
    # bound configuration seen by the model at construction time.
    payload = {field_info.name: getattr(config, field_info.name) for field_info in fields(TFTConfig)}
    return _serialize_dataclass_payload(payload)


def tcn_config_to_dict(config: TCNConfig) -> dict[str, Any]:
    """
    Serialize `TCNConfig` into a plain dictionary.

    Context:
    this keeps the TCN branch aligned with the same checkpoint/summary format
    used for the other config objects.
    """
    # The TCN branch config is already close to plain-Python data, so the main
    # benefit here is consistency with the other config serializers.
    payload = {field_info.name: getattr(config, field_info.name) for field_info in fields(TCNConfig)}
    return _serialize_dataclass_payload(payload)


def config_to_dict(config: Config) -> dict[str, Any]:
    """
    Convert the top-level config into a checkpoint-friendly plain dictionary.

    Lightning stores hyperparameters inside checkpoint metadata. Serializing the
    nested config explicitly keeps that metadata portable across local scripts,
    Colab notebooks, and `load_from_checkpoint(...)` calls without relying on
    Python object pickling for custom dataclasses, Paths, or enum-backed feature
    specs.
    """
    return {
        "data": data_config_to_dict(config.data),
        "tft": tft_config_to_dict(config.tft),
        "tcn": tcn_config_to_dict(config.tcn),
    }


def data_config_from_dict(payload: Mapping[str, Any]) -> DataConfig:
    """
    Rebuild `DataConfig` from its serialized dictionary form.

    Context:
    feature declarations must be rehydrated before `DataConfig.__post_init__`
    runs so validation sees the original semantic schema.
    """
    data_payload = dict(payload)
    # Feature declarations are stored in plain serialized form inside the
    # checkpoint payload and need to be rebuilt before `DataConfig` validation
    # runs in `__post_init__`.
    data_payload["features"] = tuple(
        _deserialize_feature_spec(feature_payload)
        for feature_payload in data_payload.get("features", ())
    )
    return DataConfig(**data_payload)


def tft_config_from_dict(payload: Mapping[str, Any]) -> TFTConfig:
    """
    Rebuild `TFTConfig` from its serialized dictionary form.

    Context:
    the feature schema is restored first so the derived variable counts are
    recomputed against the same semantic inputs seen during the original run.
    """
    tft_payload = dict(payload)
    # Rehydrate feature specs first so the derived variable counts recomputed by
    # `TFTConfig.__post_init__` see the same semantic feature contract the model
    # originally used.
    tft_payload["features"] = tuple(
        _deserialize_feature_spec(feature_payload)
        for feature_payload in tft_payload.get("features", ())
    )
    return TFTConfig(**tft_payload)


def tcn_config_from_dict(payload: Mapping[str, Any]) -> TCNConfig:
    """
    Rebuild `TCNConfig` from its serialized dictionary form.

    Context:
    `TCNConfig.__post_init__` already handles normalization and validation, so
    plain reconstruction through the dataclass constructor is sufficient.
    """
    # `TCNConfig` normalizes sequences and validates values in `__post_init__`,
    # so simply passing the plain payload back through the dataclass
    # constructor is enough to recover the typed config safely.
    return TCNConfig(**dict(payload))


def config_from_dict(payload: Mapping[str, Any]) -> Config:
    """
    Rebuild the typed top-level config from `config_to_dict(...)` output.

    This keeps checkpoint reloads and notebook workflows simple: callers can
    persist only the plain dictionary form and recover the strongly typed config
    object when constructing the fused model again.
    """
    return Config(
        data=data_config_from_dict(payload["data"]),
        tft=tft_config_from_dict(payload["tft"]),
        tcn=tcn_config_from_dict(payload["tcn"]),
    )
