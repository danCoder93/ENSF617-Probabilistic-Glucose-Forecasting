# AI-assisted implementation note:
# This file was drafted with AI assistance and then reviewed/adapted for this
# project. The refactor draws on the earlier AZT1D pipeline in this repo, prior
# work by SlickMik (https://github.com/SlickMik), the PyTorch Lightning
# DataModule docs/tutorial
# (https://lightning.ai/docs/pytorch/stable/data/datamodule.html), and the
# original AZT1D dataset release on Mendeley Data
# (https://data.mendeley.com/datasets/gk9m674wcx/1). Its purpose is to define the
# shared schema contract used across preprocessing, indexing, datasets, and
# model configuration.

from __future__ import annotations

from dataclasses import dataclass

from config import DataConfig
from utils.tft_utils import DataTypes, InputTypes


# ============================================================
# Raw-to-canonical column contract
# ============================================================
# Purpose:
#   Define the stable names used throughout the pipeline after
#   raw AZT1D files are standardized.
# ============================================================

# Raw AZT1D files use vendor-facing column names. The preprocessor rewrites them
# into these canonical names once so the rest of the pipeline can speak a single
# vocabulary. That keeps download/build concerns separate from dataset/model
# concerns and prevents "if this raw column exists then..." logic from leaking
# into training code.
RAW_TO_INTERNAL_COLUMN_MAP = {
    "EventDateTime": "timestamp",
    "Basal": "basal_insulin_u",
    "TotalBolusInsulinDelivered": "bolus_insulin_u",
    "CorrectionDelivered": "correction_insulin_u",
    "FoodDelivered": "meal_insulin_u",
    "CarbSize": "carbs_g",
    "DeviceMode": "device_mode",
    "BolusType": "bolus_type",
}


# The raw exports are not perfectly standardized, so we accept either glucose
# spelling and rewrite to the canonical target column during preprocessing.
RAW_GLUCOSE_COLUMNS = ("CGM", "Readings (CGM / BGM)")


# ============================================================
# Declared categorical vocabularies
# ============================================================
# Purpose:
#   Keep category ordering explicit so categorical IDs and
#   embedding cardinalities stay stable across the data/model
#   boundary.
# ============================================================

# These categories are intentionally centralized. The transform layer uses them
# to normalize messy raw strings, and the dataset/data module use the same
# ordering when they turn categories into integer IDs. Keeping the ordering in
# one place is what prevents data/model cardinalities from silently drifting.
DEVICE_MODE_CATEGORIES = ("none", "sleep", "exercise", "other")
BOLUS_TYPE_CATEGORIES = (
    "none",
    "automatic",
    "standard",
    "standard_correction",
    "ble_standard",
    "ble_standard_correction",
    "quick",
    "extended",
    "extended_correction",
    "other",
)


# Default feature groups mirror the current fused-pipeline sample semantics:
# subject identity is static, time features are known in advance, insulin/carb
# activity is observed history, and the glucose target is forecasted.
DEFAULT_KNOWN_CONTINUOUS_COLUMNS = (
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
)
DEFAULT_OBSERVED_CONTINUOUS_COLUMNS = (
    "basal_insulin_u",
    "bolus_insulin_u",
    "correction_insulin_u",
    "meal_insulin_u",
    "carbs_g",
)
DEFAULT_OBSERVED_CATEGORICAL_COLUMNS = ("device_mode", "bolus_type")
# These defaults are a bridge for the refactor period. The long-term goal is
# for `config.features` to be the single source of truth, but keeping sensible
# AZT1D defaults here lets the new architecture work immediately without forcing
# the rest of the codebase to migrate in one risky step.


# ============================================================
# Feature grouping contract
# ============================================================
# Purpose:
#   Represent the semantic feature groups consumed by the fused
#   forecasting pipeline.
#
# Context:
#   The dataframe is column-based, but the model contract is
#   tensor-group based. This object is the translation layer.
# ============================================================
@dataclass(frozen=True)
class FeatureGroups:
    """
    The data contract used across transforms, indexing, and dataset assembly.

    Context:
    - The raw dataframe is column-oriented.
    - The model pipeline is tensor-group oriented.
    - We need one place that explains how columns map into semantic groups.

    This object is that bridge. The DataModule derives it once from `FeatureSpec`
    (or a well-documented fallback), then every lower layer consumes the same
    grouping instead of hardcoding its own interpretation.
    """

    subject_id_column: str
    time_column: str
    target_column: str
    static_categorical: tuple[str, ...]
    static_continuous: tuple[str, ...]
    known_categorical: tuple[str, ...]
    known_continuous: tuple[str, ...]
    observed_categorical: tuple[str, ...]
    observed_continuous: tuple[str, ...]

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        # All categorical columns that need one shared category-to-ID mapping.
        return _ordered_unique(
            self.static_categorical
            + self.known_categorical
            + self.observed_categorical
        )

    @property
    def continuous_columns(self) -> tuple[str, ...]:
        # All continuous columns that must be numerically clean and ready for
        # slicing before any Dataset is instantiated.
        return _ordered_unique(
            self.static_continuous
            + self.known_continuous
            + self.observed_continuous
            + (self.target_column,)
        )

    @property
    def encoder_continuous(self) -> tuple[str, ...]:
        # Encoder history must expose every continuous signal available up to the
        # forecast origin. That includes:
        # 1. known-ahead continuous signals on the historical axis
        # 2. observed continuous signals seen only after they happen
        # 3. the historical target trajectory itself
        #
        # Including the target history here preserves the original "past observed
        # inputs" behavior while still exposing the future target separately.
        return _ordered_unique(
            self.known_continuous
            + self.observed_continuous
            + (self.target_column,)
        )

    @property
    def encoder_categorical(self) -> tuple[str, ...]:
        # Encoder history includes both known-ahead and observed categorical
        # signals because both are available on the historical axis.
        return _ordered_unique(self.known_categorical + self.observed_categorical)

    @property
    def decoder_known_continuous(self) -> tuple[str, ...]:
        # Decoder inputs are intentionally restricted to "known" features only.
        # Observed features and targets are not available from the future at
        # inference time, so including them here would create leakage.
        return self.known_continuous

    @property
    def decoder_known_categorical(self) -> tuple[str, ...]:
        return self.known_categorical


def build_feature_groups(config: DataConfig) -> FeatureGroups:
    """
    Derive the semantic column groups from the shared config contract.

    Preferred path:
    - Use `config.features` so the data layer and model layer share one schema.

    Fallback path:
    - If the feature schema has not been filled in yet, we fall back to the
      current AZT1D defaults. This keeps the refactor usable immediately without
      requiring any changes to `config.py`, which the user asked us not to touch.
    """

    # Preferred path: derive groups from the shared FeatureSpec schema so the
    # data layer and model layer literally read from the same contract.
    if config.features:
        static_categorical = tuple(
            feature.name
            for feature in config.features
            if feature.feature_type == InputTypes.STATIC
            and feature.feature_embed_type == DataTypes.CATEGORICAL
        )
        static_continuous = tuple(
            feature.name
            for feature in config.features
            if feature.feature_type == InputTypes.STATIC
            and feature.feature_embed_type == DataTypes.CONTINUOUS
        )
        known_categorical = tuple(
            feature.name
            for feature in config.features
            if feature.feature_type == InputTypes.KNOWN
            and feature.feature_embed_type == DataTypes.CATEGORICAL
        )
        known_continuous = tuple(
            feature.name
            for feature in config.features
            if feature.feature_type == InputTypes.KNOWN
            and feature.feature_embed_type == DataTypes.CONTINUOUS
        )
        observed_categorical = tuple(
            feature.name
            for feature in config.features
            if feature.feature_type == InputTypes.OBSERVED
            and feature.feature_embed_type == DataTypes.CATEGORICAL
        )
        observed_continuous = tuple(
            feature.name
            for feature in config.features
            if feature.feature_type == InputTypes.OBSERVED
            and feature.feature_embed_type == DataTypes.CONTINUOUS
        )
    else:
        # Fallback groups mirror the original handcrafted pipeline semantics.
        # They are explicitly AZT1D-specific and should eventually disappear once
        # every caller populates `config.features`.
        static_categorical = (config.subject_id_column,)
        static_continuous = ()
        known_categorical = ()
        known_continuous = DEFAULT_KNOWN_CONTINUOUS_COLUMNS
        observed_categorical = DEFAULT_OBSERVED_CATEGORICAL_COLUMNS
        observed_continuous = DEFAULT_OBSERVED_CONTINUOUS_COLUMNS

    # Subject identity is the one static signal guaranteed by the current raw
    # dataset. If the feature schema omitted it, we still inject it as a static
    # categorical so the sequence dataset can provide a stable per-subject anchor
    # and expose cardinalities needed by the model side.
    if config.subject_id_column not in static_categorical:
        static_categorical = (config.subject_id_column,) + tuple(static_categorical)

    return FeatureGroups(
        subject_id_column=config.subject_id_column,
        time_column=config.time_column,
        target_column=config.target_column,
        static_categorical=_ordered_unique(static_categorical),
        static_continuous=_ordered_unique(static_continuous),
        known_categorical=_ordered_unique(known_categorical),
        known_continuous=_ordered_unique(known_continuous),
        observed_categorical=_ordered_unique(observed_categorical),
        observed_continuous=_ordered_unique(observed_continuous),
    )


def declared_category_order(column_name: str) -> tuple[str, ...] | None:
    """
    Return the canonical category order for known categorical columns.

    Context:
    returning `None` means the column is still categorical, but its vocabulary
    should be discovered from the cleaned dataframe rather than hardcoded.
    """
    # `None` means the column is still categorical, but its vocabulary should be
    # discovered from the cleaned dataframe rather than hardcoded.
    if column_name == "device_mode":
        return DEVICE_MODE_CATEGORIES
    if column_name == "bolus_type":
        return BOLUS_TYPE_CATEGORIES
    return None


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    """
    Preserve the first occurrence of each feature name while removing duplicates.

    Context:
    tensor column order must stay stable across data preparation, checkpoint
    serialization, and model construction, so deduplication cannot scramble the
    declared order.
    """
    # Feature order matters because tensor column order must be stable across
    # every batch, checkpoint, and model initialization. A small helper keeps
    # that guarantee explicit and reusable.
    return tuple(dict.fromkeys(values))
