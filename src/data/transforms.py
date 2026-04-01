"""
AI-assisted implementation note:
This file was drafted with AI assistance and then reviewed/adapted for this
project. The refactor draws on the earlier AZT1D pipeline in this repo, prior
work by SlickMik (https://github.com/SlickMik), the PyTorch Lightning
DataModule docs/tutorial
(https://lightning.ai/docs/pytorch/stable/data/datamodule.html), and the
original AZT1D dataset release on Mendeley Data
(https://data.mendeley.com/datasets/gk9m674wcx/1). Its purpose is to isolate
cleaning and normalization as reusable dataframe transforms for the new data
contract.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pandas as pd

from data.schema import (
    FeatureGroups,
    declared_category_order,
)
from utils.config import DataConfig


# ============================================================
# Dataframe loading and normalization
# ============================================================
# Purpose:
#   Convert the processed CSV into a cleaned dataframe ready for
#   split construction and sequence indexing.
#
# Why this is separate from the Dataset:
#   These are dataframe-wide transforms that should happen once
#   during setup, not once per sample inside __getitem__.
# ============================================================

def load_processed_frame(
    processed_csv_path: str | Path,
    config: DataConfig,
    feature_groups: FeatureGroups,
) -> pd.DataFrame:
    """
    Load and normalize the processed AZT1D table.

    Architectural intent:
    - This function owns dataframe-level cleanup only.
    - It does not split subjects.
    - It does not construct window samples.
    - It does not create tensors.

    That boundary keeps all table-wide normalization work in one testable layer
    and lets the Dataset stay focused on one sample at a time.
    """

    dataframe = pd.read_csv(processed_csv_path)

    dataframe[feature_groups.subject_id_column] = (
        dataframe[feature_groups.subject_id_column].astype(str).str.strip()
    )
    dataframe[feature_groups.time_column] = pd.to_datetime(
        dataframe[feature_groups.time_column],
        errors="coerce",
    )
    dataframe[feature_groups.target_column] = pd.to_numeric(
        dataframe[feature_groups.target_column],
        errors="coerce",
    )

    # Rows missing identity, time, or target cannot participate in supervised
    # sequence forecasting, so they are removed before any further processing.
    dataframe = dataframe.dropna(
        subset=[
            feature_groups.subject_id_column,
            feature_groups.time_column,
            feature_groups.target_column,
        ]
    ).copy()
    dataframe[feature_groups.time_column] = dataframe[feature_groups.time_column].dt.round("min")
    dataframe = dataframe.sort_values(
        [feature_groups.subject_id_column, feature_groups.time_column]
    ).reset_index(drop=True)

    continuous_columns = tuple(
        column
        for column in feature_groups.continuous_columns
        if column != feature_groups.target_column
    )
    # Operational features such as insulin and carb events behave like sparse
    # signals in this dataset, so missing values are interpreted as "no event"
    # and filled with zeros after numeric coercion.
    for column in continuous_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0)

    if "device_mode" in dataframe.columns:
        dataframe["device_mode"] = _prepare_text_column(dataframe["device_mode"])
        dataframe["device_mode"] = dataframe["device_mode"].replace({"": pd.NA, "0": pd.NA})
        # Device mode is forward-filled within each subject because these states
        # often persist until the device reports a new one. Filling across
        # subjects would be nonsensical, which is why the groupby key matters.
        dataframe["device_mode"] = dataframe.groupby(feature_groups.subject_id_column)["device_mode"].ffill()
        dataframe["device_mode"] = dataframe["device_mode"].fillna("none")
        dataframe["device_mode"] = dataframe["device_mode"].apply(normalize_device_mode)

    if "bolus_type" in dataframe.columns:
        dataframe["bolus_type"] = _prepare_text_column(dataframe["bolus_type"])
        dataframe["bolus_type"] = dataframe["bolus_type"].replace({"": pd.NA, "0": pd.NA})
        # Bolus type is handled the same way: within-subject forward-fill treats
        # missing raw entries as "same event context as the previous row" when
        # the export omits repeated text values.
        dataframe["bolus_type"] = dataframe.groupby(feature_groups.subject_id_column)["bolus_type"].ffill()
        dataframe["bolus_type"] = dataframe["bolus_type"].fillna("none")
        dataframe["bolus_type"] = dataframe["bolus_type"].apply(normalize_bolus_type)

    dataframe = _add_time_features(dataframe, feature_groups.time_column)
    dataframe = _normalize_declared_categories(dataframe, feature_groups)
    _validate_required_columns(dataframe, feature_groups, config)

    return dataframe


def build_category_maps(
    dataframe: pd.DataFrame,
    feature_groups: FeatureGroups,
) -> dict[str, tuple[str, ...]]:
    """
    Build stable category vocabularies for every categorical feature.

    The DataModule fits these vocabularies once on the cleaned dataframe and
    passes them to each dataset split. That gives train/val/test identical
    integer IDs, which is required for reproducible embedding cardinalities.
    """

    category_maps: dict[str, tuple[str, ...]] = {}
    for column in feature_groups.categorical_columns:
        if column not in dataframe.columns:
            continue

        declared_order = declared_category_order(column)
        if declared_order is not None:
            category_maps[column] = declared_order
            continue

        values = dataframe[column].dropna().astype(str).str.strip()
        category_maps[column] = tuple(sorted(value for value in values.unique() if value != ""))

    return category_maps


def normalize_device_mode(value: object) -> str:
    # These normalization helpers intentionally collapse messy raw spellings into
    # a small controlled vocabulary. That keeps category IDs stable and reduces
    # the chance that the model learns spurious distinctions from export noise.
    if pd.isna(cast(Any, value)):
        return "none"

    normalized = str(value).strip().lower()
    if normalized in {"", "0", "none"}:
        return "none"
    if normalized == "sleepsleep":
        return "sleep"
    if normalized in {"sleep", "exercise"}:
        return normalized
    return "other"


def normalize_bolus_type(value: object) -> str:
    if pd.isna(cast(Any, value)):
        return "none"

    normalized = str(value).strip().lower()
    if normalized in {"", "0", "none"}:
        return "none"
    if "automatic bolus" in normalized:
        return "automatic"
    if "ble standard bolus/correction" in normalized:
        return "ble_standard_correction"
    if "ble standard bolus" in normalized:
        return "ble_standard"
    if normalized == "standard/correction":
        return "standard_correction"
    if normalized == "standard":
        return "standard"
    if normalized == "quick":
        return "quick"
    if normalized.startswith("extended/correction"):
        return "extended_correction"
    if normalized.startswith("extended"):
        return "extended"
    return "other"


def _add_time_features(dataframe: pd.DataFrame, time_column: str) -> pd.DataFrame:
    # Cyclical encodings let the model see periodic structure without the sharp
    # discontinuities that raw hour/day integers would create.
    minute_of_day = dataframe[time_column].dt.hour * 60 + dataframe[time_column].dt.minute
    day_of_week = dataframe[time_column].dt.dayofweek

    dataframe["minute_of_day_sin"] = _sin_from_period(minute_of_day, 1440.0)
    dataframe["minute_of_day_cos"] = _cos_from_period(minute_of_day, 1440.0)
    dataframe["day_of_week_sin"] = _sin_from_period(day_of_week, 7.0)
    dataframe["day_of_week_cos"] = _cos_from_period(day_of_week, 7.0)
    dataframe["is_weekend"] = (day_of_week >= 5).astype("float32")

    return dataframe


def _normalize_declared_categories(
    dataframe: pd.DataFrame,
    feature_groups: FeatureGroups,
) -> pd.DataFrame:
    for column in feature_groups.categorical_columns:
        if column not in dataframe.columns:
            continue

        declared_order = declared_category_order(column)
        if declared_order is None:
            # Free-form categories are normalized to stripped strings so later
            # vocabulary fitting sees one stable textual representation.
            dataframe[column] = dataframe[column].fillna("").astype(str).str.strip()
            continue

        # Declared vocabularies are enforced here so unexpected spellings are
        # normalized before category IDs are fitted.
        dataframe[column] = pd.Categorical(
            dataframe[column],
            categories=list(declared_order),
        ).astype(str)

    return dataframe


def _validate_required_columns(
    dataframe: pd.DataFrame,
    feature_groups: FeatureGroups,
    config: DataConfig,
) -> None:
    # Turn schema drift into a setup-time error. That is much easier to diagnose
    # than discovering the mismatch later inside model forward passes.
    required_columns = {
        feature_groups.subject_id_column,
        feature_groups.time_column,
        feature_groups.target_column,
        *feature_groups.static_categorical,
        *feature_groups.static_continuous,
        *feature_groups.known_categorical,
        *feature_groups.known_continuous,
        *feature_groups.observed_categorical,
        *feature_groups.observed_continuous,
    }
    missing_columns = sorted(column for column in required_columns if column not in dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The processed dataframe is missing required columns for the declared "
            f"feature schema: {', '.join(missing_columns)}. "
            f"Processed file: {config.processed_file_path}"
        )


def _prepare_text_column(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _sin_from_period(values: pd.Series, period: float) -> pd.Series:
    angles = values.astype("float64") * (2.0 * math.pi / period)
    return angles.apply(math.sin).astype("float32")


def _cos_from_period(values: pd.Series, period: float) -> pd.Series:
    angles = values.astype("float64") * (2.0 * math.pi / period)
    return angles.apply(math.cos).astype("float32")
