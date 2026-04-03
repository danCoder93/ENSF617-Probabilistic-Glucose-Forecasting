# AI-assisted implementation note:
# This file was drafted with AI assistance and then reviewed/adapted for this
# project. The refactor draws on the earlier AZT1D pipeline in this repo, prior
# work by SlickMik (https://github.com/SlickMik), the PyTorch Lightning
# DataModule docs/tutorial
# (https://lightning.ai/docs/pytorch/stable/data/datamodule.html), and the
# original AZT1D dataset release on Mendeley Data
# (https://data.mendeley.com/datasets/gk9m674wcx/1). Its purpose is to isolate
# cleaning and normalization as reusable dataframe transforms for the new data
# contract.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pandas as pd

from data.schema import (
    FeatureGroups,
    declared_category_order,
)
from config import DataConfig


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
    dataframe = _deduplicate_exact_rows(dataframe)
    dataframe = _collapse_duplicate_timestamps(dataframe, feature_groups)

    # The AZT1D paper treats basal insulin as a carried state on the unified
    # 5-minute grid, while bolus/correction/meal/carbohydrate values behave like
    # sparse events that should become zeros when no event was recorded.
    if "basal_insulin_u" in dataframe.columns:
        dataframe["basal_insulin_u"] = pd.to_numeric(
            dataframe["basal_insulin_u"],
            errors="coerce",
        )
        dataframe["basal_insulin_u"] = dataframe.groupby(feature_groups.subject_id_column)[
            "basal_insulin_u"
        ].ffill()
        dataframe["basal_insulin_u"] = dataframe.groupby(feature_groups.subject_id_column)[
            "basal_insulin_u"
        ].bfill()
        dataframe["basal_insulin_u"] = dataframe["basal_insulin_u"].fillna(0.0)

    event_continuous_columns = (
        "bolus_insulin_u",
        "correction_insulin_u",
        "meal_insulin_u",
        "carbs_g",
    )
    # Event-style quantities are sparse by nature in AZT1D. Once the table is on
    # the shared 5-minute grid, a missing value means no event was recorded for
    # that interval, so zero is the correct semantic fill.
    for column in event_continuous_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0)

    if "device_mode" in dataframe.columns:
        dataframe["device_mode"] = _prepare_text_column(dataframe["device_mode"])
        dataframe["device_mode"] = dataframe["device_mode"].replace(
            {"0": "regular", "0.0": "regular", "": pd.NA}
        )
        # Device mode is a persistent per-subject state in AZT1D. Blank rows mean
        # "same mode as before", and leading gaps fall back to the paper's
        # default regular mode rather than to an abstract missing category.
        dataframe["device_mode"] = dataframe.groupby(feature_groups.subject_id_column)["device_mode"].ffill()
        dataframe["device_mode"] = dataframe["device_mode"].fillna("regular")
        dataframe["device_mode"] = dataframe["device_mode"].apply(normalize_device_mode)

    if "bolus_type" in dataframe.columns:
        dataframe["bolus_type"] = _prepare_text_column(dataframe["bolus_type"])
        # Bolus type is event-local rather than stateful. Missing values should
        # stay tied to "no bolus event here" instead of being propagated across
        # later timesteps by forward-fill.
        dataframe["bolus_type"] = dataframe["bolus_type"].replace({"0": "", "none": ""})
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
    """
    Collapse raw device-mode strings into a small controlled vocabulary.

    Context:
    the raw exports contain messy spellings and placeholder values, but the
    model side needs stable category IDs and cardinalities.
    """
    # These normalization helpers intentionally collapse messy raw spellings into
    # a small controlled vocabulary. That keeps category IDs stable and reduces
    # the chance that the model learns spurious distinctions from export noise.
    if pd.isna(cast(Any, value)):
        return "regular"

    normalized = str(value).strip().lower()
    if normalized in {"", "0", "0.0", "none", "regular"}:
        return "regular"
    if normalized == "sleepsleep":
        return "sleep"
    if normalized in {"sleep", "exercise"}:
        return normalized
    return "other"


def normalize_bolus_type(value: object) -> str:
    """
    Collapse raw bolus-type strings into the canonical modeling vocabulary.

    Context:
    this mirrors `normalize_device_mode(...)`, but for the more varied bolus
    event labels found in the AZT1D exports.
    """
    if pd.isna(cast(Any, value)):
        return "none"

    normalized = str(value).strip().lower()
    if normalized in {"", "0", "0.0", "none"}:
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


def _deduplicate_exact_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows before any timestamp-level aggregation.

    Context:
    this behavior is based on inspection of the extracted AZT1D files used in
    this repo, not on a claim from the paper itself. Some subject CSVs contain
    repeated rows that are identical across every column. Dropping those copies
    first prevents later timestamp-collision logic from accidentally treating a
    pure file-level duplication artifact as multiple true events.
    """
    return dataframe.drop_duplicates(ignore_index=True)


def _collapse_duplicate_timestamps(
    dataframe: pd.DataFrame,
    feature_groups: FeatureGroups,
) -> pd.DataFrame:
    """
    Collapse same-subject/same-timestamp collisions into one representative row.

    Context:
    after exact-duplicate removal, the remaining same-subject/same-timestamp
    collisions observed in the local AZT1D files are mostly conflicting glucose
    readings or overlapping basal values at the same minute. We reduce those to
    one row so the later indexing logic can safely assume one observation per
    subject and timestamp.
    """
    group_columns = [feature_groups.subject_id_column, feature_groups.time_column]
    if not dataframe.duplicated(group_columns).any():
        return dataframe

    reduced_rows: list[dict[str, Any]] = []
    for _, group in dataframe.groupby(group_columns, sort=False, dropna=False):
        row = group.iloc[-1].to_dict()

        if feature_groups.target_column in group.columns:
            row[feature_groups.target_column] = _median_non_null(group[feature_groups.target_column])

        if "basal_insulin_u" in group.columns:
            row["basal_insulin_u"] = _most_common_non_null(group["basal_insulin_u"])

        for column in (
            "bolus_insulin_u",
            "correction_insulin_u",
            "meal_insulin_u",
            "carbs_g",
            "device_mode",
            "bolus_type",
            "source_file",
        ):
            if column in group.columns:
                row[column] = _last_non_null(group[column])

        reduced_rows.append(row)

    return pd.DataFrame(reduced_rows, columns=dataframe.columns)


def _add_time_features(dataframe: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """
    Add cyclical and calendar-derived known-ahead time features.

    Context:
    these features let the model see time-of-day and day-of-week structure
    without suffering the discontinuities of raw integer clock values.
    """
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
    """
    Normalize categorical columns according to declared or discovered vocabularies.

    Context:
    this keeps category IDs stable before the DataModule fits vocabularies and
    prevents messy raw spellings from leaking into embedding cardinalities.
    """
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
    """
    Verify that the cleaned dataframe satisfies the declared feature contract.

    Context:
    surfacing schema drift during setup is much easier to debug than allowing
    the mismatch to appear later inside tensor assembly or model forward passes.
    """
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
    """
    Normalize a text-like series into stripped string values with empty-fill.

    Context:
    the category-normalization helpers expect one consistent textual
    representation before they apply controlled-vocabulary cleanup.
    """
    return series.fillna("").astype(str).str.strip()


# ============================================================
# Duplicate-timestamp reduction helpers
# ============================================================
# Purpose:
#   Reduce one same-subject/same-timestamp collision group into
#   a single cleaned scalar per column.
#
# Why these helpers live here instead of in `data.statistics`:
#   They are part of the normalization policy used to build the
#   cleaned modeling dataframe. They do summarize a small group
#   of values, but they are not user-facing descriptive stats.
# ============================================================

def _median_non_null(series: pd.Series) -> float | None:
    """
    Return the median of a numeric series after dropping missing values.

    Context:
    same-minute glucose collisions in AZT1D can contain differing readings. The
    median is a robust way to keep one representative value without always
    trusting the first or last raw row.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.median())


def _most_common_non_null(series: pd.Series) -> float | None:
    """
    Return the most common numeric value in a series, preserving source-order ties.

    Context:
    overlapping basal values at one timestamp are usually duplicates of one
    prevailing rate with occasional alternatives. Choosing the modal value is a
    better fit for that cleanup step than averaging two competing basal rates.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None

    counts = numeric.value_counts(sort=False)
    max_count = counts.max()
    candidates = set(counts[counts == max_count].index.tolist())
    for value in numeric:
        if value in candidates:
            return float(value)
    return float(numeric.iloc[-1])


def _last_non_null(series: pd.Series) -> object:
    """
    Return the last non-missing scalar from a series, or `None` if none exist.

    Context:
    event/categorical columns in a duplicate-timestamp group are usually either
    identical or sparsely populated. Taking the last non-null value preserves a
    concrete observed label without inventing a new merged category.
    """
    values = series.dropna()
    if values.empty:
        return None
    return values.iloc[-1]


def _sin_from_period(values: pd.Series, period: float) -> pd.Series:
    """
    Encode one periodic scalar signal with a sine transform.

    Context:
    paired with the cosine transform, this gives the model a smooth cyclical
    representation of repeating time features.
    """
    angles = values.astype("float64") * (2.0 * math.pi / period)
    return angles.apply(math.sin).astype("float32")


def _cos_from_period(values: pd.Series, period: float) -> pd.Series:
    """
    Encode one periodic scalar signal with a cosine transform.

    Context:
    this complements `_sin_from_period(...)` so cyclical features can represent
    wrap-around positions without discontinuities.
    """
    angles = values.astype("float64") * (2.0 * math.pi / period)
    return angles.apply(math.cos).astype("float32")
