# AI-assisted implementation note:
# This file was drafted with AI assistance and then reviewed/adapted for this
# project. It provides descriptive-statistics utilities for the refactored
# AZT1D data pipeline without coupling that reporting logic to training code.

from __future__ import annotations

from typing import Any

import pandas as pd

from config import DataConfig
from data.indexing import build_sequence_index, split_processed_frame
from data.schema import FeatureGroups
from data.transforms import load_processed_frame


# ============================================================
# Descriptive-statistics helpers
# ============================================================
# Purpose:
#   Summarize the cleaned dataframe and split/window layout in a
#   JSON-ready structure that callers can log, inspect, or write
#   to disk.
# ============================================================

def describe_processed_data(
    processed_csv_path: str,
    config: DataConfig,
    feature_groups: FeatureGroups,
) -> dict[str, Any]:
    """
    Load the canonical processed CSV and return descriptive statistics for it.

    Context:
    callers that only have a processed file path should not need to manually
    repeat the same load-and-summarize boilerplate that the DataModule already
    uses internally.
    """
    dataframe = load_processed_frame(processed_csv_path, config, feature_groups)
    return describe_clean_frame(dataframe, config, feature_groups)


def describe_clean_frame(
    dataframe: pd.DataFrame,
    config: DataConfig,
    feature_groups: FeatureGroups,
) -> dict[str, Any]:
    """
    Return descriptive statistics for one cleaned dataframe.

    Context:
    the transform layer owns dataframe normalization, while this helper owns the
    reporting-oriented question "what does this cleaned dataset look like?".
    Keeping those responsibilities separate makes the summary logic reusable for
    notebooks, tests, and future artifact-writing code.
    """
    split_frames = split_processed_frame(dataframe, config, feature_groups)
    train_index = build_sequence_index(split_frames["train"], config, feature_groups)
    val_index = build_sequence_index(split_frames["val"], config, feature_groups)
    test_index = build_sequence_index(split_frames["test"], config, feature_groups)

    subject_counts = dataframe.groupby(config.subject_id_column).size()
    duplicate_timestamp_rows = int(
        dataframe.duplicated([config.subject_id_column, config.time_column]).sum()
    )

    return {
        "row_count": int(len(dataframe)),
        "subject_count": int(dataframe[config.subject_id_column].nunique()) if not dataframe.empty else 0,
        "timestamp_start": _optional_timestamp(dataframe[config.time_column].min())
        if config.time_column in dataframe.columns
        else None,
        "timestamp_end": _optional_timestamp(dataframe[config.time_column].max())
        if config.time_column in dataframe.columns
        else None,
        "duplicate_subject_timestamp_rows": duplicate_timestamp_rows,
        "missing_values_by_column": {
            str(column): int(dataframe[column].isna().sum())
            for column in dataframe.columns
        },
        "rows_per_subject": {
            "min": int(subject_counts.min()) if not subject_counts.empty else 0,
            "median": float(subject_counts.median()) if not subject_counts.empty else 0.0,
            "max": int(subject_counts.max()) if not subject_counts.empty else 0,
        },
        "subject_row_counts": {
            str(subject_id): int(count)
            for subject_id, count in subject_counts.sort_index().items()
        },
        "continuous_columns": {
            column: _describe_numeric_series(dataframe[column])
            for column in feature_groups.continuous_columns
            if column in dataframe.columns
        },
        "categorical_columns": {
            column: _describe_categorical_series(dataframe[column])
            for column in feature_groups.categorical_columns
            if column in dataframe.columns
        },
        "splits": {
            "train_rows": int(len(split_frames["train"])),
            "val_rows": int(len(split_frames["val"])),
            "test_rows": int(len(split_frames["test"])),
            "train_windows": int(len(train_index)),
            "val_windows": int(len(val_index)),
            "test_windows": int(len(test_index)),
            "train_subjects": int(split_frames["train"][config.subject_id_column].nunique())
            if not split_frames["train"].empty
            else 0,
            "val_subjects": int(split_frames["val"][config.subject_id_column].nunique())
            if not split_frames["val"].empty
            else 0,
            "test_subjects": int(split_frames["test"][config.subject_id_column].nunique())
            if not split_frames["test"].empty
            else 0,
        },
        "config": {
            "sampling_interval_minutes": int(config.sampling_interval_minutes),
            "encoder_length": int(config.encoder_length),
            "prediction_length": int(config.prediction_length),
            "window_stride": int(config.window_stride),
        },
    }


def _describe_numeric_series(series: pd.Series) -> dict[str, Any]:
    """Return compact descriptive statistics for one numeric dataframe column."""
    numeric = pd.to_numeric(series, errors="coerce")
    quantiles = numeric.quantile([0.25, 0.5, 0.75])

    return {
        "count": int(numeric.notna().sum()),
        "missing_count": int(numeric.isna().sum()),
        "zero_count": int((numeric.fillna(1.0) == 0.0).sum()),
        "min": _optional_float(numeric.min()),
        "max": _optional_float(numeric.max()),
        "mean": _optional_float(numeric.mean()),
        "std": _optional_float(numeric.std(ddof=0)),
        "q25": _optional_float(quantiles.loc[0.25]),
        "q50": _optional_float(quantiles.loc[0.5]),
        "q75": _optional_float(quantiles.loc[0.75]),
    }


def _describe_categorical_series(series: pd.Series) -> dict[str, Any]:
    """Return compact descriptive statistics for one categorical/text column."""
    values = series.fillna("").astype(str).str.strip()
    non_empty_values = values[values != ""]

    return {
        "count": int(len(values)),
        "missing_count": int((values == "").sum()),
        "distinct_count": int(non_empty_values.nunique()),
        "value_counts": {
            str(value): int(count)
            for value, count in non_empty_values.value_counts(sort=False).sort_index().items()
        },
    }


def _optional_float(value: object) -> float | None:
    """Convert a scalar to float unless it is missing/NaN."""
    if pd.isna(value):
        return None
    return float(value)


def _optional_timestamp(value: object) -> str | None:
    """Convert a timestamp-like scalar to ISO format unless it is missing."""
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()
