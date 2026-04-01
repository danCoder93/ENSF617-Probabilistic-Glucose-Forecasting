"""
AI-assisted implementation note:
This file was drafted with AI assistance and then reviewed/adapted for this
project. The refactor draws on the earlier AZT1D pipeline in this repo, prior
work by SlickMik (https://github.com/SlickMik), the PyTorch Lightning
DataModule docs/tutorial
(https://lightning.ai/docs/pytorch/stable/data/datamodule.html), and the
original AZT1D dataset release on Mendeley Data
(https://data.mendeley.com/datasets/gk9m674wcx/1). Its purpose is to separate
split policy and sequence indexing from dataset tensor assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.schema import FeatureGroups
from utils.config import DataConfig


# ============================================================
# Sample index contract
# ============================================================
# Purpose:
#   Represent valid encoder/decoder slice boundaries without
#   materializing tensors yet.
# ============================================================
@dataclass(frozen=True)
class SampleIndexEntry:
    """
    Slice boundaries for one model sample.

    We index with integer row positions rather than storing tensors directly so:
    - the index stays lightweight
    - dataset splits can be tested independently of tensor code
    - the Dataset remains the single owner of item assembly in `__getitem__`
    """

    subject_id: str
    encoder_start: int
    encoder_end: int
    decoder_start: int
    decoder_end: int


def split_processed_frame(
    dataframe: pd.DataFrame,
    config: DataConfig,
    feature_groups: FeatureGroups,
) -> dict[str, pd.DataFrame]:
    # Split policy belongs to orchestration, not to the Dataset. Keeping it
    # here lets us swap policies without changing sample assembly code.
    if config.split_by_subject:
        return _split_by_subject(
            dataframe,
            config,
            feature_groups.subject_id_column,
            feature_groups.time_column,
        )
    if config.split_within_subject:
        return _split_within_subject(
            dataframe,
            config,
            feature_groups.subject_id_column,
            feature_groups.time_column,
        )
    return _split_globally(dataframe, config, feature_groups.subject_id_column, feature_groups.time_column)


def build_sequence_index(
    dataframe: pd.DataFrame,
    config: DataConfig,
    feature_groups: FeatureGroups,
) -> list[SampleIndexEntry]:
    """
    Convert a cleaned dataframe into a list of valid sequence slices.

    This layer owns continuity and boundary logic. It deliberately does not
    construct tensors because "is this a legal window?" and "how do I turn that
    window into tensors?" are separate concerns with different failure modes.
    """

    if dataframe.empty:
        return []

    sample_index: list[SampleIndexEntry] = []
    # `sort=False` preserves the split-frame subject ordering chosen upstream.
    # We only sort rows within each subject by time, not the subjects
    # themselves, so split behavior stays deterministic and easier to reason
    # about.
    for subject_id, subject_frame in dataframe.groupby(feature_groups.subject_id_column, sort=False):
        # Keep the original split-frame row positions so the resulting index can
        # be used directly by the Dataset. Resetting to 0..N here would make
        # every subject window point at the wrong rows once multiple subjects are
        # concatenated into a split dataframe.
        subject_frame = subject_frame.sort_values(feature_groups.time_column).reset_index()
        absolute_row_positions = subject_frame["index"].to_numpy(dtype="int64", copy=True)

        for segment_start, segment_end in _find_contiguous_segments(
            subject_frame,
            feature_groups.time_column,
            config.sampling_interval_minutes,
        ):
            # A legal sample needs one fully contiguous block long enough to
            # cover encoder history plus decoder horizon.
            segment_length = segment_end - segment_start
            required_length = config.encoder_length + config.prediction_length

            if segment_length < required_length:
                continue

            for window_start in range(
                segment_start,
                segment_end - required_length + 1,
                config.window_stride,
            ):
                # Store half-open boundaries so the Dataset can reuse them
                # directly with normal Python slicing semantics.
                encoder_start = int(absolute_row_positions[window_start])
                encoder_end = int(absolute_row_positions[window_start + config.encoder_length - 1]) + 1
                decoder_start = int(absolute_row_positions[window_start + config.encoder_length])
                decoder_end = int(absolute_row_positions[window_start + required_length - 1]) + 1
                sample_index.append(
                    SampleIndexEntry(
                        subject_id=str(subject_id),
                        encoder_start=encoder_start,
                        encoder_end=encoder_end,
                        decoder_start=decoder_start,
                        decoder_end=decoder_end,
                    )
                )

    return sample_index


def _split_by_subject(
    dataframe: pd.DataFrame,
    config: DataConfig,
    subject_id_column: str,
    time_column: str,
) -> dict[str, pd.DataFrame]:
    # Strongest leakage barrier: one subject belongs to exactly one split.
    sorted_frame = dataframe.sort_values([subject_id_column, time_column]).reset_index(drop=True)
    subject_ids = sorted(sorted_frame[subject_id_column].astype(str).unique())
    train_ids, val_ids, test_ids = _split_ids(
        subject_ids,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
    )

    return {
        "train": sorted_frame.loc[sorted_frame[subject_id_column].isin(train_ids)].reset_index(drop=True),
        "val": sorted_frame.loc[sorted_frame[subject_id_column].isin(val_ids)].reset_index(drop=True),
        "test": sorted_frame.loc[sorted_frame[subject_id_column].isin(test_ids)].reset_index(drop=True),
    }


def _split_within_subject(
    dataframe: pd.DataFrame,
    config: DataConfig,
    subject_id_column: str,
    time_column: str,
) -> dict[str, pd.DataFrame]:
    # Closest to the original behavior: each subject timeline is split
    # chronologically into train/val/test segments.
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, subject_frame in dataframe.groupby(subject_id_column, sort=False):
        # We reset each subject frame to its own 0..N timeline here because this
        # function's job is only to choose which rows belong to each split. The
        # later indexing step will rebuild absolute split-frame row positions.
        subject_frame = subject_frame.sort_values(time_column).reset_index(drop=True)
        split_bounds = _split_bounds(
            len(subject_frame),
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
        )

        train_parts.append(subject_frame.iloc[:split_bounds.train_end].copy())
        val_parts.append(subject_frame.iloc[split_bounds.train_end:split_bounds.val_end].copy())
        test_parts.append(subject_frame.iloc[split_bounds.val_end:split_bounds.test_end].copy())

    return {
        "train": _concat_split_parts(train_parts),
        "val": _concat_split_parts(val_parts),
        "test": _concat_split_parts(test_parts),
    }


def _split_globally(
    dataframe: pd.DataFrame,
    config: DataConfig,
    subject_id_column: str,
    time_column: str,
) -> dict[str, pd.DataFrame]:
    # Kept as a fallback policy, although subject-aware splitting is usually
    # more appropriate for patient time-series forecasting.
    dataframe = dataframe.sort_values([subject_id_column, time_column]).reset_index(drop=True)
    split_bounds = _split_bounds(
        len(dataframe),
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
    )

    return {
        "train": dataframe.iloc[:split_bounds.train_end].reset_index(drop=True),
        "val": dataframe.iloc[split_bounds.train_end:split_bounds.val_end].reset_index(drop=True),
        "test": dataframe.iloc[split_bounds.val_end:split_bounds.test_end].reset_index(drop=True),
    }


def _find_contiguous_segments(
    subject_frame: pd.DataFrame,
    time_column: str,
    sampling_interval_minutes: int,
) -> list[tuple[int, int]]:
    # The forecasting pipeline assumes evenly sampled timesteps, so any time gap
    # breaks one subject trajectory into separate valid segments.
    timestamps = subject_frame[time_column]
    deltas = (
        timestamps.diff()
        .dt.total_seconds()
        .div(60)
        .fillna(sampling_interval_minutes)
    )

    segments: list[tuple[int, int]] = []
    segment_start = 0
    for row_index in range(1, len(subject_frame)):
        if deltas.iloc[row_index] != sampling_interval_minutes:
            segments.append((segment_start, row_index))
            segment_start = row_index

    segments.append((segment_start, len(subject_frame)))
    return segments


def _split_ids(
    values: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[set[str], set[str], set[str]]:
    # Integer truncation is deliberate here because split boundaries must land on
    # whole subjects. We still compute test allocation explicitly from
    # `test_ratio` rather than treating it as an implicit remainder, so the code
    # matches the DataConfig contract more directly.
    split_bounds = _split_bounds(len(values), train_ratio, val_ratio, test_ratio)
    return (
        set(values[:split_bounds.train_end]),
        set(values[split_bounds.train_end:split_bounds.val_end]),
        set(values[split_bounds.val_end:split_bounds.test_end]),
    )


@dataclass(frozen=True)
class SplitBounds:
    train_end: int
    val_end: int
    test_end: int


def _split_bounds(
    total_count: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> SplitBounds:
    # We compute each split size from its own declared ratio so `test_ratio` has
    # explicit effect rather than only acting as "whatever rows are left over".
    # Because integer truncation can leave a few unassigned rows, any remainder
    # is attached to the test split so we still cover the full dataset.
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = int(total_count * test_ratio)

    assigned_count = train_count + val_count + test_count
    if assigned_count < total_count:
        test_count += total_count - assigned_count

    train_end = train_count
    val_end = train_end + val_count
    test_end = min(total_count, val_end + test_count)

    return SplitBounds(
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
    )


def _concat_split_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        # Returning an empty dataframe keeps downstream code uniform: the
        # DataModule can still attempt index construction and then decide whether
        # to apply fallback behavior, rather than branching on `None`.
        return pd.DataFrame()
    return pd.concat(parts, axis=0, ignore_index=True)
