# AI-assisted implementation note:
# This file was drafted with AI assistance and then reviewed/adapted for this
# project. The refactor draws on the earlier AZT1D pipeline in this repo, prior
# work by SlickMik (https://github.com/SlickMik), the PyTorch Lightning
# DataModule docs/tutorial
# (https://lightning.ai/docs/pytorch/stable/data/datamodule.html), and the
# original AZT1D dataset release on Mendeley Data
# (https://data.mendeley.com/datasets/gk9m674wcx/1). Its purpose is to move
# sample assembly into a dedicated Dataset layer aligned with the fused TCN +
# TFT batch contract.

from __future__ import annotations

from typing import Any, Sequence, TypedDict, TypeAlias

import numpy as _np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.indexing import SampleIndexEntry
from data.schema import FeatureGroups


np: Any = _np
NDArray: TypeAlias = Any


# ============================================================
# Sequence dataset
# ============================================================
# Purpose:
#   Assemble one model-ready sample from a cleaned dataframe and
#   a precomputed sequence index.
# ============================================================

class BatchMetadata(TypedDict):
    """
    Human-readable sequence boundary metadata attached to one sample.

    Context:
    this payload is carried alongside tensors so debugging, report generation,
    and prediction export can still identify which subject and which time range
    produced a given window.
    """
    subject_id: str
    encoder_start: str
    encoder_end: str
    decoder_start: str
    decoder_end: str


class BatchItem(TypedDict):
    """
    Full batch-item contract emitted by `AZT1DSequenceDataset`.

    Context:
    the keys here are semantic model-input groups, not raw dataframe columns.
    Keeping that contract explicit makes the dataset, model, and observability
    layers easier to align and debug.
    """
    static_categorical: Tensor
    static_continuous: Tensor
    encoder_continuous: Tensor
    encoder_categorical: Tensor
    decoder_known_continuous: Tensor
    decoder_known_categorical: Tensor
    target: Tensor
    metadata: BatchMetadata


class AZT1DSequenceDataset(Dataset[BatchItem]):
    """
    Dataset that turns one indexed sequence slice into one training sample.

    This class is intentionally narrow:
    - it receives a cleaned dataframe
    - it receives a prebuilt sample index
    - it assembles exactly one sample in `__getitem__`

    That separation is the heart of the refactor. The dataset no longer decides
    how the corpus is split or how DataLoaders are created. It only knows how to
    answer the question "given this index entry, what tensors belong to this
    sample?"
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sample_index: Sequence[SampleIndexEntry],
        feature_groups: FeatureGroups,
        category_maps: dict[str, tuple[str, ...]],
    ) -> None:
        """
        Cache the cleaned dataframe, sample index, and pre-encoded feature arrays.

        Context:
        dataset construction does the repeated dataframe-to-array conversion once
        so per-sample retrieval can stay focused on slicing and assembling the
        semantic batch contract.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.sample_index = list(sample_index)
        self.feature_groups = feature_groups
        self.category_maps = category_maps

        # We pre-encode columns once at dataset construction time so __getitem__
        # stays focused on slicing/assembly rather than repeated dataframe work.
        # This still respects dataset responsibilities because the encoding is
        # sample-serving state, not corpus-level orchestration.
        self.continuous_arrays = {
            column: self.dataframe[column].to_numpy(dtype="float32", copy=True)
            for column in feature_groups.continuous_columns
            if column in self.dataframe.columns
        }
        self.categorical_arrays = {
            column: self._encode_categorical_column(column)
            for column in feature_groups.categorical_columns
            if column in self.dataframe.columns
        }
        self.timestamps = (
            self.dataframe[feature_groups.time_column]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )

    def __len__(self) -> int:
        """Return the number of precomputed sequence windows available to the Dataset."""
        return len(self.sample_index)

    def __getitem__(self, index: int) -> BatchItem:
        """
        Assemble one model-ready sample from the indexed sequence boundaries.

        Context:
        this is where cleaned dataframe rows are translated into the exact batch
        groups consumed by the fused TCN + TFT model stack.
        """
        sample = self.sample_index[index]
        encoder_slice = slice(sample.encoder_start, sample.encoder_end)
        decoder_slice = slice(sample.decoder_start, sample.decoder_end)

        # This dictionary is the explicit batch contract for the refactored
        # pipeline. Each entry corresponds to a semantic model input group rather
        # than to an arbitrary set of raw dataframe columns.
        return {
            "static_categorical": torch.tensor(
                self._row_categorical(sample.encoder_start, self.feature_groups.static_categorical),
                dtype=torch.long,
            ),
            "static_continuous": torch.tensor(
                self._row_continuous(sample.encoder_start, self.feature_groups.static_continuous),
                dtype=torch.float32,
            ),
            "encoder_continuous": torch.tensor(
                self._slice_continuous(encoder_slice, self.feature_groups.encoder_continuous),
                dtype=torch.float32,
            ),
            "encoder_categorical": torch.tensor(
                self._slice_categorical(encoder_slice, self.feature_groups.encoder_categorical),
                dtype=torch.long,
            ),
            "decoder_known_continuous": torch.tensor(
                self._slice_continuous(decoder_slice, self.feature_groups.decoder_known_continuous),
                dtype=torch.float32,
            ),
            "decoder_known_categorical": torch.tensor(
                self._slice_categorical(decoder_slice, self.feature_groups.decoder_known_categorical),
                dtype=torch.long,
            ),
            "target": torch.tensor(
                self.continuous_arrays[self.feature_groups.target_column][decoder_slice],
                dtype=torch.float32,
            ),
            "metadata": {
                # Metadata is intentionally kept as human-readable tracing
                # information instead of model input. It is useful for debugging,
                # inspection, and later evaluation/reporting without affecting
                # the learned computation graph.
                "subject_id": sample.subject_id,
                "encoder_start": self.timestamps[sample.encoder_start],
                "encoder_end": self.timestamps[sample.encoder_end - 1],
                "decoder_start": self.timestamps[sample.decoder_start],
                "decoder_end": self.timestamps[sample.decoder_end - 1],
            },
        }

    def _row_continuous(self, row_index: int, columns: Sequence[str]) -> NDArray:
        """Read one static continuous feature vector anchored at a single row."""
        # Static continuous features are read from a single anchor row because
        # they should not vary across timesteps within one sequence.
        if not columns:
            # Empty feature groups are represented as length-0 trailing
            # dimensions instead of `None` so batching stays uniform and model
            # code can branch on tensor shape rather than on missing keys.
            return np.empty((0,), dtype=np.float32)
        return np.asarray(
            [self.continuous_arrays[column][row_index] for column in columns],
            dtype=np.float32,
        )

    def _row_categorical(self, row_index: int, columns: Sequence[str]) -> NDArray:
        """Read one static categorical feature vector anchored at a single row."""
        # Static categorical features follow the same rule: one row anchors the
        # identity of the whole sequence.
        if not columns:
            return np.empty((0,), dtype=np.int64)
        return np.asarray(
            [self.categorical_arrays[column][row_index] for column in columns],
            dtype=np.int64,
        )

    def _slice_continuous(self, row_slice: slice, columns: Sequence[str]) -> NDArray:
        """Read one temporal continuous tensor in `[time, feature]` layout."""
        # Temporal continuous features are returned as [time, feature] so that
        # default DataLoader collation naturally yields [batch, time, feature].
        if not columns:
            length = row_slice.stop - row_slice.start
            return np.empty((length, 0), dtype=np.float32)

        stacked = [self.continuous_arrays[column][row_slice] for column in columns]
        return np.stack(stacked, axis=-1).astype(np.float32, copy=False)

    def _slice_categorical(self, row_slice: slice, columns: Sequence[str]) -> NDArray:
        """Read one temporal categorical tensor in `[time, feature]` integer-ID layout."""
        # Temporal categorical features mirror the continuous layout, but remain
        # integer encoded for later embedding layers.
        if not columns:
            length = row_slice.stop - row_slice.start
            return np.empty((length, 0), dtype=np.int64)

        stacked = [self.categorical_arrays[column][row_slice] for column in columns]
        return np.stack(stacked, axis=-1).astype(np.int64, copy=False)

    def _encode_categorical_column(self, column: str) -> NDArray:
        """
        Convert one categorical dataframe column into stable integer IDs.

        Context:
        the Dataset applies the DataModule-fitted vocabulary so train/val/test
        splits and the model-side embedding cardinalities stay aligned.
        """
        # The Dataset applies a frozen category map fitted by the DataModule.
        # This keeps train/val/test integer IDs perfectly aligned.
        categories = self.category_maps.get(column)
        if not categories:
            raise ValueError(f"Missing category map for categorical column '{column}'")

        category_to_index = {category: index for index, category in enumerate(categories)}
        values = self.dataframe[column].fillna("").astype(str).str.strip()

        # Unknown categories fall back to 0. In practice they should not appear
        # because the DataModule fits vocabularies from the same cleaned frame,
        # but the fallback keeps dataset serving robust if a new label slips in.
        return values.map(lambda value: category_to_index.get(value, 0)).to_numpy(dtype="int64")
