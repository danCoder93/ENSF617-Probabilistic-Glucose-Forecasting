# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This code is adapted from the NVIDIA/DeepLearningExamples repository:
# https://github.com/NVIDIA/DeepLearningExamples
# Modifications by danCoder93 (March 26, 2026).
# Updated to integrate PyTorch Lightning.
#
# AI-assisted maintenance note (April 1, 2026):
# This file remains intentionally small because it serves as the shared TFT
# schema vocabulary for the rest of the repository. The goal here is not to add
# logic, but to keep the meaning of the enum values, feature declarations, and
# ordering constants explicit so the data and model layers interpret the same
# semantic contract.

import enum

from collections import namedtuple, OrderedDict

import numpy as np

NP_FLOAT32 = getattr(np, "float32", float)
NP_INT64 = getattr(np, "int64", int)

class DataTypes(enum.IntEnum):
    """
    Enumerate the storage/embedding type of one declared feature column.

    Context:
    these values describe "what kind of data is this column?" rather than
    "what role does it play in forecasting?" The latter is handled by
    `InputTypes`.
    """
    CONTINUOUS = 0
    CATEGORICAL = 1
    DATE = 2
    STR = 3

class InputTypes(enum.IntEnum):
    """
    Enumerate the modeling role of one declared feature column.

    Context:
    these values answer questions like:
    - is this the target series?
    - is this a known-ahead covariate?
    - is this a static subject-level variable?
    - is this the time index or entity identifier?
    """
    TARGET = 0
    OBSERVED = 1
    KNOWN = 2
    STATIC = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index

# `FeatureSpec` is the compact declarative contract used by the config and data
# layers to describe one feature's name, modeling role, and embedding/storage
# type without introducing a heavier custom class.
FeatureSpec = namedtuple('FeatureSpec', ['name', 'feature_type', 'feature_embed_type'])

# Map the semantic `DataTypes` enum to the concrete NumPy/pandas dtype the data
# pipeline should use when preparing columns for model consumption.
DTYPE_MAP = {
        DataTypes.CONTINUOUS : NP_FLOAT32,
        DataTypes.CATEGORICAL : NP_INT64,
        DataTypes.DATE:'datetime64[ns]',
        DataTypes.STR: str
        }

# Canonical feature ordering shared by the TFT path. This keeps grouped tensors
# and derived counts aligned across the data and model layers.
FEAT_ORDER = [
        (InputTypes.STATIC, DataTypes.CATEGORICAL),
        (InputTypes.STATIC, DataTypes.CONTINUOUS),
        (InputTypes.KNOWN, DataTypes.CATEGORICAL),
        (InputTypes.KNOWN, DataTypes.CONTINUOUS),
        (InputTypes.OBSERVED, DataTypes.CATEGORICAL),
        (InputTypes.OBSERVED, DataTypes.CONTINUOUS),
        (InputTypes.TARGET, DataTypes.CONTINUOUS),
        (InputTypes.ID, DataTypes.CATEGORICAL)
        ]

# Short names used when grouping the ordered feature sets into batch-dictionary
# keys for TFT.
FEAT_NAMES = ['s_cat' , 's_cont' , 'k_cat' , 'k_cont' , 'o_cat' , 'o_cont' , 'target', 'id']

# Historical default identifier column kept for compatibility with older TFT
# utilities and lineage from the upstream implementation.
DEFAULT_ID_COL = 'id'
