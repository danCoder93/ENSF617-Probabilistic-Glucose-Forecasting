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

import enum

from collections import namedtuple, OrderedDict

import numpy as np

NP_FLOAT32 = getattr(np, "float32", float)
NP_INT64 = getattr(np, "int64", int)

class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""
    CONTINUOUS = 0
    CATEGORICAL = 1
    DATE = 2
    STR = 3

class InputTypes(enum.IntEnum):
    """Defines input types of each column."""
    TARGET = 0
    OBSERVED = 1
    KNOWN = 2
    STATIC = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index

FeatureSpec = namedtuple('FeatureSpec', ['name', 'feature_type', 'feature_embed_type'])
DTYPE_MAP = {
        DataTypes.CONTINUOUS : NP_FLOAT32,
        DataTypes.CATEGORICAL : NP_INT64,
        DataTypes.DATE:'datetime64[ns]',
        DataTypes.STR: str
        }

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

FEAT_NAMES = ['s_cat' , 's_cont' , 'k_cat' , 'k_cont' , 'o_cat' , 'o_cont' , 'target', 'id']
DEFAULT_ID_COL = 'id'
