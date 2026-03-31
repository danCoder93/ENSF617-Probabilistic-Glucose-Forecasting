from dataclasses import dataclass

from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from utils.tft_utils import FeatureSpec, InputTypes, DataTypes

@dataclass
class TCNConfig:
  num_inputs: int
  num_channels: Sequence[int] | NDArray[np.int_]
  kernel_size: int = 3
  dilations: Optional[ Sequence[int] | NDArray[np.int_] ] = [1, 2, 4]
  dilation_reset: Optional[ int ] = None
  dropout: float = 0.1
  causal: bool = True
  use_norm: str = 'weight_norm'
  activation: str = 'relu'
  kernel_initializer: str = 'xavier_uniform'
  use_skip_connections: bool = True
  input_shape: str = 'NCL'
  embedding_shapes: Optional[ Sequence[int] | NDArray[np.int_] ] = None
  embedding_mode: str = 'add'
  use_gate: bool = False
  lookahead: int = 0
  output_projection: Optional[ int ] = None
  output_activation: Optional[ str ] = None 

@dataclass
class TFTConfig:
  features: Sequence[FeatureSpec]
  # Feature sizes
  static_categorical_inp_lens = [369]
  temporal_known_categorical_inp_lens = []
  temporal_observed_categorical_inp_lens = []
  quantiles = [0.9]
  
  encoder_length = 7 * 24

  n_head = 4
  hidden_size = 128
  dropout = 0.1
  attn_dropout = 0.0

  #### Derived variables ####
  temporal_known_continuous_inp_size: int = 0
  temporal_observed_continuous_inp_size: int = 0
  temporal_target_size: int = 0
  static_continuous_inp_size: int = 0
  num_static_vars: int = 0
  num_future_vars: int = 0
  num_historic_vars: int = 0

  def __post_init__(self):
    self.temporal_known_continuous_inp_size = len([x for x in self.features 
        if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
    self.temporal_observed_continuous_inp_size = len([x for x in self.features 
        if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
    self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
    self.static_continuous_inp_size = len([x for x in self.features 
        if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

    self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
    self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
    self.num_historic_vars = sum([self.num_future_vars,
                                  self.temporal_observed_continuous_inp_size,
                                  self.temporal_target_size,
                                  len(self.temporal_observed_categorical_inp_lens),
                                  ])


@dataclass
class Config:
  tft : TFTConfig
  tcn : TCNConfig