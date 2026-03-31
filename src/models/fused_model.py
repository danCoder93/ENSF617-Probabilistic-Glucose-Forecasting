from dataclasses import replace

from pytorch_lightning import LightningModule

import torch
from torch import Tensor
from torch.nn import ModuleList

from tcn import TCN
from tft import TemporalFusionTransformer
from grn import GRN
from nn_head import NNHead

from utils.config import Config

class FusedModel(LightningModule):
  '''
  Fused model is a TCN-TFT hybrid that combines Temporal Convolution Network (TCN) for short term temporal patterns and Temporal Fusion Transformer (TFT) for observing long term temporal patterns
  '''
  def __init__(self, config: Config):
    super(FusedModel).__init__()

    self.config = config

    # backbones
    # TCN kernal - 3
    self.tcn3 = TCN(config.tcn)

    # TCN kernal - 5
    config_tcn_5 = replace(config.tcn, kernel_size=5)
    self.tcn5 = TCN(config_tcn_5)

    # TCN kernal - 7
    config_tcn_7 = replace(config.tcn, kernal_size=7)
    self.tcn7 = TCN(config_tcn_7)

    # TFT
    self.tft = TemporalFusionTransformer(config.tft)

    # GRN
    # Fused outputs from TCNs + TFT
    self.grn = GRN(config.tft.hidden_size * 4, config.tft.hidden_size, dropout=config.tft.dropout)

    # NN head
    self.fcn = NNHead(config)

  def forward(self, x: Tensor):
    # TODO: figure out the static cat, static cont, obs cat, obs cont, future cat, future cont

    # TODO: pass only observed past cat, cont to the TCN
    x_tcn_3 = self.tcn3(x)
    x_tcn_5 = self.tcn5(x)
    x_tcn_7 = self.tcn7(x)

    # pass x to the TFT
    x_tft = self.tft(x)

    # combine the output of tft + tcns
    x_final = torch.cat([x_tft, x_tcn_3, x_tcn_5, x_tcn_7])

    # pass it to grn
    x = self.grn(x_final)

    # pass it to the fcn
    x = self.fcn(x)

    return x




