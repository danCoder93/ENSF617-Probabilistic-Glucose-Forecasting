from pytorch_lightning import LightningModule

from tft import TemporalFusionTransformer
from grn import GRN
from nn_head import NNHead

class FusedModel(LightningModule):
  '''
  Fused model is a TCN-TFT hybrid that combines Temporal Convolution Network (TCN) for short term temporal patterns and Temporal Fusion Transformer (TFT) for observing long term temporal patterns
  '''
  def __init__(self, config):
    super(FusedModel).__init__()

    # backbones
    self.tcn = None
    self.tft = TemporalFusionTransformer(config)
    self.fcn = NNHead(config)

  def forward(self):
    pass

