import torch.nn as nn
from torch.nn import Module
from torch.nn.functional import relu

from torch import Tensor

class NNHead(Module):
  '''
  Fully connected Neural Network (FCN) head
  '''

  def __init__(self, config):
    super(NNHead).__init__()

    # Layer 1
    self.ln_1 = nn.LazyLinear(64)
    # Layer 2
    self.ln_2 = nn.Linear(64, 1)
  
  def forward(self, x: Tensor):
    x = relu(self.ln_1(x))
    x = relu(self.ln_2(x))
    return x

