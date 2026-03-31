import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.functional import glu, elu

from torch import Tensor
from typing import Optional, cast

from torch.nn import LayerNorm

class MaybeLayerNorm(Module):
    def __init__(self, output_size, hidden_size, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = LayerNorm(output_size if output_size else hidden_size, eps=eps)
    
    def forward(self, x):
        return self.ln(x)

class GLU(Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = glu(x)
        return x

class GRN(Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 context_hidden_size=None,
                 dropout=0.0,):
        super().__init__()
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        else:
            self.lin_c = nn.Identity()
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if self.out_proj is None else self.out_proj(a)
        x = x + y
        return self.layer_norm(x) 