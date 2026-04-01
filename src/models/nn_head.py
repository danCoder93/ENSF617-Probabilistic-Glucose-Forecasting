import torch.nn as nn
from torch import Tensor


class NNHead(nn.Module):
    """Lightweight readout head for horizon-wise fused features."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        hidden_size = max(input_size // 2, output_size)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
