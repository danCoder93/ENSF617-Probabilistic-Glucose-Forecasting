from __future__ import annotations

# AI-assisted maintenance note (April 1, 2026):
# This existing module was refined with AI assistance under direct user
# guidance. The head remains the final readout stage for fused horizon-wise
# features, but it was strengthened from a tiny MLP into a deeper residual
# position-wise predictor so the fused model has more expressive final-stage
# predictive capacity without moving fusion responsibility out of the upstream
# GRN.
#
# Context:
# this file intentionally stays narrow. It does not own temporal reasoning,
# multimodal fusion, or probabilistic loss computation. Its job is only to turn
# one already-fused hidden vector per horizon step into final forecast
# channels.

import torch.nn as nn
from torch import Tensor


class ResidualMLPBlock(nn.Module):
    """
    Position-wise residual feed-forward block.

    This gives the final head more capacity without changing the sequence
    layout: every horizon step is processed independently, but more deeply than
    with a single hidden linear layer.

    Architectural role:
    - the upstream `GRN` has already fused the branch representations into one
      shared hidden state per horizon step
    - the final head still benefits from a small amount of nonlinear
      refinement before producing quantiles
    - a residual connection lets the block learn "adjustments" to the fused
      hidden state instead of forcing each block to relearn the full signal

    This is conceptually similar to the position-wise feed-forward layers used
    in transformer-style models: it increases per-position modeling power while
    keeping temporal alignment unchanged.
    """

    def __init__(self, hidden_size: int, feedforward_size: int, dropout: float) -> None:
        super().__init__()
        # The inner width is deliberately larger than the external hidden size.
        # This lets the block briefly expand the representation, apply a
        # nonlinear transformation, and compress back to the same hidden width
        # so the residual addition remains shape-compatible.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size),
            nn.Dropout(dropout),
        )
        # LayerNorm is applied after the residual addition so the output scale
        # stays well-behaved even when several residual blocks are stacked.
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        # Input / output shape:
        #   [batch, prediction_length, hidden_size]
        #
        # The block is position-wise: it transforms each horizon step's feature
        # vector independently, while preserving batch and horizon dimensions.
        return self.norm(x + self.ffn(x))


class NNHead(nn.Module):
    """
    Strengthened readout head for `[batch, horizon, hidden]` tensors.

    The same network is applied independently at each forecast horizon
    position. PyTorch's `nn.Linear` naturally supports this because it operates
    on the last dimension and preserves the leading dimensions.

    Current design:
    - project inputs into a readout hidden space
    - refine that hidden state with residual feed-forward blocks
    - map the refined representation to the final forecast channels

    This gives the fused model a stronger last-mile predictor while keeping the
    actual branch fusion responsibility in the upstream GRN.

    Important architectural boundary:
    - this head is not where TCN and TFT are fused; that happens earlier
    - this head assumes fusion has already happened and focuses only on
      converting one fused hidden vector per horizon step into final outputs

    In other words:
    - `FusedModel` decides what information should meet
    - `GRN` decides how to mix that information into one hidden representation
    - `NNHead` decides how to turn that hidden representation into the final
      glucose forecast channels
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        hidden_size: int | None = None,
        feedforward_size: int | None = None,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

        # Default sizing logic:
        # - if the caller does not provide a readout hidden size, keep at least
        #   the incoming feature width and avoid collapsing too aggressively
        # - keep the feed-forward width wider still so each residual block has
        #   room for a richer nonlinear transformation
        hidden_size = hidden_size or max(input_size, output_size * 8)
        feedforward_size = feedforward_size or hidden_size * 2
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if feedforward_size <= 0:
            raise ValueError("feedforward_size must be > 0")

        # The input projection lets the head widen the representation before
        # the residual refinement stack. A normalization layer immediately
        # after projection helps keep the deeper readout stable.
        #
        # Shape change:
        #   [batch, horizon, input_size] -> [batch, horizon, hidden_size]
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        # Stacking multiple residual blocks increases readout capacity while
        # keeping the external hidden shape fixed from block to block.
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(hidden_size, feedforward_size, dropout)
                for _ in range(num_blocks)
            ]
        )
        # The final projector converts the refined hidden representation into
        # the desired output channels. In this project that is typically one
        # value per requested quantile at each horizon step.
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input:
        #   [batch, prediction_length, input_size]
        # Output:
        #   [batch, prediction_length, output_size]
        #
        # Algorithmically this is:
        # 1. lift the fused feature vector into the readout hidden space
        # 2. refine that representation with residual MLP blocks
        # 3. project the refined state to the final forecast channels
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)
