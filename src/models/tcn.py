# This file is adapted from the open-source `pytorch-tcn` project by Paul
# Krug:
# https://github.com/paul-krug/pytorch-tcn
#
# Within this repository, the implementation is also informed by prior
# project-specific TCN work on the `MarleyTCNClean` branch:
# https://github.com/danCoder93/ENSF617-Probabilistic-Glucose-Forecasting/blob/MarleyTCNClean/src/models/tcn.py
#
# The current version narrows and adapts that lineage for the fused glucose
# forecasting architecture used in this project. It keeps the core causal
# residual temporal-block structure while removing broader library-style
# surfaces that are not part of the current use case.
#
# AI-assisted maintenance note (April 1, 2026):
# AI assistance was used under direct user guidance to help document
# provenance, explain the project-specific architecture, and clarify the role
# of this TCN branch inside the fused model. Final responsibility for
# integration, editing, and technical intent remains with the project authors.

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.config import TCNConfig


# ============================================================
# Helper: activation factory
# ============================================================
# Purpose:
#   Convert the small set of supported activation names from
#   `TCNConfig` into actual PyTorch modules.
#
# Design note:
#   The refactored TCN intentionally supports a narrow activation
#   surface. Centralizing the mapping keeps the rest of the file
#   focused on temporal logic rather than on string branching.
# ============================================================
def _build_activation(name: str) -> nn.Module:
    activation_map: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activation_map:
        raise ValueError(f"Unsupported TCN activation '{name}'")
    return activation_map[name]()


class ChannelLayerNorm(nn.Module):
    """
    LayerNorm wrapper for channel-first temporal tensors.

    PyTorch's `LayerNorm` expects the normalized dimension to be the trailing
    dimension. Our Conv1d blocks naturally operate on `[batch, channels, time]`,
    so this module temporarily swaps time and channel axes, applies layer norm,
    then restores the original layout.

    Normalization choice:
    - In this project the TCN is only one branch inside a larger fused model.
    - Standardizing on layer norm keeps the branch behavior closer to the TFT
      side of the architecture and avoids dependence on batch statistics.
    - That is a better fit for subject-heterogeneous medical time series than a
      batch-normalization-heavy design.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: Tensor) -> Tensor:
        # Temporarily move channels to the last dimension so standard LayerNorm
        # can normalize feature channels at each timestep independently.
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Module):
    """
    Conv1d with left padding only.

    Forecasting constraint:
    - In forecasting we must prevent each timestep from seeing the future.
    - Left-only padding preserves output length while ensuring the convolution at
      time `t` only depends on timesteps `<= t`.

    Concrete intuition for this project:
    - If we are predicting glucose at the forecast origin, the representation at
      that point may use earlier insulin, meal, and glucose history.
    - It must never accidentally look at later decoder timesteps, because that
      would leak future information and make evaluation unrealistically easy.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        # Effective receptive field grows with both kernel size and dilation.
        # This is the core mechanism that lets the TCN cover short and mid-range
        # glucose dynamics without using recurrence.
        #
        # Example:
        # - larger kernels widen the local window seen by each convolution
        # - larger dilations spread those windows farther apart in time
        # Together they let the branch capture both immediate and more delayed
        # temporal relationships in the encoder history.
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Left padding ensures the convolution output stays length-preserving
        # without letting the filter see timesteps to the right of the current
        # position.
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    Residual TCN block with two causal convolutions.

    Design intent:
    - preserve the standard residual-block flavor of TCNs
    - keep the implementation minimal and readable
    - expose only the pieces that matter to the fused architecture

    Block structure:
    - The first convolution starts to transform the local temporal pattern.
    - The second convolution lets the block refine that representation before it
      is passed onward.
    - The residual connection preserves an easier information path through the
      network, which typically stabilizes optimization and makes deeper stacks
      easier to train than a plain feed-forward conv chain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation_name: str,
    ) -> None:
        super().__init__()
        # First temporal transform:
        # causal conv -> normalization -> nonlinearity -> dropout
        #
        # This stage extracts a first-pass local temporal representation from
        # the current receptive field.
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = ChannelLayerNorm(out_channels)
        self.activation1 = _build_activation(activation_name)
        self.dropout1 = nn.Dropout(dropout)

        # Second temporal transform:
        # mirrors the first so the block can learn a richer local temporal map
        # before merging back with the residual path.
        #
        # Using the same dilation within the block keeps the temporal scale of
        # the block coherent, while the outer TCN stack grows scale across
        # blocks by changing the dilation value from one block to the next.
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = ChannelLayerNorm(out_channels)
        self.activation2 = _build_activation(activation_name)
        self.dropout2 = nn.Dropout(dropout)

        # Residual projection is only needed when channel width changes between
        # blocks. Otherwise the identity path is already shape-compatible.
        self.residual_projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.output_activation = _build_activation(activation_name)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual_projection(x)

        # The main path learns a causal temporal transformation while the
        # residual path preserves information flow and stabilizes optimization.
        #
        # In effect, the block is learning "how should the input history be
        # adjusted?" rather than forcing every layer to relearn the whole signal
        # from scratch.
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)

        return self.output_activation(x + residual)


class TCN(nn.Module):
    """
    Lean project-specific TCN forecaster.

    Architecture role in this repository:
    - consume encoder-side history only
    - act as a base forecaster for one kernel size branch
    - emit horizon-aligned predictions that can be fused with TFT

    What this class does not try to be:
    - It is not a generic sequence-modeling toolkit.
    - It is not a standalone decoder with autoregressive rollout logic.
    - It is not responsible for handling static covariates or future-known
      decoder inputs; those responsibilities sit elsewhere in the fused model.

    Instead, this class focuses on one question:
    - given only past temporal history, what short/mid-range forecast signal can
      this kernel-scale branch contribute to the fused predictor?

    Input:
    - `[batch, encoder_length, num_inputs]`

    Output:
    - `forward_features(...) -> [batch, prediction_length, branch_hidden_size]`
    - `forward(...) -> [batch, prediction_length, output_size]`
    """

    def __init__(self, config: TCNConfig) -> None:
        super().__init__()
        # Each dilation value configures one residual block, so the backbone
        # depth is defined jointly by `num_channels` and `dilations`.
        if len(config.dilations) != len(config.num_channels):
            raise ValueError("Length of dilations must match length of num_channels")

        self.config = config

        # Build the temporal backbone from the dilation schedule declared in the
        # config. In the fused model we instantiate this class three times with
        # kernel sizes 3, 5, and 7, while keeping dilations fixed at [1, 2, 4].
        #
        # Receptive-field intuition:
        # - kernel size chooses how wide each local pattern detector is
        # - dilation chooses how far apart samples in that detector are
        # - stacking dilations [1, 2, 4] grows temporal coverage without making
        #   the model fully recurrent or excessively deep
        #
        # In practice:
        # - k=3 branch leans more local
        # - k=5 branch covers a broader short-term context
        # - k=7 branch covers the widest branch-local context
        #
        # The fused model then combines these complementary branch forecasts.
        blocks: list[nn.Module] = []
        in_channels = config.num_inputs
        for out_channels, dilation in zip(config.num_channels, config.dilations):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    dilation=int(dilation),
                    dropout=config.dropout,
                    activation_name=config.activation,
                )
            )
            in_channels = int(out_channels)
        self.network = nn.Sequential(*blocks)

        # The branch now exposes a horizon-aligned latent representation for the
        # fused model's final fusion stage, while still keeping a small local
        # forecast projection for standalone branch outputs.
        #
        # This is enough for our use case because:
        # - the TCN branch is a base forecaster, not the whole forecasting
        #   system
        # - the fused head is now where branch-to-branch interaction happens
        last_channels = int(config.num_channels[-1])
        self.branch_hidden_size = last_channels
        self.feature_head = nn.Sequential(
            nn.Linear(last_channels, last_channels),
            _build_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(last_channels, config.prediction_length * last_channels),
        )
        self.output_proj = nn.Linear(last_channels, config.output_size)

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "TCN expects input shaped [batch, encoder_length, num_inputs]"
            )
        if x.shape[-1] != self.config.num_inputs:
            raise ValueError(
                f"Expected {self.config.num_inputs} TCN input features, got {x.shape[-1]}"
            )

        # Conv1d operates on channel-first tensors, but the repo's data contract
        # is batch-first with features on the last axis. We transpose only at the
        # model boundary so callers can stay aligned with the dataset contract.
        #
        # Keeping the external contract in `[batch, time, features]` form makes
        # the TCN easier to plug into the same dataset outputs used by TFT and
        # keeps model code closer to how the data pipeline describes sequences.
        x = x.transpose(1, 2)
        x = self.network(x)
        return x.transpose(1, 2)

    def summarize(self, x: Tensor) -> Tensor:
        encoded = self.encode(x)

        # We summarize the final encoded timestep because it has the widest
        # receptive field across the full history window under causal masking.
        #
        # In a causal model, the last encoder position is the place where the
        # branch has seen the full available history up to the forecast origin.
        # That makes it a natural compact summary for mapping history into the
        # next `prediction_length` steps.
        return encoded[:, -1, :]

    def forward_features(self, x: Tensor) -> Tensor:
        # This is the interface the fused model uses. It expands one compact
        # branch summary into one hidden feature vector per forecast step so the
        # TCN can participate in late fusion as a representation branch rather
        # than only as a scalar base forecaster.
        summary = self.summarize(x)
        branch_features = self.feature_head(summary)
        return branch_features.view(
            x.shape[0],
            self.config.prediction_length,
            self.branch_hidden_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Keep a branch-local forecast path available so the TCN still behaves
        # like a normal forecaster when called directly. The fused model now
        # uses `forward_features(...)` instead.
        branch_features = self.forward_features(x)
        return self.output_proj(branch_features)
