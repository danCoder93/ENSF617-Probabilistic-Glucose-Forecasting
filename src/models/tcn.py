"""
Temporal Convolution Network branch for the fused glucose forecasting pipeline.

High-level role in the model:
- This TCN branch is the "short / medium range pattern specialist" inside the
  fused architecture.
- It reads only encoder-side history and learns local temporal structure such
  as short-lived glucose moves, post-meal changes, and insulin-driven dynamics.
- In the full fused model, three instances of this class are created with
  kernel sizes 3, 5, and 7 so the system can look at the same history through
  multiple temporal scales before handing those branch forecasts to the TFT.

Why this file is intentionally narrow:
- The repository does not need a general-purpose TCN library.
- It needs one readable, project-specific temporal forecaster that matches the
  batch contract produced by the data pipeline and the fusion contract expected
  by the TCN-TFT model.
- That is why this file keeps only the causal residual Conv1d backbone and a
  small horizon projection head.

Code provenance / disclaimer:
- This file is influenced by the original TCN paper and by the open-source
  `pytorch-tcn` project by Paul Krug:
  https://github.com/paul-krug/pytorch-tcn
- Within this repository, the main project-specific TCN implementation is most
  closely based on mcheema88's work on the `MarleyTCNClean` branch:
  https://github.com/danCoder93/ENSF617-Probabilistic-Glucose-Forecasting/blob/MarleyTCNClean/src/models/tcn.py
- The current file preserves that lineage at the architectural level:
  causal convolutions, residual temporal blocks, and the project-specific
  multiscale TCN direction.
- This version further simplifies and adapts that earlier implementation for
  the fused TCN-TFT forecasting pipeline used in this project.

How this repository's version differs from the upstream reference:
- keeps only the causal residual Conv1d path needed by this project
- removes generic library features that are not part of the current use case,
  such as streaming buffer management, transposed-convolution decoding,
  compatibility shims, and multiple operating modes
- standardizes on the dataset/model contract used in this repo:
  `[batch, encoder_length, features] -> [batch, prediction_length, outputs]`
- uses a lightweight horizon projection head so each TCN branch acts as a base
  forecaster inside the fused TCN-TFT architecture

Generative AI assistance disclaimer:
- Generative AI tools contributed to parts of the earlier repository TCN work
  and were also used here to help summarize provenance, document
  project-specific modifications, and draft explanatory comments.
- Final responsibility for integration, editing, and technical intent in this
  repository remains with the project authors listed above.
"""

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
# Why this exists:
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

    Why layer norm is used here:
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
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Module):
    """
    Conv1d with left padding only.

    Why causal padding matters:
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
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    Residual TCN block with two causal convolutions.

    Design intent:
    - preserve the standard residual-block flavor of TCNs
    - keep the implementation minimal and readable
    - expose only the pieces that matter to the fused architecture

    Why two convolutions plus a residual path:
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
    - `[batch, prediction_length, output_size]`
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

        # The forecast head is the project-specific "decoder" we kept:
        # instead of a large generic decoder stack, we summarize the final
        # encoded history state and project it directly to the forecast horizon.
        #
        # This is enough for our use case because:
        # - the TCN branch is a base forecaster, not the whole forecasting
        #   system
        # - TFT is the branch that later refines future-aware behavior using
        #   known decoder inputs and richer fusion logic
        last_channels = int(config.num_channels[-1])
        self.forecast_head = nn.Sequential(
            nn.Linear(last_channels, last_channels),
            _build_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(last_channels, config.prediction_length * config.output_size),
        )

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

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encode(x)

        # We summarize the final encoded timestep because it has the widest
        # receptive field across the full history window under causal masking.
        #
        # In a causal model, the last encoder position is the place where the
        # branch has seen the full available history up to the forecast origin.
        # That makes it a natural compact summary for mapping history into the
        # next `prediction_length` steps.
        summary = encoded[:, -1, :]
        forecast = self.forecast_head(summary)
        # The head emits one flat vector per batch item, which we reshape back
        # into the standard forecasting layout expected by the fused pipeline.
        return forecast.view(
            x.shape[0],
            self.config.prediction_length,
            self.config.output_size,
        )
