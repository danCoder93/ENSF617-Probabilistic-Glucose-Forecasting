from __future__ import annotations

# AI-assisted maintenance note (April 1, 2026):
# This existing module was refined with AI assistance under direct user
# guidance. The changes here are intended to clarify, harden, and better
# encapsulate the project's current GRN implementation for reuse across the TFT
# internals and the fused forecast head while preserving the established
# architecture.
#
# The refactor intentionally kept per-call structural dimensions explicit
# (`input_size`, `output_size`, and `context_hidden_size`) because those vary
# by use site, but centralized shared defaults such as `hidden_size`,
# `dropout`, and layer-norm epsilon through the TFT config path via
# `GRN.from_tft_config`.
#
# The goal of these changes is not to present new model ideas, but to improve
# the robustness, consistency, and maintainability of the existing GRN math
# block and its config-driven construction path.

from typing import Optional

import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm, Module
from torch.nn.functional import elu, glu

from config import TFTConfig


class MaybeLayerNorm(Module):
    """
    Apply layer norm unless the output collapses to a single scalar channel.

    Purpose:
    keep the GRN normalization step numerically sensible when the output width
    is only one channel.

    Context:
    a size-1 output is a special case because normalizing a single scalar per
    position is not very meaningful and can introduce avoidable instability.
    """

    def __init__(self, output_size, hidden_size, eps):
        """Choose between real layer norm and identity based on the effective output width."""
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = LayerNorm(output_size if output_size else hidden_size, eps=eps)

    def forward(self, x):
        """Apply the chosen normalization layer to the incoming tensor."""
        return self.ln(x)


class GLU(Module):
    """
    Gated linear unit used inside the GRN blocks.

    Purpose:
    turn one hidden representation into a gated output where the block can
    learn both content and how much of that content should pass through.

    Context:
    conceptually this learns:
    - a candidate transformed signal
    - a gate deciding how much of that signal should pass through
    """

    def __init__(self, hidden_size, output_size):
        """Create the linear projection that supplies content and gate channels to the GLU."""
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the learned projection and split it into gated GLU output channels."""
        x = self.lin(x)
        x = glu(x)
        return x


class GRN(Module):
    """
    Gated Residual Network block used throughout TFT and the fused head.

    High-level behavior:
    - project the input into a hidden space
    - optionally inject a context vector
    - apply a nonlinear transformation
    - gate the transformed signal
    - add a residual shortcut
    - normalize the result

    Architectural role in this codebase:
    - TFT relies on GRNs repeatedly as its standard nonlinear processing block
    - the fused model also uses a GRN to mix branch features before the final
      readout head
    - sharing the same block keeps the fused architecture stylistically and
      numerically aligned with the TFT internals
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=None,
        context_hidden_size=None,
        dropout=0.0,
        layer_norm_eps=1e-3,
    ):
        """
        Build one GRN block with optional context injection and optional output projection.

        Context:
        this same block is reused across TFT internals and the fused-model
        fusion head, so the constructor keeps the shape-defining dimensions
        explicit while validating the shared GRN math contract.
        """
        super().__init__()
        # Validate the low-level GRN contract at construction time so
        # configuration mistakes fail early and consistently. The rest of the
        # codebase already validates dataclass-backed model settings in this
        # style, so keeping the same pattern here makes the shared block safer
        # to instantiate from multiple callers.
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        if output_size is not None and output_size <= 0:
            raise ValueError("output_size must be > 0 when provided")
        if context_hidden_size is not None and context_hidden_size <= 0:
            raise ValueError("context_hidden_size must be > 0 when provided")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if layer_norm_eps <= 0.0:
            raise ValueError("layer_norm_eps must be > 0.0")

        # Layer norm epsilon is now configurable through the shared TFT config
        # path rather than hardcoded. This keeps GRN normalization aligned with
        # the rest of the TFT stack while still allowing direct GRN construction
        # in tests or future non-TFT callers.
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=layer_norm_eps)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        else:
            self.lin_c = nn.Identity()
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    @classmethod
    def from_tft_config(
        cls,
        config: TFTConfig,
        *,
        input_size: int,
        output_size: Optional[int] = None,
        context_hidden_size: Optional[int] = None,
    ) -> "GRN":
        """
        Build a GRN that inherits hidden/dropout/norm defaults from `TFTConfig`.

        Context:
        this keeps TFT-owned GRNs and the fused model's post-fusion GRN aligned
        without forcing each call site to repeat the same shared defaults.
        """
        # Keep the shape-defining dimensions explicit at the call site while
        # inheriting the shared architectural defaults that should stay aligned
        # across TFT-owned GRNs and the fused model's post-TFT fusion block.
        return cls(
            input_size=input_size,
            hidden_size=config.hidden_size,
            output_size=output_size,
            context_hidden_size=context_hidden_size,
            dropout=config.dropout,
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-3),
        )

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        """
        Apply the GRN transform, optional context injection, residual shortcut, and normalization.

        Context:
        the method accepts either rank-2 or rank-3 feature tensors so the same
        GRN implementation can serve both static and temporal code paths.
        """
        # `a` is the main feature tensor. Depending on the caller it may be:
        # - rank 2: [batch, features]
        # - rank 3: [batch, time, features]
        #
        # `c` is an optional context tensor, typically a static context vector
        # produced by TFT's static encoder.
        x = self.lin_a(a)
        if c is not None:
            context = self.lin_c(c)
            # Historically the TFT path passed rank-3 temporal inputs and rank-2
            # static context, which requires broadcasting across the time axis.
            # The new branch below preserves that behavior while also supporting
            # rank-2 feature inputs with rank-2 context so the GRN can be reused
            # more safely outside the exact original TFT call pattern.
            if x.dim() == context.dim():
                x = x + context
            elif x.dim() == context.dim() + 1:
                x = x + context.unsqueeze(1)
            else:
                raise ValueError(
                    "Context tensor rank must match the input rank or be broadcastable "
                    "across a single temporal dimension."
                )
        # The nonlinear path learns a transformed view of the input, while the
        # residual path below preserves a direct route for the original signal.
        x = elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        # If the output dimensionality changes, the shortcut must be projected
        # so the residual addition stays shape-compatible.
        y = a if self.out_proj is None else self.out_proj(a)
        x = x + y
        return self.layer_norm(x)
