from __future__ import annotations

"""
AI-assisted maintenance note:
These tests were added with AI assistance on April 1, 2026 to protect the GRN
encapsulation refactor.

They focus on the narrow behavior that changed during the refactor:
- config-backed GRN construction through `GRN.from_tft_config(...)`
- support for both rank-2 and rank-3 inputs when context is provided
- explicit failure for unsupported context-rank combinations

The intent is to keep the test surface small but targeted so future changes can
distinguish real GRN regressions from unrelated model-level failures.
"""

import pytest

torch = pytest.importorskip("torch")

from models.grn import GRN
from utils.config import TFTConfig


def test_grn_from_tft_config_uses_shared_defaults() -> None:
    # This verifies the central refactor goal: GRN instances can inherit shared
    # TFT defaults without hiding the structural dimensions that remain specific
    # to the local call site.
    config = TFTConfig(hidden_size=16, dropout=0.2, layer_norm_eps=1e-4)

    module = GRN.from_tft_config(
        config,
        input_size=4,
        output_size=8,
        context_hidden_size=6,
    )

    assert module.lin_a.in_features == 4
    assert module.lin_a.out_features == config.hidden_size
    assert module.lin_c.in_features == 6
    assert module.dropout.p == config.dropout
    assert module.layer_norm.ln.eps == config.layer_norm_eps


def test_grn_supports_rank_2_inputs_with_context() -> None:
    # Rank-2 support matters for static or non-temporal feature paths where the
    # caller provides one feature vector and one context vector per batch item.
    module = GRN(
        input_size=4,
        hidden_size=8,
        output_size=6,
        context_hidden_size=5,
    )

    out = module(torch.randn(3, 4), torch.randn(3, 5))

    assert out.shape == (3, 6)


def test_grn_supports_rank_3_inputs_with_context_broadcast() -> None:
    # Rank-3 support preserves the original TFT behavior where static context is
    # broadcast across a temporal dimension.
    module = GRN(
        input_size=4,
        hidden_size=8,
        output_size=6,
        context_hidden_size=5,
    )

    out = module(torch.randn(3, 7, 4), torch.randn(3, 5))

    assert out.shape == (3, 7, 6)


def test_grn_rejects_unsupported_context_rank_mismatch() -> None:
    # A mismatched rank should now fail loudly rather than silently relying on
    # accidental broadcasting semantics.
    module = GRN(
        input_size=4,
        hidden_size=8,
        output_size=6,
        context_hidden_size=5,
    )

    with pytest.raises(ValueError, match="Context tensor rank"):
        module(torch.randn(3, 2, 7, 4), torch.randn(3, 5))
