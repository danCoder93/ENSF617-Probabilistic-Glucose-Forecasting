from __future__ import annotations

from typing import Any

import torch


class TorchviewFusedAdapter(torch.nn.Module):
    """Present a Lightning/PyTorch model as a thin plain `nn.Module` wrapper.

    Purpose:
        Some visualization tools behave more reliably when tracing a minimal
        PyTorch module surface instead of the full higher-level training module.

    Context:
        This adapter is intentionally thin. It does not change the batch
        contract or model behavior. It only forwards the already-prepared
        batch into the wrapped model so observability tools can trace a
        narrower and more stable surface.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the wrapped model used for visualization."""
        super().__init__()
        self.model = model

    def forward(self, batch: Any) -> Any:
        """Forward the sampled visualization batch unchanged."""
        return self.model(batch)