from __future__ import annotations

from typing import Any

import torch


def extract_trace_tensor(output: Any) -> torch.Tensor:
    """Reduce model outputs to one deterministic tensor for visualization tools.

    Purpose:
        Torchview and similar tools are happiest when the wrapped forward
        returns one plain tensor. The real training model may return a tensor,
        tuple, list, or dict depending on how the architecture evolves.

    Behavior:
        - tensors are returned unchanged
        - dicts are searched in insertion order for the first tensor value
        - tuples/lists are searched in order for the first tensor entry
        - unsupported outputs raise a clear error instead of silently tracing
          something incorrect
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("Visualization output dict did not contain a tensor.")

    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("Visualization output sequence did not contain a tensor.")

    raise TypeError(
        f"Unsupported visualization output type: {type(output).__name__}."
    )


def warmup_visualization_model(model: torch.nn.Module, batch: Any) -> None:
    """Run one best-effort warmup forward before graph capture.

    Why this exists:
        Some model components finalize internal state only after their first
        forward pass. Running one warmup pass before torchview reduces the
        chance that graph tracing sees different module structure across
        repeated internal invocations.

    Failure behavior:
        Any exception is allowed to propagate to the caller so the callback can
        log whether warmup or graph rendering failed.
    """
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            _ = model(batch)
    finally:
        model.train(was_training)


class TorchviewFusedAdapter(torch.nn.Module):
    """Expose a visualization-stable surface for graph tools.

    Design intent:
        This adapter is used only by observability code. It does not change the
        real training workflow and should never be used as a substitute for the
        actual model during optimization or evaluation.

    Behavior:
        - prefers a dedicated `forward_for_visualization(...)` method when the
          wrapped model provides one
        - otherwise falls back to the real forward path
        - always reduces the result to one tensor so tracing tools see a
          simpler and more deterministic output contract
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: Any) -> torch.Tensor:
        forward_for_visualization = getattr(
            self.model, "forward_for_visualization", None
        )
        if callable(forward_for_visualization):
            output = forward_for_visualization(batch)
        else:
            output = self.model(batch)
        return extract_trace_tensor(output)