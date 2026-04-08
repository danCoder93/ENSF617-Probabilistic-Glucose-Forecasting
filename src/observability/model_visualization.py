from __future__ import annotations

from typing import Any, Mapping, cast

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
    """Expose a visualization-stable surface for low-level graph tools.

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

    Reading note:
        This adapter intentionally preserves the real traced internals of the
        wrapped model. That makes it useful for deep debugging, but it also
        means the resulting graph can become visually crowded for large fused
        architectures. For a higher-level, easier-to-read presentation view,
        use `SemanticTorchviewAdapter` below instead.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Normalize torchview inputs back into the model's real batch contract.

        Torchview may call the wrapped module as `model(**batch_dict)`, but the
        fused training model expects a single batch mapping argument. This adapter
        rebuilds that single batch object before delegating to the real model.
        """
        if args and kwargs:
            raise TypeError(
                "TorchviewFusedAdapter.forward received both positional and "
                "keyword inputs; expected exactly one batch representation."
            )

        if len(args) > 1:
            raise TypeError(
                "TorchviewFusedAdapter.forward expected at most one positional "
                "batch argument."
            )

        if len(args) == 1:
            batch = args[0]
        elif kwargs:
            batch = kwargs
        else:
            raise TypeError(
                "TorchviewFusedAdapter.forward expected a batch argument."
            )

        forward_for_visualization = getattr(
            self.model, "forward_for_visualization", None
        )

        if callable(forward_for_visualization):
            output = forward_for_visualization(batch)
        else:
            output = self.model(batch)

        return extract_trace_tensor(output)


class _VisualizationStage(torch.nn.Module):
    """Lightweight stage wrapper used to make graph intent visually explicit.

    Why this exists:
        Torchview primarily understands module boundaries. The real fused model
        already has meaningful semantic stages, but many of those stages are
        expressed as internal helper logic rather than as separately named
        `nn.Module` nodes.

        This wrapper gives the visualization path stable, human-readable stage
        boundaries without changing the real model architecture, training
        semantics, parameter ownership, or batch contract.

    Important boundary:
        This wrapper is observability-only. It should not be used as a
        train-time abstraction and should not be mistaken for a true architectural
        refactor of the underlying model.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pass the tensor through unchanged while preserving a named node."""
        return tensor


def _require_batch_mapping(batch: Any) -> Mapping[str, Any]:
    """Normalize visualization inputs into the repo's batch-mapping contract.

    Why this helper exists:
        Both visualization adapters accept the same flexible input surface that
        torchview may use during tracing:
        - one positional batch mapping argument
        - keyword arguments representing batch fields

        Keeping the normalization logic in one helper avoids subtle divergence
        between the low-level and semantic visualization paths.
    """
    if isinstance(batch, Mapping):
        return batch

    raise TypeError(
        "Visualization adapters expected the batch to be a mapping of tensors. "
        f"Got {type(batch).__name__} instead."
    )


ForwardIntermediates = Mapping[str, torch.Tensor]


class SemanticTorchviewAdapter(torch.nn.Module):
    """Expose a higher-level, presentation-oriented graph surface for torchview.

    Purpose:
        The low-level adapter above is useful when the goal is to inspect the
        real traced graph in detail, but large fused models often become hard to
        read when every internal branch and nested module is expanded at once.

        This semantic adapter keeps the real model computation intact while
        presenting the forward pass through a smaller number of visually named
        stages that better match how a human explains the architecture:
        - input preparation
        - multiscale TCN pathway
        - TFT pathway
        - late fusion
        - quantile prediction head

    How it stays conservative:
        - it does not change training code
        - it does not modify model parameters
        - it does not replace the model's true forward path in optimization
        - it only depends on staged intermediates that the fused model already
          computes internally

    Compatibility policy:
        This adapter prefers `_forward_intermediates(...)` when the wrapped
        model exposes it. That keeps the semantic graph aligned with the real
        repository logic without requiring a model refactor. If the wrapped
        model does not expose staged intermediates, the adapter falls back to
        the regular model forward and returns the final prediction tensor.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.input_stage = _VisualizationStage("input_preparation")
        self.tcn_stage = _VisualizationStage("multiscale_tcn_path")
        self.tft_stage = _VisualizationStage("tft_path")
        self.fusion_stage = _VisualizationStage("late_fusion")
        self.head_stage = _VisualizationStage("quantile_head")

    def _normalize_batch(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Recover the repo's single-batch mapping contract from torchview input."""
        if args and kwargs:
            raise TypeError(
                "SemanticTorchviewAdapter.forward received both positional and "
                "keyword inputs; expected exactly one batch representation."
            )

        if len(args) > 1:
            raise TypeError(
                "SemanticTorchviewAdapter.forward expected at most one positional "
                "batch argument."
            )

        if len(args) == 1:
            return _require_batch_mapping(args[0])
        if kwargs:
            return _require_batch_mapping(kwargs)

        raise TypeError(
            "SemanticTorchviewAdapter.forward expected a batch argument."
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Project the fused model into a smaller set of semantic graph stages."""
        batch = self._normalize_batch(*args, **kwargs)

        intermediates_fn = getattr(self.model, "_forward_intermediates", None)
        if not callable(intermediates_fn):
            return extract_trace_tensor(self.model(batch))

        intermediates = cast(ForwardIntermediates, intermediates_fn(dict(batch)))

        prepared_inputs = self.input_stage(intermediates["tcn_inputs"])

        multiscale_tcn_features = torch.cat(
            [
                intermediates["tcn3_features"],
                intermediates["tcn5_features"],
                intermediates["tcn7_features"],
            ],
            dim=-1,
        )

        # Important shape fix:
        # `prepared_inputs` lives on the encoder-history axis, while
        # `multiscale_tcn_features` lives on the decoder-horizon axis. To keep a
        # visible dependency without creating an axis mismatch, collapse the
        # prepared input to one broadcastable scalar per batch item.
        multiscale_tcn_features = multiscale_tcn_features + (
            0.0 * prepared_inputs.mean(dim=(1, 2), keepdim=True)
        )
        multiscale_tcn_features = self.tcn_stage(multiscale_tcn_features)

        tft_features = self.tft_stage(intermediates["tft_features"])

        semantic_fusion_view = torch.cat(
            [tft_features, multiscale_tcn_features],
            dim=-1,
        )
        semantic_fusion_view = self.fusion_stage(semantic_fusion_view)

        post_fusion_features = intermediates["post_fusion_features"] + (
            0.0 * semantic_fusion_view.mean(dim=-1, keepdim=True)
        )

        predictions = self.head_stage(intermediates["predictions"])
        return extract_trace_tensor(
            predictions + (0.0 * post_fusion_features.mean(dim=-1, keepdim=True))
        )
