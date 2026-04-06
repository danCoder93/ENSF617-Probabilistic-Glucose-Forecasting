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
        This adapter intentionally preserves the *real* traced internals of the
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


# Keep the staged-intermediate contract explicit for static type checkers.
#
# The fused model's private `_forward_intermediates(...)` helper already returns
# a dictionary of named tensors, but because we access that helper dynamically
# through `getattr(...)`, tools like Pylance lose the precise return type and
# treat the result as `object`. Defining the expected contract here and casting
# once at the boundary restores type information without requiring any changes
# to the real model file.
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

        # These wrappers exist purely so torchview can show a smaller number of
        # named semantic nodes in the graph. They intentionally do not own any
        # parameters or computation.
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
        """Project the fused model into a smaller set of semantic graph stages.

        Important implementation note:
            The named stage modules below are *identity* wrappers. Their job is
            not to perform the real computation themselves. Instead, they make
            the semantic flow visible to graph tools while the wrapped model
            continues to own the actual forecasting logic.
        """
        batch = self._normalize_batch(*args, **kwargs)

        intermediates_fn = getattr(self.model, "_forward_intermediates", None)
        if not callable(intermediates_fn):
            # Conservative fallback for models that do not expose staged
            # intermediates. We still return a stable tensor output, but the
            # graph will naturally be less semantically structured.
            return extract_trace_tensor(self.model(batch))

        intermediates = cast(ForwardIntermediates, intermediates_fn(dict(batch)))

        # Stage 1: surface the encoder/history-side signal that feeds the TCN
        # pathway. This is the narrowest "prepared input" representation already
        # exposed by the fused model.
        prepared_inputs = self.input_stage(intermediates["tcn_inputs"])

        # Stage 2: represent the multiscale TCN pathway as one semantic block.
        # The fused model already computes three branch tensors. We concatenate
        # them here only for visualization-stage grouping; the real model logic
        # is unchanged and continues to use the original branch tensors.
        multiscale_tcn_features = torch.cat(
            [
                intermediates["tcn3_features"],
                intermediates["tcn5_features"],
                intermediates["tcn7_features"],
            ],
            dim=-1,
        )
        multiscale_tcn_features = self.tcn_stage(multiscale_tcn_features)

        # Stage 3: surface the TFT branch representation directly.
        tft_features = self.tft_stage(intermediates["tft_features"])

        # Stage 4: represent the late-fusion hidden state. We preserve a visible
        # dependency on both semantic branches so the graph still communicates
        # that fusion depends on the TCN and TFT pathways together.
        semantic_fusion_view = torch.cat(
            [tft_features, multiscale_tcn_features],
            dim=-1,
        )
        semantic_fusion_view = self.fusion_stage(semantic_fusion_view)

        # The real fused model already computed `post_fusion_features` using its
        # proper GRN-based fusion logic. We combine that real tensor with a zero-
        # weighted semantic view so the output node keeps a visible dependency on
        # the named fusion stage without changing numerical behavior.
        post_fusion_features = intermediates["post_fusion_features"] + (
            0.0 * semantic_fusion_view.mean(dim=-1, keepdim=True)
        )

        # `prepared_inputs` is deliberately folded into the semantic branch view
        # with a zero-weight dependency so the clean graph preserves a visible
        # upstream input-preparation stage without changing tensor values.
        multiscale_tcn_features = multiscale_tcn_features + (
            0.0 * prepared_inputs.mean(dim=-1, keepdim=True)
        )

        # Stage 5: expose the final probabilistic output head.
        predictions = self.head_stage(intermediates["predictions"])

        # As above, preserve the real prediction tensor while keeping a visible
        # dependency on the semantic head stage in the traced graph.
        return extract_trace_tensor(
            predictions + (0.0 * post_fusion_features.mean(dim=-1, keepdim=True))
        )
