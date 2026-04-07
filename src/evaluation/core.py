from __future__ import annotations

# AI-assisted maintenance note:
# This module centralizes the evaluation package's normalization logic.
#
# Responsibility boundary:
# - normalize target tensors into one consistent shape
# - select a representative point forecast from probabilistic predictions
# - normalize metadata into per-sample Python containers
# - validate and assemble the canonical `EvaluationBatch` contract
#
# Why this matters:
# evaluation becomes much easier to reason about when shape validation and
# metadata normalization happen once at the package boundary instead of being
# reimplemented in the model, reports, and tests.
#
# Shape convention reminder:
# the canonical evaluation view in this repository is:
# - probabilistic predictions: `[batch, horizon, quantiles]`
# - targets: `[batch, horizon]`
# - representative point forecast: `[batch, horizon]`
#
# The rest of the package leans on those conventions heavily, so this module is
# intentionally strict about enforcing them early.

from typing import Any, Mapping, Sequence

from torch import Tensor

from evaluation.types import EvaluationBatch


def normalize_target_tensor(target: Tensor) -> Tensor:
    """
    Normalize targets to `[batch, horizon]` float tensors.

    Context:
    several code paths may carry targets as `[batch, horizon, 1]`, but the
    evaluation layer uses `[batch, horizon]` as its canonical shape.
    """

    # The evaluation layer treats target tensors as numeric comparison data
    # only, so casting to float once here keeps downstream metric code simpler
    # and avoids repeated dtype juggling.
    target = target.float()
    if target.ndim == 3 and target.shape[-1] == 1:
        # Several model/data code paths naturally produce `[batch, horizon, 1]`
        # targets because that shape aligns well with sequence-model outputs.
        # Evaluation collapses that trailing singleton so all point-forecast
        # metrics can assume `[batch, horizon]`.
        target = target.squeeze(-1)
    if target.ndim != 2:
        raise ValueError(
            "Expected target tensor with shape [batch, horizon] or "
            f"[batch, horizon, 1], got {tuple(target.shape)}."
        )
    return target


def select_point_prediction(
    predictions: Tensor,
    quantiles: Sequence[float],
    *,
    quantile: float = 0.5,
) -> Tensor:
    """
    Select the configured quantile channel closest to `quantile`.

    Context:
    the project trains probabilistic forecasts, but several human-readable
    metrics still rely on one representative deterministic curve.
    """

    if predictions.ndim != 3:
        raise ValueError(
            "Expected predictions with shape [batch, horizon, quantiles], "
            f"got {tuple(predictions.shape)}."
        )
    # Quantiles are normalized to plain floats before indexing so:
    # - comparisons behave consistently even if callers passed NumPy scalars
    # - the returned index logic is deterministic and easy to reason about
    normalized_quantiles = tuple(float(value) for value in quantiles)
    if len(normalized_quantiles) != predictions.shape[-1]:
        raise ValueError(
            "Prediction quantile dimension does not match configured quantiles: "
            f"{predictions.shape[-1]} != {len(normalized_quantiles)}."
        )
    # We intentionally choose the configured quantile that is *closest* to the
    # requested point quantile instead of demanding an exact match. That keeps
    # the helper usable for common quantile sets like `(0.1, 0.5, 0.9)` while
    # still behaving sensibly if a caller later requests something like `0.5`
    # from a denser quantile grid that does not contain an exact float match.
    quantile_index = min(
        range(len(normalized_quantiles)),
        key=lambda index: abs(normalized_quantiles[index] - quantile),
    )
    return predictions[..., quantile_index]


def quantile_key(quantile: float) -> str:
    """
    Create a stable JSON/logging-friendly key for one quantile.

    Context:
    quantiles are floats in the model contract, but report payloads and JSON
    summaries are easier to work with when their keys are stable strings.
    """

    return f"q{int(round(float(quantile) * 100)):02d}"


def normalize_batch_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    batch_size: int,
) -> dict[str, tuple[Any, ...]]:
    """
    Normalize metadata values into per-sample tuples.

    Context:
    grouped evaluation needs metadata that can be indexed sample-by-sample
    regardless of whether the original batch metadata arrived as scalars,
    tuples, lists, or tensors.
    """

    if metadata is None:
        return {}

    normalized: dict[str, tuple[Any, ...]] = {}
    for key, value in metadata.items():
        if isinstance(value, Tensor):
            # Tensor metadata is detached onto CPU so the evaluation result is a
            # plain analysis artifact rather than an object still entangled
            # with autograd or device placement.
            values = value.detach().cpu().tolist()
        elif isinstance(value, list):
            values = value
        elif isinstance(value, tuple):
            values = list(value)
        else:
            # Scalar metadata is broadcast to every sample in the batch. This
            # is useful for fields that are conceptually batch-level context
            # but still need to be indexable per sample later.
            values = [value for _ in range(batch_size)]

        if len(values) != batch_size:
            raise ValueError(
                f"Metadata field '{key}' has length {len(values)}, expected {batch_size}."
            )
        normalized[str(key)] = tuple(values)
    return normalized


def build_evaluation_batch(
    *,
    predictions: Tensor,
    target: Tensor,
    quantiles: Sequence[float],
    metadata: Mapping[str, Any] | None = None,
    point_quantile: float = 0.5,
) -> EvaluationBatch:
    """
    Build the canonical evaluation-batch contract from raw tensors.

    Context:
    this function is the evaluation package's main defensive boundary. It
    validates shape compatibility early so later metric/reporting code can
    assume one stable contract.
    """

    # This function is intentionally the main "be strict here so the rest of
    # the package can stay simple" boundary for evaluation inputs.
    if predictions.ndim != 3:
        raise ValueError(
            "Expected predictions with shape [batch, horizon, quantiles], "
            f"got {tuple(predictions.shape)}."
        )

    normalized_target = normalize_target_tensor(target)
    # Once targets are normalized, every remaining validation check is about
    # making sure the probabilistic forecast stack and target grid describe the
    # same batch of forecast windows.
    if predictions.shape[0] != normalized_target.shape[0]:
        raise ValueError(
            "Prediction batch size does not match target batch size: "
            f"{predictions.shape[0]} != {normalized_target.shape[0]}."
        )
    if predictions.shape[1] != normalized_target.shape[1]:
        raise ValueError(
            "Prediction horizon length does not match target horizon length: "
            f"{predictions.shape[1]} != {normalized_target.shape[1]}."
        )
    if predictions.shape[2] != len(tuple(float(value) for value in quantiles)):
        raise ValueError(
            "Prediction quantile dimension does not match configured quantiles: "
            f"{predictions.shape[2]} != {len(tuple(float(value) for value in quantiles))}."
        )

    # The selected point forecast is stored explicitly on the batch so later
    # metric code does not need to keep rediscovering which quantile channel is
    # currently acting as the representative deterministic curve.
    point_prediction = select_point_prediction(
        predictions,
        quantiles,
        quantile=point_quantile,
    )
    return EvaluationBatch(
        # Detach/cpu handling is left to the caller because some users may want
        # to evaluate already-normalized in-memory tensors, but dtype
        # normalization lives here so the result contract stays consistent.
        predictions=predictions.float(),
        target=normalized_target,
        point_prediction=point_prediction.float(),
        quantiles=tuple(float(value) for value in quantiles),
        point_quantile=float(point_quantile),
        metadata=normalize_batch_metadata(
            metadata,
            batch_size=int(predictions.shape[0]),
        ),
    )
