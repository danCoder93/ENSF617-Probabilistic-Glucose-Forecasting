from __future__ import annotations

# AI-assisted maintenance note:
# These helpers normalize nested tensor, batch, and metadata structures.
#
# Why they live in their own module:
# - several callbacks need the same recursive traversal logic
# - the reporting/export path also needs metadata normalization
# - these helpers describe data-shape handling, not logging policy
#
# Pulling them out keeps the callback classes easier to read because each
# callback can focus on "what should we log?" instead of "how do we recursively
# walk this batch structure safely?"

from typing import Any, Mapping

import torch
from torch import Tensor


# ============================================================================
# Tensor / Batch Normalization Helpers
# ============================================================================
# The custom callbacks often need to inspect nested batch dictionaries or
# nested model outputs. These helpers normalize that work so each callback can
# stay focused on "what to log" rather than "how to recursively walk arbitrary
# nested structures."
def _flatten_tensor_output(output: Any) -> Tensor | None:
    """
    Find the first tensor payload inside a nested model output structure.

    Context:
    several debug callbacks only need one representative tensor for summary
    statistics, even when the real output is nested.
    """
    # Many model outputs in PyTorch ecosystems are nested structures rather
    # than a single tensor. For summary statistics we only need the first
    # actual tensor payload we can find.
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _flatten_tensor_output(item)
            if tensor is not None:
                return tensor
    if isinstance(output, Mapping):
        for item in output.values():
            tensor = _flatten_tensor_output(item)
            if tensor is not None:
                return tensor
    return None


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """
    Recursively move every tensor in a nested batch structure to one device.

    Context:
    model graph logging and `torchview` rendering need example inputs located
    on the same device as the model.
    """
    # TensorBoard graph logging and torchview rendering need example inputs on
    # the same device as the model. This helper recursively mirrors a nested
    # tensor structure onto the target device while leaving non-tensor metadata
    # untouched.
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {
            key: _move_batch_to_device(value, device) for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)
    return batch


def _tensor_only_structure(batch: Any) -> Any:
    """
    Remove non-tensor values from a nested batch structure.

    Context:
    some visualization utilities only understand tensors, so metadata must be
    filtered out before those tools see the batch.
    """
    # Some visualization utilities do not know what to do with strings,
    # timestamps, or nested metadata objects. This helper removes everything
    # except tensors from a batch structure so the visualization path only sees
    # model-consumable inputs.
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, Mapping):
        filtered: dict[Any, Any] = {}
        for key, value in batch.items():
            nested = _tensor_only_structure(value)
            if nested is not None:
                filtered[key] = nested
        return filtered if filtered else None
    if isinstance(batch, list):
        filtered_list = [
            nested
            for value in batch
            if (nested := _tensor_only_structure(value)) is not None
        ]
        return filtered_list if filtered_list else None
    if isinstance(batch, tuple):
        filtered_tuple = tuple(
            nested
            for value in batch
            if (nested := _tensor_only_structure(value)) is not None
        )
        return filtered_tuple if filtered_tuple else None
    return None


def _tensor_stats(tensor: Tensor) -> dict[str, float]:
    """
    Compute a compact health summary for one tensor.

    Context:
    these statistics are reused across multiple debug callbacks, so keeping
    them centralized ensures consistent logging behavior.
    """
    # Centralized tensor summary logic keeps the callback metrics consistent.
    # Every place in observability that wants a "quick health snapshot" of a
    # tensor uses the same mean/std/min/max/finite accounting.
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    total_count = detached.numel()
    if finite_count == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "finite_fraction": 0.0,
            "nan_count": float(torch.isnan(detached).sum().item()),
            "inf_count": float(torch.isinf(detached).sum().item()),
        }

    finite_values = detached[finite_mask]
    return {
        "mean": float(finite_values.mean().item()),
        "std": float(finite_values.std(unbiased=False).item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "finite_fraction": finite_count / total_count,
        "nan_count": float(torch.isnan(detached).sum().item()),
        "inf_count": float(torch.isinf(detached).sum().item()),
    }


def _summarize_batch(batch: Any) -> Any:
    """
    Convert a nested batch structure into a JSON-friendly summary object.

    Context:
    batch-audit logging needs tensor shape, dtype, and basic statistics without
    trying to serialize raw tensors directly.
    """
    # This turns an arbitrary nested batch structure into a JSON-friendly
    # summary object that can be logged as text. Tensors become shape/dtype/
    # stats dictionaries; non-tensor values are passed through as-is.
    if isinstance(batch, Tensor):
        return {
            "shape": list(batch.shape),
            "dtype": str(batch.dtype),
            "device": str(batch.device),
            "stats": _tensor_stats(batch),
        }
    if isinstance(batch, Mapping):
        return {str(key): _summarize_batch(value) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [_summarize_batch(value) for value in batch]
    return batch


def _as_metadata_lists(
    metadata: Mapping[str, Any],
    batch_size: int,
) -> dict[str, list[Any]]:
    """
    Normalize metadata values into per-sample Python lists.

    Context:
    prediction export and figure generation need metadata that can be indexed
    row by row regardless of whether the original source was scalar, tuple, or
    tensor.
    """
    # Prediction exports and figure callbacks need metadata in a predictable
    # row-oriented shape. This helper converts tensor/list/scalar metadata into
    # "list of per-sample values" form so downstream code can index it
    # uniformly.
    normalized: dict[str, list[Any]] = {}
    for key, value in metadata.items():
        if isinstance(value, Tensor):
            normalized[key] = value.detach().cpu().tolist()
        elif isinstance(value, list):
            normalized[key] = value
        elif isinstance(value, tuple):
            normalized[key] = list(value)
        else:
            normalized[key] = [value for _ in range(batch_size)]
    return normalized
