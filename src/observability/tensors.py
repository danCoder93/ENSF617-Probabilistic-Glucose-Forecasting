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
#
# Design boundary:
# - this module should stay free of logger-specific behavior
# - this module should not emit side effects
# - this module should return plain Python structures and scalar summaries that
#   callbacks can decide to log, persist, or ignore
#
# Why this matters for the current repo:
# The fused glucose forecasting pipeline passes around nested tensor structures,
# optional feature groups, metadata dictionaries, and model outputs that are not
# always a single flat tensor. Centralizing the normalization and summary logic
# here prevents each callback from growing its own slightly different
# interpretation of "what a batch looks like" or "what a healthy tensor looks
# like".

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

# ============================================================================
# Tensor / Batch Normalization Helpers
# ============================================================================
#
# The custom callbacks often need to inspect nested batch dictionaries or
# nested model outputs. These helpers normalize that work so each callback can
# stay focused on "what to log" rather than "how to recursively walk arbitrary
# nested structures."


def _flatten_tensor_output(output: Any) -> Tensor | None:
    """Find the first tensor payload inside a nested model output structure.

    Purpose:
        Extract one representative tensor from an arbitrarily nested output
        structure.

    Context:
        Several debug callbacks only need one representative tensor for summary
        statistics, even when the real output is nested inside tuples, lists, or
        dictionaries.

    Why "first tensor wins" is acceptable here:
        These helpers are for lightweight observability, not full semantic
        parsing of every possible output contract. For activation summaries and
        similar diagnostics, the first concrete tensor payload is often enough to
        answer questions like:
        - are values finite?
        - are activations all zeros?
        - is the scale obviously broken?

    Failure behavior:
        Returns `None` when no tensor can be found anywhere in the structure.
        Callers should treat that as "nothing summarizable was available" rather
        than as an error.
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
    """Recursively move every tensor in a nested batch structure to one device.

    Purpose:
        Mirror a nested batch structure onto the same device as the model.

    Context:
        Model graph logging and `torchview` rendering need example inputs located
        on the same device as the model. The real repo batch structure is not
        guaranteed to be a single flat tensor, so this helper must recurse
        through mappings, lists, and tuples.

    Important contract:
        - tensors are moved
        - containers are reconstructed recursively
        - non-tensor metadata is preserved as-is

    Why metadata is preserved:
        Some downstream paths still want the batch shape to remain structurally
        recognizable even if only tensors are moved. Filtering out metadata is a
        separate concern handled by `_tensor_only_structure(...)`.
    """
    # TensorBoard graph logging and torchview rendering need example inputs on
    # the same device as the model. This helper recursively mirrors a nested
    # tensor structure onto the target device while leaving non-tensor metadata
    # untouched.
    if isinstance(batch, Tensor):
        return batch.to(device)

    if isinstance(batch, Mapping):
        return {
            key: _move_batch_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]

    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)

    return batch


def _tensor_only_structure(batch: Any) -> Any:
    """Remove non-tensor values from a nested batch structure.

    Purpose:
        Strip metadata and other non-tensor values from a nested batch structure.

    Context:
        Some visualization utilities only understand tensors, so metadata must
        be filtered out before those tools see the batch.

    Important behavior:
        - tensors are kept
        - container structure is preserved where possible
        - empty containers are collapsed to `None`

    Why empty containers become `None`:
        Returning empty dictionaries/lists/tuples would make downstream graphing
        and tracing code think a meaningful input branch still exists. Returning
        `None` makes absence explicit and keeps the tensor-only structure clean.
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


def _shape_list(tensor: Tensor) -> list[int]:
    """Convert a tensor shape into a JSON-friendly Python list.

    Why this tiny helper exists:
        Multiple summary paths want a stable list representation for shapes.
        Keeping the conversion in one place avoids repeated `list(tensor.shape)`
        calls and makes future shape-related formatting changes easier.
    """
    return [int(dim) for dim in tensor.shape]


def _zero_fraction(tensor: Tensor) -> float:
    """Return the fraction of tensor elements that are exactly zero.

    Why this is useful:
        Exact-zero prevalence is a quick signal for:
        - dead ReLU-style activations
        - masked or padded tensors
        - branches that may be producing empty information

    Important limitation:
        A high zero fraction is not automatically a bug. Some tensors are
        expected to contain padding, masks, or sparse features. This helper only
        provides evidence; interpretation belongs in the callback layer.
    """
    detached = tensor.detach()
    total_count = detached.numel()
    if total_count == 0:
        return 0.0
    return float((detached == 0).sum().item()) / float(total_count)


def _near_zero_fraction(
    tensor: Tensor,
    *,
    atol: float = 1e-6,
) -> float:
    """Return the fraction of finite tensor elements whose magnitude is tiny.

    Purpose:
        Detect tensors that are not exactly zero but are still effectively
        "dead" or numerically negligible.

    Why finite-only:
        NaN/inf handling is already reported separately. This helper focuses on
        the magnitude distribution of values that are at least numerically
        defined.

    Why the default threshold is small:
        The goal is not to aggressively label small values as broken. The goal
        is to catch tensors whose outputs have collapsed toward zero in a way
        that deserves a closer look during debugging.
    """
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    if finite_count == 0:
        return 0.0

    finite_values = detached[finite_mask]
    return float((torch.abs(finite_values) <= atol).sum().item()) / float(
        finite_count
    )


def _is_effectively_constant(
    tensor: Tensor,
    *,
    atol: float = 1e-6,
) -> bool:
    """Return whether all finite values in the tensor are effectively constant.

    Purpose:
        Provide a simple constant-value detector for batch contracts, branch
        activations, and predictions.

    Why this matters:
        A constant tensor can indicate:
        - a collapsed branch
        - a broken preprocessing path
        - a target/prediction bug
        - a legitimate constant feature

        The helper itself stays neutral; it only answers whether the values are
        effectively constant within tolerance.

    Finite-value policy:
        Non-finite values are excluded from the comparison because those are
        already captured by finite-fraction / NaN / inf metrics. If there are no
        finite values, the tensor is treated as not meaningfully constant.
    """
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    if finite_count == 0:
        return False

    finite_values = detached[finite_mask]
    value_range = float((finite_values.max() - finite_values.min()).item())
    return value_range <= atol


def _time_axis_constant_fraction(
    tensor: Tensor,
    *,
    atol: float = 1e-6,
) -> float:
    """Estimate how often sequences are effectively constant along time.

    Purpose:
        Provide a lightweight sequence-specific signal for tensors shaped like
        batched time series.

    Interpretation:
        For tensors with shape `[B, T, ...]`, this helper measures how many batch
        rows are effectively constant along the time axis after flattening the
        feature dimensions.

    Why this matters in this repo:
        The glucose forecasting pipeline is sequence-heavy. A tensor that looks
        statistically fine when flattened may still be suspicious if each sample
        is nearly constant across time.

    Fallback behavior:
        If the tensor has fewer than 2 dimensions, there is no meaningful time
        axis to inspect, so the helper returns `0.0`.
    """
    detached = tensor.detach().float()
    if detached.ndim < 2:
        return 0.0

    finite_mask = torch.isfinite(detached)

    # We flatten all non-batch dimensions after the leading batch axis so the
    # helper stays generic for `[B, T]`, `[B, T, F]`, or other sequence-like
    # tensors. The second dimension is interpreted as the time axis only in the
    # sense that we compare variation within each batch item across the rest of
    # the sequence payload.
    batch_size = detached.shape[0]
    flattened = detached.reshape(batch_size, -1)
    flattened_finite = finite_mask.reshape(batch_size, -1)

    constant_rows = 0
    valid_rows = 0

    for row_values, row_finite in zip(flattened, flattened_finite):
        finite_values = row_values[row_finite]
        if finite_values.numel() == 0:
            continue
        valid_rows += 1
        row_range = float((finite_values.max() - finite_values.min()).item())
        if row_range <= atol:
            constant_rows += 1

    if valid_rows == 0:
        return 0.0

    return float(constant_rows) / float(valid_rows)


def _tensor_stats(
    tensor: Tensor,
    *,
    near_zero_atol: float = 1e-6,
) -> dict[str, float]:
    """Compute a compact health summary for one tensor.

    Purpose:
        Produce one consistent tensor-health summary shared across observability
        callbacks.

    Context:
        These statistics are reused across multiple debug callbacks, so keeping
        them centralized ensures consistent logging behavior.

    Included metrics:
        - central tendency / spread: mean, std
        - value range: min, max, abs_mean, max_abs
        - numerical health: finite_fraction, nan_count, inf_count
        - sparsity / collapse hints: zero_fraction, near_zero_fraction
        - constant-tensor hint: is_constant encoded as 0.0 or 1.0

    Why booleans are encoded as floats:
        Most logger backends and metric sinks work most naturally with numeric
        scalar values. Returning `0.0` / `1.0` keeps this helper metric-friendly
        without forcing callbacks to cast values themselves.
    """
    # Centralized tensor summary logic keeps the callback metrics consistent.
    # Every place in observability that wants a "quick health snapshot" of a
    # tensor uses the same mean/std/min/max/finite accounting plus a few more
    # debugging-oriented indicators such as zero/near-zero prevalence.
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    total_count = detached.numel()

    if total_count == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_mean": 0.0,
            "max_abs": 0.0,
            "finite_fraction": 0.0,
            "nan_count": 0.0,
            "inf_count": 0.0,
            "zero_fraction": 0.0,
            "near_zero_fraction": 0.0,
            "is_constant": 0.0,
        }

    if finite_count == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_mean": 0.0,
            "max_abs": 0.0,
            "finite_fraction": 0.0,
            "nan_count": float(torch.isnan(detached).sum().item()),
            "inf_count": float(torch.isinf(detached).sum().item()),
            "zero_fraction": _zero_fraction(detached),
            "near_zero_fraction": 0.0,
            "is_constant": 0.0,
        }

    finite_values = detached[finite_mask]

    return {
        "mean": float(finite_values.mean().item()),
        "std": float(finite_values.std(unbiased=False).item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "abs_mean": float(torch.abs(finite_values).mean().item()),
        "max_abs": float(torch.abs(finite_values).max().item()),
        "finite_fraction": float(finite_count) / float(total_count),
        "nan_count": float(torch.isnan(detached).sum().item()),
        "inf_count": float(torch.isinf(detached).sum().item()),
        "zero_fraction": _zero_fraction(detached),
        "near_zero_fraction": _near_zero_fraction(
            detached,
            atol=near_zero_atol,
        ),
        "is_constant": 1.0 if _is_effectively_constant(detached) else 0.0,
    }


def _tensor_contract_summary(
    tensor: Tensor,
    *,
    near_zero_atol: float = 1e-6,
) -> dict[str, Any]:
    """Build a richer JSON-friendly summary for one tensor in a batch contract.

    Purpose:
        Provide a more descriptive structure than `_tensor_stats(...)` alone for
        one-time batch auditing and contract inspection.

    Why this helper is separate from `_tensor_stats(...)`:
        `_tensor_stats(...)` returns flat scalar metrics suitable for logger
        backends. Batch audit logs instead benefit from richer structural fields
        such as shape, dtype, device, rank, and zero-width status.

    Included structure:
        - shape / ndim / numel
        - dtype / device
        - zero-width flag
        - scalar health summary from `_tensor_stats(...)`
        - a sequence-oriented constantness hint

    Why sequence-constantness is included here:
        Contract logging is where we most want a human-readable answer to
        questions like "does this time-series tensor actually vary?"
    """
    shape = _shape_list(tensor)
    zero_width = any(dim == 0 for dim in shape)

    return {
        "shape": shape,
        "ndim": int(tensor.ndim),
        "numel": int(tensor.numel()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "zero_width": bool(zero_width),
        "time_axis_constant_fraction": _time_axis_constant_fraction(
            tensor,
            atol=near_zero_atol,
        ),
        "stats": _tensor_stats(
            tensor,
            near_zero_atol=near_zero_atol,
        ),
    }


def _summarize_batch(
    batch: Any,
    *,
    near_zero_atol: float = 1e-6,
) -> Any:
    """Convert a nested batch structure into a JSON-friendly summary object.

    Purpose:
        Turn an arbitrary nested batch structure into a human-readable summary
        that can be safely serialized to JSON.

    Context:
        Batch-audit logging needs tensor shape, dtype, device, and basic
        statistics without trying to serialize raw tensors directly.

    Important design choice:
        This helper stays generic and recursive. It does not assume any
        repo-specific batch key names. Repo-aware interpretation should happen
        one layer up in higher-level helpers or callbacks.

    Why generic recursion still matters:
        Even though this repo has a known fused-model batch contract, generic
        recursion is still useful for:
        - unexpected nested metadata
        - ad hoc debugging of intermediate structures
        - future model/data variants that add or remove fields
    """
    # This turns an arbitrary nested batch structure into a JSON-friendly
    # summary object that can be logged as text. Tensors become shape/dtype/
    # stats dictionaries; non-tensor values are passed through as-is.
    if isinstance(batch, Tensor):
        return _tensor_contract_summary(
            batch,
            near_zero_atol=near_zero_atol,
        )

    if isinstance(batch, Mapping):
        return {
            str(key): _summarize_batch(value, near_zero_atol=near_zero_atol)
            for key, value in batch.items()
        }

    if isinstance(batch, (list, tuple)):
        return [
            _summarize_batch(value, near_zero_atol=near_zero_atol)
            for value in batch
        ]

    return batch


def _batch_semantic_overview(
    batch: Any,
    *,
    expected_tensor_keys: Sequence[str] | None = None,
    near_zero_atol: float = 1e-6,
) -> dict[str, Any]:
    """Build a repo-friendly semantic overview for top-level batch inspection.

    Purpose:
        Produce a compact "contract view" of the top-level batch object that is
        easier for humans and later GenAI analysis to reason about than a raw
        recursive dump alone.

    Intended use:
        This helper is especially useful for one-time batch audit logging in
        callbacks, where we want explicit answers such as:
        - which top-level keys are present?
        - which expected tensor groups are missing?
        - which tensor groups are zero-width or constant?
        - what metadata keys exist?

    Why this helper is conservative:
        The repo may evolve its batch contract over time. Rather than hard-code
        strong assumptions about every nested field, this overview focuses on
        the top-level structure and delegates full recursive detail to
        `_summarize_batch(...)`.

    Parameters:
        batch:
            The real batch object observed at runtime.

        expected_tensor_keys:
            Optional top-level keys that callers consider important for contract
            checking. Missing keys are reported explicitly.

        near_zero_atol:
            Threshold used in the underlying tensor summaries.
    """
    overview: dict[str, Any] = {
        "batch_type": type(batch).__name__,
        "top_level_keys": [],
        "present_tensor_keys": [],
        "missing_expected_tensor_keys": [],
        "metadata_keys": [],
        "tensor_groups": {},
        "raw_structure": _summarize_batch(batch, near_zero_atol=near_zero_atol),
    }

    if not isinstance(batch, Mapping):
        # Some data paths may not use a top-level dictionary batch contract.
        # In that case we still return a useful overview with the raw recursive
        # structure attached, but there is no key-level semantic analysis to do.
        return overview

    top_level_keys = [str(key) for key in batch.keys()]
    overview["top_level_keys"] = top_level_keys

    tensor_groups: dict[str, Any] = {}
    metadata_keys: list[str] = []

    for key, value in batch.items():
        key_name = str(key)

        if isinstance(value, Tensor):
            tensor_groups[key_name] = _tensor_contract_summary(
                value,
                near_zero_atol=near_zero_atol,
            )
            continue

        nested_tensor = _flatten_tensor_output(value)
        if nested_tensor is not None:
            # Nested non-scalar structures such as lists/tuples/dicts can still
            # contain tensors that are semantically important top-level groups.
            # We summarize the first representative tensor so the top-level
            # contract view remains compact.
            tensor_groups[key_name] = {
                "top_level_type": type(value).__name__,
                "representative_tensor": _tensor_contract_summary(
                    nested_tensor,
                    near_zero_atol=near_zero_atol,
                ),
            }
            continue

        metadata_keys.append(key_name)

    overview["tensor_groups"] = tensor_groups
    overview["present_tensor_keys"] = sorted(tensor_groups.keys())
    overview["metadata_keys"] = sorted(metadata_keys)

    if expected_tensor_keys is not None:
        expected = [str(key) for key in expected_tensor_keys]
        overview["missing_expected_tensor_keys"] = sorted(
            key for key in expected if key not in tensor_groups
        )

    return overview


def _as_metadata_lists(
    metadata: Mapping[str, Any],
    batch_size: int,
) -> dict[str, list[Any]]:
    """Normalize metadata values into per-sample Python lists.

    Purpose:
        Convert mixed metadata shapes into a uniform row-oriented representation.

    Context:
        Prediction export and figure generation need metadata that can be indexed
        row by row regardless of whether the original source was scalar, tuple,
        list, or tensor.

    Important behavior:
        - tensors are detached and moved to CPU before conversion
        - tuples become lists for easier downstream JSON/tabular handling
        - scalars are broadcast across the batch size

    Why scalar broadcasting is helpful:
        Some metadata describes the whole batch rather than individual rows. By
        broadcasting scalars into per-sample lists, downstream code can treat
        all metadata uniformly during export and visualization.
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