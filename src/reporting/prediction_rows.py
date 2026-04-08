from __future__ import annotations

# AI-assisted maintenance note:
# This module owns canonical row-level prediction table construction for the
# repository's post-run reporting layer.
#
# Why this file exists:
# - the shared reporting pipeline relies on one stable, row-per-horizon table
#   surface that downstream sinks can reuse
# - that row-building logic used to live inside `builders.py`, but it is now
#   split out so the reporting package can grow without turning one file into a
#   giant mixed-responsibility module
# - keeping row construction here makes it easier to enrich prediction rows
#   later (for example, with additional row-level context columns) without
#   tangling those changes with grouped-table or narrative-text helpers
#
# Responsibility boundary:
# - align prediction batches with the original test dataloader batches
# - normalize metadata into one stable sample-level representation
# - emit the canonical flat row-per-horizon table used by the rest of the
#   reporting package
#
# What does *not* live here:
# - grouped metric table construction
# - dashboard/report text generation
# - TensorBoard sink logic
# - CSV/JSON/HTML export
#
# In other words, this module is the single source of truth for canonical
# prediction-row construction, and downstream helpers should consume the
# resulting DataFrame rather than reconstructing rows themselves.

from typing import Any, Sequence

import pandas as pd
import torch
from torch import Tensor

from evaluation import select_point_prediction
from observability.tensors import _as_metadata_lists

from reporting.types import TestDataloaderProvider


def _encoder_target_feature_index(datamodule: TestDataloaderProvider) -> int | None:
    """
    Best-effort lookup for the target column position inside encoder history.

    Context:
    the reporting layer wants to enrich exported prediction rows with the last
    observed glucose value from the encoder/history window. The Dataset already
    assembles encoder history in one packed `encoder_continuous` tensor, so the
    reporting path should read that existing batch contract rather than rebuild
    history windows from the dataframe.

    Why this helper is best-effort:
    `TestDataloaderProvider` is intentionally a small protocol that guarantees
    only `test_dataloader()`. The concrete AZT1DDataModule does expose
    `feature_groups`, but the reporting layer keeps the protocol narrow on
    purpose. This helper therefore probes for the richer runtime attributes when
    they are available and returns `None` when they are not.
    """
    feature_groups = getattr(datamodule, "feature_groups", None)
    if feature_groups is None:
        return None

    encoder_continuous = getattr(feature_groups, "encoder_continuous", None)
    target_column = getattr(feature_groups, "target_column", None)
    if encoder_continuous is None or target_column is None:
        return None

    try:
        return tuple(str(name) for name in encoder_continuous).index(str(target_column))
    except ValueError:
        return None


def _extract_last_observed_glucose(
    *,
    batch: dict[str, Any],
    sample_index: int,
    target_feature_index: int | None,
) -> float | None:
    """
    Read the final historical target value from one batched encoder window.

    Context:
    the canonical prediction-row table already includes future targets and model
    predictions. Adding the final observed glucose value from the encoder window
    gives downstream analysis a lightweight persistence-style reference point
    without changing model semantics or training behavior.

    Returned value:
    - float when the encoder history tensor is available and the target feature
      can be located reliably
    - `None` when the batch contract is missing that information or when the
      history window is empty
    """
    encoder_continuous = batch.get("encoder_continuous")
    if encoder_continuous is None:
        return None

    # Normalize the batch payload into a CPU tensor so row construction stays
    # device-agnostic and downstream reporting does not retain accelerator
    # memory through references to model batches.
    if isinstance(encoder_continuous, Tensor):
        encoder_continuous_cpu = encoder_continuous.detach().cpu()
    else:
        encoder_continuous_cpu = torch.as_tensor(encoder_continuous)

    # The canonical encoder-continuous layout is [batch, time, feature]. If the
    # tensor does not match that contract, reporting should fail soft and simply
    # omit the derived row field rather than guessing from an unexpected shape.
    if encoder_continuous_cpu.ndim != 3:
        return None

    if encoder_continuous_cpu.shape[1] == 0 or encoder_continuous_cpu.shape[2] == 0:
        return None

    # Preferred path: use the explicit target-column index from the DataModule's
    # feature-group contract. This keeps the derived reporting field aligned
    # with the same semantic column order used by the Dataset.
    if target_feature_index is not None:
        if target_feature_index < 0 or target_feature_index >= int(encoder_continuous_cpu.shape[2]):
            return None
        return float(encoder_continuous_cpu[sample_index, -1, target_feature_index].item())

    # Fallback path: the current schema contract appends the target history as
    # the final encoder continuous feature. Using the last feature position keeps
    # the reporting path useful even when the narrower protocol hides
    # `feature_groups`, while still matching the documented data/schema design.
    return float(encoder_continuous_cpu[sample_index, -1, -1].item())


def build_prediction_rows(
    *,
    datamodule: TestDataloaderProvider,
    predictions: Sequence[Tensor],
    quantiles: Sequence[float],
    sampling_interval_minutes: int,
) -> list[dict[str, Any]]:
    """
    Build the canonical flat row-per-horizon prediction table in memory.

    Context:
    the repository already relied on this denormalized table shape for CSV
    export and lightweight reports. The reporting package keeps that shape
    intact, but centralizes the row construction so downstream sinks do not
    each invent their own variant of the same logic.

    Important behavior:
    - aligns prediction batches with the original test dataloader batches
    - attaches metadata such as subject ID and decoder timing
    - emits one row per forecast horizon step for easy pandas use

    Why this shape is intentionally denormalized:
    a row-per-horizon table is not the most storage-efficient representation,
    but it is the most convenient shape for quick pandas aggregation, grouped
    plotting, CSV export, and manual inspection. That tradeoff is worthwhile in
    the reporting layer because clarity and reuse matter more here than raw
    compactness.
    """
    rows: list[dict[str, Any]] = []

    # Reuse the held-out test loader so prediction rows stay aligned with the
    # exact source metadata that produced them.
    test_loader = datamodule.test_dataloader()

    # Column names are derived deterministically from the configured quantiles.
    # Keeping this naming stable matters because downstream export/reporting
    # code expects to find quantile columns by their canonical names.
    quantile_columns = [f"pred_q{int(round(q * 100)):02d}" for q in quantiles]

    # Resolve the encoder target-column position once up front so per-row logic
    # can stay simple and avoid repeatedly re-reading DataModule metadata.
    target_feature_index = _encoder_target_feature_index(datamodule)

    for batch_index, (prediction_batch, batch) in enumerate(zip(predictions, test_loader)):
        # Post-run reporting works on CPU copies so downstream pandas/reporting
        # code stays device-agnostic and does not accidentally hold onto
        # accelerator memory.
        prediction_cpu = prediction_batch.detach().cpu()

        # Keep one representative point forecast column in addition to the
        # quantile columns. The repo's current default is the median.
        point_prediction_cpu = select_point_prediction(
            prediction_cpu,
            quantiles,
            quantile=0.5,
        )

        target = batch["target"]
        if isinstance(target, Tensor):
            target_cpu = target.detach().cpu()
        else:
            # Preserve compatibility with lighter or slightly older code paths
            # that may provide target-like values in non-tensor form.
            target_cpu = torch.as_tensor(target)

        # Normalize trailing singleton target dimensions into the canonical
        # [batch, horizon] table-friendly shape.
        if target_cpu.ndim == 3 and target_cpu.shape[-1] == 1:
            target_cpu = target_cpu.squeeze(-1)

        batch_size = int(prediction_cpu.shape[0])

        # Normalize metadata into predictable sample-level lists.
        metadata = _as_metadata_lists(batch["metadata"], batch_size)

        for sample_index in range(batch_size):
            subject_id = str(metadata.get("subject_id", ["unknown"])[sample_index])
            decoder_start = pd.Timestamp(
                str(
                    metadata.get("decoder_start", ["1970-01-01 00:00:00"])[sample_index]
                )
            )

            # Capture the final observed glucose value once per sample and then
            # copy it across all horizon rows for that forecast window. The
            # value is forecast-origin context, not horizon-specific truth.
            last_observed_glucose = _extract_last_observed_glucose(
                batch=batch,
                sample_index=sample_index,
                target_feature_index=target_feature_index,
            )

            for horizon_index in range(int(prediction_cpu.shape[1])):
                timestamp = decoder_start + pd.Timedelta(
                    minutes=sampling_interval_minutes * horizon_index
                )

                # `pd.Timestamp(...)` normally gives us a timestamp-like object
                # with `isoformat()`, but pandas typing can widen some datetime
                # expressions to include `NaTType`. Converting through `str(...)`
                # keeps the exported reporting row stable and avoids a false
                # static-analysis complaint without changing runtime behavior for
                # valid timestamps.
                row = {
                    "prediction_batch_index": batch_index,
                    "sample_index_within_batch": sample_index,
                    "subject_id": subject_id,
                    "decoder_start": str(metadata.get("decoder_start", [""])[sample_index]),
                    "decoder_end": str(metadata.get("decoder_end", [""])[sample_index]),
                    "timestamp": str(timestamp),
                    "horizon_index": horizon_index,
                    "target": float(target_cpu[sample_index, horizon_index].item()),
                    "last_observed_glucose": last_observed_glucose,
                }

                # Emit one scalar column per configured quantile so the table is
                # immediately usable for CSV export, grouped pandas summaries,
                # and lightweight plotting without an additional reshape step.
                for quantile_index, column_name in enumerate(quantile_columns):
                    row[column_name] = float(
                        prediction_cpu[
                            sample_index,
                            horizon_index,
                            quantile_index,
                        ].item()
                    )

                row["median_prediction"] = float(
                    point_prediction_cpu[sample_index, horizon_index].item()
                )

                # Residual sign convention stays stable across the repo:
                # prediction - target.
                row["residual"] = row["median_prediction"] - row["target"]

                # Expose simple interval width directly when at least two
                # quantiles are available.
                if len(quantile_columns) >= 2:
                    row["prediction_interval_width"] = (
                        row[quantile_columns[-1]] - row[quantile_columns[0]]
                    )

                rows.append(row)

    return rows
