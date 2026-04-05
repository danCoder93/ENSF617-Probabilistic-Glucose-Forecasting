from __future__ import annotations

# AI-assisted maintenance note:
# This module contains the repo's *runtime debugging callbacks*.
#
# The goal of these callbacks is not to replace normal training metrics such as
# loss, MAE, or RMSE. Those model-quality metrics already exist elsewhere in the
# codebase and answer questions like:
# - "is optimization progressing?"
# - "is validation improving?"
# - "how wide are the prediction intervals?"
#
# Instead, the callbacks in this file answer a different class of questions:
# - "what exact batch contract reached the model?"
# - "did all major branches receive gradient signal?"
# - "are activations finite, healthy, and non-collapsed?"
# - "is one fused-model branch becoming numerically dominant or silent?"
#
# Why this file exists separately from other observability modules:
# - `system_callbacks.py` focuses on run-level artifacts such as system
#   telemetry, model text, TensorBoard graphs, and torchview exports.
# - `parameter_callbacks.py` focuses on parameter histograms and scalar
#   summaries across the model.
# - `prediction_callbacks.py` focuses on human-facing prediction plots and
#   exported forecast examples.
# - this file focuses on *debugging-oriented numerical introspection* during
#   the live training loop.
#
# That separation matters because runtime debugging often becomes the noisiest
# part of observability. Keeping the policy here makes it easier to reason about
# what is being sampled, how often it is sampled, and whether a future change is
# duplicating an already existing signal elsewhere in the observability stack.
#
# Repo-specific motivation:
# This repository trains a fused glucose forecasting model with multiple major
# architectural branches. In a multi-branch setup, generic "the model is
# training" signals are not enough. A run can look superficially healthy while
# still hiding issues such as:
# - one TCN branch receiving effectively no gradient
# - the TFT path producing near-constant activations
# - a feature-group contract changing silently
# - fusion compressing the scale of incoming representations too aggressively
#
# These callbacks are therefore intentionally designed around *architecture-
# level trust questions* rather than generic MLOps dashboards.

import json
import logging
from typing import Any

import torch
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from config import ObservabilityConfig
from observability.logging_utils import _log_text_to_loggers, log_metrics_to_loggers
from observability.tensors import (
    _batch_semantic_overview,
    _flatten_tensor_output,
    _summarize_batch,
    _tensor_stats,
)

# ============================================================================
# Shared Debug Constants / Helpers
# ============================================================================
#
# This section contains shared names and small helper functions used by several
# callbacks in this file.
#
# Why keep these helpers local to `debug_callbacks.py` instead of pushing every
# tiny function into `tensors.py`?
# - `tensors.py` should stay focused on structure traversal and tensor summary
#   primitives that are broadly reusable across the observability layer.
# - the helpers below express *debugging policy specific to this file*, such as
#   how parameter names map onto architectural branch families.
# - keeping them nearby makes the callback logic easier to read because the
#   policy is visible where it is used.

_EXPECTED_BATCH_TENSOR_KEYS = (
    "static_categorical",
    "static_continuous",
    "known_categorical",
    "known_continuous",
    "observed_categorical",
    "observed_continuous",
    "target",
)

# `_EXPECTED_BATCH_TENSOR_KEYS` is intentionally treated as a *debugging
# contract* rather than a hard validation schema.
#
# Why not enforce it here?
# - the responsibility of these callbacks is observability, not control flow
# - hard validation belongs closer to config/data/model setup if we want
#   training to fail fast
# - here, we simply want to log evidence about what arrived at runtime
#
# This means the callbacks will still work if the batch structure changes. They
# will simply record which expected groups are missing and which additional keys
# appeared, which is often more useful during debugging than raising inside a
# callback.

_BRANCH_FAMILIES: dict[str, tuple[str, ...]] = {
    "tcn3": ("tcn3",),
    "tcn5": ("tcn5",),
    "tcn7": ("tcn7",),
    "tft": ("tft",),
    "grn": ("grn",),
    "fcn": ("fcn",),
}

# `_BRANCH_FAMILIES` defines the architectural aggregation level used for
# gradient and activation summaries.
#
# Why aggregate at this level?
# - leaf-layer logging is too noisy for humans to reason about quickly
# - whole-model logging is too coarse for a fused architecture
# - these names match the conceptual blocks the repo already uses when talking
#   about the model
#
# In practice, this lets us answer branch-aware questions such as:
# - are all three TCN branches active?
# - is TFT much larger or smaller than the TCN side?
# - is the GRN/fusion block collapsing or amplifying scale?
# - is the FCN head receiving healthy gradients from upstream fusion?


# Fusion-interpretation thresholds are kept local to this module because they
# are currently debugging heuristics rather than public configuration surfaces.
#
# Why define them once at module scope?
# - the same thresholds can be reused across gradient and activation summaries
# - keeping them together makes the intended interpretation easier to audit
# - future config wiring can lift these values out without changing callback
#   logic
#
# These values are intentionally conservative: they should surface obviously
# suspicious imbalance or collapse without overreacting to normal training
# variation.
_FUSION_DOMINANCE_WARN_RATIO = 10.0
_BRANCH_NEAR_DEAD_STD_THRESHOLD = 1e-6
_BRANCH_NEAR_DEAD_NEAR_ZERO_THRESHOLD = 0.95


def _module_family_for_parameter_name(parameter_name: str) -> str | None:
    """Map one parameter name to the high-level architectural family it belongs to.

    Purpose:
        Convert low-level parameter names such as
        `tft.encoder.variable_selection...weight` into a small, human-readable
        family label such as `tft`.

    Why this helper exists:
        Raw parameter names are too detailed for the kind of training-forensics
        summaries we want here. During debugging we usually care about whether a
        *branch* is receiving signal, not whether an individual sub-layer inside
        that branch had a slightly smaller norm than another.

    Matching policy:
        We intentionally use a simple prefix / substring policy rather than a
        more elaborate module graph walk. The naming in this repo is already
        descriptive enough that string-based grouping is stable, cheap, and easy
        to understand.

    Returns:
        The family name when the parameter appears to belong to a tracked
        branch, or `None` when the parameter does not match any known family.

    Why `None` is a valid result:
        Some parameters may live outside the main branches we care about, or
        the model may evolve later. Returning `None` lets the caller skip those
        parameters without treating them as an error.
    """
    for family_name, prefixes in _BRANCH_FAMILIES.items():
        if any(
            parameter_name == prefix
            or parameter_name.startswith(f"{prefix}.")
            or f".{prefix}." in parameter_name
            for prefix in prefixes
        ):
            return family_name
    return None



def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a stable numeric ratio for observability metrics.

    Purpose:
        Compute ratios such as "TFT gradient norm divided by mean TCN gradient
        norm" without ever emitting infinities or raising due to zero
        denominators.

    Why this matters in observability code:
        During early training or in partially inactive branches, a denominator
        being zero is not necessarily a bug. It can simply mean:
        - a family has no parameters yet matched by the grouping rule
        - the norm is truly zero for that sampled step
        - the branch has collapsed and that is exactly what we are trying to see

        Emitting `0.0` keeps the scalar stream easy to consume in TensorBoard,
        CSVs, and later automated analysis.

    Important interpretation note:
        A returned `0.0` does not always mean the true ratio is zero in a
        mathematical sense; sometimes it means the denominator was zero and the
        ratio was therefore intentionally clamped to a stable fallback value for
        logging purposes.
    """
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _tensor_energy_stats(tensor: Tensor) -> dict[str, float]:
    """Compute a small set of energy-style summaries for one tensor.

    Purpose:
        Complement `_tensor_stats(...)` with magnitude summaries that are more
        directly useful for branch-dominance interpretation. Mean absolute value
        and standard deviation are helpful, but they do not fully capture how
        much *activation energy* a branch is carrying.

    Returned metrics:
        `energy`
            Mean squared magnitude, i.e. `mean(x^2)`.

        `rms`
            Root-mean-square magnitude, i.e. `sqrt(mean(x^2))`. This is often a
            convenient scale summary because it stays in the same units as the
            original tensor while still emphasizing larger values.

    Why compute this here instead of in `tensors.py`?
        Energy-style summaries are currently only needed for the fusion-
        interpretation layer in this file. Keeping the helper local avoids
        widening the shared tensor-utils API until there is a broader need.
    """
    detached = tensor.detach()
    # Squared magnitude is used rather than variance because we care about the
    # total signal carried by the branch output, not only how much it varies
    # around its mean.
    energy = float(torch.mean(detached.float() ** 2).item())
    rms = float(torch.sqrt(torch.tensor(energy, dtype=torch.float32)).item())
    return {"energy": energy, "rms": rms}


def _branch_is_near_dead(stats: dict[str, float]) -> bool:
    """Heuristically flag a branch output as near-dead.

    Purpose:
        Surface branches that are producing overwhelmingly near-zero outputs
        with almost no variation. This is not a formal proof that the branch is
        mathematically useless, but it is a strong debugging hint that the
        branch may be inactive, over-masked, or numerically collapsed.

    Why this combines two conditions:
        - a high near-zero fraction alone can happen legitimately in sparse or
          masked representations
        - a tiny standard deviation alone can happen when a tensor is nearly
          constant but not necessarily small

        Requiring both conditions makes the warning more specific.
    """
    return (
        stats.get("near_zero_fraction", 0.0) >= _BRANCH_NEAR_DEAD_NEAR_ZERO_THRESHOLD
        and stats.get("std", 0.0) <= _BRANCH_NEAR_DEAD_STD_THRESHOLD
    )


# ============================================================================
# Batch Audit Callback
# ============================================================================
#
# The batch audit callback is the first line of defense against silent contract
# drift. Many model bugs are not caused by a mathematically wrong loss function
# or an optimizer bug; they are caused by the wrong tensors, wrong shapes, wrong
# groups, or unexpectedly empty feature families reaching the model while the
# code still "runs".
#
# This callback therefore logs a *small* number of representative batches in a
# form that is:
# - detailed enough for forensic debugging later
# - compact enough that text logs remain readable
# - structured enough that a later GenAI review can reason about the content


class BatchAuditCallback(Callback):
    """Log small, structured snapshots of the real runtime batch contract.

    What this callback is trying to reveal:
        - the true top-level batch schema seen during training/validation/test
        - whether expected tensor groups are present or absent
        - whether any groups are numerically suspicious (non-finite, constant,
          zero-width, heavily zeroed, etc.)
        - whether metadata is present and where it lives

    Why this callback exists even though generic batch summaries already exist:
        A generic recursive summary is useful, but for this repo we want a more
        contract-aware view that highlights the semantically important groups a
        fused time-series model expects.

    Why the callback only logs a few batches:
        The point is to capture evidence, not to create a transcript of every
        batch. Usually one or two examples per stage are enough to spot schema
        mismatches, empty tensors, or obvious preprocessing issues.
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        text_logger: logging.Logger | None = None,
    ) -> None:
        """Store batch-audit policy plus bounded per-stage logging state.

        Parameters:
            config:
                Observability settings controlling whether batch auditing is
                enabled and how many batches per stage should be logged.

            text_logger:
                Optional plain-text logger used to mirror the same payload into
                the run logs in addition to structured logger backends.

        Internal state:
            `_seen_counts` stores how many examples have already been logged for
            each stage. This is intentionally stage-specific because train, val,
            and test batches may differ subtly in ways that matter for
            debugging.
        """
        self.config = config
        self.text_logger = text_logger
        self._seen_counts = {"train": 0, "val": 0, "test": 0}

    def _build_batch_audit_payload(self, batch: Any) -> dict[str, Any]:
        """Construct one JSON-friendly payload describing the observed batch.

        Why the payload has two complementary views:
            `semantic_overview`
                A compact top-level contract summary optimized for quick human
                reading and later machine review.

            `raw_structure`
                A deeper recursive dump optimized for fidelity when someone
                needs to inspect the full nested structure.

        Why not only log the raw recursive summary?
            Because a large recursive dump is harder to scan quickly. The
            semantic overview acts as the "executive summary" of the batch while
            still preserving the deeper structure for follow-up inspection.
        """
        semantic_overview = _batch_semantic_overview(
            batch,
            expected_tensor_keys=_EXPECTED_BATCH_TENSOR_KEYS,
        )

        return {
            "semantic_overview": semantic_overview,
            "raw_structure": _summarize_batch(batch),
        }

    def _maybe_log_batch(self, trainer: Any, stage: str, batch: Any) -> None:
        """Emit one batch-audit payload if the stage-level cap allows it.

        Why the cap matters:
            Batch audits are intentionally text-heavy. Without a cap, they would
            quickly dominate logs and make later analysis harder instead of
            easier.

        Why this function does not sample by global step:
            For contract debugging, the earliest few batches are usually the
            most informative because they show the *initial* runtime shape of
            the pipeline. A simple per-stage cap is easier to reason about than
            periodic sampling.
        """
        if not self.config.enable_batch_audit:
            return

        if self._seen_counts[stage] >= self.config.batch_audit_limit:
            return

        payload = self._build_batch_audit_payload(batch)
        summary = json.dumps(payload, indent=2)

        # We write to both the plain-text logger and the experiment loggers so
        # the evidence remains available even if one surface is disabled or not
        # preserved in downstream artifacts.
        if self.text_logger is not None:
            self.text_logger.info("%s batch audit\n%s", stage, summary)

        _log_text_to_loggers(trainer, f"batch_audit/{stage}", summary)
        self._seen_counts[stage] += 1

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record an early training batch summary when batch auditing is enabled.

        Why `on_*_batch_start` is used instead of `on_*_batch_end`:
            We want to capture the raw incoming contract before the model has a
            chance to mutate anything or before any later hooks obscure what the
            batch originally looked like.
        """
        del pl_module, batch_idx
        self._maybe_log_batch(trainer, "train", batch)

    def on_validation_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record an early validation batch summary when batch auditing is enabled.

        Why validation is logged separately from training:
            In many pipelines, validation can differ subtly from training due to
            transforms, target availability, masking, or dataloader settings.
            Logging both makes those differences visible instead of assumed.
        """
        del pl_module, batch_idx, dataloader_idx
        self._maybe_log_batch(trainer, "val", batch)

    def on_test_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record an early test batch summary when batch auditing is enabled.

        Why test-stage audits are useful:
            Test-time pipelines are often the least frequently exercised path.
            A small contract snapshot here helps catch stage-specific drift that
            training/validation logs would otherwise miss.
        """
        del pl_module, batch_idx, dataloader_idx
        self._maybe_log_batch(trainer, "test", batch)


# ============================================================================
# Gradient Stats Callback
# ============================================================================
#
# The gradient callback focuses on backpropagated signal health.
#
# There are already other parameter-oriented callbacks elsewhere in the repo,
# including scalar parameter summaries and histogram logging. This callback does
# *not* try to replace those. Its specific job is to answer branch-aware
# questions at training time:
# - are gradients present and finite?
# - how large are they globally?
# - how large are they by major branch family?
# - is a branch effectively silent relative to its parameter magnitude?
#
# This branch-aware view is particularly important for fused architectures,
# because whole-model gradient norms can look healthy even while one branch is
# receiving almost no useful signal.


class GradientStatsCallback(Callback):
    """Emit sampled gradient and parameter health summaries during training.

    Why gradients and parameters are logged together:
        Looking at gradients alone can be misleading. For example, a tiny
        gradient may be acceptable for a tiny parameter family but suspicious
        for a large one. Logging both lets us compute useful scale-aware ratios.

    Why the sampling is periodic rather than per-step:
        Full gradient inspection is not free. Periodic sampling keeps the
        callback lightweight enough for normal debugging runs while still
        collecting enough evidence to diagnose branch collapse or instability.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        """Store the debug sampling policy for gradient diagnostics.

        The callback intentionally keeps very little state because gradients are
        sampled directly from the live model after each backward pass. This
        makes the measurements easy to reason about and avoids stale caches.
        """
        self.config = config

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        """Summarize gradient and parameter health immediately after backward.

        Why this specific hook is chosen:
            This is the earliest point where the complete gradient signal exists
            on parameters, but before optimizer stepping or zeroing can alter
            it. That makes it the cleanest location for gradient introspection.

        What this callback computes:
            - whole-model gradient norm and max absolute gradient
            - whole-model parameter norm and max absolute parameter value
            - counts of parameters with non-finite gradients or values
            - the same style of summaries aggregated by major architectural
              family (`tcn3`, `tcn5`, `tcn7`, `tft`, `grn`, `fcn`)
            - a few cross-family ratios to make imbalance more explicit
        """
        if not self.config.enable_gradient_stats:
            return

        # Lightning performs validation sanity checking before real training in
        # some configurations. We skip those passes so the logged gradient time
        # series represents actual training rather than pre-training probes.
        if trainer.sanity_checking:
            return

        # Sampling every debug step keeps this callback affordable while still
        # producing enough evidence for later forensic analysis.
        if trainer.global_step % self.config.debug_every_n_steps != 0:
            return

        # Whole-model accumulators. These answer "is backprop globally sane?".
        total_norm_sq = 0.0
        max_abs = 0.0
        grad_parameter_count = 0
        nonfinite_grad_parameters = 0

        # Parameter-state accumulators. These answer "what scale do the
        # parameters themselves live at?" which helps interpret gradient scale.
        parameter_norm_sq = 0.0
        parameter_max_abs = 0.0
        nonfinite_parameters = 0

        # Family-level accumulators. These are the most repo-specific part of
        # the callback because they let us inspect each major branch as a unit.
        family_stats: dict[str, dict[str, float]] = {
            family_name: {
                "grad_norm_sq": 0.0,
                "grad_max_abs": 0.0,
                "grad_parameter_count": 0.0,
                "nonfinite_grad_parameters": 0.0,
                "parameter_norm_sq": 0.0,
                "parameter_max_abs": 0.0,
                "parameter_count": 0.0,
                "nonfinite_parameters": 0.0,
            }
            for family_name in _BRANCH_FAMILIES
        }

        for parameter_name, parameter in pl_module.named_parameters():
            detached_parameter = parameter.detach()

            # Parameter norm summaries are always computed, even if a gradient is
            # currently missing for that parameter, because the parameter scale
            # is still useful context when reasoning about branch health.
            parameter_norm = float(torch.norm(detached_parameter).item())
            parameter_norm_sq += parameter_norm * parameter_norm
            parameter_max_abs = max(
                parameter_max_abs,
                float(torch.max(torch.abs(detached_parameter)).item()),
            )

            if not torch.isfinite(detached_parameter).all():
                nonfinite_parameters += 1

            family_name = _module_family_for_parameter_name(parameter_name)
            if family_name is not None:
                family_stats[family_name]["parameter_count"] += 1.0
                family_stats[family_name]["parameter_norm_sq"] += (
                    parameter_norm * parameter_norm
                )
                family_stats[family_name]["parameter_max_abs"] = max(
                    family_stats[family_name]["parameter_max_abs"],
                    float(torch.max(torch.abs(detached_parameter)).item()),
                )
                if not torch.isfinite(detached_parameter).all():
                    family_stats[family_name]["nonfinite_parameters"] += 1.0

            # A missing gradient is not automatically a bug. It can happen for
            # frozen parameters, unused parameters, or branches not participating
            # in the current backward path. We therefore skip cleanly and let the
            # aggregate counts reveal whether that pattern is suspicious.
            if parameter.grad is None:
                continue

            grad = parameter.grad.detach()
            grad_parameter_count += 1

            if not torch.isfinite(grad).all():
                nonfinite_grad_parameters += 1

            grad_norm = float(torch.norm(grad).item())
            total_norm_sq += grad_norm * grad_norm
            max_abs = max(max_abs, float(torch.max(torch.abs(grad)).item()))

            if family_name is not None:
                family_stats[family_name]["grad_parameter_count"] += 1.0
                family_stats[family_name]["grad_norm_sq"] += grad_norm * grad_norm
                family_stats[family_name]["grad_max_abs"] = max(
                    family_stats[family_name]["grad_max_abs"],
                    float(torch.max(torch.abs(grad)).item()),
                )
                if not torch.isfinite(grad).all():
                    family_stats[family_name]["nonfinite_grad_parameters"] += 1.0

        metrics = {
            "debug/grad_total_norm": total_norm_sq ** 0.5,
            "debug/grad_max_abs": max_abs,
            "debug/nonfinite_grad_parameters": float(nonfinite_grad_parameters),
            "debug/grad_parameter_count": float(grad_parameter_count),
            "debug/parameter_total_norm": parameter_norm_sq ** 0.5,
            "debug/parameter_max_abs": parameter_max_abs,
            "debug/nonfinite_parameters": float(nonfinite_parameters),
        }

        # Family metrics expose branch-level behavior that global metrics can
        # hide. For example, a healthy whole-model gradient norm could still be
        # consistent with one branch being almost silent while another dominates.
        for family_name, stats in family_stats.items():
            family_grad_total_norm = stats["grad_norm_sq"] ** 0.5
            family_parameter_total_norm = stats["parameter_norm_sq"] ** 0.5

            metrics.update(
                {
                    f"debug/{family_name}_grad_total_norm": family_grad_total_norm,
                    f"debug/{family_name}_grad_max_abs": stats["grad_max_abs"],
                    f"debug/{family_name}_grad_parameter_count": stats[
                        "grad_parameter_count"
                    ],
                    f"debug/{family_name}_nonfinite_grad_parameters": stats[
                        "nonfinite_grad_parameters"
                    ],
                    f"debug/{family_name}_parameter_total_norm": (
                        family_parameter_total_norm
                    ),
                    f"debug/{family_name}_parameter_max_abs": stats[
                        "parameter_max_abs"
                    ],
                    f"debug/{family_name}_parameter_count": stats["parameter_count"],
                    f"debug/{family_name}_nonfinite_parameters": stats[
                        "nonfinite_parameters"
                    ],
                    f"debug/{family_name}_grad_to_parameter_norm_ratio": _safe_ratio(
                        family_grad_total_norm,
                        family_parameter_total_norm,
                    ),
                }
            )

        # The TCN side has three branches, so a few aggregated TCN metrics make
        # it easier to compare the TCN ensemble against the TFT/fusion side.
        tcn_family_grad_norms = [
            metrics["debug/tcn3_grad_total_norm"],
            metrics["debug/tcn5_grad_total_norm"],
            metrics["debug/tcn7_grad_total_norm"],
        ]
        metrics["debug/tcn_branch_grad_norm_mean"] = (
            sum(tcn_family_grad_norms) / float(len(tcn_family_grad_norms))
        )
        metrics["debug/tcn_branch_grad_norm_max"] = max(tcn_family_grad_norms)
        metrics["debug/tcn_branch_grad_norm_min"] = min(tcn_family_grad_norms)

        # Cross-family ratios help turn raw magnitudes into more interpretable
        # comparisons. They are not proofs of correctness; they are compact
        # signals that make branch imbalance easier to spot later.
        metrics["debug/tft_to_tcn_mean_grad_norm_ratio"] = _safe_ratio(
            metrics["debug/tft_grad_total_norm"],
            metrics["debug/tcn_branch_grad_norm_mean"],
        )
        metrics["debug/grn_to_tft_grad_norm_ratio"] = _safe_ratio(
            metrics["debug/grn_grad_total_norm"],
            metrics["debug/tft_grad_total_norm"],
        )
        metrics["debug/fcn_to_grn_grad_norm_ratio"] = _safe_ratio(
            metrics["debug/fcn_grad_total_norm"],
            metrics["debug/grn_grad_total_norm"],
        )

        # The fusion-interpretation layer complements the raw branch norms with
        # compact imbalance signals. These metrics are designed to answer
        # questions such as: "is TFT dominating the TCN ensemble?" and "is one
        # major branch receiving much more gradient signal than the others?"
        gradient_dominance_families = ("tcn3", "tcn5", "tcn7", "tft")
        gradient_dominance_values = [
            metrics[f"debug/{family_name}_grad_total_norm"]
            for family_name in gradient_dominance_families
        ]
        metrics["debug/tft_to_tcn_max_grad_norm_ratio"] = _safe_ratio(
            metrics["debug/tft_grad_total_norm"],
            metrics["debug/tcn_branch_grad_norm_max"],
        )
        metrics["debug/tcn_branch_grad_norm_spread_ratio"] = _safe_ratio(
            metrics["debug/tcn_branch_grad_norm_max"],
            metrics["debug/tcn_branch_grad_norm_min"],
        )
        metrics["debug/fusion_branch_grad_dominance_ratio"] = _safe_ratio(
            max(gradient_dominance_values),
            min(gradient_dominance_values),
        )
        metrics["debug/fusion_branch_grad_dominance_warning"] = (
            1.0
            if metrics["debug/fusion_branch_grad_dominance_ratio"]
            >= _FUSION_DOMINANCE_WARN_RATIO
            else 0.0
        )

        log_metrics_to_loggers(trainer, metrics, step=trainer.global_step)


# ============================================================================
# Activation Stats Callback
# ============================================================================
#
# The activation callback focuses on forward-path numerical behavior at the
# level of the main architectural blocks.
#
# Important design choice:
# We do *not* hook every leaf layer. That would create too much noise and make
# the logs harder to interpret. Instead, we hook the modules the repo already
# uses as meaningful architectural units: the three TCN branches, the TFT path,
# the GRN/fusion block, and the final FCN head.
#
# This keeps the activation record aligned with the questions we actually care
# about later:
# - are branch outputs finite?
# - are they near-zero or constant?
# - is one side of the fusion much larger than the other?
# - is the head collapsing or exploding relative to upstream fusion?


class ActivationStatsCallback(Callback):
    """Sample forward-activation summaries from the model's major blocks.

    Why activation statistics matter in addition to gradient statistics:
        Gradients tell us whether a branch is receiving learning signal.
        Activations tell us what the branch is actually producing on the forward
        path. A branch can have gradients but still produce numerically weak,
        nearly constant, or non-finite outputs.

    Why outputs are staged and flushed later:
        Forward hooks can fire many times and at awkward points inside the
        model's execution. Staging the metrics first and logging once at batch
        end keeps the logging behavior predictable and easier to analyze.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        """Initialize activation-stat policy plus per-run hook state.

        Internal state:
            `_handles`
                Stores the registered forward-hook handles so they can be
                removed cleanly at fit end.

            `_pending_metrics`
                Stores scalar metrics collected during the current sampled
                forward pass and flushed once at batch end.

            `_latest_module_stats`
                Stores per-module summaries for the current sampled step so that
                cross-module comparison metrics can be derived before flushing.
        """
        self.config = config
        self._handles: list[Any] = []
        self._pending_metrics: dict[str, float] = {}
        self._latest_module_stats: dict[str, dict[str, float]] = {}

    def _stage_module_stats(self, module_name: str, tensor: Tensor) -> None:
        """Compute and cache activation summaries for one hooked module output.

        Why this helper exists:
            Forward hooks should stay lightweight and consistent. Centralizing
            the staging logic here ensures every hooked module produces the same
            metric set and derived indicators.

        What is staged:
            - the base scalar summary from `_tensor_stats(...)`
            - a deadness indicator when outputs are overwhelmingly near-zero and
              have negligible variance
            - a non-finite indicator when the output is not fully finite

        Important note on the indicators:
            These are intentionally heuristic flags, not formal proofs of a bug.
            Their job is to make suspicious steps easy to find later.
        """
        stats = dict(_tensor_stats(tensor))

        # Energy-style summaries make branch comparisons more interpretable for
        # fused models because they describe how much signal a branch is
        # carrying, not just how dispersed or non-zero it is.
        stats.update(_tensor_energy_stats(tensor))
        self._latest_module_stats[module_name] = stats

        for stat_name, value in stats.items():
            self._pending_metrics[f"activation/{module_name}_{stat_name}"] = value

        # A high near-zero fraction alone is not enough to call an activation
        # dead, because some layers or masks may legitimately produce many small
        # values. We therefore also require an extremely small standard
        # deviation before raising the deadness indicator. The helper keeps the
        # policy shared with the later branch-level fusion summary.
        self._pending_metrics[f"activation/{module_name}_deadness_indicator"] = (
            1.0 if _branch_is_near_dead(stats) else 0.0
        )

        # Non-finite outputs are always worth surfacing directly because they
        # are strong evidence of numerical instability, broken preprocessing, or
        # an invalid operation somewhere upstream.
        self._pending_metrics[f"activation/{module_name}_nonfinite_indicator"] = (
            1.0 if stats["finite_fraction"] < 1.0 else 0.0
        )

    def _stage_branch_comparison_metrics(self) -> None:
        """Derive architecture-level comparison metrics from staged activations.

        Purpose:
            Turn per-module summaries into a few higher-level comparative
            signals that are easier to interpret when reasoning about fusion.

        Why these comparisons are valuable:
            Absolute stats answer questions like "is TFT finite?".
            Comparative stats answer questions like:
            - "is TFT much stronger than the TCN side?"
            - "is GRN shrinking the representation too aggressively?"
            - "is the FCN head operating at a very different scale than fusion?"
            - "is one branch becoming numerically dominant or effectively dead?"

        Why missing modules simply skip comparisons:
            For observability, missing evidence should not be fabricated into
            placeholder numbers that might be misread as real measurements.
        """
        required_tcn_modules = ("tcn3", "tcn5", "tcn7")
        if all(name in self._latest_module_stats for name in required_tcn_modules):
            tcn_abs_means = [
                self._latest_module_stats[name]["abs_mean"]
                for name in required_tcn_modules
            ]
            tcn_stds = [
                self._latest_module_stats[name]["std"]
                for name in required_tcn_modules
            ]
            tcn_energies = [
                self._latest_module_stats[name]["energy"]
                for name in required_tcn_modules
            ]
            tcn_rms_values = [
                self._latest_module_stats[name]["rms"]
                for name in required_tcn_modules
            ]

            self._pending_metrics["activation/tcn_abs_mean_mean"] = (
                sum(tcn_abs_means) / float(len(tcn_abs_means))
            )
            self._pending_metrics["activation/tcn_abs_mean_max"] = max(tcn_abs_means)
            self._pending_metrics["activation/tcn_abs_mean_min"] = min(tcn_abs_means)
            self._pending_metrics["activation/tcn_std_mean"] = (
                sum(tcn_stds) / float(len(tcn_stds))
            )
            self._pending_metrics["activation/tcn_std_max"] = max(tcn_stds)
            self._pending_metrics["activation/tcn_std_min"] = min(tcn_stds)

            # Energy and RMS summaries provide a stronger signal for fusion
            # interpretation than abs-mean alone because they more directly
            # reflect the magnitude carried by each branch representation.
            self._pending_metrics["activation/tcn_energy_mean"] = (
                sum(tcn_energies) / float(len(tcn_energies))
            )
            self._pending_metrics["activation/tcn_energy_max"] = max(tcn_energies)
            self._pending_metrics["activation/tcn_energy_min"] = min(tcn_energies)
            self._pending_metrics["activation/tcn_rms_mean"] = (
                sum(tcn_rms_values) / float(len(tcn_rms_values))
            )
            self._pending_metrics["activation/tcn_rms_max"] = max(tcn_rms_values)
            self._pending_metrics["activation/tcn_rms_min"] = min(tcn_rms_values)
            self._pending_metrics["activation/tcn_energy_spread_ratio"] = _safe_ratio(
                self._pending_metrics["activation/tcn_energy_max"],
                self._pending_metrics["activation/tcn_energy_min"],
            )

        if (
            "tft" in self._latest_module_stats
            and "activation/tcn_abs_mean_mean" in self._pending_metrics
        ):
            self._pending_metrics["activation/tft_to_tcn_abs_mean_ratio"] = _safe_ratio(
                self._latest_module_stats["tft"]["abs_mean"],
                self._pending_metrics["activation/tcn_abs_mean_mean"],
            )
            self._pending_metrics["activation/tft_to_tcn_std_ratio"] = _safe_ratio(
                self._latest_module_stats["tft"]["std"],
                self._pending_metrics["activation/tcn_std_mean"],
            )
            self._pending_metrics["activation/tft_to_tcn_energy_ratio"] = _safe_ratio(
                self._latest_module_stats["tft"]["energy"],
                self._pending_metrics["activation/tcn_energy_mean"],
            )
            self._pending_metrics["activation/tft_to_tcn_rms_ratio"] = _safe_ratio(
                self._latest_module_stats["tft"]["rms"],
                self._pending_metrics["activation/tcn_rms_mean"],
            )

        if "grn" in self._latest_module_stats and "tft" in self._latest_module_stats:
            self._pending_metrics["activation/grn_to_tft_abs_mean_ratio"] = _safe_ratio(
                self._latest_module_stats["grn"]["abs_mean"],
                self._latest_module_stats["tft"]["abs_mean"],
            )
            self._pending_metrics["activation/grn_to_tft_std_ratio"] = _safe_ratio(
                self._latest_module_stats["grn"]["std"],
                self._latest_module_stats["tft"]["std"],
            )
            self._pending_metrics["activation/grn_to_tft_energy_ratio"] = _safe_ratio(
                self._latest_module_stats["grn"]["energy"],
                self._latest_module_stats["tft"]["energy"],
            )
            self._pending_metrics["activation/grn_to_tft_rms_ratio"] = _safe_ratio(
                self._latest_module_stats["grn"]["rms"],
                self._latest_module_stats["tft"]["rms"],
            )

        if "fcn" in self._latest_module_stats and "grn" in self._latest_module_stats:
            self._pending_metrics["activation/fcn_to_grn_abs_mean_ratio"] = _safe_ratio(
                self._latest_module_stats["fcn"]["abs_mean"],
                self._latest_module_stats["grn"]["abs_mean"],
            )
            self._pending_metrics["activation/fcn_to_grn_std_ratio"] = _safe_ratio(
                self._latest_module_stats["fcn"]["std"],
                self._latest_module_stats["grn"]["std"],
            )
            self._pending_metrics["activation/fcn_to_grn_energy_ratio"] = _safe_ratio(
                self._latest_module_stats["fcn"]["energy"],
                self._latest_module_stats["grn"]["energy"],
            )
            self._pending_metrics["activation/fcn_to_grn_rms_ratio"] = _safe_ratio(
                self._latest_module_stats["fcn"]["rms"],
                self._latest_module_stats["grn"]["rms"],
            )

        # Branch-deadness summaries are computed at the architectural-family
        # level so that later log review can answer questions like "did one TCN
        # branch collapse even though the rest of the model looked healthy?"
        dead_branch_count = 0.0
        for module_name in ("tcn3", "tcn5", "tcn7", "tft", "grn", "fcn"):
            if module_name not in self._latest_module_stats:
                continue

            is_near_dead = _branch_is_near_dead(self._latest_module_stats[module_name])
            self._pending_metrics[f"activation/{module_name}_near_dead_indicator"] = (
                1.0 if is_near_dead else 0.0
            )
            dead_branch_count += 1.0 if is_near_dead else 0.0

        self._pending_metrics["activation/near_dead_branch_count"] = dead_branch_count
        self._pending_metrics["activation/near_dead_branch_warning"] = (
            1.0 if dead_branch_count > 0.0 else 0.0
        )

        # Dominance metrics compress a potentially large set of branch magnitudes
        # into compact signals that are easy to scan in TensorBoard or CSV
        # exports. We intentionally use energy-based comparisons here because
        # they reflect branch signal strength more directly than mean alone.
        dominance_modules = tuple(
            module_name
            for module_name in ("tcn3", "tcn5", "tcn7", "tft")
            if module_name in self._latest_module_stats
        )
        if dominance_modules:
            dominance_energies = [
                self._latest_module_stats[module_name]["energy"]
                for module_name in dominance_modules
            ]
            self._pending_metrics["activation/fusion_branch_energy_max"] = max(
                dominance_energies
            )
            self._pending_metrics["activation/fusion_branch_energy_min"] = min(
                dominance_energies
            )
            self._pending_metrics["activation/fusion_branch_dominance_ratio"] = _safe_ratio(
                self._pending_metrics["activation/fusion_branch_energy_max"],
                self._pending_metrics["activation/fusion_branch_energy_min"],
            )
            self._pending_metrics["activation/fusion_branch_dominance_warning"] = (
                1.0
                if self._pending_metrics["activation/fusion_branch_dominance_ratio"]
                >= _FUSION_DOMINANCE_WARN_RATIO
                else 0.0
            )

        if "tft" in self._latest_module_stats and all(
            name in self._latest_module_stats for name in required_tcn_modules
        ):
            tcn_branch_energies = [
                self._latest_module_stats[name]["energy"]
                for name in required_tcn_modules
            ]
            self._pending_metrics["activation/tft_to_tcn_max_energy_ratio"] = _safe_ratio(
                self._latest_module_stats["tft"]["energy"],
                max(tcn_branch_energies),
            )
            self._pending_metrics["activation/tft_to_tcn_min_energy_ratio"] = _safe_ratio(
                self._latest_module_stats["tft"]["energy"],
                min(tcn_branch_energies),
            )

    def _register_hook(self, pl_module: Any, module_name: str) -> None:
        """Attach one forward hook to a named high-level module when it exists.

        Why this helper exists:
            Hook registration is repetitive and a common source of subtle bugs
            if duplicated inline. Centralizing the logic here makes it easier to
            keep registration behavior consistent across all tracked modules.

        Why missing modules are skipped quietly:
            The observability layer should adapt gracefully if the model changes
            or if some branches are absent in a future variant. Missing hooks
            are interesting information, but they are not callback-fatal.
        """
        module = getattr(pl_module, module_name, None)
        if module is None:
            return

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            # The forward hook runs inside the model execution path, so it should
            # do the minimum necessary work: extract one representative tensor
            # and stage summarized metrics. All logger interaction is deferred.
            trainer = getattr(pl_module, "_trainer", None)
            if trainer is None or trainer.sanity_checking or not pl_module.training:
                return

            # Periodic sampling avoids turning forward hooks into a noticeable
            # runtime cost during long training runs.
            if trainer.global_step % self.config.debug_every_n_steps != 0:
                return

            tensor = _flatten_tensor_output(output)
            if tensor is None:
                return

            self._stage_module_stats(module_name, tensor)

        self._handles.append(module.register_forward_hook(hook))

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Register forward hooks on the main fused-model blocks at fit start.

        Why `on_fit_start` is the right time:
            By this point the model is constructed, attached to the trainer, and
            about to execute real batches. Registering here avoids hook setup in
            constructors while still ensuring the full training run is covered.
        """
        del trainer

        if not self.config.enable_activation_stats:
            return

        for module_name in ("tcn3", "tcn5", "tcn7", "tft", "grn", "fcn"):
            self._register_hook(pl_module, module_name)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Flush staged activation metrics once the sampled train batch finishes.

        Why flushing happens here instead of inside the forward hook:
            - all modules for the batch have had a chance to stage their stats
            - a single shared global step is used for the whole activation set
            - cross-module comparisons can be derived before logging
            - the forward path itself stays as light as possible
        """
        del pl_module, outputs, batch, batch_idx

        if not self._pending_metrics:
            return

        self._stage_branch_comparison_metrics()

        log_metrics_to_loggers(trainer, self._pending_metrics, step=trainer.global_step)

        # Reset per-step caches immediately after flushing so metrics from one
        # sampled step cannot leak into the next.
        self._pending_metrics = {}
        self._latest_module_stats = {}

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Remove registered activation hooks and clear any remaining state.

        Why explicit cleanup matters:
            Forward hooks hold references and can lead to confusing duplicate
            logging if a module instance is reused or if future changes alter the
            trainer lifecycle. Cleaning them up keeps callback behavior explicit.
        """
        del trainer, pl_module

        for handle in self._handles:
            handle.remove()

        self._handles = []
        self._pending_metrics = {}
        self._latest_module_stats = {}
