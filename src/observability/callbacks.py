from __future__ import annotations

# AI-assisted maintenance note:
# This module is the public observability callback assembly facade for the repo.
#
# Why this file exists even though the concrete callbacks now live in separate
# modules:
# - callers and tests can continue importing from `observability.callbacks`
#   without needing to know where each implementation class physically lives
# - trainer construction code stays small because all observability wiring is
#   concentrated in one place
# - this file becomes the single authoritative answer to the question:
#   "given an `ObservabilityConfig`, which callbacks will actually be active?"
#
# Why this matters in the current repo:
# The project now has several distinct observability layers:
# - system/run-level instrumentation
# - debug-oriented numerical-health instrumentation
# - parameter-distribution instrumentation
# - prediction-semantic instrumentation
# - qualitative prediction visualization
#
# If callback assembly were scattered across training code, it would become much
# harder to reason about:
# - which flags enable which behavior
# - whether two callbacks overlap or duplicate each other
# - what order callbacks are attached in
# - whether a new callback belongs in the run at all
#
# Design boundary:
# - this file should assemble callbacks
# - this file should explain callback selection policy
# - this file should keep the public import surface stable
#
# This file should *not*:
# - implement the actual callback logic
# - own tensor summarization helpers
# - own filesystem/reporting logic
# - own model architecture or data-module behavior
#
# In other words, the job of this file is orchestration, not instrumentation.
#
# AI generation disclaimer:
# This file was generated and refined with AI assistance, then reviewed and
# adapted for this repository's observability structure. The intent is to keep
# the code readable, explicit, and maintainable, but maintainers should still
# review callback selection policy carefully whenever new observability features
# are added or existing config flags change.

import logging

from pytorch_lightning.callbacks import (
    Callback,
    DeviceStatsMonitor,
    LearningRateMonitor,
    RichProgressBar,
)

from config import ObservabilityConfig
from observability.debug_callbacks import (
    ActivationStatsCallback,
    BatchAuditCallback,
    GradientStatsCallback,
)
from observability.parameter_callbacks import (
    ParameterHistogramCallback,
    ParameterScalarTelemetryCallback,
)
from observability.prediction_callbacks import (
    PredictionFigureCallback,
    PredictionSanityCallback,
)
from observability.system_callbacks import (
    ModelTensorBoardCallback,
    SystemTelemetryCallback,
)
from observability.utils import _has_module

# ============================================================================
# Callback Assembly
# ============================================================================
#
# This function is intentionally the single translation layer between
# `ObservabilityConfig` and the concrete callback objects handed to the
# Lightning trainer.
#
# Why a single assembly function is preferable:
# - it prevents trainer setup code from becoming a long sequence of ad hoc
#   feature-flag checks
# - it gives future maintainers one place to inspect when they want to confirm
#   what an observability mode actually does
# - it makes it much easier to compare "desired observability policy" against
#   "actual attached callbacks"
# - it reduces the chance that one environment path (local, Slurm, notebook,
#   debug run, etc.) quietly assembles a different callback set than another
#
# Ordering policy:
# The exact callback order is not mission-critical in every case, but we still
# keep it intentional because order affects how readable the callback stack is
# to humans and, in a few cases, when a callback gets a chance to observe state.
#
# The rough order below is:
# 1. run/model-surface callbacks
#    These explain the overall run, model structure, and system context.
# 2. built-in Lightning telemetry / UX callbacks
#    These provide familiar trainer-side monitoring conveniences.
# 3. deeper debug callbacks
#    These inspect batches, gradients, activations, and numerical health.
# 4. parameter-state callbacks
#    These provide more detailed parameter/gradient distribution views.
# 5. prediction-semantic callbacks
#    These inspect whether produced forecasts themselves look numerically sane.
# 6. qualitative prediction callbacks
#    These help humans visually inspect forecast behavior and interval shape.
#
# That ordering is chosen for human reasoning first, with technical correctness
# as the guardrail. The goal is that when someone reads the resulting callback
# list, it tells a coherent story about the run.


def build_observability_callbacks(
    config: ObservabilityConfig,
    *,
    text_logger: logging.Logger | None = None,
) -> list[Callback]:
    """Translate `ObservabilityConfig` flags into the Lightning callback list.

    Purpose:
        Build the exact callback stack that should be attached to the Trainer
        for the current run.

    Why this function exists:
        The repo has grown beyond a tiny single-callback setup. At this point,
        observability is a subsystem with multiple responsibilities and multiple
        optional features. Centralizing callback construction here prevents that
        complexity from leaking into training entrypoints.

    Parameters:
        config:
            The single source of truth for whether each observability feature is
            enabled and, indirectly, for how noisy or lightweight the resulting
            callback stack should be.

        text_logger:
            Optional plain-text logger used by callbacks that emit structured
            JSON/text summaries in addition to scalar metrics. Passing it here
            keeps logging-surface dependency injection centralized instead of
            forcing each caller to wire callbacks individually.

    Returns:
        A list of Lightning `Callback` instances in a deliberate and documented
        order.

    Design note:
        This function intentionally performs *selection and assembly only*.
        It does not mutate config, infer new policy, or attempt to resolve
        conflicts between callbacks beyond choosing whether to include them.

    Practical maintenance rule:
        When adding a new observability callback to the repo, this should be one
        of the first files updated. If it is not wired here, it is effectively
        not part of the supported observability surface.
    """
    # Keep callback assembly centralized so trainer construction elsewhere in
    # the repo remains easy to read. The caller should not need to remember
    # every individual callback class or the order in which they should be
    # attached.
    callbacks: list[Callback] = []

    # ---------------------------------------------------------------------
    # 1. Model / run-surface observability
    # ---------------------------------------------------------------------
    #
    # These callbacks describe the overall run environment and model shape:
    # - model text
    # - TensorBoard graph logging
    # - torchview export
    #
    # We place the model-structure callback first because it establishes the
    # "what model is this run actually using?" context before deeper debugging
    # callbacks start describing internal behavior.
    #
    # Why the three config flags share one callback:
    # `ModelTensorBoardCallback` owns a cluster of related responsibilities
    # around model visibility. Grouping them behind one callback avoids
    # splitting tightly related model-surface logic across multiple tiny
    # callback classes without a strong architectural reason.
    if config.enable_model_graph or config.enable_model_text or config.enable_torchview:
        callbacks.append(ModelTensorBoardCallback(config, text_logger=text_logger))

    # ---------------------------------------------------------------------
    # 2. Built-in Lightning monitoring / quality-of-life callbacks
    # ---------------------------------------------------------------------
    #
    # These are not custom repo callbacks, but they still form part of the
    # effective observability surface presented to the user during training.
    #
    # They live here rather than directly in trainer construction so that:
    # - all run-monitoring features are selected in one place
    # - "observability mode" changes can reason about both custom and built-in
    #   callback surfaces together
    # - callers do not need to duplicate Lightning-specific callback wiring
    #
    # Learning-rate monitoring is a basic but still high-value signal for
    # optimizer sanity, especially when later debugging why training dynamics
    # look odd even though the model code itself appears fine.
    if config.enable_learning_rate_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    # DeviceStatsMonitor is Lightning's built-in lightweight system/device
    # monitor. We keep it separate from the repo's custom system telemetry
    # callback because:
    # - it already provides a familiar built-in surface for supported backends
    # - the custom telemetry callback also writes CSV/text artifacts and uses
    #   repo-specific logging utilities
    # - the two are related but not identical, so keeping both optional makes
    #   the observability surface more flexible
    if config.enable_device_stats:
        callbacks.append(DeviceStatsMonitor())

    # The rich progress bar is not primarily a forensic debugging tool, but it
    # still improves local observability by making step/epoch progress more
    # legible during interactive runs. We gate it both by config and by package
    # availability so environments without `rich` degrade gracefully.
    if config.enable_rich_progress_bar and _has_module("rich"):
        callbacks.append(RichProgressBar())

    # ---------------------------------------------------------------------
    # 3. Debug-oriented callbacks
    # ---------------------------------------------------------------------
    #
    # These callbacks inspect the actual runtime behavior of the training
    # pipeline:
    # - batch contract / schema visibility
    # - gradient health visibility
    # - activation health visibility
    # - system telemetry that helps contextualize numerical behavior
    #
    # They sit after the broad run/model-surface callbacks because they are
    # narrower and more intrusive conceptually: they are about "what is the
    # model doing internally right now?" rather than "what run is this?"
    #
    # We keep these separate from parameter histogram callbacks because their
    # purpose is different:
    # - debug callbacks try to surface compact actionable numerical-health
    #   signals for fast diagnosis
    # - parameter callbacks provide richer but often heavier distribution views
    if config.enable_batch_audit:
        callbacks.append(BatchAuditCallback(config, text_logger=text_logger))

    if config.enable_gradient_stats:
        callbacks.append(GradientStatsCallback(config))

    # System telemetry is grouped near the debug-oriented block because, in
    # practice, resource behavior helps explain numerical behavior. Gradient or
    # activation anomalies are easier to interpret when the same run also logs
    # memory pressure, device utilization, and step-level system context.
    if config.enable_system_telemetry:
        callbacks.append(SystemTelemetryCallback(config, text_logger=text_logger))

    if config.enable_activation_stats:
        callbacks.append(ActivationStatsCallback(config))

    # ---------------------------------------------------------------------
    # 4. Parameter-state callbacks
    # ---------------------------------------------------------------------
    #
    # These callbacks provide more detailed visibility into parameter and
    # gradient distributions over time.
    #
    # Why they are separate from `GradientStatsCallback`:
    # - gradient stats focuses on compact sampled health summaries
    # - scalar telemetry provides broad per-parameter scalar signals
    # - histograms provide richer but heavier distribution-level inspection
    #
    # Keeping them as distinct callbacks allows users to enable only the level
    # of detail they need for a given run instead of paying the cost for every
    # parameter-related artifact all the time.
    if config.enable_parameter_scalars:
        callbacks.append(ParameterScalarTelemetryCallback(config))

    if config.enable_parameter_histograms:
        callbacks.append(ParameterHistogramCallback(config))

    # ---------------------------------------------------------------------
    # 5. Prediction-semantic callbacks
    # ---------------------------------------------------------------------
    #
    # This block is the natural home for callbacks that inspect the *meaningful
    # numerical behavior* of model outputs rather than just internal model state.
    #
    # What this layer is trying to answer:
    # - Are predictions finite?
    # - Are quantiles ordered correctly?
    # - Are intervals collapsing toward zero width?
    # - Are predictions becoming nearly constant?
    # - Is forecast scale drifting away from target scale?
    #
    # Why this belongs after activation/gradient/parameter callbacks:
    # - activations and gradients tell us whether the model appears alive
    # - parameter telemetry tells us whether weights look stable
    # - prediction sanity tells us whether the produced forecast tensor itself
    #   still looks plausible
    #
    # Why this belongs before qualitative figure export:
    # - scalar sanity metrics are easier to aggregate and compare across runs
    # - if prediction figures look strange, these metrics help explain *why*
    # - it keeps the conceptual flow as:
    #   internals first -> prediction semantics second -> visualization last
    #
    # Backward-compatibility note:
    # The config may not yet expose an `enable_prediction_sanity` flag on all
    # local branches. We therefore treat the feature as enabled by default via
    # `getattr(..., True)` so that:
    # - newer branches can explicitly toggle it
    # - older branches do not crash due to a missing config field
    #
    # If you later formalize the flag in `ObservabilityConfig`, keeping the same
    # line below preserves behavior while making the policy explicit.
    if getattr(config, "enable_prediction_sanity", True):
        callbacks.append(PredictionSanityCallback(config, text_logger=text_logger))

    # ---------------------------------------------------------------------
    # 6. Prediction-visualization callbacks
    # ---------------------------------------------------------------------
    #
    # Prediction figures are valuable, but they are the most presentation-like
    # part of the observability stack. They help a human judge output behavior,
    # yet they are less foundational than:
    # - confirming batch contract correctness
    # - checking gradients/activations
    # - confirming model and system state
    # - verifying quantitative output sanity
    #
    # Keeping `PredictionFigureCallback` in this file is important because it is
    # part of the repo's existing supported observability surface. The addition
    # of prediction-sanity diagnostics should *extend* that surface, not replace
    # the older qualitative figure path.
    #
    # Putting figure logging last in the conceptual assembly order makes the
    # callback stack easier to interpret as:
    # first understand the run and internals,
    # then validate prediction semantics,
    # and finally inspect qualitative outputs.
    if config.enable_prediction_figures:
        callbacks.append(PredictionFigureCallback(config))

    # Final note:
    # Returning the assembled list directly keeps this function easy to test:
    # tests can inspect length, callback classes, and order without needing to
    # instantiate a Trainer.
    return callbacks
