from __future__ import annotations

# AI-assisted maintenance note:
# This module isolates observability policy from the rest of the configuration
# package.
#
# Why this config lives in its own file:
# - logging, telemetry, debugging, and artifact generation are runtime concerns
# - those concerns affect how we inspect a run, not what model semantics are
#   checkpointed
# - separating observability policy from model/data config makes it easier to
#   reason about what changes experiment visibility versus what changes the
#   actual forecasting system
#
# Why this matters in the current repo:
# The glucose forecasting project now has several observability layers:
# - logger surfaces such as TensorBoard and plain text logs
# - callback-driven numerical-health signals
# - system telemetry and model-structure artifacts
# - parameter distribution tracking
# - prediction figures and exported run reports
#
# As that surface grows, a single dedicated config object becomes important for
# answering questions like:
# - which debug signals were intended to be on for this run?
# - how noisy was the callback stack allowed to be?
# - where should artifacts be written on disk?
# - were we asking for baseline observability or an intentionally heavy trace?
#
# Design boundary:
# - this file defines observability *policy*
# - this file validates observability *inputs*
# - this file normalizes observability *paths*
#
# This file should *not*:
# - assemble callbacks
# - implement callback logic
# - decide model architecture
# - decide data preprocessing behavior
# - own trainer construction
#
# In other words, this file says what observability we want, not how each
# callback achieves it.

from dataclasses import dataclass
from pathlib import Path

from config.types import PathInput


@dataclass
class ObservabilityConfig:
    """Configuration for logging, telemetry, debug instrumentation, and reports.

    Purpose:
        Define the runtime observability policy for the training and evaluation
        workflow.

    Context:
        This config is deliberately separate from model/data architecture
        config. Observability settings change how much evidence we collect about
        a run, how often we collect it, and where we write it, but they should
        not change the semantic meaning of the forecasting architecture itself.

    Why a dedicated config object is useful:
        The repo has outgrown a handful of ad hoc logging flags. We now need one
        place where a maintainer can inspect:
        - logging surfaces
        - callback enable/disable policy
        - artifact output paths
        - debug sampling cadence
        - prediction/reporting verbosity

    Important disclaimer:
        Enabling every diagnostic at once can materially increase:
        - runtime overhead
        - memory pressure
        - disk usage
        - artifact volume
        - logger noise

        This config improves visibility into a run, but observability alone does
        not guarantee model correctness or experimental validity. It only makes
        those properties easier to inspect and challenge.

    Maintenance rule:
        If a new callback or observability artifact becomes part of the
        supported workflow, this dataclass should usually be updated alongside
        it so the repo keeps one coherent source of truth for runtime
        instrumentation policy.
    """

    # ------------------------------------------------------------------
    # High-level observability mode
    # ------------------------------------------------------------------
    #
    # This field is intentionally lightweight and descriptive rather than
    # magical. It acts as a human-readable declaration of observability intent:
    # - baseline: normal experiment logging with reasonable signal density
    # - debug: heavier numerical introspection during development
    # - trace: the heaviest expected observability posture in this repo
    #
    # Important note:
    # The mode itself does not automatically derive every other field in this
    # dataclass. We keep the individual flags explicit so runs remain readable
    # and override-friendly. The mode is still valuable because it communicates
    # intended observability posture in saved configs, logs, and experiment
    # records.
    mode: str = "baseline"

    # ------------------------------------------------------------------
    # Core logging surfaces
    # ------------------------------------------------------------------
    #
    # These fields control the broadest "where does observability go?" surfaces.
    #
    # `enable_tensorboard`
    #     Enables TensorBoard-backed logging when the surrounding workflow wires
    #     a TensorBoard logger. This is the main structured surface for scalar
    #     metrics, text summaries, model graphs, and images.
    #
    # `enable_text_logging`
    #     Enables plain-text log emission for human-readable event streams and
    #     structured JSON/text payloads that are easier to inspect directly in
    #     terminal logs or saved text files.
    #
    # `enable_csv_fallback_logger`
    #     Enables CSV-style fallback logging surfaces where the repo supports
    #     them. This is especially useful when richer tracking backends are not
    #     available or when simple spreadsheet-style inspection is preferred.
    enable_tensorboard: bool = True
    enable_text_logging: bool = True
    enable_csv_fallback_logger: bool = True

    # ------------------------------------------------------------------
    # Callback-driven run / trainer telemetry
    # ------------------------------------------------------------------
    #
    # These flags control the main observability callbacks assembled in
    # `observability.callbacks`.
    #
    # Why keep these as individual booleans:
    # Different debugging situations benefit from different observability
    # surfaces. For example:
    # - a routine experiment may want learning-rate and progress-bar logging
    # - a deep debugging run may want batch audit + gradient stats
    # - a lightweight cluster run may want to suppress heavy artifacts
    #
    # Keeping them independent avoids forcing every run into one rigid logging
    # package.
    #
    # `enable_learning_rate_monitor`
    #     Adds Lightning's LR monitor. This is a simple but high-value signal
    #     for optimizer schedule sanity.
    #
    # `enable_device_stats`
    #     Adds Lightning's device stats monitor where supported. This complements
    #     but does not replace the repo's custom system telemetry callback.
    #
    # `enable_rich_progress_bar`
    #     Enables the richer terminal progress UI for interactive runs. This is
    #     mostly a usability feature, but it still improves local visibility
    #     into run progress.
    #
    # `enable_system_telemetry`
    #     Enables the repo's custom host/device telemetry callback, which writes
    #     resource metrics into logger backends and CSV/text artifacts.
    #
    # `enable_parameter_histograms`
    #     Enables heavier parameter/gradient histogram logging. Useful for deep
    #     inspection, but potentially noisy and more expensive.
    #
    # `enable_parameter_scalars`
    #     Enables lighter-weight per-parameter scalar summaries such as norms.
    #
    # `enable_prediction_figures`
    #     Enables sampled qualitative forecast figures. These are especially
    #     useful for quickly judging whether model outputs look plausible.
    #
    # `enable_model_graph`
    #     Enables TensorBoard-native graph logging when possible.
    #
    # `enable_model_text`
    #     Enables plain-text model architecture dumps, usually via `repr(model)`.
    #
    # `enable_torchview`
    #     Enables the static torchview/Graphviz export path. This is useful for
    #     architecture visualization, but it is intentionally best-effort and
    #     separate from actual runtime training logic.
    #
    # `enable_profiler`
    #     Enables profiler integration when the broader workflow supports it.
    #     Profiling is generally more intrusive than normal observability and is
    #     therefore kept as a distinct explicit switch.
    enable_learning_rate_monitor: bool = True
    enable_device_stats: bool = True
    enable_rich_progress_bar: bool = True
    enable_system_telemetry: bool = True
    enable_parameter_histograms: bool = True
    enable_parameter_scalars: bool = True
    enable_prediction_figures: bool = True
    enable_model_graph: bool = True
    enable_model_text: bool = True
    enable_torchview: bool = True
    enable_profiler: bool = False

    # ------------------------------------------------------------------
    # Deeper debugging hooks
    # ------------------------------------------------------------------
    #
    # These are the most directly code-logic-oriented switches in the
    # observability stack.
    #
    # `enable_gradient_stats`
    #     Enables sampled gradient-health summaries. In the current repo this is
    #     especially valuable for confirming that major fused-model branches are
    #     actually receiving gradient signal.
    #
    # `enable_activation_stats`
    #     Enables sampled activation summaries from selected high-level modules.
    #     This is intentionally off by default in some configurations because
    #     forward-hook-based logging can add extra runtime cost and noise.
    #
    # `enable_batch_audit`
    #     Enables one-time or capped batch schema / contract summaries. This is
    #     one of the highest-value debugging signals for catching silent data
    #     contract issues early.
    enable_gradient_stats: bool = True
    enable_activation_stats: bool = False
    enable_batch_audit: bool = True

    # ------------------------------------------------------------------
    # Artifact / report outputs
    # ------------------------------------------------------------------
    #
    # These toggles control higher-level saved outputs rather than immediate
    # live scalar logging.
    #
    # `enable_prediction_exports`
    #     Enables writing tabular prediction outputs that can later be inspected
    #     offline or passed into downstream analysis/reporting steps.
    #
    # `enable_plot_reports`
    #     Enables saved report plots and richer visual run summaries where the
    #     workflow supports them.
    enable_prediction_exports: bool = True
    enable_plot_reports: bool = True

    # ------------------------------------------------------------------
    # Filesystem layout under the run output directory
    # ------------------------------------------------------------------
    #
    # These paths are optional because not every environment or logger setup
    # uses every artifact surface.
    #
    # Why path normalization happens in `__post_init__`:
    # Configs may be created from CLI strings, YAML values, or already-created
    # `Path` objects. Normalizing once here lets the rest of the codebase assume
    # path semantics consistently instead of re-checking types at every call
    # site.
    #
    # `log_dir`
    #     Base directory for run logging outputs when the broader workflow needs
    #     one.
    #
    # `text_log_path`
    #     Optional plain-text log destination.
    #
    # `telemetry_path`
    #     Optional CSV or tabular path for system telemetry snapshots.
    #
    # `prediction_table_path`
    #     Optional output path for exported prediction tables.
    #
    # `report_dir`
    #     Optional directory for saved report artifacts such as plots.
    #
    # `profiler_path`
    #     Optional output path or directory for profiler artifacts.
    #
    # `torchview_path`
    #     Optional base path for the rendered torchview artifact.
    log_dir: PathInput | None = None
    text_log_path: PathInput | None = None
    telemetry_path: PathInput | None = None
    prediction_table_path: PathInput | None = None
    report_dir: PathInput | None = None
    profiler_path: PathInput | None = None
    torchview_path: PathInput | None = None

    # ------------------------------------------------------------------
    # Sampling controls / verbosity limits
    # ------------------------------------------------------------------
    #
    # These fields constrain how often potentially noisy or expensive
    # instrumentation should fire.
    #
    # Why these controls are important:
    # Deep observability is only useful if it remains operationally manageable.
    # Without explicit limits, debugging callbacks can:
    # - swamp TensorBoard with too many points
    # - create huge text artifacts
    # - slow down training
    # - make later analysis harder because signal is buried in repetition
    #
    # `debug_every_n_steps`
    #     Global sampling cadence for step-based debug callbacks such as
    #     gradient and activation stats.
    #
    # `telemetry_every_n_steps`
    #     Sampling cadence for system telemetry snapshots.
    #
    # `batch_audit_limit`
    #     Maximum number of batches per stage that should receive detailed batch
    #     audit logging.
    #
    # `max_forecast_subjects_per_report`
    #     Cap on the number of forecast subjects surfaced in saved reports so
    #     qualitative artifacts remain readable.
    #
    # `histogram_every_n_epochs`
    #     Epoch cadence for heavier parameter/gradient histogram emission.
    #
    # `parameter_scalar_every_n_epochs`
    #     Epoch cadence for lighter parameter scalar telemetry.
    #
    # `figure_every_n_epochs`
    #     Epoch cadence for prediction figure generation.
    #
    # `max_prediction_plots`
    #     Maximum number of prediction plots to save per reporting interval.
    #
    # `profiler_type`
    #     Which profiler backend/mode the workflow should request when profiling
    #     is enabled.
    #
    # `torchview_depth`
    #     How deep the static torchview graph expansion should go.
    #
    # `torchview_roll`
    #     Whether torchview should roll repeated structures where supported.
    #
    # `torchview_expand_nested`
    #     Whether nested modules should be expanded in the torchview artifact.
    debug_every_n_steps: int = 10
    telemetry_every_n_steps: int = 10
    batch_audit_limit: int = 1
    max_forecast_subjects_per_report: int = 5
    histogram_every_n_epochs: int = 1
    parameter_scalar_every_n_epochs: int = 1
    figure_every_n_epochs: int = 1
    max_prediction_plots: int = 2
    profiler_type: str = "simple"
    torchview_depth: int = 4
    torchview_roll: bool = True
    torchview_expand_nested: bool = True

    def __post_init__(self) -> None:
        """Validate observability values and normalize filesystem fields.

        Purpose:
            Fail fast on impossible or inconsistent observability settings and
            convert path-like fields into concrete `Path` objects exactly once.

        Why this happens here:
            Observability is runtime policy rather than model semantics, so this
            is the earliest safe place to reject invalid values before callback
            assembly, trainer construction, or artifact writing begins.

        What is validated:
            - the declared high-level observability mode
            - all positive integer sampling/limit controls
            - profiler type membership
            - all optional path-like fields

        Why fail-fast validation matters:
            Without early validation, an invalid observability request can hide
            until much later in training setup, making the root cause harder to
            diagnose and easier to misattribute to model code.
        """
        # --------------------------------------------------------------
        # 1. Validate the high-level mode declaration
        # --------------------------------------------------------------
        #
        # The mode is primarily descriptive, but it still needs to stay within
        # the small supported vocabulary so saved configs and logs remain
        # consistent across environments and runs.
        valid_modes = {"baseline", "debug", "trace"}
        if self.mode not in valid_modes:
            supported = ", ".join(sorted(valid_modes))
            raise ValueError(
                f"mode must be one of {supported}, got '{self.mode}'."
            )

        # --------------------------------------------------------------
        # 2. Normalize all configured filesystem targets to `Path`
        # --------------------------------------------------------------
        #
        # Why normalize here instead of later:
        # - downstream code can assume path semantics
        # - type handling stays centralized
        # - repeated `Path(...)` wrapping throughout the codebase is avoided
        #
        # Only optional observability artifact paths are normalized here; a
        # value of `None` still means "this output surface is not explicitly
        # configured."
        for field_name in (
            "log_dir",
            "text_log_path",
            "telemetry_path",
            "prediction_table_path",
            "report_dir",
            "profiler_path",
            "torchview_path",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, Path(value))

        # --------------------------------------------------------------
        # 3. Validate positive integer controls
        # --------------------------------------------------------------
        #
        # These values drive callback frequency, report size, and artifact
        # density. We keep the bounds explicit because silently accepting zero
        # or negative values would usually create confusing downstream behavior
        # such as callbacks never firing or artifact loops misbehaving.
        if self.debug_every_n_steps <= 0:
            raise ValueError("debug_every_n_steps must be > 0")

        if self.telemetry_every_n_steps <= 0:
            raise ValueError("telemetry_every_n_steps must be > 0")

        if self.batch_audit_limit <= 0:
            raise ValueError("batch_audit_limit must be > 0")

        if self.max_forecast_subjects_per_report <= 0:
            raise ValueError("max_forecast_subjects_per_report must be > 0")

        if self.histogram_every_n_epochs <= 0:
            raise ValueError("histogram_every_n_epochs must be > 0")

        if self.parameter_scalar_every_n_epochs <= 0:
            raise ValueError("parameter_scalar_every_n_epochs must be > 0")

        if self.figure_every_n_epochs <= 0:
            raise ValueError("figure_every_n_epochs must be > 0")

        if self.max_prediction_plots <= 0:
            raise ValueError("max_prediction_plots must be > 0")

        # --------------------------------------------------------------
        # 4. Validate categorical / enumerated observability controls
        # --------------------------------------------------------------
        #
        # Profiling is especially sensitive to backend/mode selection, so we
        # validate the supported vocabulary explicitly rather than relying on a
        # later failure in trainer/profiler construction.
        if self.profiler_type not in {"simple", "advanced", "pytorch"}:
            raise ValueError(
                "profiler_type must be one of simple, advanced, pytorch"
            )

        # Torchview depth directly affects graph rendering behavior. A non-
        # positive depth would not make semantic sense for the current export
        # path, so we reject it early.
        if self.torchview_depth <= 0:
            raise ValueError("torchview_depth must be > 0")
