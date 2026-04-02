from __future__ import annotations

# AI-assisted maintenance note:
# This module isolates observability policy from the rest of the configuration
# package. Logging, telemetry, and debug instrumentation are runtime concerns,
# so they are easier to reason about when kept separate from the data/model
# contracts that define the checkpointed architecture itself.

from dataclasses import dataclass
from pathlib import Path

from config.types import PathInput


@dataclass
class ObservabilityConfig:
    """
    Configuration for logging, telemetry, debug instrumentation, and reports.

    Purpose:
    define the runtime observability policy for logging, telemetry, debug
    instrumentation, and report generation.

    Context:
    this config is deliberately separate from the model/data architecture
    config. Observability is a runtime policy concern owned by the Trainer
    wrapper and the top-level workflow, not part of the checkpointed semantic
    contract of the fused forecaster itself.

    Important disclaimer:
    enabling every diagnostic at once can materially increase runtime cost,
    storage overhead, and artifact volume, especially in notebook and Colab
    environments. This config improves visibility into a run, but it does not
    by itself guarantee model correctness or experiment validity.
    """

    # High-level behavior preset. Baseline is suitable for normal experiments,
    # debug enables more sampled diagnostics, and trace is the heaviest mode.
    mode: str = "baseline"

    # Core logging surfaces.
    enable_tensorboard: bool = True
    enable_text_logging: bool = True
    enable_csv_fallback_logger: bool = True

    # Callback-driven run telemetry.
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

    # Deeper debugging hooks.
    enable_gradient_stats: bool = True
    enable_activation_stats: bool = False
    enable_batch_audit: bool = True

    # Artifact/report outputs.
    enable_prediction_exports: bool = True
    enable_plot_reports: bool = True

    # Filesystem layout under the run output directory.
    log_dir: PathInput | None = None
    text_log_path: PathInput | None = None
    telemetry_path: PathInput | None = None
    prediction_table_path: PathInput | None = None
    report_dir: PathInput | None = None
    profiler_path: PathInput | None = None
    torchview_path: PathInput | None = None

    # Sampling controls for potentially noisy or expensive debug signals.
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
        """
        Validate observability mode/limits and normalize filesystem targets.

        Context:
        observability is runtime policy rather than model semantics, so this is
        the earliest safe place to reject impossible sampling settings or invalid
        output-path values.
        """
        # Validate high-level behavior early so invalid observability requests
        # fail near config construction time rather than much later during
        # trainer assembly.
        valid_modes = {"baseline", "debug", "trace"}
        if self.mode not in valid_modes:
            supported = ", ".join(sorted(valid_modes))
            raise ValueError(
                f"mode must be one of {supported}, got '{self.mode}'."
            )

        # Normalize all filesystem fields to `Path` once so downstream code can
        # assume path semantics consistently.
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

        # All of these controls influence callback frequency or artifact shape,
        # so we keep the bounds strict and explicit here.
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
        if self.profiler_type not in {"simple", "advanced", "pytorch"}:
            raise ValueError(
                "profiler_type must be one of simple, advanced, pytorch"
            )
        if self.torchview_depth <= 0:
            raise ValueError("torchview_depth must be > 0")
