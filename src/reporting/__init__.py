from __future__ import annotations

# AI-assisted maintenance note:
# `reporting` is the stable package-level facade for the repository's post-run
# reporting layer.
#
# Why this package now exists:
# - the repository has grown a real lifecycle boundary between
#   runtime observability and post-run reporting
# - `observability` primarily deals with live run-time visibility:
#   callbacks, logger setup, profiler setup, telemetry, model graph surfaces,
#   and other in-loop inspection tools
# - `reporting` now deals with post-run packaging and presentation after
#   predictions and structured evaluation already exist
#
# In other words:
# - `evaluation` computes metric truth
# - `reporting` packages and renders post-run artifacts from that truth
# - `observability` handles live runtime visibility during training/evaluation
#
# Why keep a package-level facade:
# - callers such as workflows, tests, notebooks, and future CLI helpers should
#   be able to import short, stable names like
#   `from reporting import build_shared_report, export_prediction_table`
# - the internal file split can evolve over time without forcing import churn
#   throughout the rest of the codebase
# - this keeps the public API easy to discover even as the implementation grows
#   into smaller, more focused files
#
# Internal layout:
# - `types.py`
#   canonical in-memory reporting contracts such as `SharedReport`
# - `builders.py`
#   construction of the shared-report surface from predictions + evaluation
# - `exports.py`
#   persistence-oriented sinks such as CSV export
# - `plotly_reports.py`
#   lightweight HTML/Plotly presentation sink
#
# What does *not* live here:
# - implementation details of report builders or sinks
# - canonical metric computation logic
# - runtime callback logic
# - trainer/logger/profiler setup
#
# This file exists to define the public reporting import surface, not to
# perform reporting work itself.


# ============================================================================
# Public Reporting Surface
# ============================================================================
# Re-export the most commonly used reporting contracts and helpers so callers
# can use short package-level imports while the implementation remains split
# across smaller files.

from reporting.builders import build_shared_report
from reporting.exports import export_prediction_table
from reporting.plotly_reports import generate_plotly_reports
from reporting.types import SharedReport, TestDataloaderProvider
from reporting.tensorboard import log_shared_report_to_tensorboard

# `__all__` is the stable package-level reporting API.
#
# Important compatibility note:
# this list intentionally exposes the same core reporting entrypoints the
# workflow currently needs:
# - build the canonical shared report
# - export the flat prediction table
# - generate lightweight HTML reports
#
# As the package grows, new sinks (for example TensorBoard) can be added here
# without changing the import style used by the rest of the repository.
__all__ = [
    "SharedReport",
    "TestDataloaderProvider",
    "build_shared_report",
    "export_prediction_table",
    "generate_plotly_reports",
    "log_shared_report_to_tensorboard"
]