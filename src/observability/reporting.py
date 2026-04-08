from __future__ import annotations

# Compatibility shim:
# the canonical post-run reporting/export implementation now lives in the
# dedicated `reporting` package. We keep this module as a re-export surface so
# older imports such as `observability.reporting` continue to work while still
# pointing at the same underlying implementation objects.

from reporting import (
    SharedReport,
    build_shared_report,
    export_grouped_tables_from_report,
    export_prediction_table,
    export_prediction_table_from_report,
    generate_plotly_reports,
    log_shared_report_to_tensorboard,
)

__all__ = [
    "SharedReport",
    "build_shared_report",
    "export_grouped_tables_from_report",
    "export_prediction_table",
    "export_prediction_table_from_report",
    "generate_plotly_reports",
    "log_shared_report_to_tensorboard",
]