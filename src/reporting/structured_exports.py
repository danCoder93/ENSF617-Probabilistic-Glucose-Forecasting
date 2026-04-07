from __future__ import annotations

# AI-assisted maintenance note:
# This module contains the structured artifact export sink for the repository's
# post-run reporting layer.
#
# Why this file exists:
# - the reporting package already distinguishes between canonical shared-report
#   construction and sink-specific rendering/export
# - TensorBoard and Plotly are useful presentation sinks, but they are not ideal
#   when a user wants a simple machine-readable artifact bundle for later
#   scripting, spreadsheet inspection, archival, or downstream notebook work
# - a dedicated structured-export sink keeps those machine-readable artifacts
#   aligned with the same canonical `SharedReport` used by the other sinks
#
# Responsibility boundary:
# - consume an already-built `SharedReport`
# - export tabular report surfaces as CSV
# - export scalar/text/metadata report surfaces as JSON
# - write one manifest describing what was exported
#
# What does *not* live here:
# - canonical metric computation
# - shared-report row construction
# - Plotly HTML generation
# - TensorBoard sink logic
# - workflow orchestration
#
# In other words, this file is a sink. It should serialize the canonical shared
# report into a structured artifact bundle, not quietly become a second report
# builder.
#
# Format-design note:
# This sink intentionally uses a mixed-format export bundle:
# - CSV for naturally tabular report tables
# - JSON for keyed, nested, or text-heavy report surfaces
#
# That split keeps the exported artifacts convenient for both humans and
# downstream tooling without forcing every surface into an awkward one-format
# representation.
#
# Packaging enhancement note:
# Earlier revisions exported everything flat under one folder. That worked, but
# the bundle became harder to scan as the shared report grew. The current
# version keeps the same shared-report truth and the same file formats while
# introducing a slightly clearer folder layout:
# - top-level JSON summaries remain easy to discover
# - tabular artifacts live under `tables/`
# - optional data-oriented add-on artifacts live under `data/`
#
# Important compatibility rule:
# This is still a sink-only change. The export layer is allowed to package files
# more clearly, but it must not invent new report truth or recompute metrics.

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from config import PathInput
from observability.utils import _ensure_parent

from reporting.types import SharedReport


def _json_ready_report_value(value: Any) -> Any:
    """Return a JSON-safe version of a shared-report value.

    Context:
        Shared-report surfaces are already lightweight, but they may still
        contain values such as:
        - pandas objects
        - tuples
        - path-like objects
        - nested dict/list structures

        This helper keeps JSON serialization centralized so the public export
        function can remain small and readable.

    Important design note:
        The goal is not to aggressively coerce every unknown object into a new
        semantic type. The goal is only to make the export payload stable and
        serializable while preserving meaning as faithfully as possible.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Normalize paths to strings so the exported JSON stays portable and easy to
    # inspect outside Python.
    if isinstance(value, Path):
        return str(value)

    # Pandas Timestamp and Timedelta objects serialize cleanly as strings and
    # preserve enough information for report-style artifact use.
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    # Pandas/Numpy scalar-like values often support `.item()`. Converting them
    # here helps keep downstream JSON clean and Python-native.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_ready_report_value(value.item())
        except Exception:
            pass

    if isinstance(value, Mapping):
        return {
            str(key): _json_ready_report_value(nested_value)
            for key, nested_value in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_ready_report_value(item) for item in value]

    # Fall back to string conversion for uncommon values rather than failing the
    # entire export. This sink is best-effort serialization of already-built
    # report surfaces, not a strict schema validator.
    return str(value)


def _write_json(path: Path, payload: Any) -> Path:
    """Write one JSON payload to disk with stable formatting.

    Context:
        The export sink writes several small JSON artifacts. Centralizing the
        write path keeps parent-directory creation, UTF-8 encoding, and
        indentation policy consistent across all of them.
    """
    _ensure_parent(path)
    path.write_text(
        json.dumps(_json_ready_report_value(payload), indent=2),
        encoding="utf-8",
    )
    return path


def _build_export_layout(*, export_dir: Path) -> dict[str, Path]:
    """Return the concrete folder layout for one structured-export bundle.

    Context:
        The reporting package now exports a few more artifact families, so the
        sink keeps the directory policy centralized here rather than scattering
        folder names across multiple helpers.

    Design note:
        The layout intentionally stays shallow and human-readable:
        - root directory for bundle-level index files and compact summaries
        - `tables/` for CSV-backed tabular surfaces
        - `data/` for optional dataset-oriented JSON artifacts
    """
    layout = {
        "root": export_dir,
        "tables": export_dir / "tables",
        "data": export_dir / "data",
    }

    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)

    return layout


def _export_report_tables(
    *,
    shared_report: SharedReport,
    tables_dir: Path,
) -> dict[str, Path]:
    """Export the tabular shared-report surfaces as CSV files.

    Context:
        Report tables are already packaged upstream in DataFrame form, so this
        helper simply persists them in a spreadsheet-friendly format rather than
        rebuilding or reshaping them here.

    Important policy:
        - only DataFrame-backed table surfaces are exported
        - empty tables are still exported when present so the artifact bundle
          preserves the table contract, not just non-empty results
    """
    table_file_names = {
        "prediction_table": "prediction_table.csv",
        "by_horizon": "by_horizon.csv",
        "by_subject": "by_subject.csv",
        "by_glucose_range": "by_glucose_range.csv",
    }

    exported_paths: dict[str, Path] = {}
    for table_name, file_name in table_file_names.items():
        frame = shared_report.tables.get(table_name)
        if not isinstance(frame, pd.DataFrame):
            continue

        output_path = tables_dir / file_name

        # Preserve the current table exactly as packaged in the shared report.
        # CSV is chosen here because these surfaces are naturally tabular and
        # often consumed later in spreadsheets, notebooks, or quick shell/data
        # tooling.
        frame.to_csv(output_path, index=False)
        exported_paths[table_name] = output_path

    return exported_paths


def _export_report_json_surfaces(
    *,
    shared_report: SharedReport,
    export_dir: Path,
) -> dict[str, Path]:
    """Export the non-tabular shared-report surfaces as JSON files.

    Context:
        Scalars, text blocks, and metadata are already structured for keyed JSON
        export. Writing them separately keeps each surface easy to inspect and
        avoids forcing unrelated content into one oversized file.
    """
    exported_paths: dict[str, Path] = {}

    # Scalars are kept in their own file because they are often useful as a
    # compact machine-readable summary without requiring any table loading.
    exported_paths["scalars"] = _write_json(
        export_dir / "scalars.json",
        shared_report.scalars,
    )

    # Text surfaces are kept separate because they tend to be sink-facing human
    # summaries rather than numeric machine features.
    exported_paths["text"] = _write_json(
        export_dir / "text.json",
        shared_report.text,
    )

    # Metadata is kept separate so downstream users can inspect run/export
    # context without conflating it with top-level scalar metrics.
    exported_paths["metadata"] = _write_json(
        export_dir / "metadata.json",
        shared_report.metadata,
    )

    # A small metrics summary file gives downstream tooling one focused place to
    # read compact numeric report outputs without loading the broader metadata or
    # text surfaces. This remains packaging-only because it is just a curated
    # subset of already-canonical shared-report content.
    exported_paths["metrics_summary"] = _write_json(
        export_dir / "metrics_summary.json",
        {
            "scalars": shared_report.scalars,
            "has_evaluation_result": shared_report.metadata.get(
                "has_evaluation_result",
                False,
            ),
            "quantiles": shared_report.metadata.get("quantiles", ()),
            "sampling_interval_minutes": shared_report.metadata.get(
                "sampling_interval_minutes"
            ),
        },
    )

    return exported_paths


def _export_optional_data_surfaces(
    *,
    export_dir: Path,
    report_dir: Path,
) -> dict[str, Path]:
    """Export optional data-oriented artifacts that already exist beside the bundle.

    Context:
        The workflow may now write a best-effort `data_summary.json` sourced from
        the DataModule. That file is not part of the canonical `SharedReport`
        itself, but it still belongs logically with the rest of the structured
        report artifacts.

    Important policy:
        - this helper never computes data summary content itself
        - it only mirrors an already-written artifact into the structured bundle
        - missing optional artifacts are skipped cleanly
    """
    exported_paths: dict[str, Path] = {}

    source_path = report_dir / "data_summary.json"
    if source_path.exists():
        output_path = export_dir / "data_summary.json"
        output_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        exported_paths["data_summary"] = output_path

    return exported_paths


def _build_manifest(
    *,
    shared_report: SharedReport,
    exported_tables: Mapping[str, Path],
    exported_json_surfaces: Mapping[str, Path],
    exported_data_surfaces: Mapping[str, Path],
    export_dir: Path,
) -> dict[str, Any]:
    """Build the manifest describing one structured-report export bundle.

    Context:
        The manifest is the discovery/index file for the exported artifact
        bundle. It allows downstream tools and users to find the exported files
        without hard-coding assumptions about what was present.
    """
    row_counts: dict[str, int] = {}
    for table_name, frame in shared_report.tables.items():
        if isinstance(frame, pd.DataFrame):
            row_counts[table_name] = int(len(frame))

    def _relative(path: Path) -> str:
        return path.relative_to(export_dir).as_posix()

    export_entries = {
        "json": {
            name: _relative(path) for name, path in exported_json_surfaces.items()
        },
        "tables": {
            name: _relative(path) for name, path in exported_tables.items()
        },
        "data": {
            name: _relative(path) for name, path in exported_data_surfaces.items()
        },
    }

    return {
        "version": 2,
        "export_type": "shared_report_artifacts",
        "layout": {
            "root": ".",
            "tables": "tables",
            "data": "data",
        },
        "has_prediction_table": bool(
            isinstance(shared_report.tables.get("prediction_table"), pd.DataFrame)
            and not shared_report.tables.get("prediction_table", pd.DataFrame()).empty
        ),
        "has_evaluation_result": bool(
            shared_report.metadata.get("has_evaluation_result", False)
        ),
        "has_data_summary": bool(exported_data_surfaces),
        "quantiles": _json_ready_report_value(
            shared_report.metadata.get("quantiles", ())
        ),
        "sampling_interval_minutes": _json_ready_report_value(
            shared_report.metadata.get("sampling_interval_minutes")
        ),
        "exports": export_entries,
        "row_counts": row_counts,
    }


def export_shared_report_artifacts(
    *,
    shared_report: SharedReport,
    report_dir: PathInput | None,
) -> dict[str, Path]:
    """Export the canonical shared report into a structured artifact bundle.

    Purpose:
        Create a mixed-format export bundle that is easy to inspect, archive,
        and consume from notebooks or downstream tooling.

    Output layout:
        <report_dir>/
          artifacts/
            shared_report/
              manifest.json
              scalars.json
              text.json
              metadata.json
              metrics_summary.json
              tables/
                prediction_table.csv
                by_horizon.csv
                by_subject.csv
                by_glucose_range.csv
              data/
                data_summary.json

    Important behavior:
        - returns an empty mapping when `report_dir` is not configured
        - creates the export folder on demand
        - writes report tables as CSV
        - writes keyed/nested surfaces as JSON
        - mirrors optional data-oriented artifacts into the bundle when present
        - returns the concrete file paths that were produced

    Important compatibility rule:
        This function consumes the already-built `SharedReport` exactly as it
        was packaged upstream. It does not recompute metrics or reshape the
        report into a second source of truth.
    """
    if report_dir is None:
        return {}

    report_dir_path = Path(report_dir)
    export_dir = report_dir_path / "artifacts" / "shared_report"
    layout = _build_export_layout(export_dir=export_dir)

    exported_tables = _export_report_tables(
        shared_report=shared_report,
        tables_dir=layout["tables"],
    )
    exported_json_surfaces = _export_report_json_surfaces(
        shared_report=shared_report,
        export_dir=layout["root"],
    )
    exported_data_surfaces = _export_optional_data_surfaces(
        export_dir=layout["data"],
        report_dir=report_dir_path,
    )

    manifest = _build_manifest(
        shared_report=shared_report,
        exported_tables=exported_tables,
        exported_json_surfaces=exported_json_surfaces,
        exported_data_surfaces=exported_data_surfaces,
        export_dir=layout["root"],
    )
    manifest_path = _write_json(layout["root"] / "manifest.json", manifest)

    exported_paths: dict[str, Path] = {
        **exported_tables,
        **exported_json_surfaces,
        **exported_data_surfaces,
        "manifest": manifest_path,
    }
    return exported_paths
