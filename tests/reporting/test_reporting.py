from __future__ import annotations

# These tests protect the reporting/export helpers that run after predictions
# already exist.

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

pytest.importorskip("torch")
pytest.importorskip("pandas")

from reporting import (
    SharedReport,
    export_grouped_tables_from_report,
    export_prediction_table,
    export_prediction_table_from_report,
    generate_plotly_reports,
    log_shared_report_to_tensorboard,
)
from reporting.builders import build_shared_report
from src.evaluation.types import EvaluationResult, GroupedMetricRow, MetricSummary


class StubDataModule:
    """Minimal datamodule exposing only the test-dataloader surface used by reporting helpers."""

    def __init__(self, test_batches: list[dict[str, object]]) -> None:
        """Store the synthetic test batches that should be exposed to the exporter."""
        self._test_batches = test_batches

    def test_dataloader(self) -> list[dict[str, object]]:
        """Return the stored synthetic batches as the held-out prediction surface."""
        return self._test_batches


class StubTensorBoardExperiment:
    """Record TensorBoard-style logging calls for sink-behavior assertions.

    Purpose:
        Keep TensorBoard sink tests lightweight and deterministic without
        requiring a real Lightning trainer, filesystem event writer, or running
        TensorBoard process.

    Context:
        `log_shared_report_to_tensorboard(...)` only relies on the small writer
        surface below. Capturing those calls in memory is sufficient for testing
        the sink contract.
    """

    def __init__(self) -> None:
        """Initialize empty call ledgers for scalar, text, and figure logging."""
        self.scalars: list[tuple[str, Any, int]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.figures: list[tuple[str, Any, int]] = []

    def add_scalar(self, tag: str, value: Any, global_step: int) -> None:
        """Record one scalar logging call exactly as the sink emitted it."""
        self.scalars.append((tag, value, global_step))

    def add_text(self, tag: str, text: str, global_step: int) -> None:
        """Record one text logging call exactly as the sink emitted it."""
        self.texts.append((tag, text, global_step))

    def add_figure(self, tag: str, figure: Any, global_step: int) -> None:
        """Record one figure logging call exactly as the sink emitted it."""
        self.figures.append((tag, figure, global_step))


class StubTensorBoardLogger:
    """Expose a TensorBoard-like `.experiment` surface to the reporting sink."""

    def __init__(self, experiment: StubTensorBoardExperiment) -> None:
        """Store the in-memory experiment recorder under the expected attribute."""
        self.experiment = experiment


class StubNonTensorBoardLogger:
    """Provide a logger shape that should be ignored by the TensorBoard sink.

    Context:
        The sink is expected to filter to logger backends that expose
        TensorBoard-style methods. This stub helps validate that incompatible
        loggers do not cause logging attempts or failures.
    """

    def __init__(self) -> None:
        """Store a placeholder experiment object without TensorBoard methods."""
        self.experiment = object()


def _build_prediction_table_frame() -> pd.DataFrame:
    """Build a compact canonical prediction table used across reporting tests.

    Purpose:
        Keep the reporting tests focused on sink behavior rather than repeating
        large inline DataFrame literals in every test.

    Context:
        The stricter reporting contract now treats the shared report as the
        canonical post-run packaging surface. This helper provides one stable
        prediction-table fixture shape that can be reused by both CSV-export and
        Plotly-sink tests.
    """
    return pd.DataFrame(
        {
            "subject_id": ["subject_a", "subject_a", "subject_b", "subject_b"],
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-01T00:05:00",
                "2026-01-02T00:00:00",
                "2026-01-02T00:05:00",
            ],
            "horizon_index": [0, 1, 0, 1],
            "target": [100.0, 101.0, 120.0, 121.0],
            "median_prediction": [99.0, 102.0, 119.5, 122.0],
            "residual": [-1.0, 1.0, -0.5, 1.0],
            "pred_q10": [95.0, 97.0, 115.0, 117.0],
            "pred_q90": [103.0, 107.0, 124.0, 126.0],
            "prediction_interval_width": [8.0, 10.0, 9.0, 9.0],
        }
    )


def _build_by_horizon_frame() -> pd.DataFrame:
    """Build the canonical grouped horizon table used by strict Plotly tests.

    Purpose:
        Mirror the grouped horizon-report table shape that the stricter Plotly
        sink now expects from `SharedReport.tables["by_horizon"]`.
    """
    return pd.DataFrame(
        {
            "group_name": ["horizon_index", "horizon_index"],
            "group_value": [0, 1],
            "count": [2, 2],
            "mae": [0.75, 1.0],
            "rmse": [0.79, 1.0],
            "bias": [-0.75, 1.0],
            "overall_pinball_loss": [0.45, 0.55],
            "mean_interval_width": [8.5, 9.5],
            "empirical_interval_coverage": [1.0, 0.5],
        }
    )


def _build_by_subject_frame() -> pd.DataFrame:
    """Build a compact grouped subject table with canonical grouped columns."""
    return pd.DataFrame(
        {
            "group_name": ["subject_id", "subject_id"],
            "group_value": ["subject_a", "subject_b"],
            "count": [2, 2],
            "mae": [1.0, 0.75],
            "rmse": [1.0, 0.80],
            "bias": [0.2, -0.5],
            "overall_pinball_loss": [0.50, 0.45],
            "mean_interval_width": [9.0, 8.5],
            "empirical_interval_coverage": [1.0, 1.0],
        }
    )


def _build_by_glucose_range_frame() -> pd.DataFrame:
    """Build a compact grouped glucose-range table with canonical grouped columns."""
    return pd.DataFrame(
        {
            "group_name": ["glucose_range", "glucose_range"],
            "group_value": ["euglycemia", "hyperglycemia"],
            "count": [2, 2],
            "mae": [0.6, 1.1],
            "rmse": [0.7, 1.2],
            "bias": [-0.1, 0.4],
            "overall_pinball_loss": [0.40, 0.60],
            "mean_interval_width": [8.0, 10.5],
            "empirical_interval_coverage": [1.0, 0.5],
        }
    )


def _build_evaluation_result() -> EvaluationResult:
    """Build a real typed evaluation result for report-builder tests.

    Purpose:
        Use the repo's actual evaluation contract rather than a lookalike stub
        so static analysis and runtime behavior stay aligned.
    """
    return EvaluationResult(
        summary=MetricSummary(
            count=4,
            mae=0.875,
            rmse=0.9,
            bias=0.1,
            overall_pinball_loss=0.5,
            pinball_loss_by_quantile={"0.1": 0.4, "0.5": 0.5, "0.9": 0.6},
            mean_interval_width=9.0,
            empirical_interval_coverage=0.75,
        ),
        by_horizon=(
            GroupedMetricRow(
                group_name="horizon_index",
                group_value=0,
                count=2,
                mae=0.75,
                rmse=0.85,
                bias=-0.25,
                overall_pinball_loss=0.40,
                mean_interval_width=8.5,
                empirical_interval_coverage=1.0,
            ),
            GroupedMetricRow(
                group_name="horizon_index",
                group_value=1,
                count=2,
                mae=1.0,
                rmse=1.05,
                bias=0.45,
                overall_pinball_loss=0.60,
                mean_interval_width=9.5,
                empirical_interval_coverage=0.5,
            ),
        ),
        by_subject=(
            GroupedMetricRow(
                group_name="subject_id",
                group_value="subject_a",
                count=2,
                mae=1.0,
                rmse=1.0,
                bias=0.2,
                overall_pinball_loss=0.50,
                mean_interval_width=9.0,
                empirical_interval_coverage=1.0,
            ),
            GroupedMetricRow(
                group_name="subject_id",
                group_value="subject_b",
                count=2,
                mae=0.75,
                rmse=0.80,
                bias=-0.5,
                overall_pinball_loss=0.45,
                mean_interval_width=8.5,
                empirical_interval_coverage=1.0,
            ),
        ),
        by_glucose_range=(
            GroupedMetricRow(
                group_name="glucose_range",
                group_value="euglycemia",
                count=2,
                mae=0.6,
                rmse=0.7,
                bias=-0.1,
                overall_pinball_loss=0.40,
                mean_interval_width=8.0,
                empirical_interval_coverage=1.0,
            ),
            GroupedMetricRow(
                group_name="glucose_range",
                group_value="hyperglycemia",
                count=2,
                mae=1.1,
                rmse=1.2,
                bias=0.4,
                overall_pinball_loss=0.60,
                mean_interval_width=10.5,
                empirical_interval_coverage=0.5,
            ),
        ),
    )


def _build_shared_report(*, include_by_horizon: bool) -> SharedReport:
    """Build a compact shared report for reporting sink regression tests."""
    prediction_table = _build_prediction_table_frame()
    by_horizon = _build_by_horizon_frame() if include_by_horizon else pd.DataFrame()

    return SharedReport(
        scalars={
            "num_prediction_rows": len(prediction_table),
            "num_subjects": int(prediction_table["subject_id"].nunique()),
            "num_horizons": int(prediction_table["horizon_index"].nunique()),
        },
        tables={
            "prediction_table": prediction_table,
            "by_horizon": by_horizon,
            "by_subject": _build_by_subject_frame(),
            "by_glucose_range": _build_by_glucose_range_frame(),
        },
        text={
            "dataset_overview": "Synthetic shared report for reporting tests.",
            "metric_overview": "Synthetic grouped metrics for reporting tests.",
            "quantile_overview": "Quantiles: 0.1, 0.5, 0.9",
            "horizon_overview": "Synthetic horizon text.",
            "probabilistic_overview": "Synthetic probabilistic text.",
            "subject_variability_overview": "Synthetic subject text.",
            "glucose_range_overview": "Synthetic glucose-range text.",
        },
        figures={},
        metadata={
            "num_prediction_batches": 1,
            "quantiles": (0.1, 0.5, 0.9),
            "sampling_interval_minutes": 5,
            "has_evaluation_result": include_by_horizon,
        },
    )


def _text_by_tag(
    experiment: StubTensorBoardExperiment,
) -> dict[str, tuple[str, str, int]]:
    """Index captured text calls by tag for concise TensorBoard assertions."""
    return {tag: (tag, text, global_step) for tag, text, global_step in experiment.texts}


def _figure_tags(experiment: StubTensorBoardExperiment) -> set[str]:
    """Return the logged TensorBoard figure tags as a simple lookup set."""
    return {tag for tag, _, _ in experiment.figures}


def test_export_prediction_table_writes_analysis_friendly_rows(tmp_path: Path) -> None:
    predictions = [
        torch.tensor(
            [
                [[95.0, 100.0, 105.0], [96.0, 101.0, 106.0]],
                [[115.0, 120.0, 125.0], [116.0, 121.0, 126.0]],
            ],
            dtype=torch.float32,
        )
    ]
    datamodule = StubDataModule(
        [
            {
                "target": torch.tensor(
                    [
                        [[102.0], [103.0]],
                        [[118.0], [119.0]],
                    ],
                    dtype=torch.float32,
                ),
                "metadata": {
                    "subject_id": ["subject_a", "subject_b"],
                    "decoder_start": [
                        "2026-01-01 00:00:00",
                        "2026-01-02 00:00:00",
                    ],
                    "decoder_end": [
                        "2026-01-01 00:05:00",
                        "2026-01-02 00:05:00",
                    ],
                },
            }
        ]
    )

    output_path = export_prediction_table(
        datamodule=datamodule,
        predictions=predictions,
        quantiles=(0.1, 0.5, 0.9),
        output_path=tmp_path / "test_predictions.csv",
        sampling_interval_minutes=5,
    )

    assert output_path == tmp_path / "test_predictions.csv"
    assert output_path is not None
    frame = pd.read_csv(output_path)
    assert len(frame) == 4
    assert {
        "subject_id",
        "timestamp",
        "target",
        "pred_q10",
        "pred_q50",
        "pred_q90",
        "median_prediction",
        "residual",
        "prediction_interval_width",
    }.issubset(frame.columns)
    assert frame.loc[0, "median_prediction"] == 100.0
    assert frame.loc[0, "residual"] == -2.0
    assert frame.loc[0, "prediction_interval_width"] == 10.0


def test_export_prediction_table_from_report_writes_canonical_prediction_table(
    tmp_path: Path,
) -> None:
    shared_report = _build_shared_report(include_by_horizon=True)

    output_path = export_prediction_table_from_report(
        shared_report=shared_report,
        output_path=tmp_path / "report_predictions.csv",
    )

    assert output_path == tmp_path / "report_predictions.csv"
    assert output_path is not None
    frame = pd.read_csv(output_path)
    assert len(frame) == 4
    assert set(shared_report.tables["prediction_table"].columns).issubset(frame.columns)


def test_export_grouped_tables_from_report_writes_grouped_csvs(tmp_path: Path) -> None:
    shared_report = _build_shared_report(include_by_horizon=True)

    written_paths = export_grouped_tables_from_report(
        shared_report=shared_report,
        output_dir=tmp_path / "grouped_reports",
    )

    assert set(written_paths) == {"by_horizon", "by_subject", "by_glucose_range"}
    for path in written_paths.values():
        assert path.exists()
        assert path.suffix == ".csv"


def test_build_shared_report_adds_richer_canonical_text_keys() -> None:
    predictions = [
        torch.tensor(
            [
                [[95.0, 100.0, 105.0], [96.0, 101.0, 106.0]],
                [[115.0, 120.0, 125.0], [116.0, 121.0, 126.0]],
            ],
            dtype=torch.float32,
        )
    ]
    datamodule = StubDataModule(
        [
            {
                "target": torch.tensor(
                    [
                        [[102.0], [103.0]],
                        [[118.0], [119.0]],
                    ],
                    dtype=torch.float32,
                ),
                "metadata": {
                    "subject_id": ["subject_a", "subject_b"],
                    "decoder_start": [
                        "2026-01-01 00:00:00",
                        "2026-01-02 00:00:00",
                    ],
                    "decoder_end": [
                        "2026-01-01 00:05:00",
                        "2026-01-02 00:05:00",
                    ],
                },
            }
        ]
    )

    shared_report = build_shared_report(
        datamodule=datamodule,
        predictions=predictions,
        quantiles=(0.1, 0.5, 0.9),
        sampling_interval_minutes=5,
        evaluation_result=_build_evaluation_result(),
    )

    assert {
        "dataset_overview",
        "metric_overview",
        "quantile_overview",
        "horizon_overview",
        "probabilistic_overview",
        "subject_variability_overview",
        "glucose_range_overview",
    }.issubset(shared_report.text)
    assert "MAE" in shared_report.text["horizon_overview"]
    assert "Probabilistic overview" in shared_report.text["probabilistic_overview"]


def test_generate_plotly_reports_creates_all_expected_artifacts_from_shared_report(
    tmp_path: Path,
) -> None:
    pytest.importorskip("plotly")

    shared_report = _build_shared_report(include_by_horizon=True)

    report_paths = generate_plotly_reports(
        None,
        report_dir=tmp_path / "reports",
        max_subjects=1,
        shared_report=shared_report,
    )

    assert set(report_paths) == {
        "residual_histogram",
        "horizon_metrics",
        "horizon_bias",
        "horizon_coverage",
        "subject_metrics",
        "glucose_range_metrics",
        "forecast_overview",
    }
    for path in report_paths.values():
        assert path.exists()
        assert path.suffix == ".html"


def test_generate_plotly_reports_skips_horizon_metrics_without_canonical_grouped_data(
    tmp_path: Path,
) -> None:
    pytest.importorskip("plotly")

    shared_report = _build_shared_report(include_by_horizon=False)

    report_paths = generate_plotly_reports(
        None,
        report_dir=tmp_path / "reports",
        max_subjects=1,
        shared_report=shared_report,
    )

    assert set(report_paths) == {
        "residual_histogram",
        "subject_metrics",
        "glucose_range_metrics",
        "forecast_overview",
    }
    assert "horizon_metrics" not in report_paths
    assert "horizon_bias" not in report_paths
    assert "horizon_coverage" not in report_paths

    for path in report_paths.values():
        assert path.exists()
        assert path.suffix == ".html"


def test_log_shared_report_to_tensorboard_logs_expected_scalars_text_tables_and_figures() -> None:
    """Validate the main TensorBoard sink path against the canonical shared report.

    Purpose:
        Protect the new TensorBoard reporting surface so later reporting changes
        do not silently drop key interpretation panels or grouped diagnostics.
    """
    shared_report = _build_shared_report(include_by_horizon=True)
    experiment = StubTensorBoardExperiment()
    logger = StubTensorBoardLogger(experiment)

    logged = log_shared_report_to_tensorboard(
        shared_report=shared_report,
        logger_or_trainer=logger,
        global_step=7,
        namespace="report",
        max_table_rows=3,
        max_forecast_subjects=2,
    )

    assert logged is True

    scalar_tags = {tag for tag, _, _ in experiment.scalars}
    assert {
        "report/scalars/num_prediction_rows",
        "report/scalars/num_subjects",
        "report/scalars/num_horizons",
    }.issubset(scalar_tags)

    text_calls = _text_by_tag(experiment)
    assert "report/text/index" in text_calls
    assert "Available report text panels:" in text_calls["report/text/index"][1]

    assert "report/text/dataset_overview" in text_calls
    assert "report/text/metric_overview" in text_calls
    assert "report/text/quantile_overview" in text_calls
    assert "report/text/horizon_overview" in text_calls
    assert "report/text/probabilistic_overview" in text_calls
    assert "report/text/subject_variability_overview" in text_calls
    assert "report/text/glucose_range_overview" in text_calls
    assert "report/text/metadata" in text_calls

    # Table previews should still be present so TensorBoard preserves the raw
    # drill-down surface alongside higher-level interpretation panels.
    assert "report/tables/prediction_table" in text_calls
    assert "report/tables/by_horizon" in text_calls
    assert "report/tables/by_subject" in text_calls
    assert "report/tables/by_glucose_range" in text_calls

    figure_tags = _figure_tags(experiment)
    assert {
        "report/figures/residual_histogram",
        "report/figures/horizon_metrics",
        "report/figures/horizon_uncertainty",
        "report/figures/horizon_bias",
        "report/figures/subject_mae",
        "report/figures/subject_bias",
        "report/figures/subject_rmse",
        "report/figures/glucose_range_mae",
        "report/figures/glucose_range_bias",
        "report/figures/glucose_range_interval_width",
        "report/figures/glucose_range_coverage",
        "report/figures/forecast_overview",
    }.issubset(figure_tags)


def test_log_shared_report_to_tensorboard_orders_text_panels_interpretation_first() -> None:
    """Ensure the sink's text panels are emitted in a stable, meaningful order.

    Context:
        The text tab is easier to navigate when broad report interpretation
        appears first and any extra custom text blocks are appended afterward in
        deterministic order.
    """
    shared_report = _build_shared_report(include_by_horizon=True)
    shared_report.text["zzz_custom_appendix"] = "Synthetic appendix text."

    experiment = StubTensorBoardExperiment()
    logger = StubTensorBoardLogger(experiment)

    logged = log_shared_report_to_tensorboard(
        shared_report=shared_report,
        logger_or_trainer=logger,
        global_step=1,
        namespace="report",
    )

    assert logged is True

    text_tags = [tag for tag, _, _ in experiment.texts]
    report_text_tags = [tag for tag in text_tags if tag.startswith("report/text/")]

    # The index should lead the text surface, followed by the canonical report
    # interpretation blocks in their preferred order, and only then any extra
    # custom text sections.
    assert report_text_tags[0] == "report/text/index"
    assert report_text_tags[1:8] == [
        "report/text/dataset_overview",
        "report/text/metric_overview",
        "report/text/quantile_overview",
        "report/text/horizon_overview",
        "report/text/probabilistic_overview",
        "report/text/subject_variability_overview",
        "report/text/glucose_range_overview",
    ]
    assert "report/text/zzz_custom_appendix" in report_text_tags
    assert report_text_tags.index("report/text/zzz_custom_appendix") > report_text_tags.index(
        "report/text/glucose_range_overview"
    )


def test_log_shared_report_to_tensorboard_skips_incompatible_logger_backends() -> None:
    """Confirm the sink returns False when no TensorBoard-compatible logger exists."""
    shared_report = _build_shared_report(include_by_horizon=True)

    logged = log_shared_report_to_tensorboard(
        shared_report=shared_report,
        logger_or_trainer=StubNonTensorBoardLogger(),
    )

    assert logged is False


def test_log_shared_report_to_tensorboard_gracefully_skips_missing_grouped_metric_figures() -> None:
    """Validate best-effort figure logging when some grouped metric columns are absent.

    Context:
        The sink should preserve all unaffected report surfaces while omitting
        only the figures whose required canonical columns are missing.
    """
    shared_report = _build_shared_report(include_by_horizon=True)

    # Remove one column from each grouped table to force the newly added figure
    # builders to opt out while leaving the rest of the report intact.
    shared_report.tables["by_subject"] = shared_report.tables["by_subject"].drop(
        columns=["rmse"]
    )
    shared_report.tables["by_glucose_range"] = shared_report.tables["by_glucose_range"].drop(
        columns=["mean_interval_width"]
    )

    experiment = StubTensorBoardExperiment()
    logger = StubTensorBoardLogger(experiment)

    logged = log_shared_report_to_tensorboard(
        shared_report=shared_report,
        logger_or_trainer=logger,
        namespace="report",
    )

    assert logged is True

    figure_tags = _figure_tags(experiment)

    # Existing unaffected figures should still be logged.
    assert "report/figures/subject_mae" in figure_tags
    assert "report/figures/subject_bias" in figure_tags
    assert "report/figures/glucose_range_mae" in figure_tags
    assert "report/figures/glucose_range_bias" in figure_tags
    assert "report/figures/glucose_range_coverage" in figure_tags

    # The newly added figures whose required columns were removed should be
    # omitted rather than causing the sink to fail.
    assert "report/figures/subject_rmse" not in figure_tags
    assert "report/figures/glucose_range_interval_width" not in figure_tags
