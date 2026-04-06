from __future__ import annotations

# These tests protect the reporting/export helpers that run after predictions
# already exist.

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
DataFrame = pd.DataFrame

from reporting import (
    SharedReport,
    export_prediction_table,
    export_prediction_table_from_report,
    generate_plotly_reports,
)


class StubDataModule:
    """Minimal datamodule exposing only the test-dataloader surface used by reporting helpers."""

    def __init__(self, test_batches: list[dict[str, object]]) -> None:
        """Store the synthetic test batches that should be exposed to the exporter."""
        self._test_batches = test_batches

    def test_dataloader(self) -> list[dict[str, object]]:
        """Return the stored synthetic batches as the held-out prediction surface."""
        return self._test_batches


def _build_prediction_table_frame() -> DataFrame:
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


def _build_by_horizon_frame() -> DataFrame:
    """Build the canonical grouped horizon table used by strict Plotly tests.

    Purpose:
        Mirror the grouped horizon-report table shape that the stricter Plotly
        sink now expects from `SharedReport.tables["by_horizon"]`.

    Context:
        The Plotly sink no longer recomputes grouped horizon metrics from the
        flat prediction table. That behavior now belongs upstream in evaluation
        and shared-report packaging, so the tests must provide the grouped table
        explicitly when they expect a horizon-metrics artifact.
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
            "empirical_interval_coverage": [1.0, 1.0],
        }
    )


def _build_shared_report(*, include_by_horizon: bool) -> SharedReport:
    """Build a compact shared report for Plotly sink regression tests.

    Purpose:
        Provide one canonical in-memory report bundle that matches the stricter
        reporting architecture, where sinks consume `SharedReport` rather than
        recomputing grouped metrics internally.

    Args:
        include_by_horizon:
            Whether the returned report should include the canonical grouped
            horizon table required for the horizon-metrics Plotly artifact.
    """
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
            "by_subject": pd.DataFrame(),
            "by_glucose_range": pd.DataFrame(),
        },
        text={
            "dataset_overview": "Synthetic shared report for reporting tests.",
            "metric_overview": "Synthetic grouped metrics for reporting tests.",
            "quantile_overview": "Quantiles: 0.1, 0.5, 0.9",
        },
        figures={},
        metadata={
            "num_prediction_batches": 1,
            "quantiles": (0.1, 0.5, 0.9),
            "sampling_interval_minutes": 5,
            "has_evaluation_result": include_by_horizon,
        },
    )


def test_export_prediction_table_writes_analysis_friendly_rows(tmp_path: Path) -> None:
    # Prediction export is the bridge from batched quantile tensors to a flat
    # analysis table, so this test checks both schema and a few key derived
    # values.
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
    # The stricter canonical export path should serialize the already-built
    # shared-report prediction table directly without needing raw prediction
    # tensors or datamodule access at sink time.
    shared_report = _build_shared_report(include_by_horizon=True)

    output_path = export_prediction_table_from_report(
        shared_report=shared_report,
        output_path=tmp_path / "report_predictions.csv",
    )

    assert output_path == tmp_path / "report_predictions.csv"
    frame = pd.read_csv(output_path)
    assert len(frame) == 4
    assert set(shared_report.tables["prediction_table"].columns).issubset(frame.columns)


def test_generate_plotly_reports_creates_all_expected_artifacts_from_shared_report(
    tmp_path: Path,
) -> None:
    # Plot generation should consume the canonical shared report directly. When
    # the grouped horizon table is present, the stricter sink should emit the
    # horizon-metrics artifact in addition to the residual and overview plots.
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
        "forecast_overview",
    }
    for path in report_paths.values():
        assert path.exists()
        assert path.suffix == ".html"


def test_generate_plotly_reports_skips_horizon_metrics_without_canonical_grouped_data(
    tmp_path: Path,
) -> None:
    # The stricter Plotly contract no longer recomputes grouped horizon metrics
    # from the flat prediction table. If canonical `by_horizon` data is absent,
    # the sink should omit only that artifact while still producing the plots
    # that are legitimately derived from the prediction table itself.
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
        "forecast_overview",
    }
    assert "horizon_metrics" not in report_paths

    for path in report_paths.values():
        assert path.exists()
        assert path.suffix == ".html"