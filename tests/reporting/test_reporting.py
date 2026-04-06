from __future__ import annotations

# These tests protect the reporting/export helpers that run after predictions
# already exist.

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from evaluation.types import EvaluationResult, GroupedMetricRow, MetricSummary
from reporting import export_prediction_table, generate_plotly_reports


class StubDataModule:
    """Minimal datamodule exposing only the test-dataloader surface used by reporting helpers."""

    def __init__(self, test_batches: list[dict[str, object]]) -> None:
        """Store the synthetic test batches that should be exposed to the exporter."""
        self._test_batches = test_batches

    def test_dataloader(self) -> list[dict[str, object]]:
        """Return the stored synthetic batches as the held-out prediction surface."""
        return self._test_batches


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


def test_generate_plotly_reports_creates_html_artifacts_from_prediction_table(
    tmp_path: Path,
) -> None:
    # Plot generation should stay a thin artifact layer over the exported
    # prediction table plus optional evaluation summaries.
    pytest.importorskip("plotly")

    prediction_table_path = tmp_path / "predictions.csv"
    pd.DataFrame(
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
    ).to_csv(prediction_table_path, index=False)
    evaluation_result = EvaluationResult(
        summary=MetricSummary(
            count=4,
            mae=0.875,
            rmse=0.901,
            bias=0.125,
            overall_pinball_loss=0.5,
            mean_interval_width=9.0,
            empirical_interval_coverage=1.0,
        ),
        by_horizon=(
            GroupedMetricRow(
                group_name="horizon_index",
                group_value=0,
                count=2,
                mae=0.75,
                rmse=0.79,
                bias=-0.75,
                overall_pinball_loss=0.45,
                mean_interval_width=8.5,
                empirical_interval_coverage=1.0,
            ),
            GroupedMetricRow(
                group_name="horizon_index",
                group_value=1,
                count=2,
                mae=1.0,
                rmse=1.0,
                bias=1.0,
                overall_pinball_loss=0.55,
                mean_interval_width=9.5,
                empirical_interval_coverage=1.0,
            ),
        ),
        quantiles=(0.1, 0.5, 0.9),
    )

    report_paths = generate_plotly_reports(
        prediction_table_path,
        report_dir=tmp_path / "reports",
        max_subjects=1,
        evaluation_result=evaluation_result,
    )

    assert set(report_paths) == {
        "residual_histogram",
        "horizon_metrics",
        "forecast_overview",
    }
    for path in report_paths.values():
        assert path.exists()
        assert path.suffix == ".html"
