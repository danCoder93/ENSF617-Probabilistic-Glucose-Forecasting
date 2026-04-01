from __future__ import annotations

# AI-assisted maintenance note:
# This module owns observability artifacts that are generated *after* model
# predictions already exist.
#
# Responsibility boundary:
# - export prediction tensors into an analysis-friendly CSV table
# - generate lightweight Plotly HTML reports from that table
#
# Why keep this apart from callbacks:
# - callbacks run inside the live Trainer loop
# - these helpers operate after the run on persisted prediction outputs
#
# Keeping that lifecycle boundary explicit makes it easier to reason about what
# can fail without affecting training itself and what artifacts are expected to
# exist only after prediction generation completes.

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch import Tensor

from config import PathInput
from data.datamodule import AZT1DDataModule
from observability.tensors import _as_metadata_lists
from observability.utils import _has_module


# ============================================================================
# Post-Run Prediction Export
# ============================================================================
# The helpers below operate after prediction batches have already been
# produced. They turn raw tensors into analysis-friendly artifacts.
def export_prediction_table(
    *,
    datamodule: AZT1DDataModule,
    predictions: Sequence[Tensor],
    quantiles: Sequence[float],
    output_path: PathInput | None,
    sampling_interval_minutes: int,
) -> Path | None:
    """
    Export test predictions as a flat analysis-friendly CSV table.

    Context:
    the raw tensor dump preserves fidelity, while this table optimizes for
    plotting, inspection, and report generation.
    """
    # This export deliberately denormalizes prediction results into one flat
    # row-per-horizon table because that format is easy to inspect in a
    # notebook, easy to plot with Plotly/pandas, and easy to archive as a run
    # artifact.
    #
    # It complements the raw tensor dump written elsewhere in the workflow:
    # - raw `.pt` files preserve full tensor fidelity for PyTorch consumers
    # - this CSV prioritizes analysis convenience
    if output_path is None:
        return None
    output_path = Path(output_path)
    if not predictions:
        return None

    rows: list[dict[str, Any]] = []
    test_loader = datamodule.test_dataloader()
    quantile_columns = [f"pred_q{int(round(q * 100)):02d}" for q in quantiles]
    median_index = min(
        range(len(quantiles)),
        key=lambda index: abs(float(quantiles[index]) - 0.5),
    )

    # The exported table intentionally lines up predictions with the original
    # test dataloader batches so metadata such as subject ID and decoder start
    # can be attached row by row.
    for batch_index, (prediction_batch, batch) in enumerate(zip(predictions, test_loader)):
        prediction_cpu = prediction_batch.detach().cpu()
        target = batch["target"]
        if isinstance(target, Tensor):
            target_cpu = target.detach().cpu()
        else:
            target_cpu = torch.as_tensor(target)
        if target_cpu.ndim == 3 and target_cpu.shape[-1] == 1:
            target_cpu = target_cpu.squeeze(-1)

        batch_size = int(prediction_cpu.shape[0])
        metadata = _as_metadata_lists(batch["metadata"], batch_size)

        for sample_index in range(batch_size):
            subject_id = str(metadata.get("subject_id", ["unknown"])[sample_index])
            decoder_start = pd.Timestamp(
                str(
                    metadata.get("decoder_start", ["1970-01-01 00:00:00"])[sample_index]
                )
            )
            for horizon_index in range(int(prediction_cpu.shape[1])):
                # The export is intentionally one row per forecast horizon step
                # rather than one row per sample window. That denormalized
                # shape is what makes later plotting and grouped metric
                # analysis with pandas/Plotly straightforward.
                timestamp = decoder_start + pd.Timedelta(
                    minutes=sampling_interval_minutes * horizon_index
                )
                row = {
                    "prediction_batch_index": batch_index,
                    "sample_index_within_batch": sample_index,
                    "subject_id": subject_id,
                    "decoder_start": str(metadata.get("decoder_start", [""])[sample_index]),
                    "decoder_end": str(metadata.get("decoder_end", [""])[sample_index]),
                    "timestamp": timestamp.isoformat(),
                    "horizon_index": horizon_index,
                    "target": float(target_cpu[sample_index, horizon_index].item()),
                }
                for quantile_index, column_name in enumerate(quantile_columns):
                    row[column_name] = float(
                        prediction_cpu[
                            sample_index,
                            horizon_index,
                            quantile_index,
                        ].item()
                    )
                row["median_prediction"] = float(
                    prediction_cpu[sample_index, horizon_index, median_index].item()
                )
                row["residual"] = row["median_prediction"] - row["target"]
                if len(quantile_columns) >= 2:
                    row["prediction_interval_width"] = (
                        row[quantile_columns[-1]] - row[quantile_columns[0]]
                    )
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


# ============================================================================
# Post-Run Report Generation
# ============================================================================
# These reports are intentionally lightweight first-pass visual artifacts. They
# are meant to make a run inspectable immediately, not to replace a full
# analytics notebook.
def generate_plotly_reports(
    prediction_table_path: PathInput | None,
    *,
    report_dir: PathInput | None,
    max_subjects: int,
) -> dict[str, Path]:
    """
    Generate lightweight Plotly HTML reports from the exported prediction
    table.

    Context:
    these reports are intended to make each run immediately inspectable
    without requiring a separate notebook.
    """
    # These reports are intentionally lightweight first-pass diagnostics, not a
    # complete experiment-reporting system. The aim is to generate a few useful
    # HTML artifacts automatically from the flat prediction table so every run
    # leaves behind something visual and shareable.
    if prediction_table_path is None or report_dir is None:
        return {}
    prediction_table_path = Path(prediction_table_path)
    report_dir = Path(report_dir)
    if not prediction_table_path.exists():
        return {}
    if not _has_module("plotly"):
        return {}

    import plotly.express as px
    import plotly.graph_objects as go

    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(prediction_table_path)
    if frame.empty:
        return {}

    report_paths: dict[str, Path] = {}

    residual_histogram = px.histogram(
        frame,
        x="residual",
        nbins=50,
        title="Residual Distribution",
    )
    residual_histogram_path = report_dir / "residual_histogram.html"
    residual_histogram.write_html(str(residual_histogram_path))
    report_paths["residual_histogram"] = residual_histogram_path

    grouped = frame.assign(abs_error=lambda data: data["residual"].abs()).groupby(
        "horizon_index",
        as_index=False,
    )
    # Grouping by horizon index gives us a simple answer to one of the most
    # important forecasting diagnostics questions:
    # "How does error behave as we predict farther into the future?"
    #
    # That horizon-wise view is often more informative than one single global
    # metric because short-horizon and long-horizon behavior can differ a lot.
    aggregation: dict[str, Any] = {
        "mae": ("abs_error", "mean"),
        "rmse": ("residual", lambda values: float((values.pow(2).mean()) ** 0.5)),
    }
    if "prediction_interval_width" in frame.columns:
        aggregation["mean_interval_width"] = ("prediction_interval_width", "mean")
    horizon_metrics = grouped.agg(**aggregation)
    horizon_metrics_fig = go.Figure()
    horizon_metrics_fig.add_trace(
        go.Scatter(
            x=horizon_metrics["horizon_index"],
            y=horizon_metrics["mae"],
            mode="lines+markers",
            name="MAE",
        )
    )
    horizon_metrics_fig.add_trace(
        go.Scatter(
            x=horizon_metrics["horizon_index"],
            y=horizon_metrics["rmse"],
            mode="lines+markers",
            name="RMSE",
        )
    )
    if "mean_interval_width" in horizon_metrics:
        horizon_metrics_fig.add_trace(
            go.Scatter(
                x=horizon_metrics["horizon_index"],
                y=horizon_metrics["mean_interval_width"],
                mode="lines+markers",
                name="Mean Interval Width",
                yaxis="y2",
            )
        )
        horizon_metrics_fig.update_layout(
            yaxis2=dict(
                title="Interval Width",
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )
    horizon_metrics_fig.update_layout(title="Error Metrics By Forecast Horizon")
    horizon_metrics_path = report_dir / "horizon_metrics.html"
    horizon_metrics_fig.write_html(str(horizon_metrics_path))
    report_paths["horizon_metrics"] = horizon_metrics_path

    overview_fig = go.Figure()
    subject_ids = list(dict.fromkeys(frame["subject_id"].tolist()))[:max_subjects]
    filtered = frame[frame["subject_id"].isin(subject_ids)].copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
    filtered.sort_values(["subject_id", "timestamp"], inplace=True)

    for subject_id in subject_ids:
        subject_frame = filtered[filtered["subject_id"] == subject_id]
        if subject_frame.empty:
            continue
        overview_fig.add_trace(
            go.Scatter(
                x=subject_frame["timestamp"],
                y=subject_frame["target"],
                mode="lines",
                name=f"{subject_id} target",
            )
        )
        overview_fig.add_trace(
            go.Scatter(
                x=subject_frame["timestamp"],
                y=subject_frame["median_prediction"],
                mode="lines",
                name=f"{subject_id} median",
            )
        )
        quantile_columns = sorted(
            column for column in subject_frame.columns if column.startswith("pred_q")
        )
        if len(quantile_columns) >= 2:
            lower = quantile_columns[0]
            upper = quantile_columns[-1]
            overview_fig.add_trace(
                go.Scatter(
                    x=subject_frame["timestamp"],
                    y=subject_frame[upper],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            overview_fig.add_trace(
                go.Scatter(
                    x=subject_frame["timestamp"],
                    y=subject_frame[lower],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    opacity=0.2,
                    name=f"{subject_id} interval",
                )
            )

    overview_fig.update_layout(title="Forecast Overview")
    overview_path = report_dir / "forecast_overview.html"
    overview_fig.write_html(str(overview_path))
    report_paths["forecast_overview"] = overview_path

    return report_paths
