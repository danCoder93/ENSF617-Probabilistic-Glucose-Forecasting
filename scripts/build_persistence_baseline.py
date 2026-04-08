from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("artifacts/main_run")
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

pred_path = ROOT / "test_predictions.csv"
if not pred_path.exists():
    raise FileNotFoundError(f"Could not find {pred_path}")

df = pd.read_csv(pred_path)

required_cols = [
    "target",
    "horizon_index",
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Try common names for the last observed glucose column.
history_candidates = [
    "last_observed_glucose",
    "encoder_last_glucose",
    "history_last_glucose",
    "input_last_glucose",
    "context_last_glucose",
]

history_col = None
for col in history_candidates:
    if col in df.columns:
        history_col = col
        break

if history_col is None:
    raise ValueError(
        "Could not find a last-observed-glucose column in test_predictions.csv. "
        "Open the CSV and identify the column that holds the most recent glucose "
        "value from the encoder/history window."
    )

# Persistence baseline: predict the last observed glucose at every future step.
df["baseline_prediction"] = df[history_col]
df["baseline_abs_error"] = (df["target"] - df["baseline_prediction"]).abs()
df["baseline_sq_error"] = (df["target"] - df["baseline_prediction"]) ** 2
df["baseline_bias"] = df["baseline_prediction"] - df["target"]

# Try to compare against model median prediction if present.
model_pred_candidates = [
    "prediction_median",
    "median_prediction",
    "pred_q50",
    "q50_prediction",
    "prediction_p50",
]

model_pred_col = None
for col in model_pred_candidates:
    if col in df.columns:
        model_pred_col = col
        break

if model_pred_col is not None:
    df["model_abs_error"] = (df["target"] - df[model_pred_col]).abs()
    df["model_sq_error"] = (df["target"] - df[model_pred_col]) ** 2
    df["model_bias"] = df[model_pred_col] - df["target"]

def summarize(group: pd.DataFrame, pred_type: str) -> dict:
    if pred_type == "baseline":
        ae = group["baseline_abs_error"]
        se = group["baseline_sq_error"]
        bias = group["baseline_bias"]
    else:
        ae = group["model_abs_error"]
        se = group["model_sq_error"]
        bias = group["model_bias"]

    return {
        "count": len(group),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(se.mean())),
        "bias": float(bias.mean()),
    }

# Overall baseline summary
baseline_summary = summarize(df, "baseline")

summary_payload = {
    "baseline_type": "persistence",
    "history_column_used": history_col,
    "overall_metrics": baseline_summary,
}

if model_pred_col is not None:
    model_summary = summarize(df, "model")
    summary_payload["model_prediction_column_used"] = model_pred_col
    summary_payload["model_overall_metrics"] = model_summary
    summary_payload["delta_model_minus_baseline"] = {
        "mae": model_summary["mae"] - baseline_summary["mae"],
        "rmse": model_summary["rmse"] - baseline_summary["rmse"],
        "bias": model_summary["bias"] - baseline_summary["bias"],
    }

summary_path = OUT / "baseline_persistence_metrics.json"
summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

# By horizon
rows = []
for horizon, g in df.groupby("horizon_index"):
    row = {"horizon_index": horizon}
    row.update({f"baseline_{k}": v for k, v in summarize(g, "baseline").items()})
    if model_pred_col is not None:
        row.update({f"model_{k}": v for k, v in summarize(g, "model").items()})
        row["delta_mae"] = row["model_mae"] - row["baseline_mae"]
        row["delta_rmse"] = row["model_rmse"] - row["baseline_rmse"]
    rows.append(row)

by_horizon = pd.DataFrame(rows).sort_values("horizon_index")
by_horizon.to_csv(OUT / "baseline_metrics_by_horizon.csv", index=False)

# Glucose range buckets on target
def glucose_bucket(x: float) -> str:
    if x < 70:
        return "lt_70"
    if x <= 180:
        return "70_to_180"
    return "gt_180"

df["glucose_range"] = df["target"].apply(glucose_bucket)

rows = []
for bucket, g in df.groupby("glucose_range"):
    row = {"glucose_range": bucket}
    row.update({f"baseline_{k}": v for k, v in summarize(g, "baseline").items()})
    if model_pred_col is not None:
        row.update({f"model_{k}": v for k, v in summarize(g, "model").items()})
        row["delta_mae"] = row["model_mae"] - row["baseline_mae"]
        row["delta_rmse"] = row["model_rmse"] - row["baseline_rmse"]
    rows.append(row)

by_range = pd.DataFrame(rows)
by_range.to_csv(OUT / "baseline_metrics_by_glucose_range.csv", index=False)

# Combined summary table
if model_pred_col is not None:
    combined = pd.DataFrame([
        {
            "comparison": "overall",
            "baseline_mae": baseline_summary["mae"],
            "model_mae": model_summary["mae"],
            "delta_mae": model_summary["mae"] - baseline_summary["mae"],
            "baseline_rmse": baseline_summary["rmse"],
            "model_rmse": model_summary["rmse"],
            "delta_rmse": model_summary["rmse"] - baseline_summary["rmse"],
        }
    ])
    combined.to_csv(OUT / "baseline_vs_model_summary.csv", index=False)

    # Plot 1: MAE by horizon
    plt.figure(figsize=(8, 5))
    plt.plot(by_horizon["horizon_index"], by_horizon["baseline_mae"], marker="o", label="Persistence baseline")
    plt.plot(by_horizon["horizon_index"], by_horizon["model_mae"], marker="o", label="Fused model")
    plt.xlabel("Forecast horizon step")
    plt.ylabel("MAE (mg/dL)")
    plt.title("Model vs persistence baseline by horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "model_vs_baseline_mae_by_horizon.png", dpi=180)
    plt.close()

    # Plot 2: RMSE by horizon
    plt.figure(figsize=(8, 5))
    plt.plot(by_horizon["horizon_index"], by_horizon["baseline_rmse"], marker="o", label="Persistence baseline")
    plt.plot(by_horizon["horizon_index"], by_horizon["model_rmse"], marker="o", label="Fused model")
    plt.xlabel("Forecast horizon step")
    plt.ylabel("RMSE (mg/dL)")
    plt.title("Model vs persistence baseline RMSE by horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "model_vs_baseline_rmse_by_horizon.png", dpi=180)
    plt.close()

    # Plot 3: MAE by glucose range
    x = np.arange(len(by_range))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, by_range["baseline_mae"], width=width, label="Persistence baseline")
    plt.bar(x + width / 2, by_range["model_mae"], width=width, label="Fused model")
    plt.xticks(x, by_range["glucose_range"])
    plt.ylabel("MAE (mg/dL)")
    plt.title("Model vs persistence baseline by glucose range")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "model_vs_baseline_mae_by_glucose_range.png", dpi=180)
    plt.close()

    # Markdown summary
    overall_delta_mae = model_summary["mae"] - baseline_summary["mae"]
    overall_delta_rmse = model_summary["rmse"] - baseline_summary["rmse"]

    md = f"""# Baseline Comparison Summary

## Baseline
Persistence baseline using `{history_col}` as the forecast for all future steps.

## Overall comparison
- Baseline MAE: **{baseline_summary["mae"]:.2f} mg/dL**
- Model MAE: **{model_summary["mae"]:.2f} mg/dL**
- Delta MAE (model - baseline): **{overall_delta_mae:.2f} mg/dL**

- Baseline RMSE: **{baseline_summary["rmse"]:.2f} mg/dL**
- Model RMSE: **{model_summary["rmse"]:.2f} mg/dL**
- Delta RMSE (model - baseline): **{overall_delta_rmse:.2f} mg/dL**

## Interpretation
A negative delta means the fused model beat the persistence baseline.
A positive delta means the persistence baseline was better.
"""
    (OUT / "baseline_comparison_summary.md").write_text(md, encoding="utf-8")

print(f"Created {summary_path}")
print(f"Created {OUT / 'baseline_metrics_by_horizon.csv'}")
print(f"Created {OUT / 'baseline_metrics_by_glucose_range.csv'}")
if model_pred_col is not None:
    print(f"Created {OUT / 'baseline_vs_model_summary.csv'}")
    print(f"Created {OUT / 'model_vs_baseline_mae_by_horizon.png'}")
    print(f"Created {OUT / 'model_vs_baseline_rmse_by_horizon.png'}")
    print(f"Created {OUT / 'model_vs_baseline_mae_by_glucose_range.png'}")
    print(f"Created {OUT / 'baseline_comparison_summary.md'}")