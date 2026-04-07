from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("artifacts/main_run")
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

pred_path = ROOT / "test_predictions.csv"
if not pred_path.exists():
    raise FileNotFoundError(f"Could not find {pred_path}")

df = pd.read_csv(pred_path)

required_cols = ["target", "median_prediction", "horizon_index"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df["abs_error"] = (df["target"] - df["median_prediction"]).abs()

thresholds = [10, 20, 30]
for t in thresholds:
    df[f"within_{t}"] = (df["abs_error"] <= t).astype(int)

def glucose_bucket(x: float) -> str:
    if x < 70:
        return "lt_70"
    if x <= 180:
        return "70_to_180"
    return "gt_180"

df["glucose_range"] = df["target"].apply(glucose_bucket)

# Overall summary
summary_row = {
    "count": len(df),
    "mae": float(df["abs_error"].mean()),
}
for t in thresholds:
    summary_row[f"pct_within_{t}mgdl"] = float(df[f"within_{t}"].mean() * 100.0)

summary_df = pd.DataFrame([summary_row])
summary_df.to_csv(OUT / "threshold_accuracy_summary.csv", index=False)

# By horizon
rows = []
for horizon, g in df.groupby("horizon_index"):
    row = {
        "horizon_index": horizon,
        "count": len(g),
        "mae": float(g["abs_error"].mean()),
    }
    for t in thresholds:
        row[f"pct_within_{t}mgdl"] = float(g[f"within_{t}"].mean() * 100.0)
    rows.append(row)

by_horizon = pd.DataFrame(rows).sort_values("horizon_index")
by_horizon.to_csv(OUT / "threshold_accuracy_by_horizon.csv", index=False)

# By glucose range
rows = []
for bucket, g in df.groupby("glucose_range"):
    row = {
        "glucose_range": bucket,
        "count": len(g),
        "mae": float(g["abs_error"].mean()),
    }
    for t in thresholds:
        row[f"pct_within_{t}mgdl"] = float(g[f"within_{t}"].mean() * 100.0)
    rows.append(row)

by_range = pd.DataFrame(rows)
by_range.to_csv(OUT / "threshold_accuracy_by_glucose_range.csv", index=False)

# Plot 1: threshold accuracy by horizon
plt.figure(figsize=(8, 5))
for t in thresholds:
    plt.plot(
        by_horizon["horizon_index"],
        by_horizon[f"pct_within_{t}mgdl"],
        marker="o",
        label=f"Within {t} mg/dL",
    )
plt.xlabel("Forecast horizon step")
plt.ylabel("Percent of predictions")
plt.title("Threshold accuracy falls as forecast horizon increases")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "threshold_accuracy_by_horizon.png", dpi=180)
plt.close()

# Plot 2: threshold accuracy by glucose range
x = range(len(by_range))
width = 0.25
plt.figure(figsize=(8, 5))
for i, t in enumerate(thresholds):
    plt.bar(
        [v + i * width for v in x],
        by_range[f"pct_within_{t}mgdl"],
        width=width,
        label=f"Within {t} mg/dL",
    )
plt.xticks([v + width for v in x], by_range["glucose_range"])
plt.ylabel("Percent of predictions")
plt.title("Threshold accuracy by glucose range")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "threshold_accuracy_by_glucose_range.png", dpi=180)
plt.close()

# Markdown summary
best_range = by_range.loc[by_range["pct_within_20mgdl"].idxmax()]
worst_range = by_range.loc[by_range["pct_within_20mgdl"].idxmin()]
best_horizon = by_horizon.loc[by_horizon["pct_within_20mgdl"].idxmax()]
worst_horizon = by_horizon.loc[by_horizon["pct_within_20mgdl"].idxmin()]

md = f"""# Threshold Accuracy Summary

## Overall
- MAE: **{summary_row['mae']:.2f} mg/dL**
- Within 10 mg/dL: **{summary_row['pct_within_10mgdl']:.2f}%**
- Within 20 mg/dL: **{summary_row['pct_within_20mgdl']:.2f}%**
- Within 30 mg/dL: **{summary_row['pct_within_30mgdl']:.2f}%**

## Best / worst range by 20 mg/dL threshold
- Best range: **{best_range['glucose_range']}** at **{best_range['pct_within_20mgdl']:.2f}%**
- Worst range: **{worst_range['glucose_range']}** at **{worst_range['pct_within_20mgdl']:.2f}%**

## Best / worst horizon by 20 mg/dL threshold
- Best horizon: **{int(best_horizon['horizon_index'])}** at **{best_horizon['pct_within_20mgdl']:.2f}%**
- Worst horizon: **{int(worst_horizon['horizon_index'])}** at **{worst_horizon['pct_within_20mgdl']:.2f}%**

## Interpretation
This analysis translates raw error into a more intuitive “close enough” framing.
It is especially useful for reporting because it shows how often predictions stay within practical error bands, not just what the average error is.
"""
(OUT / "threshold_accuracy_summary.md").write_text(md, encoding="utf-8")

print("Created:")
for name in [
    "threshold_accuracy_summary.csv",
    "threshold_accuracy_by_horizon.csv",
    "threshold_accuracy_by_glucose_range.csv",
    "threshold_accuracy_by_horizon.png",
    "threshold_accuracy_by_glucose_range.png",
    "threshold_accuracy_summary.md",
]:
    print(OUT / name)