from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("artifacts/main_run")
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

pred_path = ROOT / "test_predictions.csv"
processed_path = Path("data/processed/azt1d_processed.csv")

if not pred_path.exists():
    raise FileNotFoundError(f"Could not find {pred_path}")
if not processed_path.exists():
    raise FileNotFoundError(f"Could not find {processed_path}")

pred = pd.read_csv(pred_path)
proc = pd.read_csv(processed_path)

required_pred_cols = ["subject_id", "timestamp", "target", "median_prediction"]
for col in required_pred_cols:
    if col not in pred.columns:
        raise ValueError(f"Missing required prediction column: {col}")

required_proc_cols = [
    "subject_id",
    "timestamp",
    "carbs_g",
    "bolus_insulin_u",
    "correction_insulin_u",
    "meal_insulin_u",
    "device_mode",
]
for col in required_proc_cols:
    if col not in proc.columns:
        raise ValueError(f"Missing required processed-data column: {col}")

# Normalize timestamps for a clean join.
pred["timestamp"] = pd.to_datetime(pred["timestamp"]).dt.floor("min")
proc["timestamp"] = pd.to_datetime(proc["timestamp"]).dt.floor("min")

proc_small = proc[
    [
        "subject_id",
        "timestamp",
        "carbs_g",
        "bolus_insulin_u",
        "correction_insulin_u",
        "meal_insulin_u",
        "device_mode",
    ]
].copy()

# CHANGE: The processed table can contain repeated subject/timestamp pairs.
# For this first-pass event analysis, collapse those repeats before merging so
# the prediction rows get one clean context record each.
proc_small = (
    proc_small.groupby(["subject_id", "timestamp"], as_index=False)
    .agg(
        {
            "carbs_g": "max",
            "bolus_insulin_u": "max",
            "correction_insulin_u": "max",
            "meal_insulin_u": "max",
            "device_mode": "first",
        }
    )
)

merged = pred.merge(
    proc_small,
    on=["subject_id", "timestamp"],
    how="left",
    validate="many_to_one",
)

# Simple event flags.
merged["meal_event"] = merged["carbs_g"].fillna(0) > 0
merged["insulin_event"] = (
    (merged["bolus_insulin_u"].fillna(0) > 0)
    | (merged["correction_insulin_u"].fillna(0) > 0)
    | (merged["meal_insulin_u"].fillna(0) > 0)
)

merged["abs_error"] = (merged["target"] - merged["median_prediction"]).abs()
merged["sq_error"] = (merged["target"] - merged["median_prediction"]) ** 2
merged["bias"] = merged["median_prediction"] - merged["target"]

thresholds = [10, 20, 30]
for t in thresholds:
    merged[f"within_{t}"] = (merged["abs_error"] <= t).astype(int)

def summarize_group(df: pd.DataFrame) -> dict:
    row = {
        "count": len(df),
        "mae": float(df["abs_error"].mean()),
        "rmse": float((df["sq_error"].mean()) ** 0.5),
        "bias": float(df["bias"].mean()),
    }
    for t in thresholds:
        row[f"pct_within_{t}mgdl"] = float(df[f"within_{t}"].mean() * 100.0)
    return row

def build_group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, g in df.groupby(group_col, dropna=False):
        row = {group_col: group_value}
        row.update(summarize_group(g))
        rows.append(row)
    return pd.DataFrame(rows)

# Build tables.
by_device_mode = build_group_table(merged, "device_mode").sort_values("mae")
by_meal_event = build_group_table(merged, "meal_event")
by_insulin_event = build_group_table(merged, "insulin_event")

by_device_mode.to_csv(OUT / "metrics_by_device_mode.csv", index=False)
by_meal_event.to_csv(OUT / "metrics_by_meal_event.csv", index=False)
by_insulin_event.to_csv(OUT / "metrics_by_insulin_event.csv", index=False)

# Simple summary markdown.
best_mode = by_device_mode.loc[by_device_mode["mae"].idxmin()]
worst_mode = by_device_mode.loc[by_device_mode["mae"].idxmax()]

meal_true = by_meal_event[by_meal_event["meal_event"] == True]
meal_false = by_meal_event[by_meal_event["meal_event"] == False]

ins_true = by_insulin_event[by_insulin_event["insulin_event"] == True]
ins_false = by_insulin_event[by_insulin_event["insulin_event"] == False]

def safe_val(df: pd.DataFrame, col: str):
    if df.empty:
        return None
    return float(df.iloc[0][col])

md = f"""# Event-Aware Analysis Summary

## Device mode
- Best device mode by MAE: **{best_mode['device_mode']}** at **{best_mode['mae']:.2f} mg/dL**
- Worst device mode by MAE: **{worst_mode['device_mode']}** at **{worst_mode['mae']:.2f} mg/dL**

## Meal events
- Non-meal MAE: **{safe_val(meal_false, 'mae'):.2f} mg/dL**
- Meal-event MAE: **{safe_val(meal_true, 'mae'):.2f} mg/dL**
- Non-meal within 20 mg/dL: **{safe_val(meal_false, 'pct_within_20mgdl'):.2f}%**
- Meal-event within 20 mg/dL: **{safe_val(meal_true, 'pct_within_20mgdl'):.2f}%**

## Insulin events
- Non-insulin-event MAE: **{safe_val(ins_false, 'mae'):.2f} mg/dL**
- Insulin-event MAE: **{safe_val(ins_true, 'mae'):.2f} mg/dL**
- Non-insulin-event within 20 mg/dL: **{safe_val(ins_false, 'pct_within_20mgdl'):.2f}%**
- Insulin-event within 20 mg/dL: **{safe_val(ins_true, 'pct_within_20mgdl'):.2f}%**

## Interpretation
This analysis shows whether the model behaves differently under more realistic operating conditions rather than only in aggregate.
If meal or insulin event performance is meaningfully worse, that gives us a concrete place to focus model improvement.
"""
(OUT / "event_aware_summary.md").write_text(md, encoding="utf-8")

# Plot 1: MAE by device mode
# Plot 1: MAE by device mode
plt.figure(figsize=(8, 5))

# CHANGE: Some rows can come through with missing device_mode after the join.
# Turn those into a readable label before plotting so matplotlib does not crash
# on mixed string/float category values.
device_labels = by_device_mode["device_mode"].fillna("missing").astype(str).tolist()

plt.bar(device_labels, by_device_mode["mae"])
plt.ylabel("MAE (mg/dL)")
plt.xlabel("Device mode")
plt.title("MAE by device mode")
plt.tight_layout()
plt.savefig(OUT / "mae_by_device_mode.png", dpi=180)
plt.close()

# Plot 2: MAE by event type
event_plot = pd.DataFrame(
    [
        {
            "group": "No meal",
            "mae": safe_val(meal_false, "mae"),
        },
        {
            "group": "Meal event",
            "mae": safe_val(meal_true, "mae"),
        },
        {
            "group": "No insulin",
            "mae": safe_val(ins_false, "mae"),
        },
        {
            "group": "Insulin event",
            "mae": safe_val(ins_true, "mae"),
        },
    ]
)

plt.figure(figsize=(8, 5))
plt.bar(event_plot["group"], event_plot["mae"])
plt.ylabel("MAE (mg/dL)")
plt.xlabel("Event group")
plt.title("MAE by meal and insulin event status")
plt.tight_layout()
plt.savefig(OUT / "mae_by_event_group.png", dpi=180)
plt.close()

# Save merged table too, because it is useful for follow-up analysis.
merged.to_csv(OUT / "predictions_with_event_context.csv", index=False)

print("Created:")
for name in [
    "metrics_by_device_mode.csv",
    "metrics_by_meal_event.csv",
    "metrics_by_insulin_event.csv",
    "event_aware_summary.md",
    "mae_by_device_mode.png",
    "mae_by_event_group.png",
    "predictions_with_event_context.csv",
]:
    print(OUT / name)