from __future__ import annotations

"""
Build one post-run health + comparison summary from a completed artifact directory.

Purpose:
- combine run-level metrics and epoch-level training dynamics into one clean summary
- make lightweight hyperparameter decisions easier after each run
- give one CSV row per run that can be appended into a tracker over time

Outputs:
- run_health_summary.json
- run_health_summary.csv
- optional appended tracker CSV

What this script is trying to answer:
- did the model peak early?
- did training longer help?
- is there a train/val gap?
- how bad is horizon degradation?
- how bad is low-glucose performance?
- are intervals too wide / too narrow?
- did the model beat or lose to persistence?
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------
# Basic file readers
# ---------------------------------------------------------------------
def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


# ---------------------------------------------------------------------
# Checkpoint / epoch parsing
# ---------------------------------------------------------------------
def _extract_best_epoch(best_checkpoint_path: str | None) -> int | None:
    """
    Extract epoch number from checkpoint names like:
    - epoch=epoch=02-val_loss=val_loss=4.7425.ckpt
    - epoch=02-val_loss=4.7425.ckpt
    """
    if not best_checkpoint_path:
        return None

    filename = Path(best_checkpoint_path).name
    patterns = [
        r"epoch=epoch=(\d+)",
        r"epoch=(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def _extract_best_val_loss_from_checkpoint(best_checkpoint_path: str | None) -> float | None:
    """
    Extract val loss from checkpoint names like:
    - epoch=epoch=02-val_loss=val_loss=4.7425.ckpt
    - epoch=02-val_loss=4.7425.ckpt
    """
    if not best_checkpoint_path:
        return None

    filename = Path(best_checkpoint_path).name
    patterns = [
        r"val_loss=val_loss=([0-9.]+)",
        r"val_loss=([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _parse_epoch_history_from_run_log(run_log_text: str) -> pd.DataFrame:
    """
    Parse epoch-level train/val stats from run.log / console-like text.

    Expected patterns include lines like:
    Epoch 0/19 ... val_loss: 6.874 val_mae: 19.499 train_loss_epoch: 9.825

    This parser is intentionally forgiving and only extracts what it can find.
    """
    rows: list[dict[str, Any]] = []

    # Split into lines and parse only lines that appear to contain epoch summaries.
    for line in run_log_text.splitlines():
        if "Epoch " not in line:
            continue
        if "val_loss:" not in line and "train_loss_epoch:" not in line:
            continue

        epoch_match = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", line)
        val_loss_match = re.search(r"val_loss:\s*([0-9.]+)", line)
        val_mae_match = re.search(r"val_mae:\s*([0-9.]+)", line)
        train_loss_epoch_match = re.search(r"train_loss_epoch:\s*([0-9.]+)", line)
        train_loss_step_match = re.search(r"train_loss_step:\s*([0-9.]+)", line)

        row: dict[str, Any] = {}
        if epoch_match:
            row["epoch"] = int(epoch_match.group(1))
            row["max_epoch_index"] = int(epoch_match.group(2))
        if val_loss_match:
            row["val_loss"] = float(val_loss_match.group(1))
        if val_mae_match:
            row["val_mae"] = float(val_mae_match.group(1))
        if train_loss_epoch_match:
            row["train_loss_epoch"] = float(train_loss_epoch_match.group(1))
        if train_loss_step_match:
            row["train_loss_step_snapshot"] = float(train_loss_step_match.group(1))

        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).drop_duplicates(subset=["epoch"], keep="last")
    if "epoch" in frame.columns:
        frame = frame.sort_values("epoch").reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------
# Report table helpers
# ---------------------------------------------------------------------
def _read_optional_table_from_report_paths(run_summary: dict[str, Any], key: str) -> pd.DataFrame:
    report_paths = _get_nested(run_summary, "observability", "report_paths")
    if not isinstance(report_paths, dict):
        return pd.DataFrame()

    path_str = report_paths.get(key)
    if not path_str:
        return pd.DataFrame()

    return _safe_read_csv(Path(path_str))


def _extract_metric_row(
    frame: pd.DataFrame,
    *,
    key_column: str = "group_value",
    key_value: Any,
) -> dict[str, Any]:
    if frame.empty or key_column not in frame.columns:
        return {}
    subset = frame[frame[key_column] == key_value]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _extract_horizon_metrics(by_horizon: pd.DataFrame, horizon: int) -> dict[str, Any]:
    row = _extract_metric_row(by_horizon, key_value=horizon)
    return {
        f"h{horizon}_count": _safe_float(row.get("count")),
        f"h{horizon}_mae": _safe_float(row.get("mae")),
        f"h{horizon}_rmse": _safe_float(row.get("rmse")),
        f"h{horizon}_bias": _safe_float(row.get("bias")),
        f"h{horizon}_pinball": _safe_float(row.get("overall_pinball_loss")),
        f"h{horizon}_interval_width": _safe_float(row.get("mean_interval_width")),
        f"h{horizon}_coverage": _safe_float(row.get("empirical_interval_coverage")),
    }


def _extract_glucose_range_metrics(by_glucose_range: pd.DataFrame, group_value: str, prefix: str) -> dict[str, Any]:
    row = _extract_metric_row(by_glucose_range, key_value=group_value)
    return {
        f"{prefix}_count": _safe_float(row.get("count")),
        f"{prefix}_mae": _safe_float(row.get("mae")),
        f"{prefix}_rmse": _safe_float(row.get("rmse")),
        f"{prefix}_bias": _safe_float(row.get("bias")),
        f"{prefix}_pinball": _safe_float(row.get("overall_pinball_loss")),
        f"{prefix}_interval_width": _safe_float(row.get("mean_interval_width")),
        f"{prefix}_coverage": _safe_float(row.get("empirical_interval_coverage")),
    }


def _compute_horizon_trend_features(by_horizon: pd.DataFrame) -> dict[str, Any]:
    if by_horizon.empty or "group_value" not in by_horizon.columns:
        return {
            "horizon_mae_delta_last_minus_first": None,
            "horizon_rmse_delta_last_minus_first": None,
            "horizon_coverage_delta_last_minus_first": None,
            "horizon_interval_width_delta_last_minus_first": None,
        }

    frame = by_horizon.sort_values("group_value").reset_index(drop=True)
    first = frame.iloc[0]
    last = frame.iloc[-1]

    def _delta(col: str) -> float | None:
        a = _safe_float(first.get(col))
        b = _safe_float(last.get(col))
        if a is None or b is None:
            return None
        return b - a

    return {
        "horizon_mae_delta_last_minus_first": _delta("mae"),
        "horizon_rmse_delta_last_minus_first": _delta("rmse"),
        "horizon_coverage_delta_last_minus_first": _delta("empirical_interval_coverage"),
        "horizon_interval_width_delta_last_minus_first": _delta("mean_interval_width"),
    }


# ---------------------------------------------------------------------
# Optional external summaries
# ---------------------------------------------------------------------
def _read_first_existing_json(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        if path.exists():
            return _safe_read_json(path)
    return {}


def _read_persistence_summary(artifact_dir: Path) -> dict[str, Any]:
    return _read_first_existing_json(
        [
            artifact_dir / "reports" / "persistence_baseline_summary.json",
            artifact_dir / "reports" / "analysis" / "persistence_baseline_summary.json",
            artifact_dir / "persistence_baseline_summary.json",
        ]
    )


def _read_threshold_summary(artifact_dir: Path) -> dict[str, Any]:
    return _read_first_existing_json(
        [
            artifact_dir / "reports" / "threshold_accuracy_summary.json",
            artifact_dir / "reports" / "analysis" / "threshold_accuracy_summary.json",
            artifact_dir / "threshold_accuracy_summary.json",
        ]
    )


def _read_event_aware_summary(artifact_dir: Path) -> dict[str, Any]:
    return _read_first_existing_json(
        [
            artifact_dir / "reports" / "event_aware_analysis_summary.json",
            artifact_dir / "reports" / "analysis" / "event_aware_analysis_summary.json",
            artifact_dir / "event_aware_analysis_summary.json",
        ]
    )


# ---------------------------------------------------------------------
# Epoch summary features
# ---------------------------------------------------------------------
def _build_epoch_summary(epoch_history: pd.DataFrame, best_epoch: int | None) -> dict[str, Any]:
    if epoch_history.empty:
        return {
            "epochs_logged": None,
            "first_epoch_logged": None,
            "last_epoch_logged": None,
            "best_val_loss_logged": None,
            "best_val_epoch_logged": None,
            "final_val_loss": None,
            "final_val_mae": None,
            "final_train_loss_epoch": None,
            "train_val_gap_final": None,
            "val_loss_worsened_after_best": None,
            "epochs_after_best": None,
            "train_loss_drop_first_to_last": None,
        }

    frame = epoch_history.copy()

    best_val_loss_logged = None
    best_val_epoch_logged = None
    if "val_loss" in frame.columns and frame["val_loss"].notna().any():
        idx = frame["val_loss"].idxmin()
        best_val_loss_logged = _safe_float(frame.loc[idx, "val_loss"])
        best_val_epoch_logged = int(frame.loc[idx, "epoch"])

    final_val_loss = None
    final_val_mae = None
    final_train_loss_epoch = None
    if not frame.empty:
        final_row = frame.iloc[-1]
        final_val_loss = _safe_float(final_row.get("val_loss"))
        final_val_mae = _safe_float(final_row.get("val_mae"))
        final_train_loss_epoch = _safe_float(final_row.get("train_loss_epoch"))

    train_val_gap_final = None
    if final_train_loss_epoch is not None and final_val_loss is not None:
        train_val_gap_final = final_val_loss - final_train_loss_epoch

    val_loss_worsened_after_best = None
    epochs_after_best = None
    if best_val_epoch_logged is not None and final_val_loss is not None and best_val_loss_logged is not None:
        val_loss_worsened_after_best = final_val_loss > best_val_loss_logged
        epochs_after_best = int(frame["epoch"].max()) - best_val_epoch_logged

    train_loss_drop_first_to_last = None
    if "train_loss_epoch" in frame.columns and frame["train_loss_epoch"].notna().sum() >= 2:
        first_train = _safe_float(frame["train_loss_epoch"].dropna().iloc[0])
        last_train = _safe_float(frame["train_loss_epoch"].dropna().iloc[-1])
        if first_train is not None and last_train is not None:
            train_loss_drop_first_to_last = first_train - last_train

    return {
        "epochs_logged": len(frame),
        "first_epoch_logged": int(frame["epoch"].min()) if "epoch" in frame.columns else None,
        "last_epoch_logged": int(frame["epoch"].max()) if "epoch" in frame.columns else None,
        "best_val_loss_logged": best_val_loss_logged,
        "best_val_epoch_logged": best_val_epoch_logged,
        "final_val_loss": final_val_loss,
        "final_val_mae": final_val_mae,
        "final_train_loss_epoch": final_train_loss_epoch,
        "train_val_gap_final": train_val_gap_final,
        "val_loss_worsened_after_best": val_loss_worsened_after_best,
        "epochs_after_best": epochs_after_best,
        "train_loss_drop_first_to_last": train_loss_drop_first_to_last,
        "best_epoch_matches_logged_best": (
            best_epoch is not None and best_val_epoch_logged is not None and best_epoch == best_val_epoch_logged
        ),
    }


# ---------------------------------------------------------------------
# Core summary builder
# ---------------------------------------------------------------------
def build_run_health_row(artifact_dir: Path) -> dict[str, Any]:
    run_summary_path = artifact_dir / "run_summary.json"
    run_summary = _safe_read_json(run_summary_path)
    if not run_summary:
        raise FileNotFoundError(f"Could not find run_summary.json in {artifact_dir}")

    run_log_path = artifact_dir / "run.log"
    run_log_text = _safe_read_text(run_log_path)
    epoch_history = _parse_epoch_history_from_run_log(run_log_text)

    by_horizon = _read_optional_table_from_report_paths(run_summary, "by_horizon")
    by_subject = _read_optional_table_from_report_paths(run_summary, "by_subject")
    by_glucose_range = _read_optional_table_from_report_paths(run_summary, "by_glucose_range")

    test_metrics_list = _get_nested(run_summary, "evaluation", "test_metrics") or []
    test_metrics = test_metrics_list[0] if test_metrics_list else {}
    test_eval_summary = _get_nested(run_summary, "evaluation", "test_evaluation", "summary") or {}

    persistence_summary = _read_persistence_summary(artifact_dir)
    threshold_summary = _read_threshold_summary(artifact_dir)
    event_summary = _read_event_aware_summary(artifact_dir)

    train_config = run_summary.get("train_config", {})
    optimizer = run_summary.get("optimizer", {})
    data_config = run_summary.get("config", {}).get("data", {})
    tft_config = run_summary.get("config", {}).get("tft", {})
    tcn_config = run_summary.get("config", {}).get("tcn", {})
    snapshot_config = run_summary.get("snapshot_config", {})
    fit_info = run_summary.get("fit", {})

    best_checkpoint_path = fit_info.get("best_checkpoint_path")
    best_epoch = _extract_best_epoch(best_checkpoint_path)
    best_val_loss_from_ckpt = _extract_best_val_loss_from_checkpoint(best_checkpoint_path)
    max_epochs = _safe_float(train_config.get("max_epochs"))

    row: dict[str, Any] = {
        # Identity
        "timestamp": run_summary.get("timestamp"),
        "artifact_dir": str(artifact_dir),
        "output_dir": run_summary.get("output_dir"),
        "requested_device_profile": _get_nested(run_summary, "device_profile", "requested"),
        "resolved_device_profile": _get_nested(run_summary, "device_profile", "resolved"),

        # Hyperparameters / knobs
        "max_epochs": max_epochs,
        "batch_size": _safe_float(data_config.get("batch_size")),
        "learning_rate": _safe_float(optimizer.get("learning_rate")),
        "weight_decay": _safe_float(optimizer.get("weight_decay")),
        "optimizer_name": optimizer.get("optimizer_name"),
        "precision": train_config.get("precision"),
        "compile_model": train_config.get("compile_model"),
        "early_stopping_patience": _safe_float(train_config.get("early_stopping_patience")),
        "num_sanity_val_steps": _safe_float(train_config.get("num_sanity_val_steps")),
        "tft_hidden_size": _safe_float(tft_config.get("hidden_size")),
        "tft_dropout": _safe_float(tft_config.get("dropout")),
        "tcn_dropout": _safe_float(tcn_config.get("dropout")),
        "tcn_kernel_size": _safe_float(tcn_config.get("kernel_size")),
        "encoder_length": _safe_float(data_config.get("encoder_length")),
        "prediction_length": _safe_float(data_config.get("prediction_length")),

        # Fit / checkpoint
        "has_validation_data": fit_info.get("has_validation_data"),
        "has_test_data": fit_info.get("has_test_data"),
        "best_checkpoint_path": best_checkpoint_path,
        "best_epoch": best_epoch,
        "best_val_loss_from_ckpt": best_val_loss_from_ckpt,
        "best_epoch_fraction_of_max": (best_epoch / max_epochs) if best_epoch is not None and max_epochs else None,
        "checkpoint_monitor": snapshot_config.get("monitor"),
        "checkpoint_mode": snapshot_config.get("mode"),
        "save_top_k": _safe_float(snapshot_config.get("save_top_k")),

        # Logged test metrics
        "test_loss": _safe_float(test_metrics.get("test_loss")),
        "test_mae_logged": _safe_float(test_metrics.get("test_mae")),
        "test_rmse_logged": _safe_float(test_metrics.get("test_rmse")),
        "test_target_mean": _safe_float(test_metrics.get("test_target_mean")),
        "test_target_std": _safe_float(test_metrics.get("test_target_std")),
        "test_median_prediction_mean": _safe_float(test_metrics.get("test_median_prediction_mean")),
        "test_prediction_interval_width_logged": _safe_float(test_metrics.get("test_prediction_interval_width")),

        # Canonical evaluation summary
        "eval_count": _safe_float(test_eval_summary.get("count")),
        "eval_mae": _safe_float(test_eval_summary.get("mae")),
        "eval_rmse": _safe_float(test_eval_summary.get("rmse")),
        "eval_bias": _safe_float(test_eval_summary.get("bias")),
        "eval_pinball_loss": _safe_float(test_eval_summary.get("overall_pinball_loss")),
        "eval_interval_width": _safe_float(test_eval_summary.get("mean_interval_width")),
        "eval_coverage": _safe_float(test_eval_summary.get("empirical_interval_coverage")),
        "pinball_q10": _safe_float(_get_nested(test_eval_summary, "pinball_loss_by_quantile", "q10")),
        "pinball_q50": _safe_float(_get_nested(test_eval_summary, "pinball_loss_by_quantile", "q50")),
        "pinball_q90": _safe_float(_get_nested(test_eval_summary, "pinball_loss_by_quantile", "q90")),

        # Subject heterogeneity
        "num_subject_rows": len(by_subject) if not by_subject.empty else None,
        "best_subject_mae": _safe_float(by_subject["mae"].min()) if "mae" in by_subject.columns and not by_subject.empty else None,
        "worst_subject_mae": _safe_float(by_subject["mae"].max()) if "mae" in by_subject.columns and not by_subject.empty else None,

        # Optional threshold summary
        "within_10_mgdl": _safe_float(threshold_summary.get("within_10_mgdl")),
        "within_20_mgdl": _safe_float(threshold_summary.get("within_20_mgdl")),
        "within_30_mgdl": _safe_float(threshold_summary.get("within_30_mgdl")),

        # Optional persistence summary
        "persistence_overall_rmse": _safe_float(persistence_summary.get("overall_persistence_rmse")),
        "model_minus_persistence_rmse": _safe_float(persistence_summary.get("model_minus_persistence_rmse")),

        # Optional event-aware summary
        "meal_mae": _safe_float(event_summary.get("meal_event_mae")),
        "non_meal_mae": _safe_float(event_summary.get("non_meal_mae")),
        "insulin_event_mae": _safe_float(event_summary.get("insulin_event_mae")),
        "non_insulin_event_mae": _safe_float(event_summary.get("non_insulin_event_mae")),
    }

    # Epoch-level summary
    row.update(_build_epoch_summary(epoch_history, best_epoch))

    # Horizon snapshots
    row.update(_extract_horizon_metrics(by_horizon, 0))
    row.update(_extract_horizon_metrics(by_horizon, 11))
    row.update(_compute_horizon_trend_features(by_horizon))

    # Glucose regimes
    row.update(_extract_glucose_range_metrics(by_glucose_range, "lt_70", "low"))
    row.update(_extract_glucose_range_metrics(by_glucose_range, "70_to_180", "normal"))
    row.update(_extract_glucose_range_metrics(by_glucose_range, "gt_180", "high"))

    # Derived health flags
    row["best_epoch_very_early"] = (
        best_epoch is not None and max_epochs is not None and best_epoch <= max(2, int(max_epochs * 0.15))
    )
    row["val_peaked_early_and_worsened"] = (
        row.get("best_epoch_very_early") is True and row.get("val_loss_worsened_after_best") is True
    )
    row["coverage_too_low"] = (
        row["eval_coverage"] is not None and row["eval_coverage"] < 0.75
    )
    row["intervals_very_wide"] = (
        row["eval_interval_width"] is not None and row["eval_interval_width"] > 60.0
    )
    row["low_glucose_struggles"] = (
        row["low_mae"] is not None and row["low_mae"] > 25.0
    )
    row["horizon_degrades_strongly"] = (
        row["horizon_rmse_delta_last_minus_first"] is not None
        and row["horizon_rmse_delta_last_minus_first"] > 10.0
    )
    row["large_train_val_gap_final"] = (
        row["train_val_gap_final"] is not None and row["train_val_gap_final"] > 1.0
    )

    return row, epoch_history


# ---------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------
def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_row_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        for col in row.keys():
            if col not in existing.columns:
                existing[col] = None
        for col in existing.columns:
            if col not in row:
                row[col] = None
        ordered_cols = list(existing.columns)
        updated = pd.concat([existing, pd.DataFrame([row])[ordered_cols]], ignore_index=True)
        updated.to_csv(path, index=False)
    else:
        write_csv(path, [row])


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one post-run health summary for a completed artifact directory.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/main_run"),
        help="Artifact directory containing run_summary.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit output path for the summary JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit output path for the one-row summary CSV",
    )
    parser.add_argument(
        "--epoch-history-csv",
        type=Path,
        default=None,
        help="Optional explicit output path for parsed epoch history CSV",
    )
    parser.add_argument(
        "--append-csv",
        type=Path,
        default=None,
        help="Optional tracker CSV to append this run into",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir: Path = args.artifact_dir

    row, epoch_history = build_run_health_row(artifact_dir)

    output_json = args.output_json or (artifact_dir / "run_health_summary.json")
    output_csv = args.output_csv or (artifact_dir / "run_health_summary.csv")
    epoch_history_csv = args.epoch_history_csv or (artifact_dir / "epoch_history_summary.csv")

    write_json(output_json, row)
    write_csv(output_csv, [row])

    if not epoch_history.empty:
        epoch_history.to_csv(epoch_history_csv, index=False)

    if args.append_csv is not None:
        append_row_csv(args.append_csv, row)
        print(f"Appended run health summary to tracker CSV: {args.append_csv}")

    print(f"Saved run health JSON to: {output_json}")
    print(f"Saved run health CSV to: {output_csv}")
    if not epoch_history.empty:
        print(f"Saved parsed epoch history CSV to: {epoch_history_csv}")
    else:
        print("No parseable epoch history found in run.log")

    print("\nRun health snapshot:")
    display_keys = [
        "resolved_device_profile",
        "max_epochs",
        "best_epoch",
        "best_val_loss_from_ckpt",
        "final_val_loss",
        "final_train_loss_epoch",
        "train_val_gap_final",
        "learning_rate",
        "weight_decay",
        "eval_mae",
        "eval_rmse",
        "eval_bias",
        "eval_coverage",
        "eval_interval_width",
        "h0_rmse",
        "h11_rmse",
        "low_mae",
        "normal_mae",
        "high_mae",
        "best_epoch_very_early",
        "val_peaked_early_and_worsened",
        "coverage_too_low",
        "intervals_very_wide",
        "low_glucose_struggles",
        "horizon_degrades_strongly",
        "large_train_val_gap_final",
    ]
    for key in display_keys:
        print(f"- {key}: {row.get(key)}")


if __name__ == "__main__":
    main()