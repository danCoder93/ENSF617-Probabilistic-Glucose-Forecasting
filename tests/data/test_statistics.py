"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates descriptive-statistics helpers for the refactored
AZT1D data layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.datamodule import AZT1DDataModule
from data.schema import build_feature_groups
from data.statistics import describe_clean_frame, describe_processed_data
from data.transforms import load_processed_frame
from tests.support import BuildDataConfig, WriteProcessedCsv


# ============================================================
# Data statistics tests
# ============================================================
# Purpose:
#   Verify that descriptive-statistics helpers report stable,
#   useful dataset summaries without changing the underlying
#   data-processing contract.
# ============================================================

def test_describe_clean_frame_reports_core_dataset_statistics(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    csv_path = write_processed_csv(
        subject_ids=("subject_a", "subject_b"),
        steps_per_subject=40,
    )
    config = build_data_config(csv_path)
    feature_groups = build_feature_groups(config)
    frame = load_processed_frame(csv_path, config, feature_groups)

    summary = describe_clean_frame(frame, config, feature_groups)

    assert summary["row_count"] == 80
    assert summary["subject_count"] == 2
    assert summary["duplicate_subject_timestamp_rows"] == 0
    assert summary["rows_per_subject"] == {"min": 40, "median": 40.0, "max": 40}
    assert summary["continuous_columns"]["glucose_mg_dl"]["q50"] == pytest.approx(119.5)
    assert summary["categorical_columns"]["device_mode"]["value_counts"] == {
        "sleep": 80,
    }
    assert summary["splits"] == {
        "train_rows": 56,
        "val_rows": 12,
        "test_rows": 12,
        "train_windows": 46,
        "val_windows": 2,
        "test_windows": 2,
        "train_subjects": 2,
        "val_subjects": 2,
        "test_subjects": 2,
    }


def test_describe_processed_data_tracks_duplicate_subject_timestamps(
    tmp_path: Path,
    build_data_config: BuildDataConfig,
) -> None:
    csv_path = tmp_path / "processed.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "subject_a",
                "timestamp": "2026-01-01 00:00:00",
                "glucose_mg_dl": 100.0,
                "basal_insulin_u": 0.0,
                "bolus_insulin_u": 0.0,
                "correction_insulin_u": 0.0,
                "meal_insulin_u": 0.0,
                "carbs_g": 0.0,
                "device_mode": "sleep",
                "bolus_type": "standard",
                "source_file": "subject_a.csv",
            },
            {
                "subject_id": "subject_a",
                "timestamp": "2026-01-01 00:00:00",
                "glucose_mg_dl": 105.0,
                "basal_insulin_u": 0.0,
                "bolus_insulin_u": 0.0,
                "correction_insulin_u": 0.0,
                "meal_insulin_u": 0.0,
                "carbs_g": 0.0,
                "device_mode": "sleep",
                "bolus_type": "standard",
                "source_file": "subject_a.csv",
            },
        ]
    ).to_csv(csv_path, index=False)

    config = build_data_config(csv_path)
    feature_groups = build_feature_groups(config)

    summary = describe_processed_data(str(csv_path), config, feature_groups)

    assert summary["duplicate_subject_timestamp_rows"] == 0
    assert summary["categorical_columns"]["subject_id"]["value_counts"] == {"subject_a": 1}


def test_datamodule_describe_data_requires_setup_and_returns_cleaned_summary(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    csv_path = write_processed_csv(steps_per_subject=40)
    config = build_data_config(csv_path)
    datamodule = AZT1DDataModule(config)

    with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
        datamodule.describe_data()

    datamodule.setup()
    summary = datamodule.describe_data()

    assert summary["row_count"] == 40
    assert summary["subject_count"] == 1
    assert summary["splits"]["train_windows"] == 23
    assert summary["splits"]["val_windows"] == 1
    assert summary["splits"]["test_windows"] == 1
