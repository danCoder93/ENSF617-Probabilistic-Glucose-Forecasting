"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates the refactored AZT1D transform layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.schema import DEVICE_MODE_CATEGORIES, build_feature_groups
from data.transforms import build_category_maps, load_processed_frame
from tests.support import BuildDataConfig, WriteProcessedCsv


# ============================================================
# Transform tests
# ============================================================
# Purpose:
#   Verify that dataframe-level cleanup produces a stable, model-
#   ready table contract before indexing or tensor assembly begins.
# ============================================================

def test_load_processed_frame_normalizes_sparse_values_and_categories(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    csv_path = write_processed_csv()
    config = build_data_config(csv_path)
    groups = build_feature_groups(config)

    # `load_processed_frame` is intentionally the one place where dataframe-wide
    # cleanup happens. The rest of the pipeline should be able to trust the
    # resulting frame as numerically and categorically normalized.
    frame = load_processed_frame(csv_path, config, groups)

    # Sparse operational values should be numeric and safe for tensor conversion.
    assert frame["basal_insulin_u"].dtype.kind in {"f", "i"}
    assert frame["carbs_g"].dtype.kind in {"f", "i"}
    assert frame.loc[1, "basal_insulin_u"] == 0.1

    # Normalization should collapse messy raw strings into the controlled vocab.
    assert frame.loc[0, "device_mode"] == "sleep"
    assert frame.loc[1, "bolus_type"] == "automatic"
    assert frame.loc[2, "bolus_type"] == "none"

    # Time-derived known features are part of the cleaned dataframe contract.
    assert "minute_of_day_sin" in frame.columns
    assert "day_of_week_cos" in frame.columns
    assert "is_weekend" in frame.columns


def test_build_category_maps_uses_declared_order_for_known_vocabularies(
    write_processed_csv: WriteProcessedCsv,
    build_data_config: BuildDataConfig,
) -> None:
    csv_path = write_processed_csv(subject_ids=("subject_b", "subject_a"))
    config = build_data_config(csv_path)
    groups = build_feature_groups(config)
    frame = load_processed_frame(csv_path, config, groups)

    # Category maps are fit after cleaning because the cleaned dataframe is the
    # first point where we know labels have already been normalized into their
    # stable vocabulary.
    category_maps = build_category_maps(frame, groups)

    # Declared vocabularies should preserve the schema-defined ordering exactly
    # so categorical IDs remain stable across runs and model initializations.
    assert category_maps["device_mode"] == DEVICE_MODE_CATEGORIES

    # Subject IDs are not hardcoded, so their vocabulary is discovered from data.
    assert category_maps["subject_id"] == ("subject_a", "subject_b")


def test_load_processed_frame_defaults_device_mode_to_regular_without_forward_fill_from_future(
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
                "basal_insulin_u": "",
                "bolus_insulin_u": 0.0,
                "correction_insulin_u": 0.0,
                "meal_insulin_u": 0.0,
                "carbs_g": "",
                "device_mode": "",
                "bolus_type": "",
                "source_file": "subject_a.csv",
            },
            {
                "subject_id": "subject_a",
                "timestamp": "2026-01-01 00:05:00",
                "glucose_mg_dl": 101.0,
                "basal_insulin_u": "0.8",
                "bolus_insulin_u": 0.0,
                "correction_insulin_u": 0.0,
                "meal_insulin_u": 0.0,
                "carbs_g": "",
                "device_mode": "exercise",
                "bolus_type": "",
                "source_file": "subject_a.csv",
            },
        ]
    ).to_csv(csv_path, index=False)

    config = build_data_config(csv_path)
    groups = build_feature_groups(config)
    frame = load_processed_frame(csv_path, config, groups)

    assert frame.loc[0, "device_mode"] == "regular"
    assert frame.loc[1, "device_mode"] == "exercise"
    assert frame.loc[0, "basal_insulin_u"] == 0.8


def test_load_processed_frame_collapses_duplicate_subject_timestamps_before_indexing(
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
                "basal_insulin_u": "1.2",
                "bolus_insulin_u": "",
                "correction_insulin_u": "",
                "meal_insulin_u": "",
                "carbs_g": "",
                "device_mode": "",
                "bolus_type": "",
                "source_file": "subject_a.csv",
            },
            {
                "subject_id": "subject_a",
                "timestamp": "2026-01-01 00:00:00",
                "glucose_mg_dl": 110.0,
                "basal_insulin_u": "1.2",
                "bolus_insulin_u": "",
                "correction_insulin_u": "",
                "meal_insulin_u": "",
                "carbs_g": "",
                "device_mode": "",
                "bolus_type": "",
                "source_file": "subject_a.csv",
            },
            {
                "subject_id": "subject_a",
                "timestamp": "2026-01-01 00:05:00",
                "glucose_mg_dl": 120.0,
                "basal_insulin_u": "1.4",
                "bolus_insulin_u": "2.0",
                "correction_insulin_u": "0.0",
                "meal_insulin_u": "2.0",
                "carbs_g": "15.0",
                "device_mode": "0",
                "bolus_type": "Standard",
                "source_file": "subject_a.csv",
            },
            {
                "subject_id": "subject_a",
                "timestamp": "2026-01-01 00:05:00",
                "glucose_mg_dl": 120.0,
                "basal_insulin_u": "1.6",
                "bolus_insulin_u": "2.0",
                "correction_insulin_u": "0.0",
                "meal_insulin_u": "2.0",
                "carbs_g": "15.0",
                "device_mode": "0",
                "bolus_type": "Standard",
                "source_file": "subject_a.csv",
            },
        ]
    ).to_csv(csv_path, index=False)

    config = build_data_config(csv_path)
    groups = build_feature_groups(config)
    frame = load_processed_frame(csv_path, config, groups)

    assert len(frame) == 2
    assert frame.loc[0, "glucose_mg_dl"] == 105.0
    assert frame.loc[1, "basal_insulin_u"] == 1.4
    assert frame.loc[1, "device_mode"] == "regular"
    assert frame.loc[1, "bolus_type"] == "standard"
