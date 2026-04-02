"""
AI-assisted implementation note:
This test file was drafted with AI assistance and then reviewed/adapted for
this project. It validates the refactored AZT1D preprocessing stage.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from data.preprocessor import AZT1DPreprocessor


# ============================================================
# Preprocessor tests
# ============================================================
# Purpose:
#   Verify that raw vendor-shaped CSV files are standardized into
#   the single processed CSV contract used by the rest of the data
#   layer.
# ============================================================

def test_preprocessor_builds_one_canonical_processed_csv(tmp_path: Path) -> None:
    raw_root = tmp_path / "extracted" / "AZT1D 2025" / "CGM Records" / "subject_001"
    raw_root.mkdir(parents=True)
    raw_csv_path = raw_root / "records.csv"

    with raw_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "EventDateTime",
                "CGM",
                "Basal",
                "TotalBolusInsulinDelivered",
                "CorrectionDelivered",
                "FoodDelivered",
                "CarbSize",
                "DeviceMode",
                "BolusType",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "EventDateTime": "2026-01-01 00:00:00",
                "CGM": "105",
                "Basal": "0.1",
                "TotalBolusInsulinDelivered": "0",
                "CorrectionDelivered": "0",
                "FoodDelivered": "0",
                "CarbSize": "12",
                "DeviceMode": "sleep",
                "BolusType": "standard",
            }
        )

    output_path = tmp_path / "processed" / "azt1d_processed.csv"
    preprocessor = AZT1DPreprocessor(dataset_dir=tmp_path / "extracted", output_file=output_path)

    built_path = preprocessor.build()
    frame = pd.read_csv(built_path)

    # The exact output columns matter because every downstream layer assumes
    # this canonical schema exists after raw-data standardization.
    assert built_path == output_path
    assert list(frame.columns) == [
        "subject_id",
        "timestamp",
        "glucose_mg_dl",
        "basal_insulin_u",
        "bolus_insulin_u",
        "correction_insulin_u",
        "meal_insulin_u",
        "carbs_g",
        "device_mode",
        "bolus_type",
        "source_file",
    ]
    # Subject IDs come from directory structure in the public AZT1D archive, so
    # this assertion protects that convention explicitly.
    assert frame.loc[0, "subject_id"] == "subject_001"
    assert frame.loc[0, "glucose_mg_dl"] == 105
