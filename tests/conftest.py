"""
AI-assisted implementation note:
This test-support file was drafted with AI assistance and then reviewed/adapted
for this project. It supports validation of the refactored AZT1D data
pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# ============================================================
# Test import bootstrap
# ============================================================
# Purpose:
#   Make the project modules under `src/` importable from pytest.
#
# Why this lives in conftest.py:
#   Every test module needs the same import-path setup, so keeping it here avoids
#   repeating the same boilerplate in each file and keeps the tests focused on
#   behavior rather than environment wiring.
# ============================================================

# Pytest runs from the repository root, but the project code currently lives
# under `src/` and uses imports like `from data...` and `from utils...`.
# Adding `src` to sys.path here makes the tests match the way the project is run
# in local scripts and in notebook-style environments such as Colab.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import DataConfig
from tests.support import BuildDataConfig, WriteProcessedCsv


@pytest.fixture
def write_processed_csv(tmp_path: Path) -> WriteProcessedCsv:
    """
    Create a canonical processed CSV for data-layer tests.

    Why a factory fixture is useful:
    - Different tests need slightly different subject counts, sequence lengths,
      and gap patterns.
    - Building those tables inline every time would make the tests repetitive and
      harder to read than the behavior they are trying to verify.
    """

    def _write(
        *,
        filename: str = "processed.csv",
        subject_ids: tuple[str, ...] = ("subject_a",),
        steps_per_subject: int = 12,
        gap_after_step: int | None = None,
    ) -> Path:
        # The fixture emits the *processed* CSV contract directly rather than raw
        # AZT1D vendor files because most unit tests are targeting the cleaned
        # data path. Raw-file behavior is covered separately in the preprocessor
        # tests.
        rows: list[dict[str, Any]] = []

        for subject_offset, subject_id in enumerate(subject_ids):
            base_time = pd.Timestamp("2026-01-01 00:00:00") + pd.Timedelta(days=subject_offset)

            for step in range(steps_per_subject):
                minute_offset = step * 5
                if gap_after_step is not None and step > gap_after_step:
                    # Insert one extra sampling interval after the chosen step so
                    # the indexing tests can verify that continuity checks break
                    # windows across real time gaps.
                    minute_offset += 5

                rows.append(
                    {
                        "subject_id": subject_id,
                        "timestamp": base_time + pd.Timedelta(minutes=minute_offset),
                        "glucose_mg_dl": 100.0 + step,
                        "basal_insulin_u": 0.1 if step % 2 == 0 else "",
                        "bolus_insulin_u": 0.0,
                        "correction_insulin_u": 0.0,
                        "meal_insulin_u": 0.0,
                        "carbs_g": 10.0 if step % 3 == 0 else "",
                        "device_mode": "SleepSleep" if step == 0 else "",
                        "bolus_type": "Automatic Bolus" if step == 1 else "",
                        "source_file": f"{subject_id}.csv",
                    }
                )

        csv_path = tmp_path / filename
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    return _write


@pytest.fixture
def build_data_config(tmp_path: Path) -> BuildDataConfig:
    """
    Build a DataConfig that points entirely into pytest-managed temp folders.

    This keeps tests isolated from the real project `data/` directory and makes
    each test responsible only for the files it explicitly creates.
    """

    def _build(processed_csv_path: Path, **overrides: Any) -> DataConfig:
        # Keep the defaults intentionally small so tests generate short sequences
        # and tiny dataloaders. That makes failures easier to inspect while still
        # exercising the real production code paths.
        defaults: dict[str, Any] = {
            "dataset_url": None,
            "raw_dir": tmp_path / "raw",
            "cache_dir": tmp_path / "cache",
            "extracted_dir": tmp_path / "extracted",
            "processed_dir": processed_csv_path.parent,
            "processed_file_name": processed_csv_path.name,
            "encoder_length": 4,
            "prediction_length": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
        defaults.update(overrides)
        return DataConfig(**defaults)

    return _build
