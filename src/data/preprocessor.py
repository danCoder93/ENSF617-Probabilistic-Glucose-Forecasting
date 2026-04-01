# AI-assisted implementation note:
# This file was drafted with AI assistance and then reviewed/adapted for this
# project. The refactor draws on the earlier AZT1D pipeline in this repo, prior
# work by SlickMik (https://github.com/SlickMik), the PyTorch Lightning
# DataModule docs/tutorial
# (https://lightning.ai/docs/pytorch/stable/data/datamodule.html), and the
# original AZT1D dataset release on Mendeley Data
# (https://data.mendeley.com/datasets/gk9m674wcx/1). Its purpose is to isolate
# raw-data standardization as a dedicated preprocessing stage.

from __future__ import annotations

import csv
from pathlib import Path

from data.schema import RAW_GLUCOSE_COLUMNS, RAW_TO_INTERNAL_COLUMN_MAP


# ============================================================
# Processed CSV contract
# ============================================================
# Purpose:
#   Define the canonical file shape emitted by the raw-data
#   standardization stage.
# ============================================================

# The processed file is intentionally narrow and canonical. It preserves the raw
# measurements but strips away vendor-specific naming so every downstream stage
# can load one stable table shape.
OUTPUT_COLUMNS = [
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


class AZT1DPreprocessor:
    """
    Standardize raw AZT1D CSV exports into one canonical processed CSV.

    Purpose:
    isolate raw-file standardization from the rest of the learning pipeline.

    Context:
    - raw files are a storage concern, not a dataset-window concern
    - this class knows how to read vendor-shaped CSVs and emit one clean table
    - it intentionally does not know anything about train/val/test splits or
      tensor windows; that logic lives later in indexing, dataset, and
      DataModule code
    """

    def __init__(
        self,
        dataset_dir: str | Path = "data/extracted/azt1d",
        output_file: str | Path = "data/processed/azt1d_processed.csv",
    ) -> None:
        self.dataset_dir = self._resolve_data_dir(Path(dataset_dir))
        self.output_file = Path(output_file)

    def build(self, force: bool = False) -> Path:
        # Reusing an existing processed file keeps repeated setup runs fast and
        # makes preprocessing an explicit, controllable step instead of hidden
        # work that happens during training.
        if self.output_file.exists() and not force:
            return self.output_file

        csv_files = self._find_csv_files(self.dataset_dir)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with self.output_file.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            for csv_file in csv_files:
                relative_path = csv_file.relative_to(self.dataset_dir)

                # The subject ID is derived from the folder structure because the
                # public AZT1D export stores each subject in a separate directory.
                # Capturing it here lets the rest of the pipeline treat subject ID
                # as just another canonical column.
                subject_id = relative_path.parts[0] if len(relative_path.parts) > 1 else csv_file.stem

                with csv_file.open("r", newline="", encoding="utf-8-sig") as source:
                    reader = csv.DictReader(source)
                    fieldnames = [field.strip() for field in (reader.fieldnames or []) if field]

                    if not fieldnames:
                        raise ValueError(f"Missing header row in {csv_file}")

                    glucose_column = next(
                        (column for column in RAW_GLUCOSE_COLUMNS if column in fieldnames),
                        None,
                    )
                    # Using a set for required-column membership keeps the
                    # validation concise. We sort only when formatting the error
                    # message so the user still gets a stable, readable report.
                    required_columns = set(RAW_TO_INTERNAL_COLUMN_MAP)
                    missing_columns = [column for column in required_columns if column not in fieldnames]

                    if glucose_column is None:
                        missing_columns.append(RAW_GLUCOSE_COLUMNS[0])

                    if missing_columns:
                        raise ValueError(
                            f"Missing required columns in {csv_file}: {', '.join(sorted(missing_columns))}"
                        )

                    # The validation above guarantees we found a glucose column,
                    # but the explicit assertion helps static analyzers narrow the
                    # type from `str | None` to `str`.
                    assert glucose_column is not None

                    # Keep the mapping explicit so the canonical schema is easy
                    # to audit and does not depend on implicit raw-column loops.
                    reader.fieldnames = fieldnames

                    for row in reader:
                        writer.writerow(
                            {
                                "subject_id": subject_id,
                                "timestamp": self._text(row, "EventDateTime"),
                                "glucose_mg_dl": self._text(row, glucose_column),
                                "basal_insulin_u": self._text(row, "Basal"),
                                "bolus_insulin_u": self._text(row, "TotalBolusInsulinDelivered"),
                                "correction_insulin_u": self._text(row, "CorrectionDelivered"),
                                "meal_insulin_u": self._text(row, "FoodDelivered"),
                                "carbs_g": self._text(row, "CarbSize"),
                                "device_mode": self._text(row, "DeviceMode"),
                                "bolus_type": self._text(row, "BolusType"),
                                "source_file": relative_path.as_posix(),
                            }
                        )

        return self.output_file

    def _resolve_data_dir(self, dataset_dir: Path) -> Path:
        # The public archive layout is slightly inconsistent across extraction
        # paths, so we probe a few likely roots once here rather than spreading
        # that discovery logic throughout the pipeline.
        candidates = [
            dataset_dir / "AZT1D 2025" / "CGM Records",
            dataset_dir / "CGM Records",
            dataset_dir,
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

        return dataset_dir

    def _find_csv_files(self, dataset_dir: Path) -> list[Path]:
        # Fail fast if the extracted dataset layout is wrong instead of letting
        # later stages continue with an empty or partial processed file.
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

        csv_files = sorted(path for path in dataset_dir.rglob("*.csv") if path.is_file())
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

        return csv_files

    def _text(self, row: dict[str, str | None], column: str) -> str:
        # Trimming raw whitespace here keeps later normalization logic simpler
        # and ensures the processed CSV is already reasonably clean.
        value = row.get(column, "")
        if value is None:
            return ""
        return value.strip()
