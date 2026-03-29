from pathlib import Path
import csv


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

REQUIRED_COLUMNS = [
    "EventDateTime",
    "Basal",
    "TotalBolusInsulinDelivered",
    "CorrectionDelivered",
    "FoodDelivered",
    "CarbSize",
    "DeviceMode",
    "BolusType",
]

GLUCOSE_COLUMNS = ["CGM", "Readings (CGM / BGM)"]


class DatasetCombiner:
    def __init__(
        self,
        dataset_dir: str | Path = "data/extracted/azt1d",
        output_file: str | Path = "data/processed/azt1d_by_subject.csv",
    ) -> None:
        self.dataset_dir = self._resolve_data_dir(Path(dataset_dir))
        self.output_file = Path(output_file)

    def combine(self) -> Path:
        csv_files = self._find_csv_files(self.dataset_dir)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with self.output_file.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            for csv_file in csv_files:
                relative_path = csv_file.relative_to(self.dataset_dir)
                subject_id = relative_path.parts[0] if len(relative_path.parts) > 1 else csv_file.stem

                with csv_file.open("r", newline="", encoding="utf-8-sig") as source:
                    reader = csv.DictReader(source)
                    fieldnames = [field.strip() for field in (reader.fieldnames or []) if field]

                    if not fieldnames:
                        raise ValueError(f"Missing header row in {csv_file}")

                    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
                    glucose_column = next(
                        (column for column in GLUCOSE_COLUMNS if column in fieldnames),
                        None,
                    )

                    if glucose_column is None:
                        missing_columns.append("CGM")

                    if missing_columns:
                        raise ValueError(
                            f"Missing required columns in {csv_file}: {', '.join(missing_columns)}"
                        )

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
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

        csv_files = sorted(path for path in dataset_dir.rglob("*.csv") if path.is_file())
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

        return csv_files

    def _text(self, row: dict[str, str | None], column: str) -> str:
        value = row.get(column, "")
        if value is None:
            return ""
        return value.strip()
