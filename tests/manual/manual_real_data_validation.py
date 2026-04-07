"""
AI-assisted implementation note:
This manual validation script was drafted with AI assistance and then
reviewed/adapted for this project. It validates the real AZT1D data path
against local dataset artifacts and writes a compact JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DataConfig
from data.datamodule import AZT1DDataModule
from data.indexing import SampleIndexEntry, build_sequence_index, split_processed_frame
from data.preprocessor import AZT1DPreprocessor
from data.transforms import load_processed_frame


def _default_processed_path() -> Path:
    return ROOT_DIR / "data" / "processed" / "azt1d_processed.csv"


def _default_extracted_dir() -> Path:
    return ROOT_DIR / "data" / "extracted"


def _default_summary_path() -> Path:
    return ROOT_DIR / "artifacts" / "data_validation" / "real_data_validation_summary.json"


def _discover_extracted_dataset_root(base_dir: Path) -> Path | None:
    candidates = [
        base_dir,
        base_dir / "mendeley_dataset",
    ]
    if base_dir.exists():
        candidates.extend(path for path in sorted(base_dir.iterdir()) if path.is_dir())

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        preprocessor = AZT1DPreprocessor(dataset_dir=candidate, output_file=_default_processed_path())
        try:
            csv_files = preprocessor._find_csv_files(preprocessor.dataset_dir)
        except FileNotFoundError:
            continue
        if csv_files:
            return candidate

    return None


def _ensure_processed_file(
    *,
    processed_path: Path,
    extracted_dir: Path | None,
    force_rebuild: bool,
) -> Path:
    if processed_path.exists() and not force_rebuild:
        return processed_path

    dataset_root = None
    if extracted_dir is not None:
        dataset_root = extracted_dir
    else:
        dataset_root = _discover_extracted_dataset_root(_default_extracted_dir())

    if dataset_root is None:
        raise FileNotFoundError(
            "Could not find an extracted AZT1D dataset root. "
            "Run the manual smoke test first or pass --dataset-dir."
        )

    preprocessor = AZT1DPreprocessor(dataset_dir=dataset_root, output_file=processed_path)
    return preprocessor.build(force=force_rebuild)


def _assert_clean_frame_invariants(frame: pd.DataFrame, config: DataConfig) -> None:
    assert not frame.empty, "Cleaned dataframe is empty."
    assert frame[config.subject_id_column].notna().all(), "Cleaned dataframe has null subject IDs."
    assert frame[config.time_column].notna().all(), "Cleaned dataframe has null timestamps."
    assert frame[config.target_column].notna().all(), "Cleaned dataframe has null target values."
    assert frame[config.subject_id_column].astype(str).str.strip().ne("").all(), (
        "Cleaned dataframe has blank subject IDs."
    )
    duplicate_timestamp_rows = int(
        frame.duplicated([config.subject_id_column, config.time_column]).sum()
    )
    assert duplicate_timestamp_rows == 0, (
        "Cleaned dataframe contains duplicate subject/timestamp rows "
        f"({duplicate_timestamp_rows} duplicates), which breaks the current "
        "sequence-index assumptions on the real dataset."
    )

    for subject_id, subject_frame in frame.groupby(config.subject_id_column, sort=False):
        deltas = (
            subject_frame[config.time_column]
            .sort_values()
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(60)
        )
        assert (deltas >= 0).all(), f"Subject {subject_id} has descending timestamps."


def _validate_sample_index(
    *,
    frame: pd.DataFrame,
    sample_index: list[SampleIndexEntry],
    config: DataConfig,
) -> None:
    required_length = config.encoder_length + config.prediction_length

    for sample in sample_index:
        window = frame.iloc[sample.encoder_start:sample.decoder_end].reset_index(drop=True)
        assert len(window) == required_length, "Indexed sample length does not match encoder+decoder length."

        subject_ids = window[config.subject_id_column].astype(str).unique().tolist()
        assert subject_ids == [sample.subject_id], f"Sample crossed subject boundaries: {subject_ids}"

        deltas = (
            window[config.time_column]
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(60)
        )
        assert (
            deltas == config.sampling_interval_minutes
        ).all(), f"Sample for subject {sample.subject_id} crossed a time gap."


def _tensor_shape(shape: Any) -> list[int]:
    return [int(dimension) for dimension in tuple(shape)]


def _describe_numeric(series: pd.Series) -> dict[str, float]:
    quantiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "q05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]),
        "q50": float(quantiles.loc[0.5]),
        "q75": float(quantiles.loc[0.75]),
        "q95": float(quantiles.loc[0.95]),
    }


def _basic_summary(
    *,
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    config: DataConfig,
) -> dict[str, Any]:
    rows_per_subject = clean_frame.groupby(config.subject_id_column).size()
    duplicate_timestamp_rows = int(
        clean_frame.duplicated([config.subject_id_column, config.time_column]).sum()
    )
    target_series = clean_frame[config.target_column]

    return {
        "processed_file": str(config.processed_file_path),
        "raw_processed_csv": {
            "row_count": int(len(raw_frame)),
            "column_count": int(len(raw_frame.columns)),
            "columns": [str(column) for column in raw_frame.columns],
            "missing_values_by_column": {
                str(column): int(raw_frame[column].isna().sum()) for column in raw_frame.columns
            },
        },
        "clean_frame": {
            "row_count": int(len(clean_frame)),
            "subject_count": int(clean_frame[config.subject_id_column].nunique()),
            "timestamp_start": clean_frame[config.time_column].min().isoformat(),
            "timestamp_end": clean_frame[config.time_column].max().isoformat(),
            "duplicate_subject_timestamp_rows": duplicate_timestamp_rows,
            "rows_per_subject": {
                "min": int(rows_per_subject.min()),
                "median": float(rows_per_subject.median()),
                "max": int(rows_per_subject.max()),
            },
            "glucose_mg_dl": _describe_numeric(target_series),
            "device_mode_counts": {
                str(key): int(value)
                for key, value in clean_frame["device_mode"].value_counts(dropna=False).sort_index().items()
            }
            if "device_mode" in clean_frame.columns
            else {},
            "bolus_type_counts": {
                str(key): int(value)
                for key, value in clean_frame["bolus_type"].value_counts(dropna=False).sort_index().items()
            }
            if "bolus_type" in clean_frame.columns
            else {},
        },
    }


def _build_summary(
    *,
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    datamodule: AZT1DDataModule,
    config: DataConfig,
) -> dict[str, Any]:
    summary = _basic_summary(raw_frame=raw_frame, clean_frame=clean_frame, config=config)
    split_frames = split_processed_frame(clean_frame, config, datamodule.feature_groups)
    train_index = build_sequence_index(split_frames["train"], config, datamodule.feature_groups)
    val_index = build_sequence_index(split_frames["val"], config, datamodule.feature_groups)
    test_index = build_sequence_index(split_frames["test"], config, datamodule.feature_groups)

    _assert_clean_frame_invariants(clean_frame, config)
    _validate_sample_index(frame=split_frames["train"], sample_index=train_index, config=config)
    _validate_sample_index(frame=split_frames["val"], sample_index=val_index, config=config)
    _validate_sample_index(frame=split_frames["test"], sample_index=test_index, config=config)

    assert len(split_frames["train"]) + len(split_frames["val"]) + len(split_frames["test"]) == len(clean_frame), (
        "Split frames do not cover the cleaned dataframe exactly."
    )
    assert len(train_index) > 0, "Train split did not produce any valid windows on the real dataset."
    assert len(val_index) > 0, "Validation split did not produce any valid windows on the real dataset."
    assert len(test_index) > 0, "Test split did not produce any valid windows on the real dataset."

    train_batch = next(iter(datamodule.train_dataloader()))
    first_train_sample = datamodule.train_dataset[0] if datamodule.train_dataset is not None else None
    assert first_train_sample is not None, "Training dataset is unexpectedly empty."

    summary["splits"] = {
            "train_rows": int(len(split_frames["train"])),
            "val_rows": int(len(split_frames["val"])),
            "test_rows": int(len(split_frames["test"])),
            "train_windows": int(len(train_index)),
            "val_windows": int(len(val_index)),
            "test_windows": int(len(test_index)),
    }
    summary["categorical_cardinalities"] = {
        str(column): int(size) for column, size in datamodule.categorical_cardinalities.items()
    }
    summary["batch_contract"] = {
        "train_batch_target_shape": _tensor_shape(train_batch["target"].shape),
        "train_batch_encoder_continuous_shape": _tensor_shape(
            train_batch["encoder_continuous"].shape
        ),
        "first_train_sample_metadata": {
            key: str(value) for key, value in first_train_sample["metadata"].items()
        },
    }
    summary["checks"] = {
        "validation_passed": True,
        "clean_frame_non_empty": True,
        "split_rows_cover_clean_frame": True,
        "train_windows_non_zero": True,
        "val_windows_non_zero": True,
        "test_windows_non_zero": True,
        "sample_windows_preserve_subject_boundaries": True,
        "sample_windows_preserve_expected_sampling_interval": True,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the real AZT1D data path against local dataset artifacts."
    )
    parser.add_argument(
        "--processed-file",
        default=str(_default_processed_path()),
        help="Processed CSV to validate. Built from the extracted dataset if missing.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Optional extracted dataset root to preprocess when the processed CSV is missing.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the processed CSV from the extracted dataset before validation.",
    )
    parser.add_argument(
        "--summary-file",
        default=str(_default_summary_path()),
        help="JSON file path for the persisted validation summary.",
    )
    args = parser.parse_args()

    processed_path = Path(args.processed_file)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    summary_path = Path(args.summary_file)

    processed_path = _ensure_processed_file(
        processed_path=processed_path,
        extracted_dir=dataset_dir,
        force_rebuild=args.force_rebuild,
    )

    data_config = DataConfig(
        dataset_url=None,
        processed_dir=processed_path.parent,
        processed_file_name=processed_path.name,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    datamodule = AZT1DDataModule(data_config)
    datamodule.setup()

    raw_frame = pd.read_csv(processed_path)
    clean_frame = load_processed_frame(
        processed_path,
        datamodule.config,
        datamodule.feature_groups,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary = _build_summary(
            raw_frame=raw_frame,
            clean_frame=clean_frame,
            datamodule=datamodule,
            config=data_config,
        )
        exit_code = 0
        status_line = "Real-data validation passed"
    except AssertionError as exc:
        summary = _basic_summary(
            raw_frame=raw_frame,
            clean_frame=clean_frame,
            config=data_config,
        )
        summary["checks"] = {"validation_passed": False}
        summary["failure"] = str(exc)
        exit_code = 1
        status_line = "Real-data validation failed"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(status_line)
    print(f"Processed file: {processed_path}")
    print(f"Summary file: {summary_path}")
    if "splits" in summary:
        print(
            "Rows / subjects / windows: "
            f"{summary['clean_frame']['row_count']} / "
            f"{summary['clean_frame']['subject_count']} / "
            f"{summary['splits']['train_windows'] + summary['splits']['val_windows'] + summary['splits']['test_windows']}"
        )
    else:
        print(
            "Rows / subjects / duplicate subject-timestamp rows: "
            f"{summary['clean_frame']['row_count']} / "
            f"{summary['clean_frame']['subject_count']} / "
            f"{summary['clean_frame']['duplicate_subject_timestamp_rows']}"
        )
        print(f"Failure: {summary['failure']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
