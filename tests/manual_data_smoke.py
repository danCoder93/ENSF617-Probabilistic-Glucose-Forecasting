"""
AI-assisted implementation note:
This smoke-test script was drafted with AI assistance and then reviewed/adapted
for this project. It exercises the refactored AZT1D data path end to end.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sized, cast


# This script is kept as a manual smoke test rather than an automated pytest
# case because it can perform real dataset download and end-to-end data-module
# preparation. That makes it useful for developer sanity checks, but too heavy
# and too network-dependent for the regular unit-test suite.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.datamodule import AZT1DDataModule
from data.dataset import BatchItem
from config import DataConfig


# ============================================================
# Manual smoke test
# ============================================================
# Purpose:
#   Provide a developer-facing end-to-end check that exercises the
#   real DataModule with real filesystem side effects and optional
#   network download.
#
# Why this is not a pytest test:
#   It is intentionally heavier and less deterministic than unit
#   tests. Keeping it separate preserves a fast, isolated pytest
#   suite while still giving developers an easy full-pipeline check.
# ============================================================

AZT1D_URL = (
    "https://data.mendeley.com/public-files/datasets/"
    "gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded"
)


def summarize_csv(file_path: Path) -> tuple[int, list[str]]:
    # This helper keeps the smoke script readable by separating the quick CSV
    # summary logic from the DataModule orchestration flow below.
    with file_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader, [])
        row_count = sum(1 for _ in reader)
    return row_count, headers


def describe_batch(batch: BatchItem) -> None:
    # Printing shapes is usually enough for a manual smoke check because it tells
    # us whether the batch contract materialized in the expected structure.
    print(f"Static categorical shape: {tuple(batch['static_categorical'].shape)}")
    print(f"Static continuous shape: {tuple(batch['static_continuous'].shape)}")
    print(f"Encoder continuous shape: {tuple(batch['encoder_continuous'].shape)}")
    print(f"Encoder categorical shape: {tuple(batch['encoder_categorical'].shape)}")
    print(
        "Decoder known continuous shape: "
        f"{tuple(batch['decoder_known_continuous'].shape)}"
    )
    print(
        "Decoder known categorical shape: "
        f"{tuple(batch['decoder_known_categorical'].shape)}"
    )
    print(f"Target shape: {tuple(batch['target'].shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a manual end-to-end smoke test for the AZT1D DataModule."
    )
    parser.add_argument(
        "--url",
        default=AZT1D_URL,
        help="Direct download URL for the AZT1D zip file.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Directory for temporary download artifacts.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory for the downloaded raw archive.",
    )
    parser.add_argument(
        "--extract-dir",
        default="data/extracted",
        help="Directory for the extracted dataset.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the zip again even if it is already cached.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/azt1d_processed.csv",
        help="Path for the processed CSV output.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    data_config = DataConfig(
        dataset_url=args.url,
        raw_dir=Path(args.raw_dir),
        cache_dir=Path(args.cache_dir),
        extracted_dir=Path(args.extract_dir),
        processed_dir=output_path.parent,
        processed_file_name=output_path.name,
        redownload=args.force_download,
        rebuild_processed=args.force_download,
        num_workers=0,
    )
    datamodule = AZT1DDataModule(data_config)
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    row_count, headers = summarize_csv(data_config.processed_file_path)

    print("DataModule prepare/setup complete")
    print(f"Processed file: {data_config.processed_file_path}")
    print(f"Row count: {row_count}")
    print(f"Columns: {', '.join(headers)}")
    print(f"Train windows: {len(cast(Sized, train_loader.dataset))}")
    print(f"Validation windows: {len(cast(Sized, val_loader.dataset))}")
    print(f"Test windows: {len(cast(Sized, test_loader.dataset))}")
    print(f"Feature groups: {datamodule.feature_groups}")
    print(f"Categorical cardinalities: {datamodule.categorical_cardinalities}")

    train_iterator = iter(train_loader)
    try:
        first_batch = next(train_iterator)
    except StopIteration:
        first_batch = None

    if first_batch is not None:
        describe_batch(first_batch)
