import argparse
import csv
from pathlib import Path

from azt1d_dataset import WindowConfig, create_dataloaders
from dataset_combiner import DatasetCombiner
from dataset_downloader import DatasetDownloader


AZT1D_URL = (
    "https://data.mendeley.com/public-files/datasets/"
    "gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded"
)


def summarize_csv(file_path: Path) -> tuple[int, list[str]]:
    with file_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader, [])
        row_count = sum(1 for _ in reader)
    return row_count, headers


def describe_batch(batch: dict[str, object]) -> None:
    print(f"Static covariates shape: {tuple(batch['static_covariates'].shape)}")
    print(f"Past observed inputs shape: {tuple(batch['past_observed_inputs'].shape)}")
    print(f"Future known inputs shape: {tuple(batch['future_known_inputs'].shape)}")
    print(f"Target shape: {tuple(batch['target'].shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AZT1D and combine it into one CSV.")
    parser.add_argument(
        "--url",
        default=AZT1D_URL,
        help="Direct download URL for the AZT1D zip file.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw",
        help="Directory for the downloaded zip file.",
    )
    parser.add_argument(
        "--extract-dir",
        default="data/extracted",
        help="Directory for the extracted dataset.",
    )
    parser.add_argument(
        "--filename",
        default="azt1d.zip",
        help="Filename to use for the downloaded archive.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the zip again even if it is already cached.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/azt1d_by_subject.csv",
        help="Path for the combined CSV output.",
    )
    args = parser.parse_args()

    downloader = DatasetDownloader(
        cache_dir=args.cache_dir,
        extract_dir=args.extract_dir,
    )
    result = downloader.download(
        url=args.url,
        filename=args.filename,
        extract=True,
        force=args.force_download,
    )

    if result.extracted_path is None:
        raise ValueError("The downloaded file was not extracted.")

    combiner = DatasetCombiner(dataset_dir=result.extracted_path, output_file=args.output)
    output_file = combiner.combine()
    train_loader, val_loader, test_loader = create_dataloaders(
        output_file,
        config=WindowConfig(),
    )

    row_count, headers = summarize_csv(output_file)

    print("Download complete")
    print(f"Zip file: {result.file_path}")
    print(f"Extracted path: {result.extracted_path}")
    print(f"From cache: {result.from_cache}")
    print(f"Size (bytes): {result.size_bytes}")
    print("Combine complete")
    print(f"Output file: {output_file}")
    print(f"Row count: {row_count}")
    print(f"Columns: {', '.join(headers)}")
    print(f"Train windows: {len(train_loader.dataset)}")
    print(f"Validation windows: {len(val_loader.dataset)}")
    print(f"Test windows: {len(test_loader.dataset)}")

    first_batch = next(iter(train_loader), None)
    if first_batch is not None:
        describe_batch(first_batch)
