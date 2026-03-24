from dataset_downloader import DatasetDownloader

if __name__ == "__main__":
    url = "https://data.mendeley.com/public-files/datasets/gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded"

    downloader = DatasetDownloader(
        cache_dir="data/raw",
        extract_dir="data/extracted",
    )

    result = downloader.download(
        url=url,
        filename="mendeley_dataset.zip",  # set this if you know it's a zip
        extract=True,                     # auto-extract if it is a zip
        force=False,
    )

    print("Download complete")
    print(f"File path     : {result.file_path}")
    print(f"Extracted path: {result.extracted_path}")
    print(f"From cache    : {result.from_cache}")
    print(f"Content-Type  : {result.content_type}")
    print(f"Size (bytes)  : {result.size_bytes}")