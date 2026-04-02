from __future__ import annotations

# AI-assisted maintenance note:
# This module owns the data-pipeline configuration contract. It is kept
# separate from model, runtime, and observability settings so the dataset layer
# can evolve independently without making the broader configuration surface
# harder to navigate.

from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from typing import Optional, Sequence

from config.types import PathInput
from utils.tft_utils import FeatureSpec


@dataclass
class DataConfig:
    """
    Configuration for dataset access, preprocessing, splits, and dataloaders.

    Purpose:
    hold dataset access, preprocessing, split, and DataLoader settings shared
    across the refactored data pipeline.

    Context:
    the data refactor separated downloading, preprocessing, schema derivation,
    indexing, dataset assembly, and loader creation into distinct layers, but
    those layers still need one common configuration contract.

    Operational role:
    this dataclass gives the rest of the codebase one place to agree on paths,
    sequence lengths, split policy, and feature semantics.
    """

    # --------------------------------------------------------
    # Dataset identity and source location
    # --------------------------------------------------------
    # A short logical dataset name used for logging, debugging,
    # and future support of alternate dataset sources.
    dataset_name: str = "azt1d"

    # Public download URL for the raw dataset archive/file.
    # Set to `None` when the processed/raw files will already
    # exist locally and no download should occur.
    dataset_url: Optional[str] = (
        "https://data.mendeley.com/public-files/datasets/"
        "gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded"
    )

    # --------------------------------------------------------
    # Filesystem locations
    # --------------------------------------------------------
    # `raw_dir` stores the original downloaded vendor files.
    raw_dir: PathInput = Path("data/raw")

    # `cache_dir` stores temporary/cache artifacts used during
    # download and extraction.
    cache_dir: PathInput = Path("data/cache")

    # `extracted_dir` stores unpacked raw dataset contents when
    # the source arrives as an archive.
    extracted_dir: PathInput = Path("data/extracted")

    # `processed_dir` holds the canonical cleaned file consumed
    # by the dataset and datamodule layers.
    processed_dir: PathInput = Path("data/processed")

    # Name of the canonical processed file inside
    # `processed_dir`.
    processed_file_name: str = "azt1d_processed.csv"

    # --------------------------------------------------------
    # Canonical column names used throughout the pipeline
    # --------------------------------------------------------
    # These are the shared semantic names used by schema,
    # transforms, indexing, and dataset assembly.
    subject_id_column: str = "subject_id"
    time_column: str = "timestamp"
    target_column: str = "glucose_mg_dl"

    # --------------------------------------------------------
    # Sequence-construction parameters
    # --------------------------------------------------------
    # Expected spacing between cleaned samples after
    # preprocessing/standardization.
    sampling_interval_minutes: int = 5

    # Historical window length consumed by the model.
    encoder_length: int = 168

    # Forecast horizon length emitted by the model.
    prediction_length: int = 12

    # Sliding-window step size used when generating consecutive
    # sequence examples from a longer timeline.
    window_stride: int = 1

    # --------------------------------------------------------
    # Dataset split behavior
    # --------------------------------------------------------
    # Ratios for train/validation/test. These are validated to
    # sum to 1.0.
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # If True, entire subjects are assigned to only one split.
    split_by_subject: bool = False

    # If True, each subject timeline is split chronologically
    # into train/validation/test segments.
    split_within_subject: bool = True

    # --------------------------------------------------------
    # DataLoader behavior
    # --------------------------------------------------------
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last_train: bool = False

    # --------------------------------------------------------
    # Rebuild / redownload controls
    # --------------------------------------------------------
    # If True, rebuild the processed file even if it already
    # exists.
    rebuild_processed: bool = False

    # If True, force a fresh download even if the raw file
    # already exists locally.
    redownload: bool = False

    # --------------------------------------------------------
    # Shared feature schema
    # --------------------------------------------------------
    # This is the semantic bridge between the data layer and the
    # model layer. When populated, downstream code can derive
    # feature groups from it instead of relying on AZT1D-specific
    # fallback defaults.
    features: Sequence[FeatureSpec] = field(default_factory=tuple)

    @property
    def processed_file_path(self) -> Path:
        """
        Full path to the canonical processed dataset file.

        Keeping this as a property avoids repeating
        `processed_dir / processed_file_name` throughout the
        pipeline.
        """
        # `processed_dir` is normalized to `Path` in `__post_init__`, but the
        # public field still accepts `str | Path` at construction time. We
        # coerce again here so the property's static type stays accurate even
        # under strict type checking.
        return Path(self.processed_dir) / self.processed_file_name

    def __post_init__(self) -> None:
        """
        Normalize path inputs and validate the core data-pipeline contract.

        Context:
        this keeps obvious configuration mistakes close to construction time and
        ensures downstream data code can assume normalized `Path` objects and a
        coherent split/dataloader policy.
        """
        # Normalize path-like inputs once so the rest of the
        # codebase can assume `Path` instances.
        self.raw_dir = Path(self.raw_dir)
        self.cache_dir = Path(self.cache_dir)
        self.extracted_dir = Path(self.extracted_dir)
        self.processed_dir = Path(self.processed_dir)
        self.features = tuple(self.features)

        # Validate basic numeric parameters early so config
        # mistakes fail fast near construction time.
        if self.encoder_length <= 0:
            raise ValueError("encoder_length must be > 0")
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be > 0")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be > 0")
        if self.sampling_interval_minutes <= 0:
            raise ValueError("sampling_interval_minutes must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be > 0 or None")

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"train/val/test ratios must sum to 1.0, got {ratio_sum:.4f}"
            )

        if self.split_by_subject and self.split_within_subject:
            raise ValueError(
                "split_by_subject and split_within_subject cannot both be True"
            )
