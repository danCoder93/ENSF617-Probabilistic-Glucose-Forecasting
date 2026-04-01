from __future__ import annotations

"""
AI-assisted maintenance note:
This config file has evolved across multiple refactor passes in this project
and was updated with AI assistance before being reviewed/adapted for the
current codebase.

Recent evolution captured in this version:
- the data-layer refactor established `DataConfig` as the shared contract for
  dataset access, preprocessing, sequence construction, and loader behavior
- the model-layer refactor narrowed `TCNConfig` to match the lean fused-model
  TCN branch rather than an older generic-library-style surface
- `TFTConfig` was kept aligned with the newer runtime-bound path where the
  DataModule and fused model inject discovered metadata such as categorical
  cardinalities, sequence lengths, and auxiliary future features
- unresolved merge-conflict markers and duplicate/contradictory definitions
  from earlier edits were removed so this module again provides one coherent
  source of truth for configuration

The goal of this file is to keep the shared configuration contract readable and
stable while remaining compatible with both older declarative construction and
the newer refactored data/model integration path.
"""

from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from typing import Optional, Sequence

from utils.tft_utils import DataTypes, FeatureSpec, InputTypes


# ============================================================
# DataConfig
# ============================================================
# Purpose:
#   Hold dataset access, preprocessing, split, and DataLoader
#   settings shared across the refactored data pipeline.
#
# Why this exists:
#   The data refactor separated downloading, preprocessing,
#   schema derivation, indexing, dataset assembly, and loader
#   creation into distinct layers, but those layers still need
#   one common configuration contract.
# ============================================================
@dataclass
class DataConfig:
    """
    Configuration for dataset access, preprocessing, splits, and dataloaders.
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
    raw_dir: Path = Path("data/raw")

    # `cache_dir` stores temporary/cache artifacts used during
    # download and extraction.
    cache_dir: Path = Path("data/cache")

    # `extracted_dir` stores unpacked raw dataset contents when
    # the source arrives as an archive.
    extracted_dir: Path = Path("data/extracted")

    # `processed_dir` holds the canonical cleaned file consumed
    # by the dataset and datamodule layers.
    processed_dir: Path = Path("data/processed")

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
        return self.processed_dir / self.processed_file_name

    def __post_init__(self) -> None:
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

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"train/val/test ratios must sum to 1.0, got {ratio_sum:.4f}"
            )

        if self.split_by_subject and self.split_within_subject:
            raise ValueError(
                "split_by_subject and split_within_subject cannot both be True"
            )


# ============================================================
# TCNConfig
# ============================================================
# Purpose:
#   Hold architecture settings for the lean project-specific
#   TCN branch used inside the fused model.
#
# Why this exists:
#   Earlier versions of the repo carried a broader, more generic
#   TCN configuration surface. The refactor intentionally
#   narrowed that contract to the options the local TCN
#   implementation actually supports.
# ============================================================
@dataclass
class TCNConfig:
    """
    Configuration for the lean project-specific TCN branch.

    This config intentionally matches the narrowed TCN implementation in
    `src/models/tcn.py` rather than the older generic-library surface.
    """

    # Number of encoder-side temporal input features presented
    # to one TCN branch.
    num_inputs: int = 1

    # Output channel widths for the stacked residual temporal
    # blocks. One entry per block.
    num_channels: Sequence[int] = field(
        default_factory=lambda: (64, 64, 128)
    )

    # Temporal convolution kernel width.
    kernel_size: int = 3

    # Dilation schedule for the residual blocks. In the current
    # fused design this commonly stays at `(1, 2, 4)`.
    dilations: Sequence[int] = field(
        default_factory=lambda: (1, 2, 4)
    )

    # Dropout applied inside temporal blocks and in the forecast
    # head.
    dropout: float = 0.1

    # Activation used inside the residual blocks and forecast
    # head.
    activation: str = "relu"

    # The refactored TCN branch is always causal and always uses channel-wise
    # layer norm internally. Keep this field only as an explicit contract check
    # so config values do not imply unsupported alternatives.
    use_norm: str = "layer_norm"

    # Forecast horizon length emitted by each TCN branch.
    prediction_length: int = 12

    # Number of forecast channels emitted at each horizon step.
    output_size: int = 1

    def __post_init__(self) -> None:
        # Convert list-like user inputs to tuples so the config
        # becomes immutable in practice after construction.
        self.num_channels = tuple(int(channel) for channel in self.num_channels)
        self.dilations = tuple(int(dilation) for dilation in self.dilations)

        if self.num_inputs <= 0:
            raise ValueError("num_inputs must be > 0")
        if not self.num_channels:
            raise ValueError("num_channels must not be empty")
        if any(channel <= 0 for channel in self.num_channels):
            raise ValueError("num_channels must contain only positive values")
        if not self.dilations:
            raise ValueError("dilations must not be empty")
        if any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("dilations must contain only positive values")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be > 0")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be > 0")
        if self.output_size <= 0:
            raise ValueError("output_size must be > 0")
        if self.use_norm != "layer_norm":
            raise ValueError(
                "The refactored TCN implementation currently supports only layer_norm"
            )


# ============================================================
# TFTConfig
# ============================================================
# Purpose:
#   Hold architecture settings and derived input counts for the
#   Temporal Fusion Transformer branch.
#
# Why this exists:
#   The TFT branch depends on both declarative feature semantics
#   and runtime-discovered categorical metadata, so this config
#   acts as the bridge between those two worlds.
# ============================================================
@dataclass
class TFTConfig:
    """
    Configuration for the Temporal Fusion Transformer branch.
    """

    # --------------------------------------------------------
    # Shared feature schema
    # --------------------------------------------------------
    # These semantic feature declarations are used to derive the
    # continuous input counts expected by the TFT embedding and
    # variable-selection blocks.
    features: Sequence[FeatureSpec] = field(default_factory=tuple)

    # --------------------------------------------------------
    # Runtime categorical cardinalities
    # --------------------------------------------------------
    # These are typically discovered by the DataModule after it
    # inspects the cleaned dataframe and builds category maps.
    static_categorical_inp_lens: Sequence[int] = field(default_factory=tuple)
    temporal_known_categorical_inp_lens: Sequence[int] = field(default_factory=tuple)
    temporal_observed_categorical_inp_lens: Sequence[int] = field(default_factory=tuple)

    # --------------------------------------------------------
    # Core TFT hyperparameters
    # --------------------------------------------------------
    n_head: int = 4
    hidden_size: int = 128
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # Quantiles predicted by the TFT output head.
    quantiles: Sequence[float] = field(default_factory=lambda: (0.1, 0.5, 0.9))

    # These are runtime-bound to the data contract by the DataModule / fused
    # model path, so they remain explicit config fields.
    #
    # `encoder_length` is the history length, `example_length`
    # is the total history+future sequence length expected by
    # attention masking, and `num_aux_future_features` counts
    # additional future continuous channels such as TCN branch
    # forecasts injected into the TFT decoder inputs.
    encoder_length: int = 168
    example_length: int = 180
    num_aux_future_features: int = 0

    # --------------------------------------------------------
    # Derived input counts
    # --------------------------------------------------------
    # These are recomputed in `__post_init__` from `features`
    # plus runtime metadata. They remain explicit fields so the
    # rest of the model code can rely on a clear typed contract.
    temporal_known_continuous_inp_size: int = 0
    temporal_observed_continuous_inp_size: int = 0
    temporal_target_size: int = 0
    static_continuous_inp_size: int = 0

    num_static_vars: int = 0
    num_future_vars: int = 0
    num_historic_vars: int = 0

    def __post_init__(self) -> None:
        # Normalize list-like inputs to tuples so callers can
        # build configs flexibly while downstream code receives a
        # consistent immutable-looking shape.
        self.features = tuple(self.features)
        self.static_categorical_inp_lens = tuple(
            int(cardinality) for cardinality in self.static_categorical_inp_lens
        )
        self.temporal_known_categorical_inp_lens = tuple(
            int(cardinality)
            for cardinality in self.temporal_known_categorical_inp_lens
        )
        self.temporal_observed_categorical_inp_lens = tuple(
            int(cardinality)
            for cardinality in self.temporal_observed_categorical_inp_lens
        )
        self.quantiles = tuple(float(q) for q in self.quantiles)

        # Derive continuous input counts from semantic
        # `FeatureSpec` declarations. Categorical counts are
        # represented separately via the cardinality lists above.
        self.temporal_known_continuous_inp_size = len(
            [
                feature
                for feature in self.features
                if feature.feature_type == InputTypes.KNOWN
                and feature.feature_embed_type == DataTypes.CONTINUOUS
            ]
        )
        self.temporal_observed_continuous_inp_size = len(
            [
                feature
                for feature in self.features
                if feature.feature_type == InputTypes.OBSERVED
                and feature.feature_embed_type == DataTypes.CONTINUOUS
            ]
        )

        # Targets are counted by semantic role rather than embed
        # type because the TFT target path has its own dedicated
        # handling.
        self.temporal_target_size = len(
            [
                feature
                for feature in self.features
                if feature.feature_type == InputTypes.TARGET
            ]
        )
        self.static_continuous_inp_size = len(
            [
                feature
                for feature in self.features
                if feature.feature_type == InputTypes.STATIC
                and feature.feature_embed_type == DataTypes.CONTINUOUS
            ]
        )

        # Aggregate variable counts used by the TFT variable
        # selection networks.
        self.num_static_vars = (
            self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        )
        self.num_future_vars = (
            self.temporal_known_continuous_inp_size
            + self.num_aux_future_features
            + len(self.temporal_known_categorical_inp_lens)
        )
        self.num_historic_vars = (
            self.num_future_vars
            + self.temporal_observed_continuous_inp_size
            + self.temporal_target_size
            + len(self.temporal_observed_categorical_inp_lens)
        )

        # Validate the runtime-bound categorical metadata before
        # model construction.
        if any(cardinality <= 0 for cardinality in self.static_categorical_inp_lens):
            raise ValueError("static_categorical_inp_lens must contain only positive values")
        if any(
            cardinality <= 0
            for cardinality in self.temporal_known_categorical_inp_lens
        ):
            raise ValueError(
                "temporal_known_categorical_inp_lens must contain only positive values"
            )
        if any(
            cardinality <= 0
            for cardinality in self.temporal_observed_categorical_inp_lens
        ):
            raise ValueError(
                "temporal_observed_categorical_inp_lens must contain only positive values"
            )
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.n_head <= 0:
            raise ValueError("n_head must be > 0")
        if self.hidden_size % self.n_head != 0:
            raise ValueError("hidden_size must be divisible by n_head")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.attn_dropout < 0.0 or self.attn_dropout >= 1.0:
            raise ValueError("attn_dropout must be in [0.0, 1.0)")
        if self.encoder_length <= 0:
            raise ValueError("encoder_length must be > 0")
        if self.example_length < self.encoder_length:
            raise ValueError("example_length must be >= encoder_length")
        if self.num_aux_future_features < 0:
            raise ValueError("num_aux_future_features must be >= 0")

        for q in self.quantiles:
            if q <= 0.0 or q >= 1.0:
                raise ValueError(f"quantile values must be in (0, 1), got {q}")


# ============================================================
# Top-level Config
# ============================================================
# Purpose:
#   Group together the data contract and the two model-branch
#   contracts into one object passed through training/bootstrap
#   code.
# ============================================================
@dataclass
class Config:
    # Shared data-pipeline configuration.
    data: DataConfig

    # Temporal Fusion Transformer branch configuration.
    tft: TFTConfig

    # Temporal Convolution Network branch configuration.
    tcn: TCNConfig
