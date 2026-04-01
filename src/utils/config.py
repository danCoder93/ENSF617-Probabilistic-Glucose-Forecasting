from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
<<<<<<< Updated upstream
try:
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover - compatibility for older numpy versions
    NDArray = np.ndarray
=======
from numpy.typing import NDArray
>>>>>>> Stashed changes

from utils.tft_utils import FeatureSpec, InputTypes, DataTypes


# ============================================================
# DataConfig
# ============================================================
# Purpose:
#   Holds all configuration related to dataset access, caching,
#   preprocessing, sequence construction, dataset splitting,
#   and DataLoader behavior.
#
# Why this exists:
#   These settings are shared by the DataModule / Dataset side
#   of the pipeline
# ============================================================
@dataclass
class DataConfig:
    # --------------------------------------------------------
    # Dataset identity and source location
    # --------------------------------------------------------
    # A short logical name for the dataset. Useful for logging,
    # debugging, or future support of multiple datasets.
    dataset_name: str = "azt1d"

    # Public download URL for the raw dataset archive/file.
    # Set to None if the dataset will already exist locally.
    dataset_url: Optional[str] = (
        "https://data.mendeley.com/public-files/datasets/"
        "gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded"
    )

    # --------------------------------------------------------
    # Filesystem locations
    # --------------------------------------------------------
    # raw_dir:
    #   Directory where the original downloaded file is stored.
    raw_dir: Path = Path("data/raw")

    # cache_dir:
    #   Optional directory for HTTP cache / temporary artifacts.
    cache_dir: Path = Path("data/cache")

    # extracted_dir:
    #   Directory where extracted raw files go if the download
    #   is an archive (zip, tar, etc.).
    extracted_dir: Path = Path("data/extracted")

    # processed_dir:
    #   Directory for processed / canonical data files that are
    #   ready for the Dataset / DataModule to consume.
    processed_dir: Path = Path("data/processed")

    # processed_file_name:
    #   Name of the processed dataset file that downstream code
    #   loads after download/extract/standardization.
    processed_file_name: str = "azt1d_processed.csv"

    # --------------------------------------------------------
    # Canonical column names used throughout the pipeline
    # --------------------------------------------------------
    subject_id_column: str = "subject_id"
    time_column: str = "timestamp"
    target_column: str = "glucose_mg_dl"

    # --------------------------------------------------------
    # Time-series sample construction parameters
    # --------------------------------------------------------
    # sampling_interval_minutes:
    #   Expected time spacing between consecutive rows after
    #   standardization.
    sampling_interval_minutes: int = 5

    # encoder_length:
    #   Number of historical time steps provided to the model.
    #   This is the "past context" length.
    #
    # Example:
    #   168 steps at 5 min/step = 14 hours of history.
    encoder_length: int = 168

    # prediction_length:
    #   Number of future time steps the model predicts.
    #
    # Example:
    #   12 steps at 5 min/step = 60 minutes ahead.
    prediction_length: int = 12

    # window_stride:
    #   Step size used when generating consecutive training
    #   samples from a long time series.
    #
    # Example:
    #   stride=1  -> every possible window
    #   stride=12 -> start each new sample 1 hour later
    window_stride: int = 1

    # --------------------------------------------------------
    # Dataset split behavior
    # --------------------------------------------------------
    # Ratios for train / validation / test.
    # These should sum to 1.0.
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # split_by_subject:
    #   If True, entire subjects are assigned to one split only.
    #   This is stronger leakage prevention across subjects.
    split_by_subject: bool = False

    # split_within_subject:
    #   If True, each subject timeline is split chronologically
    #   into train/val/test segments.
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
    # rebuild_processed:
    #   If True, regenerate processed data even if it exists.
    rebuild_processed: bool = False

    # redownload:
    #   If True, force redownload even if the raw file exists.
    redownload: bool = False

    # --------------------------------------------------------
    # Shared feature schema
    # --------------------------------------------------------
    # This is the canonical feature description used by both
    # the DataModule / Dataset and the TFT configuration.
    features: Sequence[FeatureSpec] = field(default_factory=list)

    @property
    def processed_file_path(self) -> Path:
        """
        Full path to the processed CSV file.

        Keeping this as a property avoids repeating:
            processed_dir / processed_file_name
        all over the codebase.
        """
        return self.processed_dir / self.processed_file_name

    def __post_init__(self) -> None:
        """
        Validate configuration values early so mistakes fail fast.
        """
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
        if not np.isclose(ratio_sum, 1.0):
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
#   Holds architecture settings specific to the TCN branch.
#
# ============================================================
@dataclass
class TCNConfig:
    # num_inputs:
    #   Number of input channels/features passed to the TCN.
    #   If the TCN consumes encoder continuous features only,
    num_inputs: int

    # num_channels:
    #   Output channel width for each stacked temporal block.
    #   Example: [64, 64, 128]
    num_channels: Sequence[int] | NDArray[np.int_]

    # kernel_size:
    #   Temporal convolution kernel size.
    kernel_size: int = 3

    # dilations:
    #   Dilation factors for temporal convolutions.
    #   Using default_factory avoids a shared mutable default.
    dilations: Sequence[int] | NDArray[np.int_] = field(
        default_factory=lambda: [1, 2, 4]
    )

    # dilation_reset:
    #   Optional reset period for dilation growth.
    dilation_reset: Optional[int] = None

    # dropout:
    #   Dropout used inside the TCN.
    dropout: float = 0.1

<<<<<<< Updated upstream
    # use_norm:
    #   Normalization used in the TCN residual blocks.
    #   The lean fused implementation standardizes on layer norm so the
    #   convolution branch behaves consistently with the TFT-heavy pipeline.
    use_norm: str = "layer_norm"
=======
    # causal:
    #   If True, convolutions only use current and past time steps.
    causal: bool = True

    # use_norm:
    #   Type of normalization used in the TCN blocks.
    #   Common options may include: weight_norm, batch_norm, layer_norm.
    use_norm: str = "weight_norm"
>>>>>>> Stashed changes

    # activation:
    #   Activation function used in temporal blocks.
    activation: str = "relu"

<<<<<<< Updated upstream
    # prediction_length:
    #   Number of future steps each TCN branch forecasts.
    prediction_length: int = 12

    # output_size:
    #   Number of forecast channels emitted by each TCN branch.
    output_size: int = 1
=======
    # kernel_initializer:
    #   Weight initialization strategy for convolution kernels.
    kernel_initializer: str = "xavier_uniform"

    # use_skip_connections:
    #   Whether to add skip connections across TCN blocks.
    use_skip_connections: bool = True

    # input_shape:
    #   Layout expected by the TCN implementation.
    #   "NCL" usually means:
    #       N = batch
    #       C = channels/features
    #       L = temporal length
    input_shape: str = "NCL"

    # embedding_shapes / embedding_mode:
    #   Optional support for categorical embeddings if the TCN
    #   implementation supports them.
    embedding_shapes: Optional[Sequence[int] | NDArray[np.int_]] = None
    embedding_mode: str = "add"

    # use_gate:
    #   Whether to use gated activations in the TCN blocks.
    use_gate: bool = False

    # lookahead:
    #   Optional future context used by non-causal variants.
    #   Should normally stay 0 for causal forecasting.
    lookahead: int = 0

    # output_projection:
    #   Optional final projection size after TCN blocks.
    output_projection: Optional[int] = None

    # output_activation:
    #   Optional final activation applied to TCN output.
    output_activation: Optional[str] = None
>>>>>>> Stashed changes

    def __post_init__(self) -> None:
        """
        Basic validation for TCN architecture parameters.
        """
        if self.num_inputs <= 0:
            raise ValueError("num_inputs must be > 0")

        if len(self.num_channels) == 0:
            raise ValueError("num_channels must not be empty")

        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be > 0")

        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

<<<<<<< Updated upstream
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be > 0")

        if self.output_size <= 0:
            raise ValueError("output_size must be > 0")

        if self.use_norm != "layer_norm":
            raise ValueError("The fused TCN implementation currently supports only layer_norm")
=======
        if self.lookahead < 0:
            raise ValueError("lookahead must be >= 0")
>>>>>>> Stashed changes


# ============================================================
# TFTConfig
# ============================================================
# Purpose:
#   Holds architecture settings specific to the Temporal Fusion
#   Transformer branch.
#
# ============================================================
@dataclass
class TFTConfig:
    # --------------------------------------------------------
    # Shared feature schema
    # --------------------------------------------------------
    # This is required because __post_init__ derives input counts
    # from the declared feature semantics.
    features: Sequence[FeatureSpec] = field(default_factory=list)

    # --------------------------------------------------------
    # Categorical cardinalities
    # --------------------------------------------------------
    # These store the vocabulary size / cardinality for each
    # categorical feature group. Example:
    #   static_categorical_inp_lens = [num_subjects]
    #
    # These are model-facing embedding metadata.
    static_categorical_inp_lens: Sequence[int] = field(default_factory=list)
    temporal_known_categorical_inp_lens: Sequence[int] = field(default_factory=list)
    temporal_observed_categorical_inp_lens: Sequence[int] = field(default_factory=list)

    # --------------------------------------------------------
    # Core TFT architecture hyperparameters
    # --------------------------------------------------------
    n_head: int = 4
    hidden_size: int = 128
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # Quantiles predicted by the TFT output head.
    # Example:
    #   [0.1, 0.5, 0.9] for probabilistic forecasting.
    quantiles: Sequence[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

<<<<<<< Updated upstream
    # Sequence lengths used by the current TFT implementation.
    encoder_length: int = 168
    example_length: int = 180

    # Number of auxiliary continuous decoder channels injected from the
    # external TCN forecasters.
    num_aux_future_features: int = 0

=======
>>>>>>> Stashed changes
    # --------------------------------------------------------
    # Derived variable counts
    # --------------------------------------------------------
    # These are computed from `features` and categorical cardinality
    # metadata during __post_init__.
    temporal_known_continuous_inp_size: int = 0
    temporal_observed_continuous_inp_size: int = 0
    temporal_target_size: int = 0
    static_continuous_inp_size: int = 0

    num_static_vars: int = 0
    num_future_vars: int = 0
    num_historic_vars: int = 0

    def __post_init__(self) -> None:
        """
        Derive TFT input sizes from the shared feature schema.

        We separate continuous and categorical counts because TFT
        often embeds categorical variables while projecting continuous
        variables differently.

        Semantic meanings:
        - KNOWN:
            features known in advance over the forecast horizon
            (e.g., time-of-day encodings).
        - OBSERVED:
            features only observed up to the prediction time
            (e.g., delivered insulin, carbs, measured device state).
        - TARGET:
            variable(s) we want to forecast
            (e.g., glucose).
        - STATIC:
            time-invariant metadata
            (e.g., subject identity).
        """
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

        # Total number of static variables entering TFT.
        # This includes:
        #   - static continuous variables
        #   - static categorical variables
        self.num_static_vars = (
            self.static_continuous_inp_size
            + len(self.static_categorical_inp_lens)
        )

        # Total number of future-known variables entering the decoder.
        self.num_future_vars = (
            self.temporal_known_continuous_inp_size
<<<<<<< Updated upstream
            + self.num_aux_future_features
=======
>>>>>>> Stashed changes
            + len(self.temporal_known_categorical_inp_lens)
        )

        # Total number of historic variables entering the encoder.
        #
        # Historic variables include:
        #   - all future-known variables that also exist on the history axis
        #   - observed continuous variables
        #   - target variable(s)
        #   - observed categorical variables
        self.num_historic_vars = (
            self.num_future_vars
            + self.temporal_observed_continuous_inp_size
            + self.temporal_target_size
            + len(self.temporal_observed_categorical_inp_lens)
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

<<<<<<< Updated upstream
        if self.encoder_length <= 0:
            raise ValueError("encoder_length must be > 0")

        if self.example_length < self.encoder_length:
            raise ValueError("example_length must be >= encoder_length")

        if self.num_aux_future_features < 0:
            raise ValueError("num_aux_future_features must be >= 0")

=======
>>>>>>> Stashed changes
        for q in self.quantiles:
            if q <= 0.0 or q >= 1.0:
                raise ValueError(f"quantile values must be in (0, 1), got {q}")


# ============================================================
# Top-level pipeline config
# ============================================================
# Purpose:
#   Groups together data configuration and model branch configs
#   into a single object passed around the pipeline.
# ============================================================
@dataclass
class Config:
    data: DataConfig
    tft: TFTConfig
<<<<<<< Updated upstream
    tcn: TCNConfig
=======
    tcn: TCNConfig
>>>>>>> Stashed changes
