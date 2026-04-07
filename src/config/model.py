from __future__ import annotations

# AI-assisted maintenance note:
# This module owns the model-side configuration contracts. It groups the fused
# model's branch-specific configuration in one place while leaving runtime
# orchestration and observability policy in separate modules.

from dataclasses import dataclass, field
from typing import Sequence

from config.data import DataConfig
from utils.tft_utils import DataTypes, FeatureSpec, InputTypes


@dataclass
class TCNConfig:
    """
    Configuration for the lean project-specific TCN branch.

    Purpose:
    hold architecture settings for the lean project-specific TCN branch used
    inside the fused model.

    Context:
    earlier versions of the repository carried a broader, more generic TCN
    configuration surface. The refactor intentionally narrowed that contract to
    the options the local implementation in `src/models/tcn.py` actually
    supports.

    Important note:
    this config keeps the fused model aligned with the current local branch
    implementation instead of implying support for a broader generic TCN API
    that the repository does not expose.
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
        """
        Normalize sequence-like inputs and validate the narrow project TCN contract.

        Context:
        the refactored TCN intentionally supports a smaller configuration
        surface than older generic variants, so unsupported combinations should
        fail here rather than later during model construction.
        """
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


@dataclass
class TFTConfig:
    """
    Configuration for the Temporal Fusion Transformer branch.

    Purpose:
    hold architecture settings and derived input counts for the Temporal Fusion
    Transformer branch.

    Context:
    the TFT branch depends on both declarative feature semantics and
    runtime-discovered categorical metadata, so this config acts as the bridge
    between those two worlds.

    Operational role:
    the DataModule provides categorical cardinalities and sequence details,
    while the model relies on the resulting bound config as a stable typed
    contract for the TFT branch.
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
    # Shared epsilon for layer normalization across TFT submodules, including
    # GRN-backed blocks. Keeping this in TFTConfig lets the model remain
    # numerically consistent without introducing a separate GRN config surface
    # before the project actually needs one.
    layer_norm_eps: float = 1e-3

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
        """
        Derive TFT input counts from the semantic feature schema and validate the result.

        Context:
        this config is the bridge between declarative feature semantics and the
        runtime-bound metadata the TFT branch needs, so the derived counts live
        here instead of being recomputed ad hoc inside model code.
        """
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
        if self.layer_norm_eps <= 0.0:
            raise ValueError("layer_norm_eps must be > 0.0")
        if self.encoder_length <= 0:
            raise ValueError("encoder_length must be > 0")
        if self.example_length < self.encoder_length:
            raise ValueError("example_length must be >= encoder_length")
        if self.num_aux_future_features < 0:
            raise ValueError("num_aux_future_features must be >= 0")

        for q in self.quantiles:
            if q <= 0.0 or q >= 1.0:
                raise ValueError(f"quantile values must be in (0, 1), got {q}")


@dataclass
class Config:
    """
    Top-level project configuration grouping data and model-branch contracts.

    Purpose:
    group together the data contract and the two model-branch contracts into
    one object passed through training and bootstrap code.

    Context:
    the repository needs to move one coherent configuration payload through
    data preparation, runtime binding, model construction, checkpoint
    serialization, and top-level orchestration.
    """

    # Shared data-pipeline configuration.
    data: DataConfig

    # Temporal Fusion Transformer branch configuration.
    tft: TFTConfig

    # Temporal Convolution Network branch configuration.
    tcn: TCNConfig
