from __future__ import annotations

# AI-assisted maintenance note (April 1, 2026):
# This existing module was refined with AI assistance under direct user
# guidance. The current implementation aligns the fused path more closely with
# the project's architecture diagram by making the TCN branches and the TFT
# branch meet at the fusion layer, rather than feeding TCN forecasts into TFT
# as auxiliary future decoder inputs.
#
# The post-branch fusion GRN still uses the same config-backed construction
# path as the internal TFT GRNs, but the fused model now concatenates
# horizon-wise latent branch features before the final `NNHead` readout. This
# makes the final head a true predictive head over fused representations rather
# than a late reconciliation layer over already-decoded TFT quantiles.

from dataclasses import replace
from typing import Any, Mapping

from pytorch_lightning import LightningModule

import torch
from torch import Tensor
from torch.optim import Adam, AdamW, Optimizer

try:
    from torchmetrics import MeanAbsoluteError, MeanSquaredError
except ImportError:  # pragma: no cover - optional dependency until installed
    # `torchmetrics` is preferred for Lightning-native metric state handling,
    # but the model keeps manual fallbacks so unit tests and lightweight
    # environments do not fail purely because the extra package is missing.
    MeanAbsoluteError = None  # type: ignore[assignment]
    MeanSquaredError = None  # type: ignore[assignment]

from models.tcn import TCN
from models.tft import TemporalFusionTransformer
from models.grn import GRN
from models.nn_head import NNHead
from evaluation import (
    mean_absolute_error,
    mean_prediction_interval_width,
    normalize_target_tensor,
    root_mean_squared_error,
    select_point_prediction,
)

from config import Config, config_from_dict, config_to_dict


class FusedModel(LightningModule):
    """
    Hybrid glucose forecaster built from fused TCN and TFT branches.

    Purpose:
    provide one Lightning-native model that owns the fused architecture, the
    probabilistic training objective, the optimizer contract, and the core
    train/validation/test logging behavior.

    Context:
    this module reflects the repository's current late-fusion design rather
    than the earlier idea of feeding TCN forecasts directly into the TFT
    decoder. The TCN branches and TFT branch now produce horizon-wise latent
    features that meet at the fusion layer before final quantile prediction.

    Architecture overview:
    - three multi-kernel TCN branches for short/mid-range forecasts
    - a TFT branch that models future-aware decoder context
    - a final fusion path where TCN and TFT latent features meet before output

    Tensor-flow summary:
    1. Split encoder history into known / observed / target groups.
    2. Send observed history + target history to the TCN branches.
    3. Send the semantically grouped batch inputs to TFT.
    4. Fuse the resulting horizon-wise latent features with a GRN.
    5. Project the fused hidden representation to final quantile outputs.
    """

    def __init__(
        self,
        config: Config | Mapping[str, Any],
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_name: str = "adam",
    ) -> None:
        """
        Bind the fused architecture, optimizer defaults, and runtime-aware TFT config.

        Context:
        construction does more than instantiate submodules. It also normalizes
        checkpoint-friendly config payloads, binds the TFT branch to the real
        data contract, and materializes any lazy TFT parameters so Lightning can
        configure optimizers immediately after model creation.
        """
        super().__init__()

        # Accept either the typed project config or its serialized checkpoint
        # form. This keeps normal code paths strongly typed while allowing
        # Lightning's `load_from_checkpoint(...)` to reconstruct the model from
        # the plain hyperparameter dictionary saved in the checkpoint.
        config = self._coerce_config(config)
        self.config = config
        # These optimization hyperparameters now live on the model object
        # instead of being deferred to an external training loop for two
        # reasons:
        #
        # 1. This file is intentionally being strengthened into a true
        #    LightningModule, following the Lightning tutorial pattern where the
        #    model owns its optimizer and loss-related training behavior.
        # 2. Keeping the defaults here makes the future `train.py` much thinner.
        #    The trainer/bootstrap layer will only need to instantiate the
        #    model and hand it to Lightning's `Trainer`, rather than rebuilding
        #    optimizer configuration logic in a second location.
        #
        # In other words, these are "model-side training semantics" rather than
        # "experiment orchestration"; that makes them a good fit for the module
        # itself.
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name.lower()
        # `save_hyperparameters(...)` is a Lightning convention that stores
        # constructor values inside checkpoints/logs. That gives us a useful
        # paper trail when resuming runs later: the saved model can tell us
        # which optimizer settings it expected without requiring a separate
        # experiment notebook or manually maintained metadata file.
        #
        # The top-level `Config` is serialized to a plain nested dictionary
        # first rather than saved as the dataclass object directly. That extra
        # step is deliberate:
        # - Lightning checkpoints should remain easy to reload in notebook and
        #   Colab environments
        # - plain dict/list/str/number payloads are much more portable than
        #   custom Python objects with enums, Paths, and namedtuples inside
        # - `load_from_checkpoint(...)` can then call this constructor with the
        #   saved payload and let `_coerce_config(...)` rebuild the typed config
        self.save_hyperparameters(
            {
                "config": config_to_dict(config),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "optimizer_name": self.optimizer_name,
            }
        )

        # The top-level config contains both declarative architecture settings
        # and data-dependent values that are only fully known once the
        # DataModule has established the sequence lengths and feature schema.
        #
        # `replace(...)` is used here to bind the TFT branch to the actual
        # dataset contract used by this fused model instance:
        # - encoder length comes from the data config
        # - example length is encoder + prediction horizon
        # - auxiliary future features are disabled in the aligned architecture
        #   because TCN and TFT now meet only at the late fusion stage
        tft_config = replace(
            config.tft,
            encoder_length=config.data.encoder_length,
            example_length=config.data.encoder_length + config.data.prediction_length,
            num_aux_future_features=0,
        )
        self.tft_config = tft_config

        self.num_known_cont = tft_config.temporal_known_continuous_inp_size
        self.num_observed_cont = tft_config.temporal_observed_continuous_inp_size
        self.num_target = max(1, tft_config.temporal_target_size)
        self.num_known_cat = len(tft_config.temporal_known_categorical_inp_lens)
        self.num_observed_cat = len(tft_config.temporal_observed_categorical_inp_lens)
        # These cached counts make the forward pass easier to read and ensure
        # the semantic feature splits stay driven by the same config that built
        # the TFT branch.

        # The TCN branches only consume signals that are legitimately available
        # on the encoder history axis:
        # - observed continuous variables
        # - the historical target trajectory itself
        tcn_input_size = self.num_observed_cont + self.num_target
        tcn_config = replace(
            config.tcn,
            num_inputs=tcn_input_size,
            prediction_length=config.data.prediction_length,
            output_size=1,
        )

        self.tcn3 = TCN(tcn_config)
        self.tcn5 = TCN(replace(tcn_config, kernel_size=5))
        self.tcn7 = TCN(replace(tcn_config, kernel_size=7))
        # The three kernel sizes give three different receptive-field biases
        # over the same encoder history. They all operate on the same semantic
        # inputs, but each branch can emphasize a different temporal scale.

        self.tft = TemporalFusionTransformer(tft_config)
        # TFT handles the future-aware side of the problem: known decoder inputs,
        # static context, and richer temporal reasoning over the full
        # encoder+decoder example axis.
        #
        # Lightning-specific lifecycle note:
        # `TemporalFusionTransformer` still uses a lazily materialized embedding
        # block for some continuous-input parameters. In plain PyTorch that can
        # be fine if the first real forward pass happens before optimizer
        # construction, but Lightning expects `configure_optimizers()` to work
        # against a fully materialized parameter set.
        #
        # To keep the model Lightning-safe, we proactively initialize any lazy
        # TFT embedding parameters here using a tiny synthetic batch whose
        # feature widths are derived from the already-bound config. This turns
        # optimizer setup into a deterministic constructor-time property instead
        # of relying on a later first batch.
        self._materialize_tft_lazy_parameters()

        fused_feature_size = (
            tft_config.hidden_size
            + self.tcn3.branch_hidden_size
            + self.tcn5.branch_hidden_size
            + self.tcn7.branch_hidden_size
        )
        # Keep the fusion feature width explicit because it depends on this
        # model's branch composition, while the GRN's shared defaults come from
        # the bound TFT config so the post-fusion projector stays aligned with
        # the TFT branch's hidden size, dropout, and normalization behavior.
        self.grn = GRN.from_tft_config(
            tft_config,
            input_size=fused_feature_size,
            output_size=tft_config.hidden_size,
        )
        # The final head emits one set of quantile predictions per horizon step.
        # It no longer has to invent the fusion logic itself; it only reads the
        # hidden representation produced by the fusion GRN.
        self.fcn = NNHead(
            tft_config.hidden_size,
            len(tft_config.quantiles),
            hidden_size=tft_config.hidden_size,
            feedforward_size=tft_config.hidden_size * 2,
            num_blocks=2,
            dropout=tft_config.dropout,
        )
        # Cache the quantiles once on the fused model so every training/eval
        # helper reads from the exact same ordered tuple:
        # - the final head width is derived from it
        # - the pinball loss uses it
        # - the point-forecast extraction picks from it
        #
        # Centralizing this here avoids subtle bugs where the model predicts one
        # quantile order but the loss or metrics accidentally assume another.
        self.quantiles = tuple(float(quantile) for quantile in tft_config.quantiles)
        # These stage-specific metric objects are optional by design:
        # - when `torchmetrics` is installed, Lightning can manage their state
        #   and aggregate them naturally across epochs
        # - when it is not installed, the model still computes scalar MAE/RMSE
        #   manually so training remains usable
        self.train_mae_metric = MeanAbsoluteError() if MeanAbsoluteError is not None else None
        self.val_mae_metric = MeanAbsoluteError() if MeanAbsoluteError is not None else None
        self.test_mae_metric = MeanAbsoluteError() if MeanAbsoluteError is not None else None
        self.train_rmse_metric = (
            MeanSquaredError(squared=False) if MeanSquaredError is not None else None
        )
        self.val_rmse_metric = (
            MeanSquaredError(squared=False) if MeanSquaredError is not None else None
        )
        self.test_rmse_metric = (
            MeanSquaredError(squared=False) if MeanSquaredError is not None else None
        )

    @staticmethod
    def _coerce_config(config: Config | Mapping[str, Any]) -> Config:
        """
        Normalize constructor inputs into the repository's typed `Config`.

        Context:
        local call sites pass a real dataclass config, while Lightning
        checkpoint reloads pass the serialized hyperparameter payload that was
        saved by `save_hyperparameters(...)`.
        """
        # Constructor compatibility boundary:
        # - local training code will usually pass a real `Config`
        # - checkpoint reloads pass the serialized hyperparameter payload back
        #   into `__init__`
        #
        # Rehydrating here keeps the rest of the class simple because all
        # downstream logic can continue assuming `self.config` is strongly typed.
        if isinstance(config, Config):
            return config
        return config_from_dict(config)

    def _synthetic_tft_input_group(
        self,
        *,
        time_steps: int,
        feature_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor | None:
        """
        Build one placeholder grouped tensor for TFT lazy-parameter initialization.

        Context:
        the lazy embedding only needs the eventual feature widths. Returning
        `None` for zero-width groups preserves the same optional-group contract
        used by the real forward path.
        """
        # Helper for the lazy-parameter dry run above.
        #
        # Returning `None` for zero-width groups matches the real grouped TFT
        # input contract and avoids inventing fake tensors for feature families
        # that do not exist in the current data/schema binding.
        if feature_size == 0:
            return None
        return torch.zeros(1, time_steps, feature_size, dtype=dtype)

    def _materialize_tft_lazy_parameters(self) -> None:
        """
        Force TFT lazy embedding parameters to materialize during model construction.

        Context:
        Lightning expects a fully known parameter set by the time it asks the
        module to configure optimizers, so the fused model performs a tiny
        synthetic dry run instead of waiting for the first real batch.
        """
        embedding = getattr(self.tft, "embedding", None)
        has_uninitialized_params = getattr(embedding, "has_uninitialized_params", None)
        initialize_parameters = getattr(embedding, "initialize_parameters", None)

        # If the embedding is already eagerly initialized, or if the TFT
        # implementation changes in the future to remove the lazy path
        # altogether, there is nothing to do here.
        if not callable(has_uninitialized_params) or not has_uninitialized_params():
            return
        if not callable(initialize_parameters):
            raise RuntimeError(
                "TFT embedding reports uninitialized parameters but does not expose "
                "an initialization hook."
            )

        # The TFT lazy embedding only needs feature-width information to
        # materialize its continuous-embedding matrices. Those widths are now
        # known from `tft_config`, so a one-item synthetic batch is sufficient.
        #
        # The important detail here is that we are not trying to simulate a
        # meaningful training example. We only need to trigger parameter-shape
        # binding for the lazy embedding tensors. Zero-valued placeholders are
        # therefore fine because the initializer cares about trailing feature
        # widths, not the actual numeric content.
        #
        # Shapes mirror the grouped contract used at runtime:
        # - `s_cont`: [batch, 1, num_static_cont]
        # - `k_cont`: [batch, encoder + horizon, num_known_cont]
        # - `o_cont`: [batch, encoder, num_observed_cont]
        # - `target`: [batch, encoder, num_target]
        initialize_parameters(
            {
                "s_cont": self._synthetic_tft_input_group(
                    time_steps=1,
                    feature_size=self.tft_config.static_continuous_inp_size,
                ),
                "k_cont": self._synthetic_tft_input_group(
                    time_steps=self.tft_config.example_length,
                    feature_size=self.num_known_cont,
                ),
                "o_cont": self._synthetic_tft_input_group(
                    time_steps=self.tft_config.encoder_length,
                    feature_size=self.num_observed_cont,
                ),
                "target": self._synthetic_tft_input_group(
                    time_steps=self.tft_config.encoder_length,
                    feature_size=self.num_target,
                ),
            }
        )

    def _split_encoder_continuous(self, encoder_continuous: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Recover known, observed, and target slices from the packed continuous history tensor.

        Context:
        the shared batch contract stores continuous encoder features in one
        combined tensor, but the TCN and TFT branches consume different semantic
        subsets of that history.
        """
        # `encoder_continuous` follows the shared data contract:
        # [known continuous | observed continuous | target history]
        #
        # This helper recovers the semantic groups so each branch only receives
        # the information it is supposed to consume.
        #
        # That separation matters because the branches do not share the same
        # information budget:
        # - TCN only gets encoder-side observed dynamics plus target history
        # - TFT gets a semantically richer grouped view with known future inputs
        # - the fusion layer assumes both branches are already aligned to the
        #   same decoder horizon by the time their latent features meet
        known_history = encoder_continuous[:, :, : self.num_known_cont]
        observed_end = self.num_known_cont + self.num_observed_cont
        observed_history = encoder_continuous[:, :, self.num_known_cont:observed_end]
        target_history = encoder_continuous[:, :, observed_end:]

        if target_history.shape[-1] == 0:
            # The data contract is supposed to include at least one target
            # channel, but keep a defensive fallback so the last continuous
            # feature can still be treated as target history if older callers
            # produce a slightly different layout.
            observed_history = encoder_continuous[:, :, self.num_known_cont:-1]
            target_history = encoder_continuous[:, :, -1:]

        return known_history, observed_history, target_history

    def _split_encoder_categorical(self, encoder_categorical: Tensor) -> tuple[Tensor, Tensor]:
        """
        Recover known and observed categorical histories from the packed encoder tensor.

        Context:
        categorical history follows the same semantic ordering as the continuous
        path, minus the target slice, so this helper keeps that split logic in
        one explicit place.
        """
        # Historical categorical features are ordered as:
        # [known categorical | observed categorical]
        #
        # Unlike the continuous path, there is no target slice here because the
        # target series is represented only in the continuous history channel.
        known_history = encoder_categorical[:, :, : self.num_known_cat]
        observed_history = encoder_categorical[:, :, self.num_known_cat :]
        return known_history, observed_history

    def _optional_group(self, tensor: Tensor) -> Tensor | None:
        """
        Convert zero-width grouped tensors into the `None` markers expected by TFT.

        Context:
        the data pipeline may emit empty trailing feature axes for absent groups,
        but the TFT grouped-input contract uses `None` to represent a missing
        semantic family.
        """
        # TFT expects missing feature groups to be passed as `None`, not as
        # empty tensors with zero trailing width.
        return None if tensor.shape[-1] == 0 else tensor

    def _build_tft_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
        """
        Assemble the semantic grouped-input dictionary consumed by the TFT branch.

        Context:
        this is the boundary where the DataModule's packed batch contract is
        translated into the richer grouped representation expected by the
        project-specific TFT implementation.
        """
        # Build the exact grouped tensor contract expected by `TemporalFusionTransformer`.
        #
        # Importantly, the aligned architecture no longer injects TCN outputs
        # into these TFT inputs. TFT now models the sequence using only its own
        # proper static, historical, and future-known covariates.
        #
        # Expected batch keys from the data pipeline:
        # - `static_categorical`, `static_continuous`
        # - `encoder_continuous`, `encoder_categorical`
        # - `decoder_known_continuous`, `decoder_known_categorical`
        # - `target`
        encoder_continuous = batch["encoder_continuous"]
        encoder_categorical = batch["encoder_categorical"]

        known_history_cont, observed_history_cont, target_history = self._split_encoder_continuous(
            encoder_continuous
        )
        known_history_cat, observed_history_cat = self._split_encoder_categorical(
            encoder_categorical
        )

        known_continuous = torch.cat(
            [known_history_cont, batch["decoder_known_continuous"]],
            dim=1,
        )
        # Concatenating along time recreates the full example axis expected by
        # TFT: encoder history first, decoder-known future inputs second.
        #
        # Resulting shape:
        # - `[batch, encoder_length + prediction_length, num_known_continuous]`
        #
        # This is one of the key semantic differences between the TFT path and
        # the TCN path:
        # - the TCN branches never see decoder-known future covariates
        # - TFT explicitly consumes them because it is designed to reason over
        #   the combined historical+future-known example axis
        known_categorical = torch.cat(
            [known_history_cat, batch["decoder_known_categorical"]],
            dim=1,
        )

        return {
            "s_cat": self._optional_group(batch["static_categorical"].unsqueeze(1)),
            "s_cont": self._optional_group(batch["static_continuous"].unsqueeze(1)),
            "k_cat": self._optional_group(known_categorical),
            "k_cont": known_continuous,
            "o_cat": self._optional_group(observed_history_cat),
            "o_cont": self._optional_group(observed_history_cont),
            "target": target_history,
        }

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Produce horizon-aligned quantile forecasts from one prepared batch.

        Context:
        the forward pass is intentionally staged into branch-specific feature
        extraction followed by late fusion so the TCN and TFT paths can learn
        complementary representations before final quantile prediction.
        """
        # -----------------------------
        # Step 1: Build TCN branch input
        # -----------------------------
        # The TCN branches specialize in history-only dynamics, so they consume
        # only the observed encoder-side signals plus the target trajectory.
        _, observed_history_cont, target_history = self._split_encoder_continuous(
            batch["encoder_continuous"]
        )
        tcn_inputs = torch.cat([observed_history_cont, target_history], dim=-1)
        # `tcn_inputs` shape:
        #   [batch, encoder_length, num_observed_cont + num_target]
        #
        # This tensor is intentionally narrower than the full encoder input.
        # We do not pass known-ahead covariates into the TCN branches because
        # those branches are meant to specialize in "what can be inferred from
        # observed history dynamics alone?" The TFT branch carries the richer
        # future-aware context in parallel.

        # -----------------------------
        # Step 2: Run multiscale TCNs
        # -----------------------------
        # Each branch produces horizon-aligned latent features rather than just
        # scalar predictions, which lets the fusion layer combine richer branch
        # representations.
        tcn3_features = self.tcn3.forward_features(tcn_inputs)
        tcn5_features = self.tcn5.forward_features(tcn_inputs)
        tcn7_features = self.tcn7.forward_features(tcn_inputs)
        # Each tensor has shape:
        #   [batch, prediction_length, branch_hidden_size]

        # -----------------------------
        # Step 3: Run TFT branch
        # -----------------------------
        tft_inputs = self._build_tft_inputs(batch)
        # Use TFT decoder features before quantile projection so fusion happens
        # in representation space, not after TFT has already collapsed itself
        # into final probabilistic outputs.
        tft_features = self.tft.forward_features(tft_inputs)
        # `tft_features` shape:
        #   [batch, prediction_length, tft_hidden_size]

        # -----------------------------
        # Step 4: Fuse branch features
        # -----------------------------
        fused_features = torch.cat(
            [tft_features, tcn3_features, tcn5_features, tcn7_features],
            dim=-1,
        )
        # Concatenation happens along the feature axis because all four branch
        # outputs are already aligned across:
        # - batch items
        # - forecast horizon positions
        #
        # The model is therefore fusing "different views of the same forecast
        # step" rather than mixing different timesteps together at this stage.
        #
        # That is an important architectural boundary:
        # - temporal reasoning happened inside the individual branches
        # - this stage is purely feature fusion at each decoder position
        # - the GRN and final head do not need to rediscover sequence alignment

        # -----------------------------
        # Step 5: Predict outputs
        # -----------------------------
        # The GRN acts as the nonlinear gated mixer over branch features, and
        # the final MLP head converts that fused hidden state into quantiles.
        fused_features = self.grn(fused_features)
        return self.fcn(fused_features)

    def _target_tensor(self, batch: dict[str, Any]) -> Tensor:
        """Normalize the batch target into the `[batch, horizon]` shape used by loss and metrics."""
        return normalize_target_tensor(batch["target"])

    def quantile_loss(self, predictions: Tensor, target: Tensor) -> Tensor:
        """
        Compute the pinball loss over the model's configured quantile channels.

        Context:
        this is the canonical interpretation of the fused model's output
        contract during optimization, so every training/evaluation path routes
        through this helper rather than re-implementing quantile supervision.
        """
        # This is the standard pinball loss used for quantile regression.
        #
        # Supervision contract:
        # - the final layer emits multiple quantiles at each horizon step
        # - the meaning and ordering of those channels is part of the model's
        #   output contract
        # - this method is therefore the canonical place where those output
        #   channels are interpreted during optimization
        #
        # Shape contract:
        # - predictions: [batch, horizon, num_quantiles]
        # - target:      [batch, horizon]
        #
        # The target is broadcast across the quantile axis so each configured
        # quantile receives its own asymmetric penalty:
        # - under-predicting a high quantile is penalized strongly
        # - over-predicting a low quantile is penalized strongly
        #
        # That asymmetry is what makes the outputs learn different points of the
        # predictive distribution instead of collapsing toward the same mean.
        if predictions.ndim != 3:
            raise ValueError(
                "Expected predictions to have shape [batch, horizon, quantiles], "
                f"got {tuple(predictions.shape)}."
            )
        if predictions.shape[-1] != len(self.quantiles):
            raise ValueError(
                "Prediction quantile dimension does not match configured quantiles: "
                f"{predictions.shape[-1]} != {len(self.quantiles)}."
            )

        quantiles = predictions.new_tensor(self.quantiles).view(1, 1, -1)
        errors = target.unsqueeze(-1) - predictions
        # Broadcasting works as follows:
        # - `target.unsqueeze(-1)` turns `[batch, horizon]` into
        #   `[batch, horizon, 1]`
        # - subtracting predictions expands the target across the quantile axis
        # - each quantile channel therefore receives its own signed error tensor
        #
        # The two terms inside `torch.maximum(...)` are the upper and lower
        # branches of the pinball loss. Which one is active at each position
        # depends on the sign of the prediction error.
        return torch.maximum((quantiles - 1.0) * errors, quantiles * errors).mean()

    def point_prediction(self, predictions: Tensor, *, quantile: float = 0.5) -> Tensor:
        """
        Extract one representative deterministic forecast from the probabilistic output tensor.

        Context:
        human-facing metrics such as MAE and RMSE are easier to interpret on one
        point forecast, so the model uses the configured quantile closest to the
        requested value, defaulting to the median.
        """
        # The fused model is trained probabilistically, but MAE/RMSE are easier
        # to interpret on a single deterministic forecast. The most natural
        # choice is the median (0.5 quantile), so this helper selects the
        # closest configured quantile channel and exposes it as the "point
        # forecast" for human-readable metrics.
        #
        # We intentionally select the *closest* configured quantile rather than
        # requiring that 0.5 be present exactly. That keeps the method usable if
        # a future experiment changes the quantile set to something like
        # (0.1, 0.4, 0.9) while still wanting one representative point curve for
        # logging or visualization.
        return select_point_prediction(
            predictions,
            self.quantiles,
            quantile=quantile,
        )

    def _metric_pair_for_stage(self, stage: str) -> tuple[Any | None, Any | None]:
        """Return the stage-specific torchmetrics objects, if that optional dependency is available."""
        # Keep the stage-to-metric-object mapping explicit in one place so the
        # step methods do not each have to duplicate the same branching logic.
        if stage == "train":
            return self.train_mae_metric, self.train_rmse_metric
        if stage == "val":
            return self.val_mae_metric, self.val_rmse_metric
        return self.test_mae_metric, self.test_rmse_metric

    def _log_metrics(
        self,
        stage: str,
        *,
        loss: Tensor,
        mae: Tensor,
        rmse: Tensor,
        mae_metric: Any | None,
        rmse_metric: Any | None,
        point_forecast: Tensor,
        target: Tensor,
        predictions: Tensor,
        batch_size: int,
    ) -> None:
        """
        Publish loss, point metrics, and distribution summaries for one stage.

        Context:
        this helper centralizes the model's logging policy so train/validation/
        test steps share the same observability semantics while still allowing
        stage-specific on-step and distributed-sync behavior.
        """
        # Direct `training_step(...)` calls in unit tests do not attach a
        # Trainer. In that case we still want the method to remain usable for
        # smoke tests and loss verification, so logging becomes a no-op.
        if getattr(self, "_trainer", None) is None:
            return

        on_step = stage == "train"
        # Logging policy:
        # - training metrics are logged both per step and per epoch so Lightning
        #   can show near-real-time progress while still aggregating epoch-level
        #   summaries
        # - validation/test metrics are logged on epoch only to avoid noisy,
        #   harder-to-read progress output
        #
        # `batch_size` is passed explicitly so Lightning can perform weighted
        # reductions correctly when batch sizes vary, which becomes important
        # once the real training loop is added.
        #
        # `sync_dist` is enabled for validation/test metrics so distributed
        # Lightning runs report globally reduced values instead of one process's
        # local shard.
        #
        # We leave `train_*` metrics unsynchronized on purpose:
        # - per-step training logs are primarily for local progress feedback
        # - synchronizing every training step across devices can add overhead
        # - the higher-value correctness issue is on validation/test, where the
        #   reported metrics are commonly treated as run-level results
        #
        # Additional observability note:
        # - besides loss/MAE/RMSE, we also log target distribution summaries and
        #   prediction distribution summaries so TensorBoard can reveal drift,
        #   collapse, or implausibly narrow/wide prediction intervals
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=stage != "train",
        )
        if mae_metric is not None:
            self.log(
                f"{stage}_mae",
                mae_metric,
                prog_bar=stage != "train",
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=stage != "train",
            )
        else:
            self.log(
                f"{stage}_mae",
                mae,
                prog_bar=stage != "train",
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=stage != "train",
            )
        if rmse_metric is not None:
            self.log(
                f"{stage}_rmse",
                rmse_metric,
                prog_bar=False,
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=stage != "train",
            )
        else:
            self.log(
                f"{stage}_rmse",
                rmse,
                prog_bar=False,
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=stage != "train",
            )
        self.log(
            f"{stage}_target_mean",
            target.mean(),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=stage != "train",
        )
        self.log(
            f"{stage}_target_std",
            target.std(unbiased=False),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=stage != "train",
        )
        self.log(
            f"{stage}_median_prediction_mean",
            point_forecast.mean(),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=stage != "train",
        )
        self.log(
            f"{stage}_quantile_prediction_mean",
            predictions.mean(),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=stage != "train",
        )
        interval_width = mean_prediction_interval_width(predictions)
        if interval_width is not None:
            self.log(
                f"{stage}_prediction_interval_width",
                interval_width,
                prog_bar=False,
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=stage != "train",
            )

    def _shared_step(self, batch: dict[str, Any], stage: str) -> Tensor:
        """
        Run the common prediction, loss, metric, and logging path for one stage.

        Context:
        Lightning exposes separate hooks for train/validation/test, but this
        model intentionally keeps their supervision logic identical and varies
        only the stage label used for metrics and logging policy.
        """
        # Lightning encourages the train/val/test steps to stay small and to
        # share as much logic as possible. This helper is the fused-model
        # equivalent of the tutorial's "compute predictions, compute loss, log
        # metrics" pattern.
        #
        # Centralizing the logic here ensures:
        # - the three stages supervise the exact same forecast tensor in the
        #   exact same way
        # - future metric changes are made once instead of three times
        # - the public step methods remain simple enough that they are easy to
        #   scan in `train.py` or in notebook experiments
        predictions = self(batch)
        target = self._target_tensor(batch)
        loss = self.quantile_loss(predictions, target)

        # Quantile training objective + point-metric reporting is a deliberate
        # split of responsibilities:
        # - optimization uses the full probabilistic forecast
        # - human-facing summary metrics use one representative curve
        #
        # This preserves the probabilistic nature of the model while still
        # giving us MAE/RMSE values that are intuitive to compare across runs.
        point_forecast = self.point_prediction(predictions)
        mae_metric, rmse_metric = self._metric_pair_for_stage(stage)
        # We prefer stateful torchmetrics when present because they integrate
        # cleanly with Lightning logging, but we intentionally keep exact
        # scalar fallbacks so the model remains robust in reduced environments.
        if mae_metric is not None:
            mae_metric.update(point_forecast, target)
            mae = mae_metric.compute()
        else:
            mae = mean_absolute_error(point_forecast, target)
        if rmse_metric is not None:
            rmse_metric.update(point_forecast, target)
            rmse = rmse_metric.compute()
        else:
            rmse = root_mean_squared_error(point_forecast, target)

        self._log_metrics(
            stage,
            loss=loss,
            mae=mae,
            rmse=rmse,
            mae_metric=mae_metric,
            rmse_metric=rmse_metric,
            point_forecast=point_forecast,
            target=target,
            predictions=predictions,
            batch_size=target.shape[0],
        )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Lightning training hook that delegates to the shared stage implementation."""
        del batch_idx
        # Returning the loss tensor is the Lightning contract: the Trainer will
        # take care of backward propagation, optimizer stepping, gradient
        # accumulation, mixed precision, multi-device reduction, and the rest of
        # the boilerplate that the tutorial is trying to remove from user code.
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Lightning validation hook that reuses the same supervision path as training."""
        del batch_idx
        # Validation reuses the same supervision path but leaves optimization to
        # Lightning. The explicit method still matters because Lightning uses it
        # as the hook boundary for validation scheduling and metric collection.
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Lightning test hook that keeps held-out evaluation aligned with train/val semantics."""
        del batch_idx
        # Test follows the same pattern as validation so the model's reported
        # held-out performance is computed with exactly the same loss/metric
        # semantics used during development.
        return self._shared_step(batch, "test")

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """
        Lightning prediction hook returning the full quantile forecast tensor.

        Context:
        prediction keeps the probabilistic output intact so downstream reporting
        and evaluation code can derive medians, intervals, and calibration views
        without rerunning the model.
        """
        del batch_idx, dataloader_idx
        # Prediction intentionally returns the raw quantile tensor rather than a
        # median-only projection. That keeps inference maximally informative:
        # callers can compute intervals, extract the median, or evaluate
        # calibration later without rerunning the model.
        return self(batch)

    def configure_optimizers(self) -> Optimizer:
        """
        Build the optimizer owned by this LightningModule's training contract.

        Context:
        optimizer choice and defaults live on the model because they describe
        how this model family trains, while outer workflow code is responsible
        only for orchestrating runs.
        """
        # The optimizer is configured inside the LightningModule because that is
        # the Lightning design boundary:
        # - the module knows which parameters should be trained
        # - the module owns the learning-rate / weight-decay defaults attached
        #   to this model family
        # - the outer Trainer should not need to reconstruct optimizer logic
        #
        # We support a very small surface on purpose (`adam`, `adamw`):
        # enough flexibility for common experiments, but not a sprawling API
        # before the project has a demonstrated need for it.
        optimizer_map = {
            "adam": Adam,
            "adamw": AdamW,
        }
        optimizer_class = optimizer_map.get(self.optimizer_name)
        if optimizer_class is None:
            supported = ", ".join(sorted(optimizer_map))
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer_name}'. Supported values: {supported}."
            )

        return optimizer_class(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
