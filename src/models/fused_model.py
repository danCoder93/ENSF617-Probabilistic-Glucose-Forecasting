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

from pytorch_lightning import LightningModule

import torch
from torch import Tensor

from tcn import TCN
from tft import TemporalFusionTransformer
from grn import GRN
from nn_head import NNHead

from utils.config import Config


class FusedModel(LightningModule):
    """
    Hybrid glucose forecaster with:
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

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

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

    def _split_encoder_continuous(self, encoder_continuous: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # `encoder_continuous` follows the shared data contract:
        # [known continuous | observed continuous | target history]
        #
        # This helper recovers the semantic groups so each branch only receives
        # the information it is supposed to consume.
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
        # Historical categorical features are ordered as:
        # [known categorical | observed categorical]
        known_history = encoder_categorical[:, :, : self.num_known_cat]
        observed_history = encoder_categorical[:, :, self.num_known_cat :]
        return known_history, observed_history

    def _optional_group(self, tensor: Tensor) -> Tensor | None:
        # TFT expects missing feature groups to be passed as `None`, not as
        # empty tensors with zero trailing width.
        return None if tensor.shape[-1] == 0 else tensor

    def _build_tft_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
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

        # -----------------------------
        # Step 5: Predict outputs
        # -----------------------------
        # The GRN acts as the nonlinear gated mixer over branch features, and
        # the final MLP head converts that fused hidden state into quantiles.
        fused_features = self.grn(fused_features)
        return self.fcn(fused_features)
