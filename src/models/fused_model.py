from __future__ import annotations

"""
AI-assisted maintenance note:
This existing module was refined with AI assistance on April 1, 2026 under
direct user guidance. The updates align the post-TFT fusion GRN with the same
config-backed construction path used by the internal TFT GRNs so the codebase
is more consistent and maintainable, without introducing a new model concept.

The fused model still owns the fusion-specific dimensions, especially the size
of the concatenated `tft_output + aux_future` feature tensor, but shared GRN
defaults now come from the TFT config so the fusion head stays numerically and
architecturally consistent with the TFT branch it builds on.
"""

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
    - a TFT branch that refines those forecasts with known future inputs
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        tft_config = replace(
            config.tft,
            encoder_length=config.data.encoder_length,
            example_length=config.data.encoder_length + config.data.prediction_length,
            num_aux_future_features=3,
        )
        self.tft_config = tft_config

        self.num_known_cont = tft_config.temporal_known_continuous_inp_size
        self.num_observed_cont = tft_config.temporal_observed_continuous_inp_size
        self.num_target = max(1, tft_config.temporal_target_size)
        self.num_known_cat = len(tft_config.temporal_known_categorical_inp_lens)
        self.num_observed_cat = len(tft_config.temporal_observed_categorical_inp_lens)

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

        self.tft = TemporalFusionTransformer(tft_config)

        fused_feature_size = len(tft_config.quantiles) + tft_config.num_aux_future_features
        # Keep the fusion feature width explicit because it depends on this
        # model's branch composition, while the GRN's shared defaults come from
        # the bound TFT config so the post-fusion projector stays aligned with
        # the TFT branch's hidden size, dropout, and normalization behavior.
        self.grn = GRN.from_tft_config(
            tft_config,
            input_size=fused_feature_size,
            output_size=tft_config.hidden_size,
        )
        self.fcn = NNHead(tft_config.hidden_size, len(tft_config.quantiles))

    def _split_encoder_continuous(self, encoder_continuous: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        known_history = encoder_continuous[:, :, : self.num_known_cont]
        observed_end = self.num_known_cont + self.num_observed_cont
        observed_history = encoder_continuous[:, :, self.num_known_cont:observed_end]
        target_history = encoder_continuous[:, :, observed_end:]

        if target_history.shape[-1] == 0:
            observed_history = encoder_continuous[:, :, self.num_known_cont:-1]
            target_history = encoder_continuous[:, :, -1:]

        return known_history, observed_history, target_history

    def _split_encoder_categorical(self, encoder_categorical: Tensor) -> tuple[Tensor, Tensor]:
        known_history = encoder_categorical[:, :, : self.num_known_cat]
        observed_history = encoder_categorical[:, :, self.num_known_cat :]
        return known_history, observed_history

    def _optional_group(self, tensor: Tensor) -> Tensor | None:
        return None if tensor.shape[-1] == 0 else tensor

    def _build_tft_inputs(
        self, batch: dict[str, Tensor], aux_future: Tensor
    ) -> dict[str, Tensor | None]:
        encoder_continuous = batch["encoder_continuous"]
        encoder_categorical = batch["encoder_categorical"]

        known_history_cont, observed_history_cont, target_history = self._split_encoder_continuous(
            encoder_continuous
        )
        known_history_cat, observed_history_cat = self._split_encoder_categorical(
            encoder_categorical
        )

        aux_history = torch.zeros(
            aux_future.shape[0],
            self.config.data.encoder_length,
            aux_future.shape[-1],
            device=aux_future.device,
            dtype=aux_future.dtype,
        )

        known_continuous = torch.cat(
            [
                torch.cat([known_history_cont, aux_history], dim=-1),
                torch.cat([batch["decoder_known_continuous"], aux_future], dim=-1),
            ],
            dim=1,
        )
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
        _, observed_history_cont, target_history = self._split_encoder_continuous(
            batch["encoder_continuous"]
        )
        tcn_inputs = torch.cat([observed_history_cont, target_history], dim=-1)

        tcn3_forecast = self.tcn3(tcn_inputs)
        tcn5_forecast = self.tcn5(tcn_inputs)
        tcn7_forecast = self.tcn7(tcn_inputs)

        aux_future = torch.cat([tcn3_forecast, tcn5_forecast, tcn7_forecast], dim=-1)
        tft_inputs = self._build_tft_inputs(batch, aux_future)
        tft_output = self.tft(tft_inputs)

        fused_features = torch.cat([tft_output, aux_future], dim=-1)
        fused_features = self.grn(fused_features)
        return self.fcn(fused_features)
