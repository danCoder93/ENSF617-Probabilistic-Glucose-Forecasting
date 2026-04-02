from __future__ import annotations

"""
AI-assisted maintenance note:
These tests were added with AI assistance on April 1, 2026 to protect the
LightningModule responsibilities now carried by `FusedModel`.

They intentionally stay narrow:
- verify the fused model still produces horizon-aligned quantile forecasts
- verify `training_step(...)` uses the public quantile-loss path
- verify optimizer configuration stays attached to the model rather than
  leaking into a future training script
"""

import pytest

pytest.importorskip("pytorch_lightning")
torch = pytest.importorskip("torch")
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from models.fused_model import FusedModel
from config import Config, DataConfig, TCNConfig, TFTConfig, config_to_dict
from utils.tft_utils import DataTypes, FeatureSpec, InputTypes


def _build_config() -> Config:
    features = (
        FeatureSpec("age_years", InputTypes.STATIC, DataTypes.CONTINUOUS),
        FeatureSpec("hour_of_day", InputTypes.KNOWN, DataTypes.CONTINUOUS),
        FeatureSpec("carbs_g", InputTypes.OBSERVED, DataTypes.CONTINUOUS),
        FeatureSpec("glucose_mg_dl", InputTypes.TARGET, DataTypes.CONTINUOUS),
    )

    return Config(
        data=DataConfig(
            dataset_url=None,
            features=features,
            encoder_length=6,
            prediction_length=3,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
        tft=TFTConfig(
            features=features,
            hidden_size=8,
            n_head=2,
            dropout=0.0,
            attn_dropout=0.0,
            encoder_length=6,
            example_length=9,
        ),
        tcn=TCNConfig(
            num_inputs=2,
            num_channels=(4, 4),
            dilations=(1, 2),
            kernel_size=3,
            dropout=0.0,
            prediction_length=3,
            output_size=1,
        ),
    )


def _build_batch(config: Config, batch_size: int = 2) -> dict[str, Tensor]:
    encoder_length = config.data.encoder_length
    prediction_length = config.data.prediction_length

    return {
        "static_categorical": torch.empty(batch_size, 0, dtype=torch.long),
        "static_continuous": torch.randn(batch_size, 1),
        "encoder_continuous": torch.randn(batch_size, encoder_length, 3),
        "encoder_categorical": torch.empty(batch_size, encoder_length, 0, dtype=torch.long),
        "decoder_known_continuous": torch.randn(batch_size, prediction_length, 1),
        "decoder_known_categorical": torch.empty(
            batch_size,
            prediction_length,
            0,
            dtype=torch.long,
        ),
        "target": torch.randn(batch_size, prediction_length),
    }


def test_fused_model_forward_emits_quantile_forecasts() -> None:
    torch.manual_seed(0)
    config = _build_config()
    model = FusedModel(config)

    predictions = model(_build_batch(config))

    assert predictions.shape == (
        2,
        config.data.prediction_length,
        len(config.tft.quantiles),
    )
    assert torch.isfinite(predictions).all()


def test_fused_model_materializes_lazy_tft_parameters_during_init() -> None:
    model = FusedModel(_build_config())

    assert not any(
        isinstance(parameter, UninitializedParameter)
        for parameter in model.parameters()
    )


def test_fused_model_accepts_serialized_config_payload() -> None:
    config = _build_config()

    model = FusedModel(config_to_dict(config))

    assert model.config == config
    assert model.hparams["config"]["data"]["encoder_length"] == config.data.encoder_length


def test_fused_model_training_step_matches_quantile_loss() -> None:
    torch.manual_seed(0)
    config = _build_config()
    model = FusedModel(config)
    batch = _build_batch(config)

    predictions = model(batch)
    expected_loss = model.quantile_loss(predictions, batch["target"])

    actual_loss = model.training_step(batch, batch_idx=0)

    assert torch.allclose(actual_loss, expected_loss)


def test_fused_model_point_prediction_uses_median_quantile() -> None:
    model = FusedModel(_build_config())
    predictions = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        dtype=torch.float32,
    )

    point_forecast = model.point_prediction(predictions)

    assert torch.equal(point_forecast, torch.tensor([[2.0, 5.0]], dtype=torch.float32))


def test_fused_model_configure_optimizers_uses_model_hyperparameters() -> None:
    model = FusedModel(
        _build_config(),
        learning_rate=5e-4,
        weight_decay=1e-2,
        optimizer_name="adamw",
    )

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-2)
