from __future__ import annotations

# AI-assisted maintenance note:
# This callback focuses on qualitative forecast examples rather than on raw
# scalar telemetry or parameter-state inspection.
#
# Why keep it separate:
# - it depends on matplotlib and figure construction, unlike the lighter
#   scalar callbacks
# - its logic is centered on prediction batches and metadata formatting
# - a dedicated file keeps the heavier plotting path isolated from the rest of
#   the callback assembly code

from typing import Any, Mapping

import torch
from pytorch_lightning.callbacks import Callback

from config import ObservabilityConfig
from observability.logging_utils import _tensorboard_experiments
from observability.tensors import _as_metadata_lists
from observability.utils import _has_module


class PredictionFigureCallback(Callback):
    """
    Log a few qualitative forecast examples directly into TensorBoard.

    Purpose:
    push qualitative forecast examples directly into TensorBoard.

    Context:
    the callback renders small matplotlib figures comparing targets, median
    predictions, and prediction intervals for representative validation/test
    examples. It is intentionally scoped to a handful of examples per epoch so
    it acts as a monitoring aid rather than a full report generator.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        self.config = config
        self._logged_validation_epochs: set[int] = set()
        self._logged_test_epochs: set[int] = set()

    def _should_log(self, trainer: Any, stage: str) -> bool:
        # We only log one small figure set per stage per eligible epoch. The
        # goal is qualitative inspection, not exhaustive visualization of every
        # batch.
        if not self.config.enable_prediction_figures:
            return False
        if not _has_module("matplotlib"):
            return False
        target_set = (
            self._logged_validation_epochs if stage == "val" else self._logged_test_epochs
        )
        if trainer.current_epoch in target_set:
            return False
        if (trainer.current_epoch + 1) % self.config.figure_every_n_epochs != 0:
            return False
        target_set.add(trainer.current_epoch)
        return True

    def _log_prediction_figure(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Mapping[str, Any],
        stage: str,
    ) -> None:
        # This uses the current model directly on the observed batch so the
        # figure reflects the live state of the run at that point in training.
        if not self._should_log(trainer, stage):
            return

        import matplotlib.pyplot as plt

        with torch.no_grad():
            predictions = pl_module(batch).detach().cpu()
        target = pl_module._target_tensor(batch).detach().cpu()
        metadata = _as_metadata_lists(batch["metadata"], int(predictions.shape[0]))
        # We intentionally run the live model on the current batch rather than
        # reusing some cached predictions because the point of this callback is
        # to show what the model currently believes at this stage of training.

        max_plots = min(self.config.max_prediction_plots, int(predictions.shape[0]))
        figure, axes = plt.subplots(
            max_plots,
            1,
            figsize=(10, max(4, 3 * max_plots)),
            squeeze=False,
        )
        median_index = min(
            range(len(pl_module.quantiles)),
            key=lambda index: abs(float(pl_module.quantiles[index]) - 0.5),
        )

        for plot_index in range(max_plots):
            axis = axes[plot_index][0]
            horizon = list(range(int(predictions.shape[1])))
            # Each subplot shows one forecast window:
            # - black line: ground-truth target trajectory
            # - blue line: median / representative point forecast
            # - shaded band: outer prediction interval if multiple quantiles
            #   exist
            #
            # This gives TensorBoard a quick qualitative "does this forecast
            # look reasonable?" surface alongside the scalar metrics.
            axis.plot(
                horizon,
                target[plot_index].tolist(),
                label="target",
                color="black",
                linewidth=2,
            )
            axis.plot(
                horizon,
                predictions[plot_index, :, median_index].tolist(),
                label="median prediction",
                color="tab:blue",
            )
            if predictions.shape[-1] >= 2:
                lower = predictions[plot_index, :, 0].tolist()
                upper = predictions[plot_index, :, -1].tolist()
                axis.fill_between(
                    horizon,
                    lower,
                    upper,
                    color="tab:blue",
                    alpha=0.2,
                    label="prediction interval",
                )
            subject_id = str(metadata.get("subject_id", ["unknown"])[plot_index])
            decoder_start = str(metadata.get("decoder_start", [""])[plot_index])
            axis.set_title(f"{stage} subject={subject_id} decoder_start={decoder_start}")
            axis.set_xlabel("Horizon Step")
            axis.set_ylabel("Glucose")
            axis.legend(loc="best")

        figure.tight_layout()
        for experiment in _tensorboard_experiments(trainer):
            add_figure = getattr(experiment, "add_figure", None)
            if callable(add_figure):
                add_figure(
                    f"predictions/{stage}",
                    figure,
                    global_step=trainer.global_step,
                )
        plt.close(figure)

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch_idx, dataloader_idx
        self._log_prediction_figure(trainer, pl_module, batch, "val")

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch_idx, dataloader_idx
        self._log_prediction_figure(trainer, pl_module, batch, "test")
