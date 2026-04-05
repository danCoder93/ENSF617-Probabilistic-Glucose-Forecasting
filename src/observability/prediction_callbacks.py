from __future__ import annotations

# NOTE:
# This regenerated file only changes one behavioral detail relative to the
# version you uploaded:
# - PredictionSanityCallback now uses `debug_every_n_steps` from
#   ObservabilityConfig instead of the non-existent `log_every_n_steps`.
#
# Everything else is left intentionally aligned with your current local file.

# =============================================================================
# Prediction-level observability callbacks for probabilistic forecasting.
#
# AI-assisted maintenance note:
#     This module owns prediction-facing observability callbacks. In the current
#     repo design, this means callbacks that operate *after* the model has
#     already produced forecast tensors and we want to inspect those forecasts
#     from either of two perspectives:
#
#     1. Qualitative prediction inspection
#        - representative forecast plots in TensorBoard
#        - target / prediction / interval visualization
#        - human-friendly sanity checking of forecast shape and trend behavior
#
#     2. Quantitative prediction sanity checking
#        - quantile crossing rate
#        - interval width behavior
#        - non-finite prediction detection
#        - collapse / near-constant prediction detection
#        - prediction-vs-target scale comparison
#
# Why this file exists as its own module:
#     The repo already has separate files for:
#     - debug callbacks focused on batches, gradients, and activations
#     - parameter callbacks focused on parameter-state telemetry
#     - system callbacks focused on model graph / system-level instrumentation
#
#     Prediction-facing logic deserves its own home because it answers a
#     different question:
#         "Even if the training loop is numerically healthy, do the produced
#          forecasts themselves actually look sane?"
#
# Design constraints for this file:
#     - keep the implementation local and artifact-oriented
#     - avoid changing model architecture or dataset contract
#     - preserve existing qualitative prediction figures already used by the repo
#     - add prediction sanity diagnostics without removing older functionality
#     - stay reasonably robust while the training-step output contract evolves
#     - use Lightning's built-in callback and logging surfaces rather than
#       introducing external observability services
#
# Important maintenance rule:
#     When patching this file, do not treat new prediction callbacks as
#     replacements for existing ones unless that is a deliberate repo decision.
#     In particular, `PredictionFigureCallback` remains important because it is
#     the qualitative forecast visualization surface used in TensorBoard, while
#     `PredictionSanityCallback` adds scalar diagnostics that complement those
#     figures rather than replacing them.
#
# AI generation disclaimer:
#     This file was generated and refined with AI assistance, then adapted to
#     fit the repository's observability structure. The code and comments aim to
#     be explicit and maintainable, but maintainers should still review the
#     extraction logic and metric semantics against the project's canonical
#     prediction contract.
# =============================================================================

from typing import Any, Mapping

import torch
from pytorch_lightning.callbacks import Callback

from config import ObservabilityConfig
from observability.logging_utils import _tensorboard_experiments
from observability.tensors import _as_metadata_lists
from observability.utils import _has_module


class PredictionFigureCallback(Callback):
    """Log a few qualitative forecast examples directly into TensorBoard.

    Purpose:
        Push a small number of human-readable forecast plots into TensorBoard so
        that a run can be inspected visually without opening a separate notebook
        or report pipeline.

    What this callback tries to show:
        - whether the predicted trajectory broadly follows the target
        - whether the prediction interval is visibly too wide or too narrow
        - whether forecasts appear shifted, flat, unstable, or otherwise odd
        - whether qualitative behavior differs between validation and test

    Why this callback still matters even after adding prediction sanity metrics:
        Scalar metrics are excellent for detecting anomalies systematically, but
        they do not replace human visual judgment. A model can have acceptable
        scalar diagnostics while still producing forecast shapes that look
        implausible to a human reviewing representative examples.

    Scope boundary:
        This callback is intentionally lightweight and sampled. It is *not*
        designed to produce exhaustive visual reports for every batch.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        """Initialize the callback and per-stage epoch throttling state.

        Parameters:
            config:
                Observability configuration object that controls whether figures
                are enabled, how frequently they should be logged, and how many
                forecast examples should appear in one figure.
        """
        # Store config so all later decisions draw from the repo's centralized
        # observability policy rather than hard-coded callback behavior.
        self.config = config

        # Keep track of epochs that have already emitted validation figures.
        # This prevents multiple validation batches within the same eligible
        # epoch from each creating redundant plots.
        self._logged_validation_epochs: set[int] = set()

        # Keep track of epochs that have already emitted test figures for the
        # same reason as validation: one sampled figure set per eligible epoch
        # is enough for qualitative inspection.
        self._logged_test_epochs: set[int] = set()

    def _should_log(self, trainer: Any, stage: str) -> bool:
        """Return whether one prediction figure set should be emitted now.

        Parameters:
            trainer:
                Lightning trainer object. We use it mainly for current epoch.

            stage:
                Expected to be either ``"val"`` or ``"test"``. This determines
                which per-stage epoch tracking set should be consulted.

        Returns:
            ``True`` if the current batch is the first eligible batch for a
            figure log in this stage/epoch, otherwise ``False``.
        """
        # Respect the main feature gate first. If qualitative prediction figures
        # are disabled in config, nothing else in this callback should run.
        if not self.config.enable_prediction_figures:
            return False

        # Matplotlib is required for the figure path. We guard availability here
        # so environments without plotting dependencies degrade gracefully.
        if not _has_module("matplotlib"):
            return False

        # Select the correct per-stage epoch set so validation and test logging
        # are throttled independently.
        target_set = (
            self._logged_validation_epochs
            if stage == "val"
            else self._logged_test_epochs
        )

        # Only emit one sampled figure set per stage per eligible epoch. This
        # keeps the callback lightweight and avoids spamming TensorBoard with a
        # figure for every batch.
        if trainer.current_epoch in target_set:
            return False

        # Respect configured epoch frequency. For example, a value of 5 means
        # log figures on epochs 5, 10, 15, ... when using one-based wording.
        if (trainer.current_epoch + 1) % self.config.figure_every_n_epochs != 0:
            return False

        # Mark the current epoch as already logged for this stage so future
        # batches in the same epoch do not duplicate the figure set.
        target_set.add(trainer.current_epoch)
        return True

    def _log_prediction_figure(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Mapping[str, Any],
        stage: str,
    ) -> None:
        """Render and publish one small qualitative forecast figure set.

        Parameters:
            trainer:
                Lightning trainer used to access the global step and attached
                TensorBoard experiment handles.

            pl_module:
                Active Lightning module. The callback runs the live model on the
                current batch so the figure reflects the model's actual current
                state at this point in training/evaluation.

            batch:
                Current batch mapping. This callback assumes the repo's current
                batch contract includes ``metadata`` and that the module can
                derive the target tensor via ``pl_module._target_tensor(batch)``.

            stage:
                Either ``"val"`` or ``"test"``.
        """
        # Exit immediately unless this batch is the single eligible batch for a
        # figure log in the current stage/epoch.
        if not self._should_log(trainer, stage):
            return

        # Import matplotlib lazily so environments that never log prediction
        # figures do not pay the import cost during module import.
        import matplotlib.pyplot as plt

        # Use torch.no_grad() because this callback is purely observational. We
        # do not want figure generation to participate in autograd or increase
        # memory pressure by storing graph history.
        with torch.no_grad():
            # Run the live model on the current batch. This is intentional: the
            # point of the figure is to show what the model currently predicts,
            # not what it predicted earlier in some cached intermediate.
            predictions = pl_module(batch).detach().cpu()

            # Obtain the canonical target tensor using the module's own helper so
            # the plotting path stays aligned with the model's expected target
            # extraction logic.
            target = pl_module._target_tensor(batch).detach().cpu()

            # Convert batch metadata into simple per-example lists so subject and
            # decoder-start information can be added to figure titles.
            metadata = _as_metadata_lists(batch["metadata"], int(predictions.shape[0]))

        # Cap the number of plotted examples so the figure remains readable and
        # plotting overhead remains modest for routine local runs.
        max_plots = min(self.config.max_prediction_plots, int(predictions.shape[0]))

        # Create one subplot per selected example. The height is scaled so each
        # subplot has enough vertical space to remain readable.
        figure, axes = plt.subplots(
            max_plots,
            1,
            figsize=(10, max(4, 3 * max_plots)),
            squeeze=False,
        )

        # Find the quantile closest to 0.5 and use it as the representative
        # point forecast line in the plot.
        median_index = min(
            range(len(pl_module.quantiles)),
            key=lambda index: abs(float(pl_module.quantiles[index]) - 0.5),
        )

        for plot_index in range(max_plots):
            # Each row contains a single axis because we created a column of
            # subplots. Access the first element explicitly for clarity.
            axis = axes[plot_index][0]

            # Use a simple integer horizon on the x-axis so the plot answers the
            # forecasting question "what happens over decoder step 0..H-1?"
            horizon = list(range(int(predictions.shape[1])))

            # Plot the ground-truth target trajectory as the reference series.
            axis.plot(
                horizon,
                target[plot_index].tolist(),
                label="target",
                color="black",
                linewidth=2,
            )

            # Plot the median or nearest-to-median quantile as the main forecast
            # line. This gives a readable single-line forecast summary.
            axis.plot(
                horizon,
                predictions[plot_index, :, median_index].tolist(),
                label="median prediction",
                color="tab:blue",
            )

            # If the prediction tensor contains multiple quantiles, shade the
            # outermost interval as a quick visual uncertainty band.
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

            # Pull a small amount of metadata into the title so plotted examples
            # can be linked back to their source subject/window.
            subject_id = str(metadata.get("subject_id", ["unknown"])[plot_index])
            decoder_start = str(metadata.get("decoder_start", [""])[plot_index])
            axis.set_title(f"{stage} subject={subject_id} decoder_start={decoder_start}")
            axis.set_xlabel("Horizon Step")
            axis.set_ylabel("Glucose")
            axis.legend(loc="best")

        # Tight layout reduces overlap between titles, labels, and legends.
        figure.tight_layout()

        # Publish the figure to every TensorBoard experiment attached to the
        # trainer. The helper keeps this callback decoupled from logger internals.
        for experiment in _tensorboard_experiments(trainer):
            add_figure = getattr(experiment, "add_figure", None)
            if callable(add_figure):
                add_figure(
                    f"predictions/{stage}",
                    figure,
                    global_step=trainer.global_step,
                )

        # Always close the figure so repeated logging does not leak figure
        # objects or consume unnecessary memory.
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
        """Emit at most one sampled validation figure set per eligible epoch."""
        # These hook parameters are part of Lightning's callback signature but
        # are not needed by this callback's current logic.
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
        """Emit at most one sampled test figure set per eligible epoch."""
        del outputs, batch_idx, dataloader_idx
        self._log_prediction_figure(trainer, pl_module, batch, "test")


class PredictionSanityCallback(Callback):
    """Log compact numerical sanity diagnostics for forecast tensors.

    Purpose:
        Complement qualitative prediction figures with scalar diagnostics that
        help answer whether the forecast tensor is numerically and semantically
        healthy.

    What this callback is trying to detect:
        - quantile crossing, which indicates invalid quantile ordering
        - near-zero interval widths, which can indicate collapsed uncertainty
        - very low prediction variance, which can indicate flat forecasts
        - non-finite values, which indicate a serious numerical failure
        - strong mismatch between prediction scale and target scale

    Design philosophy:
        This callback should be lightweight enough for local debug runs and
        resilient enough to survive minor output-contract changes while the repo
        is still evolving. It therefore uses best-effort extraction rather than
        requiring one rigid output dict schema.
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        text_logger: Any | None = None,
    ) -> None:
        """Initialize the prediction sanity callback.

        Parameters:
            config:
                Repository observability config. We reuse it mainly for cadence
                and to stay consistent with the repo's central observability
                policy surface.

            text_logger:
                Optional plain-text logger. If provided, the callback can emit
                human-readable warning messages in addition to Lightning scalar
                logs.
        """
        self.config = config
        self.text_logger = text_logger

        # Keep logging cadence modest. This callback is useful, but it should
        # not spam every step unless the repo later decides it should.
        self._log_every_n_steps = max(1, int(getattr(config, "debug_every_n_steps", 10)))

        # Thresholds below are intentionally conservative. They are not claims of
        # universal correctness; they are practical heuristics for surfacing
        # suspicious behavior during debugging.
        self._constant_std_threshold = 1e-6
        self._near_zero_width_threshold = 1e-6
        self._crossing_tolerance = 0.0

    def _should_log_now(self, trainer: Any) -> bool:
        """Return whether this global step should emit sanity metrics."""
        global_step = int(getattr(trainer, "global_step", 0))
        return global_step % self._log_every_n_steps == 0

    def _extract_predictions(self, outputs: Any, batch: Mapping[str, Any], pl_module: Any) -> torch.Tensor | None:
        """Best-effort extraction of the forecast tensor.

        Extraction priority:
            1. common keys in the step output structure
            2. common cached attributes on the module
            3. direct model forward pass on the current batch as fallback

        Why best-effort extraction is used here:
            During active repo development, the exact structure returned from
            training/validation steps may evolve. A sanity callback should avoid
            becoming brittle if the naming changes slightly.
        """
        # First handle mapping-like outputs because many Lightning step methods
        # return dictionaries with loss plus extra tensors.
        if isinstance(outputs, Mapping):
            for key in (
                "predictions",
                "prediction",
                "y_hat",
                "forecast",
                "forecasts",
                "quantiles",
                "preds",
            ):
                value = outputs.get(key)
                if isinstance(value, torch.Tensor):
                    return value.detach()

        # Some repos cache the latest predictions on the module for later hooks.
        for attr_name in (
            "latest_predictions",
            "_latest_predictions",
            "predictions",
            "_predictions",
        ):
            value = getattr(pl_module, attr_name, None)
            if isinstance(value, torch.Tensor):
                return value.detach()

        # As a final fallback, run the model directly on the batch. This is less
        # ideal than extracting from outputs because it repeats the forward pass,
        # but it is still useful for observability when contracts are in flux.
        try:
            with torch.no_grad():
                value = pl_module(batch)
            if isinstance(value, torch.Tensor):
                return value.detach()
        except Exception:
            return None

        return None

    def _extract_target(self, outputs: Any, batch: Mapping[str, Any], pl_module: Any) -> torch.Tensor | None:
        """Best-effort extraction of the target tensor."""
        if isinstance(outputs, Mapping):
            for key in ("target", "targets", "y", "y_true"):
                value = outputs.get(key)
                if isinstance(value, torch.Tensor):
                    return value.detach()

        # Prefer the module's canonical target extraction helper when available.
        target_helper = getattr(pl_module, "_target_tensor", None)
        if callable(target_helper):
            try:
                value = target_helper(batch)
                if isinstance(value, torch.Tensor):
                    return value.detach()
            except Exception:
                pass

        # Fallback to common batch keys.
        for key in ("target", "targets", "y", "decoder_target"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value.detach()

        return None

    def _as_cpu_float_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """Detach, move to CPU, and cast to float for metric computation."""
        if tensor is None:
            return None
        return tensor.detach().float().cpu()

    def _log_scalar(self, pl_module: Any, name: str, value: float, *, on_step: bool = True) -> None:
        """Safely log one scalar through Lightning's built-in logging surface."""
        if not torch.isfinite(torch.tensor(value)):
            return
        pl_module.log(name, value, on_step=on_step, on_epoch=False, prog_bar=False, logger=True)

    def _log_warning(self, message: str) -> None:
        """Emit an optional text warning through the injected text logger."""
        if self.text_logger is not None:
            self.text_logger.warning(message)

    def _compute_quantile_crossing_rate(self, predictions: torch.Tensor) -> float | None:
        """Compute the fraction of adjacent quantile pairs that cross.

        Expected prediction shape:
            [batch, horizon, quantile]

        Interpretation:
            A value greater than zero means at least some lower quantiles exceed
            higher quantiles, which violates quantile ordering.
        """
        if predictions.ndim < 3 or predictions.shape[-1] < 2:
            return None

        adjacent_diffs = predictions[..., 1:] - predictions[..., :-1]
        crossings = adjacent_diffs < self._crossing_tolerance
        return float(crossings.float().mean().item())

    def _compute_interval_width_stats(self, predictions: torch.Tensor) -> tuple[float | None, float | None, float | None]:
        """Return mean width, width std, and near-zero width rate."""
        if predictions.ndim < 3 or predictions.shape[-1] < 2:
            return None, None, None

        widths = predictions[..., -1] - predictions[..., 0]
        width_mean = float(widths.mean().item())
        width_std = float(widths.std(unbiased=False).item())
        near_zero_rate = float((widths.abs() <= self._near_zero_width_threshold).float().mean().item())
        return width_mean, width_std, near_zero_rate

    def _compute_variance_stats(self, predictions: torch.Tensor) -> tuple[float, float]:
        """Return overall variance and horizon-mean variance.

        Overall variance:
            Variance over all prediction elements.

        Horizon-mean variance:
            Variance computed across the batch for each horizon/quantile position,
            then averaged. This is often more interpretable for spotting whether
            forecasts are becoming too similar across examples.
        """
        overall_variance = float(predictions.var(unbiased=False).item())

        if predictions.ndim >= 2:
            batch_axis = 0
            horizon_variance = predictions.var(dim=batch_axis, unbiased=False).mean()
            horizon_mean_variance = float(horizon_variance.item())
        else:
            horizon_mean_variance = overall_variance

        return overall_variance, horizon_mean_variance

    def _compute_constant_rate(self, predictions: torch.Tensor) -> float:
        """Return the fraction of examples that look nearly constant.

        Heuristic meaning:
            For each example, flatten all non-batch dimensions and compute the
            standard deviation. If that std is extremely small, we mark the
            forecast as near-constant.
        """
        if predictions.ndim == 0:
            return 1.0

        flattened = predictions.reshape(predictions.shape[0], -1)
        per_example_std = flattened.std(dim=1, unbiased=False)
        constant_mask = per_example_std <= self._constant_std_threshold
        return float(constant_mask.float().mean().item())

    def _compute_prediction_target_scale_ratio(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor | None,
    ) -> float | None:
        """Compare prediction magnitude scale against target magnitude scale.

        Interpretation:
            A ratio near 1.0 means prediction absolute magnitude and target
            absolute magnitude are in the same ballpark. A very small or very
            large value may indicate drift, collapse, or target mismatch.
        """
        if target is None or target.numel() == 0:
            return None

        prediction_scale = float(predictions.abs().mean().item())
        target_scale = float(target.abs().mean().item())

        # Avoid division by zero while still surfacing mismatch meaningfully.
        if target_scale <= 1e-12:
            return None

        return prediction_scale / target_scale

    def _log_prediction_metrics(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Mapping[str, Any],
        stage: str,
    ) -> None:
        """Compute and emit prediction sanity metrics for one hook invocation."""
        if not self._should_log_now(trainer):
            return

        predictions = self._as_cpu_float_tensor(self._extract_predictions(outputs, batch, pl_module))
        target = self._as_cpu_float_tensor(self._extract_target(outputs, batch, pl_module))

        # If predictions cannot be extracted, emit one warning and stop quietly.
        # We avoid raising because observability should not crash training.
        if predictions is None:
            self._log_warning(
                f"PredictionSanityCallback could not extract predictions during stage={stage} "
                f"at global_step={getattr(trainer, 'global_step', 'unknown')}."
            )
            return

        prefix = f"prediction_sanity/{stage}"

        # Basic non-finite rate is the highest-value first-line anomaly signal.
        non_finite_rate = float((~torch.isfinite(predictions)).float().mean().item())
        self._log_scalar(pl_module, f"{prefix}/non_finite_rate", non_finite_rate)

        # Replace non-finite values with zero for downstream metrics so one NaN
        # does not poison every later summary. We still keep the non-finite rate
        # metric above as the real anomaly indicator.
        clean_predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

        crossing_rate = self._compute_quantile_crossing_rate(clean_predictions)
        if crossing_rate is not None:
            self._log_scalar(pl_module, f"{prefix}/quantile_crossing_rate", crossing_rate)

        width_mean, width_std, near_zero_width_rate = self._compute_interval_width_stats(clean_predictions)
        if width_mean is not None:
            self._log_scalar(pl_module, f"{prefix}/interval_width_mean", width_mean)
        if width_std is not None:
            self._log_scalar(pl_module, f"{prefix}/interval_width_std", width_std)
        if near_zero_width_rate is not None:
            self._log_scalar(pl_module, f"{prefix}/near_zero_interval_rate", near_zero_width_rate)

        overall_variance, horizon_mean_variance = self._compute_variance_stats(clean_predictions)
        self._log_scalar(pl_module, f"{prefix}/variance_overall", overall_variance)
        self._log_scalar(pl_module, f"{prefix}/variance_horizon_mean", horizon_mean_variance)

        constant_rate = self._compute_constant_rate(clean_predictions)
        self._log_scalar(pl_module, f"{prefix}/near_constant_prediction_rate", constant_rate)

        scale_ratio = self._compute_prediction_target_scale_ratio(clean_predictions, target)
        if scale_ratio is not None:
            self._log_scalar(pl_module, f"{prefix}/prediction_target_scale_ratio", scale_ratio)

        # Emit lightweight warnings only when the metric suggests something
        # meaningfully suspicious. These warnings are intended to aid log review,
        # not to act as hard assertions.
        if non_finite_rate > 0.0:
            self._log_warning(
                f"PredictionSanityCallback detected non-finite predictions during stage={stage} "
                f"at global_step={getattr(trainer, 'global_step', 'unknown')}: "
                f"non_finite_rate={non_finite_rate:.6f}."
            )

        if crossing_rate is not None and crossing_rate > 0.0:
            self._log_warning(
                f"PredictionSanityCallback detected quantile crossing during stage={stage} "
                f"at global_step={getattr(trainer, 'global_step', 'unknown')}: "
                f"crossing_rate={crossing_rate:.6f}."
            )

        if constant_rate >= 0.95:
            self._log_warning(
                f"PredictionSanityCallback detected near-collapsed forecasts during stage={stage} "
                f"at global_step={getattr(trainer, 'global_step', 'unknown')}: "
                f"near_constant_prediction_rate={constant_rate:.6f}."
            )

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> None:
        """Training hook for compact prediction sanity metrics."""
        del batch_idx
        self._log_prediction_metrics(trainer, pl_module, outputs, batch, "train")

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Validation hook for compact prediction sanity metrics."""
        del batch_idx, dataloader_idx
        self._log_prediction_metrics(trainer, pl_module, outputs, batch, "val")

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test hook for compact prediction sanity metrics."""
        del batch_idx, dataloader_idx
        self._log_prediction_metrics(trainer, pl_module, outputs, batch, "test")
