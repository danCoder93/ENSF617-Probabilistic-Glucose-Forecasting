from __future__ import annotations

# AI-assisted maintenance note:
# This module contains logger-facing helpers that are shared across the package.
#
# Why keep these separate from runtime setup:
# - runtime setup answers "which logger objects exist for this run?"
# - these helpers answer "how do other parts of the code talk to those logger
#   objects once they exist?"
#
# That distinction keeps callback code focused on observability policy rather
# than repeatedly re-implementing Lightning logger shape normalization.

import json
from pathlib import Path
from typing import Any, Mapping


def _active_loggers(trainer: Any) -> list[Any]:
    """
    Return the trainer's active logger objects as a normalized list.

    Context:
    Lightning can expose either `trainer.logger` or `trainer.loggers`, so the
    rest of the package should not have to care which shape is active.
    """
    # Lightning may expose a single logger as `trainer.logger` or multiple
    # loggers as `trainer.loggers`. This helper normalizes both cases so the
    # rest of the package can treat "active loggers" as a simple list.
    loggers = getattr(trainer, "loggers", None)
    if loggers is not None:
        return list(loggers)
    logger = getattr(trainer, "logger", None)
    return [] if logger is None else [logger]


def _tensorboard_experiments(trainer: Any) -> list[Any]:
    """
    Return TensorBoard-compatible experiment backends from the active loggers.

    Context:
    only some logger backends expose TensorBoard methods like `add_text`,
    `add_histogram`, or `add_figure`.
    """
    # Not every active logger is a TensorBoard logger. This helper filters the
    # active logger set down to logger backends whose `.experiment` object
    # supports TensorBoard-style methods like `add_scalar`, `add_text`, and
    # `add_histogram`.
    experiments: list[Any] = []
    for logger in _active_loggers(trainer):
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_scalar"):
            experiments.append(experiment)
    return experiments


def log_metrics_to_loggers(
    trainer: Any,
    metrics: Mapping[str, float],
    step: int,
) -> None:
    """
    Push a precomputed metric mapping to every compatible active logger.

    Context:
    custom callbacks often compute metrics outside `LightningModule.self.log`,
    but they still need one shared path to the configured loggers.
    """
    # This helper is used by custom callbacks that compute metrics outside the
    # model's `self.log(...)` path. It pushes a ready-made metric dictionary to
    # every active logger that exposes Lightning's `log_metrics(...)` API.
    for logger in _active_loggers(trainer):
        log_metrics = getattr(logger, "log_metrics", None)
        if callable(log_metrics):
            log_metrics(dict(metrics), step=step)


def _log_text_to_loggers(trainer: Any, tag: str, text: str) -> None:
    """
    Publish a text payload to TensorBoard-compatible logger backends.

    Context:
    this is used for structured debug artifacts such as batch audits that are
    more readable as text than as scalar metrics.
    """
    # TensorBoard has a useful text surface for structured debug payloads such
    # as batch audits. This helper publishes text only to compatible
    # TensorBoard-backed experiments and skips other logger types quietly.
    for experiment in _tensorboard_experiments(trainer):
        add_text = getattr(experiment, "add_text", None)
        if callable(add_text):
            add_text(tag, text, global_step=getattr(trainer, "global_step", 0))


def _flatten_for_hparams(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, str | int | float | bool]:
    """
    Flatten nested config-like payloads into scalar logger-friendly entries.

    Context:
    hyperparameter backends usually expect a flat mapping rather than nested
    dictionaries or Python-specific objects.
    """
    # Hyperparameter logging is much more readable when nested config objects
    # are flattened into stable slash-delimited keys such as
    # `config/data/encoder_length`. This also avoids surprising behavior in
    # logger backends that expect scalar-like values rather than nested dicts.
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in payload.items():
        joined_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_for_hparams(value, prefix=joined_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[joined_key] = "None" if value is None else value
        elif isinstance(value, Path):
            flattened[joined_key] = str(value)
        else:
            flattened[joined_key] = json.dumps(value, sort_keys=True)
    return flattened


def log_hyperparameters(
    trainer: Any,
    payload: Mapping[str, Any],
) -> None:
    """
    Log a nested hyperparameter payload to the active logger set.

    Context:
    this helper keeps hyperparameter logging consistent across TensorBoard and
    any other configured Lightning logger backends.
    """
    # Flattening here keeps the logged hparams readable in TensorBoard and
    # compatible with logger backends that expect simple scalar/string values.
    flattened = _flatten_for_hparams(payload)
    for logger in _active_loggers(trainer):
        log_hparams = getattr(logger, "log_hyperparams", None)
        if callable(log_hparams):
            log_hparams(flattened)
