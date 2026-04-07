from __future__ import annotations

# AI-assisted maintenance note:
# These helpers sit between raw CLI values and the richer reusable workflow
# functions.
#
# Why keep them together:
# - both the CLI layer and training workflow need small normalization helpers
# - most are deliberately tiny and mechanical, so keeping them in one file
#   avoids scattering one-off parsing utilities across the package
# - they are implementation detail helpers, not part of the data/model domain

import argparse
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import platform
from typing import TYPE_CHECKING, Any, Sequence

from config import TrainConfig
from environment import (
    RuntimeDiagnostic,
    apply_runtime_environment_overrides,
    format_runtime_diagnostics,
)

if TYPE_CHECKING:
    from train import CheckpointSelection, FitArtifacts


def _json_ready(value: Any) -> Any:
    """
    Normalize workflow metadata into JSON-friendly values.

    Purpose:
    convert the mixed Python objects used by the workflow layer into plain
    JSON-serializable structures.

    Context:
    run summaries in this repository include paths, dataclasses, tuples, and
    nested mappings. This helper centralizes that normalization so every
    summary-writing path produces the same readable JSON shape.
    """
    # Run summaries are persisted as JSON, so this helper normalizes a mix of
    # Paths, dataclasses, lists, and plain mappings into JSON-friendly values.
    #
    # The goal is not to serialize every arbitrary Python object; it is simply
    # to make this workflow package's own artifact metadata readable and
    # portable.
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if is_dataclass(value):
        return {
            field_info.name: _json_ready(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    return value


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    """
    Parse a comma-separated integer CLI value into a tuple.

    Purpose:
    keep list-shaped CLI inputs shell-friendly without giving up typed
    downstream config values.

    Context:
    several top-level flags expose compact comma-separated syntax because that
    is easier to type and read than repeating the flag or embedding JSON on
    the command line.
    """
    # Several CLI flags accept comma-separated lists because that is friendlier
    # to shell usage than repeating a flag or forcing JSON syntax.
    #
    # Example:
    # `--tcn-channels 64,64,128`
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    """
    Parse a comma-separated float CLI value into a tuple.

    Purpose:
    mirror `_parse_csv_ints(...)` for float-valued settings such as quantiles.

    Context:
    keeping the integer and float parsers parallel makes the CLI conversion
    layer easier to reason about and avoids duplicating ad-hoc parsing logic
    at each call site.
    """
    # Mirrors `_parse_csv_ints(...)`, but for values like quantiles.
    # Example:
    # `--quantiles 0.1,0.5,0.9`
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_devices(value: str) -> str | int | list[int]:
    """
    Translate the CLI `--devices` value into a Lightning-compatible shape.

    Purpose:
    preserve the simple CLI surface while still returning the richer device
    value types Lightning accepts.

    Context:
    the command line can only pass strings directly, but the training config
    intentionally stores the semantically correct type so later layers do not
    have to re-interpret device syntax again.
    """
    # Lightning accepts several device formats:
    # - `"auto"` for automatic placement
    # - a single int like `1` for one device / one process
    # - a list like `[0, 1]` for explicit device indices
    #
    # This helper lets the CLI stay simple while still mapping cleanly onto the
    # richer Trainer API.
    cleaned = value.strip()
    if cleaned == "auto":
        return "auto"
    if "," in cleaned:
        return [int(part.strip()) for part in cleaned.split(",") if part.strip()]
    try:
        return int(cleaned)
    except ValueError:
        return cleaned


def _parse_limit(value: str) -> int | float:
    """
    Parse a Lightning loop-limit CLI value while preserving int/float meaning.

    Purpose:
    keep the important semantic distinction between "fixed number of batches"
    and "fraction of the loader" intact from the CLI boundary onward.

    Context:
    Lightning treats integer and floating loop limits differently, so a plain
    string-to-float conversion would lose useful intent for debugging runs.
    """
    # Lightning interprets integers and floats differently for loop limits:
    # - `10` means "run exactly 10 batches"
    # - `0.25` means "run 25% of the loader"
    #
    # Parsing here preserves that distinction from the command line.
    cleaned = value.strip()
    if any(character in cleaned for character in (".", "e", "E")):
        return float(cleaned)
    return int(cleaned)


def _normalize_optional_string(value: str | None) -> str | None:
    """
    Interpret common "empty" CLI strings as `None`.

    Purpose:
    make top-level flags behave closer to user intent when a caller types
    values like `none`, `null`, or an empty string.

    Context:
    command-line parsing surfaces optional values as strings, but the rest of
    the workflow code wants a real optional value rather than many string
    spellings of "unset".
    """
    # The CLI passes many values through as strings. This helper makes flags
    # like `--dataset-url none` or `--fit-ckpt-path null` behave the way users
    # usually mean them: "treat this as no value".
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "null"}:
        return None
    return cleaned


def _apply_early_apple_silicon_environment_defaults(
    *,
    requested_device_profile: str,
    train_config: TrainConfig,
) -> None:
    """
    Apply Apple Silicon runtime defaults before Torch import side effects.

    Purpose:
    export the small subset of MPS-related defaults that must exist before
    deeper Torch runtime initialization happens.

    Context:
    most device-profile logic can wait until full runtime detection, but some
    Apple Silicon environment variables only take effect if they are present
    very early in process startup.
    """
    # MPS allocator/fallback environment variables need to be present before
    # torch is imported. The full profile resolver runs after runtime detection,
    # so Apple Silicon needs one early pass here when the profile is explicit
    # or when `auto` is likely to resolve to the Apple path.
    if requested_device_profile not in {"auto", "apple-silicon"}:
        return
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "arm64e"}:
        return

    apply_runtime_environment_overrides(
        train_config=replace(
            train_config,
            mps_high_watermark_ratio=(
                1.3
                if train_config.mps_high_watermark_ratio is None
                else train_config.mps_high_watermark_ratio
            ),
            mps_low_watermark_ratio=(
                1.0
                if train_config.mps_low_watermark_ratio is None
                else train_config.mps_low_watermark_ratio
            ),
            enable_mps_fallback=(
                True
                if train_config.enable_mps_fallback is None
                else train_config.enable_mps_fallback
            ),
        )
    )


def _collect_explicit_cli_overrides(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> set[str]:
    """
    Record which CLI destinations were explicitly provided by the caller.

    Purpose:
    let profile resolution distinguish between user-authored overrides and
    profile-supplied defaults.

    Context:
    device profiles in this repository are intended to be helpful defaults,
    not unconditional rewrites of every runtime-facing field.
    """
    # The profile resolver needs to know which flags were explicitly provided
    # so it can treat the device profile as a source of defaults rather than an
    # unconditional override.
    explicit_overrides: set[str] = set()
    option_actions = getattr(parser, "_option_string_actions", {})
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None:
            explicit_overrides.add(action.dest)
    return explicit_overrides


def _print_runtime_diagnostics(
    diagnostics: Sequence[RuntimeDiagnostic],
) -> None:
    """
    Print formatted runtime diagnostics when any are present.

    Purpose:
    keep diagnostic printing behavior centralized so CLI and future helper
    flows present the same output shape.

    Context:
    the formatting logic itself lives in the environment package; this helper
    only owns the lightweight "should anything be printed?" policy.
    """
    if not diagnostics:
        return
    print("Runtime diagnostics:")
    print(format_runtime_diagnostics(diagnostics))


def _resolve_eval_ckpt_path(
    fit_artifacts: FitArtifacts,
    eval_ckpt_path: CheckpointSelection,
) -> CheckpointSelection:
    """
    Normalize evaluation checkpoint selection after fitting completes.

    Purpose:
    convert the user-facing checkpoint preference into a choice that remains
    valid even when a best checkpoint was never produced.

    Context:
    validation can be absent in some runs, so asking for `"best"` needs to
    degrade gracefully to the in-memory fitted model rather than failing late
    during test or prediction.
    """
    # A top-level workflow should degrade gracefully when there is no
    # validation-ranked "best" checkpoint to reload. In that case we fall back
    # to the in-memory model weights from the just-finished fit run.
    if eval_ckpt_path == "best" and not fit_artifacts.best_checkpoint_path:
        return None
    return eval_ckpt_path
