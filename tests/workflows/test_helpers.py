from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import TrainConfig
from tests.support import build_runtime_environment
from workflows.cli import build_argument_parser
from workflows.helpers import (
    _apply_early_apple_silicon_environment_defaults,
    _collect_explicit_cli_overrides,
    _json_ready,
    _normalize_optional_string,
    _parse_csv_floats,
    _parse_csv_ints,
    _parse_devices,
    _parse_limit,
    _resolve_eval_ckpt_path,
)


def test_json_ready_normalizes_paths_dataclasses_and_nested_sequences() -> None:
    payload = {
        "path": Path("artifacts/run"),
        "environment": build_runtime_environment(cuda_available=True),
        "values": (1, Path("logs"), {"flag": True}),
    }

    rendered = _json_ready(payload)

    assert rendered["path"] == "artifacts/run"
    assert rendered["environment"]["cuda_available"] is True
    assert rendered["values"] == [1, "logs", {"flag": True}]


def test_parse_helpers_preserve_cli_value_types() -> None:
    assert _parse_csv_ints("64, 32,16") == (64, 32, 16)
    assert _parse_csv_floats("0.1, 0.5, 0.9") == (0.1, 0.5, 0.9)
    assert _parse_devices("auto") == "auto"
    assert _parse_devices("2") == 2
    assert _parse_devices("0,1") == [0, 1]
    assert _parse_limit("0.25") == 0.25
    assert _parse_limit("4") == 4
    assert _normalize_optional_string("") is None
    assert _normalize_optional_string("null") is None
    assert _normalize_optional_string("weights.ckpt") == "weights.ckpt"


def test_collect_explicit_cli_overrides_tracks_cli_destinations() -> None:
    parser = build_argument_parser()

    overrides = _collect_explicit_cli_overrides(
        parser,
        ["--max-epochs", "3", "--allow-tf32", "--devices=0,1"],
    )

    assert {"max_epochs", "allow_tf32", "devices"}.issubset(overrides)


def test_resolve_eval_ckpt_path_falls_back_when_best_checkpoint_is_missing() -> None:
    assert _resolve_eval_ckpt_path(
        SimpleNamespace(best_checkpoint_path=""),
        "best",
    ) is None
    assert _resolve_eval_ckpt_path(
        SimpleNamespace(best_checkpoint_path="model.ckpt"),
        "best",
    ) == "best"


def test_apply_early_apple_silicon_environment_defaults_sets_mps_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("workflows.helpers.platform.system", lambda: "Darwin")
    monkeypatch.setattr("workflows.helpers.platform.machine", lambda: "arm64")
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    _apply_early_apple_silicon_environment_defaults(
        requested_device_profile="auto",
        train_config=TrainConfig(),
    )

    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "1.3"
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "1.0"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
