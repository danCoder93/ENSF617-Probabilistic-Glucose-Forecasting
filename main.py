from __future__ import annotations

# AI-assisted implementation note:
# This root file remains the repository's stable runnable entrypoint, but the
# heavier workflow logic now lives under `src/workflows/`.
#
# Why keep this facade:
# - users still expect `python main.py` and `from main import ...` to work
# - tests and notebooks already import top-level workflow helpers from here
# - keeping this file thin makes the entrypoint easier to scan while preserving
#   the existing public surface

from defaults import (
    build_default_config,
    build_default_observability_config,
    build_default_snapshot_config,
    build_default_train_config,
)
from workflows.cli import build_argument_parser, main
from workflows.helpers import _apply_early_apple_silicon_environment_defaults
from workflows.training import (
    run_environment_benchmark_workflow,
    run_training_workflow,
)
from workflows.types import EnvironmentBenchmarkArtifacts, MainRunArtifacts

__all__ = [
    "EnvironmentBenchmarkArtifacts",
    "MainRunArtifacts",
    "_apply_early_apple_silicon_environment_defaults",
    "build_argument_parser",
    "build_default_config",
    "build_default_observability_config",
    "build_default_snapshot_config",
    "build_default_train_config",
    "main",
    "run_environment_benchmark_workflow",
    "run_training_workflow",
]


if __name__ == "__main__":
    main()
