from __future__ import annotations

# AI-assisted maintenance note:
# `workflows` houses the repository's top-level execution flows.
#
# Why keep this package:
# - `main.py` is still the stable user-facing entrypoint
# - the actual run logic grew beyond what one root script should reasonably own
# - splitting the orchestration into smaller modules makes the CLI, reusable
#   training workflow, and shared helper logic easier to inspect separately
#
# Responsibility boundary:
# - `types.py` defines stable artifact/result containers
# - `helpers.py` contains small normalization and parsing helpers shared across
#   CLI and workflow code
# - `training.py` owns the reusable train/evaluate/predict flows
# - `cli.py` turns command-line arguments into structured config and delegates
#   to the reusable workflows

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
    "main",
    "run_environment_benchmark_workflow",
    "run_training_workflow",
]
