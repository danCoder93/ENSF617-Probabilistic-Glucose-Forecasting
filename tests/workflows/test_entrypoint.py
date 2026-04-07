from __future__ import annotations

"""
AI-assisted maintenance note:
This test protects the root `main.py` facade that re-exports the project's
workflow and default-builder surface.

Context:
the repository intentionally keeps `main.py` as the stable top-level import and
runnable entrypoint even though most implementation now lives under `src/`.
"""

import main as entrypoint
from defaults import (
    build_default_config,
    build_default_observability_config,
    build_default_snapshot_config,
    build_default_train_config,
)
from workflows.cli import build_argument_parser, main as cli_main
from workflows.helpers import _apply_early_apple_silicon_environment_defaults
from workflows.training import (
    run_environment_benchmark_workflow,
    run_training_workflow,
)


def test_root_entrypoint_reexports_workflow_and_default_builders() -> None:
    # This file is intentionally tiny because the behavior under test is also
    # tiny: `main.py` should remain a stable facade over the reorganized
    # workflow/default modules.
    assert entrypoint.build_default_config is build_default_config
    assert entrypoint.build_default_train_config is build_default_train_config
    assert entrypoint.build_default_snapshot_config is build_default_snapshot_config
    assert entrypoint.build_default_observability_config is build_default_observability_config
    assert entrypoint.build_argument_parser is build_argument_parser
    assert entrypoint.main is cli_main
    assert (
        entrypoint._apply_early_apple_silicon_environment_defaults
        is _apply_early_apple_silicon_environment_defaults
    )
    assert entrypoint.run_training_workflow is run_training_workflow
    assert entrypoint.run_environment_benchmark_workflow is run_environment_benchmark_workflow
