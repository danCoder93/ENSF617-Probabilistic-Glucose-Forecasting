# Extension And Constraints

Role: Focused current-state reference for safe modification paths and active
repository constraints.
Audience: Contributors and maintainers planning changes to the codebase.
Owns: Common modification starting points, stable boundaries, current
limitations, and active assumptions.
Related docs: [`../current_architecture.md`](../current_architecture.md),
[`package_boundaries.md`](package_boundaries.md),
[`runtime_and_config_flow.md`](runtime_and_config_flow.md),
[`current_architecture_reference.md`](current_architecture_reference.md).

## How To Extend The Codebase Safely

When adding new work, start by placing it in the right subsystem.

### Add a new data feature or preprocessing rule

Usually touch:

- `src/data/preprocessor.py`
- `src/data/schema.py`
- `src/data/transforms.py`
- `src/config/data.py`
- `src/data/datamodule.py`
- matching tests in `tests/data/`

### Add a new model input, branch, or loss behavior

Usually touch:

- `src/config/model.py`
- `src/models/fused_model.py`
- one or more branch modules in `src/models/`
- `tests/models/`

### Add a new runtime flag or Trainer behavior

Usually touch:

- `defaults.py`
- `main.py`
- `src/workflows/cli.py`
- `src/workflows/training.py`
- `src/environment/profiles.py`
- `src/environment/diagnostics.py`
- `src/config/runtime.py`
- `src/train.py`
- `tests/workflows/`
- `tests/training/`

### Add a new metric or grouped evaluation summary

Usually touch:

- `src/evaluation/metrics.py`
- `src/evaluation/grouping.py`
- `src/evaluation/evaluator.py`
- `tests/evaluation/`

### Add a new post-run export or report sink

Usually touch:

- `src/reporting/builders.py` only if the canonical shared-report surface needs
  to change
- `src/reporting/exports.py`
- `src/reporting/structured_exports.py`
- `src/reporting/tensorboard.py`
- `src/reporting/plotly_reports.py`
- maybe `src/workflows/training.py`
- `tests/reporting/`

## Boundaries That Should Stay Stable

- the DataModule discovers runtime metadata; the model consumes it
- the environment layer interprets runtime context; it does not define model or
  data semantics
- `FusedModel` owns forecasting and supervision semantics
- `FusedModelTrainer` owns Lightning orchestration, not model math
- `main.py` stays thin and user-facing
- `src/workflows/` owns reusable entry-surface orchestration logic
- `evaluation` computes canonical metric truth
- `reporting` packages and renders post-run artifacts from that truth
- `observability` and `reporting` remain separate subsystems
- typed config remains the canonical runtime contract

## Intentional Limitations And Current Constraints

Important current constraints include:

- `FusedModelTrainer.test(...)` and `predict_test(...)` still depend on prior
  `fit()` state
- structured evaluation currently depends on the prediction path, not only the
  reduced test-metric path
- some observability features depend on optional extras and may be skipped
  gracefully in minimal environments
- fallback feature-spec synthesis still exists while the repo transitions
  toward explicit feature declarations as the single source of truth
- the workflow's best-effort dataset summary export is still an additive side
  artifact rather than a required input to every reporting sink

## Known Assumptions

The current architecture assumes:

- AZT1D-style data preprocessing and feature semantics
- a grouped batch contract rather than a single raw input tensor
- runtime binding of TFT categorical metadata
- a Lightning-centered training workflow
- prediction-driven detailed evaluation
- post-run reporting built around a canonical shared-report package

## When You Need More Detail

- [`current_architecture_reference.md`](current_architecture_reference.md) for
  the preserved full reference
- [`../history/index.md`](../history/index.md) for milestone-level rationale
