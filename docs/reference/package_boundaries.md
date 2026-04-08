# Package Boundaries

Role: Focused current-state reference for package ownership and architectural
boundaries.
Audience: Engineers and contributors scoping changes to the codebase.
Owns: Source-tree responsibility map, subsystem ownership, and stable
architectural boundary lines.
Related docs: [`../current_architecture.md`](../current_architecture.md),
[`runtime_and_config_flow.md`](runtime_and_config_flow.md),
[`data_and_model_contract.md`](data_and_model_contract.md),
[`extension_and_constraints.md`](extension_and_constraints.md).

## Repository Map By Responsibility

For day-to-day navigation, the most useful way to think about the repository is
by task rather than by alphabetical file listing.

### Entry surfaces

- [`../../main.py`](../../main.py)
- [`../../defaults.py`](../../defaults.py)
- `main.ipynb`

These stay intentionally thin and user-facing.

### Configuration

- [`../../src/config/`](../../src/config/)

This package owns typed contracts for data, model, training, snapshot, and
observability behavior.

### Runtime environment

- [`../../src/environment/`](../../src/environment/)

This package owns environment detection, profile resolution, diagnostics, and
backend tuning.

### Data

- [`../../src/data/`](../../src/data/)

This package owns download, preprocessing, schema semantics, split/index
construction, dataset materialization, and the DataModule lifecycle.

### Models

- [`../../src/models/`](../../src/models/)

This package owns the TCN branch, TFT branch, fusion path, neural head, and
the forecasting semantics of `FusedModel`.

### Training runtime

- [`../../src/train.py`](../../src/train.py)

This layer owns DataModule preparation, runtime model binding, Lightning
assembly, and fit/test/predict orchestration.

### Workflows

- [`../../src/workflows/`](../../src/workflows/)

This package owns CLI assembly, top-level workflow execution, benchmark mode,
summary writing, and post-run handoff into evaluation and reporting.

### Evaluation

- [`../../src/evaluation/`](../../src/evaluation/)

This package owns canonical metric computation and grouped held-out evaluation.

### Observability

- [`../../src/observability/`](../../src/observability/)

This package owns logger setup, profiler setup, callback-driven telemetry, and
other runtime-facing visibility surfaces.

### Reporting

- [`../../src/reporting/`](../../src/reporting/)

This package owns the canonical post-run packaging and export surfaces.

### Tests

- [`../../tests/`](../../tests/)

The test tree mirrors the package structure closely enough to act as evidence
for which subsystem boundaries are considered stable.

## Architectural Principles

The current architecture is built around a few stable ideas:

- semantic data contracts are preferred over raw positional tensors
- model behavior, data lifecycle, runtime orchestration, evaluation,
  observability, and reporting are separate concerns
- runtime policy is expressed through typed config plus an explicit
  environment-aware interpretation layer
- root-level entrypoints stay thin and user-facing
- evaluation computes metric truth once, then reporting packages and renders it
- post-run sinks consume canonical shared-report surfaces rather than silently
  rebuilding metric logic

## Stable Boundaries

These boundaries are deliberate and should be preserved unless there is a clear
reason to change them:

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

## Related Deep Reads

- [`current_architecture_reference.md`](current_architecture_reference.md) for
  the preserved long-form reference
- [`../system_walkthrough.md`](../system_walkthrough.md) for the guided system
  read
- [`../history/index.md`](../history/index.md) for historical subsystem changes
