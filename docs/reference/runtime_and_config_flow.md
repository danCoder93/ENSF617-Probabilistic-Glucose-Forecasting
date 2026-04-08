# Runtime And Config Flow

Role: Focused current-state reference for how execution starts and how runtime
policy is resolved.
Audience: Engineers and contributors following the runnable lifecycle from CLI
to workflow to trainer.
Owns: End-to-end runtime flow, config states, entry surfaces, and
environment/runtime ownership.
Related docs: [`../current_architecture.md`](../current_architecture.md),
[`package_boundaries.md`](package_boundaries.md),
[`data_and_model_contract.md`](data_and_model_contract.md),
[`../execution_guide.md`](../execution_guide.md),
[`../cli_reference.md`](../cli_reference.md).

## End-To-End Runtime Flow

The normal run path through the repository is:

1. [`../../defaults.py`](../../defaults.py) builds baseline config objects for
   data, model, training, snapshots, and observability.
2. [`../../main.py`](../../main.py) or `main.ipynb` parses user-supplied
   overrides and converts them into typed config objects plus entry-surface
   runtime options.
3. [`../../src/environment/`](../../src/environment/) detects the current
   runtime context and resolves the requested device profile into effective
   runtime defaults.
4. Diagnostics and backend tuning are applied before training starts.
5. [`../../src/workflows/training.py`](../../src/workflows/training.py)
   constructs the DataModule and training wrapper.
6. The DataModule prepares the data and binds runtime-discovered metadata into a
   runtime-aware model config.
7. The trainer wrapper constructs the model, callbacks, logger surfaces, and
   Lightning `Trainer`.
8. Lightning executes `fit(...)`.
9. Optional test/predict flows produce the richer evaluation and reporting
   surfaces used later by artifact exports.

## Config States

The repository has two important config states:

- declarative config
- runtime-bound config

### Declarative config

This is the config created by defaults, CLI parsing, tests, or notebook code.
It expresses intended run setup:

- dataset paths and sequence lengths
- model hyperparameters
- Trainer policy
- snapshot policy
- observability policy

### Runtime-bound config

This is produced after the DataModule has prepared and inspected the actual
dataset. It includes declarative settings plus runtime-discovered facts such as:

- categorical cardinalities
- fallback feature specs when needed
- finalized TFT-facing sequence-aligned metadata

That distinction explains why the trainer wrapper prepares the DataModule
before constructing the model.

## Root-Level Entry Surfaces

### `defaults.py`

`defaults.py` is the baseline configuration surface for script and notebook
usage. It provides builders for:

- the top-level model/data config
- `TrainConfig`
- `SnapshotConfig`
- `ObservabilityConfig`

The defaults are intentionally convenience-oriented research defaults. They are
meant to make the repo runnable end to end, not to encode one final or optimal
experiment configuration.

### `main.py`

`main.py` is the primary script entrypoint and the stable top-level user-facing
facade.

Its role is deliberately narrow:

- parse CLI arguments
- normalize them into typed config
- delegate CLI assembly and workflow execution to `src/workflows/`
- preserve a convenient public import surface for scripts, notebooks, and tests

### `main.ipynb`

The notebook is a convenience surface for interactive work, not a second
architecture. It reuses the same environment-aware workflow as `main.py`
instead of forking notebook-only logic.

## Configuration Layer

[`../../src/config/`](../../src/config/) is the canonical source of runtime
contracts.

Main objects to know:

- `DataConfig`
- `TCNConfig`
- `TFTConfig`
- `Config`
- `TrainConfig`
- `SnapshotConfig`
- `ObservabilityConfig`

These are not peripheral helpers. They define the data contract, model
structure, trainer behavior, checkpoint policy, and visibility policy of a run.

## Environment Layer

[`../../src/environment/`](../../src/environment/) is the runtime
interpretation layer that sits between entry-surface intent and the lower-level
training stack.

It owns:

- runtime-environment detection
- device-profile inference and resolution
- preflight diagnostics
- environment-sensitive failure explanation
- backend-level tuning actions such as TF32, thread counts, and optional
  `torch.compile(...)`

The precedence rule is intentionally simple:

- explicit user override wins
- otherwise profile default applies
- otherwise repository baseline default applies

## Training Runtime Layer

[`../../src/train.py`](../../src/train.py) owns the orchestration above the
model and data stack.

Its main public surface is `FusedModelTrainer`, which owns:

- preparing and setting up the DataModule before model construction
- binding runtime config through the DataModule
- constructing `FusedModel`
- assembling callbacks
- assembling the Lightning `Trainer`
- fit/test/predict orchestration

It intentionally does not own model math, preprocessing internals, canonical
metric definitions, or reporting/export sink rendering.

## Workflow Layer

[`../../src/workflows/`](../../src/workflows/) sits above `src/train.py` and
below the thin root entry surfaces.

It is the first layer that can see all of these at once:

- resolved runtime policy
- fit artifacts
- optional test metrics
- optional prediction tensors
- grouped evaluation outputs
- reporting/export sinks
- final artifact paths

That is why post-run packaging and sink orchestration live here rather than in
`main.py` or in callbacks.

## Best Companion Reads

- [`../execution_guide.md`](../execution_guide.md) for practical run commands
- [`../cli_reference.md`](../cli_reference.md) for exact flags and CLI behavior
- [`data_and_model_contract.md`](data_and_model_contract.md) for what happens
  after data preparation starts
- [`current_architecture_reference.md`](current_architecture_reference.md) for
  the preserved full current-state reference
