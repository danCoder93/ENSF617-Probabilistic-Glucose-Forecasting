# Current Architecture

This document describes the repository as it exists today: the major packages,
their ownership boundaries, the runtime flow, the artifact outputs, and the
constraints that are still intentional.

Use this file as the current-system reference. For how the repository reached
this shape, read [`codebase_evolution.md`](codebase_evolution.md).

## System Goal

The repository is a research-oriented probabilistic glucose forecasting system
built around a fused TCN + TFT architecture, a Lightning-oriented training
runtime, structured held-out evaluation, and a richer observability surface for
inspection and debugging.

At a high level, the system is trying to do four things well:

- prepare AZT1D data into a stable model-facing contract
- train a probabilistic hybrid forecasting model with clear runtime ownership
- generate structured evaluation outputs rather than only scalar test metrics
- leave behind enough logs and artifacts that a run can be understood later

## Repository Map

A simplified view of the repository today looks like this:

```text
defaults.py
main.py
main.ipynb
docs/
  assets/
  history/
  codebase_evolution.md
  current_architecture.md
src/
  config/
  data/
  evaluation/
  models/
  observability/
  train.py
  utils/
tests/
```

Each of those areas exists for a specific reason.

## Repository Map By Responsibility

For day-to-day navigation, the most useful way to think about the repository is
by task rather than by alphabetical file listing.

### If you are working on data ingestion or dataset semantics

Look at:

- `src/config/data.py`
- `src/data/downloader.py`
- `src/data/preprocessor.py`
- `src/data/schema.py`
- `src/data/transforms.py`
- `src/data/indexing.py`
- `src/data/dataset.py`
- `src/data/datamodule.py`
- `tests/data/`

### If you are working on model architecture or forecast behavior

Look at:

- `src/config/model.py`
- `src/models/fused_model.py`
- `src/models/tcn.py`
- `src/models/tft.py`
- `src/models/grn.py`
- `src/models/nn_head.py`
- `tests/test_fused_model.py`
- `tests/test_grn.py`

### If you are working on run orchestration or checkpoint behavior

Look at:

- `defaults.py`
- `main.py`
- `src/config/runtime.py`
- `src/train.py`
- `tests/test_main.py`
- `tests/test_train.py`

### If you are working on observability, exports, or reports

Look at:

- `src/config/observability.py`
- `src/observability/runtime.py`
- `src/observability/callbacks.py`
- `src/observability/reporting.py`
- `tests/test_observability_package.py`
- `tests/test_observability_reporting.py`
- `tests/test_observability_runtime_and_callbacks.py`

### If you are working on canonical metric computation

Look at:

- `src/evaluation/core.py`
- `src/evaluation/metrics.py`
- `src/evaluation/grouping.py`
- `src/evaluation/evaluator.py`
- `tests/test_evaluation_metrics.py`
- `tests/test_evaluation_evaluator.py`

## Architectural Principles

The current architecture is built around a few stable principles:

- semantic data contracts are preferred over raw positional tensors
- model behavior, data lifecycle, runtime orchestration, evaluation, and
  observability are separate concerns
- runtime policy is expressed through typed config objects
- root-level entrypoints should stay thin and user-facing
- documentation should explain intent and boundaries, not only APIs

## End-To-End Runtime Flow

The normal run path through the repository is:

1. `defaults.py` builds baseline config objects for data, model, training,
   snapshots, and observability.
2. `main.py` parses CLI arguments and converts them into those typed config
   objects.
3. `main.py` constructs `AZT1DDataModule` from `config.data`.
4. `main.py` constructs `FusedModelTrainer` from the top-level config plus
   runtime policy configs.
5. `FusedModelTrainer` calls `datamodule.prepare_data()` and `datamodule.setup()`
   before model construction.
6. `AZT1DDataModule` discovers runtime categorical cardinalities and final
   sequence-aligned feature details, then binds them into a new runtime config.
7. `FusedModelTrainer` builds `FusedModel` from that runtime-bound config.
8. `FusedModelTrainer` assembles callbacks, loggers, checkpoint policy, and the
   Lightning `Trainer`.
9. Lightning executes `fit(...)`.
10. If test windows exist, the wrapper can run `test(...)` and `predict(...)`
    against the resolved checkpoint or the current in-memory weights.
11. `main.py` uses raw predictions plus aligned test batches to compute
    structured held-out evaluation through `src/evaluation/`.
12. `src/observability/` exports prediction tables, reports, and runtime
    artifacts.
13. `main.py` writes a compact `run_summary.json` describing the run.

That same underlying workflow is shared by the notebook path. `main.ipynb`
reuses the same Python surfaces rather than keeping its own independent
training logic.

## Data Flow

The most important end-to-end data flow in the repository is:

1. raw AZT1D archive is downloaded and extracted by `src/data/downloader.py`
2. the raw export is standardized into one canonical processed CSV by
   `src/data/preprocessor.py`
3. the cleaned dataframe is loaded and normalized by
   `src/data/transforms.py`
4. semantic feature groups are derived through `src/data/schema.py`
5. split frames and legal sequence windows are built through
   `src/data/indexing.py`
6. `src/data/dataset.py` turns each index entry into one structured sample
7. `AZT1DDataModule` wraps those datasets in train/validation/test dataloaders
8. batches flow into `FusedModel` through the grouped batch contract
9. predictions and aligned test batches later flow into `src/evaluation/` and
   `src/observability/reporting.py`

The important architectural point is that the data is progressively made more
structured as it moves through the system. The repo does not jump directly from
downloaded files to model tensors in one opaque step.

## Config Flow

The repository has two important config states:

- declarative config
- runtime-bound config

### Declarative config

This is the config created by `defaults.py`, the CLI, tests, or notebook code.
It expresses the intended run setup:

- dataset paths and sequence lengths
- model hyperparameters
- Trainer policy
- snapshot policy
- observability policy

### Runtime-bound config

This is the config produced after the DataModule has prepared and inspected the
actual dataset. It includes the declarative settings plus runtime-discovered
details such as:

- categorical cardinalities
- fallback feature specs when needed
- finalized TFT-facing sequence-aligned metadata

The handoff happens in `AZT1DDataModule.bind_model_config(...)`.

### Why this distinction matters

This distinction explains several otherwise unusual design choices:

- why the trainer wrapper prepares the DataModule before constructing the model
- why config serialization matters for checkpoints
- why the repo cannot treat the initial config as the complete final truth for
  model construction

## Root-Level Entry Surfaces

### `defaults.py`

`defaults.py` is the baseline configuration surface for script and notebook
usage.

It provides builders for:

- the top-level model/data config
- `TrainConfig`
- `SnapshotConfig`
- `ObservabilityConfig`

It also bootstraps `src/` imports for root-level consumers. That means root
entrypoints and notebooks can use imports like `from data...` and `from config...`
without duplicating path setup logic in multiple places.

The defaults are intentionally convenience-oriented research defaults. They are
meant to make the repo runnable end to end, not to encode one final or optimal
experiment configuration.

### `main.py`

`main.py` is the primary script entrypoint and the main top-level workflow
surface.

Its role is deliberately narrow:

- parse CLI arguments
- normalize them into typed config
- construct the DataModule and trainer wrapper
- call the shared run workflow
- persist summaries and artifacts

It is intentionally not the place where:

- data preprocessing logic lives
- model internals live
- evaluation metrics are defined
- callback implementations are written

That separation is what keeps the script readable even as the rest of the
system grows.

### `main.ipynb`

The notebook is a convenience surface for interactive work, not a second
architecture. It exists so teammates can explore or debug runs interactively
without having to fork the actual pipeline into notebook-only logic.

## Configuration Layer: `src/config/`

The config package is the canonical source of runtime contracts.

### Main modules

- `data.py`
  data locations, split policy, sequence lengths, loader settings, and feature
  grouping inputs
- `model.py`
  `TCNConfig`, `TFTConfig`, and the top-level `Config`
- `runtime.py`
  `TrainConfig` and `SnapshotConfig`
- `observability.py`
  `ObservabilityConfig`
- `serde.py`
  conversion helpers between typed config objects and plain checkpoint-friendly
  dictionaries
- `types.py`
  shared type aliases such as path-like inputs
- `__init__.py`
  public facade for convenient imports

### Why this package matters

Config is not a peripheral utility in this repository. It defines:

- the data contract
- the model's structural parameters
- Trainer behavior
- checkpointing policy
- observability policy
- checkpoint serialization behavior

The move into `src/config/` reflects the fact that these are central project
contracts, not miscellaneous helpers.

### The Most Important Config Objects

When reading or modifying the repo, the main config classes to know are:

- `DataConfig`
  dataset access, sequence construction, split policy, loader behavior
- `TCNConfig`
  project-specific TCN branch settings
- `TFTConfig`
  TFT branch settings plus runtime-bound feature/count metadata
- `Config`
  top-level data + model config bundle
- `TrainConfig`
  Lightning `Trainer` behavior
- `SnapshotConfig`
  checkpoint policy
- `ObservabilityConfig`
  logging, telemetry, and reporting policy

## Data Layer: `src/data/`

The data pipeline is organized around `AZT1DDataModule`.

### Responsibility split

- `downloader.py`
  download and extract the raw AZT1D archive
- `preprocessor.py`
  build one canonical processed CSV from the raw dataset
- `schema.py`
  define feature groups, category vocabularies, and model-facing schema rules
- `transforms.py`
  load and normalize the processed dataframe
- `indexing.py`
  build legal encoder/decoder windows and split-specific sample indices
- `dataset.py`
  materialize one structured sample per index entry
- `datamodule.py`
  own the Lightning lifecycle and DataLoader creation

### Data lifecycle

The important lifecycle split is:

- `prepare_data()`
  disk-side effects such as download/extraction/preprocessing
- `setup()`
  in-memory dataframe loading, category-map fitting, split creation, and
  dataset construction
- loader methods
  batching only

That split is intentional because the repo leans on Lightning conventions
without giving up control over its runtime-bound model setup.

### Model-Facing Batch Contract

The current batch contract is explicit:

```python
{
    "static_categorical": ...,
    "static_continuous": ...,
    "encoder_continuous": ...,
    "encoder_categorical": ...,
    "decoder_known_continuous": ...,
    "decoder_known_categorical": ...,
    "target": ...,
    "metadata": ...,
}
```

That grouped layout matters because the model does not treat all inputs as one
anonymous tensor. The groups correspond to real semantic roles in the fused
architecture.

### Runtime Metadata Binding

One of the most important current contracts is that the DataModule owns
data-derived runtime metadata.

After `setup()`, it can provide:

- categorical embedding cardinalities in TFT order
- fallback feature specs when explicit feature specs are absent
- sequence lengths aligned with the actual prepared dataset

`bind_model_config(...)` returns a new config rather than mutating the original
in place. That is an intentional design choice:

- the declarative config remains inspectable
- the runtime-bound config becomes explicit
- training code can log or compare both if needed

This contract is why the training wrapper prepares the DataModule before
constructing the model.

### Data-Layer Extension Rule

If you are changing feature semantics, split policy, preprocessing behavior, or
sample structure, the data layer is the first place to update. Do not patch
those concerns ad hoc in `main.py` or inside `FusedModel`.

## Model Layer: `src/models/`

The model package contains the forecasting architecture.

### Main files

- `fused_model.py`
  top-level Lightning-native forecasting model
- `tft.py`
  Temporal Fusion Transformer branch
- `tcn.py`
  project-specific causal residual TCN branch
- `grn.py`
  gated residual network blocks
- `nn_head.py`
  final prediction head

### Fused Forecasting Design

The current model is a late-fusion hybrid:

- three TCN branches at kernel sizes `3`, `5`, and `7`
- one TFT branch over grouped static, historical, and future-known inputs
- one post-branch GRN fusion layer
- one final head that emits quantile forecasts

The forward logic is conceptually:

1. split `encoder_continuous` into known-history, observed-history, and target
   history slices
2. build narrower history-only inputs for the TCN branches
3. build grouped TFT inputs from static features, encoder history, and
   decoder-known future features
4. run the TCN branches to get horizon-aligned latent features
5. run the TFT branch to get horizon-aligned decoder features before final
   quantile projection
6. concatenate those latent features
7. fuse them with a GRN
8. project them through `NNHead` into final quantile outputs

This means the fusion happens in representation space, not after each branch
has already collapsed to its own final forecast.

### Probabilistic Output Contract

`FusedModel` predicts quantiles and owns the quantile-loss interpretation of
those channels. It does not leave that semantic contract to outer training
code.

The model also owns:

- `training_step(...)`
- `validation_step(...)`
- `test_step(...)`
- `predict_step(...)`
- `configure_optimizers(...)`

That boundary keeps output semantics and supervision behavior close together.

### Lightning-Specific Model Decisions

The current model contains a few architectural decisions that exist because the
repo is Lightning-oriented:

- config can be passed either as a typed `Config` or as a serialized mapping so
  `load_from_checkpoint(...)` works cleanly
- lazy TFT parameters are proactively materialized during model construction so
  optimizer setup is deterministic
- quantiles are cached on the model so output width, pinball loss, and point
  forecast extraction all use the same ordered tuple

### Model-Layer Extension Rule

If you change forecast semantics, branch composition, fusion behavior, or loss
interpretation, that change should be reflected in:

- `src/models/fused_model.py`
- the relevant branch module
- config contracts in `src/config/model.py`
- the corresponding tests

Try not to smuggle model semantics into the training wrapper or reporting code.

## Training Runtime Layer: `src/train.py`

`src/train.py` is the orchestration layer above the model and data stack.

Its main public surface is `FusedModelTrainer`.

### What it owns

- preparing and setting up the DataModule before model construction
- binding runtime config through the DataModule
- constructing `FusedModel`
- assembling callbacks
- assembling the Lightning `Trainer`
- fit/test/predict orchestration
- caching the best-checkpoint path and the current in-memory runtime state

### What it intentionally does not own

- data preprocessing internals
- model forward math
- metric definitions
- report generation internals

### Callback and checkpoint policy

The wrapper uses validation presence to decide how to build checkpoints:

- with validation data, snapshots can be ranked on `val_loss`
- without validation data, it falls back to a last-checkpoint-only policy

This is an important detail because the repo does not pretend a meaningful
`"best"` checkpoint exists when no validation signal exists to rank snapshots.

## Runtime Artifact Flow

The runtime artifact flow is:

1. `main.py` creates or chooses the output directory
2. `defaults.py` derives default artifact paths under that directory
3. `src/observability/runtime.py` assembles logger/profiler/text-log objects
4. Lightning callbacks emit run-time diagnostics during training
5. `main.py` optionally saves raw prediction tensors after prediction
6. `src/observability/reporting.py` optionally exports a flat prediction CSV
7. `src/observability/reporting.py` optionally generates Plotly HTML reports
8. `main.py` writes `run_summary.json` with config, environment, evaluation,
   and artifact metadata

This matters because not all artifacts are produced at the same lifecycle
stage. Some exist during training, while others exist only after prediction has
completed.

### Current wrapper limitation

`test(...)` and `predict_test(...)` still assume that `fit()` has already been
called on the current wrapper instance.

Explicit checkpoint paths can be used to choose evaluation weights, but the
wrapper does not yet rebuild a fully fresh evaluation-only Trainer session from
scratch. That limitation is documented in the code and should be considered
part of the current runtime contract.

## Observability Layer: `src/observability/`

Observability is a dedicated subsystem rather than a side effect of training.

### Main modules

- `runtime.py`
  logger, profiler, and artifact-path setup
- `callbacks.py`
  callback-driven telemetry, diagnostics, and visualization hooks
- `logging_utils.py`
  shared logger-aware helpers
- `tensors.py`
  normalization helpers for nested batch/tensor structures
- `reporting.py`
  prediction table export and Plotly report generation
- `utils.py`
  small utility helpers
- `__init__.py`
  package-level convenience facade

### What observability means in this repo

Observability includes:

- TensorBoard or CSV logger setup
- text run logging
- optional profiler setup
- callback-driven telemetry
- parameter and gradient monitoring
- prediction figure generation
- graph/model visualization support
- prediction-table export
- Plotly HTML report generation

### Observability policy is runtime policy

`ObservabilityConfig` lives separately from the model/data architecture config
because observability is treated as a runtime concern:

- it changes how visible a run is
- it does not redefine the checkpointed forecasting architecture itself

### Optional dependency philosophy

The observability stack is designed to degrade gracefully where possible:

- TensorBoard is preferred
- CSV logger can act as fallback
- optional extras improve the run rather than defining whether the whole system
  is conceptually valid

## Evaluation Layer: `src/evaluation/`

The evaluation package owns structured model-quality analysis.

### Main modules

- `types.py`
  evaluation result contracts
- `core.py`
  target normalization, metadata normalization, and input validation helpers
- `metrics.py`
  primitive metrics such as MAE, RMSE, bias, pinball loss, interval width, and
  empirical coverage
- `grouping.py`
  grouped aggregation helpers such as horizon, subject, and glucose-range views
- `evaluator.py`
  end-to-end evaluation assembly

### Why it is separate from observability

The distinction is intentional:

- observability is about what happened during the run and what artifacts help a
  human inspect it
- evaluation is about the canonical computation of model-quality metrics

This keeps metric logic from being duplicated across the model, the reporting
layer, and top-level scripts.

### Current evaluation boundary

The richer structured evaluation currently hangs off the prediction path rather
than directly off Lightning's reduced `trainer.test(...)` metric output. That
is because the evaluator needs:

- raw prediction tensors
- aligned targets
- aligned metadata

Scalar test metrics alone are not enough for the richer grouped evaluation the
repo now supports.

## Artifact Outputs

The repo is designed to leave behind a run you can inspect later.

With the default output directory of `artifacts/main_run/`, the main workflow
can emit:

- `run_summary.json`
  compact machine-readable summary of config, runtime, evaluation, and artifact
  locations
- `test_predictions.pt`
  raw prediction tensors
- `test_predictions.csv`
  flat exported prediction table
- `reports/`
  Plotly HTML reports
- `checkpoints/`
  model checkpoints
- `logs/`
  TensorBoard or CSV logger output
- `run.log`
  plain-text lifecycle/debug log
- `telemetry.csv`
  system telemetry output
- `profiler/`
  profiler output when enabled
- `model_viz/`
  torchview/model-visualization artifacts when enabled

The artifact strategy is intentionally additive:

- raw tensors are preserved for flexible downstream analysis
- flat exports exist for easy plotting and tabular inspection
- structured evaluation is embedded in the run summary
- logs and telemetry capture runtime context around the same run

### Why the artifact strategy is layered

Different consumers need different artifact shapes:

- PyTorch users may want raw tensors
- analysts may want CSVs
- teammates may want HTML reports
- automation may want JSON summaries
- debugging sessions may need text logs and telemetry

The repo supports all of those without forcing one representation to replace
the others.

## Test Suite

The test suite now protects multiple architectural layers, not just the model.

It covers:

- config validation and serialization
- fused-model behavior
- training-wrapper behavior
- entrypoint behavior
- evaluation package behavior
- observability package behavior

This matters because the repo is now a system, not just a model file.

## Documentation Layer

The documentation structure under `docs/` now has three roles:

- `current_architecture.md`
  current-system reference
- `codebase_evolution.md`
  historical narrative
- `history/`
  archived milestone notes written during specific refactors

The diagrams under `docs/assets/` support the model-side architecture story
without mixing binary assets into the source package.

## How To Extend The Codebase Safely

When adding new work, start by placing it in the right subsystem.

### Add a new data feature or preprocessing rule

Usually touch:

- `src/data/preprocessor.py`
- `src/data/schema.py`
- `src/data/transforms.py`
- `src/config/data.py`
- `src/data/datamodule.py`
- the matching tests in `tests/data/`

### Add a new model input, branch, or loss behavior

Usually touch:

- `src/config/model.py`
- `src/models/fused_model.py`
- one or more branch modules in `src/models/`
- model tests

### Add a new runtime flag or Trainer behavior

Usually touch:

- `defaults.py`
- `main.py`
- `src/config/runtime.py`
- `src/train.py`
- `tests/test_main.py` and/or `tests/test_train.py`

### Add a new metric or grouped evaluation summary

Usually touch:

- `src/evaluation/metrics.py`
- `src/evaluation/grouping.py`
- `src/evaluation/evaluator.py`
- evaluation tests

### Add a new export or report

Usually touch:

- `src/config/observability.py`
- `src/observability/reporting.py`
- maybe `main.py`
- observability/reporting tests

## Common Modification Guide

For common team tasks, here is where to start:

- "I want to change the default experiment settings"
  Start in `defaults.py`
- "I want to add a CLI flag"
  Start in `main.py`
- "I want to change how test predictions are saved"
  Start in `main.py` and `src/observability/reporting.py`
- "I want to change the grouped batch contract"
  Start in `src/data/dataset.py`, `src/data/datamodule.py`, and
  `src/models/fused_model.py`
- "I want to change checkpoint policy"
  Start in `src/config/runtime.py`, `defaults.py`, and `src/train.py`
- "I want to add a new evaluation view"
  Start in `src/evaluation/`
- "I want to add a new callback or runtime diagnostic"
  Start in `src/observability/callbacks.py`

## Boundaries That Should Stay Stable

The following boundaries are deliberate and should be preserved unless there is
a clear reason to change them:

- the DataModule discovers runtime metadata; the model consumes it
- `FusedModel` owns forecasting and supervision semantics
- `FusedModelTrainer` owns Lightning orchestration, not model math
- `main.py` stays thin and user-facing
- observability and evaluation remain separate subsystems
- typed config remains the canonical runtime contract

## Intentional Limitations And Current Constraints

A few current constraints are important to understand:

- `FusedModelTrainer.test(...)` and `predict_test(...)` still depend on prior
  `fit()` state
- structured evaluation currently depends on the prediction path, not only the
  reduced test-metric path
- some observability features depend on optional extras and may be skipped
  gracefully in minimal environments
- fallback feature-spec synthesis still exists while the repo transitions
  toward `config.data.features` as the single source of truth

These are not hidden bugs in the documentation. They are part of the current
shape of the codebase.

## Known Constraints And Assumptions

The current architecture assumes:

- AZT1D-style data preprocessing and feature semantics
- a grouped batch contract rather than a single raw input tensor
- runtime binding of TFT categorical metadata
- a Lightning-centered training workflow
- prediction-driven detailed evaluation

The current codebase also still carries a few transitional assumptions:

- fallback feature-spec synthesis still exists alongside explicit feature specs
- evaluation-only wrapper flows are not fully standalone yet
- some observability features remain optional and environment-dependent

These assumptions should be revisited deliberately if the repository grows into
multi-dataset support, alternative entry surfaces, or richer deployment-style
inference paths.

## Recommended Reading Order

For a teammate new to the repository, the best reading order is:

1. this file for the current system shape
2. [`codebase_evolution.md`](codebase_evolution.md) for the historical why
3. the relevant milestone note in [`history/`](history/) for subsystem-specific
   depth
