# Training Wrapper Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the introduction of the reusable PyTorch Lightning
training wrapper and the related config/test updates.

This document summarizes the work that added `src/train.py` as the
repository-level Lightning orchestration layer for the fused TCN + TFT
forecasting stack.

## Goals

- Add a reusable training/evaluation wrapper that can be called from `main.py`
  or notebooks.
- Keep the Lightning responsibility split clean:
  model behavior in `FusedModel`,
  data orchestration in `AZT1DDataModule`,
  trainer/bootstrap flow in `src/train.py`.
- Add explicit typed configs for Trainer-side execution policy and checkpoint
  snapshot policy.
- Document the current training/evaluation workflow and its present
  limitations.

## Files Added or Updated

- `src/train.py`
- `src/config/runtime.py`
- `src/config/observability.py`
- `src/config/__init__.py`
- `tests/test_train.py`
- `README.md`

## Architectural Role of `src/train.py`

The new `src/train.py` does **not** redefine model-side learning logic.

Instead, it acts as the orchestration layer above the already-refactored model
and data stack:

- `src/models/fused_model.py`
  still owns `training_step(...)`, `validation_step(...)`, `test_step(...)`,
  `predict_step(...)`, and `configure_optimizers(...)`
- `src/data/datamodule.py`
  still owns `prepare_data()`, `setup()`, and the train/validation/test
  dataloaders
- `src/train.py`
  now owns:
  Trainer construction,
  callback construction,
  checkpoint snapshot policy,
  fit/test/predict orchestration,
  and runtime model construction from the DataModule-bound config

This keeps the project aligned with the intended PyTorch Lightning structure:

1. data preparation in the `LightningDataModule`
2. optimization behavior in the `LightningModule`
3. outer-loop run orchestration in a thin training wrapper

## Main Additions

### 1. `FusedModelTrainer`

`src/train.py` introduces `FusedModelTrainer`, a reusable wrapper around
Lightning's `Trainer`.

Its public entry points are:

- `fit(datamodule, ckpt_path=None)`
- `test(datamodule, ckpt_path="best")`
- `predict_test(datamodule, ckpt_path="best")`
- `fit_test_predict(datamodule, ...)`
- helper inspection methods such as
  `build_model(...)`,
  `has_validation_data(...)`,
  and `has_test_data(...)`

### 2. Runtime Config Binding Before Model Construction

One important requirement of this repository is that the final TFT-facing model
config depends on metadata discovered by the DataModule at runtime, such as:

- categorical cardinalities
- fallback feature-schema reconstruction when explicit `FeatureSpec` entries are
  absent
- final sequence lengths aligned with the real prepared dataset

Because of that, `FusedModelTrainer.build_model(...)` does not construct the
model directly from the original declarative `Config`.

Instead, it:

1. calls `datamodule.prepare_data()`
2. calls `datamodule.setup()`
3. calls `datamodule.bind_model_config(config)`
4. constructs `FusedModel` from the resulting runtime-bound config

This is the key design reason the project needed its own training wrapper
instead of only a bare `Trainer(...)` call in ad hoc notebooks.

### 3. Snapshot and Trainer Config Types

The work also added two project-level config dataclasses in
`src/config/runtime.py`:

- `SnapshotConfig`
- `TrainConfig`

`SnapshotConfig` centralizes checkpoint/snapshot policy such as:

- whether snapshots are enabled
- where snapshots are written
- which validation metric is monitored
- whether to save weights only or full checkpoints
- whether to keep only the best checkpoint, the last checkpoint, or more

`TrainConfig` centralizes Trainer-side run settings such as:

- accelerator/device choice
- precision
- epoch limits
- sanity-validation settings
- batch limits
- progress/model-summary toggles
- early-stopping patience

Keeping these as typed configs makes notebook and `main.py` integration easier
because run policy can be passed around as data instead of repeatedly
reconstructing long `Trainer(...)` argument lists.

## Callback Policy

The wrapper currently builds callbacks with the following behavior:

- if validation data exists and snapshots are enabled:
  create a `ModelCheckpoint` callback monitored on `val_loss`
- if validation data does not exist and snapshots are enabled:
  keep only the latest snapshot (`save_last=True`) without pretending a
  validation-ranked "best" checkpoint exists
- if validation data exists and early stopping is enabled:
  create an `EarlyStopping` callback that uses the same monitor/mode as the
  snapshot policy

This keeps checkpoint ranking and early-stopping behavior aligned rather than
letting them drift onto different metrics.

## Evaluation and Prediction Flow

After `fit(...)`, the wrapper caches:

- the in-memory `FusedModel`
- the runtime-bound `Config`
- the built Lightning `Trainer`
- the best-checkpoint path reported by Lightning's checkpoint callback

That allows later calls to:

- `test(..., ckpt_path="best")`
- `predict_test(..., ckpt_path="best")`

to reuse the current run state in a notebook-friendly way.

## Current Limitation

One important limitation remains intentionally documented in the code:

- `test(...)` and `predict_test(...)` still assume that `fit()` has already
  been called in the current wrapper instance

In other words, the wrapper currently supports the common experiment workflow:

- train
- validate
- test
- predict on held-out test windows

but it is not yet a fully standalone "load a checkpoint and evaluate without a
prior fit call" entry point.

The model itself is already checkpoint-friendly through
`FusedModel.load_from_checkpoint(...)`; the remaining gap is rebuilding the full
evaluation `Trainer` session around that path inside the wrapper.

## Testing Added

The work added `tests/test_train.py` to protect the new orchestration contract.

That test module covers:

- runtime TFT metadata binding during model construction
- validation-aware callback construction
- weights-only snapshot policy without validation ranking
- guardrails around checkpoint-alias usage before fitting

## Practical Usage

The intended usage pattern is:

```python
from data.datamodule import AZT1DDataModule
from train import FusedModelTrainer
from config import SnapshotConfig, TrainConfig

datamodule = AZT1DDataModule(config.data)

runner = FusedModelTrainer(
    config,
    trainer_config=TrainConfig(max_epochs=20),
    snapshot_config=SnapshotConfig(save_weights_only=True),
)

artifacts = runner.fit_test_predict(datamodule)
```

This keeps notebook or `main.py` setup code small while still making the
Lightning run policy explicit and configurable.

## Later Structural Update

After the initial training-wrapper work, the repository also promoted the
configuration layer into a dedicated top-level package:

- `src/config/data.py`
- `src/config/model.py`
- `src/config/runtime.py`
- `src/config/observability.py`
- `src/config/serde.py`

That means the training wrapper now sits on top of a clearer config ownership
boundary instead of relying on one oversized `utils/config.py` module.

The behavioral intent stayed the same, but the architectural story improved:

- config is no longer treated as a generic utility
- runtime policy now lives in a purpose-specific config module
- imports better reflect the true structure of the repository

## Later Observability Package Update

The training wrapper still imports observability through the same public module
name:

- `from observability import ...`

Internally, though, observability now lives in a package under
`src/observability/` rather than one oversized `src/observability.py` file.

That follow-up refactor did not change the training wrapper's architectural
role. It simply made the observability internals easier to navigate while
preserving the wrapper's existing integration points.
