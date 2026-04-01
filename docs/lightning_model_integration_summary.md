# PyTorch Lightning Model Integration Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the recent work that strengthened the model layer around
PyTorch Lightning without yet introducing the separate training script.

The intent of this document is to record:

- what changed
- why those changes were made
- how the work relates to the Lightning tutorial used as guidance
- what remains intentionally deferred to `train.py`

## Tutorial Reference

This pass was guided by the PyTorch Lightning tutorial provided for the task:

- [From PyTorch to PyTorch Lightning — A gentle introduction](https://medium.com/data-science/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)

The tutorial's central pattern is:

1. Keep the model architecture in `forward(...)`
2. Move optimization and supervision logic into the `LightningModule`
3. Keep data orchestration in a `LightningDataModule`
4. Let Lightning's `Trainer` own the training loop boilerplate

That pattern matches the direction of this repository well because the data
layer had already been refactored into a `LightningDataModule`, while the fused
model still needed to absorb the rest of the model-side Lightning contract.

## Scope of This Pass

The goal of this pass was to strengthen `src/models/fused_model.py` so it could
serve as the project's model-side Lightning entry point before creating a
future `train.py`.

This work did not aim to:
- redesign the fused TCN + TFT architecture
- move training orchestration into the model
- finalize experiment management, callbacks, or trainer configuration

Instead, the goal was to make the model itself Lightning-ready while leaving
trainer/bootstrap concerns for a later step.

## Files Updated

- `src/models/fused_model.py`
- `src/models/tft.py`
- `src/models/grn.py`
- `src/models/nn_head.py`
- `src/models/tcn.py`
- `src/utils/config.py`
- `tests/test_fused_model.py`
- `tests/test_config.py`
- `README.md`

## Main Model-Side Changes

### 1. `FusedModel` now acts as a fuller `LightningModule`

`FusedModel` already inherited from `LightningModule`, but it previously
behaved mostly like a plain architecture wrapper with a custom `forward(...)`.

This pass added the missing model-side Lightning responsibilities:

- `quantile_loss(...)`
- `training_step(...)`
- `validation_step(...)`
- `test_step(...)`
- `predict_step(...)`
- `configure_optimizers(...)`
- shared metric logging through `_shared_step(...)` and `_log_metrics(...)`

### Lightning alignment

This follows the tutorial's basic Lightning refactor pattern:

- architecture remains in `forward(...)`
- supervision logic moves into the model
- optimizer configuration moves into the model
- the Trainer later handles the actual training loop

That division keeps `train.py` small and focused on orchestration rather than
rebuilding model-specific optimization logic elsewhere.

## Quantile-Loss Integration

The fused model predicts quantiles, so this pass added a pinball-loss based
`quantile_loss(...)` method.

### Implementation details

- the model now caches the configured quantile tuple
- loss computation validates the expected output shape
- targets are normalized to `[batch, horizon]`
- point-metric reporting uses a representative quantile channel through
  `point_prediction(...)`

### Design rationale

The output semantics of the fused model are probabilistic, not single-value
regression. The interpretation of those output channels belongs to the model's
own supervision contract.

Keeping the quantile loss on the model ensures:

- the configured quantile ordering is defined in one place
- training, validation, and testing all supervise the same output contract
- future trainer code does not need model-specific loss branching

## Shared Train/Val/Test Step Logic

The model now uses a shared internal step helper instead of duplicating the
same logic three times.

### Implementation details

`_shared_step(...)` now:

- runs the forward pass
- normalizes the target shape
- computes quantile loss
- computes MAE/RMSE from a point forecast
- logs stage-specific metrics

### Design rationale

This keeps Lightning hook methods small and readable while ensuring:

- training, validation, and test stay behaviorally aligned
- future metric changes happen in one place
- the model follows the tutorial's "organize, do not duplicate" spirit

## Optimizer Configuration in the Model

The fused model now accepts:

- `learning_rate`
- `weight_decay`
- `optimizer_name`

and uses them in `configure_optimizers(...)`.

### Design rationale

The Lightning tutorial explicitly moves optimizer creation into the
`LightningModule`. That change makes the optimizer part of the model's
declared training behavior rather than part of outer loop boilerplate.

This also makes future checkpoint metadata and experiment reconstruction much
clearer because the model carries the optimizer settings it expects.

## Lazy-Parameter Materialization for Lightning

One important follow-up was required because the TFT path still used a lazy
embedding block.

### Compatibility issue

`TemporalFusionTransformer` used `LazyEmbedding`, which could leave
`UninitializedParameter`s in place until the first real forward pass.

In a plain PyTorch script that may be acceptable if the first forward happens
before optimizer creation. In Lightning, however, `configure_optimizers()` is
expected to operate over a fully materialized parameter set.

### Implementation details

`FusedModel` now proactively materializes the TFT lazy embedding parameters
during model construction using a tiny synthetic grouped input whose widths are
derived from the already-bound config.

### Design rationale

This makes model construction deterministic and Lightning-safe:

- the optimizer no longer depends on a first training batch side effect
- model parameters are fully realized before trainer setup
- checkpoint reload behavior becomes more predictable

## TFT Input Typing Cleanup

The fused model constructs grouped TFT inputs where some groups may be absent.

### Type-contract issue

`FusedModel._build_tft_inputs(...)` correctly returned optional groups such as:

- `s_cat: None`
- `o_cat: None`
- `o_cont: None`

but `TemporalFusionTransformer.forward_features(...)` had still been typed as
if every group were always a concrete `Tensor`.

This caused a Pylance type error at the fused-model call site.

### Implementation details

`src/models/tft.py` now accepts a read-only grouped input type based on:

- `Mapping[str, Tensor | None]`

and keeps `target` explicitly required at runtime through a defensive check.

### Design rationale

This change better matches the actual grouped-input contract already used by
the model and avoids misleading type annotations that imply nonexistent feature
groups must always be present.

## Checkpoint-Friendly Config Serialization

Another important Lightning follow-up was checkpoint reload ergonomics.

### Checkpointing issue

`FusedModel.__init__` requires a project `Config`, but Lightning checkpoint
reloads work most smoothly when constructor hyperparameters can be restored
from plain saved metadata.

Saving a nested config object directly is not ideal because it includes:

- dataclasses
- `Path` objects
- enum-backed `FeatureSpec` structures

Those are convenient in live Python code but are less portable across notebook
and checkpoint-loading workflows.

### Implementation details

`src/utils/config.py` now includes:

- `config_to_dict(...)`
- `config_from_dict(...)`
- typed helpers for `DataConfig`, `TFTConfig`, and `TCNConfig`
- serialization / deserialization of `FeatureSpec`

`FusedModel` now:

- accepts either a typed `Config` or a serialized config mapping
- stores the serialized config in `save_hyperparameters(...)`
- reconstructs the typed config through `_coerce_config(...)`

### Design rationale

This improves Lightning and Colab compatibility:

- `load_from_checkpoint(...)` can rebuild the model without manually injecting
  `config`
- checkpoint metadata stays plain and portable
- notebook workflows become simpler and less error-prone

## Distributed Logging Improvement

The logging path was also adjusted for better Lightning behavior in distributed
setups.

### Implementation details

`sync_dist=True` is now enabled for validation and test logs.

### Design rationale

In distributed Lightning runs, unsynchronized scalar logs can reflect only a
single process's local shard. Validation and test metrics are usually treated
as run-level results, so they should be globally reduced.

Training-step logs were intentionally left unsynchronized to avoid adding
unnecessary per-step distributed overhead for progress reporting.

## Import and Colab Compatibility Follow-Up

The model imports were also cleaned up to be more consistent with the
repository's `src/`-based import style.

### Implementation details

- model files now use `from models...` style imports consistently where needed
- the README Colab section now uses `sys.path.insert(0, ...)`
- the README now shows direct checkpoint loading with `FusedModel.load_from_checkpoint(...)`

### Design rationale

This keeps:

- local pytest usage
- notebook usage
- Google Colab usage
- checkpoint reload usage

more consistent under the same `src` import-root convention.

## Comment and Documentation Cleanup

This pass also cleaned up documentation style inside the model files.

### Implementation details

- new inline comments were added around Lightning integration points
- comment headings were normalized away from conversational "Why this..."
  phrasing toward neutral headings such as:
  - `Supervision contract:`
  - `Architectural role:`
  - `Design note:`
  - `Normalization choice:`
  - `Forecasting constraint:`
  - `Block structure:`
- the top AI-maintenance note in `src/utils/config.py` was reformatted to match
  the style used in `src/models/`

### Design rationale

The model layer has become more complex due to Lightning integration, lazy
parameter handling, and checkpoint config serialization. The extra comments are
intended to preserve reasoning near the code so later refactors do not have to
reverse-engineer design intent from commit history.

## Test Coverage Added

New tests were added to protect the newer Lightning-oriented behavior.

### `tests/test_fused_model.py`

Coverage now includes:

- fused forward output shape
- quantile-loss behavior through `training_step(...)`
- median/point forecast selection
- optimizer wiring
- lazy-parameter materialization at init time
- construction from serialized checkpoint-style config payloads

### `tests/test_config.py`

Coverage now includes:

- round-tripping the top-level config through a checkpoint-friendly plain dict

### Coverage rationale

These additions protect the exact areas most likely to regress during future
trainer or Colab work:

- model initialization
- checkpoint reload behavior
- config serialization
- Lightning hook expectations

## What Is Still Intentionally Left for `train.py`

This pass did not move trainer/bootstrap concerns into the model.

Those remain appropriate for a future `train.py`, including:

- `Trainer(...)` construction
- accelerator/device selection
- callbacks
- checkpoint directory policy
- logger setup
- precision settings
- early stopping
- learning-rate scheduling, if desired

That separation is intentional. The model now owns model-specific behavior; the
trainer script can remain focused on run orchestration.

## Current Outcome

After this pass, the repository has:

- a stronger Lightning-oriented fused model
- better type accuracy around optional TFT inputs
- a safer lazy-parameter path for Lightning optimizer setup
- checkpoint-friendly config serialization
- improved Colab reload ergonomics
- clearer inline documentation around the new behavior

In short, the model layer is now much closer to the organization advocated by
the Lightning tutorial:

- data in the `DataModule`
- model behavior in the `LightningModule`
- trainer boilerplate deferred to the future training entry point
