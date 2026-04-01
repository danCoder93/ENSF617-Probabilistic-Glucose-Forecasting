# Entrypoint And Defaults Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the work that introduced a repository-level runnable
entrypoint, a shared defaults module, a notebook entry surface, and the
related documentation/tooling updates.

This document summarizes the pass that added:

- a root-level `main.py`
- a root-level `main.ipynb`
- a shared root-level `defaults.py`
- supporting tests and typing configuration updates
- detailed comments and disclaimers around the new entrypoint layer

Later follow-up note:
the current top-level workflow has since grown richer post-run observability
and structured evaluation outputs. This document has been updated where needed
so its artifact descriptions reflect the current `main.py` behavior.

## Goals

- Make the repository runnable from a single script entrypoint.
- Provide an equivalent notebook workflow that uses the same underlying Python
  logic as the script rather than maintaining a second copy of the training
  flow.
- Centralize baseline data/model/training/checkpoint defaults in one file so
  the script and notebook do not drift.
- Record run artifacts in a lightweight, inspectable way.
- Make the new top-level workflow easier to understand through comments,
  disclaimers, and explicit documentation.

## Files Added or Updated

Files added:

- `main.py`
- `main.ipynb`
- `defaults.py`
- `tests/test_main.py`
- `docs/entrypoint_defaults_summary.md`

Files updated:

- `src/train.py`
- `tests/test_train.py`
- `pyrightconfig.json`
- `README.md`

## Main Architectural Addition

The repository now has a clearer top-level execution stack:

1. `defaults.py`
   provides shared baseline config builders and entrypoint constants
2. `main.py`
   turns CLI arguments into typed config objects and runs the full workflow
3. `main.ipynb`
   calls the same workflow helpers in a notebook-friendly way
4. `src/train.py`
   remains the reusable Lightning orchestration layer beneath both entrypoints

This preserves the intended responsibility split:

- `FusedModel` owns model-side learning behavior
- `AZT1DDataModule` owns data preparation and loaders
- `FusedModelTrainer` owns Lightning fit/test/predict orchestration
- `main.py` and `main.ipynb` own user-facing run setup and artifact reporting

## `defaults.py`

The new `defaults.py` exists so the repository has one shared source of truth
for baseline experiment settings.

It currently provides:

- `DEFAULT_AZT1D_URL`
- `DEFAULT_OUTPUT_DIR`
- `build_default_config(...)`
- `build_default_train_config(...)`
- `build_default_snapshot_config(...)`

### Why this file was needed

Before this pass, the project already had a reusable training wrapper, but the
top-level runnable defaults still needed a home.

Keeping those builders in `main.py` alone would have made the notebook either:

- duplicate the same setup logic, or
- import large pieces of script-only code just to access the defaults

Separating them into `defaults.py` keeps:

- the CLI simpler
- the notebook simpler
- the default run policy explicit and centrally editable

### Default-policy intent

The defaults are explicitly documented as:

- convenience-oriented research defaults
- baseline settings to make the repository runnable end to end
- not claims of optimal performance
- not a clinically validated forecasting configuration

## `main.py`

The new root-level `main.py` is the main repository entrypoint for running the
full training/evaluation workflow.

### What it does

`main.py` now:

- adds the repository import bootstrap it needs through the shared defaults
  module
- parses common experiment flags from the command line
- builds typed project config objects
- constructs the `AZT1DDataModule`
- constructs the `FusedModelTrainer`
- runs fit/test/predict through one shared function
- writes lightweight run artifacts to disk

### Shared workflow function

The most important implementation detail is that `main.py` does not keep all
logic trapped inside `argparse` handling.

Instead, it exposes `run_training_workflow(...)`, which:

1. optionally seeds the run
2. creates the output directory
3. builds the DataModule and trainer wrapper
4. fits the model
5. optionally evaluates on the test split
6. optionally collects raw test predictions
7. optionally computes structured detailed held-out evaluation from the raw
   predictions plus aligned source batches
8. writes a compact JSON summary

That function is reusable from:

- the CLI entrypoint
- the notebook
- tests

This makes the script and notebook meaningfully share code rather than only
share concepts.

### Artifact outputs

The top-level workflow can now produce:

- `run_summary.json`
- `test_predictions.pt`
- `test_predictions.csv`
- Plotly HTML reports
- checkpoint files under the configured output/checkpoint directory

The summary is intentionally lightweight. It records:

- the declared config
- trainer and snapshot config
- optimizer settings
- fit/test availability information
- checkpoint selection information
- prediction batch shapes and saved locations
- structured held-out evaluation when prediction-based evaluation runs

## `main.ipynb`

The new root-level notebook provides the same end-to-end flow in notebook form.

### Notebook design intent

The notebook is intentionally thin:

- one cell bootstraps imports
- one cell exposes editable run variables
- one cell builds typed config objects
- one cell runs the shared workflow
- later cells inspect metrics and raw predictions

This is important because it avoids the common failure mode where notebooks
quietly fork the real project workflow into a second, partially maintained
training path.

## `src/train.py` follow-up change

One small but important follow-up was made in `src/train.py`.

### Fallback when `"best"` does not exist

`FusedModelTrainer.fit_test_predict(...)` now falls back to the current
in-memory weights when:

- the caller requests `eval_ckpt_path="best"`
- but the fit run did not actually produce a validation-ranked best checkpoint

This matters for the new entrypoint layer because a top-level script or
notebook should still be able to complete the common:

- fit
- test
- predict

workflow even when no meaningful `"best"` checkpoint exists.

Without that fallback, the new one-command workflow would be more fragile in
small-data or no-validation scenarios.

## Tests Added Or Updated

This pass added `tests/test_main.py` and extended `tests/test_train.py`.

### `tests/test_main.py`

The new main-entrypoint test coverage checks that:

- the top-level workflow can run through a fake trainer surface
- `"best"` evaluation falls back to in-memory weights when no best checkpoint
  exists
- summary, prediction, and structured evaluation artifacts are written

### `tests/test_train.py`

The training-wrapper tests now also protect the new fallback behavior in
`fit_test_predict(...)`.

## Typing / Tooling Update

The new root-level files surfaced a tooling issue:

- Pylance/Pyright could resolve imports under `src/`
- but root-level modules like `defaults.py` were not part of the configured
  analysis path

To fix that, `pyrightconfig.json` was updated to:

- include `main.py`
- include `defaults.py`
- add `"."` to `extraPaths`

This keeps editor diagnostics aligned with the actual new repository layout.

## Commenting And Disclaimers

This pass also intentionally added more detailed comments around the new
entrypoint layer.

The goal was not to add decorative comments, but to explain:

- why the defaults are shaped the way they are
- how CLI parsing maps to Lightning config expectations
- why some values are normalized from strings into typed config fields
- why prediction tensors are saved in raw form
- why the notebook is kept thin
- what the top-level disclaimers are trying to communicate

The top-level comments in `defaults.py` and `main.py` also explicitly state
that this entrypoint layer is for research/development workflows and not for
clinical decision-making.

## README Update

The README now includes:

- usage notes for `main.py`
- usage notes for `main.ipynb`
- a refactor-notes link to this new summary document

The README also now points users at the current repository test layout rather
than an older `tests/data/` path that no longer matches the tree.

That makes the new entrypoints discoverable from the repository landing page.

## Practical Usage

### Script

```bash
python main.py --max-epochs 5 --batch-size 32
```

### Notebook

```bash
jupyter notebook main.ipynb
```

### Shared Python surface

```python
from defaults import build_default_config, build_default_snapshot_config, build_default_train_config
from main import run_training_workflow

config = build_default_config()
train_config = build_default_train_config()
snapshot_config = build_default_snapshot_config()

artifacts = run_training_workflow(
    config,
    train_config=train_config,
    snapshot_config=snapshot_config,
)
```

## Summary

This pass did not replace the existing `src/train.py` wrapper.

Instead, it added the missing user-facing top layer above it:

- shared defaults
- a runnable script
- a runnable notebook
- artifact writing
- supporting tests
- editor configuration updates
- more explicit inline documentation

Together, those changes make the repository easier to run, easier to read, and
less likely to drift between script, notebook, and test workflows.
