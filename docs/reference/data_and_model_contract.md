# Data And Model Contract

Role: Focused current-state reference for the data pipeline, grouped batch
contract, and fused model surface.
Audience: Engineers, contributors, and researchers who need the exact
data-to-model story without starting in the research-facing methods docs.
Owns: Data lifecycle, grouped batch interface, runtime metadata binding, model
organization, and probabilistic output contract.
Related docs: [`../current_architecture.md`](../current_architecture.md),
[`runtime_and_config_flow.md`](runtime_and_config_flow.md),
[`../research/dataset.md`](../research/dataset.md),
[`../research/methods.md`](../research/methods.md),
[`../research/results_and_discussion.md`](../research/results_and_discussion.md).

## Data Flow

The most important end-to-end data flow in the repository is:

1. raw AZT1D archive is downloaded and extracted
2. the raw export is standardized into one canonical processed CSV
3. the cleaned dataframe is loaded and normalized
4. semantic feature groups are derived through the schema layer
5. split frames and legal sequence windows are built by indexing
6. the Dataset turns each index entry into one structured sample
7. the DataModule wraps those datasets in train/validation/test dataloaders
8. batches flow into `FusedModel` through the grouped batch contract

The important architectural point is that the data is progressively made more
structured as it moves through the system. The repo does not jump directly from
downloaded files to model tensors in one opaque step.

## Data Layer

[`../../src/data/`](../../src/data/) is organized around `AZT1DDataModule`.

Main responsibilities:

- download and extract the raw AZT1D archive
- build one canonical processed CSV from the raw dataset
- define feature groups, category vocabularies, and model-facing schema rules
- load and normalize the processed dataframe
- build legal encoder/decoder windows and split-specific sample indices
- materialize one structured sample per index entry
- own the Lightning data lifecycle and DataLoader creation

## Model-Facing Batch Contract

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

## Runtime Metadata Binding

One of the most important current contracts is that the DataModule owns
data-derived runtime metadata.

After `setup()`, it can provide:

- categorical embedding cardinalities in TFT order
- fallback feature specs when explicit feature specs are absent
- sequence lengths aligned with the actual prepared dataset
- descriptive statistics for the cleaned dataframe and split/window layout

`bind_model_config(...)` returns a new config rather than mutating the original
in place. That keeps the declarative config inspectable and makes the
runtime-bound config explicit.

## Model Layer

[`../../src/models/`](../../src/models/) contains the forecasting
architecture.

The current model is a late-fusion hybrid:

- three TCN branches at kernel sizes `3`, `5`, and `7`
- one TFT branch over grouped static, historical, and future-known inputs
- one post-branch GRN fusion layer
- one final head that emits quantile forecasts

Conceptually, the forward logic is:

1. split encoder inputs into semantically meaningful slices
2. build narrower history-only inputs for the TCN branches
3. build grouped TFT inputs from static features, encoder history, and
   decoder-known future features
4. run the TCN branches and TFT branch separately
5. concatenate those latent features
6. fuse them with a GRN
7. project them through `NNHead` into final quantile outputs

## Probabilistic Output Contract

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

## Where To Go Deeper

- [`../research/dataset.md`](../research/dataset.md)
  for the research-side dataset and preprocessing treatment
- [`../research/methods.md`](../research/methods.md)
  for the paper-style methods path and probabilistic supervision story
- [`../research/results_and_discussion.md`](../research/results_and_discussion.md)
  for the uncertainty and metric interpretation lens
- [`current_architecture_reference.md`](current_architecture_reference.md) for
  the preserved full current-state reference
