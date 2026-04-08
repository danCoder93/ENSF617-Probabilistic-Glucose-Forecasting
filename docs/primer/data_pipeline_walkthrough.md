# Data Pipeline Walkthrough

Role: Focused walkthrough of how raw AZT1D data becomes model-facing input.
Audience: Readers who need the data story without absorbing the entire methods
section first.
Owns: Data lifecycle, semantic feature contract, and windowing explanation at a
guided level.
Related docs: [`../system_walkthrough.md`](../system_walkthrough.md),
[`../current_architecture.md`](../current_architecture.md),
[`../research/dataset.md`](../research/dataset.md),
[`../research/methods.md`](../research/methods.md).

## 1. Why The Data Layer Matters Here

This repository does not treat the dataset as a flat tensor that happens to be
available already.

The data layer is responsible for:

- acquiring and normalizing AZT1D data
- building a canonical processed table
- assigning semantic roles to features
- splitting timelines safely
- constructing legal encoder/decoder windows
- emitting the grouped batch contract expected by the model

## 2. Raw To Processed

At a high level, the path is:

1. download and extract raw data
2. normalize raw columns into a canonical tabular representation
3. clean and transform the table
4. derive semantic feature groups and categorical vocabularies
5. split the data and index valid windows
6. emit grouped tensors through the dataset and datamodule

That staged shape exists because real-world glucose data is messy and the model
depends on more than one kind of temporal signal.

## 3. Why The Semantic Contract Exists

The repository distinguishes features by causal and modeling role, such as:

- static
- known in advance
- historically observed
- target

This matters especially for the TFT branch, which needs to reason about
different categories of temporal inputs instead of one undifferentiated tensor.

## 4. Why Windowing Is Not Trivial

The model operates on a history window plus a forecast horizon, not on isolated
rows.

That means the data layer must:

- respect encoder length and prediction length
- avoid illegal windows
- keep target alignment correct
- preserve subject/time structure during split and indexing

## 5. Why The DataModule Exists

The DataModule is doing more than DataLoader plumbing. It participates in
runtime binding because some facts needed by the model are only known after the
data has been prepared, such as vocabulary sizes and final grouped feature
surfaces.

That is why the repo does not fully finalize the model contract before the data
layer runs.

## 6. Best Next Reads

- For the concise research-facing version:
  [`../research/dataset.md`](../research/dataset.md)
- For the companion methods story:
  [`../research/methods.md`](../research/methods.md)
- For exact present-state package boundaries:
  [`../current_architecture.md`](../current_architecture.md)
