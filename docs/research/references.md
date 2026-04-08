# References And Provenance

Role: Canonical references and provenance section for the research companion.
Audience: Readers who want the source trail behind the research-facing notes.
Owns: Repository source mapping, external literature anchors, and dataset
source pointers.
Related docs: [`index.md`](index.md), [`dataset.md`](dataset.md),
[`methods.md`](methods.md), [`related_work.md`](related_work.md).

## Current Repository-Aligned Sources

The research companion is grounded first in the present repository state. The
most important repository sources examined are:

- `README.md`
- `defaults.py`
- `src/config/data.py`
- `src/config/model.py`
- `src/data/preprocessor.py`
- `src/data/transforms.py`
- `src/data/schema.py`
- `src/data/indexing.py`
- `src/data/dataset.py`
- `src/data/datamodule.py`
- `src/models/tcn.py`
- `src/models/tft.py`
- `src/models/grn.py`
- `src/models/fused_model.py`
- `src/evaluation/metrics.py`
- `src/train.py`
- `src/workflows/training.py`

Those files provide the implementation facts behind the research-facing claims
about the data contract, fused architecture, quantile supervision, and runtime
orchestration.

## Dataset Sources

The current research notes are aligned to the AZT1D dataset materials preserved
in this repository:

- AZT1D dataset materials in [`../publications/azt1d-dataset.pdf`](../publications/azt1d-dataset.pdf)

The repository defaults also point to the public AZT1D release URL through the
data configuration.

## Canonical Literature Anchors

The most important literature anchors for the current implementation are:

1. Lim, Arik, Loeff, and Pfister. Temporal Fusion Transformers for
   Interpretable Multi-horizon Time Series Forecasting.
2. Bai, Kolter, and Koltun. An Empirical Evaluation of Generic Convolutional
   and Recurrent Networks for Sequence Modeling.
3. Ba, Kiros, and Hinton. Layer Normalization.
4. PyTorch Lightning documentation on `LightningDataModule`.

These references matter because the repository's current data and model story
is explicitly organized around TFT-style semantic feature roles, TCN-style
causal temporal convolutions, layer-normalized sequence processing, and
Lightning-oriented runtime structure.

## Bibliography Status

This file is not yet a fully formatted manuscript bibliography. It is a
repository-aligned provenance note and reading map. The final human-authored
paper should eventually convert these materials into the citation style
required by the target venue.

## Suggested Usage

- Use [`dataset.md`](dataset.md) when you want the data provenance and
  preprocessing story.
- Use [`methods.md`](methods.md) when you want the model, loss, and training
  story.
- Use [`related_work.md`](related_work.md) when you want the paper-style
  literature-positioning surface.
