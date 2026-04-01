# Data Refactor Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the AZT1D data-pipeline refactor and related testing and
tooling changes.

This document summarizes the AZT1D data-pipeline refactor completed for the
PyTorch Lightning `DataModule` pattern and the fused TCN + TFT model pipeline.

## Goals

- Stop growing the old all-in-one dataset module.
- Define a clearer data contract for the fused model inputs.
- Separate download, preprocessing, indexing, dataset assembly, and loader
  orchestration into distinct layers.
- Keep the data layer aligned with the shared config and feature schema used by
  the model layer.

## Architectural Changes

The old design mixed multiple responsibilities into one module. The refactor now
separates the data stack into the following layers:

- `src/data/downloader.py`
  Downloads and extracts the raw AZT1D archive. It does not know anything about
  splits, tensors, or model input structure.
- `src/data/preprocessor.py`
  Standardizes the raw AZT1D export into one canonical processed CSV.
- `src/data/schema.py`
  Defines canonical column names, feature groups, category vocabularies, and
  the bridge from `FeatureSpec` to actual dataframe columns.
- `src/data/transforms.py`
  Loads and cleans the processed dataframe, normalizes categories, creates time
  features, and builds shared category maps.
- `src/data/indexing.py`
  Handles split logic, continuity checks, and encoder/decoder window indexing.
- `src/data/dataset.py`
  Implements `AZT1DSequenceDataset`, which assembles one model-ready sample per
  index entry.
- `src/data/datamodule.py`
  Implements `AZT1DDataModule`, which owns `prepare_data()`, `setup()`, and the
  train/validation/test dataloaders.

## Legacy Files Removed

The previous monolithic modules were removed:

- `src/data/azt1d_dataset.py`
- `src/data/combiner.py`
- `src/test.py`

## Batch Contract

The dataset now exposes an explicit structured batch dictionary for the fused
pipeline:

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

This keeps the sample contract readable and matches the semantic grouping used
by the fused TCN + TFT model path.

## Config Alignment

The refactor was designed to work with the existing `DataConfig` and broader
project config layout.

Notable alignment work included:

- using `cache_dir` in the downloader instead of leaving it unused
- using `test_ratio` explicitly in split-boundary calculations
- deriving feature groups from `config.data.features` when available
- keeping documented AZT1D fallback feature groups while the wider project is
  still migrating

## Data-to-Model Bridge

One important follow-up completed during the refactor was closing the runtime
categorical-cardinality gap between the data layer and the TFT config.

`AZT1DDataModule` now exposes:

- `get_tft_categorical_cardinalities()`
- `bind_model_config(config)`

These methods allow the training/bootstrap layer to bind discovered category
sizes and feature metadata into `Config.tft` before constructing the model.

## Testing and Tooling

The refactor also introduced a dedicated pytest-based test layout:

- `tests/conftest.py`
- `tests/support.py`
- `tests/data/test_schema.py`
- `tests/data/test_transforms.py`
- `tests/data/test_indexing.py`
- `tests/data/test_dataset.py`
- `tests/data/test_preprocessor.py`
- `tests/data/test_downloader.py`
- `tests/data/test_datamodule.py`
- `tests/manual_data_smoke.py`

Additional tooling/config support:

- `requirements.txt` updated with missing runtime/test dependencies
- `pyrightconfig.json` added for better `src/` and `tests/` import resolution
- README testing instructions added for local runs and Google Colab

## Compatibility Notes

The refactored data layer is intended to work both locally and in Google Colab.

Current practical notes:

- Colab/notebook usage still needs `src/` added to `sys.path`
- the data layer itself is device-agnostic and does not depend on CUDA-only code
- full end-to-end confidence would still benefit from one integration test that
  runs `AZT1DDataModule` output directly through `FusedModel.forward()`

## Recommended Commit Strategy

Because the repository may contain unrelated in-progress changes, the safest
commit strategy is to group the refactor into a focused commit that includes:

- `src/data/*`
- `tests/*`
- `README.md`
- `requirements.txt`
- `pyrightconfig.json`
- this document

Suggested commit title:

```text
Refactor AZT1D data pipeline into Lightning DataModule architecture
```
