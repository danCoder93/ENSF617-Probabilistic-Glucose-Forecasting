# Model Refactor Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the current model-folder refactor around the fused TCN +
TFT forecasting path and related documentation asset moves.

This document summarizes the model-side cleanup and integration work completed
for the fused glucose forecasting architecture.

## Goals

- Make the TCN path leaner and more aligned with the actual project use case.
- Remove unused generic TCN library machinery from the local implementation.
- Wire the fused model to the explicit dataset batch contract instead of a
  placeholder raw tensor interface.
- Keep the multi-kernel TCN idea while making the branch behavior easier to
  reason about and document.
- Move architecture images into `docs/` so they live alongside written design
  documentation.

## Architectural Changes

The current model refactor keeps the overall hybrid direction intact while
making the responsibilities of each model file narrower and clearer.

- `src/models/tcn.py`
  Replaced the previous large, library-style TCN implementation with a
  project-specific causal residual forecaster. The new file keeps:
  causal Conv1d, residual temporal blocks, fixed NLC input handling, and a
  small horizon projection head. It removes unused features such as streaming
  buffer management, transposed-convolution decoding, and compatibility paths.
- `src/models/fused_model.py`
  Refactored `FusedModel` to consume the structured batch dictionary emitted by
  the data pipeline. The model now:
  splits encoder history into known / observed / target groups,
  runs three TCN branches with kernel sizes 3, 5, and 7,
  injects the three TCN horizon forecasts into TFT as auxiliary future
  continuous features, and fuses TFT output with raw TCN forecasts before the
  final readout.
- `src/models/nn_head.py`
  Simplified the readout head into a lightweight MLP that maps fused
  horizon-wise features to the final output dimension.
- `src/models/tft.py`
  Kept the core TFT implementation intact, but added a CUDA availability guard
  around `torch.cuda.synchronize()` so the model path is safer on CPU-only
  environments.

## TCN Refactor

The TCN refactor was the main change in this model pass.

The old local `tcn.py` behaved like a generic imported library module. The new
design is intentionally narrower:

- input contract:
  `[batch, encoder_length, num_inputs]`
- output contract:
  `[batch, prediction_length, output_size]`
- backbone:
  stacked residual temporal blocks
- temporal logic:
  causal convolutions only
- branch role:
  one TCN instance forecasts one kernel-scale view of the future

The fused model instantiates three such branches:

- kernel size `3`
- kernel size `5`
- kernel size `7`

while keeping the dilation schedule fixed at:

- `[1, 2, 4]`

This preserves the intended multi-scale temporal pattern extraction while
making the local implementation much easier to inspect and maintain.

## Data-to-Model Alignment

One important outcome of this refactor is that the fused model now follows the
semantic batch contract produced by the refactored data pipeline.

The forward path now works with grouped tensors such as:

```python
{
    "static_categorical": ...,
    "static_continuous": ...,
    "encoder_continuous": ...,
    "encoder_categorical": ...,
    "decoder_known_continuous": ...,
    "decoder_known_categorical": ...,
    "target": ...,
}
```

This replaces the earlier placeholder design where `FusedModel.forward()`
accepted a single raw tensor and still had TODOs around splitting inputs for TCN
and TFT.

## Config Alignment and Cleanup

One important follow-up to the model/data refactor was repairing and narrowing
the shared config layer so it once again matched the current code paths.

The main config-side changes were:

- `src/utils/config.py`
  Cleaned up unresolved merge-conflict markers and removed duplicate /
  contradictory definitions that had accumulated during earlier edits.
- `DataConfig`
  Kept aligned with the refactored data stack and documented again as the
  shared contract for dataset access, preprocessing, split behavior, feature
  schema, and DataLoader settings.
- `TCNConfig`
  Narrowed to the settings actually used by the lean local TCN branch instead
  of preserving a broader generic-library-style surface that the refactored
  implementation no longer supports.
- `TFTConfig`
  Kept aligned with the runtime-bound path used by `AZT1DDataModule` and
  `FusedModel`, including explicit support for sequence lengths, categorical
  cardinalities, and auxiliary future features injected from the TCN branches.
- `src/utils/tft_utils.py`
  Received a small compatibility guard around numpy scalar-type lookups so the
  shared config and feature-schema imports remain usable in environments with a
  partial numpy install.

This cleanup restored `config.py` as one coherent source of truth instead of a
partially merged mix of older and newer contracts.

## Fusion Behavior

The current fused behavior is:

1. `encoder_continuous` history is split into known, observed, and target
   history groups.
2. Observed history plus target history are passed into the TCN branches.
3. Each TCN branch emits a horizon-aligned forecast.
4. Those three TCN forecasts are appended to TFT's future continuous inputs as
   auxiliary forecast features.
5. TFT output and the three raw TCN forecasts are concatenated.
6. The concatenated representation is passed through `GRN` and then through the
   final `NNHead`.

This keeps TFT as the future-aware refinement branch while allowing the TCNs to
contribute explicit short/mid-range forecast signals.

## Documentation and Assets

Model diagrams were moved out of `src/models/` and into `docs/`:

- `docs/FusedModel_architecture.png`
- `docs/TFT_architecture.PNG`
- `docs/Time_Series.jpg`

An additional TCN-specific diagram was added:

- `docs/TCN_architecture.png`

This keeps implementation files and design/reference assets more clearly
separated.

## Documentation Work Inside `tcn.py`

The new `tcn.py` also received a full provenance and architectural comment pass.

That file now documents:

- the TCN branch's role in the fused model
- why the implementation is intentionally narrow
- provenance relative to the original TCN paper, `pytorch-tcn`, and the
  repository's earlier `MarleyTCNClean` TCN work
- where generative AI assistance was involved in documentation/provenance text
- key implementation choices such as causal padding, residual blocks, receptive
  field scaling, and horizon projection

## Verification Notes

Verification completed during this refactor:

- `python -m py_compile` passed for the modified model files
- `python -m py_compile` also passed for the repaired shared config and new
  config test file
- static inspection confirmed the new code paths match the structured batch
  contract used by the data refactor
- a dedicated `tests/test_config.py` file was added to document and protect the
  shared config contract, including:
  older declarative config construction,
  runtime rebinding via `replace(...)`,
  TCN config narrowing,
  and derived TFT metadata counts

Verification still pending in a runtime environment with `torch` installed:

- one end-to-end synthetic forward pass through `FusedModel`
- one integration pass using actual `AZT1DDataModule` batches
- shape checks for the TCN-to-TFT auxiliary future feature bridge
- full pytest execution once the test environment includes `pytest`
