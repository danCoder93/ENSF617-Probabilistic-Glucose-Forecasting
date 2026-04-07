# Model Refactor Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the current model-folder refactor around the fused TCN +
TFT forecasting path and related documentation asset moves.

Where AI assistance is mentioned in this document, it refers to user-directed
refinement of existing project code and documentation. The intent was to help
clarify, harden, and document the current codebase rather than to claim new
model ideas or original architectural authorship.

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
  horizon projection path that now supports both fusion-ready latent features
  and branch-local forecast outputs. It removes unused features such as
  streaming buffer management, transposed-convolution decoding, and
  compatibility paths.
- `src/models/fused_model.py`
  Refactored `FusedModel` to consume the structured batch dictionary emitted by
  the data pipeline. The model now:
  splits encoder history into known / observed / target groups,
  runs three TCN branches with kernel sizes 3, 5, and 7,
  keeps the TFT branch focused on its own future-known decoder inputs,
  and fuses TCN branch features with TFT decoder features before the final
  readout.
- `src/models/nn_head.py`
  Evolved from a tiny readout MLP into a stronger residual position-wise
  prediction head with GELU, LayerNorm, dropout, and stacked feed-forward
  blocks so the final fused representation has a more expressive last-stage
  predictor.
- `src/models/tft.py`
  Kept the core TFT implementation intact, added a CUDA availability guard
  around `torch.cuda.synchronize()` so the model path is safer on CPU-only
  environments, and exposed a latent-feature interface so the fused model can
  consume decoder representations before `quantile_proj`.

## TCN Refactor

The TCN refactor was the main change in this model pass.

The old local `tcn.py` behaved like a generic imported library module. The new
design is intentionally narrower:

- input contract:
  `[batch, encoder_length, num_inputs]`
- output contract:
  `forward_features(...) -> [batch, prediction_length, branch_hidden_size]`
  and `forward(...) -> [batch, prediction_length, output_size]`
- backbone:
  stacked residual temporal blocks
- temporal logic:
  causal convolutions only
- branch role:
  one TCN instance encodes one kernel-scale view of the history and contributes
  a horizon-aligned latent feature stream to the fused predictor

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

- `src/config/`
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
  cardinalities, and optional auxiliary future-feature counts when a caller
  chooses to use that TFT capability outside the current default fused path.
- `src/utils/tft_utils.py`
  Received a small compatibility guard around numpy scalar-type lookups so the
  shared config and feature-schema imports remain usable in environments with a
  partial numpy install.

This cleanup restored the config layer as one coherent source of truth instead
of a partially merged mix of older and newer contracts.

## Fusion Behavior

The current fused behavior is:

1. `encoder_continuous` history is split into known, observed, and target
   history groups.
2. Observed history plus target history are passed into the TCN branches.
3. Each TCN branch emits a horizon-aligned latent feature tensor for fusion.
4. TFT processes the structured batch inputs and exposes horizon-aligned
   decoder features before its final quantile projection.
5. TFT features and the three TCN branch feature tensors are concatenated.
6. The concatenated representation is passed through `GRN` and then through the
   final `NNHead`.

This keeps TFT as the future-aware refinement branch while allowing the TCNs to
contribute explicit short/mid-range temporal representations at the same fusion
stage instead of influencing TFT internally.

## Fusion Realignment Follow-up

After the earlier fused-model cleanup, a second follow-up pass aligned the
actual implementation more closely with `docs/assets/FusedModel_architecture.png`.

The key architectural shifts were:

- `src/models/fused_model.py`
  Removed the old TCN-to-TFT auxiliary future bridge from the default fused
  path by binding `num_aux_future_features=0` in the runtime TFT config.
- `src/models/fused_model.py`
  Reworked the fused forward pass so:
  TCN branches read only encoder-history observed+target signals,
  TFT reads only its own proper grouped static / historical / future-known
  inputs,
  and the two branches meet only at the late fusion stage.
- `src/models/tft.py`
  Added `forward_with_features(...)` and `forward_features(...)` so the fused
  model can consume the decoder representation before the final quantile
  projection.
- `src/models/tcn.py`
  Added `summarize(...)` and `forward_features(...)` so each branch can produce
  a horizon-aligned latent feature tensor for fusion, while still preserving a
  standard forecast-producing `forward(...)`.
- `src/models/fused_model.py`
  Updated the post-branch fusion width calculation to reflect:
  one TFT hidden stream plus three TCN branch hidden streams, followed by the
  fusion GRN and the final `NNHead`.

This changed the fused architecture from a late reconciliation layer over
already-decoded TFT outputs into a true latent-space fusion path where TCN and
TFT meet before final prediction.

## GRN Encapsulation Follow-up

After the broader fused-model and config cleanup, a smaller follow-up pass was
completed around `src/models/grn.py` to make the gated residual network easier
to reuse safely across both the TFT internals and the fused model's final
fusion head.

This work was intentionally conservative. It did not redesign the GRN math or
change the model's high-level architecture. Instead, it made the existing GRN
construction path more explicit, more consistent, and better aligned with the
shared config contract already used by the rest of the model stack.

The main changes were:

- `src/config/model.py`
  Added `layer_norm_eps` to `TFTConfig` so TFT-owned normalization behavior,
  including GRN-backed blocks, can share one validated epsilon value rather
  than relying on a hardcoded constant inside `grn.py`.
- `src/models/grn.py`
  Added constructor validation for dimensions, dropout, and normalization
  epsilon so invalid GRN configurations fail early and clearly.
- `src/models/grn.py`
  Added `GRN.from_tft_config(...)` as a narrow factory method. This keeps
  per-call structural dimensions such as `input_size`, `output_size`, and
  `context_hidden_size` explicit at the call site while inheriting shared
  defaults such as `hidden_size`, dropout, and layer-norm epsilon from
  `TFTConfig`.
- `src/models/grn.py`
  Tightened the context-broadcasting behavior so the module supports both:
  rank-2 feature tensors with rank-2 context, and the original TFT pattern of
  rank-3 temporal inputs with rank-2 static context broadcast across time.
  Unsupported rank combinations now fail loudly instead of relying on implicit
  broadcasting assumptions.
- `src/models/tft.py`

### Later structural note

After the earlier config cleanup described above, the repository promoted that
shared config layer into the dedicated `src/config/` package. The behavioral
intent described in this summary is unchanged, but the canonical home of those
config objects is now `src/config/` rather than `src/utils/config.py`.
  Replaced repeated direct GRN construction sites with
  `GRN.from_tft_config(...)` in the variable-selection network, static context
  encoder, enrichment GRN, and position-wise GRN paths.
- `src/models/tft.py`
  Updated local `LayerNorm` construction to use `config.layer_norm_eps` so the
  surrounding TFT normalization layers remain numerically aligned with the
  config-backed GRN path.
- `src/models/fused_model.py`
  Updated the post-fusion GRN construction to use the same config-backed
  factory while keeping the fusion-specific feature width explicit in the fused
  model itself.

One key design decision in this follow-up was *not* adding a separate `GRNConfig`
surface yet. At the time of this refactor, GRN-specific structural dimensions
still vary by call site and the shared defaults already belong naturally to the
TFT branch contract. Introducing a second config object at this stage would
have duplicated fields without adding much practical flexibility.

This leaves the door open for a future `GRNConfig` only if the project later
needs GRN-specific behavior that diverges from the current TFT-level defaults.

## Documentation and Assets

Model diagrams were moved out of `src/models/` and into `docs/`:

- `docs/assets/FusedModel_architecture.png`
- `docs/assets/TFT_architecture.PNG`
- `docs/assets/Time_Series.jpg`

An additional TCN-specific diagram was added:

- `docs/assets/TCN_architecture.png`

This keeps implementation files and design/reference assets more clearly
separated.

## Documentation Standardization

The model files also received a documentation/style cleanup so the top-of-file
provenance and AI-assistance notes follow a pattern closer to the existing
`src/models/tft.py` style:

- provenance / adaptation notes live at the top of the file as comment blocks
- class docstrings focus on architecture role, tensor contracts, and behavior
- inline `#` comments explain implementation logic locally where it matters

This was applied across the active model files:

- `src/models/fused_model.py`
- `src/models/grn.py`
- `src/models/nn_head.py`
- `src/models/tcn.py`
- `src/models/tft.py`

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

## NNHead Follow-up

The final `NNHead` also received a targeted architecture and documentation pass.

That follow-up changed the head from a minimal two-layer MLP into a stronger
position-wise predictor that now:

- projects fused hidden features into a dedicated readout space
- refines them through residual feed-forward blocks
- uses GELU activations, LayerNorm, and dropout for a deeper but still stable
  final-stage predictor
- keeps the head's responsibility narrow:
  `GRN` performs branch fusion, while `NNHead` performs final prediction

The file was also expanded with comments that explain:

- the purpose of the residual feed-forward blocks
- why the head remains position-wise across the horizon
- how the readout stack relates to the upstream fusion GRN
- the default sizing logic for hidden and feed-forward widths

## Verification Notes

Verification completed during this refactor:

- `python -m py_compile` passed for the modified model files
- `python -m py_compile src/models/*.py` passed after the later fusion
  realignment and model-comment cleanup
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
- a dedicated `tests/test_grn.py` file was added to document and protect the
  GRN encapsulation follow-up, including:
  config-backed GRN construction from `TFTConfig`,
  rank-2 input plus context handling,
  rank-3 temporal input plus broadcast context handling,
  and explicit failure for unsupported context-rank combinations

Later follow-up note:
that config/model coverage now lives under `tests/config/` and `tests/models/`.

Verification still pending in a runtime environment with `torch` installed:

- one end-to-end synthetic forward pass through `FusedModel`
- one integration pass using actual `AZT1DDataModule` batches
- shape checks for the new latent-space TCN/TFT fusion path
- full pytest execution once the test environment includes `pytest`
