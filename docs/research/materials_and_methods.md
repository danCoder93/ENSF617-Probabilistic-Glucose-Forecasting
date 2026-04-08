# Materials And Methods

Role: Canonical materials-and-methods section for the research companion.
Audience: Researchers who want the paper-style methods path before the full
technical monographs.
Owns: Condensed data, model, training, and evaluation methodology narrative.
Related docs: [`introduction.md`](introduction.md),
[`results_and_discussion.md`](results_and_discussion.md),
[`methods_exposition.md`](methods_exposition.md),
[`../current_architecture.md`](../current_architecture.md).

## Materials

The current implementation is centered on an AZT1D-oriented preparation path.
The data workflow:

- acquires and normalizes raw data into a canonical processed table
- builds semantically typed feature groups
- constructs encoder/decoder sequence windows
- emits grouped model-facing batches rather than one flat tensor surface

## Methods

The forecasting model combines:

- TCN branches for local temporal pattern extraction
- a TFT branch for feature-aware temporal reasoning
- late fusion
- a quantile output head for probabilistic forecasting

The runtime path is Lightning-oriented, but the repository itself owns the
broader method design around:

- runtime-bound model construction
- device-profile-aware execution
- structured held-out evaluation
- observability and reporting artifacts

## Experimental-Protocol Notes

Some experimental-setup detail is still placeholder-friendly and should remain
human-shaped later. That includes:

- final benchmark framing
- baseline comparison narrative
- manuscript-ready protocol wording

## Deep Method Reads

- Clean research companion path:
  [`index.md`](index.md)
- Full methods monograph:
  [`methods_exposition.md`](methods_exposition.md)
- Probabilistic interpretation lens:
  [`probabilistic_forecasting.md`](probabilistic_forecasting.md)
- Preserved paper prompts:
  [`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md)
