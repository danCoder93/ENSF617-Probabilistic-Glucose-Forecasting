# Conclusions

Role: Canonical conclusions section for the research companion.
Audience: Researchers and collaborators who want the closing interpretation and
next-step framing in a paper-style structure.
Owns: Closing synthesis, scientific framing, and future-facing conclusion
guidance.
Related docs: [`results_and_discussion.md`](results_and_discussion.md),
[`methodology.md`](methodology.md),
[`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md).

## Current Closing Framing

The strongest current claim is not merely that the repository implements a
forecasting model. It is that the project is trying to become a layered,
inspectable, artifact-rich research codebase for probabilistic glucose
forecasting.

Its scientific goal is broader than "predict the next glucose value." The
repository is trying to:

1. preserve physiologically meaningful inputs
2. represent different kinds of time-series information according to their
   causal availability
3. use a hybrid model to capture both short-range patterns and structured
   multi-horizon reasoning
4. predict not only what will happen, but how uncertain that forecast is
5. make the whole process reproducible and inspectable

In that sense, the repository is trying to become not just a model
implementation, but a research artifact.

## Concluding Synthesis

The cleanest summary of the current design is:

- the repository solves a sequence-forecasting problem rather than a static
  tabular regression problem
- the target is future glucose over a 12-step horizon in the default
  configuration
- the input data is grouped semantically into static, known, observed, and
  target roles
- the TCN branches use observed continuous inputs plus target history to model
  history-only temporal structure
- the TFT branch uses semantic feature grouping, static context, historical
  context, and future-known inputs to model feature-aware temporal structure
- the model fuses latent horizon-aligned representations rather than already
  decoded branch forecasts
- the final head emits one output channel per quantile at each future timestep
- pinball loss trains those channels to behave as \(q_{0.1}\), \(q_{0.5}\), and
  \(q_{0.9}\)
- point metrics summarize the median-like forecast, while interval width and
  coverage summarize the probabilistic quality of the forecast

Taken together, that makes the current branch simultaneously:

- a regression model
- a probabilistic forecaster
- an interval-producing uncertainty model
- a structured research-software system

## Future Work Themes

The current structure suggests a practical research agenda.

### Data-side improvements

- add missingness masks
- add time-since-event features
- add insulin-on-board approximations
- add carb-on-board approximations
- test alternative imputation policies for basal insulin and device mode

### Split and evaluation improvements

- standardize both within-subject and subject-held-out benchmarks
- report horizon-wise performance
- report performance by glucose range
- include richer calibration diagnostics beyond coarse interval coverage

### Architecture improvements

- ablate TCN-only, TFT-only, and fused variants
- test denser quantile sets
- test subject-ID removal
- test learned or mechanistic future conditioning for the TCN path
- compare late fusion against earlier fusion alternatives

### Reporting improvements

- store explicit shape summaries in run artifacts
- log feature-group cardinalities in run summaries
- emit calibration plots by horizon
- add representation-level diagnostics for each branch

## Conclusion Status

The final manuscript conclusion should remain human-authored. This file exists
so the research companion mirrors a conventional paper structure now, while the
exact final claims, limitations language, and future-facing wording remain
open.

## Follow-On Material

- methods and technical framing: [`methodology.md`](methodology.md)
- evidence and discussion surfaces: [`results_and_discussion.md`](results_and_discussion.md)
- preserved prompts and legacy section seeds:
  [`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md)
