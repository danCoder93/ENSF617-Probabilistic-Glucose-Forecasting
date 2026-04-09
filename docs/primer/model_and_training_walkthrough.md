# Model And Training Walkthrough

Role: Focused walkthrough of the fused model and training path.
Audience: Readers who want the modeling story without jumping immediately into
the research-facing methods docs.
Owns: High-level explanation of model branches, fusion, training loop, and
runtime-bound construction.
Related docs: [`../system_walkthrough.md`](../system_walkthrough.md),
[`../research/methodology.md`](../research/methodology.md),
[`../research/results.md`](../research/results.md),
[`../research/discussion.md`](../research/discussion.md).

## 1. The Core Modeling Choice

The repository uses a fused architecture rather than one single sequence model.

The model combines:

- TCN branches to capture local temporal patterns over multiple receptive
  fields
- a TFT branch to reason over richer structured temporal inputs
- a fusion layer that combines those representations
- a quantile output head that emits probabilistic forecasts

## 2. Why TCN And TFT Together

The design is trying to combine two complementary strengths:

- convolutional temporal pattern extraction from the history
- semantically structured temporal reasoning with future-known information

That is why the repository talks about "hybrid" forecasting rather than simply
"using TFT" or "using TCN."

## 3. Why Runtime-Bound Model Construction Exists

The final model surface depends on facts learned from the prepared data, not
just on static code defaults.

Examples include:

- vocabulary sizes for categorical inputs
- exact grouped input surfaces
- data-driven feature cardinalities

So the repository binds some model details after the data layer has run.

## 4. What Training Actually Optimizes

The training loop is optimizing a probabilistic forecasting objective rather
than a point-only regression loss.

In practice that means:

- the model predicts multiple quantiles per horizon step
- pinball-style supervision matters
- point-style metrics are still useful, but they are only one view of behavior

For the uncertainty interpretation itself, continue to
[`../research/results.md`](../research/results.md) and
[`../research/discussion.md`](../research/discussion.md).

## 5. Where Lightning Fits

Lightning manages the epoch loop once the repository has already done the more
repository-specific work of:

- preparing the data contract
- binding runtime config
- constructing the model correctly
- choosing observability and reporting policy

So Lightning owns the loop mechanics, but not the whole system design.

## 6. Best Next Reads

- For the research-facing methodology narrative:
  [`../research/methodology.md`](../research/methodology.md)
- For the probabilistic interpretation layer:
  [`../research/results.md`](../research/results.md) and
  [`../research/discussion.md`](../research/discussion.md)
