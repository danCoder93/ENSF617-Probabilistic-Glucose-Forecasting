# Introduction

Role: Canonical introduction section for the research companion.
Audience: Researchers and collaborators orienting themselves to the problem,
motivation, and contribution shape.
Owns: Problem framing, scientific objective, and the high-level forecasting lens
for the repository.
Related docs: [`abstract.md`](abstract.md), [`dataset.md`](dataset.md),
[`methods.md`](methods.md),
[`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md).

## Problem Framing

The repository targets probabilistic glucose forecasting from multivariate
time-series data. Instead of asking only for a single best future glucose
estimate, it asks for a horizon of future predictions together with uncertainty
information expressed through quantiles.

At a classical machine-learning level, the repository is still solving a
supervised learning problem with inputs \(X\), a target \(Y\), and a model
that learns a function \(f(X)\) to predict \(Y\). What changes is the structure
of both the input and the output:

- the input is a structured history window rather than one flat row
- the output is a future sequence rather than one scalar
- the prediction is probabilistic rather than point-only

This makes the current branch best understood as a probabilistic,
multi-horizon sequence-forecasting system rather than as a conventional
tabular regressor.

## From Point Prediction To Probabilistic Forecasting

Ordinary regression usually aims to produce one central answer. A
quantile-based forecaster instead produces several conditional cutoffs of the
same future target.

For example, a point model might say:

- future glucose = 120

The current model may instead say:

- \(q_{0.1} = 100\)
- \(q_{0.5} = 120\)
- \(q_{0.9} = 145\)

This does not introduce three different clinical targets. It gives three
distributional views of the same target: a lower conditional cutoff, a median,
and an upper conditional cutoff. That matters because glucose dynamics are
inherently uncertain. Meals may be partially logged, correction boluses may
have delayed effects, device modes may change, and human behavior is never
perfectly measured. A probabilistic output therefore fits the problem better
than a single deterministic scalar.

## Scientific Objective

At the repository level, the project is implementing a probabilistic,
multi-horizon blood-glucose forecasting system built around a late-fused TCN +
TFT architecture. The scientific question can be stated as follows:

given a historical sequence of glucose and related covariates for a subject,
estimate the future glucose trajectory over a prediction horizon while
preserving enough information about uncertainty that the system can report not
only a central estimate but also lower- and upper-plausible outcomes.

The repository therefore sits at the intersection of three ideas:

1. physiologically informed multimodal glucose forecasting
2. multi-horizon sequence modeling
3. probabilistic forecasting through quantile prediction

## Forecast Horizon And Experimental Scale

The forecasting question is sequence-based. Let a subject's cleaned timeline be
a multivariate sequence \(x_1, x_2, \dots, x_T\), where each \(x_t\) contains
glucose information, insulin-related signals, carbohydrate intake,
time-derived covariates, device-state covariates, and subject identity
information.

Given an encoder history of length \(L_e\), the model predicts the next
\(L_d\) target values:

\[
y_{t+1}, y_{t+2}, \dots, y_{t+L_d},
\]

where \(y_t\) is glucose in mg/dL.

In the default configuration:

- encoder length \(L_e = 168\)
- prediction length \(L_d = 12\)
- sampling interval = 5 minutes

So the default sample uses:

- 168 historical steps = 14 hours of history
- 12 forecast steps = 1 hour of future

This default is a design choice, not a dataset-mandated fact.

## Why This Repository Is Research-Relevant

At the repository level, the project is trying to combine:

- a hybrid TCN + TFT model for probabilistic forecasting
- a semantic data contract rather than a flat anonymous input tensor
- runtime-aware execution across multiple environments
- a stronger artifact and interpretation surface than a minimal training script

This means the repository is not only a model implementation. It is also an
attempt to become a reproducible and inspectable research artifact.

## Reading Lenses

Three complementary lenses are useful throughout the research notes:

### Lens 1: Classical regression

Start with many inputs, one target, and then extend from one central estimate
to several conditional cutoffs.

### Lens 2: Conditional distribution

Interpret the quantile outputs as selected percentile cut points of the
conditional future-glucose distribution.

### Lens 3: Forecast interval

Interpret lower and upper quantiles as an uncertainty interval over each future
timestep, with the middle quantile acting as the median forecast.

These lenses are not competitors. They describe the same model from different
levels of abstraction.

## Scope Note

This introduction is the research-facing overview. For the dataset and
preprocessing story, continue to [`dataset.md`](dataset.md). For the model,
loss, training, and runtime story, continue to [`methods.md`](methods.md). For
richer seed prose, legacy framing, and paper-writing prompts, continue to:

- [`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md)
