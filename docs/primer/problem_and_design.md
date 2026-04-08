# Problem And Design

Role: Primer chapter for the forecasting problem framing and the core design
ideas behind the repository.
Audience: Readers who want to understand why the repository is shaped this way
before diving into packages or implementation details.
Owns: Problem framing, system-level scope, and the main architectural
principles.
Related docs: [`../repository_primer.md`](../repository_primer.md),
[`../system_walkthrough.md`](../system_walkthrough.md),
[`../research/introduction.md`](../research/introduction.md),
[`../research/probabilistic_forecasting.md`](../research/probabilistic_forecasting.md).

## 1. Problem Statement

At the highest level, the repository addresses the following problem:

> Given a historical sequence of glucose and related covariates, estimate the
> future glucose trajectory over a forecast horizon while preserving some
> representation of predictive uncertainty rather than emitting only one point
> estimate.

Three characteristics of that statement are essential:

### Sequential nature

This is not an ordinary row-wise supervised learning problem. Each prediction
depends on a history window, future-known covariates, and a forecast horizon.

### Probabilistic target

The model does not only output one predicted glucose value per horizon step. It
emits multiple quantiles, interpreted as low, central, and high forecasts.

### System-level scope

The problem is not simply "implement a neural network." The repository must
also:

- acquire and clean a real dataset
- establish a stable semantic feature contract
- support multiple runtime environments
- leave reproducible artifacts after a run
- support post-run evaluation beyond training-time logging

## 2. System Overview

A useful compressed description of the whole repository is:

> The repository converts a raw glucose dataset into semantically typed
> sequence windows, binds those runtime-discovered data facts into a hybrid
> TCN-TFT forecasting model, delegates the epoch loop to PyTorch Lightning, and
> then performs structured evaluation plus observability- and reporting-driven
> artifact generation from the resulting probabilistic forecasts.

That sentence already contains the most important causal chain:

1. raw files
2. canonical processed data
3. typed sequence windows
4. runtime-bound config
5. model construction
6. Lightning training
7. evaluation and observability artifacts

## 3. Design Principles

The repository's current shape is not accidental. It reflects several recurring
architectural principles:

### Separation of concerns

The code distinguishes between:

- declarative configuration
- runtime environment policy
- data preparation
- model behavior
- training orchestration
- evaluation
- observability
- reporting

### Semantic data contracts

The repository prefers semantically grouped inputs to anonymous raw tensors.
This matters especially for the TFT branch, where variables are treated
differently based on whether they are static, known, observed, or target-like.

### Runtime-bound construction

The repository does not assume that all model-relevant information is known at
startup. Some facts, such as categorical vocabulary sizes and final feature
cardinalities, are only discoverable after the data has been prepared and
inspected.

### Post-run evaluation

Training-time metrics are not treated as the whole evaluation story. The
repository computes richer held-out evaluation after prediction so that raw
quantile outputs, aligned targets, and metadata can all contribute to the final
assessment.

### First-class observability and reporting

The repository treats observability and reporting as distinct but adjacent
concerns:

- observability captures runtime visibility during a run
- reporting packages and renders post-run outputs for later reading

## 4. Best Next Reads

- [`repository_structure.md`](repository_structure.md) for package roles and the
  execution lifecycle
- [`runtime_and_entrypoints.md`](runtime_and_entrypoints.md) for the top-level
  control flow
- [`../research/introduction.md`](../research/introduction.md) for the
  paper-style introduction
