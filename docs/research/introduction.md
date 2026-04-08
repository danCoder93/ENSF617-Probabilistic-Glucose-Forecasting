# Introduction

Role: Canonical introduction section for the research companion.
Audience: Researchers and collaborators orienting themselves to the problem,
motivation, and contribution shape.
Owns: Problem framing, scope, and why this repository matters as a research
artifact.
Related docs: [`abstract.md`](abstract.md),
[`materials_and_methods.md`](materials_and_methods.md),
[`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md).

## Problem Framing

The repository targets probabilistic glucose forecasting from multivariate
time-series data. Instead of asking only for a single best future glucose
estimate, it asks for a horizon of future predictions together with uncertainty
information expressed through quantiles.

That framing matters because glucose dynamics are uncertain and influenced by
multiple interacting signals, including historical glucose, insulin-related
inputs, carbohydrate events, and device-state context.

## Why This Repository Is Research-Relevant

At the repository level, the project is trying to combine:

- a hybrid TCN + TFT model for probabilistic forecasting
- a semantic data contract rather than a flat anonymous input tensor
- runtime-aware execution across multiple environments
- a stronger artifact and interpretation surface than a minimal training script

This means the repository is not only a model implementation. It is also an
attempt to become a reproducible and inspectable research artifact.

## Scope Note

This introduction is the clean research-facing version. For richer seed prose,
legacy framing, and paper-writing prompts, continue to:

- [`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md)
