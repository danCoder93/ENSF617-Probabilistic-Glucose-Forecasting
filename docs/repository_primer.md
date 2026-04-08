# Repository Primer

Role: Hub for the repository's full primer path.
Audience: Readers who want a structured, chaptered primer rather than one large
continuous document.
Owns: Primer reading order, chapter map, and the handoff to the preserved full
primer.
Related docs: [`../README.md`](../README.md),
[`system_walkthrough.md`](system_walkthrough.md),
[`current_architecture.md`](current_architecture.md),
[`primer/full_primer.md`](primer/full_primer.md).

This file is the entry point into the repository primer.

The primer is now organized as a small set of focused chapter reads so that
someone can understand the system without starting from one 2000-line document.
If you want the old continuous long-form treatment, it is still preserved in
[`primer/full_primer.md`](primer/full_primer.md).

Important disclaimer: this repository is a research codebase for probabilistic
glucose forecasting. It is not a clinically validated medical decision-support
system.

## Recommended Primer Order

1. [`primer/problem_and_design.md`](primer/problem_and_design.md)
2. [`primer/repository_structure.md`](primer/repository_structure.md)
3. [`primer/runtime_and_entrypoints.md`](primer/runtime_and_entrypoints.md)
4. [`primer/data_pipeline_walkthrough.md`](primer/data_pipeline_walkthrough.md)
5. [`primer/model_and_training_walkthrough.md`](primer/model_and_training_walkthrough.md)
6. [`primer/evaluation_reporting_walkthrough.md`](primer/evaluation_reporting_walkthrough.md)

That sequence is the full chaptered primer path.

## Chapter Guide

### Problem and design

[`primer/problem_and_design.md`](primer/problem_and_design.md) explains:

- the forecasting problem the repository is trying to solve
- why uncertainty-aware forecasting matters here
- the architectural ideas that shape the rest of the codebase

### Repository structure

[`primer/repository_structure.md`](primer/repository_structure.md) explains:

- how the major packages fit together
- how the execution lifecycle is staged
- how to navigate the repository as a system

### Runtime and entrypoints

[`primer/runtime_and_entrypoints.md`](primer/runtime_and_entrypoints.md)
explains:

- how `main.py`, `defaults.py`, CLI parsing, and runtime profiles fit together

### Data pipeline

[`primer/data_pipeline_walkthrough.md`](primer/data_pipeline_walkthrough.md)
explains:

- how raw AZT1D data becomes grouped model-facing input

### Model and training

[`primer/model_and_training_walkthrough.md`](primer/model_and_training_walkthrough.md)
explains:

- the fused TCN + TFT design
- why runtime-bound model binding exists
- what training is actually optimizing

### Evaluation, reporting, and artifacts

[`primer/evaluation_reporting_walkthrough.md`](primer/evaluation_reporting_walkthrough.md)
explains:

- how the repository evaluates a run
- how artifacts are packaged
- how to inspect outputs afterward

## Preserved Full Primer

The previous continuous primer is still available as:

- [`primer/full_primer.md`](primer/full_primer.md)

Use it when you genuinely want a book-like read rather than the chaptered
primer path above.
