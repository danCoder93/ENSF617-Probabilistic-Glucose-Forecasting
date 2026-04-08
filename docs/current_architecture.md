# Current Architecture

Role: Hub for the repository's current-state technical reference.
Audience: Engineers, contributors, and researchers who want the present system
shape without starting from one monolithic file.
Owns: Reference reading order, topic map, and the handoff into focused
current-state reference docs.
Related docs: [`../README.md`](../README.md),
[`system_walkthrough.md`](system_walkthrough.md),
[`repository_primer.md`](repository_primer.md),
[`codebase_evolution.md`](codebase_evolution.md),
[`reference/current_architecture_reference.md`](reference/current_architecture_reference.md).

This is the entry point into the repository's current-state reference layer.
It replaces the old habit of putting every architectural detail into one very
large file.

The repository still keeps a preserved full-length reference in
[`reference/current_architecture_reference.md`](reference/current_architecture_reference.md),
but the recommended path is now to read the focused reference docs below
instead of starting with one monograph-sized document.

## System Goal

The repository is a research-oriented probabilistic glucose forecasting system
built around a fused TCN + TFT architecture, a Lightning-oriented training
runtime, structured held-out evaluation, a dedicated post-run reporting layer,
and a richer observability surface for inspection and debugging.

At a high level, the system is trying to do five things well:

- prepare AZT1D data into a stable model-facing contract
- train a probabilistic hybrid forecasting model with clear runtime ownership
- compute structured held-out evaluation rather than only scalar test metrics
- package post-run results into one canonical shared-report surface
- leave behind enough logs and artifacts that a run can be understood later

## Recommended Reading Order

For most readers, the best sequence is:

1. [`../README.md`](../README.md) for the front-door view
2. [`system_walkthrough.md`](system_walkthrough.md) for the guided second pass
3. this reference hub for current-state topic routing
4. the focused reference docs below
5. [`codebase_evolution.md`](codebase_evolution.md) and
   [`history/index.md`](history/index.md) when you need historical rationale

## Focused Reference Docs

### Package boundaries and ownership

Use [`reference/package_boundaries.md`](reference/package_boundaries.md) when
you need to answer:

- which package owns which responsibility
- how the source tree is split by concern
- which boundaries are intended to stay stable

### Runtime and configuration flow

Use [`reference/runtime_and_config_flow.md`](reference/runtime_and_config_flow.md)
when you need to answer:

- how a run starts
- where CLI behavior hands off to reusable workflows
- how declarative config becomes runtime-bound config
- where environment policy and trainer policy live

### Data and model contract

Use [`reference/data_and_model_contract.md`](reference/data_and_model_contract.md)
when you need to answer:

- how raw data becomes grouped model input
- what the batch contract looks like
- how the fused model is organized
- why runtime model binding exists

### Artifact and output contract

Use [`reference/artifact_contract.md`](reference/artifact_contract.md) when
you need to answer:

- what a run can write under `artifacts/main_run/`
- how evaluation, observability, and reporting differ
- which artifact surfaces are canonical for later diagnosis

### Extension points and current constraints

Use [`reference/extension_and_constraints.md`](reference/extension_and_constraints.md)
when you need to answer:

- where to start for common modifications
- which subsystem boundaries should remain stable
- which limitations and assumptions are currently intentional

## Visual And Historical Companions

- [`dependency_graphs/summary.md`](dependency_graphs/summary.md) for the static
  import-graph summary
- [`codebase_evolution.md`](codebase_evolution.md) for the historical narrative
- [`history/index.md`](history/index.md) for archived milestone-level detail
- [`artifact_diagnosis.md`](artifact_diagnosis.md) for one concrete case study

## Preserved Full Reference

The previous long-form architecture reference is still available as:

- [`reference/current_architecture_reference.md`](reference/current_architecture_reference.md)

That file remains useful when you want one continuous, exhaustive technical
read. It is no longer the default starting point.
