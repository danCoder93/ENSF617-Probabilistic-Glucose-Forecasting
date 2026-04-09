# System Walkthrough

Role: Recommended second-pass walkthrough after the root README.
Audience: Engineers, contributors, reviewers, and researchers who want one
coherent system-level read before diving into specialized docs.
Owns: Stitched end-to-end narrative of how the repository works and where to
read next.
Related docs: [`../README.md`](../README.md),
[`current_architecture.md`](current_architecture.md),
[`repository_primer.md`](repository_primer.md),
[`research/index.md`](research/index.md).

This document is the main follow-on read after the root `README.md`. It is
shorter and more guided than the full primer path exposed through
[`repository_primer.md`](repository_primer.md), but broader than the narrower
reference and research docs.

If you want one answer to "how does this repository actually work as a whole?",
start here.

## 1. What This Repository Is Trying To Do

At its core, this repository is trying to do three things together:

1. train a probabilistic glucose forecasting model rather than a point-only predictor
2. preserve enough runtime and artifact detail that a run can be inspected later
3. keep the implementation layered enough that data, model, training,
   evaluation, observability, and reporting can evolve without collapsing into
   one script

That combination is what makes the repository feel different from a typical
"train a model and print a metric" project.

## 2. The End-To-End Story

A normal run follows this shape:

1. `main.py` exposes the stable runnable entrypoint.
2. `src/workflows/cli.py` turns flat CLI arguments into typed config and
   workflow settings.
3. `src/environment/` resolves the effective runtime profile for the current
   machine or platform.
4. `src/data/` downloads, preprocesses, validates, and windows the AZT1D data
   into the model-facing semantic contract.
5. `src/models/` builds the fused TCN + TFT forecasting model against the
   resolved data contract.
6. `src/train.py` and `src/workflows/training.py` run training, evaluation,
   prediction export, and post-run artifact generation.
7. `src/evaluation/`, `src/observability/`, and `src/reporting/` turn the run
   into summaries, grouped metrics, exports, logs, and report surfaces.

That means the repository is better understood as a staged forecasting system
than as a single model file.

## 3. Why The Layering Exists

The current layout is deliberate.

- `src/config/` holds typed configuration so CLI, defaults, and workflows all
  speak a consistent contract.
- `src/environment/` exists because runtime behavior is machine-sensitive and
  profile-sensitive.
- `src/data/` exists because real-world glucose data requires preparation,
  semantic typing, and safe window construction before model code should see
  it.
- `src/models/` exists because the forecasting architecture itself is already a
  meaningful subsystem, not just a helper function.
- `src/evaluation/`, `src/observability/`, and `src/reporting/` are separate
  because post-run interpretation is a first-class project goal.

## 4. The Main Model Story

The central model combines:

- TCN branches for local temporal pattern extraction
- a TFT branch for feature-aware sequence reasoning
- a late-fusion stage
- a quantile output head for probabilistic forecasting

This matters because the repository is not merely forecasting one future scalar
value. It is forecasting a horizon of future glucose values while trying to
retain uncertainty information through quantiles.

For the deeper model and tensor semantics, continue to:

- [`research/methodology.md`](research/methodology.md)
- [`research/dataset.md`](research/dataset.md)
- [`research/results_and_discussion.md`](research/results_and_discussion.md)

## 5. What A Run Leaves Behind

The workflow is designed to leave more than a checkpoint and a metric.

The main artifact surface is rooted by default at `artifacts/main_run/` and can
include:

- `run_summary.json`
- `report_index.json`
- `metrics_summary.json`
- grouped evaluation CSVs
- raw prediction tensors and exported prediction tables
- `reports/` outputs built from the canonical shared report
- `run.log`, `telemetry.csv`, logger directories, profiler outputs, and model
  visualizations when enabled

For the exact artifact contract, use
[`reference/artifact_contract.md`](reference/artifact_contract.md). For a
concrete example of how to interpret one run, use
[`artifact_diagnosis.md`](artifact_diagnosis.md).

## 6. Reading Paths After This Walkthrough

### If you are here for engineering and reproducibility

- [`execution_guide.md`](execution_guide.md)
- [`cli_reference.md`](cli_reference.md)
- [`current_architecture.md`](current_architecture.md)
- [`reference/artifact_contract.md`](reference/artifact_contract.md)
- [`artifact_diagnosis.md`](artifact_diagnosis.md) for one concrete case study

### If you are here to modify the repository safely

- [`reference/package_boundaries.md`](reference/package_boundaries.md)
- [`reference/runtime_and_config_flow.md`](reference/runtime_and_config_flow.md)
- [`reference/extension_and_constraints.md`](reference/extension_and_constraints.md)

### If you are here for research understanding

- [`research/index.md`](research/index.md)
- [`research/methodology.md`](research/methodology.md)
- [`research/dataset.md`](research/dataset.md)
- [`research/results_and_discussion.md`](research/results_and_discussion.md)
- [`codebase_evolution.md`](codebase_evolution.md)

### If you want the preserved long continuous reads

- [`primer/full_primer.md`](primer/full_primer.md): preserved long-form,
  continuous systems primer
- [`reference/current_architecture_reference.md`](reference/current_architecture_reference.md):
  preserved exhaustive current-state technical reference

## 7. Chapter Map

This walkthrough is backed by smaller chapter docs when you want a focused
second-step explanation without jumping straight into the long monographs.

- [`primer/problem_and_design.md`](primer/problem_and_design.md):
  why the repository is solving this problem and what design principles shape it
- [`primer/repository_structure.md`](primer/repository_structure.md):
  how the major packages fit together and how to navigate them
- [`primer/runtime_and_entrypoints.md`](primer/runtime_and_entrypoints.md):
  how `main.py`, defaults, CLI, workflows, and runtime profiles fit together
- [`primer/data_pipeline_walkthrough.md`](primer/data_pipeline_walkthrough.md):
  how raw data becomes semantically typed model input
- [`primer/model_and_training_walkthrough.md`](primer/model_and_training_walkthrough.md):
  how the fused model is built and trained
- [`primer/evaluation_reporting_walkthrough.md`](primer/evaluation_reporting_walkthrough.md):
  how the repository evaluates, interprets, and exports a run

## 8. Relationship To The Longer Docs

The repository still keeps very long documents because some readers genuinely
want a continuous, essay-style explanation. That depth is being preserved on
purpose.

What changes here is not the availability of depth, but the default reading
experience:

- `README.md` is the fast interface
- this file is the guided system walkthrough
- `repository_primer.md` is the chaptered primer hub
- chapter docs provide manageable next reads
- the chaptered research companion carries the paper-style treatment
- the preserved continuous reads remain available when one long document is
  genuinely the better fit
