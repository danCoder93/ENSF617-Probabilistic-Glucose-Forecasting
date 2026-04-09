# Repository Structure

Role: Primer chapter for the major source areas and how they fit together.
Audience: Readers who want the package map and navigation logic before drilling
into specific implementation topics.
Owns: Responsibility-based repository structure, lifecycle overview, and primer
navigation guidance.
Related docs: [`../repository_primer.md`](../repository_primer.md),
[`../current_architecture.md`](../current_architecture.md),
[`../reference/package_boundaries.md`](../reference/package_boundaries.md),
[`../history/index.md`](../history/index.md).

## 1. Responsibility-Based Layout

The most useful way to understand the repository is by responsibility:

- root entry surfaces: `main.py`, `defaults.py`, `main.ipynb`
- typed contracts: `src/config/`
- runtime/environment policy: `src/environment/`
- data preparation and semantic contracts: `src/data/`
- model implementation: `src/models/`
- Lightning-oriented training runtime: `src/train.py`
- top-level orchestration: `src/workflows/`
- held-out evaluation: `src/evaluation/`
- live runtime visibility: `src/observability/`
- post-run packaging and exports: `src/reporting/`
- tests mirrored by subsystem: `tests/`

## 2. Execution Lifecycle

The normal execution lifecycle is the operational spine of the codebase:

1. the user launches `python main.py ...`
2. the CLI layer parses arguments and builds typed configuration objects
3. the environment layer detects the host and resolves runtime defaults
4. the workflow layer creates output directories and preflight metadata
5. the DataModule prepares and loads the dataset
6. the DataModule discovers runtime facts such as category cardinalities
7. the trainer wrapper binds those runtime facts into the model config
8. the fused model is instantiated
9. PyTorch Lightning runs the epoch loop
10. the workflow optionally runs held-out test and prediction
11. the evaluation package computes structured metrics from raw predictions
12. observability and reporting surfaces package the run into inspectable
    artifacts

That order is not interchangeable. Later layers depend on facts or artifacts
produced by earlier layers.

## 3. Navigation By Goal

### If you care about top-level control flow

Read:

- [`runtime_and_entrypoints.md`](runtime_and_entrypoints.md)
- [`../reference/runtime_and_config_flow.md`](../reference/runtime_and_config_flow.md)

### If you care about the data contract

Read:

- [`data_pipeline_walkthrough.md`](data_pipeline_walkthrough.md)
- [`../reference/data_and_model_contract.md`](../reference/data_and_model_contract.md)

### If you care about the forecasting model and supervision

Read:

- [`model_and_training_walkthrough.md`](model_and_training_walkthrough.md)
- [`../research/methodology.md`](../research/methodology.md)
- [`../research/results.md`](../research/results.md)
- [`../research/discussion.md`](../research/discussion.md)

### If you care about outputs and interpretation

Read:

- [`evaluation_reporting_walkthrough.md`](evaluation_reporting_walkthrough.md)
- [`../reference/artifact_contract.md`](../reference/artifact_contract.md)
- [`../artifact_diagnosis.md`](../artifact_diagnosis.md)

## 4. Relationship To The Other Layers

- [`../../README.md`](../../README.md) is the front page
- [`../system_walkthrough.md`](../system_walkthrough.md) is the short guided
  second read
- [`../current_architecture.md`](../current_architecture.md) is the current
  reference hub
- [`../history/index.md`](../history/index.md) is the archive navigation layer

## 5. Preserved Full Primer

If you want the previous continuous essay-style read, use:

- [`full_primer.md`](full_primer.md)
