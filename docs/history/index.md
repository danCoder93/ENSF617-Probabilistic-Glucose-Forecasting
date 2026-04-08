# History Index

Role: Archive navigation index for milestone documentation.
Audience: Maintainers and researchers who need historical change context.
Owns: Categorized map of milestone summaries and when to read them.
Related docs: [`../codebase_evolution.md`](../codebase_evolution.md), [`../current_architecture.md`](../current_architecture.md).

Use this index when you need subsystem-specific historical detail that sits behind
`codebase_evolution.md`.

## Data

- [`data_refactor_summary.md`](data_refactor_summary.md): Explains how the data pipeline was split into cleaner layers and aligned with the model-facing contract.

## Model And Training

- [`model_refactor_summary.md`](model_refactor_summary.md): Captures the fused-model and TCN/TFT refactor details.
- [`lightning_model_integration_summary.md`](lightning_model_integration_summary.md): Documents the LightningModule integration pass and design tradeoffs.
- [`train_wrapper_summary.md`](train_wrapper_summary.md): Covers the reusable training wrapper introduction and behavior.
- [`entrypoint_defaults_summary.md`](entrypoint_defaults_summary.md): Describes root-level defaults and entrypoint policy decisions.

## Evaluation, Reporting, And Observability

- [`evaluation_package_summary.md`](evaluation_package_summary.md): Explains the dedicated evaluation package extraction.
- [`observability_integration_summary.md`](observability_integration_summary.md): Documents first-pass observability/reporting integration.
- [`observability_package_refactor_summary.md`](observability_package_refactor_summary.md): Covers the later observability package split.

## Environment And Runtime

- [`environment_runtime_profiles_summary.md`](environment_runtime_profiles_summary.md): Introduces runtime profiles, diagnostics, and environment-aware tuning.
- [`test_layout_and_runtime_modernization_summary.md`](test_layout_and_runtime_modernization_summary.md): Captures runtime modernization and test-layout follow-up.

## Documentation And Commenting

- [`source_refactor_and_documentation_update.md`](source_refactor_and_documentation_update.md): Records config refactor and source documentation cleanup wave.
- [`commenting_conventions_summary.md`](commenting_conventions_summary.md): Defines code-comment and docstring conventions adopted in the repo.
- [`comments_disclaimer_evaluation_2026-04-01.md`](comments_disclaimer_evaluation_2026-04-01.md): Stores the rubric-based comment/disclaimer audit snapshot.
