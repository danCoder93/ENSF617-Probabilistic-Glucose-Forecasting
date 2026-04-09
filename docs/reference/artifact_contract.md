# Artifact Contract

Role: Focused current-state reference for evaluation, observability, reporting,
and artifact outputs.
Audience: Engineers, analysts, contributors, and researchers diagnosing or
consuming run outputs.
Owns: Artifact flow, output locations, and the distinction between runtime
observability, evaluation, and post-run reporting.
Related docs: [`../current_architecture.md`](../current_architecture.md),
[`../artifact_diagnosis.md`](../artifact_diagnosis.md),
[`../research/results.md`](../research/results.md),
[`../research/discussion.md`](../research/discussion.md),
[`current_architecture_reference.md`](current_architecture_reference.md).

## Runtime Artifact Flow

The runtime artifact flow is:

1. `main.py` creates or chooses the output directory
2. `defaults.py` derives default artifact paths under that directory
3. runtime observability assembles logger, profiler, and text-log objects
4. Lightning callbacks emit training-time diagnostics during the run
5. the workflow can save raw prediction tensors after prediction
6. reporting exports can write flat prediction CSV and structured report bundles
7. the workflow writes `run_summary.json` with config, runtime, evaluation, and
   artifact metadata

Not all artifacts are produced at the same lifecycle stage. Some exist during
training, while others only exist after prediction has completed.

## Evaluation, Observability, And Reporting

The repository keeps these layers separate on purpose.

### Evaluation

Evaluation owns the canonical computation of model-quality metrics, including:

- scalar metrics such as MAE, RMSE, bias, pinball loss, interval width, and
  empirical coverage
- grouped metrics by horizon, subject, and glucose range

### Observability

Observability is about visibility during the run:

- TensorBoard or CSV logger setup
- text logging
- profiler setup
- callback-driven telemetry
- parameter and gradient monitoring
- prediction figure generation during training/evaluation

### Reporting

Reporting is about packaging and rendering what is known after predictions and
grouped evaluation already exist:

- packaging raw predictions and grouped evaluation into a canonical
  `SharedReport`
- exporting flat prediction tables and grouped tables
- emitting structured JSON summaries from the same packaged report
- rendering lightweight post-run HTML reports
- mirroring that same packaged report into TensorBoard as a post-run dashboard

## Artifact Outputs

With the default output directory of `artifacts/main_run/`, the main workflow
can emit:

- `run_summary.json`
- `report_index.json`
- `test_predictions.pt`
- `test_predictions.csv`
- `reports/`
- `reports/artifacts/shared_report/`
- `checkpoints/`
- `logs/`
- `run.log`
- `telemetry.csv`
- `profiler/`
- `model_viz/`

The artifact strategy is intentionally layered:

- raw tensors are preserved for flexible downstream analysis
- flat exports exist for easy plotting and tabular inspection
- structured JSON and grouped CSV exports come from the same shared-report
  package
- HTML and TensorBoard sinks consume that same post-run package
- logs and telemetry capture runtime context around the same run

## Reading A Run

A practical inspection order is:

1. `report_index.json`
2. `run_summary.json`
3. `metrics_summary.json`
4. grouped metric tables
5. prediction exports
6. logs, telemetry, profiler outputs, and TensorBoard surfaces as needed

## Best Companion Reads

- [`../artifact_diagnosis.md`](../artifact_diagnosis.md) for a concrete run
  interpretation case study
- [`../research/results.md`](../research/results.md) for the paper-style
  evidence path
- [`../research/discussion.md`](../research/discussion.md) for the paper-style
  interpretation path
- [`current_architecture_reference.md`](current_architecture_reference.md) for
  the preserved long-form reference
