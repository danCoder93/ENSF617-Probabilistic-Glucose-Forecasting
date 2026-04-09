# Evaluation, Reporting, And Artifact Walkthrough

Role: Focused walkthrough of how the repository interprets and packages a run.
Audience: Engineers, contributors, and researchers who care about outputs and
post-run diagnosis.
Owns: Guided explanation of evaluation, observability, reporting, and artifact
surfaces.
Related docs: [`../system_walkthrough.md`](../system_walkthrough.md),
[`../artifact_diagnosis.md`](../artifact_diagnosis.md),
[`../current_architecture.md`](../current_architecture.md),
[`../research/results.md`](../research/results.md),
[`../research/discussion.md`](../research/discussion.md).

## 1. Why This Layer Exists

The repository is deliberately artifact-rich.

It is not satisfied with:

- one scalar metric
- one checkpoint
- one console printout

Instead, it tries to leave behind enough evidence that a run can be understood
later.

## 2. Evaluation Is Bigger Than Training-Time Logging

Training-time metrics are useful, but this repository also performs structured
held-out evaluation after prediction so that it can reason about:

- grouped error patterns
- quantile behavior
- subject-level and horizon-level variation
- exported prediction rows

That is why the evaluation layer is separate from the general training loop.

## 3. Observability Versus Reporting

The repo distinguishes:

- observability: runtime traces, callbacks, logger integration, telemetry,
  profiler, model visualization
- reporting: post-run packaging of predictions and evaluation into stable
  summary/export surfaces

Those two layers are adjacent, but not identical.

## 4. What A Run Can Produce

The default artifact tree can include:

- `run_summary.json`
- `report_index.json`
- `metrics_summary.json`
- grouped CSV tables
- raw prediction tensors and exported prediction tables
- `reports/` outputs
- text logs, telemetry, logger directories, profiler outputs, and model
  visualizations when enabled

## 5. How To Read A Run Quickly

A practical inspection order is:

1. `report_index.json`
2. `run_summary.json`
3. `metrics_summary.json`
4. grouped metric tables
5. prediction exports
6. logs, telemetry, profiler outputs, and TensorBoard surfaces as needed

## 6. Best Next Reads

- For the exact artifact contract:
  [`../current_architecture.md`](../current_architecture.md)
- For a concrete case study:
  [`../artifact_diagnosis.md`](../artifact_diagnosis.md)
- For the research-facing summary of experimental evidence:
  [`../research/results.md`](../research/results.md)
- For the research-facing interpretation of those results:
  [`../research/discussion.md`](../research/discussion.md)
