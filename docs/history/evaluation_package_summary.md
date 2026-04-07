# Evaluation Package Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the later follow-up that introduced a dedicated
evaluation package for structured forecasting metrics and detailed held-out
evaluation artifacts.

Later follow-up note:
the automated evaluation tests now live under `tests/evaluation/` rather than
at the repository root. The architectural ownership described below is still
the same; only the test paths were grouped more explicitly later.

## Goal

The goal of this pass was to stop treating detailed evaluation as scattered
model glue and lightweight report-side recomputation.

Instead, the repository now has a clearer split:

- `FusedModel` still owns forward pass, loss, and optimizer behavior
- `src/train.py` still owns Lightning orchestration
- `src/observability/` still owns logging, callbacks, telemetry, and reports
- `src/evaluation/` now owns metric definitions, grouped aggregations, and the
  structured detailed evaluation contract

## Files Added

- `src/evaluation/__init__.py`
- `src/evaluation/types.py`
- `src/evaluation/core.py`
- `src/evaluation/metrics.py`
- `src/evaluation/grouping.py`
- `src/evaluation/evaluator.py`
- `tests/test_evaluation_metrics.py`
- `tests/test_evaluation_evaluator.py`

Later follow-up note:
those evaluation tests now live in `tests/evaluation/test_metrics.py` and
`tests/evaluation/test_evaluator.py`. The list above reflects the layout when
the evaluation package first landed.

## Files Updated

- `src/models/fused_model.py`
- `src/observability/reporting.py`
- `main.py`
- `tests/test_main.py`

Later follow-up note:
the entrypoint/workflow tests discussed here now live under `tests/workflows/`.

## Main Architectural Addition: `src/evaluation/`

The new `src/evaluation/` package is now the canonical home for model-quality
evaluation logic.

It is intentionally split into focused modules:

- `types.py`
  structured contracts such as `EvaluationBatch`, `MetricSummary`,
  `GroupedMetricRow`, and `EvaluationResult`
- `core.py`
  target normalization, point-forecast selection, metadata normalization, and
  evaluation-batch validation
- `metrics.py`
  primitive metrics such as MAE, RMSE, bias, pinball loss, interval width, and
  empirical coverage
- `grouping.py`
  grouped aggregation helpers, including default glucose-range buckets
- `evaluator.py`
  end-to-end evaluation assembly from batch-aligned predictions, targets, and
  metadata

## Runtime Flow Follow-up

### Model-side metric parity helpers

`FusedModel` still logs the same high-level live metrics during training,
validation, and testing, but the shared fallback logic now reuses the new
evaluation helpers for:

- target normalization
- point-prediction selection
- MAE fallback computation
- RMSE fallback computation
- interval-width computation

This keeps the model-side live metric surface familiar while reducing metric
logic duplication.

### Structured held-out evaluation

`main.py` now computes a richer structured detailed evaluation payload after
held-out prediction generation.

That payload is exposed as:

- `MainRunArtifacts.test_evaluation`

and is also persisted inside the run summary under:

- `summary["evaluation"]["test_evaluation"]`

The structured result currently includes:

- scalar summary metrics
- grouped metrics by forecast horizon
- grouped metrics by subject ID
- grouped metrics by glucose range
- per-quantile pinball loss summaries
- interval-width and empirical-coverage summaries when available

## Reporting Follow-up

`src/observability/reporting.py` now has a partial integration with the
structured evaluation layer.

Specifically:

- prediction-table export now uses the shared point-prediction helper from the
  evaluation package
- Plotly horizon metrics can consume the canonical grouped horizon metrics from
  `EvaluationResult` when available
- the existing flat CSV fallback remains in place for lighter workflows

This means the reporting layer is no longer forced to derive every metric from
the exported CSV alone.

## Validation And Guardrails

One important follow-up added during this work was explicit shape validation at
the evaluation boundary.

`build_evaluation_batch(...)` now fails early when:

- predictions are not rank-3 `[batch, horizon, quantiles]`
- target rank does not normalize to `[batch, horizon]`
- prediction batch size and target batch size differ
- prediction horizon and target horizon differ
- prediction quantile width does not match the configured quantile set

This makes evaluation failures clearer and earlier than letting mismatches
surface later during reporting or grouped aggregation.

## Current Limitations

The evaluation architecture is much better structured now, but a few
intentional gaps remain:

- detailed evaluation currently hangs off the prediction path, not the scalar
  `trainer.test(...)` path
- Plotly reports still require the flat prediction table, even when structured
  evaluation exists
- the grouped glucose-range view is intentionally lightweight and should not be
  mistaken for a complete clinical evaluation taxonomy
- richer probabilistic diagnostics such as CRPS or fuller calibration analysis
  are still future work

## Practical Interpretation

After this pass, the repository has a clearer layered story for model-quality
inspection:

- live Lightning metrics for quick feedback
- callback-driven observability for debugging and runtime diagnostics
- flat prediction exports for convenient table/plot workflows
- a canonical structured evaluation result for richer held-out analysis

That combination is a better foundation for future detailed metrics, reporting,
and domain-specific glucose evaluation work than the earlier model-centric
metric path.
