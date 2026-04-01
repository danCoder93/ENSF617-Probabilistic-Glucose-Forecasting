# Observability Integration Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the work that introduced a repository-level
observability, logging, telemetry, and visualization stack on top of the
existing PyTorch Lightning training flow.

This document focuses on:

- what changed
- why those changes were added
- how the observability pieces fit together
- where each responsibility now lives in the codebase
- what artifacts a run can produce
- what remains intentionally incomplete or best-effort

## High-Level Goal

The goal of this pass was not simply to "add logging."

The real objective was to make the repository much easier to inspect, debug,
and validate while preserving the current Lightning-oriented architecture.

In practical terms, that meant adding visibility at multiple levels:

- run configuration and hyperparameters
- trainer-level callbacks and telemetry
- model-side metrics and prediction summaries
- architecture visualization
- parameter and gradient monitoring
- system telemetry
- post-run prediction exports and visual reports

The intent is to support both:

- normal experiment runs
- deeper debugging sessions where the user needs to understand whether the
  model, data pipeline, or runtime environment is behaving as expected

## Scope Of This Pass

This pass did not redesign the fused model architecture or the data pipeline.

Instead, it built an observability layer around the stack that already existed:

- `FusedModel` remained the model-side LightningModule
- `AZT1DDataModule` remained the data-preparation and loader layer
- `FusedModelTrainer` remained the reusable Trainer orchestration layer
- `main.py` remained the top-level workflow entrypoint

The observability work was intentionally added around those surfaces rather
than by collapsing responsibilities into one file.

## Files Added Or Updated

Files added:

- `src/observability.py`
- `docs/observability_integration_summary.md`

Files updated:

- `requirements.txt`
- `src/config/observability.py`
- `src/config/runtime.py`
- `src/config/model.py`
- `src/config/serde.py`
- `src/config/__init__.py`
- `defaults.py`
- `src/train.py`
- `src/models/fused_model.py`
- `src/models/tft.py`
- `src/models/tcn.py`
- `src/models/grn.py`
- `src/models/nn_head.py`
- `main.py`
- `tests/test_config.py`
- `tests/test_train.py`
- `tests/test_main.py`

## Dependencies Added

The observability stack added or formalized the use of:

- `tensorboard`
- `torchmetrics`
- `plotly`
- `psutil`
- `matplotlib`
- `torchview`

### Dependency intent

Each dependency serves a different visibility function:

- `tensorboard`
  the main live experiment dashboard for scalars, figures, histograms, graph
  views, text, and profiler output
- `torchmetrics`
  stateful metric integration that fits naturally with Lightning logging
- `plotly`
  lightweight interactive HTML reports generated after prediction export
- `psutil`
  CPU and RAM telemetry
- `matplotlib`
  prediction figures that can be pushed into TensorBoard
- `torchview`
  static model-architecture rendering that complements TensorBoard's graph view

### Important dependency disclaimer

The observability stack is intentionally dependency-aware and best-effort.
Optional extras should improve the run when installed, but the core workflow
should still degrade gracefully when some of those extras are absent.

Examples:

- if TensorBoard is not installed, the code can fall back to a CSV logger
- if `torchview` is unavailable, the model-diagram path is skipped
- if `pynvml` is unavailable, GPU utilization percentage is omitted while
  memory telemetry still works

## Main Architectural Addition: `src/observability.py`

The new `src/observability.py` file centralizes the project-level
observability logic so that:

- observability configuration stays explicit
- the training wrapper stays readable
- model code does not have to own every logging concern
- top-level workflow code can reuse the same artifact helpers

This module is the main place where the repository now defines:

- logger construction
- profiler construction
- logger-aware helper utilities
- Lightning callbacks for runtime diagnostics
- prediction-table export
- Plotly report generation

## Logger Strategy

### Primary logger

The preferred logging path is Lightning's native `TensorBoardLogger`.

This is important because Lightning already knows how to route many built-in
signals into TensorBoard naturally:

- `self.log(...)` metrics from the model
- callback-generated metrics
- hyperparameters
- text summaries
- figures
- histograms
- graph logging
- profiler output

### Fallback logger

If TensorBoard support is unavailable, the observability layer can fall back to
Lightning's `CSVLogger`.

This keeps the training workflow usable in stripped-down environments while
still preserving a basic metrics trail.

### Plain-text logging

The observability stack also adds a plain text file logger.

This file logger complements TensorBoard rather than replacing it. It is useful
for:

- lifecycle notes
- warning/failure messages
- batch-audit output
- torchview/graph-rendering failures
- telemetry snapshots that are easier to grep later

## Profiler Integration

Profiler support is now wired through the same observability config surface.

Supported profiler types:

- `simple`
- `advanced`
- `pytorch`

### Design intent

Profiler support is intentionally opt-in because profiling can add noticeable
runtime overhead. The repository now has the plumbing for profiler artifacts,
but the default path does not force profiling on every experiment run.

## Observability Configuration

The main typed runtime observability policy now lives in:

- `ObservabilityConfig` in `src/config/observability.py`

That config includes toggles and controls for:

- logger selection
- text logging
- CSV fallback behavior
- learning-rate monitoring
- device stats
- rich progress bars
- system telemetry
- parameter histograms
- parameter scalar telemetry
- prediction figures
- model graph logging
- model text logging
- torchview rendering
- profiler use
- gradient statistics
- activation statistics
- batch audit behavior
- prediction export behavior
- Plotly report generation
- output paths for logs, reports, profiler traces, telemetry, and torchview
- debug sampling intervals

### Why this config is separate

The observability policy is intentionally not part of the fused model's
checkpointed semantic config. Data and architecture settings describe what the
model is; observability settings describe how a given run should be inspected.

That separation keeps checkpoint semantics cleaner and makes it possible to run
the same model with different observability levels depending on the task:

- baseline experiment
- debug run
- trace-heavy troubleshooting run

## Default Observability Policy

`defaults.py` now provides `build_default_observability_config(...)`.

The default policy is intentionally visibility-friendly without immediately
forcing the heaviest debug hooks.

The defaults aim to provide:

- TensorBoard when available
- text logs
- telemetry CSV output
- Lightning callback telemetry
- parameter summaries
- prediction figures
- model graph and model text
- torchview rendering
- prediction-table export
- Plotly report generation

At the same time, activation hooks remain more conservative and are promoted
more aggressively only in debug-oriented modes.

## Later Structural Update

Since the initial observability integration, the repository has also promoted
its configuration layer into a dedicated `src/config/` package.

That means the observability work now sits alongside a cleaner config surface:

- `src/config/data.py`
- `src/config/model.py`
- `src/config/runtime.py`
- `src/config/observability.py`
- `src/config/serde.py`

This did not change the core observability intent, but it did improve the
ownership boundaries around that intent:

- observability config is no longer framed as a generic utility
- runtime policy and observability policy now live in explicit config modules
- imports now better reflect architectural meaning

The canonical import path is now:

- `from config import ObservabilityConfig`

## Callback Stack

The main observability callback assembly now lives in
`build_observability_callbacks(...)` inside `src/observability.py`.

## TensorBoard And Model Visualization Coverage

The repository now treats TensorBoard as the primary live inspection surface
for the training stack.

That includes:

- scalar metrics from `self.log(...)`
- learning-rate traces
- device stats
- host telemetry
- parameter scalar summaries
- parameter and gradient histograms
- model architecture text
- Lightning graph logging
- optional profiler output
- qualitative prediction figures
- optional `torchview` architecture rendering surfaced back into TensorBoard

`torchview` is intentionally treated as complementary to TensorBoard rather
than as a replacement for it.

The design split is:

- TensorBoard
  the main live experiment dashboard
- torchview
  a more presentation-oriented static architecture artifact

## Prediction Export And Reporting

The observability stack now supports two different kinds of post-run prediction
artifacts:

- raw tensor outputs for PyTorch-native reuse
- flat CSV/HTML artifacts for analysis convenience

This distinction is deliberate:

- raw `.pt` tensors preserve fidelity
- prediction tables and Plotly reports preserve inspectability

The flat prediction table is intentionally one row per forecast horizon step so
that later analysis can group and visualize by:

- subject
- forecast window
- horizon index
- residual
- interval width

without needing to unpack nested tensors first.

### Model visualization callback

`ModelTensorBoardCallback` is responsible for pushing model-oriented views into
TensorBoard.

It covers:

- plain-text architecture via `repr(pl_module)`
- Lightning graph logging via `add_graph(...)`
- optional `torchview` rendering
- optional TensorBoard image/text logging of the torchview output

This work is intentionally best-effort because graph tracing and rendering can
fail depending on:

- dependency availability
- Graphviz availability
- the shape and contents of the sampled batch
- tracing limitations of the underlying modules

Those failures are logged, but they do not intentionally stop training.

### Batch audit callback

`BatchAuditCallback` logs a small number of representative batches with:

- shape information
- dtype information
- device information
- lightweight tensor statistics

This is meant to catch data-contract problems early without drowning the run in
batch-by-batch dumps.

### Gradient statistics callback

`GradientStatsCallback` logs sampled compact summaries such as:

- total gradient norm
- gradient max absolute value
- non-finite gradient parameter count
- parameter norm summaries

This provides a first-pass answer to questions like:

- are gradients exploding?
- are gradients collapsing to zero?
- are non-finite values appearing?

### System telemetry callback

`SystemTelemetryCallback` records:

- CPU percent
- RAM percent
- RAM used in GB
- GPU memory allocated
- GPU memory reserved
- GPU utilization percentage when NVML is available
- current epoch
- global step

It writes that telemetry both to:

- the active logger
- a CSV file for later offline inspection

### Activation statistics callback

`ActivationStatsCallback` attaches sampled forward hooks to major fused-model
blocks and records summary statistics for their outputs.

Current target blocks include:

- `tcn3`
- `tcn5`
- `tcn7`
- `tft`
- `grn`
- `fcn`

This is intentionally a deeper-debug feature rather than baseline always-on
logging.

### Parameter scalar telemetry callback

`ParameterScalarTelemetryCallback` logs compact epoch-level summaries for each
parameter tensor, including:

- mean
- standard deviation
- norm
- max absolute value
- gradient norm
- gradient max absolute value

This is helpful when histograms are too detailed for the question at hand.

### Parameter histogram callback

`ParameterHistogramCallback` pushes full parameter and gradient histograms into
TensorBoard at configurable epoch intervals.

This complements the lighter scalar telemetry with richer distribution detail.

### Prediction figure callback

`PredictionFigureCallback` logs a small number of qualitative forecast figures
into TensorBoard for validation and test batches.

The figures show:

- target curves
- median predictions
- prediction intervals when available

This gives the user a quick visual check of qualitative forecast behavior
without leaving TensorBoard.

## Model-Side Logging Additions

The observability pass also expanded what the model itself logs in
`src/models/fused_model.py`.

In addition to the previously expected loss/metric hooks, the model now logs:

- loss
- MAE
- RMSE
- target mean
- target standard deviation
- median prediction mean
- quantile prediction mean
- prediction interval width

### TorchMetrics integration

The model now prefers `torchmetrics` objects when available for MAE and RMSE,
while still preserving manual fallback computation when the extra dependency is
not present.

That design keeps the model more Lightning-native without making the repository
fragile in reduced environments.

## Hyperparameter Logging

The training wrapper now logs a flattened hyperparameter payload before
training begins.

That payload includes:

- the runtime-bound config
- optimizer settings
- Trainer settings
- snapshot policy
- observability settings

This makes TensorBoard's hparams view more representative of the actual run
contract rather than only the original top-level declarative config object.

## Top-Level Workflow Integration

`main.py` now treats observability as a first-class runtime concern.

The workflow can now:

- build an effective observability config
- pass it into the training wrapper
- surface observability artifacts in the run summary
- export prediction tables
- generate Plotly reports
- report logger, telemetry, profiler, and torchview paths back to the user

## Artifact Outputs

The top-level workflow can now produce, depending on configuration:

- `run_summary.json`
- `test_predictions.pt`
- `test_predictions.csv`
- Plotly HTML reports
- TensorBoard log directory contents
- text run log
- telemetry CSV
- profiler output directory
- torchview render output

### Prediction table export

The prediction-table export is deliberately denormalized into a flat
row-per-horizon table. It includes fields such as:

- batch index
- sample index
- subject ID
- decoder start and end metadata
- timestamp
- horizon index
- target
- quantile prediction columns
- median prediction
- residual
- prediction interval width

This complements the raw `.pt` prediction tensor artifact:

- `.pt` keeps the original tensor structure
- `.csv` is easier for manual inspection, pandas analysis, and plotting

### Plotly reports

The initial automatic Plotly reports include:

- residual histogram
- horizon metrics
- forecast overview

These are intentionally lightweight first-pass reports rather than a complete
experiment-reporting system.

## CLI Surface

`main.py` now exposes observability-oriented CLI switches such as:

- `--observability-mode`
- `--disable-tensorboard`
- `--disable-plot-reports`
- `--disable-system-telemetry`
- `--disable-gradient-stats`
- `--enable-activation-stats`
- `--disable-parameter-histograms`
- `--disable-parameter-scalars`
- `--disable-prediction-figures`
- `--disable-model-graph`
- `--disable-model-text`
- `--disable-torchview`
- `--torchview-depth`
- `--enable-profiler`
- `--profiler-type`

This lets the user control the observability profile from the top-level
entrypoint without directly editing Python config code.

## Tests Updated

The observability work also updated tests so the new config and artifact
surfaces are checked more explicitly.

The updated test coverage now touches:

- observability config validation and path normalization
- callback presence in the training wrapper
- top-level artifact behavior such as prediction table presence

## Current Limitations And Intentional Gaps

The observability stack is much stronger now, but it is not the final
end-state of debugging instrumentation.

Notable remaining gaps include:

- no fail-fast NaN/Inf callback yet
- no full probabilistic calibration/coverage/CRPS reporting yet
- no guarantee that graph tracing or torchview rendering succeeds in every
  environment
- no end-to-end runtime verification in every target environment documented in
  this file

These are not hidden issues; they are known areas for future follow-up.

## Practical Interpretation

After this pass, the repository has a layered observability stack rather than
one monolithic logger:

- Lightning handles native metric and callback logging
- TensorBoard acts as the main live inspection surface
- torchview complements TensorBoard with a static architecture view
- text logging captures lifecycle/debug notes
- telemetry CSV captures runtime system behavior
- prediction tables and Plotly reports support post-run analysis

That combination makes the repository much better suited for:

- local debugging
- experiment inspection
- notebook analysis
- future Colab packaging

without forcing all observability logic directly into the model or into the
top-level script.
