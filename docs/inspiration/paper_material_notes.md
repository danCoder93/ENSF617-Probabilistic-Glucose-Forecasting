# Paper Material Notes

Role: Inspiration-only workspace for later human-authored research writing.
Audience: Human authors preparing manuscript text, bibliography, and report
narrative.
Owns: Preserved README paper scaffolding, legacy seed prose, literature
tracker prompts, and writing checklists.
Related docs: [`../repository_primer.md`](../repository_primer.md),
[`../research/methods.md`](../research/methods.md),
[`../artifact_diagnosis.md`](../artifact_diagnosis.md),
[`../research/index.md`](../research/index.md),
[`observability_reporting_case_study.md`](observability_reporting_case_study.md),
[`readme_migration_ledger.md`](readme_migration_ledger.md).

This file intentionally preserves research-writing seed material moved out of
the public README. It is not a finalized paper and should not be treated as a
canonical technical reference.

The structured research companion now lives under [`../research/`](../research/).
Use this file as the raw inspiration and placeholder layer behind that cleaner
research-facing surface.

For the exact section-by-section preservation map from the legacy README to the
current documentation system, see
[`readme_migration_ledger.md`](readme_migration_ledger.md).

## Legacy README Framing

Living research-style README for the repository. This file was intended to be
the top-level source of truth for the project's problem framing, architecture
story, implementation provenance, current literature inventory, and paper
draft placeholders.

Important status note:
this was not yet a finished paper manuscript. Several sections were
deliberately written as structured placeholders so the team could keep one
central document updated while experiments, citations, and results continued to
evolve.

Important disclaimer:
this repository is a research codebase for probabilistic glucose forecasting.
It is not a clinically validated decision-support system.

## Current Status Snapshot

What was in good shape:

- the repository had a runnable end-to-end training + evaluation workflow
- the codebase was documented by both hand-authored architecture prose and
  generated dependency graphs
- typed config, runtime profiles, diagnostics, structured evaluation, and
  observability/reporting were all first-class subsystems
- TensorBoard integration was being actively enhanced with richer logging,
  hierarchical views, and interpretable summaries
- reporting supported structured outputs (JSON, CSV) and visual artifacts
  (Plotly), with a direction toward unifying everything under TensorBoard
- prediction-level outputs (per-row, per-horizon) and aggregated summaries were
  available for downstream analysis
- the test suite covered multiple layers beyond just the model itself

What was still in progress:

- consolidation of reporting outputs into a fully TensorBoard-native experience
- improving interpretability of logged metrics (legends, grouping, semantic
  naming)
- expanding data-level statistics logging
- bibliography and related-work sections incomplete
- AZT1D-oriented pipeline

Who that older README was trying to serve best:

- contributors extending observability/reporting
- collaborators converting the repo into a research artifact
- developers improving interpretability and diagnostics

## Environment Requirements Notes

Current practical requirements:

- Python with the packages listed in `requirements.txt`
- a working PyTorch installation appropriate for the target machine
- optional GPU, MPS, or Slurm environment if the corresponding device profiles
  are desired
- Graphviz `dot` if regenerating the static dependency graphs

Notes:

- the repository did not pin one exact Python version in the old README
- Torch installation might need to be environment-specific, especially for CUDA
  or Apple Silicon
- the dependency-graph generator only needs Graphviz when regenerating graph
  assets, not for normal training or inference workflows

## Abstract Seed Material

This repository implements a research-oriented probabilistic glucose
forecasting system built around a hybrid Temporal Convolutional Network (TCN)
plus Temporal Fusion Transformer (TFT) architecture. The current codebase is
organized to support multimodal time-series preparation, runtime-aware
training, structured held-out evaluation, and artifact-rich observability.
Rather than acting only as an engineering project, the repository now aims to
carry its own architecture narrative, provenance notes, and literature map.

TODO for final paper version:

- replace this abstract with the final problem statement, dataset description,
  model summary, evaluation protocol, and top-line result numbers
- add explicit forecast horizon, cohort/split definition, and the final claim
  being made
- cite the canonical papers for every major modeling choice

## Reading Guide Seed

Use the public `README.md` for the fast interface and the following deep docs
for serious reading:

- [`../current_architecture.md`](../current_architecture.md): current-system
  reference for packages, runtime flow, artifact outputs, and intended
  subsystem boundaries
- [`../codebase_evolution.md`](../codebase_evolution.md): historical narrative
  explaining why the repository evolved into its current layered shape
- [`../history/index.md`](../history/index.md): archive map for milestone notes
- [`../dependency_graphs/summary.md`](../dependency_graphs/summary.md):
  generated import-graph summary and structural evidence
- `../assets/`: hand-authored architecture images used to explain the
  forecasting problem and model structure

## Research Problem Seed

The repository is focused on the following research problem:

- forecast future blood glucose values from time-series data
- support uncertainty-aware prediction rather than only point prediction
- combine local temporal pattern extraction with longer-range sequence
  reasoning
- keep enough documentation, evaluation detail, and runtime traceability that
  a run can be interpreted later

The present codebase assumes an AZT1D-style data preparation path and a
grouped batch contract rather than a single raw tensor interface.

TODO for final paper version:

- rewrite this section as a formal problem statement
- clarify prediction target, forecast horizon, population, and intended use
- distinguish clearly between research motivation and validated claim

## Repository Contributions Seed

At the repository level, the project contributes:

- fused TCN + TFT probabilistic forecasting architecture with quantile outputs
- typed configuration layer
- Lightning-based training workflow
- structured evaluation system
- observability system (TensorBoard, diagnostics, tracing)
- modular reporting system (`src/reporting/`)
- prediction-level exports (row + aggregated)
- runtime-aware execution profiles
- dependency graph documentation
- subsystem-aligned tests

TODO for final paper:

- formalize contributions
- separate scientific vs engineering contributions

## Current Source Map Seed

This repository contains substantial internal documentation. The table below
preserves the older source-map concept, updated to current filenames.

| Topic | Primary source in repo | Role |
| --- | --- | --- |
| Current system architecture | [`../current_architecture.md`](../current_architecture.md) | Canonical description of the system as it exists now |
| Repository history and design rationale | [`../codebase_evolution.md`](../codebase_evolution.md) | Explains why the current boundaries and layers exist |
| Refactor- and milestone-specific context | [`../history/index.md`](../history/index.md) | Entry point for detailed notes about data, model, training, evaluation, observability, and runtime refactors |
| Static code architecture evidence | [`../dependency_graphs/summary.md`](../dependency_graphs/summary.md) | Import-graph summary and structural evidence |
| Graph generation logic | [`../../scripts/generate_dependency_graphs.py`](../../scripts/generate_dependency_graphs.py) | Reproducible generator for graph artifacts |
| Model visuals | `../assets/FusedModel_architecture.png`, `../assets/TCN_architecture.png`, `../assets/TFT_architecture.PNG` | Visual explanation of the hybrid model |
| Forecasting task visual | `../assets/Time_Series.jpg` | High-level reminder that the problem is sequential forecasting |
| Data contract and feature grouping | [`../../src/data/schema.py`](../../src/data/schema.py) | Shared schema vocabulary and grouped feature semantics |
| Model configuration contract | [`../../src/config/model.py`](../../src/config/model.py) | Typed config for TCN and TFT behavior |
| Runtime defaults and entrypoint policy | [`../../defaults.py`](../../defaults.py), [`../../main.py`](../../main.py) | Baseline experiment setup and user-facing entry surface |

## Literature And Provenance Inventory

This section tracks what literature or external lineage is already explicitly
named inside the repository.

### Internal literature already present in the repo

- [`../current_architecture.md`](../current_architecture.md)
- [`../codebase_evolution.md`](../codebase_evolution.md)
- [`../history/data_refactor_summary.md`](../history/data_refactor_summary.md)
- [`../history/model_refactor_summary.md`](../history/model_refactor_summary.md)
- [`../history/lightning_model_integration_summary.md`](../history/lightning_model_integration_summary.md)
- [`../history/train_wrapper_summary.md`](../history/train_wrapper_summary.md)
- [`../history/evaluation_package_summary.md`](../history/evaluation_package_summary.md)
- [`../history/observability_integration_summary.md`](../history/observability_integration_summary.md)
- [`../history/environment_runtime_profiles_summary.md`](../history/environment_runtime_profiles_summary.md)

### External sources already named in code or docs

| Source or lineage | Where it is already referenced | Current role |
| --- | --- | --- |
| AZT1D dataset release on Mendeley Data | `src/data/` modules and config comments | Dataset/source provenance |
| PyTorch Lightning DataModule docs | `src/data/` module headers | Guidance for data-layer organization |
| NVIDIA `DeepLearningExamples` | `src/models/tft.py`, `src/utils/tft_utils.py` | TFT implementation lineage |
| `pytorch-tcn` by Paul Krug | `src/models/tcn.py` | TCN implementation lineage |
| Prior work by SlickMik | `src/data/` module headers | Early pipeline lineage/context |
| Lightning tutorial article | `docs/history/lightning_model_integration_summary.md` | Guidance for Lightning integration decisions |

### Literature still missing from the repo

These should be added before treating any human-written paper document as
finished:

- canonical citation for the original Temporal Fusion Transformer paper
- canonical citation for foundational TCN literature
- citations for glucose forecasting domain papers most relevant to this task
- citations for probabilistic forecasting and quantile-loss methodology
- verified citation for the dataset release if a paper-form citation exists in
  addition to the dataset landing page
- citations for any baseline models used in future comparisons

TODO for final paper version:

- replace the missing-literature list with a proper bibliography
- add a short note for each citation explaining exactly what design choice it
  supports

## Related Work Notes

This section should eventually become the paper-style synthesis of the
literature rather than a raw inventory.

Suggested subsection structure:

- glucose forecasting literature
- probabilistic forecasting and uncertainty estimation
- TCN-based medical or physiological sequence models
- TFT and feature-aware temporal forecasting models
- hybrid architectures most similar to this repository

TODO for final paper version:

- summarize what each paper contributes
- explain how this repository differs from or builds on each line of work
- avoid listing references without discussing their relevance

## Bibliography Tracker

Use this section to hold the verified final citations once the team confirms
the exact sources and formatting.

Suggested entry template:

- `TODO citation`: full verified citation
  Used for: exact design choice, experiment baseline, or dataset provenance

## Visual Architecture Guide Notes

The repository contains both hand-authored model diagrams and generated static
dependency graphs.

These answer two different questions:

- what the forecasting system is trying to model
- how the current codebase is organized to implement it

### Static architecture graphs

The generated dependency graphs are derived from the repository's internal
Python import structure.

Useful entry points:

- [`../dependency_graphs/package_graph.svg`](../dependency_graphs/package_graph.svg)
- [`../dependency_graphs/production_module_graph.svg`](../dependency_graphs/production_module_graph.svg)
- [`../dependency_graphs/entrypoint_flow_graph.svg`](../dependency_graphs/entrypoint_flow_graph.svg)
- [`../dependency_graphs/test_dependency_graph.svg`](../dependency_graphs/test_dependency_graph.svg)
- [`../dependency_graphs/summary.md`](../dependency_graphs/summary.md)
- [`../dependency_graphs/dependency_graph.json`](../dependency_graphs/dependency_graph.json)

To regenerate the static graph set from the repository root, run:

```bash
python scripts/generate_dependency_graphs.py
```

This requires Python and the Graphviz `dot` executable.

### Model diagrams

- `../assets/FusedModel_architecture.png`
- `../assets/TCN_architecture.png`
- `../assets/TFT_architecture.PNG`
- `../assets/Time_Series.jpg`

## Dataset And Data Contract Notes

The current repository is organized around an AZT1D-oriented pipeline.

At a high level, the data path is:

1. download raw dataset material when needed
2. preprocess it into one canonical processed CSV
3. derive semantic feature groups and sequence indices
4. construct grouped batch items for model consumption
5. bind runtime-discovered categorical metadata back into the model config

The data contract is intentionally semantic rather than purely positional. The
system distinguishes static, known, observed, and target features so that the
data layer and model layer can agree on what each feature means.

Current normalization semantics worth knowing:

- raw AZT1D field names are rewritten into canonical columns such as
  `glucose_mg_dl`, `basal_insulin_u`, `bolus_insulin_u`,
  `correction_insulin_u`, `meal_insulin_u`, and `carbs_g`
- `*_insulin_u` columns represent insulin amounts in units, and `carbs_g`
  represents carbohydrate grams
- exact duplicate rows are dropped before later cleanup
- same-subject/same-timestamp collisions are collapsed into one cleaned row so
  sequence indexing can assume one observation per subject and timestamp
- `basal_insulin_u` is treated as a carried state on the shared 5-minute grid
  and is forward/back filled within each subject
- bolus, correction, meal-insulin, and carbohydrate features are treated as
  sparse event quantities and are zero-filled when no event is present
- `device_mode` is normalized to `regular`, `sleep`, `exercise`, or `other`
- `bolus_type` is treated as event-local rather than stateful and is not
  forward-filled across future rows
- the data layer can produce JSON-ready descriptive statistics for the cleaned
  dataframe and split/window layout

Current config-default semantics worth knowing:

- the public AZT1D download URL and the 5-minute sampling interval are treated
  as dataset-derived defaults
- sequence lengths, split ratios, split mode, and window stride remain
  repository baseline experiment defaults rather than claims from the dataset
  paper

Practical default locations:

- raw downloads: `data/raw/`
- cache and extracted intermediate files: `data/cache/`, `data/extracted/`
- canonical processed dataset: `data/processed/azt1d_processed.csv`

Current practical behavior:

- if the processed CSV is missing, the main workflow can download and prepare
  the public AZT1D source automatically
- the heavier manual verification path lives at
  `tests/manual/manual_data_smoke.py`
- the pipeline is still documented as AZT1D-oriented rather than
  dataset-agnostic

Best current sources:

- `src/data/schema.py`
- `src/data/datamodule.py`
- `src/data/dataset.py`
- `docs/history/data_refactor_summary.md`
- `docs/current_architecture.md`

TODO for final paper version:

- add a formal dataset subsection with cohort/source details
- specify split policy exactly
- add feature table with units, semantics, and availability timing
- clarify any preprocessing exclusions, leakage protections, and missing-data
  handling

## Dataset Access, Governance, And Licensing Prompts

Use this section later to describe the practical and ethical status of the
dataset rather than only its tensor/dataflow role.

Suggested fill-in checklist:

- exact dataset name and version used for the reported experiments
- where the dataset comes from and how it is accessed
- whether the workflow downloads it automatically or expects manual setup
- dataset license or terms of use
- any access restrictions, approvals, or rate limits
- privacy, de-identification, or ethics notes relevant to the data source
- local storage expectations and approximate disk footprint

Important instruction for later:

- do not write anything here that has not been verified directly from the
  dataset source or its official documentation
- if dataset terms are uncertain, say so explicitly instead of guessing

## Observability And Reporting Architecture Notes

The repository separates:

- evaluation -> compute metrics
- reporting -> structure outputs
- observability -> logging and visualization

### Reporting layer (`src/reporting/`)

- report_tables.py
- prediction_rows.py
- plotly_reports.py
- builders.py
- tensorboard.py

### Observability layer (`src/observability/`)

- TensorBoard logging
- runtime diagnostics
- model visualization
- environment-aware logging

### Design direction

- unify outputs into TensorBoard
- enable drill-down views
- ensure reproducibility and traceability

## Model Overview Notes

The current forecasting model is a late-fusion hybrid:

- three TCN branches at kernel sizes `3`, `5`, and `7`
- one TFT branch over grouped static, historical, and future-known inputs
- one post-branch GRN fusion layer
- one final head that emits quantile forecasts

The current implementation treats quantile prediction as part of the model
contract rather than leaving loss interpretation to outer training code.

Best current sources:

- `src/models/fused_model.py`
- `src/models/tcn.py`
- `src/models/tft.py`
- `src/models/grn.py`
- `src/models/nn_head.py`
- `docs/history/model_refactor_summary.md`
- `docs/current_architecture.md`

TODO for final paper version:

- add a formal method section with notation
- include exact input/output tensor descriptions
- state the loss function mathematically
- clarify which architectural components are inherited, adapted, or novel
- add citations for every major modeling block

## Training, Evaluation, And Reproducibility Notes

The repository exposes both script and notebook entry surfaces:

- `main.py`
- `main.ipynb`
- `defaults.py`

The entrypoint path is intentionally thin and delegates reusable orchestration
to `src/workflows/`, `src/train.py`, `src/environment/`, and the config layer.

### Install dependencies

Local:

```bash
pip install -r requirements.txt
```

Apple Silicon often benefits from installing PyTorch separately first:

```bash
pip install torch
pip install -r requirements.txt
```

`torchvision` and `torchaudio` are not required by this repository directly.
If you choose to install them, make sure they match the exact PyTorch build in
your active environment; mismatched `torch` / `torchvision` versions can fail
during import before the project code runs.

### Run tests

Run the full tracked test suite:

```bash
pytest tests -q
```

Run a few representative subsystems:

```bash
pytest tests/config tests/training tests/workflows/test_training_workflow.py -q
```

Run evaluation-focused tests:

```bash
pytest tests/evaluation -q
```

### Manual data smoke test

```bash
python tests/manual/manual_data_smoke.py
```

This is intentionally separate from the normal pytest suite because it may
touch the network and real filesystem state.

### Run the pipeline

Minimal example:

```bash
python main.py --max-epochs 5 --batch-size 32
```

Runtime-profile examples:

```bash
python main.py --device-profile local-cpu
python main.py --device-profile local-cuda
python main.py --device-profile apple-silicon
python main.py --device-profile slurm-cpu
python main.py --device-profile slurm-cuda
python main.py --device-profile colab-cpu
python main.py --device-profile colab-cuda
```

Diagnostics-only run:

```bash
python main.py --device-profile auto --run-diagnostics-only
```

Short benchmark-style run:

```bash
python main.py --device-profile auto --run-benchmark-only --benchmark-train-batches 10
```

### Expected outputs

Core artifacts:

- run_summary.json
- checkpoints
- test_predictions.pt / .csv

Structured outputs:

- per-horizon metrics
- aggregated summaries
- prediction rows

Visualization outputs:

- Plotly HTML reports
- TensorBoard dashboards

Default directory:

`artifacts/main_run/`

Direction:

- move toward TensorBoard-native reporting
- enable hierarchical drill-down

## Experimental Setup Prompts

Use this section later for the paper-style experiment protocol.

Suggested subsection structure:

- train/validation/test split definition
- forecast horizon and sampling interval
- hardware used for the headline runs
- training duration, epoch count, and early-stopping policy
- optimizer, learning-rate, batch-size, and precision settings
- checkpoint-selection policy
- ablations or baseline-comparison protocol

Instruction for later:

- keep this section factual and reproducible
- distinguish clearly between defaults in the codebase and the exact settings
  used for a reported experiment

## Metrics Definition Prompts

Use this section later to define the evaluation metrics before inserting any
tables.

Suggested fill-in checklist:

- primary point-forecast metrics
- primary probabilistic or quantile-quality metrics
- which metric selects the best checkpoint
- which metrics are reported only on held-out test data
- any per-horizon, per-subject, or grouped metrics
- any calibration, sharpness, or uncertainty-specific measures

Instruction for later:

- define every metric in words before showing result numbers
- state clearly whether lower or higher is better
- avoid mixing validation-selection metrics with final test-report metrics

## Run Artifacts And Reporting Notes

Each training run writes a set of artifacts meant to make the workflow easier
to inspect, debug, and reuse in reporting. The point is not just to train a
model and hope for the best. It is to leave behind a clean record of what data
was used, what happened during the run, and what results came out of it.

By default, run outputs are written under:

`artifacts/main_run/`

Depending on the observability settings, that folder may contain logs,
summaries, prediction exports, grouped metric tables, and generated visual
reports.

### Core run artifacts

`run_summary.json`

This is the main high-level summary for a run. It captures the overall run
setup and points to the major artifacts that were produced.

It includes things like:

- resolved runtime configuration
- optimizer settings
- device profile information
- fit and evaluation details
- paths to prediction exports and report files

If someone wants the quickest overview of a run, this is the best file to open
first.

`data_summary.json`

This file captures dataset-level observability information. It is written early
in the workflow so that even if training fails later, the run still leaves
behind a clear record of what data it was built on.

This summary is meant to help answer questions like:

- what data did the run actually use
- how large was the dataset
- what did the split or descriptive footprint look like

This is especially useful when writing the report because it gives a cleaner
way to describe the dataset and preprocessing context.

`metrics_summary.json`

This is the compact evaluation summary artifact. Its job is simple. It should
tell you how the run performed without forcing you to dig through the full run
summary or raw logs.

It includes:

- the checkpoint used for evaluation
- scalar test metrics
- structured evaluation output
- links to related prediction and report artifacts

This is the file to use for headline results.

### Grouped evaluation tables

When grouped evaluation results are available, the workflow exports them as
flat CSV tables. These are much easier to inspect, plot, and reuse in the
report than nested JSON alone.

These can include:

- metrics_by_horizon.csv
- metrics_by_subject.csv
- metrics_by_glucose_range.csv

These tables matter because they move the analysis beyond one average number.
They let the reader look at how performance changes across:

- different forecast horizons
- different subjects
- different glucose regimes

That is much more useful for reporting than only quoting one overall metric.

`report_index.json`

This is the artifact map for the run. It points to the main files produced
during execution so that someone opening the folder does not have to guess
where things live.

It links out to:

- run_summary.json
- data_summary.json
- metrics_summary.json
- grouped evaluation tables
- prediction artifacts
- generated report files
- logging outputs

If you are opening a run folder for the first time, this is the cleanest place
to start.

### Prediction and visualization artifacts

Prediction tensor export

If enabled, the workflow saves raw prediction tensors so downstream analysis
can still access direct model outputs instead of only reduced summaries.

Typical file:

`test_predictions.pt`

This is more useful for debugging and custom analysis than for reporting
directly.

Prediction table export

If enabled, predictions are also flattened into a tabular export. This makes
it much easier to build plots, inspect examples, and reuse results in
reporting.

This artifact is useful for:

- forecast plots
- residual analysis
- subject-level examples
- report figures

Plotly reports

If enabled, the workflow generates Plotly-based reports from the prediction
table. These are meant to support quick visual inspection and later reuse in
the final writeup.

To keep the reporting path robust, Plotly reports are only generated when the
prediction table exists.

### Logs

`run.log`

This is the plain-text run log. It helps track what happened during execution
and where the run may have slowed down or failed.

`logs/`

This directory may contain additional logger outputs depending on the
observability configuration.

In general, the log files are most useful for debugging, while the JSON and CSV
artifacts are better for reporting.

### How to read the results

A good way to inspect a completed run is:

1. Open `report_index.json` to see the full artifact map.
2. Open `run_summary.json` for the high-level run overview.
3. Open `data_summary.json` to understand the dataset used in the run.
4. Open `metrics_summary.json` for the main evaluation results.
5. Use the grouped evaluation CSVs to look at performance by horizon, subject,
   or glucose range.
6. Use the prediction table and visual reports for examples and final figures.

This makes the workflow easier to debug during development and much easier to
reuse when writing the final report.

## Results Table Prompts

No finalized benchmark table is claimed in these notes yet.

Suggested structure for future insertion:

| Experiment | Split definition | Horizon | Main metrics | Uncertainty metrics | Artifact path | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| TODO baseline | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO fused model | TODO | TODO | TODO | TODO | TODO | TODO |

TODO for final paper version:

- add exact metrics used for comparison
- separate validation selection metrics from held-out reporting metrics
- report both point and probabilistic quality where relevant
- link each row to reproducible artifact outputs

## Limitations And Current Constraints Seed

The current repository intentionally still carries several constraints:

- the data path is AZT1D-oriented rather than fully multi-dataset
- some feature-spec behavior still includes transitional fallback logic
- detailed evaluation currently depends on the prediction path
- some observability features are optional and environment-dependent
- the literature inventory is incomplete and not yet a finished bibliography

These are not hidden issues; they are part of the current documented state of
the project.

## Final Paper Checklist

Before treating any paper-facing overview as final, fill in:

- the final abstract with actual results
- the exact research question and claim
- the canonical citations for TFT, TCN, quantile forecasting, and glucose
  forecasting literature
- the formal dataset description and split protocol
- the exact hyperparameter table
- the final experiment and baseline comparison table
- the limitations/validity section in paper language
- the bibliography section with verified citation formatting

## Development Workflow Notes

Use this section later for contributor-facing repo-maintenance guidance.

Suggested fill-in checklist:

- how to run the main test suite
- how to run a focused subsystem test file
- how to regenerate dependency graphs
- where to add new architecture/history docs
- how to update README-facing figures or assets
- any formatting, linting, or type-checking commands the team wants to
  standardize

Instruction for later:

- keep this section short and operational
- link to deeper docs instead of duplicating long contributor instructions

## Citation Notes

Use this section later once the team decides the preferred project citation.

Suggested fill-in checklist:

- repository citation text
- paper citation text, if a paper/preprint exists
- version, commit, or release guidance for reproducibility

Instruction for later:

- do not invent a citation format before the team agrees on one

## License And Reuse Notes

The repository includes a license file at `LICENSE`.

Placeholder for later:

- add one short sentence summarizing the intended reuse expectations for code
- if needed, separately clarify dataset reuse expectations, since dataset terms
  may differ from repository code licensing

## Appendix: Practical Navigation

If you are trying to navigate the codebase quickly, start here:

- data pipeline: `src/data/`
- model architecture: `src/models/`
- training wrapper: `src/train.py`
- workflow and CLI orchestration: `src/workflows/`
- runtime profiles and diagnostics: `src/environment/`
- evaluation: `src/evaluation/`
- observability and reporting: `src/observability/`

If you want the shortest recommended reading order:

1. the public `README.md`
2. `docs/current_architecture.md`
3. `docs/codebase_evolution.md`
4. the most relevant document under `docs/history/`
