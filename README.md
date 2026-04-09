# ENSF617 Probabilistic Glucose Forecasting

A probabilistic glucose forecasting repository built around a fused TCN + TFT
model and a layered training and evaluation workflow.

The project is designed as a full forecasting system rather than a single
training script. Data preparation, runtime policy, evaluation, observability,
and reporting are all treated as part of the pipeline, so a run leaves behind
more than a checkpoint or one headline metric. It produces a readable bundle
of summaries, metrics, predictions, and logs that can be inspected afterward.

This repository is research software for probabilistic glucose forecasting. It
is not a clinically validated medical decision-support system.

![Fused model architecture](docs/assets/FusedModel_architecture.png)

At a glance:

- the model predicts quantiles rather than only point estimates
- each run leaves behind structured summaries, metrics, predictions, and logs
- observability and reporting are first-class parts of the workflow
- the codebase is layered across config, environment, data, models, training,
  evaluation, observability, and reporting

## Quickstart

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run a short end-to-end pass:

```bash
python main.py --max-epochs 3 --batch-size 32
```

If you want a quick confidence check for the local environment:

```bash
pytest tests -q
```

That single command drives the full workflow: runtime resolution, data
preparation, model binding, training, held-out evaluation, and post-run
artifact generation.

## After A Run

By default, a run writes its outputs under `artifacts/main_run/`. The most
useful files to expect are:

- `run_summary.json` for the run configuration, environment, and high-level outcome
- `report_index.json` for a top-level map of the generated artifacts
- `metrics_summary.json` for the consolidated evaluation summary when enabled
- grouped evaluation tables for horizon-, subject-, and range-level behavior
- prediction exports such as `test_predictions.pt` and `test_predictions.csv`
- logs, telemetry, and report artifacts for deeper inspection

A finished run is meant to be readable, not just successful.

A simple first pass is:

1. `report_index.json` for the top-level map of what was written
2. `run_summary.json` for the run configuration and overall outcome
3. `metrics_summary.json` for the main evaluation summary
4. grouped metrics tables for horizon-, subject-, and range-level behavior
5. prediction exports and report artifacts for example-level inspection
6. logs, telemetry, and TensorBoard when you need deeper runtime diagnosis

For the full artifact reference, see
[docs/reference/artifact_contract.md](docs/reference/artifact_contract.md). For
one concrete example of how to interpret a real run, see
[docs/artifact_diagnosis.md](docs/artifact_diagnosis.md).

## Repository Layout

The implementation follows the lifecycle of a run, with the major concerns kept
separate enough to stay readable:

- `src/config/` holds typed configuration
- `src/environment/` resolves runtime profiles and diagnostics
- `src/data/` owns dataset preparation and the model-facing data contract
- `src/models/` contains the TCN, TFT, fusion, and prediction head
- `src/train.py` and `src/workflows/` own training orchestration
- `src/evaluation/`, `src/observability/`, and `src/reporting/` own post-run
  interpretation surfaces
- `docs/` contains the deeper technical and research-facing material

For a fuller walkthrough of how those pieces fit together as one system, start
with [docs/system_walkthrough.md](docs/system_walkthrough.md).

## Where To Go Next

For running the system and understanding the day-to-day workflow, start with:

- [docs/execution_guide.md](docs/execution_guide.md)
- [docs/cli_reference.md](docs/cli_reference.md)
- [docs/reference/artifact_contract.md](docs/reference/artifact_contract.md)

Use [docs/artifact_diagnosis.md](docs/artifact_diagnosis.md) after that when
you want one concrete forensic read of a finished run rather than the general
artifact map.

If you want the codebase story rather than just the execution path, continue
with:

- [docs/system_walkthrough.md](docs/system_walkthrough.md)
- [docs/repository_primer.md](docs/repository_primer.md)
- [docs/current_architecture.md](docs/current_architecture.md)

If you want to modify or extend the repository safely, the most useful starting
path is:

- [docs/reference/package_boundaries.md](docs/reference/package_boundaries.md)
- [docs/reference/runtime_and_config_flow.md](docs/reference/runtime_and_config_flow.md)
- [docs/reference/extension_and_constraints.md](docs/reference/extension_and_constraints.md)

For the research-facing view, the repository also keeps a paper-style
companion under:

- [docs/research/index.md](docs/research/index.md)

It follows a familiar paper structure:

- `Abstract`
- `Introduction`
- `Related Work`
- `Methods`
- `Results and Discussion`
- `Conclusions`
- `References`

For the deeper research-side reads behind that companion, continue with the
dataset, methods, and discussion sections, then use the evolution and history
docs when you want broader context:

- [docs/research/dataset.md](docs/research/dataset.md)
- [docs/research/methodology.md](docs/research/methodology.md)
- [docs/research/results_and_discussion.md](docs/research/results_and_discussion.md)
- [docs/codebase_evolution.md](docs/codebase_evolution.md)
- [docs/history/index.md](docs/history/index.md)

Code is licensed under [LICENSE](LICENSE). Dataset terms may differ from code
license terms and should be validated against the dataset source.
