# Runtime And Entrypoints Walkthrough

Role: Focused walkthrough of how execution starts and how runtime behavior is
resolved.
Audience: Engineers and contributors trying to understand the top-level control
flow quickly.
Owns: Entrypoint, CLI, defaults, workflow, and runtime-profile explanation.
Related docs: [`../system_walkthrough.md`](../system_walkthrough.md),
[`../current_architecture.md`](../current_architecture.md),
[`../cli_reference.md`](../cli_reference.md),
[`../execution_guide.md`](../execution_guide.md).

## 1. Stable Entrypoints

The public runnable surface is intentionally simple:

- `python main.py`
- notebook-oriented use through `main.ipynb`
- reusable default builders through `defaults.py`

That simplicity hides a more layered workflow below it.

## 2. What `main.py` Actually Does

`main.py` is intentionally thin. It re-exports the workflow-facing surfaces and
delegates the heavy logic to `src/workflows/`.

That means:

- the repo preserves a friendly public entrypoint
- tests and notebooks can still import top-level helpers
- orchestration logic does not get trapped in the root file

## 3. CLI To Typed Configuration

`src/workflows/cli.py` is the bridge from flat command-line flags to the
repository's typed runtime surfaces.

It is responsible for:

- registering the flat CLI
- building typed config objects
- normalizing special arguments such as `--device-profile`
- dispatching into the training workflow or diagnostics-only / benchmark-only
  modes

Use [`../cli_reference.md`](../cli_reference.md) when you need the exact flag
mapping.

## 4. Why Defaults Exist Separately

`defaults.py` is not just a convenience file. It encodes the baseline policies
for:

- data lengths and splits
- model dimensions and quantiles
- snapshot behavior
- observability behavior

Keeping that policy in one place makes the CLI, notebook, and workflow layers
share a common baseline instead of quietly drifting apart.

## 5. Runtime Profiles

The repository does not assume the same runtime policy for every environment.
It supports profiles such as:

- `local-cpu`
- `local-cuda`
- `apple-silicon`
- `colab-cpu`
- `colab-cuda`
- Slurm-oriented profiles

The runtime layer can adjust effective settings based on what machine or
platform is actually available.

That is why "CLI default" and "effective runtime behavior" are not always
identical.

## 6. Where The Real Orchestration Lives

Once config and runtime state are resolved, `src/workflows/training.py` takes
over the high-level run lifecycle:

1. build effective config
2. prepare data and bind runtime-discovered facts
3. construct the trainer and model
4. fit, test, and predict
5. export summaries, reports, and observability artifacts

This is the main place to read when you want to understand how one command
turns into a full run.

## 7. What To Read Next

- For data preparation and semantic contracts:
  [`data_pipeline_walkthrough.md`](data_pipeline_walkthrough.md)
- For model construction and training:
  [`model_and_training_walkthrough.md`](model_and_training_walkthrough.md)
- For artifact and evaluation behavior:
  [`evaluation_reporting_walkthrough.md`](evaluation_reporting_walkthrough.md)
- For exact present-state ownership boundaries:
  [`../current_architecture.md`](../current_architecture.md)
