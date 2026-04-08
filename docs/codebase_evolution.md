# Codebase Evolution

Role: Historical narrative for why the repository took its current shape.
Audience: Maintainers and researchers who need design rationale over time.
Owns: Milestone timeline, design pressures, and links to archival notes.
Related docs: [`current_architecture.md`](current_architecture.md),
[`history/`](history/), [`system_walkthrough.md`](system_walkthrough.md),
[`repository_primer.md`](repository_primer.md).

This document is the historical companion to
[`current_architecture.md`](current_architecture.md). Its purpose is to explain
how the repository moved from its early prototype form to the current layered
system, and why the major refactors happened when they did.

Use this file when you want to understand:

- what the repository looked like before the first formal documentation pass
- which architectural pressures drove each major refactor
- why the current package boundaries exist
- how the repository's vision became more explicit over time

For the point-in-time notes written during each major refactor, see the
archived documents under [`history/`](history/).

Important note:
this document stays intentionally historical. Some milestones below describe
the architectural shape at the time they landed, while later follow-up notes
call out where the current repository moved further without changing the core
reason those milestones mattered.

## Reading Guide

This file is intentionally narrative rather than commit-by-commit exhaustive.
It is the bridge between:

- the pre-documentation codebase
- the archived milestone summaries in [`history/`](history/)
- the current system layout in [`current_architecture.md`](current_architecture.md)

One useful way to read it is:

1. start with the baseline snapshot
2. read the milestones in order
3. compare the recurring design pressures to the current architecture

## The Pre-Documentation Baseline

The most useful baseline snapshot is commit `ffd190c` on March 31, 2026. It is
the last state of the repository before the first `docs/*.md` summary appeared
in commit `a769de1` on April 1, 2026.

At that point, the project already had the core forecasting idea and several
important implementation pieces, but it still behaved much more like an
assembled prototype than a deliberately layered research system.

### Baseline Repository Shape

A simplified view of the repository around `ffd190c` looked like this:

```text
README.md
requirements.txt
src/
  data/
    azt1d_dataset.py
    combiner.py
    downloader.py
  models/
    FusedModel_architecture.png
    Time_Series.jpg
    TFT_architecture.PNG
    fused_model.py
    grn.py
    nn_head.py
    tcn.py
    tft.py
  test.py
  utils/
    config.py
    tft_utils.py
```

Several important things were already true:

- the repo already targeted probabilistic glucose forecasting
- the fused TCN + TFT direction already existed
- the TCN, TFT, GRN, and prediction-head building blocks were present
- dataset download and AZT1D-specific dataset logic existed

But several important things were also missing:

- no dedicated `src/config/` package
- no `src/train.py` wrapper
- no root-level `main.py`, `main.ipynb`, or `defaults.py`
- no `src/observability/` package
- no `src/evaluation/` package
- no organized `tests/` tree
- no durable architecture documentation

### Baseline Architectural Characteristics

The baseline code had the right ambition, but several responsibilities were
still tightly coupled.

#### Data Was Too Monolithic

The older data path was concentrated in the earlier AZT1D dataset and combiner
modules. Downloading, preprocessing, split logic, feature-group handling, and
sample assembly had not yet been separated into clearly named layers.

That made the pipeline harder to reason about and harder to align cleanly with
the model contract.

#### The Model Contract Was Still In Transition

The older `FusedModel` in the baseline commit still reflected an unfinished
integration boundary. Its constructor directly assembled three TCN branches, a
TFT branch, a GRN, and a final head, but the forward path still exposed the
earlier uncertainty around structured inputs:

- `forward(...)` accepted a single raw tensor argument
- TODO comments still existed around how to split static, observed, and future
  inputs
- TCN and TFT outputs were concatenated more directly, before the later
  late-fusion cleanup

This is important because it shows that the repo already had the hybrid model
idea, but not yet the clean batch contract or the final architectural story for
how TCN and TFT should meet.

#### Configuration Existed, But As One Utility Surface

The configuration layer lived in `src/utils/config.py`. That module already
carried important project knowledge, but it had not yet been promoted into a
first-class package with explicit ownership boundaries for:

- data configuration
- model configuration
- runtime training policy
- observability policy
- config serialization

#### Training And Evaluation Were Not Yet Layered

There was no reusable training wrapper, no shared top-level workflow, and no
separate evaluation package. The codebase had model components and data
components, but it did not yet have the later explicit separation between:

- model behavior
- Trainer orchestration
- top-level execution
- structured evaluation
- runtime observability

### What The Baseline Did Well

It is worth being explicit that the early repo was not "bad." It already
established the core project direction:

- a hybrid forecasting architecture rather than a single-model baseline
- probabilistic outputs rather than only point regression
- AZT1D-specific dataset support
- an interest in combining short-range temporal signals with richer structured
  sequence reasoning

The later refactors should be understood as making that original direction
coherent and maintainable, not as replacing it.

### The Most Important Baseline Gap

If there is one summary of the pre-documentation repo, it is this:

- the project already knew what it wanted to predict
- it did not yet have a stable shared story for how data, model, runtime, and
  evaluation should fit together

Most later refactors can be understood as resolving that gap.

## Why The Repository Needed To Evolve

The refactors that followed were driven by real architectural pressure rather
than by cosmetic cleanup.

The main pressure points were:

- the data pipeline needed explicit layers
- the model needed a stable semantic input contract
- Lightning responsibilities needed a clearer split
- config had become too important to live as a generic utility
- the repo needed a shared script and notebook entry surface
- observability and evaluation needed to become first-class concerns
- documentation had to explain intent, not just implementation

These pressures all pointed in the same direction: move from a promising
prototype to a repository that other teammates could reliably run, inspect,
extend, and re-enter later.

## Evolution Timeline

### March 18-31, 2026: Prototype Assembly

Before the first documentation commit, the repository went through a quick
assembly phase:

- March 18, 2026, `75592ae`: initial commit
- March 23, 2026, `afa03f4`: dataset download script added
- March 25, 2026, `397f025`: TFT checkpoint work
- March 26, 2026, `1550421`: TFT model added
- March 26, 2026, `307f0d5`: configuration utilities added
- March 28-29, 2026: data combiner and preprocessing work landed
- March 30, 2026, `9273e74`: fused model introduced
- March 31, 2026, `1cb1a76`: TCN added and fused model improved
- March 31, 2026, `5e527c1` and `5994dd1`: AZT1D dataset and dataloader work
- March 31, 2026, `ffd190c`: fused-model architecture image updated

This phase produced the first working version of the project vision, but not
yet the final codebase structure.

The key outcome of this phase was viability, not maintainability. It proved the
project direction was worth structuring more carefully.

### April 1, 2026, later follow-up: Test Layout Cleanup And Runtime Modernization

This later follow-up is documented in
[`history/test_layout_and_runtime_modernization_summary.md`](history/test_layout_and_runtime_modernization_summary.md).

#### What changed

The repository took a cleanup pass across two areas that had grown somewhat
independently:

- the test tree was reorganized into package-aligned folders such as
  `tests/config/`, `tests/environment/`, `tests/models/`, `tests/training/`,
  and `tests/workflows/`
- the manual smoke script moved into `tests/manual/`
- the TFT path stopped relying on `torch.jit.script(...)`
- the benchmark workflow gained explicit runtime/device synchronization
  boundaries for CUDA timing

#### Why it changed

The test tree had already grown into a real subsystem, but the root-level
layout still mixed:

- shared support files
- package-aligned folders
- broad catch-all test modules
- manual smoke scripts

At the same time, the runtime-performance story had evolved:

- the repository already had environment-aware `torch.compile(...)` support
- PyTorch had begun warning that TorchScript is deprecated
- benchmark timing wanted CUDA synchronization at workflow boundaries rather
  than inside model code

The follow-up therefore moved the codebase toward a cleaner long-term story:

- tests should reflect ownership boundaries the same way `src/` does
- eager model code plus runtime-owned `torch.compile(...)` is the preferred
  optimization path
- timing-only CUDA behavior belongs to runtime/benchmark orchestration rather
  than the model forward path

#### Why this matters historically

This pass did not introduce a new subsystem, but it tightened the coherence of
two existing ones:

- `tests/` became more obviously part of the architecture rather than a pile of
  coverage files
- the runtime layer became the single place where acceleration and timing
  policy are interpreted

### April 1, 2026, `a769de1`: Data Refactor Into A Lightning DataModule

This is the first milestone documented in
[`history/data_refactor_summary.md`](history/data_refactor_summary.md).

#### What changed

The older monolithic data flow was replaced by a layered pipeline:

- `src/data/downloader.py`
- `src/data/preprocessor.py`
- `src/data/schema.py`
- `src/data/transforms.py`
- `src/data/indexing.py`
- `src/data/dataset.py`
- `src/data/datamodule.py`

The earlier `src/data/azt1d_dataset.py`, `src/data/combiner.py`, and `src/test.py`
paths were removed.

#### Why it changed

The repository needed:

- a clearer split between disk-side work and in-memory dataset construction
- an explicit sample contract for the fused model
- one place to discover runtime categorical metadata and sequence details
- a data layer that matched the intended PyTorch Lightning lifecycle

#### Lasting impact

This refactor established one of the most important long-term design decisions
in the codebase: the DataModule owns data-derived runtime metadata and binds
that metadata into the model config before model construction.

That design still shapes the repository today.

It also changed the team's mental model of the data layer. The data path was no
longer just "how we get tensors"; it became a proper subsystem with clear
stages and contracts.

### April 1, 2026, `1723892`, `22439d0`, `aaaf8d7`: Model Refactor And Fusion Cleanup

These model-side passes are summarized in
[`history/model_refactor_summary.md`](history/model_refactor_summary.md).

#### What changed

The model stack was narrowed and clarified:

- the local TCN implementation was simplified into a project-specific causal
  residual forecasting path
- `FusedModel` was aligned with the structured batch contract coming from the
  data layer
- the fusion path between TCN and TFT was cleaned up
- the final head became more expressive
- GRN construction and config alignment were tightened

Architecture images were also moved out of `src/models/` and eventually became
part of the project documentation surface.

#### Why it changed

The baseline model architecture still showed signs of being mid-transition:

- ambiguous input semantics
- leftover TODOs around feature-group handling
- a broader TCN implementation surface than the project actually used
- a fusion story that was harder to explain than it needed to be

The repository needed the model layer to reflect the actual project use case
rather than preserving a more generic, partially inherited shape.

#### Lasting impact

This phase locked in the repo's current late-fusion direction:

- TCN branches operate on history-only dynamics
- TFT operates on the structured grouped temporal inputs
- both branches produce latent horizon-aligned features
- fusion happens before final quantile prediction, not after independent branch
  outputs have already collapsed

That is still the defining high-level architectural decision in the model
layer.

This phase also made the model more documentable. Once the fusion story became
clearer, it became possible to explain the architecture to teammates in a way
that mapped cleanly onto the code.

### April 1, 2026, `e904cae`: Full LightningModule Integration In `FusedModel`

This work is summarized in
[`history/lightning_model_integration_summary.md`](history/lightning_model_integration_summary.md).

#### What changed

`FusedModel` became a fuller Lightning-native model:

- quantile loss moved into the model
- train/validation/test step logic was added
- prediction step logic was added
- optimizer construction moved into the model
- checkpoint-friendly config serialization was added
- lazy TFT parameter materialization was made Lightning-safe

#### Why it changed

The data layer had already moved toward Lightning through the DataModule, but
the model layer was still not carrying its full side of the Lightning contract.

The repo needed the model to own:

- its supervision semantics
- its probabilistic output interpretation
- its optimizer configuration
- its checkpoint-reload contract

#### Lasting impact

This refactor made it possible to introduce a thin orchestration wrapper later
instead of burying model-specific training logic in a top-level script.

It also made checkpoint reload and notebook usage materially cleaner. The model
stopped depending on the surrounding run environment to reconstruct core
training semantics.

### April 1, 2026, `60e0db6`: Reusable Training Wrapper

This pass is recorded in
[`history/train_wrapper_summary.md`](history/train_wrapper_summary.md).

#### What changed

`src/train.py` was added, centered on `FusedModelTrainer`.

That wrapper took ownership of:

- preparing and setting up the DataModule
- binding runtime config
- constructing the model
- assembling callbacks
- constructing the Lightning `Trainer`
- coordinating fit/test/predict

#### Why it changed

The repo needed a layer between:

- the model and data implementation details
- the user-facing entry surfaces

The critical reason this wrapper exists is that the final model config cannot
be built purely from a static declarative config file. It depends on metadata
the DataModule only knows after preparation and setup.

#### Lasting impact

This pass created the core runtime ownership split that still defines the repo:

- `AZT1DDataModule` owns data lifecycle
- `FusedModel` owns model behavior
- `FusedModelTrainer` owns Trainer orchestration

That split is one of the clearest examples of the repository moving from
prototype-style composition to deliberate architecture.

### April 1, 2026, `e120edd`: Root-Level Entrypoints And Shared Defaults

This pass is summarized in
[`history/entrypoint_defaults_summary.md`](history/entrypoint_defaults_summary.md).

#### What changed

Three root-level entry surfaces were introduced:

- `defaults.py`
- `main.py`
- `main.ipynb`

#### Why it changed

The repository had grown beyond the point where ad hoc notebook setup or
custom one-off scripts were enough. The team needed:

- a runnable script entrypoint
- a notebook surface that did not drift from the script
- one shared defaults layer for baseline experiments

#### Lasting impact

This is when the repo became meaningfully runnable as a full workflow rather
than as a collection of modules that a user had to assemble manually.

It also established a team-facing usability principle that still matters: the
script and notebook should share one workflow, not just similar ideas.

#### Later follow-up

The current repository still preserves that same principle, but the heavier
entry-surface orchestration no longer lives only in the root script.

A later refactor moved the reusable CLI/workflow logic into `src/workflows/`
while keeping:

- `main.py` as the stable user-facing facade
- `main.ipynb` as the thin notebook surface over the same workflow
- the same public import surface available to tests and notebooks

That follow-up did not replace the original entrypoint milestone. It refined
it by making the public entry surfaces thinner and the shared orchestration
easier to navigate internally.

### April 1, 2026, `3f6eb48`: Config Promotion, Source Cleanup, And Observability Integration

This broad pass is covered by:

- [`history/source_refactor_and_documentation_update.md`](history/source_refactor_and_documentation_update.md)
- [`history/observability_integration_summary.md`](history/observability_integration_summary.md)
- [`history/commenting_conventions_summary.md`](history/commenting_conventions_summary.md)

#### What changed

Three big things happened together:

- config moved from `src/utils/config.py` into the dedicated `src/config/`
  package
- source-level comments, docstrings, and typing cleanup were expanded
- a richer observability and reporting surface was added around the run flow

#### Why it changed

By this point the repository had enough moving parts that maintainability had
become part of the architecture. The code needed:

- clearer import and ownership boundaries
- a more explicit config surface
- stronger runtime visibility
- documentation inside the code, not only in external markdown files

#### Lasting impact

This phase transformed two major parts of the repo:

- config became a first-class domain instead of a generic utility
- observability became a real subsystem rather than an informal collection of
  logging helpers

Just as importantly, this phase recognized documentation and typing cleanup as
part of architectural work. The repo had become complex enough that readability
and re-entry were now system concerns.

### April 1, 2026, `2f036e4`: Observability Package Split

This pass is documented in
[`history/observability_package_refactor_summary.md`](history/observability_package_refactor_summary.md).

#### What changed

The earlier single-file observability implementation was split into the current
package:

- runtime setup
- callbacks
- tensor helpers
- logging helpers
- reporting

#### Later follow-up

The current repository kept the stable callback facade in
`src/observability/callbacks.py`, but later split the concrete callback
implementations into smaller responsibility-focused modules:

- `debug_callbacks.py`
- `system_callbacks.py`
- `parameter_callbacks.py`
- `prediction_callbacks.py`

That later change preserved the same import story while making the callback
layer itself less monolithic.

#### Why it changed

The initial observability integration landed quickly as one large surface, but
that surface became too large to navigate safely. The package split was a
maintainability refactor.

#### Lasting impact

This refactor improved internal structure without breaking the external import
surface used by `train.py`, `main.py`, notebooks, or tests.

That "split internally, preserve the outer surface" approach is a recurring
theme in the repo's evolution and is usually the right instinct for future
refactors too.

### April 1, 2026, `3e510c3`: Dedicated Evaluation Package

This pass is documented in
[`history/evaluation_package_summary.md`](history/evaluation_package_summary.md).

#### What changed

The repo gained `src/evaluation/` as the canonical home for:

- metric definitions
- grouped metric aggregation
- prediction-batch normalization
- structured held-out evaluation

#### Why it changed

The codebase needed a clearer distinction between:

- runtime observability
- canonical model-quality evaluation

Before this pass, richer evaluation logic was too close to reporting or to the
model path itself.

#### Lasting impact

This established another important architectural boundary that still matters
today:

- observability answers "what happened during the run?"
- evaluation answers "how well did the model perform?"

That distinction matters for future work too. A new metric does not
automatically belong in observability, and a new report does not automatically
belong in the evaluation core.

### April 1, 2026, `1686482`: Test Hardening Follow-Up

The latest inspected follow-up expanded observability test coverage and relaxed
some reporting typing assumptions.

This was not as large a structural change as the earlier milestones, but it
reinforced a healthy pattern in the repo's evolution: once a subsystem becomes
real, it gets real tests.

### April 1, 2026: Runtime Environment Profiles, Diagnostics, And `src/environment/`

This pass is summarized in
[`history/environment_runtime_profiles_summary.md`](history/environment_runtime_profiles_summary.md).

#### What changed

The repository gained a dedicated runtime-environment layer:

- `src/environment/detection.py`
- `src/environment/profiles.py`
- `src/environment/diagnostics.py`

At the same time:

- `main.py` gained high-level `--device-profile` support
- `main.py` and `main.ipynb` were aligned around one environment-aware
  workflow
- preflight diagnostics and diagnostics-only execution were added
- runtime environment metadata became part of the run summary
- the old config-adjacent `src/config/environment.py` surface was replaced by
  the dedicated `src/environment/` package

#### Why it changed

The repository had become portable enough to be run in several environments,
but the runtime policy for those environments was still too implicit.

The project needed:

- a better story for repeated Colab, local CUDA, CPU-only, Slurm, and Apple
  Silicon runs
- a way to choose sensible runtime defaults without hard-coding one machine's
  assumptions
- a more helpful response when failures came from backend or dependency
  mismatches rather than model logic
- a cleaner boundary between typed config contracts and environment/runtime
  interpretation

#### Lasting impact

This milestone made environment interpretation a first-class subsystem rather
than a side effect of entrypoint code.

It also clarified an important architectural distinction:

- `src/config/` defines runtime contracts
- `src/environment/` interprets the current runtime context and chooses
  environment-sensitive defaults

That separation matters for future work too. New backend checks, profile
policies, or preflight diagnostics now have an obvious home that does not blur
the config layer or the training wrapper.

### Later April 1, 2026 Follow-Up: Runtime Tuning, Robustness, And Editor Cleanup

After the initial `src/environment/` milestone, the repository went through a
smaller but still important follow-up wave.

#### What changed

The newer follow-up added:

- `src/environment/tuning.py` as the backend-knob companion to detection,
  profiles, and diagnostics
- backend-aware defaults for TF32, cuDNN benchmark, thread counts, BF16, and
  optional `torch.compile(...)`
- a benchmark-only top-level workflow for comparing environments more quickly
- compile fallback and workflow-level robustness fixes
- Pyright/Pylance cleanup for the newer runtime and notebook surfaces
- a final consistency pass over the lighter facade/helper modules so their
  comments matched the denser documentation style already used in the more
  complex model/runtime files

#### Why it changed

The earlier environment milestone answered:

- "what machine am I on?"
- "which profile should I use?"
- "does this requested setup look valid?"

The follow-up answered the next set of questions:

- "how do those policy choices actually reach Torch backend knobs?"
- "how can we compare environments without running the full evaluation stack?"
- "how do we keep newer runtime/editor surfaces from becoming a static-analysis
  weak spot?"

#### Lasting impact

This follow-up completed the runtime story more fully:

- config declares policy
- environment resolves policy
- tuning applies policy
- diagnostics validate policy
- entry surfaces can benchmark and report policy

It also reinforced another recurring theme in the repository's evolution:
once a subsystem becomes first-class, documentation, typing, and editor
correctness eventually have to catch up so that future work is easier rather
than harder.



## April 2–7, 2026: Reporting Refactor And Observability Expansion

This is the most recent evolution phase and reflects a shift toward a **clear separation
between runtime observability and post-run reporting**, while improving both systems.

#### What changed

The repository introduced a structured reporting layer under
`src/reporting/` and expanded the observability system:

- reporting logic was decomposed into responsibility-focused modules:
  - `prediction_rows.py`
  - `report_tables.py`
  - `report_text.py`
  - `builders.py`
- observability callbacks were further specialized:
  - system-level signals
  - parameter/gradient inspection
  - prediction-level logging
- structured outputs were standardized into:
  - CSV/JSON artifacts
  - shared report bundles
- TensorBoard was strengthened as a **primary runtime visualization surface**

#### Why it changed

Earlier reporting approaches relied heavily on external HTML artifacts,
which created friction:

- harder to navigate compared to interactive dashboards
- duplication between logging and reporting layers
- limited drill-down capability during training

At the same time, observability had matured enough that:

- most useful signals were already being computed during runtime
- TensorBoard provided strong real-time inspection capabilities

This phase therefore shifted the design toward:

- richer runtime visibility via observability
- cleaner, structured post-run reporting outputs
- reduced reliance on large monolithic reports

#### Lasting impact

This phase clarifies the final architectural relationship:

- **observability** is responsible for understanding runtime behavior
- **reporting** is responsible for packaging structured outputs after a run

TensorBoard remains a **primary observability sink**, but not the entire system.

This keeps:

- training logic unchanged
- observability centralized during execution
- reporting as a distinct, structured post-run system


## Cross-Cutting Design Decisions

Several design decisions repeat throughout the codebase's evolution. These are
the ideas that best explain why the repository looks the way it does today.

### Runtime-Bound Model Configuration

The model config is not treated as purely static. Category cardinalities and
some feature/schema details are discovered from prepared data, then bound into
the final model config. This is why the DataModule and training wrapper are so
central to the repo's architecture.

### Lightning Responsibilities Are Deliberately Layered

The repository converged on a clear split:

- data lifecycle in `AZT1DDataModule`
- model behavior in `FusedModel`
- Trainer orchestration in `FusedModelTrainer`
- top-level run setup in `main.py` and `main.ipynb`

This split did not exist fully at the start. It was earned through refactoring.

### Config Became A First-Class Domain

The move from `src/utils/config.py` to `src/config/` was not just cleanup. It
was an explicit decision that configuration contracts are one of the core
architectural surfaces in the project.

### Environment Policy Became Explicit

The later addition of `src/environment/` extended that same architectural
thinking into runtime portability and diagnostics. The repo no longer relies
only on raw low-level flags to express where and how it should run.

### Observability And Evaluation Are Adjacent, Not The Same

The repo now treats telemetry/reporting and structured evaluation as separate
subsystems. That boundary is part of the project vision, not an incidental file
split.

### Documentation Is Part Of The System

The current docs, archived milestone summaries, architecture diagrams, and
source-level comments are a direct response to the complexity of the codebase.
The team chose to treat documentation as part of maintainability and onboarding,
not as an afterthought.

### Stable Philosophy That Emerged

Across the full evolution, the repository gradually converged on a practical
engineering philosophy:

- make semantic contracts explicit
- prefer small responsibility-focused modules over convenience-driven large ones
- keep runtime orchestration separate from model semantics
- preserve shared entry surfaces for teammates and future notebooks
- record architectural intent while the reasoning is still fresh

## Baseline Versus Current Repository

The easiest way to understand the scale of the evolution is to compare the
baseline repository at `ffd190c` with the current one.

### Baseline

- compact `src/` tree with mixed responsibilities
- old data modules
- one large config utility
- early fused-model integration
- no reusable runtime wrapper
- no dedicated evaluation or observability package
- minimal high-level documentation

### Current

- layered data pipeline under `src/data/`
- explicit typed config package under `src/config/`
- Lightning-native model in `src/models/fused_model.py`
- reusable orchestration layer in `src/train.py`
- root-level runnable entrypoints through `main.py`, `main.ipynb`, and
  `defaults.py`
- reusable entry-surface workflow package under `src/workflows/`
- environment-aware runtime layer under `src/environment/`
- observability package under `src/observability/`
- evaluation package under `src/evaluation/`
- broader automated test coverage under `tests/`
- organized documentation under `docs/`

The core forecasting idea is continuous across both states. What changed is the
clarity of contracts, runtime ownership, maintainability, and the ability for
other people to understand and extend the system.

## How To Use This History Going Forward

This history is useful for more than retrospective understanding. It should
also guide future decisions.

When deciding where a new feature belongs, ask:

- is this data preparation, model behavior, runtime policy, observability, or
  evaluation?
- does this change preserve the current responsibility split, or blur it?
- is this a new contract that should be documented as part of the architecture?

In other words, this document is not only about where the repository has been.
It is also a reminder of the architectural lessons the repo already paid to
learn.

## What This Means For The Team

When reading the current codebase, it is important to understand that the repo
did not start from a cleanly layered architecture. Many current boundaries
exist because earlier versions made the friction visible:

- unclear data/model boundaries led to the DataModule refactor
- unfinished model input semantics led to the structured batch contract
- scattered Lightning concerns led to the training wrapper
- oversized config utilities led to `src/config/`
- opaque environment failures and multi-platform runtime needs led to
  `src/environment/`
- mixed reporting/metric logic led to `src/evaluation/`
- limited runtime visibility led to the observability package

That history matters because it also explains the current vision:

- keep boundaries explicit
- keep model/data/runtime concerns separate
- keep the repo runnable from stable entrypoints
- keep evaluation and observability first-class
- keep documentation close to the architecture

## Archived Milestone Documents

For deeper detail on any one phase, use the archived summaries:

- [`history/data_refactor_summary.md`](history/data_refactor_summary.md)
- [`history/model_refactor_summary.md`](history/model_refactor_summary.md)
- [`history/lightning_model_integration_summary.md`](history/lightning_model_integration_summary.md)
- [`history/train_wrapper_summary.md`](history/train_wrapper_summary.md)
- [`history/entrypoint_defaults_summary.md`](history/entrypoint_defaults_summary.md)
- [`history/source_refactor_and_documentation_update.md`](history/source_refactor_and_documentation_update.md)
- [`history/commenting_conventions_summary.md`](history/commenting_conventions_summary.md)
- [`history/observability_integration_summary.md`](history/observability_integration_summary.md)
- [`history/observability_package_refactor_summary.md`](history/observability_package_refactor_summary.md)
- [`history/evaluation_package_summary.md`](history/evaluation_package_summary.md)
- [`history/environment_runtime_profiles_summary.md`](history/environment_runtime_profiles_summary.md)
- [`history/test_layout_and_runtime_modernization_summary.md`](history/test_layout_and_runtime_modernization_summary.md)
