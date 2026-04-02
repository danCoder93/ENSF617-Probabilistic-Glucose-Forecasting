# Source Refactor And Documentation Update

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the later structural cleanup wave that followed the
initial model, training-wrapper, and observability integration work.

Later follow-up note:
Subsequent work also added a dedicated `src/evaluation/` package for
structured held-out metrics. This document remains focused on the source-layout,
typing, and commenting cleanup wave rather than that later evaluation-layer
addition.

Another later follow-up moved runtime-environment detection, profile
resolution, and diagnostics into a dedicated `src/environment/` package. That
follow-up kept the spirit of this cleanup wave intact by preserving
`src/config/` as the home of typed contracts rather than operational runtime
interpretation. See
[`environment_runtime_profiles_summary.md`](environment_runtime_profiles_summary.md).

This document focuses on the changes that were made after the repository had
already become functionally richer:

- moving config into its own package
- cleaning up import structure
- tightening static typing around path-like config values
- fixing editor/type-checker issues
- standardizing the in-source documentation style
- expanding inline implementation comments in the complex model and
  observability paths

## High-Level Intent

The goal of this pass was not to add new forecasting behavior.

The goal was to make the repository easier to:

- navigate
- maintain
- statically analyze
- debug
- and re-enter later without reverse-engineering the architecture from scratch

In practice, this meant treating code structure and documentation as
maintainability work rather than cosmetic cleanup.

## Major Structural Change: `src/config/`

One of the most important refactors in this pass was moving the repository's
configuration system out of the old `utils` location and into a dedicated
top-level package:

- `src/config/__init__.py`
- `src/config/data.py`
- `src/config/model.py`
- `src/config/runtime.py`
- `src/config/observability.py`
- `src/config/serde.py`
- `src/config/types.py`

### Why this change mattered

The old config module had grown too large and too broad in responsibility.

It was carrying:

- dataset config
- model config
- Trainer/runtime config
- observability config
- serialization/deserialization helpers

Moving that surface into `src/config/` made the ownership boundaries much
clearer:

- config is a first-class domain, not a generic utility
- runtime and observability policy now sit in explicit modules
- serialization logic is separated from the dataclass definitions
- import paths better reflect architecture

Later work reinforced that same boundary by moving environment-sensitive
runtime policy and diagnostics out of config-adjacent code and into the newer
`src/environment/` package.

For example:

- `from config import Config, TrainConfig`

now communicates intent more clearly than:

- `from utils.config import Config, TrainConfig`

### Old compatibility path removed

After the import migration was complete, the compatibility shim under
`src/utils/config.py` was removed.

That means `src/config/` is now the single source of truth for the repository's
config system.

## Static Typing And Pylance Cleanup

Another large part of this pass was cleaning up type-checker friction,
especially around:

- `Path` versus `str | Path`
- fake/test doubles that no longer matched real method signatures
- broad placeholder objects in typed fields
- Torch/Pylance export warnings in the TFT code

### Path-like config inputs

The config layer now uses a shared `PathInput = str | Path` alias for fields
that intentionally accept plain strings at construction time.

That was important because the codebase now supports several entry styles:

- direct Python usage
- CLI wiring
- notebook/Colab usage
- tests using temporary string paths

Those path-like values are normalized to concrete `Path` objects at the right
runtime boundaries rather than pretending the constructor always receives a
`Path`.

### Observability path normalization

The observability runtime and callback code also received a cleanup pass so
path-like config values are no longer treated as guaranteed `Path` objects
before they are normalized.

This removed a class of Pylance/type-checker errors around:

- `.parent`
- `.mkdir(...)`
- `.open(...)`

and made the runtime typing story match the code's actual intended usage.

### Torch/Pylance cleanup in the TFT implementation

The TFT implementation also received a few targeted type/editor fixes,
including:

- targeted Pyright ignores on `torch.jit.script` use sites that are valid at
  runtime but can trigger false-positive export warnings in some editors
- replacing `nn.Parameter(...)` with explicit `Parameter(...)` imports to avoid
  private-export warnings in some stub combinations

These changes were not architectural; they were about keeping the editor and
the runtime contract aligned.

## Source Documentation Standardization

Another major part of this work was a full pass over `src/` to standardize the
commenting and docstring style.

The enforced convention is now:

- top comments in a file: `#`
- class comments: `"""..."""`
- function comments: `"""..."""`
- inline and detailed walkthrough comments: `#`

### Why this convention was chosen

It matches how the repository is actually used:

- file-level `#` comments work well for maintenance notes, scope disclaimers,
  and architecture prefaces
- class/function docstrings work well for API-level contract explanation
- detailed `#` comments work well for tensor logic, Lightning lifecycle notes,
  broadcasting behavior, and branch-fusion walkthroughs

The project intentionally does **not** treat dense inline comments as a smell
in the complex files. For tensor-heavy code, more explanation is often a net
positive.

## Deep Comment Pass In Complex Files

The heaviest documentation improvements were made in the most complex
implementation paths:

- `src/models/fused_model.py`
- `src/models/tft.py`
- `src/train.py`
- `src/observability/`

Those files now carry more detailed explanations around:

- grouped feature semantics
- branch-specific information budgets
- lazy parameter materialization
- quantile loss interpretation
- variable-selection logic
- attention shapes and masking
- Trainer construction and checkpoint semantics
- observability callback ordering and artifact flow

The goal of those comments is not to restate the code line by line, but to make
the architecture readable for a future contributor who may not remember every
shape convention or design boundary.

## Source-Wide Documentation Result

After the cleanup pass:

- every top-level class in `src/` has a docstring
- every top-level function in `src/` has a docstring
- file-level notes in `src/` use `#` comments instead of module docstrings
- the complex files also include denser `#` walkthrough comments where the
  implementation would otherwise be hard to re-enter

This means the source tree is now both:

- structurally documented
- locally explainable

without forcing every small file to become comment-heavy.

## Files Most Affected By This Cleanup Wave

The most affected implementation files were:

- `src/config/__init__.py`
- `src/config/data.py`
- `src/config/model.py`
- `src/config/runtime.py`
- `src/config/observability.py`
- `src/config/serde.py`
- `src/config/types.py`
- `src/models/fused_model.py`
- `src/models/tft.py`
- `src/models/tcn.py`
- `src/models/grn.py`
- `src/models/nn_head.py`
- `src/train.py`
- `src/observability/`
- `src/data/datamodule.py`
- `src/data/dataset.py`
- `src/data/downloader.py`
- `src/data/indexing.py`
- `src/data/preprocessor.py`
- `src/data/schema.py`
- `src/data/transforms.py`
- `src/utils/tft_utils.py`

## Verification Performed During This Work

This cleanup wave relied primarily on:

- source inspection
- targeted type-fix iterations
- repeated `py_compile` validation over the edited files

It did **not** serve as a substitute for full runtime validation.

The architectural and documentation work is in much better shape now, but the
most important remaining verification step is still:

- running the test suite in the real project environment
- running at least one short training job
- confirming the current TensorBoard / observability path behaves as intended

## Practical Outcome

The repository now has:

- cleaner config ownership
- clearer imports
- a clearer distinction between typed config contracts and operational
  environment/runtime logic
- fewer static-analysis surprises
- more explicit architecture boundaries
- stronger in-code explanation in the complex files
- and a source tree that is much easier to navigate and maintain than before

## Later Observability Package Split

After this cleanup wave, the observability implementation was also split from a
single large module into a dedicated `src/observability/` package.

That follow-up refactor preserved the public import surface but made the
internal ownership boundaries clearer:

- `runtime.py`
  logger/profiler/artifact assembly
- `logging_utils.py`
  logger-aware helpers
- `tensors.py`
  tensor and nested-batch normalization
- `callbacks.py`
  observability callbacks and callback assembly
- `reporting.py`
  post-run CSV/HTML artifact generation

This is consistent with the broader maintainability direction documented in
this file:

- explicit ownership boundaries
- easier navigation through smaller modules
- stronger in-source explanation in complex paths

This refactor wave did not change the core modeling goals of the project, but
it substantially improved the maintainability of the implementation that
supports those goals.
